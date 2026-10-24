"""Behavior cloning with episode splits, full-frame RGB encoders and a GRU."""
import csv
import hashlib
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from .core import canonical_config, normalized, save_json
from .data import validate_episode


RESNET18_ARCH = "resnet18-proj256-gru64-mlp32"
MOBILENET_V2_ARCH = "mobilenet-v2-proj256-gru64-mlp32"
ARCHITECTURES = (RESNET18_ARCH, MOBILENET_V2_ARCH)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class StopHead(nn.Module):
    """Binary terminal-stop logit from the recurrent policy representation."""
    def __init__(self):
        super(StopHead, self).__init__()
        self.network = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, recurrent):
        return self.network(recurrent).squeeze(-1)


def image_tensor(image, size=224):
    """Match collection/deployment colors; resize full view without crop or flip."""
    if not isinstance(size, int) or size < 32:
        raise ValueError("Image size must be an integer of at least 32")
    image = Image.fromarray(image) if isinstance(image, np.ndarray) else image.convert("RGB")
    image = image.resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1).copy()) / 255.
    return (x - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]


class ParkingPolicy(nn.Module):
    def __init__(self, pretrained=True, architecture=RESNET18_ARCH):
        super(ParkingPolicy, self).__init__()
        self.architecture = architecture
        if architecture == RESNET18_ARCH:
            if hasattr(models, "ResNet18_Weights"):
                self.encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            else:  # JetPack 4 images often have an older torchvision
                self.encoder = models.resnet18(pretrained=pretrained)
            self.encoder.fc = nn.Identity()
            feature_count = 512
        elif architecture == MOBILENET_V2_ARCH:
            if hasattr(models, "MobileNet_V2_Weights"):
                self.encoder = models.mobilenet_v2(
                    weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
            else:
                self.encoder = models.mobilenet_v2(pretrained=pretrained)
            self.encoder.classifier = nn.Identity()
            feature_count = 1280
        else:
            raise ValueError("Unsupported architecture: " + str(architecture))
        self.projection = nn.Linear(feature_count, 256)
        self.gru = nn.GRU(258, 64, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2), nn.Tanh())
        self.finetune = False
        self.set_stage(False)

    def set_stage(self, finetune=False):
        self.finetune = finetune
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        if finetune:
            for parameter in self.finetune_parameters():
                parameter.requires_grad = True
        self.train(self.training)

    def finetune_parameters(self):
        if self.architecture == RESNET18_ARCH:
            return self.encoder.layer4.parameters()
        return self.encoder.features[-4:].parameters()

    def train(self, mode=True):
        super(ParkingPolicy, self).train(mode)
        # Keep pretrained BatchNorm statistics fixed in both stages, including layer4.
        self.encoder.eval()
        return self

    def recurrent_features(self, images, previous_actions, hidden=None):
        batch, steps = images.shape[:2]
        features = self.encoder(images.reshape(-1, *images.shape[2:]))
        embedding = self.projection(features).reshape(batch, steps, 256)
        return self.gru(torch.cat((embedding, previous_actions), dim=-1), hidden)

    def forward(self, images, previous_actions, hidden=None):
        recurrent, hidden = self.recurrent_features(images, previous_actions, hidden)
        return self.head(recurrent), hidden


def eligible_episodes(root, allow_synthetic=False, include_completed_unreviewed=False):
    episodes = []
    for path in sorted(Path(root).glob("episode_*")):
        meta_path = path / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        explicitly_accepted = bool(meta.get("accepted"))
        user_selected_completed = (include_completed_unreviewed and
                                   meta.get("status") == "completed" and
                                   meta.get("outcome") not in ("collision", "discard"))
        if ((not explicitly_accepted and not user_selected_completed) or
                meta.get("source") not in ("human", "dagger")):
            continue
        if meta.get("synthetic") and not allow_synthetic:
            continue
        report = validate_episode(path)
        if not report["valid"]:
            raise ValueError("Invalid accepted episode: " + str(report))
        episodes.append(path)
    return episodes


def make_split(episodes, seed=7):
    if len(episodes) < 3:
        raise ValueError("Need at least 3 reviewed episodes (ideally 30+) for separate train/val/test sets")
    ordered = list(episodes)
    random.Random(seed).shuffle(ordered)
    n = len(ordered)
    n_val = n_test = max(1, int(round(n * .15)))
    n_train = n - n_val - n_test
    return {"train": ordered[:n_train], "val": ordered[n_train:n_train + n_val], "test": ordered[n_train + n_val:]}


def reuse_split(episodes, split_file):
    """Preserve held-out episodes when new recovery demonstrations are added."""
    saved = json.loads(Path(split_file).read_text())
    if set(saved) != {"train", "val", "test"} or any(not saved[k] for k in saved):
        raise ValueError("Split must contain nonempty train/val/test lists")
    names = saved["train"] + saved["val"] + saved["test"]
    available = {p.name: p for p in episodes}
    if len(set(names)) != len(names) or set(names) - set(available):
        raise ValueError("Split contains duplicates or missing/ineligible episodes")
    result = {k: [available[name] for name in values] for k, values in saved.items()}
    result["train"].extend(p for p in episodes if p.name not in set(names))
    return result


class Sequences(Dataset):
    def __init__(self, episodes, length=16, stride=8, augmentation=None):
        if length < 1 or stride < 1:
            raise ValueError("Sequence length and stride must be positive")
        self.samples = []
        self.augmentation = augmentation
        for path in episodes:
            with (path / "actions.csv").open() as f:
                rows = list(csv.DictReader(f))
            for start in range(0, len(rows) - length + 1, stride):
                self.samples.append((path, rows[start:start + length]))
        if not self.samples:
            raise ValueError("No complete sequences in split; collect longer episodes or reduce --sequence-length")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, rows = self.samples[index]
        metadata = json.loads((path / "metadata.json").read_text())
        dagger_labels = metadata.get("source") == "dagger"
        images = []
        parameters = self.augmentation.sample() if self.augmentation is not None else None
        for row in rows:
            with Image.open(str(path / row["frame"])) as im:
                if self.augmentation is not None:
                    im = self.augmentation.apply(im, parameters)
                images.append(image_tensor(im))
        previous = [[float(r["prev_steering_norm"]), float(r["prev_throttle_norm"])] for r in rows]
        # A DAgger episode executes a policy/expert mixture, but the requested
        # action columns contain the expert label for the visited state.
        target_keys = (("requested_steering_norm", "requested_throttle_norm")
                       if dagger_labels else ("steering_norm", "throttle_norm"))
        target = [[float(r[target_keys[0]]), float(r[target_keys[1]])] for r in rows]
        return torch.stack(images), torch.tensor(previous, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


def signature(cfg):
    cfg = canonical_config(cfg)
    return {"limits": cfg["limits"], "hardware": {k: cfg["hardware"][k] for k in (
        "driver", "steering_gain", "steering_offset", "throttle_gain", "right_steering_sign")},
        "camera": cfg["camera"], "hz": cfg["control"]["hz"]}


def load_checkpoint(path, device="cpu"):
    # Only load checkpoints you created or trust. Legacy torch lacks weights_only.
    try:
        return torch.load(str(path), map_location=device, weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location=device)


def train(root, output, epochs=20, finetune_epochs=0, batch_size=4, sequence_length=16,
          stride=8, seed=7, patience=5, device="cpu", pretrained=True, allow_synthetic=False,
          split_file=None, include_completed_unreviewed=False, architecture=RESNET18_ARCH,
          augment=False, init_checkpoint=None, learning_rate=1e-4, encoder_learning_rate=1e-5):
    if min(epochs, batch_size, patience) < 1 or finetune_epochs < 0:
        raise ValueError("Invalid training settings")
    if not np.isfinite([learning_rate, encoder_learning_rate]).all() or min(learning_rate, encoder_learning_rate) <= 0:
        raise ValueError("Learning rates must be finite and positive")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    output = Path(output)
    episodes = eligible_episodes(root, allow_synthetic, include_completed_unreviewed)
    configs = [json.loads((p / "metadata.json").read_text())["config"] for p in episodes]
    if not configs or any(signature(c) != signature(configs[0]) for c in configs):
        raise ValueError("Episodes must use the same camera, rate, calibration and limits")
    split = reuse_split(episodes, split_file) if split_file else make_split(episodes, seed)
    from .augmentation import SequenceAugment
    augmentation = SequenceAugment(seed=seed) if augment else None
    loaders = {k: DataLoader(Sequences(v, sequence_length, stride,
                            augmentation=augmentation if k == "train" else None), batch_size=batch_size,
                            shuffle=(k == "train"), num_workers=0) for k, v in split.items() if k != "test"}
    Sequences(split["test"], sequence_length, stride)  # verify test lengths before training
    output.mkdir(parents=True, exist_ok=False)  # never overwrite a checkpoint / split accidentally
    save_json(output / "split.json", {k: [p.name for p in v] for k, v in split.items()})
    model = ParkingPolicy(pretrained and init_checkpoint is None, architecture).to(device)
    initial = load_checkpoint(init_checkpoint, "cpu") if init_checkpoint else None
    if initial is not None:
        if initial["architecture"] != architecture or signature(initial["config"]) != signature(configs[0]):
            raise ValueError("Initial checkpoint architecture or calibration differs")
        # Never fine-tune on episodes previously used as held-out examples.
        held_out = set(initial["split"]["val"] + initial["split"]["test"])
        if held_out.intersection(p.name for p in split["train"]):
            raise ValueError("Initial held-out episodes entered training; pass the original --split")
        model.load_state_dict(initial["state_dict"])
    training_settings = {"augmentation": augmentation.config if augmentation else None,
                         "init_checkpoint_sha256": file_sha256(init_checkpoint) if init_checkpoint else None,
                         "learning_rate": learning_rate, "encoder_learning_rate": encoder_learning_rate}
    save_json(output / "training_settings.json", training_settings)
    best, stale, history = float("inf"), 0, []
    checkpoint = output / "best.pt"
    for stage, count in ((False, epochs), (True, finetune_epochs)):
        if count == 0:
            continue
        if stage:
            model.load_state_dict(load_checkpoint(checkpoint, device)["state_dict"])
        model.set_stage(stage)
        head = list(model.projection.parameters()) + list(model.gru.parameters()) + list(model.head.parameters())
        groups = [{"params": head, "lr": learning_rate}]
        if stage:
            groups.append({"params": model.finetune_parameters(), "lr": encoder_learning_rate})
        optimizer = torch.optim.Adam(groups)
        stale = 0
        for epoch in range(count):
            metrics = {"stage": "layer4" if stage else "frozen", "epoch": epoch + 1}
            for phase, loader in loaders.items():
                model.train(phase == "train")
                total, elements = 0., 0
                for images, previous, target in loader:
                    images, previous, target = images.to(device), previous.to(device), target.to(device)
                    with torch.set_grad_enabled(phase == "train"):
                        prediction, _ = model(images, previous)
                        # Each independent window starts with h=0; never cross episodes.
                        loss = ((prediction - target) ** 2).sum(dim=-1).mean()
                        if not torch.isfinite(loss):
                            raise RuntimeError("Non-finite behavior cloning loss")
                        if phase == "train":
                            optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                            optimizer.step()
                    count_elements = target.shape[0] * target.shape[1]
                    total += loss.item() * count_elements
                    elements += count_elements
                metrics[phase] = total / elements
            history.append(metrics)
            print(metrics, flush=True)
            save_json(output / "history.json", history)
            if metrics["val"] < best:
                best, stale = metrics["val"], 0
                payload = {"schema_version": 1, "architecture": architecture,
                           "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                           "config": configs[0], "sequence_length": sequence_length, "stride": stride,
                           "pretrained": pretrained, "seed": seed, "best_val_loss": best,
                           "training_settings": training_settings,
                           "included_completed_unreviewed": include_completed_unreviewed,
                           "trained_with_synthetic": any(json.loads((p / "metadata.json").read_text())["synthetic"] for p in episodes),
                           "split": {k: [p.name for p in v] for k, v in split.items()}}
                temp = output / "best.tmp"
                torch.save(payload, str(temp))
                temp.replace(checkpoint)
            else:
                stale += 1
            if stale >= patience:
                break
    return checkpoint


class Predictor:
    def __init__(self, checkpoint, cfg, device="cpu", real=False, fp16=False, trt_encoder=None,
                 stop_head=None):
        saved = load_checkpoint(checkpoint, device)
        cfg = canonical_config(cfg)
        if signature(saved["config"]) != signature(cfg):
            raise ValueError("Checkpoint and runtime calibration/camera/control rate differ")
        if real and saved.get("trained_with_synthetic"):
            raise ValueError("A synthetic-data checkpoint cannot drive the real car")
        if fp16 and not (str(device).startswith("cuda") or str(device).startswith("mps")):
            raise ValueError("FP16 inference requires a CUDA or MPS device")
        architecture = saved.get("architecture", RESNET18_ARCH)
        self.model = ParkingPolicy(pretrained=False, architecture=architecture).to(device).eval()
        self.model.load_state_dict(saved["state_dict"])
        self.fp16 = bool(fp16)
        if self.fp16:
            self.model.half()
        self.backend = "pytorch"
        self.stop_head = None
        self.stop_decision = {}
        self.last_stop_probability = None
        if trt_encoder is not None:
            if not str(device).startswith("cuda") or not self.fp16:
                raise ValueError("The TensorRT encoder requires CUDA with --fp16")
            from torch2trt import TRTModule
            try:
                artifact = torch.load(str(trt_encoder), map_location=device, weights_only=True)
            except TypeError:  # PyTorch 1.8 on JetPack 4
                artifact = torch.load(str(trt_encoder), map_location=device)
            if artifact.get("architecture") != architecture:
                raise ValueError("TensorRT encoder architecture does not match checkpoint")
            if artifact.get("checkpoint_sha256") != file_sha256(checkpoint):
                raise ValueError("TensorRT encoder was built from a different checkpoint")
            if artifact.get("precision") != "fp16" or artifact.get("input_shape") != [1, 3, 224, 224]:
                raise ValueError("Unsupported TensorRT encoder precision or input shape")
            encoder = TRTModule()
            encoder.load_state_dict(artifact["state_dict"])
            self.model.encoder = encoder
            self.backend = "tensorrt-fp16-encoder"
            torch.cuda.empty_cache()
        if stop_head is not None:
            try:
                stop_artifact = torch.load(str(stop_head), map_location=device, weights_only=True)
            except TypeError:  # PyTorch 1.8 on JetPack 4
                stop_artifact = torch.load(str(stop_head), map_location=device)
            if stop_artifact.get("artifact_type") != "recurrent_stop_head":
                raise ValueError("Not a recurrent stop-head artifact")
            if stop_artifact.get("architecture") != architecture:
                raise ValueError("Stop-head architecture does not match checkpoint")
            if stop_artifact.get("checkpoint_sha256") != file_sha256(checkpoint):
                raise ValueError("Stop head was trained from a different checkpoint")
            self.stop_head = StopHead().to(device).eval()
            self.stop_head.load_state_dict(stop_artifact["state_dict"])
            if self.fp16:
                self.stop_head.half()
            self.stop_decision = dict(stop_artifact.get("decision", {}))
        if str(device).startswith("cuda"):
            torch.backends.cudnn.benchmark = True
        self.architecture = architecture
        self.cfg, self.device, self.hidden = cfg, device, None

    def reset(self):
        self.hidden = None
        self.last_stop_probability = None

    def __call__(self, rgb, previous):
        dtype = torch.float16 if self.fp16 else torch.float32
        image = image_tensor(rgb)[None, None].to(device=self.device, dtype=dtype)
        prev = torch.tensor(normalized(previous, self.cfg["limits"]), dtype=dtype)[None, None].to(self.device)
        with torch.no_grad():
            if self.stop_head is None:
                action, self.hidden = self.model(image, prev, self.hidden)
                self.last_stop_probability = None
            else:
                recurrent, self.hidden = self.model.recurrent_features(image, prev, self.hidden)
                action = self.model.head(recurrent)
                stop_logit = self.stop_head(recurrent)
                self.last_stop_probability = float(torch.sigmoid(stop_logit).float().cpu().item())
        return tuple(action[0, 0].cpu().tolist())


def evaluate(root, checkpoint, device="cpu", fp16=False):
    """Held-out test episodes only, replaying full recurrent histories with teacher actions."""
    saved = load_checkpoint(checkpoint, device)
    predictor = Predictor(checkpoint, saved["config"], device, fp16=fp16)
    total, count = np.zeros(2), 0
    results = []
    for name in saved["split"]["test"]:
        path = Path(root) / name
        report = validate_episode(path)
        if not report["valid"]:
            raise ValueError(str(report))
        predictor.reset()
        subtotal, n = np.zeros(2), 0
        with (path / "actions.csv").open() as f:
            for row in csv.DictReader(f):
                with Image.open(str(path / row["frame"])) as im:
                    action = predictor(np.array(im.convert("RGB")), (float(row["prev_steering"]), float(row["prev_throttle"])))
                error = np.array(action) - [float(row["steering_norm"]), float(row["throttle_norm"])]
                subtotal += error ** 2
                n += 1
        results.append({"episode": name, "frames": n, "mse": (subtotal / n).tolist()})
        total += subtotal
        count += n
    return {"normalized_steering_mse": float(total[0] / count), "normalized_throttle_mse": float(total[1] / count),
            "episodes": results, "precision": "fp16" if fp16 else "fp32",
            "mode": "full-episode teacher-forced replay; not physical success rate"}
