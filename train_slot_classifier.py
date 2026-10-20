#!/usr/bin/env python3
"""Train a left-empty-slot classifier from three starting frames per episode."""
import argparse
import csv
import json
import random
import time
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


class StartFrames(Dataset):
    def __init__(self, items, archive, augment=False, seed=17):
        from parking_lib.augmentation import SequenceAugment
        self.items = items
        self.archive = archive
        self.augment = SequenceAugment(seed=seed) if augment else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        from parking_lib.learning import image_tensor
        item = self.items[index]
        params = self.augment.sample() if self.augment else None
        images = []
        with ZipFile(str(self.archive)) as archive:
            for name in item["frames"]:
                with archive.open(name) as stream:
                    image = Image.open(stream).convert("RGB")
                    if self.augment:
                        image = self.augment.apply(image, params)
                    images.append(image_tensor(image))
        return torch.stack(images), torch.tensor(float(item["label"])), item["episode"]


def stratified_split(items, seed):
    rng = random.Random(seed)
    result = {"train": [], "val": [], "test": []}
    for label in (0, 1):
        group = [item for item in items if item["label"] == label]
        rng.shuffle(group)
        n_val = max(1, int(round(len(group) * .15)))
        n_test = max(1, int(round(len(group) * .15)))
        result["val"].extend(group[:n_val])
        result["test"].extend(group[n_val:n_val + n_test])
        result["train"].extend(group[n_val + n_test:])
    for values in result.values():
        rng.shuffle(values)
    return result


def metrics(target, probability, threshold=.5):
    target = np.asarray(target, dtype=np.int64)
    probability = np.asarray(probability)
    prediction = probability >= threshold
    tp = int(((prediction == 1) & (target == 1)).sum())
    tn = int(((prediction == 0) & (target == 0)).sum())
    fp = int(((prediction == 1) & (target == 0)).sum())
    fn = int(((prediction == 0) & (target == 1)).sum())
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    return {"accuracy": (tp + tn) / len(target),
            "balanced_accuracy": (recall + specificity) / 2.,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(1e-12, precision + recall),
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}


def evaluate(model, loader, device, loss_function):
    model.eval()
    total, count, targets, probabilities, episodes = 0., 0, [], [], []
    with torch.no_grad():
        for images, target, names in loader:
            images, target = images.to(device), target.to(device)
            logits = model(images)
            loss = loss_function(logits, target)
            total += float(loss.item()) * len(target)
            count += len(target)
            targets.extend(target.cpu().numpy().tolist())
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            episodes.extend(names)
    return total / count, targets, probabilities, episodes


def main():
    args = arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    labels = list(csv.DictReader(Path(args.labels).open()))
    labels = [{**row, "label": int(row["label"]), "frames": json.loads(row["frames"])}
              for row in labels]
    split = stratified_split(labels, args.seed)
    (output / "split.json").write_text(json.dumps(
        {key: [item["episode"] for item in values] for key, values in split.items()}, indent=2) + "\n")

    device = torch.device(args.device if args.device != "mps" or
                          (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
                          else "cpu")
    from parking_lib.learning import MOBILENET_V2_ARCH, ParkingPolicy, load_checkpoint
    from parking_lib.slot_classifier import SlotClassifier
    checkpoint = load_checkpoint(args.policy_checkpoint, "cpu")
    if checkpoint["architecture"] != MOBILENET_V2_ARCH:
        raise ValueError("The reusable policy encoder is not MobileNetV2")
    policy = ParkingPolicy(False, MOBILENET_V2_ARCH)
    policy.load_state_dict(checkpoint["state_dict"])
    model = SlotClassifier(policy.encoder).to(device)

    def set_stage(finetune):
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
        if finetune:
            for parameter in model.encoder.features[-4:].parameters():
                parameter.requires_grad = True

    counts = {label: sum(item["label"] == label for item in split["train"]) for label in (0, 1)}
    positive_weight = torch.tensor([counts[0] / float(counts[1])], device=device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    loaders = {
        key: DataLoader(StartFrames(values, args.dataset, key == "train", args.seed),
                        batch_size=args.batch_size, shuffle=(key == "train"), num_workers=0)
        for key, values in split.items()}
    history, best, best_state = [], -1., None
    started = time.time()
    for finetune, epochs in ((False, args.epochs), (True, args.finetune_epochs)):
        set_stage(finetune)
        groups = [{"params": model.head.parameters(), "lr": 3e-4}]
        if finetune:
            groups.append({"params": model.encoder.features[-4:].parameters(), "lr": 1e-5})
        optimizer = torch.optim.Adam(groups, weight_decay=1e-4)
        for epoch in range(epochs):
            model.train(True)
            running, count = 0., 0
            for images, target, _ in loaders["train"]:
                images, target = images.to(device), target.to(device)
                loss = loss_function(model(images), target)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                running += float(loss.item()) * len(target)
                count += len(target)
            val_loss, val_target, val_probability, _ = evaluate(
                model, loaders["val"], device, loss_function)
            val_metrics = metrics(val_target, val_probability)
            row = {"stage": "finetune" if finetune else "frozen", "epoch": epoch + 1,
                   "train_loss": running / count, "val_loss": val_loss,
                   **{("val_" + key): value for key, value in val_metrics.items()
                      if key != "confusion"}}
            history.append(row)
            print(row, flush=True)
            if val_metrics["balanced_accuracy"] > best or (
                    val_metrics["balanced_accuracy"] == best and
                    (best_state is None or val_loss < best_state[0])):
                best = val_metrics["balanced_accuracy"]
                best_state = (val_loss, {key: value.detach().cpu().clone()
                                         for key, value in model.state_dict().items()})

    model.load_state_dict(best_state[1])
    test_loss, target, probability, episodes = evaluate(model, loaders["test"], device, loss_function)
    report = metrics(target, probability)
    report.update(test_loss=test_loss, threshold=.5, episodes=len(labels), frames_per_episode=3,
                  class_counts={"left": sum(item["label"] == 1 for item in labels),
                                "other": sum(item["label"] == 0 for item in labels)},
                  split_counts={key: len(value) for key, value in split.items()},
                  device=str(device), elapsed_seconds=time.time() - started,
                  architecture="mobilenet-v2-mean3-mlp128", encoder_source=str(args.policy_checkpoint))
    predictions = [{"episode": episode, "target": int(y), "probability_left": float(p)}
                   for episode, y, p in zip(episodes, target, probability)]
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (output / "test_predictions.json").write_text(json.dumps(predictions, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    payload = {"schema_version": 1, "architecture": report["architecture"],
               "state_dict": best_state[1], "threshold": .5,
               "frames_per_episode": 3, "image_size": 224,
               "class_names": ["other", "left"], "metrics": report}
    torch.save(payload, str(output / "slot_left_classifier.pt"),
               _use_new_zipfile_serialization=False)

    epochs = np.arange(1, len(history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, [x["train_loss"] for x in history], label="train")
    axes[0].plot(epochs, [x["val_loss"] for x in history], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Weighted BCE loss", title="Slot classifier loss")
    axes[0].legend(); axes[0].grid(alpha=.25)
    confusion = report["confusion"]
    matrix = np.array([[confusion["tn"], confusion["fp"]],
                       [confusion["fn"], confusion["tp"]]])
    axes[1].imshow(matrix, cmap="Blues")
    axes[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["other", "left"],
                yticklabels=["other", "left"], xlabel="Predicted", ylabel="True",
                title="Held-out episode confusion matrix")
    for row in range(2):
        for col in range(2):
            axes[1].text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=15)
    fig.tight_layout(); fig.savefig(str(output / "training_report.png"), dpi=180)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
