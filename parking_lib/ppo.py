"""Recurrent PPO initialized from behavior cloning, plus persistent updates."""
import hashlib
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
from .core import save_json
from .learning import RESNET18_ARCH, ParkingPolicy, load_checkpoint


class PPOPolicy(nn.Module):
    """Tanh-Gaussian actor and value head sharing the imitation network."""
    def __init__(self, imitation_state, action_scale=(1., 1.), exploration_std=.08,
                 train_projection=False, train_gru=False, architecture=RESNET18_ARCH):
        super(PPOPolicy, self).__init__()
        if len(action_scale) != 2 or any(not 0 < float(value) <= 1 for value in action_scale):
            raise ValueError("PPO action scales must be in (0, 1]")
        if not 0 < exploration_std < 1:
            raise ValueError("Exploration standard deviation must be in (0, 1)")
        base = ParkingPolicy(pretrained=False, architecture=architecture)
        base.load_state_dict(imitation_state)
        self.encoder = base.encoder
        self.projection = base.projection
        self.gru = base.gru
        self.actor_fc = base.head[0]
        self.actor_out = base.head[2]
        self.value_fc = nn.Linear(64, 32)
        self.value_out = nn.Linear(32, 1)
        nn.init.orthogonal_(self.value_fc.weight, gain=math.sqrt(2.))
        nn.init.zeros_(self.value_fc.bias)
        nn.init.zeros_(self.value_out.weight)
        nn.init.zeros_(self.value_out.bias)
        self.log_std = nn.Parameter(torch.full((2,), math.log(exploration_std)))
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        self.train_projection = bool(train_projection)
        self.train_gru = bool(train_gru)
        self.set_trainable()

    def set_trainable(self):
        for parameter in self.parameters():
            parameter.requires_grad = False
        modules = [self.actor_fc, self.actor_out, self.value_fc, self.value_out]
        if self.train_gru:
            modules.append(self.gru)
        if self.train_projection:
            modules.append(self.projection)
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
        self.log_std.requires_grad = True

    def train(self, mode=True):
        super(PPOPolicy, self).train(mode)
        self.encoder.eval()
        return self

    def recurrent(self, images, previous_actions, hidden=None):
        batch, steps = images.shape[:2]
        features = self.encoder(images.reshape(-1, 3, 224, 224))
        features = features.to(dtype=self.projection.weight.dtype)
        embedding = self.projection(features).reshape(batch, steps, 256)
        return self.gru(torch.cat((embedding, previous_actions), dim=-1), hidden)

    def actor_value_recurrent(self, recurrent):
        actor_hidden = torch.relu(self.actor_fc(recurrent))
        location = self.actor_out(actor_hidden)
        value = self.value_out(torch.relu(self.value_fc(recurrent))).squeeze(-1)
        return location, value

    def actor_value(self, images, previous_actions, hidden=None):
        recurrent, hidden = self.recurrent(images, previous_actions, hidden)
        location, value = self.actor_value_recurrent(recurrent)
        return location, value, hidden

    def act_recurrent(self, recurrent, deterministic=False):
        location, value = self.actor_value_recurrent(recurrent)
        distribution = Normal(location, self.log_std.exp().expand_as(location))
        raw_action = location if deterministic else distribution.sample()
        action = torch.tanh(raw_action) * self.action_scale
        log_prob = self._log_prob(distribution, raw_action)
        return action, raw_action, log_prob, value

    def _log_prob(self, distribution, raw_action):
        squashed = torch.tanh(raw_action)
        jacobian = self.action_scale * (1. - squashed.pow(2)) + 1e-6
        return (distribution.log_prob(raw_action) - torch.log(jacobian)).sum(dim=-1)

    def act(self, images, previous_actions, hidden=None, deterministic=False):
        recurrent, hidden = self.recurrent(images, previous_actions, hidden)
        action, raw_action, log_prob, value = self.act_recurrent(recurrent, deterministic)
        return action, raw_action, log_prob, value, hidden

    def evaluate_recurrent(self, recurrent, raw_actions):
        location, value = self.actor_value_recurrent(recurrent)
        distribution = Normal(location, self.log_std.exp().expand_as(location))
        log_prob = self._log_prob(distribution, raw_actions)
        entropy = distribution.entropy().sum(dim=-1)
        return log_prob, entropy, value

    def evaluate_raw_actions(self, images, previous_actions, raw_actions, hidden=None):
        recurrent, hidden = self.recurrent(images, previous_actions, hidden)
        log_prob, entropy, value = self.evaluate_recurrent(recurrent, raw_actions)
        return log_prob, entropy, value, hidden


def file_sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def initialize(imitation_checkpoint, output, online_config, device="cpu"):
    """Create version zero without overwriting an existing PPO run."""
    imitation_checkpoint = Path(imitation_checkpoint)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    saved = load_checkpoint(imitation_checkpoint, device)
    scale = online_config["action_scale"]
    architecture = saved.get("architecture", RESNET18_ARCH)
    model = PPOPolicy(saved["state_dict"],
                      (float(scale["steering"]), float(scale["throttle"])),
                      float(online_config["algorithm"]["initial_exploration_std"]),
                      bool(online_config["algorithm"].get("train_projection", False)),
                      bool(online_config["algorithm"].get("train_gru", False)),
                      architecture).to(device)
    gain = online_config["algorithm"].get("initial_actor_logit_gain", {})
    gains = torch.tensor([float(gain.get("steering", 1.)), float(gain.get("throttle", 1.))],
                         dtype=model.actor_out.weight.dtype, device=device)
    if bool((gains <= 0).any()):
        raise ValueError("Initial actor logit gains must be positive")
    with torch.no_grad():
        model.actor_out.weight.mul_(gains[:, None])
        model.actor_out.bias.mul_(gains)
    payload = {"schema_version": 1, "algorithm": "recurrent_ppo", "update": 0,
               "architecture": architecture + "-tanh_gaussian_actor-value",
               "imitation_architecture": architecture,
               "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
               "imitation_checkpoint_sha256": file_sha256(imitation_checkpoint),
               "imitation_best_val_loss": saved.get("best_val_loss"),
               "config": saved["config"], "online_config": online_config,
               "source_split": saved.get("split"), "optimizer_state_dict": None,
               "update_metrics": None}
    checkpoint = output / "ppo_0000.pt"
    torch.save(payload, str(checkpoint))
    save_json(output / "online_config.json", online_config)
    return checkpoint


def load_policy(checkpoint, device="cpu"):
    # Keep the checkpoint payload and optimizer state in host memory. Loading the
    # whole payload onto CUDA duplicates the frozen visual trunk and can starve
    # TensorRT/cuDNN when the CSI camera also owns NVMM buffers on a 4 GB Nano.
    saved = load_checkpoint(checkpoint, "cpu")
    if saved.get("algorithm") != "recurrent_ppo":
        raise ValueError("Not a recurrent PPO checkpoint")
    online = saved["online_config"]
    scale = online["action_scale"]
    # Build through a temporary imitation-shaped model only to allocate modules.
    architecture = saved.get("imitation_architecture", RESNET18_ARCH)
    template = ParkingPolicy(pretrained=False, architecture=architecture).state_dict()
    model = PPOPolicy(template, (float(scale["steering"]), float(scale["throttle"])),
                      float(online["algorithm"]["initial_exploration_std"]),
                      bool(online["algorithm"].get("train_projection", False)),
                      bool(online["algorithm"].get("train_gru", False)),
                      architecture).to(device)
    model.load_state_dict(saved["state_dict"])
    return model, saved


def generalized_advantage(rewards, values, next_values, terminated, truncated,
                          gamma=.99, gae_lambda=.95):
    """GAE with bootstrap on time limits and no leakage across episode ends."""
    advantages = torch.zeros_like(rewards)
    following = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for index in range(rewards.numel() - 1, -1, -1):
        bootstrap = 1. - terminated[index].to(rewards.dtype)
        delta = rewards[index] + gamma * next_values[index] * bootstrap - values[index]
        episode_continues = 1. - (terminated[index] | truncated[index]).to(rewards.dtype)
        following = delta + gamma * gae_lambda * episode_continues * following
        advantages[index] = following
    return advantages, advantages + values


def clipped_objective(new_log_prob, old_log_prob, advantages, values, returns, entropy,
                      clip_ratio=.2, value_weight=.5, entropy_weight=.001):
    normalized = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    ratio = torch.exp(new_log_prob - old_log_prob)
    unclipped = ratio * normalized
    clipped = torch.clamp(ratio, 1. - clip_ratio, 1. + clip_ratio) * normalized
    policy_loss = -torch.min(unclipped, clipped).mean()
    value_loss = (values - returns).pow(2).mean()
    entropy_mean = entropy.mean()
    total = policy_loss + value_weight * value_loss - entropy_weight * entropy_mean
    approx_kl = (old_log_prob - new_log_prob).mean().detach()
    clipped_fraction = ((ratio - 1.).abs() > clip_ratio).to(values.dtype).mean().detach()
    return total, {"policy_loss": policy_loss.detach(), "value_loss": value_loss.detach(),
                   "entropy": entropy_mean.detach(), "approx_kl": approx_kl,
                   "clipped_fraction": clipped_fraction}


ROLLOUT_FIELDS = ("recurrent", "actions", "applied_actions", "raw_actions", "old_log_probs",
                  "values", "next_values", "rewards", "terminated", "truncated")


def validate_rollout(rollout, checkpoint_saved, checkpoint_sha256=None):
    """Validate one on-policy episode recorded from the frozen recurrent trunk."""
    if rollout.get("schema_version") != 1 or rollout.get("artifact_type") != "ppo_rollout":
        raise ValueError("Not a PPO rollout artifact")
    if int(rollout.get("policy_update", -1)) != int(checkpoint_saved["update"]):
        raise ValueError("Rollout was collected by a different PPO update")
    if checkpoint_sha256 and rollout.get("policy_checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Rollout checkpoint hash does not match the PPO checkpoint")
    if checkpoint_saved["online_config"]["algorithm"].get("train_gru", False):
        raise ValueError("Feature rollouts require train_gru=false")
    if checkpoint_saved["online_config"]["algorithm"].get("train_projection", False):
        raise ValueError("Feature rollouts require train_projection=false")
    missing = [key for key in ROLLOUT_FIELDS if key not in rollout]
    if missing:
        raise ValueError("Rollout is missing fields: " + ", ".join(missing))
    length = int(rollout["rewards"].numel())
    if length < 1:
        raise ValueError("Rollout contains no transitions")
    shapes = {"recurrent": (length, 64), "actions": (length, 2),
              "applied_actions": (length, 2),
              "raw_actions": (length, 2), "old_log_probs": (length,),
              "values": (length,), "next_values": (length,), "rewards": (length,),
              "terminated": (length,), "truncated": (length,)}
    for key, shape in shapes.items():
        tensor = rollout[key]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError("Invalid rollout shape for {}: expected {}".format(key, shape))
        if key not in ("terminated", "truncated") and not torch.isfinite(tensor).all():
            raise ValueError("Non-finite rollout tensor: " + key)
    terminated = rollout["terminated"].to(torch.bool)
    truncated = rollout["truncated"].to(torch.bool)
    if bool((terminated & truncated).any()):
        raise ValueError("A transition cannot be both terminated and truncated")
    if not bool(terminated[-1] or truncated[-1]):
        raise ValueError("The last rollout transition must end the episode")
    if bool(terminated[:-1].any() or truncated[:-1].any()):
        raise ValueError("Episode-ending flags may appear only on the final transition")
    scale_config = checkpoint_saved["online_config"]["action_scale"]
    scale = torch.tensor([float(scale_config["steering"]), float(scale_config["throttle"])])
    expected_actions = torch.tanh(rollout["raw_actions"].float()) * scale
    if not torch.allclose(rollout["actions"].float(), expected_actions, atol=1e-5, rtol=1e-5):
        raise ValueError("Saved PPO actions do not match their pre-tanh samples")
    transform = rollout.get("runtime_action_transform", {})
    steering_scale = float(transform.get("steering_scale", 1.))
    throttle_scale = float(transform.get("throttle_scale", 1.))
    if not math.isfinite(steering_scale) or steering_scale <= 0.:
        raise ValueError("Invalid rollout steering scale")
    if not math.isfinite(throttle_scale) or not 0. < throttle_scale <= 1.:
        raise ValueError("Invalid rollout throttle scale")
    cfg = checkpoint_saved["config"]
    steering_offset = float(transform.get(
        "steering_offset", cfg["hardware"]["steering_offset"]))
    steering_limit = float(transform.get(
        "steering_limit", cfg["limits"]["steering"]))
    steering_gain = float(cfg["hardware"]["steering_gain"])
    if (not math.isfinite(steering_offset) or not math.isfinite(steering_limit) or
            not 0. < steering_limit <= 1. or
            abs(steering_offset) + abs(steering_gain) * steering_limit > 1.000001):
        raise ValueError("Invalid rollout steering calibration")
    expected_applied = rollout["actions"].float().clone()
    expected_applied[:, 0] = (expected_applied[:, 0] * steering_scale).clamp(-1., 1.)
    expected_applied[:, 1] = expected_applied[:, 1] * throttle_scale
    physical_error = (rollout["applied_actions"].float() - expected_applied).abs().max()
    if float(physical_error) > .021:
        raise ValueError("Applied motor action differs from the recorded runtime transform")
    return length


def load_rollouts(paths, checkpoint, checkpoint_saved):
    checkpoint_hash = file_sha256(checkpoint)
    result = []
    runtime_transform = None
    for path in paths:
        rollout = load_checkpoint(path, "cpu")
        validate_rollout(rollout, checkpoint_saved, checkpoint_hash)
        transform = rollout.get("runtime_action_transform", {})
        cfg = checkpoint_saved["config"]
        signature = (
            float(transform.get("steering_scale", 1.)),
            float(transform.get("throttle_scale", 1.)),
            float(transform.get("steering_offset", cfg["hardware"]["steering_offset"])),
            float(transform.get("steering_limit", cfg["limits"]["steering"])),
        )
        if runtime_transform is None:
            runtime_transform = signature
        elif signature != runtime_transform:
            raise ValueError("PPO rollouts for one update use different runtime action settings")
        rollout["source_path"] = str(path)
        result.append(rollout)
    return result


def _optimizer(model, algorithm, state=None):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=float(algorithm["learning_rate"]))
    if state:
        optimizer.load_state_dict(state)
    return optimizer


def update_policy(model, rollouts, online_config, device="cpu", optimizer_state=None):
    """Run PPO epochs over complete on-policy episodes from a frozen trunk."""
    if not rollouts:
        raise ValueError("No rollouts supplied")
    algorithm = online_config["algorithm"]
    recurrent_parts, raw_parts, old_log_parts, advantage_parts, return_parts = [], [], [], [], []
    total_reward = 0.
    for rollout in rollouts:
        rewards = rollout["rewards"].to(device=device, dtype=torch.float32)
        values = rollout["values"].to(device=device, dtype=torch.float32)
        next_values = rollout["next_values"].to(device=device, dtype=torch.float32)
        terminated = rollout["terminated"].to(device=device, dtype=torch.bool)
        truncated = rollout["truncated"].to(device=device, dtype=torch.bool)
        advantage, returns = generalized_advantage(
            rewards, values, next_values, terminated, truncated,
            float(algorithm["gamma"]), float(algorithm["gae_lambda"]))
        recurrent_parts.append(rollout["recurrent"].to(dtype=torch.float32))
        raw_parts.append(rollout["raw_actions"].to(dtype=torch.float32))
        old_log_parts.append(rollout["old_log_probs"].to(dtype=torch.float32))
        advantage_parts.append(advantage.cpu())
        return_parts.append(returns.cpu())
        total_reward += float(rewards.sum().cpu())
    tensors = tuple(torch.cat(parts) for parts in (
        recurrent_parts, raw_parts, old_log_parts, advantage_parts, return_parts))
    with torch.no_grad():
        expected_log, _, expected_values = model.evaluate_recurrent(
            tensors[0].to(device), tensors[1].to(device))
    if not torch.allclose(expected_log.cpu(), tensors[2], atol=5e-4, rtol=1e-4):
        raise ValueError("Rollout log probabilities do not match the checkpoint policy")
    recorded_values = torch.cat([rollout["values"].float() for rollout in rollouts])
    if not torch.allclose(expected_values.cpu(), recorded_values, atol=5e-4, rtol=1e-4):
        raise ValueError("Rollout values do not match the checkpoint policy")
    dataset = TensorDataset(*tensors)
    batch_size = max(1, int(algorithm["sequence_length"]) * int(algorithm["sequences_per_batch"]))
    generator = torch.Generator().manual_seed(int(online_config.get("seed", 7)) + len(rollouts))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    optimizer = _optimizer(model, algorithm, optimizer_state)
    model.train()
    history = []
    for epoch in range(int(algorithm["update_epochs"])):
        sums = {key: 0. for key in ("loss", "policy_loss", "value_loss", "entropy",
                                    "approx_kl", "clipped_fraction")}
        count = 0
        for recurrent, raw, old_log, advantages, returns in loader:
            recurrent, raw = recurrent.to(device), raw.to(device)
            old_log, advantages, returns = old_log.to(device), advantages.to(device), returns.to(device)
            new_log, entropy, value = model.evaluate_recurrent(recurrent, raw)
            loss, metrics = clipped_objective(
                new_log, old_log, advantages, value, returns, entropy,
                float(algorithm["clip_ratio"]), float(algorithm["value_weight"]),
                float(algorithm["entropy_weight"]))
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                float(algorithm["max_grad_norm"]))
            if not torch.isfinite(grad_norm):
                raise RuntimeError("Non-finite PPO gradient")
            optimizer.step()
            minimum_std = float(algorithm.get("minimum_exploration_std", .02))
            maximum_std = float(algorithm.get("maximum_exploration_std", .30))
            with torch.no_grad():
                model.log_std.clamp_(math.log(minimum_std), math.log(maximum_std))
            size = len(recurrent)
            sums["loss"] += float(loss.detach().cpu()) * size
            for key in metrics:
                sums[key] += float(metrics[key].cpu()) * size
            count += size
        history.append(dict({"epoch": epoch + 1}, **{key: value / count for key, value in sums.items()}))
    result = {"episodes": len(rollouts), "transitions": len(dataset),
              "mean_episode_reward": total_reward / len(rollouts), "epochs": history,
              "log_std": model.log_std.detach().cpu().tolist()}
    return result, optimizer.state_dict()


def save_policy_update(model, previous_saved, previous_checkpoint, metrics, optimizer_state):
    """Save the next numbered checkpoint atomically, retaining the frozen encoder."""
    previous_checkpoint = Path(previous_checkpoint)
    update = int(previous_saved["update"]) + 1
    destination = previous_checkpoint.parent / "ppo_{:04d}.pt".format(update)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite: " + str(destination))
    current = model.state_dict()
    state = {}
    for key, value in previous_saved["state_dict"].items():
        state[key] = current[key].detach().cpu() if key in current and current[key].shape == value.shape else value
    def cpu_tree(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: cpu_tree(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cpu_tree(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cpu_tree(item) for item in value)
        return value
    payload = dict(previous_saved)
    payload.update(update=update, state_dict=state, optimizer_state_dict=optimizer_state,
                   update_metrics=metrics, parent_checkpoint_sha256=file_sha256(previous_checkpoint))
    payload["optimizer_state_dict"] = cpu_tree(payload["optimizer_state_dict"])
    temporary = destination.with_suffix(".tmp")
    torch.save(payload, str(temporary))
    temporary.replace(destination)
    history_path = destination.parent / "ppo_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(dict({"update": update, "checkpoint": destination.name}, **metrics))
    save_json(history_path, history)
    return destination
