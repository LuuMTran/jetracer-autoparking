"""Persistent physical rollout collection and PPO checkpoint updates."""
import json
import math
import os
import copy
import select
import shutil
import sys
import time
import uuid
from pathlib import Path

import torch

from .camera import Camera
from .core import Car, normalized
from .learning import StopHead, image_tensor, load_checkpoint
from . import ppo


SCORES = {
    "g": (1.0, "success", True),
    "n": (0.5, "near", False),
    "z": (0.0, "neutral", False),
    "p": (-0.5, "poor", False),
    "x": (-1.0, "collision", True),
}

CHECKPOINTS_TO_KEEP = 3


def parse_reward(text):
    """A scored attempt ends the task; optional 't' marks an unfinished truncation."""
    fields = text.strip().lower().split()
    if len(fields) not in (1, 2) or (len(fields) == 2 and fields[1] != "t"):
        raise ValueError("Enter a number from -1 to 1, optionally followed by t")
    reward = float(fields[0])
    if not math.isfinite(reward) or not -1. <= reward <= 1.:
        raise ValueError("Reward must be a finite number from -1 to 1")
    return {"reward": reward, "outcome": "human_scored", "terminated": len(fields) == 1}


class TerminalKeys(object):
    """Single-key terminal input for the autonomous episode loop."""
    def __init__(self):
        self.state = None
        self.buffer = ""

    def __enter__(self):
        if not sys.stdin.isatty():
            raise ValueError("The PPO loop requires an SSH/terminal TTY")
        import termios
        import tty
        self.state = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def read(self, timeout=0.):
        if self.buffer:
            key, self.buffer = self.buffer[0], self.buffer[1:]
            return key.lower()
        ready = select.select([sys.stdin], [], [], timeout)[0]
        if not ready:
            return None
        data = os.read(sys.stdin.fileno(), 64).decode("ascii", errors="ignore")
        if not data:
            return "\x1b"
        key, self.buffer = data[0], data[1:]
        return key.lower()

    def wait_start(self):
        print("READY: reposition the stopped car; Enter starts an episode; Escape exits.", flush=True)
        while True:
            key = self.read(.1)
            if key == "\x1b":
                return False
            if key in ("\r", "\n"):
                return True

    def poll_episode(self):
        result = None
        while True:
            key = self.read(0.)
            if key is None:
                return result
            if key == "\x1b":
                return "escape"
            if key == "f":
                result = "finish"

    def wait_score(self):
        print("REWARD: type -1 to +1 and Enter (e.g. -0.7 or 0.35). "
              "Append t for an unfinished attempt (0.35 t). D discards; Escape exits.", flush=True)
        print("> ", end="", flush=True)
        entered = ""
        while True:
            key = self.read(.1)
            if key is None:
                continue
            if key == "\x1b":
                print(flush=True)
                return "exit"
            if key == "d" and not entered:
                print("discard", flush=True)
                return "discard"
            if key in ("\r", "\n"):
                print(flush=True)
                try:
                    return parse_reward(entered)
                except ValueError as error:
                    print(str(error), flush=True)
                    entered = ""
                    print("> ", end="", flush=True)
            elif key in ("\x7f", "\b") and entered:
                entered = entered[:-1]
                print("\b \b", end="", flush=True)
            elif key == "\x15":
                print("\b \b" * len(entered), end="", flush=True)
                entered = ""
            elif key.isprintable() and len(entered) < 32:
                entered += key
                print(key, end="", flush=True)

    def __exit__(self, *exc):
        if self.state is not None:
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.state)


def attach_tensorrt_encoder(model, artifact_path, checkpoint_saved, device):
    if not str(device).startswith("cuda"):
        raise ValueError("TensorRT encoder requires a CUDA device")
    artifact = load_checkpoint(artifact_path, device)
    if artifact.get("checkpoint_sha256") != checkpoint_saved["imitation_checkpoint_sha256"]:
        raise ValueError("TensorRT encoder was built from a different imitation checkpoint")
    if artifact.get("architecture") != checkpoint_saved["imitation_architecture"]:
        raise ValueError("TensorRT encoder architecture does not match PPO")
    if artifact.get("precision") != "fp16" or artifact.get("input_shape") != [1, 3, 224, 224]:
        raise ValueError("Unsupported TensorRT encoder artifact")
    from torch2trt import TRTModule
    encoder = TRTModule()
    encoder.load_state_dict(artifact["state_dict"])
    model.encoder = encoder
    torch.cuda.empty_cache()
    return "tensorrt-fp16-encoder"


def load_stop_head(path, checkpoint_saved, device):
    if path is None:
        return None, {}
    artifact = load_checkpoint(path, device)
    if artifact.get("artifact_type") != "recurrent_stop_head":
        raise ValueError("Not a recurrent stop-head artifact")
    if artifact.get("checkpoint_sha256") != checkpoint_saved["imitation_checkpoint_sha256"]:
        raise ValueError("Stop head was trained from a different imitation checkpoint")
    if artifact.get("architecture") != checkpoint_saved["imitation_architecture"]:
        raise ValueError("Stop-head architecture does not match PPO")
    head = StopHead().to(device).eval()
    head.load_state_dict(artifact["state_dict"])
    return head, dict(artifact.get("decision", {}))


def rollout_directory(run_dir, update):
    path = Path(run_dir) / "rollouts" / "update_{:04d}".format(int(update))
    path.mkdir(parents=True, exist_ok=True)
    return path


def rollout_paths(run_dir, update):
    return sorted(rollout_directory(run_dir, update).glob("episode_*.pt"))


def prune_completed_update(run_dir, completed_update, checkpoint_dir):
    """Delete consumed rollouts and retain the initial plus three newest policies."""
    consumed = Path(run_dir) / "rollouts" / "update_{:04d}".format(int(completed_update))
    if consumed.exists():
        shutil.rmtree(str(consumed))
    checkpoints = sorted(
        Path(checkpoint_dir).glob("ppo_[0-9][0-9][0-9][0-9].pt"),
        key=lambda path: int(path.stem.split("_")[-1]))
    keep = set(checkpoints[-CHECKPOINTS_TO_KEEP:])
    keep.update(path for path in checkpoints if path.name == "ppo_0000.pt")
    deleted = []
    for path in checkpoints:
        if path not in keep:
            path.unlink()
            deleted.append(path.name)
    return deleted


def latest_checkpoint(checkpoint, run_dir):
    """Resume the newest numbered policy in the run directory."""
    requested = Path(checkpoint)
    run_dir = Path(run_dir)
    candidates = list(run_dir.glob("ppo_[0-9][0-9][0-9][0-9].pt"))
    if requested.is_dir():
        candidates.extend(requested.glob("ppo_[0-9][0-9][0-9][0-9].pt"))
    elif requested.exists():
        candidates.append(requested)
    if not candidates:
        raise FileNotFoundError("No PPO checkpoint found in {} or {}".format(requested, run_dir))

    def update_number(path):
        try:
            return int(path.stem.split("_")[-1])
        except (TypeError, ValueError):
            return -1
    return max(set(path.resolve() for path in candidates), key=update_number)


def inference_checkpoint(checkpoint):
    """Use an exact policy file, or select the newest policy in a directory."""
    requested = Path(checkpoint)
    if requested.is_file():
        return requested.resolve()
    if requested.is_dir():
        return latest_checkpoint(requested, requested)
    raise FileNotFoundError("PPO checkpoint does not exist: {}".format(requested))


def save_rollout(run_dir, rollout):
    directory = rollout_directory(run_dir, rollout["policy_update"])
    name = "episode_{}_{}.pt".format(time.strftime("%Y%m%d_%H%M%S"), uuid.uuid4().hex[:8])
    destination = directory / name
    temporary = destination.with_suffix(".tmp")
    torch.save(rollout, str(temporary))
    temporary.replace(destination)
    return destination


def _image_input(rgb, device, backend):
    dtype = torch.float16 if backend in (
        "tensorrt-fp16-encoder", "pytorch-fp16-encoder") else torch.float32
    return image_tensor(rgb)[None, None].to(device=device, dtype=dtype)


def _previous_input(previous, cfg, device):
    return torch.tensor(normalized(previous, cfg["limits"]), dtype=torch.float32,
                        device=device)[None, None]


def _require_finite_recurrent(recurrent, backend):
    if not torch.isfinite(recurrent).all():
        raise RuntimeError(
            "Non-finite recurrent features from {}; the CUDA/TensorRT execution failed".format(
                backend))


def _configure_inference_backend(model, saved, device, trt_encoder=None,
                                 fp16_encoder=False):
    if trt_encoder is not None and fp16_encoder:
        raise ValueError("Choose either --trt-encoder or --fp16-encoder")
    if trt_encoder is not None:
        return attach_tensorrt_encoder(model, trt_encoder, saved, device)
    if fp16_encoder:
        if not str(device).startswith("cuda"):
            raise ValueError("--fp16-encoder requires a CUDA device")
        model.encoder.half()
        torch.cuda.empty_cache()
        return "pytorch-fp16-encoder"
    return "pytorch"


def _runtime_config(saved, max_frame_age_ms=None):
    cfg = copy.deepcopy(saved["config"])
    if max_frame_age_ms is not None:
        if not 50. <= float(max_frame_age_ms) <= 2000.:
            raise ValueError("--max-frame-age-ms must be from 50 to 2000")
        cfg["control"]["max_frame_age_s"] = float(max_frame_age_ms) / 1000.
    print("Frame-age limit: {:.0f}ms".format(
        cfg["control"]["max_frame_age_s"] * 1000.), flush=True)
    expected_hz = int(saved["online_config"]["rollout"]["control_hz"])
    if int(cfg["control"]["hz"]) != expected_hz:
        raise ValueError("PPO rollout rate differs from the imitation checkpoint")
    return cfg


def apply_runtime_steering_config(cfg, steering_offset=None, steering_limit=None):
    """Apply safe runtime steering calibration overrides to a config copy."""
    checkpoint_offset = float(cfg["hardware"]["steering_offset"])
    checkpoint_limit = float(cfg["limits"]["steering"])
    if steering_limit is not None:
        cfg["limits"]["steering"] = float(steering_limit)
    if steering_offset is not None:
        cfg["hardware"]["steering_offset"] = float(steering_offset)

    offset = float(cfg["hardware"]["steering_offset"])
    gain = float(cfg["hardware"]["steering_gain"])
    requested_limit = float(cfg["limits"]["steering"])
    safe_limit = (1. - abs(offset)) / abs(gain)
    if requested_limit > safe_limit:
        if steering_limit is not None:
            raise ValueError("--steering-limit {:.3f} is unsafe with offset {:+.3f} and "
                             "gain {:+.3f}; maximum safe limit is {:.3f}".format(
                                 requested_limit, offset, gain, safe_limit))
        cfg["limits"]["steering"] = safe_limit
        print("Steering limit reduced from {:.3f} to {:.3f} to keep the driver "
              "within range.".format(requested_limit, safe_limit), flush=True)
    return checkpoint_offset, checkpoint_limit


def _runtime_action(policy_action, steering_scale, throttle_scale, step_index,
                    initial_forward_thrust, initial_forward_steps,
                    initial_forward_cap):
    steering = max(-1., min(1., float(policy_action[0]) * steering_scale))
    throttle = float(policy_action[1]) * throttle_scale
    if step_index < initial_forward_steps and initial_forward_thrust > 0.:
        # Forward is negative on this car. Add a negative startup bias, then
        # limit its forward magnitude so the initial pulse remains bounded.
        throttle = max(-initial_forward_cap, throttle - initial_forward_thrust)
    return steering, max(-1., min(1., throttle))


def _validate_runtime_action_controls(throttle_scale, steering_scale,
                                      initial_forward_thrust,
                                      initial_forward_steps,
                                      initial_forward_cap):
    throttle_scale = float(throttle_scale)
    steering_scale = float(steering_scale)
    initial_forward_thrust = float(initial_forward_thrust)
    initial_forward_steps = int(initial_forward_steps)
    initial_forward_cap = float(initial_forward_cap)
    if not math.isfinite(throttle_scale) or not 0. < throttle_scale <= 1.:
        raise ValueError("Throttle scale must be above 0 and no greater than 1")
    if not math.isfinite(steering_scale) or steering_scale <= 0.:
        raise ValueError("Steering scale must be a positive finite number")
    if not math.isfinite(initial_forward_thrust) or not 0. <= initial_forward_thrust <= 1.:
        raise ValueError("Initial forward thrust must be from 0 to 1")
    if initial_forward_steps < 0:
        raise ValueError("Initial forward steps must be zero or positive")
    if not math.isfinite(initial_forward_cap) or not 0. < initial_forward_cap <= 1.:
        raise ValueError("Initial forward cap must be above 0 and no greater than 1")
    if initial_forward_thrust > 0. and initial_forward_steps < 1:
        raise ValueError("Initial forward steps must be positive when startup thrust is used")
    return (throttle_scale, steering_scale, initial_forward_thrust,
            initial_forward_steps, initial_forward_cap)


def follow_episode(model, cfg, camera, car, keys, device, backend="pytorch",
                   stop_head=None, stop_decision=None, warmup_s=.5,
                   maximum_steps=400, throttle_scale=1., telemetry_callback=None,
                   steering_scale=1., initial_forward_thrust=0.,
                   initial_forward_steps=0, initial_forward_cap=1.):
    """Run the deterministic policy without recording, rewards, or updates."""
    (throttle_scale, steering_scale, initial_forward_thrust,
     initial_forward_steps, initial_forward_cap) = _validate_runtime_action_controls(
         throttle_scale, steering_scale, initial_forward_thrust,
         initial_forward_steps, initial_forward_cap)
    control_hz = float(cfg["control"]["hz"])
    period = 1. / control_hz
    hidden = None
    previous = (0., 0.)
    last_id = -1
    stop_decision = stop_decision or {}
    stop_threshold = float(stop_decision.get("threshold", .65))
    stop_frames = int(stop_decision.get("consecutive_frames", 4))
    stop_min_s = float(stop_decision.get("minimum_elapsed_s", 1.))
    stop_streak = 0
    warmup_frames = max(0, int(round(float(warmup_s) * control_hz)))
    model.eval()
    if warmup_frames:
        print("WARMUP: initializing deterministic policy on {} frames...".format(
            warmup_frames), flush=True)
    with torch.no_grad():
        for warmup_index in range(warmup_frames):
            warmup_started = time.monotonic()
            frame = camera.read(after=last_id, timeout=max(
                2., cfg["control"]["max_frame_age_s"]))
            last_id = frame["frame_id"]
            recurrent, hidden = model.recurrent(
                _image_input(frame["rgb"], device, backend),
                _previous_input(previous, cfg, device), hidden)
            _require_finite_recurrent(recurrent, backend)
            action, raw, log_prob, value = model.act_recurrent(
                recurrent, deterministic=True)
            synchronized = (action.float().sum() + raw.float().sum() +
                            log_prob.float().sum() + value.float().sum())
            if stop_head is not None:
                synchronized = synchronized + torch.sigmoid(
                    stop_head(recurrent)).float().sum()
            float(synchronized.cpu().item())
            print("WARMUP: frame {}/{} {:.0f}ms".format(
                warmup_index + 1, warmup_frames,
                (time.monotonic() - warmup_started) * 1000.), flush=True)

    started = time.monotonic()
    deadline = started
    reason = "duration_limit"
    applied_steps = 0
    print("FOLLOWING: deterministic policy; F ends this run; "
          "Escape emergency-stops and exits.", flush=True)
    if throttle_scale != 1.:
        print("FOLLOWING: throttle output scaled to {:.0f}%.".format(
            throttle_scale * 100.), flush=True)
    if steering_scale != 1.:
        print("FOLLOWING: steering output scaled to {:.0f}%.".format(
            steering_scale * 100.), flush=True)
    if initial_forward_thrust > 0.:
        print("FOLLOWING: adding -{:.3f} forward throttle for the first {} actions; "
              "forward magnitude capped at {:.3f}.".format(
                  initial_forward_thrust, initial_forward_steps,
                  initial_forward_cap), flush=True)
    while applied_steps < int(maximum_steps):
        frame = camera.read(after=last_id, timeout=max(
            2., cfg["control"]["max_frame_age_s"]))
        last_id = frame["frame_id"]
        command_key = keys.poll_episode()
        if command_key == "escape":
            car.park("emergency_stop")
            return "emergency_stop", True, applied_steps

        inference_started = time.monotonic()
        with torch.no_grad():
            recurrent, next_hidden = model.recurrent(
                _image_input(frame["rgb"], device, backend),
                _previous_input(previous, cfg, device), hidden)
            _require_finite_recurrent(recurrent, backend)
            action, raw, log_prob, value = model.act_recurrent(
                recurrent, deterministic=True)
            stop_probability = (float(torch.sigmoid(stop_head(recurrent)).cpu().item())
                                if stop_head is not None else None)
        if not torch.isfinite(action).all() or not torch.isfinite(value).all():
            raise RuntimeError("Non-finite deterministic PPO decision")
        inference_s = time.monotonic() - inference_started
        elapsed = time.monotonic() - started
        current_value = float(value.cpu().item())

        if command_key == "finish":
            reason = "operator_finished"
            break
        if stop_probability is not None and elapsed >= stop_min_s:
            stop_streak = stop_streak + 1 if stop_probability >= stop_threshold else 0
            if stop_streak >= stop_frames:
                reason = "policy_stop"
                break

        frame_age_s = time.monotonic() - frame["frame_monotonic_s"]
        if frame_age_s > cfg["control"]["max_frame_age_s"]:
            raise RuntimeError("Frame too old after policy inference: {:.0f}ms > {:.0f}ms".format(
                frame_age_s * 1000., cfg["control"]["max_frame_age_s"] * 1000.))
        policy_action = tuple(float(item) for item in action[0, 0].cpu().tolist())
        requested = _runtime_action(
            policy_action, steering_scale, throttle_scale, applied_steps,
            initial_forward_thrust, initial_forward_steps, initial_forward_cap)
        command = car.apply(requested)
        applied = tuple(float(item) for item in command["normalized"])
        if max(abs(requested[index] - applied[index]) for index in range(2)) > .02:
            raise RuntimeError("Applied motor command differs from deterministic PPO action")
        previous = command["action"]
        hidden = next_hidden
        applied_steps += 1

        if telemetry_callback is not None:
            telemetry_callback(frame, {
                "step": int(applied_steps),
                "elapsed_s": float(elapsed),
                "steering": float(applied[0]),
                "throttle": float(applied[1]),
                "value": float(current_value),
                "stop_probability": (float(stop_probability)
                                     if stop_probability is not None else None),
                "inference_ms": float(inference_s * 1000.),
                "frame_age_ms": float(frame_age_s * 1000.),
            })

        if applied_steps == 1 or applied_steps % max(1, int(control_hz)) == 0:
            message = ("t={:.1f}s step={} action=({:+.3f},{:+.3f}) value={:+.3f} "
                       "inference={:.0f}ms age={:.0f}ms").format(
                           elapsed, applied_steps, applied[0], applied[1], current_value,
                           inference_s * 1000., frame_age_s * 1000.)
            if stop_probability is not None:
                message += " stop_p={:.3f} streak={}".format(
                    stop_probability, stop_streak)
            print(message, flush=True)
        deadline += period
        if deadline < time.monotonic():
            deadline = time.monotonic()
        time.sleep(max(0., deadline - time.monotonic()))

    car.park(reason)
    return reason, False, applied_steps


def collect_episode(model, saved, checkpoint, cfg, camera, car, keys, device,
                    backend="pytorch", stop_head=None, stop_decision=None,
                    warmup_s=.5, maximum_steps=400, throttle_scale=1.,
                    steering_scale=1., initial_forward_thrust=0.,
                    initial_forward_steps=0, initial_forward_cap=1.):
    """Collect one autonomous on-policy episode and ask for a terminal score."""
    (throttle_scale, steering_scale, initial_forward_thrust,
     initial_forward_steps, initial_forward_cap) = _validate_runtime_action_controls(
         throttle_scale, steering_scale, initial_forward_thrust,
         initial_forward_steps, initial_forward_cap)
    control_hz = float(cfg["control"]["hz"])
    period = 1. / control_hz
    hidden = None
    previous = (0., 0.)
    last_id = -1
    stop_decision = stop_decision or {}
    stop_threshold = float(stop_decision.get("threshold", .65))
    stop_frames = int(stop_decision.get("consecutive_frames", 4))
    stop_min_s = float(stop_decision.get("minimum_elapsed_s", 1.))
    stop_streak = 0
    warmup_frames = max(0, int(round(float(warmup_s) * control_hz)))
    model.eval()
    warmup_latencies = []
    if warmup_frames:
        print("WARMUP: initializing complete policy on {} frames...".format(
            warmup_frames), flush=True)
    with torch.no_grad():
        for warmup_index in range(warmup_frames):
            warmup_started = time.monotonic()
            frame = camera.read(after=last_id, timeout=max(2., cfg["control"]["max_frame_age_s"]))
            last_id = frame["frame_id"]
            image = _image_input(frame["rgb"], device, backend)
            recurrent, hidden = model.recurrent(image, _previous_input(previous, cfg, device), hidden)
            _require_finite_recurrent(recurrent, backend)
            # Warm every CUDA path used by the first real decision. Previously
            # only the visual encoder and GRU ran here, so the actor, critic,
            # distribution and stop head could add a large one-time delay to
            # the first applied action on JetPack 4.
            action, raw, log_prob, value = model.act_recurrent(recurrent)
            synchronized = (action.float().sum() + raw.float().sum() +
                            log_prob.float().sum() + value.float().sum())
            if stop_head is not None:
                synchronized = synchronized + torch.sigmoid(stop_head(recurrent)).float().sum()
            float(synchronized.cpu().item())
            warmup_latencies.append(time.monotonic() - warmup_started)
            print("WARMUP: frame {}/{} {:.0f}ms".format(
                warmup_index + 1, warmup_frames, warmup_latencies[-1] * 1000.), flush=True)
    if warmup_latencies:
        print("WARMUP: frames={} first={:.0f}ms last={:.0f}ms max={:.0f}ms".format(
            len(warmup_latencies), warmup_latencies[0] * 1000.,
            warmup_latencies[-1] * 1000., max(warmup_latencies) * 1000.), flush=True)

    data = {key: [] for key in ppo.ROLLOUT_FIELDS}
    started = time.monotonic()
    deadline = started
    reason = "duration_limit"
    bootstrap_value = 0.
    print("RUNNING: policy is autonomous; F ends for scoring; Escape emergency-stops and exits.", flush=True)
    if throttle_scale != 1.:
        print("RUNNING: throttle output scaled to {:.0f}%.".format(
            throttle_scale * 100.), flush=True)
    if steering_scale != 1.:
        print("RUNNING: steering output scaled to {:.0f}%.".format(
            steering_scale * 100.), flush=True)
    if initial_forward_thrust > 0.:
        print("RUNNING: adding -{:.3f} forward throttle for the first {} actions; "
              "forward magnitude capped at {:.3f}.".format(
                  initial_forward_thrust, initial_forward_steps,
                  initial_forward_cap), flush=True)
    while len(data["rewards"]) < int(maximum_steps):
        frame = camera.read(after=last_id, timeout=max(2., cfg["control"]["max_frame_age_s"]))
        last_id = frame["frame_id"]
        command_key = keys.poll_episode()
        if command_key == "escape":
            car.park("emergency_stop")
            return None, True
        image = _image_input(frame["rgb"], device, backend)
        previous_input = _previous_input(previous, cfg, device)
        inference_started = time.monotonic()
        with torch.no_grad():
            recurrent, next_hidden = model.recurrent(image, previous_input, hidden)
            _require_finite_recurrent(recurrent, backend)
            action, raw, log_prob, value = model.act_recurrent(recurrent)
            stop_probability = (float(torch.sigmoid(stop_head(recurrent)).cpu().item())
                                if stop_head is not None else None)
        inference_s = time.monotonic() - inference_started
        current_value = float(value.cpu().item())
        if data["next_values"]:
            data["next_values"][-1] = current_value
        elapsed = time.monotonic() - started
        if command_key == "finish":
            reason, bootstrap_value = "operator_finished", current_value
            break
        if stop_probability is not None and elapsed >= stop_min_s:
            stop_streak = stop_streak + 1 if stop_probability >= stop_threshold else 0
            if stop_streak >= stop_frames:
                reason, bootstrap_value = "policy_stop", current_value
                break
        hidden = next_hidden
        frame_age_s = time.monotonic() - frame["frame_monotonic_s"]
        if frame_age_s > cfg["control"]["max_frame_age_s"]:
            raise RuntimeError("Frame too old after PPO inference: {:.0f}ms > {:.0f}ms".format(
                frame_age_s * 1000., cfg["control"]["max_frame_age_s"] * 1000.))
        sampled = tuple(float(value) for value in action[0, 0].cpu().tolist())
        requested = _runtime_action(
            sampled, steering_scale, throttle_scale, len(data["rewards"]),
            initial_forward_thrust, initial_forward_steps, initial_forward_cap)
        command = car.apply(requested)
        applied = tuple(float(value) for value in command["normalized"])
        if max(abs(requested[index] - applied[index]) for index in range(2)) > .02:
            raise RuntimeError("Applied motor command differs from scaled PPO action")
        previous = command["action"]
        data["recurrent"].append(recurrent[0, 0].float().cpu())
        data["actions"].append(torch.tensor(sampled, dtype=torch.float32))
        data["applied_actions"].append(torch.tensor(applied, dtype=torch.float32))
        data["raw_actions"].append(raw[0, 0].float().cpu())
        data["old_log_probs"].append(log_prob[0, 0].float().cpu())
        data["values"].append(value[0, 0].float().cpu())
        data["next_values"].append(0.)
        step_penalty = float(saved["online_config"]["reward"]["weights"].get("step_penalty", 0.))
        data["rewards"].append(-step_penalty)
        data["terminated"].append(False)
        data["truncated"].append(False)
        if len(data["rewards"]) == 1 or len(data["rewards"]) % max(1, int(control_hz)) == 0:
            message = ("t={:.1f}s step={} action=({:+.3f},{:+.3f}) value={:+.3f} "
                       "inference={:.0f}ms age={:.0f}ms").format(
                elapsed, len(data["rewards"]), applied[0], applied[1], current_value,
                inference_s * 1000., frame_age_s * 1000.)
            if stop_probability is not None:
                message += " stop_p={:.3f} streak={}".format(stop_probability, stop_streak)
            print(message, flush=True)
        deadline += period
        if deadline < time.monotonic():
            deadline = time.monotonic()
        time.sleep(max(0., deadline - time.monotonic()))
    else:
        reason = "duration_limit"
        frame = camera.read(after=last_id, timeout=max(2., cfg["control"]["max_frame_age_s"]))
        last_id = frame["frame_id"]
        with torch.no_grad():
            recurrent, _ = model.recurrent(
                _image_input(frame["rgb"], device, backend),
                _previous_input(previous, cfg, device), hidden)
            _require_finite_recurrent(recurrent, backend)
            _, bootstrap = model.actor_value_recurrent(recurrent)
        bootstrap_value = float(bootstrap.cpu().item())
        if data["next_values"]:
            data["next_values"][-1] = bootstrap_value

    car.park(reason)
    if not data["rewards"]:
        print("Episode contained no applied policy actions; discarded.", flush=True)
        return None, False
    label = keys.wait_score()
    if label == "exit":
        return None, True
    if label == "discard":
        print("Episode discarded.", flush=True)
        return None, False
    if isinstance(label, dict):
        terminal_score, outcome, true_terminal = label["reward"], label["outcome"], label["terminated"]
    else:  # Retain the original score shortcuts for existing operators.
        terminal_score, outcome, true_terminal = SCORES[label]
    data["rewards"][-1] += terminal_score
    data["terminated"][-1] = bool(true_terminal)
    data["truncated"][-1] = not true_terminal
    data["next_values"][-1] = 0. if true_terminal else float(bootstrap_value)
    rollout = {
        "schema_version": 1, "artifact_type": "ppo_rollout",
        "policy_update": int(saved["update"]),
        "policy_checkpoint_sha256": ppo.file_sha256(checkpoint),
        "imitation_checkpoint_sha256": saved["imitation_checkpoint_sha256"],
        "created_unix_s": time.time(), "reward_source": "human_terminal",
        "outcome": outcome, "terminal_score": terminal_score, "end_reason": reason,
        "control_hz": control_hz, "backend": backend,
        "runtime_action_transform": {
            "steering_scale": steering_scale,
            "throttle_scale": throttle_scale,
            "initial_forward_thrust": initial_forward_thrust,
            "initial_forward_steps": initial_forward_steps,
            "initial_forward_cap": initial_forward_cap,
            "steering_offset": float(cfg["hardware"]["steering_offset"]),
            "steering_limit": float(cfg["limits"]["steering"]),
        },
    }
    rollout.update({
        "recurrent": torch.stack(data["recurrent"]),
        "actions": torch.stack(data["actions"]),
        "applied_actions": torch.stack(data["applied_actions"]),
        "raw_actions": torch.stack(data["raw_actions"]),
        "old_log_probs": torch.stack(data["old_log_probs"]),
        "values": torch.stack(data["values"]),
        "next_values": torch.tensor(data["next_values"], dtype=torch.float32),
        "rewards": torch.tensor(data["rewards"], dtype=torch.float32),
        "terminated": torch.tensor(data["terminated"], dtype=torch.bool),
        "truncated": torch.tensor(data["truncated"], dtype=torch.bool),
    })
    ppo.validate_rollout(rollout, saved, ppo.file_sha256(checkpoint))
    return rollout, False


def update_when_ready(model, saved, checkpoint, run_dir, device, minimum_episodes):
    paths = rollout_paths(run_dir, saved["update"])
    if len(paths) < int(minimum_episodes):
        return checkpoint, saved, None
    rollouts = ppo.load_rollouts(paths, checkpoint, saved)
    metrics, optimizer_state = ppo.update_policy(
        model, rollouts, saved["online_config"], device,
        saved.get("optimizer_state_dict"))
    next_checkpoint = ppo.save_policy_update(
        model, saved, checkpoint, metrics, optimizer_state)
    next_saved = load_checkpoint(next_checkpoint, "cpu")
    deleted = prune_completed_update(run_dir, saved["update"], next_checkpoint.parent)
    print("PPO UPDATE {}: {}".format(next_saved["update"], json.dumps(metrics)), flush=True)
    if deleted:
        print("CHECKPOINT CLEANUP: removed {}".format(", ".join(deleted)), flush=True)
    return next_checkpoint, next_saved, metrics


def run_physical_loop(checkpoint, run_dir, device="cuda", trt_encoder=None, stop_head_path=None,
                      maximum_updates=0, maximum_episodes=0, warmup_s=.5,
                      max_frame_age_ms=None, fp16_encoder=False, episodes_per_update=5,
                      steering_offset=None, steering_scale=1., steering_limit=None,
                      throttle_scale=1., initial_forward_thrust=0.,
                      initial_forward_steps=0, initial_forward_cap=1.):
    """Keep camera/driver open while collecting, scoring, and updating PPO."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = latest_checkpoint(checkpoint, run_dir)
    print("RESUME: {}".format(checkpoint), flush=True)
    model, saved = ppo.load_policy(checkpoint, device)
    saved = configure_update_schedule(saved, episodes_per_update)
    print("PPO update schedule: {} scored episodes".format(episodes_per_update), flush=True)
    online = saved["online_config"]
    if not bool(online.get("physical_rollout_enabled")):
        raise ValueError("Set physical_rollout_enabled=true only after configuring the physical run")
    if bool(online["algorithm"].get("train_gru", False)) or bool(
            online["algorithm"].get("train_projection", False)):
        raise ValueError("The persistent feature-rollout loop requires frozen projection and GRU")
    cfg = _runtime_config(saved, max_frame_age_ms)
    checkpoint_offset, checkpoint_limit = apply_runtime_steering_config(
        cfg, steering_offset, steering_limit)
    print("Steering offset: checkpoint={:+.3f} runtime={:+.3f}".format(
        checkpoint_offset, float(cfg["hardware"]["steering_offset"])), flush=True)
    print("Steering limit: checkpoint={:.3f} runtime={:.3f}".format(
        checkpoint_limit, float(cfg["limits"]["steering"])), flush=True)
    print("Steering scale: {:.3f}".format(float(steering_scale)), flush=True)
    print("Throttle scale: {:.3f}".format(float(throttle_scale)), flush=True)
    if float(initial_forward_thrust) > 0.:
        print("Initial forward throttle: bias=-{:.3f} steps={} cap={:.3f}".format(
            float(initial_forward_thrust), int(initial_forward_steps),
            float(initial_forward_cap)), flush=True)
    backend = _configure_inference_backend(
        model, saved, device, trt_encoder, fp16_encoder)
    print("Inference backend: {}".format(backend), flush=True)
    stop_head, stop_decision = load_stop_head(stop_head_path, saved, device)
    minimum = int(online["rollout"]["minimum_episodes_per_update"])
    maximum_steps = int(online["rollout"]["maximum_episode_steps"])
    completed_updates = 0
    collected_episodes = 0
    # Keep OpenCV/GStreamer and TensorRT on the same thread. The old Jetson
    # Nano stack can corrupt native camera buffers when a background cap.read()
    # overlaps GPU inference or runs continuously at the READY prompt.
    with TerminalKeys() as keys, Camera(
            cfg, True, threaded=False, release_on_close=False) as camera, Car(cfg, True) as car:
        car.park("ppo_ready")
        while True:
            checkpoint, saved, metrics = update_when_ready(
                model, saved, checkpoint, run_dir, device, minimum)
            if metrics is not None:
                completed_updates += 1
                model.eval()
                if maximum_updates and completed_updates >= int(maximum_updates):
                    break
            if maximum_episodes and collected_episodes >= int(maximum_episodes):
                break
            if not keys.wait_start():
                break
            rollout, exit_requested = collect_episode(
                model, saved, checkpoint, cfg, camera, car, keys, device, backend,
                stop_head, stop_decision, warmup_s, maximum_steps,
                throttle_scale, steering_scale, initial_forward_thrust,
                initial_forward_steps, initial_forward_cap)
            if exit_requested:
                break
            if rollout is not None:
                path = save_rollout(run_dir, rollout)
                collected_episodes += 1
                print("SAVED: {} ({}/{})".format(
                    path, len(rollout_paths(run_dir, saved["update"])), minimum), flush=True)
    return {"checkpoint": str(checkpoint), "update": int(saved["update"]),
            "episodes_collected": collected_episodes, "updates_completed": completed_updates}


def run_follow_loop(checkpoint, device="cuda", trt_encoder=None, stop_head_path=None,
                    warmup_s=.5, max_frame_age_ms=None, fp16_encoder=False,
                    seconds=None):
    """Follow the newest PPO policy without collecting data or training."""
    checkpoint = inference_checkpoint(checkpoint)
    print("FOLLOW CHECKPOINT: {}".format(checkpoint), flush=True)
    model, saved = ppo.load_policy(checkpoint, device)
    cfg = _runtime_config(saved, max_frame_age_ms)
    backend = _configure_inference_backend(
        model, saved, device, trt_encoder, fp16_encoder)
    print("Inference backend: {}".format(backend), flush=True)
    stop_head, stop_decision = load_stop_head(stop_head_path, saved, device)
    control_hz = float(cfg["control"]["hz"])
    if seconds is None:
        maximum_steps = int(saved["online_config"]["rollout"]["maximum_episode_steps"])
    else:
        if not 0. < float(seconds) <= float(cfg["control"]["max_episode_s"]):
            raise ValueError("--seconds must be above 0 and no greater than {:.0f}".format(
                cfg["control"]["max_episode_s"]))
        maximum_steps = max(1, int(round(float(seconds) * control_hz)))

    completed_runs = 0
    print("Evaluation only: actions use the actor mean; no rollouts, rewards, or updates.",
          flush=True)
    # Use the same synchronous, process-lifetime camera handling as physical PPO.
    with TerminalKeys() as keys, Camera(
            cfg, True, threaded=False, release_on_close=False) as camera, Car(cfg, True) as car:
        car.park("follow_ready")
        while keys.wait_start():
            reason, exit_requested, steps = follow_episode(
                model, cfg, camera, car, keys, device, backend, stop_head,
                stop_decision, warmup_s, maximum_steps)
            completed_runs += 1
            print("FOLLOW STOPPED: {} after {} applied actions".format(
                reason, steps), flush=True)
            if exit_requested:
                break
    return {"checkpoint": str(checkpoint), "update": int(saved["update"]),
            "runs_completed": completed_runs, "mode": "deterministic_follow"}


def configure_update_schedule(saved, episodes_per_update=5):
    if isinstance(episodes_per_update, bool) or int(episodes_per_update) != episodes_per_update or episodes_per_update < 1:
        raise ValueError("Episodes per update must be a positive integer")
    saved = dict(saved)
    saved["online_config"] = copy.deepcopy(saved["online_config"])
    saved["online_config"]["rollout"]["minimum_episodes_per_update"] = int(episodes_per_update)
    return saved


def train_watch_loop(checkpoint, run_dir, device="cpu", poll_s=2., maximum_updates=0,
                     episodes_per_update=5):
    """Wait for rollout artifacts and update forever without restarting."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = latest_checkpoint(checkpoint, run_dir)
    print("RESUME: {}".format(checkpoint), flush=True)
    print("PPO update schedule: {} scored episodes".format(episodes_per_update), flush=True)
    completed = 0
    while True:
        model, saved = ppo.load_policy(checkpoint, device)
        saved = configure_update_schedule(saved, episodes_per_update)
        minimum = int(saved["online_config"]["rollout"]["minimum_episodes_per_update"])
        next_checkpoint, next_saved, metrics = update_when_ready(
            model, saved, checkpoint, run_dir, device, minimum)
        if metrics is None:
            time.sleep(float(poll_s))
            continue
        checkpoint = next_checkpoint
        completed += 1
        if maximum_updates and completed >= int(maximum_updates):
            return checkpoint
