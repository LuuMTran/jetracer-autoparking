"""Human stop labels, stop-head fine-tuning, and two-stage policy execution."""
import json
import random
import time
import uuid
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .camera import Camera
from .core import Car, canonical_config, save_json
from .learning import StopHead, load_checkpoint, signature
from . import ppo
from .ppo_loop import (TerminalKeys, _configure_inference_backend, _image_input,
                       _previous_input, _require_finite_recurrent, _runtime_config,
                       follow_episode, inference_checkpoint, load_stop_head)


def stop_label_window(labels, signal_step, pre_frames=2, post_frames=2):
    """Return labels with a short positive window around one human stop signal."""
    if signal_step < 0 or signal_step >= len(labels):
        raise ValueError("Stop signal step is outside the episode")
    result = list(bool(value) for value in labels)
    start = max(0, int(signal_step) - int(pre_frames))
    end = min(len(result), int(signal_step) + int(post_frames) + 1)
    for index in range(start, end):
        result[index] = True
    return result, start, end


def read_stop_key(keys):
    """Read all pending keys; S labels a stop state without stopping the car."""
    result = None
    while True:
        key = keys.read(0.)
        if key is None:
            return result
        if key == "\x1b":
            return "escape"
        if key == "f":
            result = "finish"
        elif key == "s" and result is None:
            result = "stop_label"


def save_stop_labels(output, recurrent, actions, elapsed, signal_step, pre_frames,
                     post_frames, checkpoint, saved):
    if signal_step is None:
        return None
    labels, positive_start, positive_end = stop_label_window(
        [False] * len(recurrent), signal_step, pre_frames, post_frames)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    name = "stop_labels_{}_{}.pt".format(
        time.strftime("%Y%m%d_%H%M%S"), uuid.uuid4().hex[:8])
    destination = output / name
    payload = {
        "schema_version": 1,
        "artifact_type": "human_stop_labels",
        "created_unix_s": time.time(),
        "policy_checkpoint": str(checkpoint),
        "policy_checkpoint_sha256": ppo.file_sha256(checkpoint),
        "policy_update": int(saved["update"]),
        "imitation_checkpoint_sha256": saved["imitation_checkpoint_sha256"],
        "imitation_architecture": saved["imitation_architecture"],
        "frozen_recurrent_trunk": True,
        "signal_step": int(signal_step),
        "positive_start": int(positive_start),
        "positive_end": int(positive_end),
        "recurrent": torch.stack(recurrent).float(),
        "actions": torch.tensor(actions, dtype=torch.float32),
        "elapsed_s": torch.tensor(elapsed, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.bool),
    }
    temporary = destination.with_suffix(".tmp")
    torch.save(payload, str(temporary))
    temporary.replace(destination)
    return destination


def collect_stop_labels(checkpoint, output, device="cuda", fp16_encoder=False,
                        seconds=40., max_frame_age_ms=None, warmup_s=.5,
                        pre_frames=2, post_frames=2, maximum_episodes=0):
    """Follow a PPO actor while S records a stop target without changing its action."""
    checkpoint = inference_checkpoint(checkpoint)
    model, saved = ppo.load_policy(checkpoint, device)
    online = saved["online_config"]["algorithm"]
    if bool(online.get("train_gru", False)) or bool(online.get("train_projection", False)):
        raise ValueError("Stop labels require the projection and GRU to remain frozen")
    cfg = _runtime_config(saved, max_frame_age_ms)
    if not 0 < float(seconds) <= float(cfg["control"]["max_episode_s"]):
        raise ValueError("Collection seconds exceed the checkpoint episode limit")
    if min(int(pre_frames), int(post_frames)) < 0:
        raise ValueError("Label-window frame counts cannot be negative")
    backend = _configure_inference_backend(model, saved, device, None, fp16_encoder)
    control_hz = float(cfg["control"]["hz"])
    maximum_steps = max(1, int(round(float(seconds) * control_hz)))
    warmup_frames = max(0, int(round(float(warmup_s) * control_hz)))
    completed = 0
    saved_paths = []
    print("STOP LABEL COLLECTION: actor mean controls the car.", flush=True)
    print("S marks one desired stop state and the car continues; F saves; Escape exits.", flush=True)
    with TerminalKeys() as keys, Camera(
            cfg, True, threaded=False, release_on_close=False) as camera, Car(cfg, True) as car:
        car.park("stop_label_ready")
        while not maximum_episodes or completed < int(maximum_episodes):
            if not keys.wait_start():
                break
            recurrent_rows, actions, elapsed_rows = [], [], []
            hidden = None
            previous = (0., 0.)
            last_id = -1
            model.eval()
            with torch.no_grad():
                for index in range(warmup_frames):
                    frame = camera.read(after=last_id, timeout=max(
                        2., cfg["control"]["max_frame_age_s"]))
                    last_id = frame["frame_id"]
                    recurrent, hidden = model.recurrent(
                        _image_input(frame["rgb"], device, backend),
                        _previous_input(previous, cfg, device), hidden)
                    _require_finite_recurrent(recurrent, backend)
                    model.act_recurrent(recurrent, deterministic=True)
                    float(recurrent.float().sum().cpu().item())
                    print("WARMUP: frame {}/{}".format(index + 1, warmup_frames), flush=True)

            started = time.monotonic()
            deadline = started
            signal_step = None
            exit_requested = False
            for step in range(maximum_steps):
                frame = camera.read(after=last_id, timeout=max(
                    2., cfg["control"]["max_frame_age_s"]))
                last_id = frame["frame_id"]
                key = read_stop_key(keys)
                if key == "escape":
                    car.park("emergency_stop")
                    exit_requested = True
                    break
                if key == "finish":
                    break
                if key == "stop_label" and signal_step is None:
                    signal_step = step
                    print("STOP LABEL: step {}; policy continues.".format(step), flush=True)

                inference_started = time.monotonic()
                with torch.no_grad():
                    recurrent, next_hidden = model.recurrent(
                        _image_input(frame["rgb"], device, backend),
                        _previous_input(previous, cfg, device), hidden)
                    _require_finite_recurrent(recurrent, backend)
                    action, raw, log_prob, value = model.act_recurrent(
                        recurrent, deterministic=True)
                    synchronized = (action.float().sum() + raw.float().sum() +
                                    log_prob.float().sum() + value.float().sum())
                    float(synchronized.cpu().item())
                frame_age = time.monotonic() - frame["frame_monotonic_s"]
                if frame_age > cfg["control"]["max_frame_age_s"]:
                    raise RuntimeError("Frame too old after stop-label inference")
                requested = tuple(float(item) for item in action[0, 0].cpu().tolist())
                command = car.apply(requested)
                applied = tuple(float(item) for item in command["normalized"])
                if max(abs(requested[i] - applied[i]) for i in range(2)) > .02:
                    raise RuntimeError("Applied action differs from deterministic PPO action")
                recurrent_rows.append(recurrent[0, 0].detach().float().cpu())
                actions.append(applied)
                elapsed_rows.append(time.monotonic() - started)
                previous = command["action"]
                hidden = next_hidden
                if step == 0 or (step + 1) % max(1, int(control_hz)) == 0:
                    print("t={:.1f}s step={} action=({:+.3f},{:+.3f}) inference={:.0f}ms".format(
                        elapsed_rows[-1], step + 1, applied[0], applied[1],
                        (time.monotonic() - inference_started) * 1000.), flush=True)
                deadline += 1. / control_hz
                if deadline < time.monotonic():
                    deadline = time.monotonic()
                time.sleep(max(0., deadline - time.monotonic()))

            car.park("stop_label_episode_end")
            if exit_requested:
                break
            path = save_stop_labels(
                output, recurrent_rows, actions, elapsed_rows, signal_step,
                pre_frames, post_frames, checkpoint, saved)
            if path is None:
                print("DISCARDED: no S stop label was recorded.", flush=True)
            else:
                completed += 1
                saved_paths.append(str(path))
                print("SAVED: {} ({})".format(path, completed), flush=True)
    return {"checkpoint": str(checkpoint), "episodes_saved": completed,
            "files": saved_paths}


def first_trigger(probabilities, threshold, consecutive):
    streak = 0
    for index, probability in enumerate(probabilities):
        streak = streak + 1 if float(probability) >= float(threshold) else 0
        if streak >= int(consecutive):
            return index
    return None


def decision_metrics(probabilities, episodes, threshold, consecutive):
    details = []
    detected = early = late = missed = 0
    for item in episodes:
        values = probabilities[item["start"]:item["end"]]
        trigger = first_trigger(values, threshold, consecutive)
        if trigger is None:
            outcome = "missed"
            missed += 1
        elif trigger < item["positive_start"]:
            outcome = "early"
            early += 1
        elif trigger >= item["positive_end"]:
            outcome = "late"
            late += 1
        else:
            outcome = "detected"
            detected += 1
        details.append({"file": item["file"], "trigger_step": trigger,
                        "positive_start": item["positive_start"],
                        "positive_end": item["positive_end"], "outcome": outcome})
    return {"threshold": float(threshold), "episodes": len(episodes),
            "detected": detected, "early": early, "late": late, "missed": missed,
            "details": details}


def load_stop_label_files(dataset, saved):
    files = sorted(Path(dataset).glob("stop_labels_*.pt"))
    if len(files) < 5:
        raise ValueError("Collect at least 5 stop-labelled runs before fine-tuning")
    episodes = []
    for path in files:
        item = load_checkpoint(path, "cpu")
        if item.get("artifact_type") != "human_stop_labels":
            raise ValueError("Unexpected stop-label artifact: " + str(path))
        if item.get("imitation_checkpoint_sha256") != saved["imitation_checkpoint_sha256"]:
            raise ValueError("Stop labels came from a different imitation policy: " + str(path))
        if item.get("imitation_architecture") != saved["imitation_architecture"]:
            raise ValueError("Stop-label architecture differs: " + str(path))
        recurrent = item["recurrent"].float()
        labels = item["labels"].to(dtype=torch.float32)
        if recurrent.ndim != 2 or recurrent.shape[1] != 64 or len(recurrent) != len(labels):
            raise ValueError("Invalid recurrent stop-label shapes: " + str(path))
        # PyTorch 1.6 only implements Tensor.all/any for uint8 and bool.  The
        # labels are floats here because BCEWithLogitsLoss expects float targets,
        # so count the positive class instead of calling all/any directly.
        positive_count = int((labels >= .5).sum().item())
        if positive_count == 0 or positive_count == int(labels.numel()):
            raise ValueError("Each run needs positive and negative stop labels: " + str(path))
        episodes.append({"file": path.name, "recurrent": recurrent, "labels": labels,
                         "positive_start": int(item["positive_start"]),
                         "positive_end": int(item["positive_end"])})
    return episodes


def head_probabilities(head, features, device, batch_size=512):
    output = []
    head.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            logits = head(features[start:start + batch_size].to(device))
            output.append(torch.sigmoid(logits).float().cpu())
    return torch.cat(output).numpy()


def fine_tune_stop_head(checkpoint, dataset, initial_stop_head, output, device="cpu",
                        epochs=100, patience=10, batch_size=128, learning_rate=1e-4,
                        seed=7):
    """Fine-tune only the stop MLP from human signals collected under frozen PPO features."""
    checkpoint = inference_checkpoint(checkpoint)
    saved = load_checkpoint(checkpoint, "cpu")
    if saved.get("algorithm") != "recurrent_ppo":
        raise ValueError("Stop fine-tuning requires a recurrent PPO checkpoint")
    algorithm = saved["online_config"]["algorithm"]
    if bool(algorithm.get("train_gru", False)) or bool(algorithm.get("train_projection", False)):
        raise ValueError("Collected stop features require a frozen projection and GRU")
    initial = load_checkpoint(initial_stop_head, "cpu")
    if initial.get("artifact_type") != "recurrent_stop_head":
        raise ValueError("Initial file is not a stop head")
    if initial.get("checkpoint_sha256") != saved["imitation_checkpoint_sha256"]:
        raise ValueError("Initial stop head came from a different imitation checkpoint")
    if initial.get("architecture") != saved["imitation_architecture"]:
        raise ValueError("Initial stop-head architecture differs from PPO")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    episodes = load_stop_label_files(dataset, saved)
    random.Random(seed).shuffle(episodes)
    validation_count = max(1, int(round(len(episodes) * .2)))
    validation = episodes[:validation_count]
    training = episodes[validation_count:]
    train_features = torch.cat([item["recurrent"] for item in training])
    train_labels = torch.cat([item["labels"] for item in training])
    val_features = torch.cat([item["recurrent"] for item in validation])
    val_labels = torch.cat([item["labels"] for item in validation])
    offset = 0
    val_episode_ranges = []
    for item in validation:
        copied = dict(item)
        copied["start"], copied["end"] = offset, offset + len(item["labels"])
        copied.pop("recurrent")
        copied.pop("labels")
        val_episode_ranges.append(copied)
        offset = copied["end"]

    torch.manual_seed(seed)
    head = StopHead().to(device)
    head.load_state_dict(initial["state_dict"])
    positives = float(train_labels.sum().item())
    negatives = float(len(train_labels) - positives)
    if positives < 1 or negatives < 1:
        raise ValueError("Training split needs both stop and continue labels")
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(head.parameters(), lr=float(learning_rate))
    loader = DataLoader(TensorDataset(train_features, train_labels),
                        batch_size=int(batch_size), shuffle=True)
    history, best_loss, best_state, stale = [], float("inf"), None, 0
    for epoch in range(1, int(epochs) + 1):
        head.train()
        total = 0.
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            loss = criterion(head(features), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(labels)
        head.eval()
        with torch.no_grad():
            val_loss = float(criterion(
                head(val_features.to(device)), val_labels.to(device)).item())
        row = {"epoch": epoch, "train_loss": total / len(train_labels),
               "val_loss": val_loss}
        history.append(row)
        print(row, flush=True)
        if val_loss < best_loss - 1e-6:
            best_loss, stale = val_loss, 0
            best_state = {name: value.detach().cpu().clone()
                          for name, value in head.state_dict().items()}
        else:
            stale += 1
        if stale >= int(patience):
            break
    head.load_state_dict(best_state)
    validation_probabilities = head_probabilities(head, val_features, device)
    consecutive = int(initial.get("decision", {}).get("consecutive_frames", 4))
    thresholds = [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95, .97, .99]
    candidates = [decision_metrics(validation_probabilities, val_episode_ranges,
                                   threshold, consecutive) for threshold in thresholds]
    zero_early = [item for item in candidates if item["early"] == 0]
    pool = zero_early or candidates
    selected = max(pool, key=lambda item: (
        item["detected"], -item["late"], -item["missed"], item["threshold"]))
    decision = dict(initial.get("decision", {}))
    decision.update(threshold=selected["threshold"], consecutive_frames=consecutive)
    report = {
        "checkpoint": str(checkpoint),
        "imitation_checkpoint_sha256": saved["imitation_checkpoint_sha256"],
        "initial_stop_head": str(initial_stop_head),
        "initial_stop_head_sha256": ppo.file_sha256(initial_stop_head),
        "episodes": len(episodes),
        "train_files": [item["file"] for item in training],
        "validation_files": [item["file"] for item in validation],
        "epochs_completed": len(history),
        "best_validation_loss": best_loss,
        "decision": decision,
        "validation_selected": selected,
        "validation_candidates": candidates,
    }
    payload = {
        "schema_version": 1,
        "artifact_type": "recurrent_stop_head",
        "architecture": saved["imitation_architecture"],
        "checkpoint_sha256": saved["imitation_checkpoint_sha256"],
        "state_dict": best_state,
        "decision": decision,
        "fine_tuned_from": ppo.file_sha256(initial_stop_head),
        "human_stop_label_runs": len(episodes),
        "metrics": {"best_validation_loss": best_loss,
                    "validation": {key: value for key, value in selected.items()
                                   if key != "details"}},
    }
    temporary = output / "stop_head.tmp"
    torch.save(payload, str(temporary))
    temporary.replace(output / "stop_head.pt")
    save_json(output / "history.json", history)
    save_json(output / "metrics.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return output / "stop_head.pt"


def load_sequence_policy(checkpoint, stop_head_path, device, fp16_encoder,
                         max_frame_age_ms):
    checkpoint = inference_checkpoint(checkpoint)
    model, saved = ppo.load_policy(checkpoint, device)
    cfg = _runtime_config(saved, max_frame_age_ms)
    backend = _configure_inference_backend(model, saved, device, None, fp16_encoder)
    stop_head, decision = load_stop_head(stop_head_path, saved, device)
    if stop_head is None:
        raise ValueError("Each sequence stage requires a stop head")
    return {"checkpoint": checkpoint, "model": model, "saved": saved, "cfg": cfg,
            "backend": backend, "stop_head": stop_head, "decision": decision}


def run_policy_sequence(first_checkpoint, first_stop_head, second_checkpoint,
                        second_stop_head, device="cuda", fp16_encoder=False,
                        first_seconds=40., second_seconds=40., warmup_s=.5,
                        max_frame_age_ms=None, second_throttle_scale=1.,
                        slot_classifier_checkpoint=None, slot_device="cpu"):
    """Choose a route from three start frames, then run the required policies."""
    first = load_sequence_policy(first_checkpoint, first_stop_head, device,
                                 fp16_encoder, max_frame_age_ms)
    second = load_sequence_policy(second_checkpoint, second_stop_head, device,
                                  fp16_encoder, max_frame_age_ms)
    if signature(first["cfg"]) != signature(second["cfg"]):
        raise ValueError("The two policies use different camera, calibration, limits, or rates")
    cfg = canonical_config(first["cfg"])
    first_steps = max(1, int(round(float(first_seconds) * cfg["control"]["hz"])))
    second_steps = max(1, int(round(float(second_seconds) * cfg["control"]["hz"])))
    maximum = int(round(cfg["control"]["max_episode_s"] * cfg["control"]["hz"]))
    if first_steps > maximum or second_steps > maximum:
        raise ValueError("A sequence stage exceeds the configured episode duration")
    if not 0. < float(second_throttle_scale) <= 1.:
        raise ValueError("Second-stage throttle scale must be above 0 and no greater than 1")
    slot_predictor = None
    if slot_classifier_checkpoint is not None:
        from .slot_classifier import SlotPredictor
        slot_predictor = SlotPredictor(slot_classifier_checkpoint, slot_device)
    completed = 0
    decisions = []
    print("CONDITIONAL POLICY: classify 3 start frames; left -> stages 1+2; "
          "middle/right -> stage 1 only.", flush=True)
    print("F forces the current stage to advance/end; Escape stops and exits.", flush=True)
    with TerminalKeys() as keys, Camera(
            cfg, True, threaded=False, release_on_close=False) as camera, Car(cfg, True) as car:
        car.park("sequence_ready")
        while keys.wait_start():
            decision = {"left_empty": True, "left_probability": None, "threshold": None}
            if slot_predictor is not None:
                from .slot_classifier import classify_start
                car.park("slot_classification")
                decision = classify_start(camera, slot_predictor)
                route = "DIRECT + REVERSE" if decision["left_empty"] else "DIRECT ONLY"
                print("SLOT ROUTER: p(left)={:.3f}, threshold={:.2f} -> {}".format(
                    decision["left_probability"], decision["threshold"], route), flush=True)
            decisions.append(decision)
            print("STAGE 1: {}".format(first["checkpoint"]), flush=True)
            reason, exit_requested, steps = follow_episode(
                first["model"], cfg, camera, car, keys, device, first["backend"],
                first["stop_head"], first["decision"], warmup_s, first_steps)
            print("STAGE 1 ENDED: {} after {} actions".format(reason, steps), flush=True)
            if exit_requested:
                break
            if not decision["left_empty"]:
                print("ROUTE COMPLETE: middle/right slot uses direct policy only", flush=True)
                completed += 1
                car.park("conditional_route_complete")
                continue
            print("SWITCHING TO STAGE 2", flush=True)
            reason, exit_requested, steps = follow_episode(
                second["model"], cfg, camera, car, keys, device, second["backend"],
                second["stop_head"], second["decision"], warmup_s, second_steps,
                second_throttle_scale)
            print("STAGE 2 ENDED: {} after {} actions".format(reason, steps), flush=True)
            completed += 1
            if exit_requested:
                break
    return {"completed_sequences": completed,
            "first_checkpoint": str(first["checkpoint"]),
            "second_checkpoint": str(second["checkpoint"]),
            "second_throttle_scale": float(second_throttle_scale),
            "slot_classifier": (str(slot_classifier_checkpoint)
                                if slot_classifier_checkpoint is not None else None),
            "route_decisions": decisions}
