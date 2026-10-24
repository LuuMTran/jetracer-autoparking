#!/usr/bin/env python3
"""Collect expert labels on policy states and fine-tune a PPO actor with DAgger."""
import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def _clamp(value):
    return max(-1., min(1., float(value)))


def collect(args):
    """Execute a policy/expert mixture while saving the expert action as the label."""
    from parking_lib import ppo
    from parking_lib.camera import Camera
    from parking_lib.controls import Controls, KEYBOARD_HELP
    from parking_lib.core import Car, normalized
    from parking_lib.data import AsyncRecorder, validate_episode
    from parking_lib.ppo_loop import (
        _configure_inference_backend, _image_input, _previous_input,
        _require_finite_recurrent, _runtime_action, _runtime_config,
        _validate_runtime_action_controls, inference_checkpoint,
    )

    checkpoint = inference_checkpoint(args.checkpoint)
    model, saved = ppo.load_policy(checkpoint, args.device)
    cfg = _runtime_config(saved, args.max_frame_age_ms)
    _validate_runtime_action_controls(
        args.throttle_scale, args.steering_scale, args.initial_forward_thrust,
        args.initial_forward_steps, args.initial_forward_cap)
    backend = _configure_inference_backend(
        model, saved, args.device, args.trt_encoder, args.fp16_encoder)
    model.eval()
    period = 1. / float(cfg["control"]["hz"])
    maximum_steps = min(
        int(args.seconds * float(cfg["control"]["hz"])),
        int(saved["online_config"]["rollout"]["maximum_episode_steps"]))

    print("DAGGER CHECKPOINT: {}".format(checkpoint), flush=True)
    print("Inference backend: {}".format(backend), flush=True)
    print("Mixture beta: {:.3f} expert + {:.3f} policy".format(
        args.beta, 1. - args.beta), flush=True)
    print(KEYBOARD_HELP, flush=True)
    print("Keep the deadman/key active and continuously provide the action the car SHOULD take.", flush=True)
    print("Releasing the control ends the episode immediately; Escape emergency-stops.", flush=True)

    recorder = None
    car = None
    final = None
    status, reason = "aborted", "initialization"
    last_id = -1
    previous = (0., 0.)
    hidden = None
    started = None
    step = 0

    with Controls(cfg, args.controller) as expert, Camera(
            cfg, True, threaded=False, release_on_close=False) as camera:
        # Warm CUDA/cuDNN before the motor driver is opened.
        with torch.no_grad():
            for warmup_index in range(max(1, args.warmup_frames)):
                frame = camera.read(after=last_id, timeout=max(
                    2., cfg["control"]["max_frame_age_s"]))
                last_id = frame["frame_id"]
                recurrent, hidden = model.recurrent(
                    _image_input(frame["rgb"], args.device, backend),
                    _previous_input(previous, cfg, args.device), hidden)
                _require_finite_recurrent(recurrent, backend)
                action, raw, log_prob, value = model.act_recurrent(
                    recurrent, deterministic=True)
                float((action.float().sum() + raw.float().sum() +
                       log_prob.float().sum() + value.float().sum()).cpu().item())
        hidden = None

        recorder = AsyncRecorder(
            args.dataset, cfg, False, args.controller, args.scenario, source="dagger")
        recorder.meta.update({
            "accepted": False,
            "outcome": "unreviewed",
            "dagger": {
                "beta": float(args.beta),
                "policy_checkpoint": str(checkpoint),
                "policy_update": int(saved["update"]),
                "label_columns": ["requested_steering_norm", "requested_throttle_norm"],
                "executed_action": "beta * expert + (1-beta) * deterministic_policy",
                "policy_action_transform": {
                    "steering_scale": float(args.steering_scale),
                    "throttle_scale": float(args.throttle_scale),
                    "initial_forward_thrust": float(args.initial_forward_thrust),
                    "initial_forward_steps": int(args.initial_forward_steps),
                    "initial_forward_cap": float(args.initial_forward_cap),
                },
            },
        })
        print("Recording:", recorder.path, flush=True)

        try:
            with Car(cfg, True) as car:
                recorder.meta["driver_runtime"] = car.describe()
                deadline = time.monotonic()
                while step < maximum_steps:
                    recorder.check()
                    frame = camera.read(after=last_id, timeout=max(
                        2., cfg["control"]["max_frame_age_s"]))
                    last_id = frame["frame_id"]
                    expert_action = expert.poll()
                    input_stamp = time.monotonic()
                    if expert.stop:
                        reason = "emergency_stop"
                        break
                    if expert.finish:
                        status, reason = "completed", "operator_finished"
                        break

                    inference_started = time.monotonic()
                    with torch.no_grad():
                        recurrent, next_hidden = model.recurrent(
                            _image_input(frame["rgb"], args.device, backend),
                            _previous_input(previous, cfg, args.device), hidden)
                        _require_finite_recurrent(recurrent, backend)
                        policy, _, _, _ = model.act_recurrent(
                            recurrent, deterministic=True)
                    inference_s = time.monotonic() - inference_started
                    raw_policy_action = tuple(
                        float(v) for v in policy[0, 0].cpu().tolist())
                    policy_action = _runtime_action(
                        raw_policy_action, args.steering_scale, args.throttle_scale,
                        step, args.initial_forward_thrust,
                        args.initial_forward_steps, args.initial_forward_cap)

                    if not expert.active:
                        if started is None:
                            # Keep the GRU warm on stationary frames while the
                            # operator gets ready, as in normal policy following.
                            hidden = next_hidden
                            previous = (0., 0.)
                            time.sleep(period)
                            continue
                        status, reason = "completed", "expert_released"
                        break

                    if started is None:
                        started = time.monotonic()
                        deadline = started

                    mixed = tuple(_clamp(args.beta * expert_action[i] +
                                         (1. - args.beta) * policy_action[i])
                                  for i in range(2))
                    frame_age = time.monotonic() - frame["frame_monotonic_s"]
                    if frame_age > cfg["control"]["max_frame_age_s"]:
                        raise RuntimeError("Frame too old after inference: {:.0f}ms".format(
                            frame_age * 1000.))

                    old_previous = previous
                    command = car.apply(mixed)
                    command["input_monotonic_s"] = input_stamp
                    missed = max(0, int((command["command_start_monotonic_s"] - deadline) / period))
                    # requested_* stores the expert label; steering/throttle stores
                    # the action actually sent to the car.
                    recorder.append(frame, command, expert_action, old_previous, True, missed)
                    previous = command["action"]
                    hidden = next_hidden
                    step += 1
                    if step == 1 or step % max(1, int(cfg["control"]["hz"])) == 0:
                        print("step={} expert=({:+.2f},{:+.2f}) policy=({:+.2f},{:+.2f}) "
                              "applied=({:+.2f},{:+.2f}) inference={:.0f}ms".format(
                                  step, expert_action[0], expert_action[1],
                                  policy_action[0], policy_action[1], mixed[0], mixed[1],
                                  inference_s * 1000.), flush=True)
                    deadline += period
                    if deadline < time.monotonic():
                        deadline = time.monotonic()
                    time.sleep(max(0., deadline - time.monotonic()))
                else:
                    status, reason = "completed", "duration_limit"
                car.park(reason)
                try:
                    final = camera.read(after=last_id, timeout=.3)
                except RuntimeError:
                    pass
        except KeyboardInterrupt:
            status, reason = "aborted", "keyboard_interrupt"
        except BaseException as exc:
            status, reason = "aborted", repr(exc)
            raise
        finally:
            enough = step >= args.sequence_length
            if status == "completed" and enough:
                recorder.meta.update(accepted=True, outcome="dagger_labelled",
                                     notes="Automatically accepted complete DAgger episode")
            elif status == "completed":
                status = "aborted"
                reason = "fewer_than_{}_labelled_frames".format(args.sequence_length)
            recorder.finish(status, reason, car.events if car else [], final)

    report = validate_episode(recorder.path)
    print(json.dumps(report, indent=2), flush=True)
    if not report["valid"]:
        raise RuntimeError("DAgger episode failed validation")
    return recorder.path


def _split_for_dagger(episodes, saved):
    source = saved.get("source_split")
    if not source or any(not source.get(name) for name in ("train", "val", "test")):
        raise ValueError("PPO checkpoint has no reusable imitation train/val/test split")
    available = {path.name: path for path in episodes}
    held_out = set(source["val"] + source["test"])
    missing = held_out - set(available)
    if missing:
        raise ValueError("Dataset is missing original held-out episodes: " + ", ".join(sorted(missing)))
    dagger = [path for path in episodes
              if json.loads((path / "metadata.json").read_text()).get("source") == "dagger"]
    if not dagger:
        raise ValueError("No accepted DAgger episodes found in the dataset")
    train = [path for path in episodes if path.name not in held_out]
    return {"train": train,
            "val": [available[name] for name in source["val"]],
            "test": [available[name] for name in source["test"]]}, dagger


def train(args):
    """Supervised DAgger update of a PPO policy using the aggregated dataset."""
    from parking_lib import ppo
    from parking_lib.augmentation import SequenceAugment
    from parking_lib.learning import Sequences, eligible_episodes, signature
    from parking_lib.core import save_json

    checkpoint = Path(args.checkpoint)
    model, saved = ppo.load_policy(checkpoint, args.device)
    # The original training workflow admitted completed demonstrations without
    # rewriting their review metadata. Preserve that exact eligibility rule,
    # then add accepted DAgger episodes to the training side only.
    episodes = eligible_episodes(args.dataset, include_completed_unreviewed=True)
    split, dagger = _split_for_dagger(episodes, saved)
    if any(signature(json.loads((path / "metadata.json").read_text())["config"]) !=
           signature(saved["config"]) for path in episodes):
        raise ValueError("Dataset calibration/camera/control rate differs from the checkpoint")

    torch.manual_seed(args.seed)
    augmentation = SequenceAugment(seed=args.seed) if args.augment else None
    loaders = {
        "train": DataLoader(Sequences(split["train"], args.sequence_length, args.stride,
                                       augmentation=augmentation),
                            batch_size=args.batch_size, shuffle=True, num_workers=0),
        "val": DataLoader(Sequences(split["val"], args.sequence_length, args.stride),
                          batch_size=args.batch_size, shuffle=False, num_workers=0),
    }
    Sequences(split["test"], args.sequence_length, args.stride)

    for parameter in model.parameters():
        parameter.requires_grad = False
    modules = [model.actor_fc, model.actor_out]
    if args.train_projection:
        modules.append(model.projection)
    if args.train_gru:
        modules.append(model.gru)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite " + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    history_path = output.with_suffix(".history.json")
    best, stale, history = float("inf"), 0, []

    scale = model.action_scale.to(device=args.device, dtype=torch.float32)[None, None]
    for epoch in range(args.epochs):
        metrics = {"epoch": epoch + 1}
        for phase in ("train", "val"):
            model.train(phase == "train")
            total, elements = 0., 0
            for images, previous, target in loaders[phase]:
                images = images.to(args.device)
                previous = previous.to(args.device)
                target = target.to(args.device)
                with torch.set_grad_enabled(phase == "train"):
                    recurrent, _ = model.recurrent(images, previous)
                    location, _ = model.actor_value_recurrent(recurrent)
                    prediction = torch.tanh(location) * scale
                    bounded_target = torch.max(torch.min(target, scale), -scale)
                    loss = ((prediction - bounded_target) ** 2).sum(dim=-1).mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite DAgger loss")
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
                        optimizer.step()
                count = int(target.shape[0] * target.shape[1])
                total += float(loss.detach().cpu().item()) * count
                elements += count
            metrics[phase + "_loss"] = total / elements
        history.append(metrics)
        print(metrics, flush=True)
        save_json(history_path, history)
        if metrics["val_loss"] < best:
            best, stale = metrics["val_loss"], 0
            payload = dict(saved)
            payload["state_dict"] = {
                key: value.detach().cpu() for key, value in model.state_dict().items()}
            payload["optimizer_state_dict"] = None
            rounds = list(saved.get("dagger_rounds", []))
            rounds.append({
                "source_checkpoint": str(checkpoint),
                "dataset": str(args.dataset),
                "dagger_episodes": [path.name for path in dagger],
                "aggregated_train_episodes": len(split["train"]),
                "epochs_completed": epoch + 1,
                "best_val_loss": float(best),
                "learning_rate": float(args.learning_rate),
                "train_projection": bool(args.train_projection),
                "train_gru": bool(args.train_gru),
            })
            payload["dagger_rounds"] = rounds
            temporary = output.with_suffix(output.suffix + ".tmp")
            torch.save(payload, str(temporary))
            temporary.replace(output)
        else:
            stale += 1
            if stale >= args.patience:
                break
    print("Best DAgger checkpoint: {} (val_loss={:.6f})".format(output, best), flush=True)
    if args.train_gru:
        print("WARNING: the GRU changed; retrain the stop head before deployment.", flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="task")

    collect_parser = sub.add_parser("collect", help="Collect one physical DAgger episode")
    collect_parser.add_argument("--checkpoint", type=Path, required=True)
    collect_parser.add_argument("--dataset", type=Path, required=True,
                                help="Original dataset root; DAgger episodes are added here")
    collect_parser.add_argument("--controller", choices=("keyboard", "gamepad"), default="keyboard")
    collect_parser.add_argument("--beta", type=float, default=.5,
                                help="Executed expert fraction; decay toward zero each round")
    collect_parser.add_argument("--seconds", type=float, default=40.)
    collect_parser.add_argument("--scenario", default="dagger_recovery")
    collect_parser.add_argument("--sequence-length", type=int, default=16)
    collect_parser.add_argument("--warmup-frames", type=int, default=5)
    collect_parser.add_argument("--device", default="cuda")
    collect_parser.add_argument("--trt-encoder", type=Path)
    collect_parser.add_argument("--fp16-encoder", action="store_true")
    collect_parser.add_argument("--max-frame-age-ms", type=float, default=500.)
    collect_parser.add_argument("--steering-scale", type=float, default=1.)
    collect_parser.add_argument("--throttle-scale", type=float, default=1.)
    collect_parser.add_argument("--initial-forward-thrust", type=float, default=0.,
                                help="Positive magnitude subtracted from initial policy throttle")
    collect_parser.add_argument("--initial-forward-steps", type=int, default=0)
    collect_parser.add_argument("--initial-forward-cap", type=float, default=1.)

    train_parser = sub.add_parser("train", help="Fine-tune the PPO actor on aggregated expert labels")
    train_parser.add_argument("--checkpoint", type=Path, required=True,
                              help="Exact PPO checkpoint used for collection")
    train_parser.add_argument("--dataset", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--sequence-length", type=int, default=16)
    train_parser.add_argument("--stride", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=3e-5)
    train_parser.add_argument("--max-grad-norm", type=float, default=1.)
    train_parser.add_argument("--patience", type=int, default=4)
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.add_argument("--augment", action="store_true")
    train_parser.add_argument("--train-projection", action="store_true")
    train_parser.add_argument("--train-gru", action="store_true",
                              help="Also adapt temporal memory; requires retraining the stop head")
    train_parser.add_argument("--device", default="cuda")

    args = parser.parse_args(argv)
    if args.task is None:
        parser.print_help()
        return
    if args.task == "collect":
        if not math.isfinite(args.beta) or not 0. <= args.beta <= 1.:
            parser.error("--beta must be from 0 to 1")
        if not 0. < args.seconds <= 300.:
            parser.error("--seconds must be above 0 and at most 300")
        if args.sequence_length < 1 or args.warmup_frames < 0:
            parser.error("sequence length must be positive and warmup frames cannot be negative")
        if args.trt_encoder is not None and args.fp16_encoder:
            parser.error("choose either --trt-encoder or --fp16-encoder")
        try:
            from parking_lib.ppo_loop import _validate_runtime_action_controls
            _validate_runtime_action_controls(
                args.throttle_scale, args.steering_scale,
                args.initial_forward_thrust, args.initial_forward_steps,
                args.initial_forward_cap)
        except ValueError as error:
            parser.error(str(error))
        collect(args)
    else:
        if min(args.epochs, args.batch_size, args.sequence_length, args.stride,
               args.patience) < 1:
            parser.error("training counts must be positive")
        if (not math.isfinite(args.learning_rate) or args.learning_rate <= 0. or
                not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0.):
            parser.error("learning rate and maximum gradient norm must be positive")
        train(args)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # JetPack 4 can invalid-free the CSI/GStreamer object during teardown.
        if len(sys.argv) > 1 and sys.argv[1] == "collect":
            os._exit(1)
        raise
    if len(sys.argv) > 1 and sys.argv[1] == "collect":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
