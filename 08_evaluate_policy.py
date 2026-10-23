#!/usr/bin/env python3
"""Evaluate an exact direct- or reverse-parking PPO checkpoint on the car."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as destination:
        json.dump(value, destination, indent=2, allow_nan=False)
        destination.write("\n")
    temporary.replace(path)


def atomic_csv(path, rows):
    fields = [
        "episode", "score", "outcome", "end_reason", "steps", "elapsed_s",
        "policy", "condition", "checkpoint", "update", "recorded_at",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summarize(rows):
    scores = [float(row["score"]) for row in rows]
    total = len(rows)
    return {
        "completed_episodes": total,
        "mean_score": (sum(scores) / total if total else None),
        "score_standard_deviation": (
            statistics.pstdev(scores) if len(scores) > 1 else (0.0 if scores else None)),
        "minimum_score": (min(scores) if scores else None),
        "maximum_score": (max(scores) if scores else None),
        "successes": sum(row["outcome"] == "success" for row in rows),
        "failures": sum(row["outcome"] == "failure" for row in rows),
        "collisions": sum(row["outcome"] == "collision" for row in rows),
        "success_rate": (
            sum(row["outcome"] == "success" for row in rows) / float(total)
            if total else None),
        "collision_rate": (
            sum(row["outcome"] == "collision" for row in rows) / float(total)
            if total else None),
    }


class EvaluationKeys(object):
    """Add a categorical evaluation outcome to the existing terminal controls."""
    def __init__(self):
        from parking_lib.ppo_loop import TerminalKeys
        self.keys = TerminalKeys()

    def __enter__(self):
        self.keys.__enter__()
        return self

    def __exit__(self, *exc):
        return self.keys.__exit__(*exc)

    def wait_start(self):
        return self.keys.wait_start()

    def poll_episode(self):
        return self.keys.poll_episode()

    def wait_score(self):
        return self.keys.wait_score()

    def wait_outcome(self):
        print("OUTCOME: Y=successful parking, N=unsuccessful, C=collision, "
              "D=discard, Escape=exit.", flush=True)
        while True:
            key = self.keys.read(.1)
            if key == "y":
                print("success", flush=True)
                return "success"
            if key == "n":
                print("failure", flush=True)
                return "failure"
            if key == "c":
                print("collision", flush=True)
                return "collision"
            if key == "d":
                print("discard", flush=True)
                return "discard"
            if key == "\x1b":
                print(flush=True)
                return "exit"


def unique_run_directory(root, policy, update):
    root.mkdir(parents=True, exist_ok=True)
    stem = "{}_{}_update_{:04d}".format(
        policy, time.strftime("%Y%m%d_%H%M%S"), int(update))
    destination = root / stem
    suffix = 1
    while destination.exists():
        destination = root / (stem + "_{}".format(suffix))
        suffix += 1
    destination.mkdir()
    return destination


def evaluate(args):
    import torch
    from parking_lib.camera import Camera
    from parking_lib.core import Car
    from parking_lib import ppo
    from parking_lib.ppo_loop import (
        _configure_inference_backend, _runtime_config, follow_episode,
        inference_checkpoint, load_stop_head,
    )

    checkpoint = inference_checkpoint(args.checkpoint)
    model, saved = ppo.load_policy(checkpoint, args.device)
    cfg = _runtime_config(saved, args.max_frame_age_ms)
    backend = _configure_inference_backend(
        model, saved, args.device, args.trt_encoder, args.fp16_encoder)
    stop_head, stop_decision = load_stop_head(args.stop_head, saved, args.device)
    control_hz = float(cfg["control"]["hz"])
    if args.seconds is None:
        maximum_steps = int(saved["online_config"]["rollout"]["maximum_episode_steps"])
    else:
        if not 0. < float(args.seconds) <= float(cfg["control"]["max_episode_s"]):
            raise ValueError("--seconds must be above 0 and no greater than {:.0f}".format(
                cfg["control"]["max_episode_s"]))
        maximum_steps = max(1, int(round(float(args.seconds) * control_hz)))

    run_directory = unique_run_directory(args.output_dir, args.policy, saved["update"])
    json_path = run_directory / "evaluation.json"
    csv_path = run_directory / "episodes.csv"
    metadata = {
        "schema_version": 1,
        "evaluation_mode": "deterministic_human_scored",
        "policy": args.policy,
        "condition": args.condition,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": ppo.file_sha256(checkpoint),
        "update": int(saved["update"]),
        "requested_episodes": int(args.episodes),
        "device": str(args.device),
        "backend": backend,
        "stop_head": str(args.stop_head) if args.stop_head is not None else None,
        "throttle_scale": float(args.throttle_scale),
        "maximum_steps": int(maximum_steps),
        "control_hz": control_hz,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    rows = []

    def save():
        payload = dict(metadata)
        payload["summary"] = summarize(rows)
        payload["episodes"] = rows
        atomic_json(json_path, payload)
        atomic_csv(csv_path, rows)

    save()
    print("EVALUATION CHECKPOINT: {}".format(checkpoint), flush=True)
    print("POLICY: {} | CONDITION: {}".format(args.policy, args.condition), flush=True)
    print("Deterministic actions; the checkpoint will not be updated.", flush=True)
    print("Results: {}".format(run_directory), flush=True)

    with EvaluationKeys() as keys, Camera(
            cfg, True, threaded=False, release_on_close=False) as camera, Car(cfg, True) as car:
        car.park("evaluation_ready")
        while len(rows) < int(args.episodes) and keys.wait_start():
            started = time.monotonic()
            reason, exit_requested, steps = follow_episode(
                model, cfg, camera, car, keys, args.device, backend, stop_head,
                stop_decision, args.warmup_seconds, maximum_steps,
                args.throttle_scale)
            elapsed = time.monotonic() - started
            print("EVALUATION STOPPED: {} after {} actions".format(reason, steps), flush=True)
            if exit_requested:
                break

            score = keys.wait_score()
            if score == "exit":
                break
            if score == "discard":
                print("Episode discarded; it will not count toward the target.", flush=True)
                continue
            outcome = keys.wait_outcome()
            if outcome == "exit":
                break
            if outcome == "discard":
                print("Episode discarded; it will not count toward the target.", flush=True)
                continue

            row = {
                "episode": len(rows) + 1,
                "score": float(score["reward"]),
                "outcome": outcome,
                "end_reason": reason,
                "steps": int(steps),
                "elapsed_s": round(float(elapsed), 3),
                "policy": args.policy,
                "condition": args.condition,
                "checkpoint": str(checkpoint),
                "update": int(saved["update"]),
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            rows.append(row)
            save()
            current = summarize(rows)
            print("RECORDED: episode {}/{} score={:+.3f} outcome={} "
                  "running_mean={:+.3f} success_rate={:.1%}".format(
                      len(rows), args.episodes, row["score"], outcome,
                      current["mean_score"], current["success_rate"]), flush=True)

    metadata["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()
    result = summarize(rows)
    result.update({"results_json": str(json_path), "episodes_csv": str(csv_path)})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Exact PPO .pt file, or a directory whose newest checkpoint is used")
    parser.add_argument("--policy", choices=("direct", "reverse"), required=True)
    parser.add_argument("--condition", default="unspecified",
                        help="Fixed test condition, such as left-slot-start-A")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluations"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trt-encoder", type=Path)
    parser.add_argument("--fp16-encoder", action="store_true")
    parser.add_argument("--stop-head", type=Path)
    parser.add_argument("--seconds", type=float,
                        help="Maximum time per episode; the checkpoint limit is used by default")
    parser.add_argument("--throttle-scale", type=float, default=1.)
    parser.add_argument("--warmup-seconds", type=float, default=.5)
    parser.add_argument("--max-frame-age-ms", type=float)
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if not math.isfinite(args.throttle_scale) or not 0. < args.throttle_scale <= 1.:
        parser.error("--throttle-scale must be above 0 and no greater than 1")
    if args.trt_encoder is not None and args.fp16_encoder:
        parser.error("choose either --trt-encoder or --fp16-encoder")
    try:
        result = evaluate(args)
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    # JetPack 4 may invalid-free a CSI/GStreamer object during normal teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
