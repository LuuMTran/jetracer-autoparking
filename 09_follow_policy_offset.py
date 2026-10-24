#!/usr/bin/env python3
"""Run a PPO policy with runtime steering-offset and throttle-scale overrides."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback


def follow(args):
    from parking_lib.camera import Camera
    from parking_lib.core import Car
    from parking_lib import ppo
    from parking_lib.ppo_loop import (
        TerminalKeys, _configure_inference_backend, _runtime_config,
        apply_runtime_steering_config, follow_episode, inference_checkpoint,
        load_stop_head,
    )

    checkpoint = inference_checkpoint(args.checkpoint)
    model, saved = ppo.load_policy(checkpoint, args.device)
    cfg = _runtime_config(saved, args.max_frame_age_ms)

    checkpoint_offset, checkpoint_limit = apply_runtime_steering_config(
        cfg, args.steering_offset, args.steering_limit)

    print("FOLLOW CHECKPOINT: {}".format(checkpoint), flush=True)
    print("Steering offset: checkpoint={:+.3f} runtime={:+.3f}".format(
        checkpoint_offset, float(cfg["hardware"]["steering_offset"])), flush=True)
    print("Steering limit: checkpoint={:.3f} runtime={:.3f}".format(
        checkpoint_limit, float(cfg["limits"]["steering"])), flush=True)
    print("Steering scale: {:.3f}".format(args.steering_scale), flush=True)
    print("Throttle scale: {:.3f}".format(args.throttle_scale), flush=True)
    if args.initial_forward_thrust > 0.:
        print("Initial forward throttle: bias=-{:.3f} steps={} cap={:.3f}".format(
            args.initial_forward_thrust, args.initial_forward_steps,
            args.initial_forward_cap), flush=True)

    backend = _configure_inference_backend(
        model, saved, args.device, args.trt_encoder, args.fp16_encoder)
    print("Inference backend: {}".format(backend), flush=True)
    stop_head, stop_decision = load_stop_head(args.stop_head, saved, args.device)

    control_hz = float(cfg["control"]["hz"])
    if args.seconds is None:
        maximum_steps = int(saved["online_config"]["rollout"]["maximum_episode_steps"])
    else:
        if not 0. < float(args.seconds) <= float(cfg["control"]["max_episode_s"]):
            raise ValueError("--seconds must be above 0 and no greater than {:.0f}".format(
                cfg["control"]["max_episode_s"]))
        maximum_steps = max(1, int(round(float(args.seconds) * control_hz)))

    completed_runs = 0
    print("Deterministic follow only; no rollouts, rewards, or updates.", flush=True)
    with TerminalKeys() as keys, Camera(
            cfg, True, threaded=False, release_on_close=False) as camera, Car(cfg, True) as car:
        car.park("follow_ready")
        while keys.wait_start():
            reason, exit_requested, steps = follow_episode(
                model, cfg, camera, car, keys, args.device, backend, stop_head,
                stop_decision, args.warmup_seconds, maximum_steps,
                args.throttle_scale, steering_scale=args.steering_scale,
                initial_forward_thrust=args.initial_forward_thrust,
                initial_forward_steps=args.initial_forward_steps,
                initial_forward_cap=args.initial_forward_cap)
            completed_runs += 1
            print("FOLLOW STOPPED: {} after {} applied actions".format(
                reason, steps), flush=True)
            if exit_requested:
                break

    return {
        "checkpoint": str(checkpoint),
        "update": int(saved["update"]),
        "runs_completed": completed_runs,
        "steering_offset": float(cfg["hardware"]["steering_offset"]),
        "steering_limit": float(cfg["limits"]["steering"]),
        "steering_scale": float(args.steering_scale),
        "throttle_scale": float(args.throttle_scale),
        "initial_forward_thrust": float(args.initial_forward_thrust),
        "initial_forward_steps": int(args.initial_forward_steps),
        "initial_forward_cap": float(args.initial_forward_cap),
        "mode": "deterministic_follow_with_runtime_calibration",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Exact PPO checkpoint, or a directory using its newest checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trt-encoder", type=Path)
    parser.add_argument("--fp16-encoder", action="store_true")
    parser.add_argument("--stop-head", type=Path)
    parser.add_argument("--seconds", type=float,
                        help="Maximum time per run; checkpoint limit used by default")
    parser.add_argument("--throttle-scale", type=float, default=1.,
                        help="Multiply policy throttle at runtime (above 0 to 1)")
    parser.add_argument("--steering-offset", type=float,
                        help="Override hardware steering offset (-0.6 to 0.6)")
    parser.add_argument("--steering-scale", type=float, default=1.,
                        help="Multiply policy steering before limiting (default 1)")
    parser.add_argument("--steering-limit", type=float,
                        help="Override physical steering limit (above 0 to 1)")
    parser.add_argument("--initial-forward-thrust", type=float, default=0.,
                        help="Positive magnitude subtracted from initial throttle")
    parser.add_argument("--initial-forward-steps", type=int, default=0,
                        help="Number of initial applied actions receiving forward thrust")
    parser.add_argument("--initial-forward-cap", type=float, default=1.,
                        help="Maximum negative throttle magnitude during initial thrust")
    parser.add_argument("--warmup-seconds", type=float, default=.5)
    parser.add_argument("--max-frame-age-ms", type=float)
    args = parser.parse_args(argv)

    if not math.isfinite(args.throttle_scale) or not 0. < args.throttle_scale <= 1.:
        parser.error("--throttle-scale must be above 0 and no greater than 1")
    if args.steering_offset is not None:
        if not math.isfinite(args.steering_offset):
            parser.error("--steering-offset must be finite")
        if abs(args.steering_offset) > .6:
            parser.error("--steering-offset must be from -0.6 to 0.6")
    if not math.isfinite(args.steering_scale) or args.steering_scale <= 0.:
        parser.error("--steering-scale must be a positive finite number")
    if args.steering_limit is not None:
        if not math.isfinite(args.steering_limit) or not 0. < args.steering_limit <= 1.:
            parser.error("--steering-limit must be above 0 and no greater than 1")
    if (not math.isfinite(args.initial_forward_thrust) or
            not 0. <= args.initial_forward_thrust <= 1.):
        parser.error("--initial-forward-thrust must be from 0 to 1")
    if args.initial_forward_steps < 0:
        parser.error("--initial-forward-steps must be zero or positive")
    if (not math.isfinite(args.initial_forward_cap) or
            not 0. < args.initial_forward_cap <= 1.):
        parser.error("--initial-forward-cap must be above 0 and no greater than 1")
    if args.initial_forward_thrust > 0. and args.initial_forward_steps < 1:
        parser.error("--initial-forward-steps must be positive when startup thrust is used")
    if args.trt_encoder is not None and args.fp16_encoder:
        parser.error("choose either --trt-encoder or --fp16-encoder")

    try:
        print(json.dumps(follow(args), indent=2, allow_nan=False), flush=True)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    # Avoid JetPack 4 invalid-free errors during CSI/GStreamer teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
