#!/usr/bin/env python3
"""Initialize, train, inspect, or deterministically run a recurrent PPO policy."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback


def load_json(path):
    with open(str(path)) as source:
        return json.load(source)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="task")
    init = sub.add_parser("init", help="Create PPO version zero from best.pt")
    init.add_argument("--imitation-checkpoint", type=Path, required=True)
    init.add_argument("--config", type=Path, default=Path(__file__).with_name("ppo_config.json"))
    init.add_argument("--output", type=Path, default=Path("runs/ppo_v1"))
    init.add_argument("--device", default="cpu")
    inspect = sub.add_parser("inspect", help="Inspect a PPO checkpoint and readiness gates")
    inspect.add_argument("--checkpoint", type=Path, required=True)
    inspect.add_argument("--device", default="cpu")
    train_loop = sub.add_parser("train-loop", help="Watch for rollouts and keep creating PPO updates")
    train_loop.add_argument("--checkpoint", type=Path, required=True,
                            help="PPO checkpoint or run directory; the newest version is resumed")
    train_loop.add_argument("--run-dir", type=Path, required=True,
                            help="Run directory containing rollout update folders")
    train_loop.add_argument("--device", default="cpu")
    train_loop.add_argument("--episodes-per-update", type=int, default=5,
                            help="Scored episodes per update, including resumed runs (default 5)")
    train_loop.add_argument("--poll-seconds", type=float, default=2.)
    train_loop.add_argument("--max-updates", type=int, default=0,
                            help="0 keeps waiting and training until interrupted")
    physical = sub.add_parser("physical-loop", help="Repeated physical episodes and PPO updates")
    physical.add_argument("--checkpoint", type=Path, required=True,
                          help="PPO checkpoint or run directory; the newest version is resumed")
    physical.add_argument("--run-dir", type=Path, required=True)
    physical.add_argument("--device", default="cuda")
    physical.add_argument("--episodes-per-update", type=int, default=5,
                          help="Scored episodes per update, including resumed runs (default 5)")
    physical.add_argument("--trt-encoder", type=Path,
                          help="TensorRT FP16 encoder built from the imitation checkpoint")
    physical.add_argument("--fp16-encoder", action="store_true",
                          help="Use the PyTorch FP16 encoder instead of TensorRT")
    physical.add_argument("--stop-head", type=Path,
                          help="Frozen terminal stop head built from the imitation checkpoint")
    physical.add_argument("--max-updates", type=int, default=0,
                          help="0 continues until Escape")
    physical.add_argument("--max-episodes", type=int, default=0,
                          help="0 continues until Escape")
    physical.add_argument("--warmup-seconds", type=float, default=.5)
    physical.add_argument("--max-frame-age-ms", type=float,
                          help="Runtime frame-age limit; for example 400 after benchmarking")
    physical.add_argument("--throttle-scale", type=float, default=1.,
                          help="Multiply policy throttle during collection (above 0 to 1)")
    physical.add_argument("--steering-offset", type=float,
                          help="Override hardware steering offset (-0.6 to 0.6)")
    physical.add_argument("--steering-scale", type=float, default=1.,
                          help="Multiply policy steering before limiting (default 1)")
    physical.add_argument("--steering-limit", type=float,
                          help="Override physical steering limit (above 0 to 1)")
    physical.add_argument("--initial-forward-thrust", type=float, default=0.,
                          help="Positive magnitude subtracted from initial throttle")
    physical.add_argument("--initial-forward-steps", type=int, default=0,
                          help="Number of initial applied actions receiving forward thrust")
    physical.add_argument("--initial-forward-cap", type=float, default=1.,
                          help="Maximum negative throttle magnitude during initial thrust")
    follow = sub.add_parser("follow", help="Run the latest PPO policy without training or recording")
    follow.add_argument("--checkpoint", type=Path, required=True,
                        help="Exact PPO checkpoint, or a directory whose newest version is loaded")
    follow.add_argument("--device", default="cuda")
    follow.add_argument("--trt-encoder", type=Path,
                        help="TensorRT FP16 encoder built from the imitation checkpoint")
    follow.add_argument("--fp16-encoder", action="store_true",
                        help="Use the PyTorch FP16 encoder instead of TensorRT")
    follow.add_argument("--stop-head", type=Path,
                        help="Frozen terminal stop head built from the imitation checkpoint")
    follow.add_argument("--seconds", type=float,
                        help="Maximum time per run; default uses the checkpoint episode limit")
    follow.add_argument("--warmup-seconds", type=float, default=.5)
    follow.add_argument("--max-frame-age-ms", type=float,
                        help="Runtime frame-age limit; for example 500 with PyTorch FP16")
    args = parser.parse_args(argv)
    if args.task is None:
        parser.print_help()
        return
    if args.task == "physical-loop":
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
    from parking_lib import ppo
    if args.task == "init":
        online = load_json(args.config)
        checkpoint = ppo.initialize(args.imitation_checkpoint, args.output, online, args.device)
        print("Initialized:", checkpoint)
        print("Physical rollout enabled:", bool(online.get("physical_rollout_enabled")))
        if online.get("reward_mode") == "human_terminal":
            print("Reward mode: score each completed episode in the persistent terminal loop.")
        else:
            print("Calibrate and replay the configured automatic reward before physical collection.")
    elif args.task == "inspect":
        model, saved = ppo.load_policy(args.checkpoint, args.device)
        online = saved["online_config"]
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())
        print(json.dumps({"algorithm": saved["algorithm"], "update": saved["update"],
                          "imitation_checkpoint_sha256": saved["imitation_checkpoint_sha256"],
                          "action_scale": online["action_scale"],
                          "trainable_parameters": trainable, "total_parameters": total,
                          "train_projection": bool(online["algorithm"].get("train_projection", False)),
                          "train_gru": bool(online["algorithm"].get("train_gru", False)),
                          "physical_rollout_enabled": bool(online.get("physical_rollout_enabled"))}, indent=2))
    elif args.task == "train-loop":
        from parking_lib.ppo_loop import train_watch_loop
        checkpoint = train_watch_loop(args.checkpoint, args.run_dir, args.device,
                                      args.poll_seconds, args.max_updates, args.episodes_per_update)
        print("Latest checkpoint:", checkpoint)
    else:
        from parking_lib.ppo_loop import run_follow_loop, run_physical_loop
        try:
            if args.task == "physical-loop":
                report = run_physical_loop(
                    args.checkpoint, args.run_dir, args.device, args.trt_encoder, args.stop_head,
                    args.max_updates, args.max_episodes, args.warmup_seconds,
                    args.max_frame_age_ms, args.fp16_encoder, args.episodes_per_update,
                    args.steering_offset, args.steering_scale, args.steering_limit,
                    args.throttle_scale, args.initial_forward_thrust,
                    args.initial_forward_steps, args.initial_forward_cap)
            else:
                report = run_follow_loop(
                    args.checkpoint, args.device, args.trt_encoder, args.stop_head,
                    args.warmup_seconds, args.max_frame_age_ms, args.fp16_encoder,
                    args.seconds)
            print(json.dumps(report, indent=2), flush=True)
        except BaseException:
            # Car and terminal context managers have already cleaned up. Avoid
            # JetPack 4's invalid free while destroying a TensorRT-used CSI
            # VideoCapture during interpreter teardown.
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
