#!/usr/bin/env python3
"""Collect human stop signals, fine-tune the stop head, or run two policies in sequence."""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="task")

    collect = sub.add_parser("collect", help="Follow PPO while S records a desired stop state")
    collect.add_argument("--checkpoint", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--device", default="cuda")
    collect.add_argument("--fp16-encoder", action="store_true")
    collect.add_argument("--seconds", type=float, default=40.)
    collect.add_argument("--warmup-seconds", type=float, default=.5)
    collect.add_argument("--max-frame-age-ms", type=float)
    collect.add_argument("--pre-label-frames", type=int, default=2)
    collect.add_argument("--post-label-frames", type=int, default=2)
    collect.add_argument("--max-episodes", type=int, default=0,
                         help="0 keeps collecting until Escape")

    train = sub.add_parser("train", help="Fine-tune only the stop head from collected signals")
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--initial-stop-head", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", default="cpu")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=7)

    sequence = sub.add_parser("sequence", help="Run stage one and switch to stage two on stop")
    sequence.add_argument("--first-checkpoint", type=Path, required=True)
    sequence.add_argument("--first-stop-head", type=Path, required=True)
    sequence.add_argument("--second-checkpoint", type=Path, required=True)
    sequence.add_argument("--second-stop-head", type=Path, required=True)
    sequence.add_argument("--device", default="cuda")
    sequence.add_argument("--fp16-encoder", action="store_true")
    sequence.add_argument("--first-seconds", type=float, default=40.)
    sequence.add_argument("--second-seconds", type=float, default=40.)
    sequence.add_argument("--warmup-seconds", type=float, default=.5)
    sequence.add_argument("--max-frame-age-ms", type=float)
    sequence.add_argument("--second-throttle-scale", type=float, default=1.,
                          help="multiply only stage-two throttle by 0 < scale <= 1")
    sequence.add_argument("--slot-classifier", type=Path,
                          help="3-frame classifier; left runs both stages, otherwise only stage one")
    sequence.add_argument("--slot-device", default="cpu",
                          help="device for the small route classifier (default: cpu)")

    args = parser.parse_args(argv)
    if args.task is None:
        parser.print_help()
        return
    from parking_lib.stop_tuning import (collect_stop_labels, fine_tune_stop_head,
                                         run_policy_sequence)
    if args.task == "collect":
        report = collect_stop_labels(
            args.checkpoint, args.output, args.device, args.fp16_encoder,
            args.seconds, args.max_frame_age_ms, args.warmup_seconds,
            args.pre_label_frames, args.post_label_frames, args.max_episodes)
    elif args.task == "train":
        result = fine_tune_stop_head(
            args.checkpoint, args.dataset, args.initial_stop_head, args.output,
            args.device, args.epochs, args.patience, args.batch_size,
            args.learning_rate, args.seed)
        report = {"stop_head": str(result)}
    else:
        report = run_policy_sequence(
            args.first_checkpoint, args.first_stop_head,
            args.second_checkpoint, args.second_stop_head,
            args.device, args.fp16_encoder, args.first_seconds,
            args.second_seconds, args.warmup_seconds, args.max_frame_age_ms,
            args.second_throttle_scale, args.slot_classifier, args.slot_device)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
