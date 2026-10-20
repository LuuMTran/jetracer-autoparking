#!/usr/bin/env python3
"""Lesson 2: train a recurrent behavior-cloning policy, evaluate it, then drive."""
import argparse
import json
import time
from pathlib import Path
from parking_lib.core import load_config, save_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="task")
    train = sub.add_parser("train", help="Train using reviewed demonstrations from lesson 3")
    train.add_argument("--dataset", type=Path, default=Path("dataset"))
    train.add_argument("--output", type=Path, default=Path("runs/parking_v1"))
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--finetune-epochs", type=int, default=0)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--sequence-length", type=int, default=16)
    train.add_argument("--stride", type=int, default=8)
    train.add_argument("--patience", type=int, default=5)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--split", type=Path, help="Reuse an earlier split.json; add new episodes to training only")
    train.add_argument("--augment", action="store_true", help="Mild sequence-consistent camera augmentation")
    train.add_argument("--init-checkpoint", type=Path, help="Fine-tune an existing imitation policy")
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    train.add_argument("--device", default="cpu", help="cpu or cuda; start with a small batch on Jetson")
    train.add_argument("--no-pretrained", action="store_true", help="Offline smoke tests only; normal training uses ImageNet")
    train.add_argument("--allow-synthetic", action="store_true", help="Allow synthetic smoke-test episodes")
    train.add_argument("--architecture",
                       choices=("resnet18-proj256-gru64-mlp32", "mobilenet-v2-proj256-gru64-mlp32"),
                       default="resnet18-proj256-gru64-mlp32",
                       help="Visual encoder; MobileNetV2 is intended for Jetson Nano deployment")
    train.add_argument("--include-completed-unreviewed", action="store_true",
                       help="Train completed human episodes without changing their review metadata")
    evaluate = sub.add_parser("evaluate", help="Replay held-out test episodes without moving the car")
    evaluate.add_argument("--dataset", type=Path, default=Path("dataset"))
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--fp16", action="store_true",
                          help="Evaluate using half precision on CUDA or MPS")
    evaluate.add_argument("--report", type=Path)
    drive = sub.add_parser("drive", help="Camera + previous action -> network -> bounded motor commands")
    drive.add_argument("--checkpoint", type=Path, required=True)
    drive.add_argument("--config", type=Path,
                       help="Runtime config; default uses the calibrated config embedded in the checkpoint")
    drive.add_argument("--real", action="store_true")
    drive.add_argument("--controller", choices=["keyboard", "gamepad", "synthetic"], default="keyboard")
    drive.add_argument("--seconds", type=float, default=20)
    drive.add_argument("--dataset", type=Path, default=Path("policy_trials"))
    drive.add_argument("--no-record", action="store_true",
                       help="Drive without saving camera frames or action rows")
    drive.add_argument("--device", default="cpu")
    drive.add_argument("--fp16", action="store_true",
                       help="Use CUDA half precision to reduce Jetson inference latency")
    drive.add_argument("--trt-encoder", type=Path,
                       help="TensorRT FP16 encoder built from this exact checkpoint on the Jetson")
    drive.add_argument("--stop-head", type=Path,
                       help="Recurrent terminal-stop head trained from this exact checkpoint")
    drive.add_argument("--steering-cap", type=float, default=1.,
                       help="Absolute normalized policy steering cap, 0 to 1")
    drive.add_argument("--throttle-cap", type=float, default=1.,
                       help="Absolute normalized policy throttle cap, 0 to 1")
    drive.add_argument("--throttle-scale", type=float, default=1.,
                       help="Multiply normalized policy throttle before applying the cap")
    drive.add_argument("--warmup-seconds", type=float, default=.5,
                       help="Stationary recurrent warm-up before policy commands; default 0.5")
    drive.add_argument("--stop-threshold", type=float,
                       help="Stop probability threshold; default comes from the stop-head artifact")
    drive.add_argument("--stop-frames", type=int,
                       help="Required consecutive stop predictions; default comes from the artifact")
    drive.add_argument("--stop-min-seconds", type=float,
                       help="Earliest terminal stop; default comes from the artifact")
    benchmark = sub.add_parser("benchmark", help="Measure camera + inference latency without motors")
    benchmark.add_argument("--checkpoint", type=Path, required=True)
    benchmark.add_argument("--config", type=Path,
                           help="Runtime config; default uses the config embedded in the checkpoint")
    benchmark.add_argument("--real-camera", action="store_true")
    benchmark.add_argument("--device", default="cpu")
    benchmark.add_argument("--fp16", action="store_true")
    benchmark.add_argument("--trt-encoder", type=Path,
                           help="TensorRT FP16 encoder built from this exact checkpoint on the Jetson")
    benchmark.add_argument("--samples", type=int, default=30)
    args = parser.parse_args(argv)
    if args.task is None:
        parser.print_help()
        return
    from parking_lib import learning
    if args.task == "train":
        result = learning.train(args.dataset, args.output, epochs=args.epochs,
                                finetune_epochs=args.finetune_epochs, batch_size=args.batch_size,
                                sequence_length=args.sequence_length, stride=args.stride,
                                seed=args.seed, patience=args.patience, device=args.device,
                                pretrained=not args.no_pretrained, allow_synthetic=args.allow_synthetic,
                                split_file=args.split,
                                include_completed_unreviewed=args.include_completed_unreviewed,
                                architecture=args.architecture, augment=args.augment,
                                init_checkpoint=args.init_checkpoint, learning_rate=args.learning_rate,
                                encoder_learning_rate=args.encoder_learning_rate)
        print("Best checkpoint:", result)
    elif args.task == "evaluate":
        report = learning.evaluate(args.dataset, args.checkpoint, args.device, args.fp16)
        print(json.dumps(report, indent=2))
        if args.report:
            save_json(args.report, report)
    elif args.task in ("drive", "benchmark"):
        from parking_lib.core import canonical_config
        cfg = load_config(args.config) if args.config else canonical_config(
            learning.load_checkpoint(args.checkpoint, "cpu")["config"])
        predictor = learning.Predictor(args.checkpoint, cfg, args.device,
                                       getattr(args, "real", False), args.fp16,
                                       getattr(args, "trt_encoder", None),
                                       getattr(args, "stop_head", None))
        if args.task == "drive":
            from parking_lib.session import run_session
            print("Hold/repeat W to enable network actions; Space releases; Escape emergency-stops; F finishes.")
            drive_dataset = None if args.no_record else args.dataset
            if drive_dataset is None:
                print("Recording disabled; this run will not create a policy-trial episode.", flush=True)
            stop_threshold = (args.stop_threshold if args.stop_threshold is not None else
                              predictor.stop_decision.get("threshold", .8))
            stop_frames = (args.stop_frames if args.stop_frames is not None else
                           predictor.stop_decision.get("consecutive_frames", 4))
            stop_min_s = (args.stop_min_seconds if args.stop_min_seconds is not None else
                          predictor.stop_decision.get("minimum_elapsed_s", 1.))
            run_session(cfg, args.real, args.controller, args.seconds, drive_dataset, "policy_trial",
                        policy=predictor, policy_caps=(args.steering_cap, args.throttle_cap),
                        policy_warmup_s=args.warmup_seconds,
                        policy_throttle_scale=args.throttle_scale,
                        policy_stop_threshold=stop_threshold, policy_stop_frames=stop_frames,
                        policy_stop_min_s=stop_min_s)
        else:
            if args.samples < 5:
                raise ValueError("Use at least 5 benchmark samples")
            from parking_lib.camera import Camera
            from parking_lib.core import physical
            latencies, ages = [], []
            with Camera(cfg, args.real_camera) as camera:
                frame = camera.read()
                for _ in range(3):
                    predictor.reset()
                    predictor(frame["rgb"], (0., 0.))
                predictor.reset()
                previous, last_id = (0., 0.), -1
                for _ in range(args.samples):
                    frame = camera.read(after=last_id, timeout=2.)
                    last_id = frame["frame_id"]
                    started = time.monotonic()
                    predicted = predictor(frame["rgb"], previous)
                    previous = physical(predicted, cfg["limits"])
                    latencies.append(time.monotonic() - started)
                    ages.append(time.monotonic() - frame["frame_monotonic_s"])
            def percentile(values, fraction):
                ordered = sorted(values)
                return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))]
            report = {"samples": len(latencies), "device": args.device, "fp16": args.fp16,
                      "backend": predictor.backend,
                      "architecture": predictor.architecture,
                      "camera_config": cfg["camera"],
                      "captured_frame_shape": list(frame["rgb"].shape),
                      "inference_ms": {"median": percentile(latencies, .5) * 1000.,
                                       "p95": percentile(latencies, .95) * 1000.,
                                       "max": max(latencies) * 1000.},
                      "frame_age_ms": {"median": percentile(ages, .5) * 1000.,
                                       "p95": percentile(ages, .95) * 1000.,
                                       "max": max(ages) * 1000.},
                      "estimated_decisions_per_second": 1. / percentile(latencies, .5),
                      "configured_decisions_per_second": cfg["control"]["hz"],
                      "configured_max_frame_age_ms": cfg["control"]["max_frame_age_s"] * 1000.,
                      "frame_age_p95_within_limit": percentile(ages, .95) <= cfg["control"]["max_frame_age_s"]}
            print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
