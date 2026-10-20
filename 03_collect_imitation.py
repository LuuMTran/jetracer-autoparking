#!/usr/bin/env python3
"""Lesson 3: collect synchronized demonstrations, validate them, and review outcomes."""
import argparse
import json
from pathlib import Path
from parking_lib.core import load_config
from parking_lib.controls import KEYBOARD_HELP
from parking_lib.data import review_episode, validate_episode
from parking_lib.session import run_session


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="task")
    collect = sub.add_parser("collect")
    collect.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    collect.add_argument("--real", action="store_true")
    collect.add_argument("--controller", choices=["keyboard", "gamepad", "synthetic"], default="keyboard")
    collect.add_argument("--dataset", type=Path, default=Path("dataset"))
    collect.add_argument("--seconds", type=float, default=60)
    collect.add_argument("--scenario", default="normal", help="E.g. forward_start, close_to_curb, recovery")
    validate = sub.add_parser("validate")
    validate.add_argument("path", type=Path, help="An episode directory or dataset root")
    review = sub.add_parser("review")
    review.add_argument("episode", type=Path)
    review.add_argument("--outcome", required=True, choices=["success", "near", "neutral", "poor", "collision", "recovery", "discard"])
    review.add_argument("--accept", action="store_true", help="Explicitly admit this demonstration to behavior cloning")
    review.add_argument("--score", type=float, help="Optional human terminal reward, -1 to +1")
    review.add_argument("--notes", default="")
    args = parser.parse_args(argv)
    if args.task == "collect":
        cfg = load_config(args.config)
        print("REAL HARDWARE" if args.real else "DRY RUN: synthetic images and simulated motors")
        print(KEYBOARD_HELP)
        run_session(cfg, args.real, args.controller, args.seconds, args.dataset, args.scenario)
    elif args.task == "validate":
        paths = [args.path] if (args.path / "metadata.json").exists() else sorted(args.path.glob("episode_*"))
        if not paths:
            raise SystemExit("No episodes found")
        reports = [validate_episode(p) for p in paths]
        print(json.dumps(reports, indent=2))
        if any(not r["valid"] for r in reports):
            raise SystemExit(1)
    elif args.task == "review":
        print(json.dumps(review_episode(args.episode, args.outcome, args.accept, args.score, args.notes), indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
