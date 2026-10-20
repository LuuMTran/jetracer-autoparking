#!/usr/bin/env python3
"""Record repeated parking episodes with Enter, keeping camera and driver open."""
import argparse
from pathlib import Path
from parking_lib.core import load_config
from parking_lib.continuous import run_continuous


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--seconds", type=float, default=60, help="Maximum duration per episode; session remains open")
    parser.add_argument("--scenario", default="normal")
    parser.add_argument("--demo-episodes", type=int, default=0, help="Automatic synthetic episodes for testing; cannot use --real")
    args = parser.parse_args(argv)
    print("REAL HARDWARE" if args.real else "DRY RUN: synthetic images and mock motors")
    print("W/S straight; U/I forward left/right; J/K reverse left/right; Space neutral")
    print("Enter starts/saves an episode; F saves and exits; Escape aborts and exits")
    run_continuous(load_config(args.config), real=args.real, dataset=args.dataset,
                   seconds=args.seconds, scenario=args.scenario, demo_episodes=args.demo_episodes)


if __name__ == "__main__":
    main()
