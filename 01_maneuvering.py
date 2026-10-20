#!/usr/bin/env python3
"""Lesson 1: bounded manual control and a stationary direction-check exercise."""
import argparse
import threading
from pathlib import Path
from parking_lib.core import Car, load_config, stop_signals
from parking_lib.controls import KEYBOARD_HELP
from parking_lib.session import run_session

DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def steering_exercise(cfg, real=False):
    """Throttle remains zero; verify which physical direction each sign means."""
    stop = threading.Event()
    with stop_signals(stop), Car(cfg, real) as car:
        for steering in (0., -.5, 0., .5, 0.):
            print("Normalized steering:", steering, "throttle: 0")
            for _ in range(int(cfg["control"]["hz"])):
                if stop.is_set():
                    return
                car.apply((steering, 0.))
                stop.wait(1. / cfg["control"]["hz"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--real", action="store_true", help="Enable verified physical hardware")
    parser.add_argument("--controller", choices=["keyboard", "gamepad", "synthetic"], default="keyboard")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--steering-exercise", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    print("REAL HARDWARE" if args.real else "DRY RUN: motors are simulated")
    if args.steering_exercise:
        steering_exercise(cfg, args.real)
    else:
        print(KEYBOARD_HELP)
        run_session(cfg, args.real, args.controller, args.seconds)


if __name__ == "__main__":
    main()
