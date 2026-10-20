#!/usr/bin/env python3
"""Hardware worker launched by the Streamlit parking controller."""
import json
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path


DIRECT_CHECKPOINT = Path(os.environ.get(
    "PARKING_DIRECT_CHECKPOINT", "runs/direct_ppo"))
DIRECT_STOP_HEAD = Path(os.environ.get(
    "PARKING_DIRECT_STOP_HEAD", "runs/direct_imitation/stop_head.pt"))
REVERSE_CHECKPOINT = Path(os.environ.get(
    "PARKING_REVERSE_CHECKPOINT", "runs/reverse_ppo"))
REVERSE_STOP_HEAD = Path(os.environ.get(
    "PARKING_REVERSE_STOP_HEAD", "runs/reverse_stop/stop_head.pt"))
SLOT_CLASSIFIER = Path(os.environ.get(
    "PARKING_SLOT_CLASSIFIER", "runs/slot_classifier/slot_left_classifier.pt"))
DEVICE = "cuda"
FP16_ENCODER = True
DIRECT_SECONDS = 40.
REVERSE_SECONDS = 40.
WARMUP_SECONDS = .5
MAX_FRAME_AGE_MS = 500.
REVERSE_THROTTLE_SCALE = .6


def write_error(run_dir, message):
    path = Path(run_dir) / "status.json"
    current = {}
    try:
        current = json.loads(path.read_text())
    except Exception:
        pass
    current.update(state="error", stage="Stopped", message=message,
                   error=traceback.format_exc(), finished_unix_s=time.time())
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(current, indent=2) + "\n")
    temporary.replace(path)


def main(run_dir):
    stop_event = threading.Event()

    def stop(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    from parking_lib.gui_sequence import run_recorded_sequence
    return run_recorded_sequence(
        DIRECT_CHECKPOINT, DIRECT_STOP_HEAD,
        REVERSE_CHECKPOINT, REVERSE_STOP_HEAD,
        run_dir, stop_event, DEVICE, FP16_ENCODER,
        DIRECT_SECONDS, REVERSE_SECONDS, WARMUP_SECONDS,
        MAX_FRAME_AGE_MS, REVERSE_THROTTLE_SCALE,
        SLOT_CLASSIFIER, "cpu")


if __name__ == "__main__":
    destination = Path(sys.argv[1]).resolve()
    code = 0
    try:
        report = main(destination)
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as exc:
        code = 1
        traceback.print_exc()
        write_error(destination, str(exc))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
