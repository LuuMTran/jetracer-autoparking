"""Durable episodes, explicit review labels, and dataset integrity checks."""
import csv
import hashlib
import io
import json
import math
import os
import time
import platform
import queue
import threading
import uuid
from pathlib import Path
from PIL import Image
from .core import save_json, normalized


FIELDS = ["episode_id", "step", "frame", "frame_id", "rgb_sha256", "frame_unix_s", "frame_monotonic_s",
          "input_monotonic_s", "command_unix_s", "command_start_monotonic_s", "command_end_monotonic_s",
          "elapsed_s", "dt_s", "frame_age_s", "missed_periods",
          "requested_steering_norm", "requested_throttle_norm", "steering", "throttle",
          "steering_norm", "throttle_norm", "prev_steering", "prev_throttle",
          "prev_steering_norm", "prev_throttle_norm", "deadman_active",
          "reward", "topdown_x_m", "topdown_y_m", "topdown_heading_rad"]


class AsyncRecorder:
    """Bounded, ordered disk writer. A full queue or write error aborts the run.

    Only the worker touches the CSV/images. finish() runs after motors stop and
    waits for queued samples before finalizing metadata. A stuck writer leaves
    status=recording (ineligible for training), rather than claiming completion.
    """
    def __init__(self, root, cfg, synthetic, controller, scenario, source="human", capacity=8):
        # Pillow discovers codecs lazily. Pay the first-JPEG startup cost before
        # creating the motor driver, not after issuing the first physical command.
        Image.new("RGB", (8, 8)).save(io.BytesIO(), format="JPEG", quality=95)
        self.recorder = Recorder(root, cfg, synthetic, controller, scenario, source)
        self.path, self.meta = self.recorder.path, self.recorder.meta
        self.meta["disk_writer"] = {"mode": "bounded_background", "capacity_frames": capacity}
        self.queue = queue.Queue(maxsize=capacity)
        self.closing = threading.Event()
        self.error = None
        self.max_write_s = 0.
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def _write(self):
        while True:
            try:
                sample = self.queue.get(timeout=.05)
            except queue.Empty:
                if self.closing.is_set():
                    return
                continue
            try:
                started = time.monotonic()
                self.recorder.append(*sample)
                self.max_write_s = max(self.max_write_s, time.monotonic() - started)
            except BaseException as exc:
                self.error = exc
                return
            finally:
                self.queue.task_done()

    def check(self):
        if self.error is not None:
            raise RuntimeError("Recording disk writer failed: " + repr(self.error))
        if not self.thread.is_alive() and not self.closing.is_set():
            raise RuntimeError("Recording disk writer stopped unexpectedly")

    def append(self, frame, command, requested, previous, active, missed):
        self.check()
        # Camera capture publishes new arrays, never mutates old ones. Keep this
        # frame reference and its original timestamps together until persisted.
        sample = (dict(frame), dict(command), tuple(requested), tuple(previous), active, missed)
        try:
            self.queue.put_nowait(sample)
        except queue.Full:
            raise RuntimeError("Recording queue full: storage cannot keep up; stopping without dropping labels")

    def finish(self, status, reason, events, final_frame=None):
        self.closing.set()
        self.thread.join(timeout=10.)
        if self.thread.is_alive():
            raise RuntimeError("Disk writer did not finish within 10 seconds; episode remains incomplete")
        if self.error is not None:
            status, reason = "aborted", "disk_writer_failed: " + repr(self.error)
        self.meta["disk_writer"]["max_write_s"] = self.max_write_s
        self.recorder.finish(status, reason, events, final_frame)
        self.check()


class Recorder:
    def __init__(self, root, cfg, synthetic, controller, scenario, source="human"):
        self.path = Path(root) / ("episode_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
        (self.path / "frames").mkdir(parents=True, exist_ok=False)
        self.meta = {"schema_version": 1, "episode_id": self.path.name, "created_unix_s": time.time(),
                     "status": "recording", "accepted": False, "outcome": "unreviewed", "terminal_reward": None,
                     "reward_source": None, "synthetic": synthetic, "source": source, "controller": controller,
                     "scenario": scenario, "config": cfg, "frame_color": "RGB", "frame_format": "JPEG quality=95",
                     "timestamp_semantics": "host frame delivery; no sensor exposure timestamp",
                     "action_semantics": "jetracer property readback; no speed/angle feedback",
                     "notes": "", "samples": 0}
        self.meta["runtime"] = {"python": platform.python_version(), "platform": platform.platform()}
        save_json(self.path / "metadata.json", self.meta)
        self.file = (self.path / "actions.csv").open("w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=FIELDS)
        self.writer.writeheader()
        self.first = None
        self.last = None
        self.step = 0

    def append(self, frame, command, requested, previous, active, missed):
        stamp = command["command_start_monotonic_s"]
        if self.first is None:
            self.first = stamp
        filename = "frames/{:06d}.jpg".format(self.step)
        dest = self.path / filename
        temp = dest.with_suffix(".tmp")
        Image.fromarray(frame["rgb"]).save(str(temp), format="JPEG", quality=95)
        # Persist image before making the CSV row visible.
        with temp.open("rb") as image_file:
            os.fsync(image_file.fileno())
        temp.replace(dest)
        prev_norm = normalized(previous, self.meta["config"]["limits"])
        row = dict(episode_id=self.path.name, step=self.step, frame=filename, frame_id=frame["frame_id"],
                   rgb_sha256=hashlib.sha256(frame["rgb"].tobytes()).hexdigest(),
                   elapsed_s=stamp - self.first, dt_s=0. if self.last is None else stamp - self.last,
                   frame_age_s=stamp - frame["frame_monotonic_s"], missed_periods=missed,
                   requested_steering_norm=requested[0], requested_throttle_norm=requested[1],
                   steering=command["action"][0], throttle=command["action"][1],
                   steering_norm=command["normalized"][0], throttle_norm=command["normalized"][1],
                   prev_steering=previous[0], prev_throttle=previous[1],
                   prev_steering_norm=prev_norm[0], prev_throttle_norm=prev_norm[1], deadman_active=int(active))
        for key in ("frame_unix_s", "frame_monotonic_s"):
            row[key] = frame[key]
        for key in ("input_monotonic_s", "command_unix_s", "command_start_monotonic_s", "command_end_monotonic_s"):
            row[key] = command[key]
        self.writer.writerow(row)  # missing rewards/poses stay empty, never fabricated zeroes
        self.file.flush()
        os.fsync(self.file.fileno())
        self.last = stamp
        self.step += 1

    def finish(self, status, reason, events, final_frame=None):
        self.file.close()
        self.meta.update(status=status, end_reason=reason, samples=self.step, ended_unix_s=time.time(), events=events)
        if final_frame is not None:
            Image.fromarray(final_frame["rgb"]).save(str(self.path / "final.jpg"), quality=95)
            self.meta["final_observation"] = {k: v for k, v in final_frame.items() if k != "rgb"}
            self.meta["final_observation"]["frame"] = "final.jpg"
        save_json(self.path / "metadata.json", self.meta)


def validate_episode(path):
    path = Path(path)
    errors, warnings = [], []
    try:
        meta = json.loads((path / "metadata.json").read_text())
        with (path / "actions.csv").open() as f:
            reader = csv.DictReader(f)
            if set(FIELDS) - set(reader.fieldnames or []):
                raise ValueError("Missing CSV columns")
            rows = list(reader)
        if not rows:
            errors.append("No samples")
        if meta["status"] != "completed":
            errors.append("Episode was not completed: " + meta["status"])
        if meta["samples"] != len(rows):
            errors.append("Metadata sample count differs from CSV")
        if any(e["event"] == "watchdog_timeout" for e in meta.get("events", [])):
            errors.append("Watchdog interrupted action history")
        previous, last_stamp, last_id = (0., 0.), None, -1
        repeated = 0
        previous_hash = None
        for i, row in enumerate(rows):
            def val(k):
                value = float(row[k])
                if not math.isfinite(value):
                    raise ValueError("Non-finite " + k)
                return value
            if int(row["step"]) != i or row["episode_id"] != path.name:
                errors.append("Step/episode identity mismatch at {}".format(i))
            image_path = (path / row["frame"]).resolve()
            if path.resolve() not in image_path.parents:
                raise ValueError("Frame path escapes episode")
            with Image.open(str(image_path)) as im:
                im.verify()
            stamp = val("command_start_monotonic_s")
            frame_stamp = val("frame_monotonic_s")
            if frame_stamp > stamp or val("command_end_monotonic_s") < stamp:
                errors.append("Observation/action ordering error at {}".format(i))
            if not frame_stamp <= val("input_monotonic_s") <= stamp:
                errors.append("Input timestamp ordering error")
            if abs(val("frame_age_s") - (stamp - frame_stamp)) > 1e-6:
                errors.append("Frame age mismatch")
            if last_stamp is not None and stamp <= last_stamp:
                errors.append("Non-increasing command time")
            if int(row["frame_id"]) <= last_id:
                errors.append("Repeated/non-increasing frame ID")
            repeated = repeated + 1 if row["rgb_sha256"] == previous_hash else 0
            if repeated == 10:
                warnings.append("11 identical raw frames: inspect for a frozen camera or a stationary scene")
            previous_hash = row["rgb_sha256"]
            if abs(val("dt_s") - (0 if last_stamp is None else stamp - last_stamp)) > 1e-6:
                errors.append("dt mismatch")
            if val("frame_age_s") > meta["config"]["control"]["max_frame_age_s"]:
                errors.append("Stale frame")
            action = (val("steering"), val("throttle"))
            norm = normalized(action, meta["config"]["limits"])
            for j, key in enumerate(("steering", "throttle")):
                if abs(val("prev_" + key) - previous[j]) > 1e-6:
                    errors.append("Broken previous-action chain at {}".format(i))
                if abs(val(key + "_norm") - norm[j]) > 1e-6 or abs(norm[j]) > 1.000001:
                    errors.append("Action scaling mismatch")
                if abs(val("prev_" + key + "_norm") - normalized(previous, meta["config"]["limits"])[j]) > 1e-6:
                    errors.append("Previous-action normalization mismatch")
            if val("dt_s") > 1.5 / meta["config"]["control"]["hz"]:
                warnings.append("Slow control interval at step {}".format(i))
            previous, last_stamp, last_id = action, stamp, int(row["frame_id"])
        if len(list((path / "frames").glob("*.jpg"))) != len(rows):
            errors.append("Orphan or missing image files")
    except Exception as exc:
        errors.append(str(exc))
    return {"episode": str(path), "valid": not errors, "errors": errors, "warnings": warnings}


def review_episode(path, outcome, accepted, score=None, notes=""):
    path = Path(path)
    if outcome not in ("success", "near", "neutral", "poor", "collision", "recovery", "discard"):
        raise ValueError("Unknown outcome")
    result = validate_episode(path)
    if accepted and (not result["valid"] or outcome in ("collision", "discard")):
        raise ValueError("Cannot accept invalid/collision/discard episode: " + str(result["errors"]))
    if score is not None and (not math.isfinite(score) or not -1 <= score <= 1):
        raise ValueError("Human terminal score must be between -1 and 1")
    meta = json.loads((path / "metadata.json").read_text())
    meta.update(outcome=outcome, accepted=accepted, terminal_reward=score,
                reward_source="human_terminal" if score is not None else None, notes=notes)
    save_json(path / "metadata.json", meta)
    return meta
