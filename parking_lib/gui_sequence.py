"""Run the two parking policies while publishing a preview and video."""
import csv
import json
import math
import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2

from .camera import Camera
from .core import Car, canonical_config
from .learning import signature
from .ppo_loop import follow_episode
from .stop_tuning import load_sequence_policy


class RunStatus:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "status.json"
        self.data = {
            "state": "starting",
            "stage": "Loading policies",
            "message": "Preparing the car",
            "frame_count": 0,
            "started_unix_s": time.time(),
        }
        self.write()

    def write(self, **changes):
        self.data.update(changes)
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w") as handle:
            json.dump(self.data, handle, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.replace(self.path)


class StopKeys:
    """The small key interface expected by follow_episode."""
    def __init__(self, stop_event):
        self.stop_event = stop_event

    def poll_episode(self):
        return "escape" if self.stop_event.is_set() else None


class VideoRecorder:
    def __init__(self, run_dir, fps, status):
        self.run_dir = Path(run_dir)
        self.preview_path = self.run_dir / "preview.jpg"
        self.video_path = self.run_dir / "parking_run.mp4"
        self.fps = float(fps)
        self.status = status
        self.writer = None
        self.frame_count = 0
        self.last_preview_s = 0.
        self.telemetry_path = self.run_dir / "telemetry.csv"
        self.telemetry_file = self.telemetry_path.open("w", newline="")
        self.telemetry_writer = csv.DictWriter(self.telemetry_file, fieldnames=[
            "stage", "frame", "step", "elapsed_s", "steering", "throttle",
            "value", "stop_probability", "inference_ms", "frame_age_ms"])
        self.telemetry_writer.writeheader()

    def _open(self, rgb):
        height, width = rgb.shape[:2]
        for codec in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(
                str(self.video_path), cv2.VideoWriter_fourcc(*codec),
                self.fps, (width, height))
            if writer.isOpened():
                self.writer = writer
                return
            writer.release()
        raise RuntimeError("OpenCV could not create an MP4 video")

    def add(self, rgb, telemetry, stage):
        if self.writer is None:
            self._open(rgb)
        annotated = draw_overlay(rgb, telemetry, stage)
        bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        self.writer.write(bgr)
        self.frame_count += 1
        row = dict(telemetry)
        row.update(stage=stage, frame=self.frame_count)
        self.telemetry_writer.writerow(row)
        self.telemetry_file.flush()
        now = time.monotonic()
        if now - self.last_preview_s >= .2:
            ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not ok:
                raise RuntimeError("Could not encode the camera preview")
            temporary = self.preview_path.with_suffix(".tmp")
            temporary.write_bytes(encoded.tobytes())
            temporary.replace(self.preview_path)
            self.status.write(frame_count=self.frame_count, latest_frame_unix_s=time.time(),
                              telemetry=telemetry, stage=stage)
            self.last_preview_s = now

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.telemetry_file is not None:
            self.telemetry_file.close()
            self.telemetry_file = None

    def browser_video(self):
        """Use ffmpeg when available so the browser gets H.264/yuv420p."""
        if not self.video_path.exists() or self.video_path.stat().st_size == 0:
            return None
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            return self.video_path
        converted = self.run_dir / "parking_run_h264.mp4"
        command = [ffmpeg, "-y", "-loglevel", "error", "-i", str(self.video_path),
                   "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", str(converted)]
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return self.video_path
        if result.returncode == 0 and converted.exists() and converted.stat().st_size:
            return converted
        return self.video_path


def draw_overlay(rgb, telemetry, stage):
    """Draw stage and action meters into one RGB video frame."""
    image = rgb.copy()
    height, width = image.shape[:2]
    panel_top = max(0, height - 126)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, panel_top), (width, height), (8, 18, 35), -1)
    cv2.addWeighted(overlay, .82, image, .18, 0, image)

    font = cv2.FONT_HERSHEY_SIMPLEX
    steering = max(-1., min(1., float(telemetry.get("steering", 0.))))
    throttle = max(-1., min(1., float(telemetry.get("throttle", 0.))))
    stage_text = str(stage).upper()
    badge_color = (37, 99, 235) if "DIRECT" in stage_text else (225, 29, 72)
    cv2.rectangle(image, (18, 16), (218, 52), badge_color, -1)
    cv2.putText(image, stage_text, (30, 41), font, .62, (255, 255, 255), 2,
                cv2.LINE_AA)
    elapsed = float(telemetry.get("elapsed_s", 0.))
    step = int(telemetry.get("step", 0))
    timing = "{:04.1f}s  STEP {:03d}".format(elapsed, step)
    size = cv2.getTextSize(timing, font, .55, 1)[0]
    cv2.putText(image, timing, (width - size[0] - 18, 40), font, .55,
                (255, 255, 255), 1, cv2.LINE_AA)

    # Steering meter: zero is centred, cyan extends toward the commanded side.
    steer_x0, steer_x1 = 116, max(240, width - 154)
    steer_y = height - 55
    steer_mid = (steer_x0 + steer_x1) // 2
    cv2.putText(image, "STEERING", (18, steer_y + 6), font, .45,
                (203, 213, 225), 1, cv2.LINE_AA)
    cv2.rectangle(image, (steer_x0, steer_y - 11), (steer_x1, steer_y + 11),
                  (51, 65, 85), -1)
    steer_end = int(round(steer_mid + steering * (steer_x1 - steer_x0) / 2.))
    cv2.rectangle(image, (min(steer_mid, steer_end), steer_y - 9),
                  (max(steer_mid, steer_end), steer_y + 9), (34, 211, 238), -1)
    cv2.line(image, (steer_mid, steer_y - 15), (steer_mid, steer_y + 15),
             (255, 255, 255), 2)
    cv2.putText(image, "L", (steer_x0 - 18, steer_y + 6), font, .45,
                (148, 163, 184), 1, cv2.LINE_AA)
    cv2.putText(image, "R", (steer_x1 + 7, steer_y + 6), font, .45,
                (148, 163, 184), 1, cv2.LINE_AA)
    cv2.putText(image, "{:+.2f}".format(steering), (steer_mid - 30, steer_y - 24),
                font, .48, (255, 255, 255), 1, cv2.LINE_AA)

    # Throttle meter: green rises for forward, amber drops for reverse.
    throttle_x = width - 62
    throttle_top, throttle_bottom = panel_top + 18, height - 18
    throttle_mid = (throttle_top + throttle_bottom) // 2
    cv2.rectangle(image, (throttle_x - 13, throttle_top),
                  (throttle_x + 13, throttle_bottom), (51, 65, 85), -1)
    throttle_end = int(round(throttle_mid - throttle *
                             (throttle_bottom - throttle_top) / 2.))
    throttle_color = (34, 197, 94) if throttle >= 0 else (251, 146, 60)
    cv2.rectangle(image, (throttle_x - 11, min(throttle_mid, throttle_end)),
                  (throttle_x + 11, max(throttle_mid, throttle_end)),
                  throttle_color, -1)
    cv2.line(image, (throttle_x - 17, throttle_mid),
             (throttle_x + 17, throttle_mid), (255, 255, 255), 2)
    cv2.putText(image, "THROTTLE", (width - 145, panel_top + 16), font, .40,
                (203, 213, 225), 1, cv2.LINE_AA)
    cv2.putText(image, "{:+.2f}".format(throttle), (width - 145, panel_top + 43),
                font, .52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "F", (throttle_x + 22, throttle_top + 10), font, .4,
                (148, 163, 184), 1, cv2.LINE_AA)
    cv2.putText(image, "R", (throttle_x + 22, throttle_bottom), font, .4,
                (148, 163, 184), 1, cv2.LINE_AA)
    return image


def run_recorded_sequence(first_checkpoint, first_stop_head, second_checkpoint,
                          second_stop_head, run_dir, stop_event, device="cuda",
                          fp16_encoder=True, first_seconds=40., second_seconds=40.,
                          warmup_s=.5, max_frame_age_ms=500.,
                          second_throttle_scale=.6,
                          slot_classifier_checkpoint=None, slot_device="cpu"):
    """Classify the start, run direct, and run reverse only for a left slot."""
    if not 0. < float(second_throttle_scale) <= 1.:
        raise ValueError("Reverse throttle scale must be above 0 and no greater than 1")
    status = RunStatus(run_dir)
    status.write(stage="Loading direct policy", message="Loading model 1 of 2")
    first = load_sequence_policy(first_checkpoint, first_stop_head, device,
                                 fp16_encoder, max_frame_age_ms)
    status.write(stage="Loading reverse policy", message="Loading model 2 of 2")
    second = load_sequence_policy(second_checkpoint, second_stop_head, device,
                                  fp16_encoder, max_frame_age_ms)
    if signature(first["cfg"]) != signature(second["cfg"]):
        raise ValueError("The direct and reverse policies use different car settings")
    slot_predictor = None
    if slot_classifier_checkpoint is not None:
        status.write(stage="Loading route classifier",
                     message="Loading the three-frame slot classifier")
        from .slot_classifier import SlotPredictor
        slot_predictor = SlotPredictor(slot_classifier_checkpoint, slot_device)
    cfg = canonical_config(first["cfg"])
    hz = float(cfg["control"]["hz"])
    first_steps = max(1, int(round(float(first_seconds) * hz)))
    second_steps = max(1, int(round(float(second_seconds) * hz)))
    maximum = int(round(float(cfg["control"]["max_episode_s"]) * hz))
    if first_steps > maximum or second_steps > maximum:
        raise ValueError("A sequence stage exceeds the configured episode duration")
    if not math.isfinite(hz) or hz <= 0:
        raise ValueError("Invalid policy control rate")

    recorder = VideoRecorder(run_dir, hz, status)
    keys = StopKeys(stop_event)
    result = {"state": "stopped", "first": None, "second": None,
              "first_checkpoint": str(first["checkpoint"]),
              "second_checkpoint": str(second["checkpoint"]),
              "second_throttle_scale": float(second_throttle_scale),
              "slot_classifier": (str(slot_classifier_checkpoint)
                                  if slot_classifier_checkpoint is not None else None),
              "slot_decision": None}
    try:
        with Camera(cfg, True, threaded=False, release_on_close=False) as raw_camera, \
                Car(cfg, True) as car:
            car.park("gui_sequence_ready")
            if stop_event.is_set():
                car.park("gui_stop_before_start")
                return result

            route = {"left_empty": True, "left_probability": None,
                     "threshold": None}
            if slot_predictor is not None:
                from .slot_classifier import classify_start
                status.write(state="running", stage="Classifying parking slot",
                             message="Reading three stationary camera frames")
                empty_telemetry = {
                    "step": 0, "elapsed_s": 0., "steering": 0., "throttle": 0.,
                    "value": 0., "stop_probability": 0., "inference_ms": 0.,
                    "frame_age_ms": 0.}
                def classification_frame(frame, index):
                    recorder.add(frame["rgb"], empty_telemetry,
                                 "Slot frame {}/3".format(index))
                route = classify_start(raw_camera, slot_predictor, classification_frame)
                result["slot_decision"] = route
                route_name = "Direct + reverse" if route["left_empty"] else "Direct only"
                status.write(
                    state="running", stage="Route selected",
                    message="{} · left-slot probability {:.1%}".format(
                        route_name, route["left_probability"]),
                    slot_decision=route)

            status.write(state="running", stage="Direct parking",
                         message="Direct policy is controlling the car")
            def direct_telemetry(frame, telemetry):
                recorder.add(frame["rgb"], telemetry, "Direct parking")
            reason, exit_requested, steps = follow_episode(
                first["model"], cfg, raw_camera, car, keys, device, first["backend"],
                first["stop_head"], first["decision"], warmup_s, first_steps, 1.,
                direct_telemetry)
            result["first"] = {"reason": reason, "steps": int(steps)}
            if exit_requested or stop_event.is_set():
                result["state"] = "stopped"
                return result

            if not route["left_empty"]:
                car.park("gui_conditional_route_complete")
                result["state"] = "completed"
                return result

            status.write(state="running", stage="Reverse parking",
                         message="Left slot selected; reverse policy is controlling the car")
            def reverse_telemetry(frame, telemetry):
                recorder.add(frame["rgb"], telemetry, "Reverse parking")
            reason, exit_requested, steps = follow_episode(
                second["model"], cfg, raw_camera, car, keys, device, second["backend"],
                second["stop_head"], second["decision"], warmup_s, second_steps,
                second_throttle_scale, reverse_telemetry)
            result["second"] = {"reason": reason, "steps": int(steps)}
            result["state"] = "stopped" if exit_requested else "completed"
            return result
    finally:
        recorder.close()
        video = recorder.browser_video()
        result["frames"] = recorder.frame_count
        result["video"] = str(video) if video else None
        status.write(state=result["state"], stage="Finished",
                     message="Run completed" if result["state"] == "completed" else "Car stopped",
                     result=result, finished_unix_s=time.time(), frame_count=recorder.frame_count,
                     video=result["video"],
                     telemetry={"steering": 0., "throttle": 0.})
