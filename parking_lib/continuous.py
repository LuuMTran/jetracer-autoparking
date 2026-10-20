"""Repeated human demonstrations with one camera and motor-driver lifetime."""
import os
import select
import sys
import threading
import time
from .camera import Camera
from .controls import Controls, KEYBOARD_ACTIONS
from .core import Car, stop_signals
from .data import AsyncRecorder


class ContinuousKeyboard(Controls):
    """Lesson 04 input: Enter toggles recording; Escape stops; F finishes and exits."""
    def __init__(self, cfg):
        super(ContinuousKeyboard, self).__init__(cfg, "keyboard")
        self.boundary = False
        self.last_boundary = float("-inf")

    def reset_input(self):
        # Never carry a held movement or buffered Enter across a recording boundary.
        self.action, self.last_key, self.active = (0., 0.), 0., False
        self.boundary = False
        self.last_boundary = time.monotonic()
        import termios
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)

    def poll(self):
        self.boundary = False
        if select.select([sys.stdin], [], [], 0)[0]:
            # Read bytes directly: TextIO buffering can hide already-buffered keys
            # from select(), including Enter after several movement keys.
            keys = os.read(sys.stdin.fileno(), 4096).decode("ascii", errors="ignore").lower()
            if not keys:
                self.stop = True
            for key in keys:
                if key == "\x1b":
                    self.stop = True
                elif key == "f":
                    self.finish = True
                elif key in ("\n", "\r"):
                    if time.monotonic() - self.last_boundary >= .35:
                        self.boundary = True
                elif key == " ":
                    self.action, self.last_key = (0., 0.), 0.
                elif key in KEYBOARD_ACTIONS:
                    self.action, self.last_key = KEYBOARD_ACTIONS[key], time.monotonic()
        if self.boundary or self.stop or self.finish:
            self.action, self.last_key = (0., 0.), 0.
        self.active = time.monotonic() - self.last_key < .25
        return (self.action[0] * self.cfg["hardware"]["right_steering_sign"], self.action[1]) if self.active else (0., 0.)


def _final_frame(camera, last_id, stopped_at):
    deadline = time.monotonic() + .3
    while time.monotonic() < deadline:
        try:
            frame = camera.read(after=last_id, timeout=max(.001, deadline - time.monotonic()))
        except RuntimeError:
            return None
        if frame["frame_monotonic_s"] >= stopped_at:
            return frame
        last_id = frame["frame_id"]
    return None


def run_continuous(cfg, real=False, dataset="dataset", seconds=60, scenario="normal",
                   demo_episodes=0, stop_event=None, controls=None):
    """Enter starts/ends episodes; idle repositioning is never recorded.

    demo_episodes > 0 automatically records a finite number of synthetic episodes
    for software checks. It is forbidden with real hardware.
    """
    if not 0 < seconds <= cfg["control"]["max_episode_s"]:
        raise ValueError("Duration exceeds config max_episode_s")
    if demo_episodes < 0 or (demo_episodes and real):
        raise ValueError("Automatic demo episodes are allowed only without real hardware")
    event = stop_event if stop_event is not None else threading.Event()
    if controls is None:
        controls = Controls(cfg, "synthetic") if demo_episodes else ContinuousKeyboard(cfg)
    controller = "synthetic" if demo_episodes else "keyboard"
    paths = []
    with stop_signals(event), controls as human, Camera(cfg, real) as camera:
        camera.read()
        with Car(cfg, real) as car:
            recorder = None
            event_start = 0
            last_id = -1
            status, reason = "aborted", "session_exit"
            period = 1. / cfg["control"]["hz"]
            car.park("ready")

            def reset_input():
                if hasattr(human, "reset_input"):
                    human.reset_input()

            reset_input()
            print("READY: motors neutral. Reposition, then press Enter to record. F exits; Escape stops.", flush=True)
            try:
                while not event.is_set():
                    if recorder:
                        recorder.check()
                    frame = camera.read(after=last_id, timeout=cfg["control"]["max_frame_age_s"])
                    last_id = frame["frame_id"]
                    action = human.poll()
                    input_stamp = time.monotonic()
                    if human.stop:
                        status, reason = "aborted", "emergency_stop"
                        break
                    if human.finish:
                        status, reason = "completed", "operator_finished"
                        break
                    if event.is_set():
                        break
                    boundary = getattr(human, "boundary", False)
                    expired = recorder is not None and time.monotonic() - started >= seconds
                    if recorder is not None and (boundary or expired):
                        reason = "operator_finished" if boundary else "duration_limit"
                        car.park(reason)
                        final = _final_frame(camera, last_id, car.events[-1]["monotonic_s"])
                        writer, recorder = recorder, None
                        writer.finish("completed", reason, car.events[event_start:], final)
                        paths.append(writer.path)
                        reset_input()
                        print("SAVED: {} (unreviewed)".format(writer.path), flush=True)
                        if demo_episodes and len(paths) >= demo_episodes:
                            break
                        print("READY: reposition the stopped car; Enter starts next episode. F exits.", flush=True)
                        continue
                    if recorder is None:
                        if boundary or demo_episodes:
                            # Car is already parked before codec initialization or disk I/O.
                            recorder = AsyncRecorder(dataset, cfg, not real, controller, scenario, source="human")
                            recorder.meta["driver_runtime"] = car.describe()
                            recorder.meta["collection_mode"] = "continuous"
                            event_start = len(car.events)
                            previous = (0., 0.)
                            started = time.monotonic()
                            deadline = started + period
                            step = 0
                            reset_input()
                            print("RECORDING: {} | Enter saves; F saves/exits; Escape aborts/exits".format(recorder.path), flush=True)
                        event.wait(period)
                        continue
                    if time.monotonic() - frame["frame_monotonic_s"] > cfg["control"]["max_frame_age_s"]:
                        raise RuntimeError("Frame too old before motor command")
                    command = car.apply(action)
                    command["input_monotonic_s"] = input_stamp
                    missed = max(0, int((command["command_start_monotonic_s"] - deadline) / period))
                    recorder.append(frame, command, action, previous, human.active, missed)
                    previous = command["action"]
                    if step % max(1, int(cfg["control"]["hz"])) == 0:
                        print("t={:.1f}s step={} steering={:+.3f} throttle={:+.3f}".format(
                            time.monotonic() - started, step, *previous), flush=True)
                    step += 1
                    deadline = max(deadline + period, time.monotonic())
                    event.wait(max(0, deadline - time.monotonic()))
                if event.is_set():
                    status, reason = "aborted", getattr(event, "reason", "external_stop")
            except KeyboardInterrupt:
                status, reason = "aborted", "keyboard_interrupt"
            except BaseException as exc:
                status, reason = "aborted", repr(exc)
                raise
            finally:
                try:
                    car.stop(reason)
                except BaseException as exc:
                    status, reason = "aborted", "stop_write_failed: " + repr(exc)
                    raise
                finally:
                    if recorder is not None:
                        if car.fault:
                            status, reason = "aborted", car.fault
                        final = _final_frame(camera, last_id, car.events[-1]["monotonic_s"])
                        recorder.finish(status, reason, car.events[event_start:], final)
                        paths.append(recorder.path)
                        print("SAVED: {} ({}, unreviewed)".format(recorder.path, status), flush=True)
    print("Session ended. {} new episode(s).".format(len(paths)), flush=True)
    return paths
