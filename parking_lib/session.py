"""One camera-observe / controller-decide / motor-write / record loop."""
import time
import threading
from .camera import Camera
from .controls import Controls
from .core import Car, stop_signals
from .data import AsyncRecorder


def bounded_policy_action(raw_predicted, policy_caps, throttle_scale=1.):
    """Scale throttle in normalized action space, then apply final safety caps."""
    adjusted = (float(raw_predicted[0]), float(raw_predicted[1]) * float(throttle_scale))
    capped = tuple(max(-policy_caps[i], min(policy_caps[i], adjusted[i])) for i in range(2))
    return adjusted, capped


def run_session(cfg, real=False, controller="synthetic", seconds=5, dataset=None,
                scenario="normal", policy=None, stop_event=None, controls=None,
                policy_caps=(1., 1.), policy_warmup_s=.5, policy_throttle_scale=1.,
                policy_stop_threshold=.8, policy_stop_frames=4, policy_stop_min_s=1.):
    event = stop_event if stop_event is not None else threading.Event()
    with stop_signals(event):
        return _run_session(cfg, real, controller, seconds, dataset, scenario, policy, event, controls,
                            policy_caps, policy_warmup_s, policy_throttle_scale,
                            policy_stop_threshold, policy_stop_frames, policy_stop_min_s)


def _run_session(cfg, real, controller, seconds, dataset, scenario, policy, stop_event, controls,
                 policy_caps=(1., 1.), policy_warmup_s=.5, policy_throttle_scale=1.,
                 policy_stop_threshold=.8, policy_stop_frames=4, policy_stop_min_s=1.):
    if real and controller == "synthetic" and controls is None:
        raise ValueError("Synthetic inputs may never drive real hardware")
    if not 0 < seconds <= cfg["control"]["max_episode_s"]:
        raise ValueError("Duration exceeds config max_episode_s")
    if len(policy_caps) != 2 or any(not 0 <= float(value) <= 1 for value in policy_caps):
        raise ValueError("Policy steering/throttle caps must be between 0 and 1")
    if not 0 <= float(policy_warmup_s) <= 5:
        raise ValueError("Policy warm-up must be between 0 and 5 seconds")
    if not 0 < float(policy_throttle_scale) <= 3:
        raise ValueError("Policy throttle scale must be greater than 0 and at most 3")
    if not 0 < float(policy_stop_threshold) < 1:
        raise ValueError("Policy stop threshold must be between 0 and 1")
    if not 1 <= int(policy_stop_frames) <= 30:
        raise ValueError("Policy stop frames must be between 1 and 30")
    if not 0 <= float(policy_stop_min_s) <= cfg["control"]["max_episode_s"]:
        raise ValueError("Policy stop minimum time exceeds the configured episode limit")
    recorder = None
    car = None
    status, reason, final = "aborted", "initialization", None
    last_id = -1
    # Construct controls/camera before the motor lease starts.
    with (controls or Controls(cfg, controller)) as human, Camera(cfg, real) as camera:
        initial_frame = camera.read()
        if policy:
            # Warm backend kernels before creating the motor-command lease.
            for _ in range(3):
                policy.reset()
                policy(initial_frame["rgb"], (0., 0.))
            policy.reset()
        try:
            if dataset is not None:
                recorder = AsyncRecorder(dataset, cfg, not real, controller, scenario,
                                    source="policy" if policy else "human")
                if policy:
                    recorder.meta["policy_safety_caps_norm"] = {
                        "steering": float(policy_caps[0]), "throttle": float(policy_caps[1])}
                    recorder.meta["policy_warmup_s"] = float(policy_warmup_s)
                    recorder.meta["policy_throttle_scale"] = float(policy_throttle_scale)
                    recorder.meta["policy_stop"] = {
                        "enabled": bool(getattr(policy, "stop_head", None) is not None),
                        "threshold": float(policy_stop_threshold),
                        "consecutive_frames": int(policy_stop_frames),
                        "minimum_elapsed_s": float(policy_stop_min_s)}
                print("Recording:", recorder.path, flush=True)
            with Car(cfg, real) as car:
                if recorder:
                    recorder.meta["driver_runtime"] = car.describe()
                started = time.monotonic()
                deadline = started
                previous = (0., 0.)
                period = 1. / cfg["control"]["hz"]
                step = 0
                stop_streak = 0
                stop_latched = False
                if policy:
                    policy.reset()
                while time.monotonic() - started < seconds:
                    if recorder:
                        recorder.check()
                    if stop_event is not None and stop_event.is_set():
                        reason = getattr(stop_event, "reason", "external_stop")
                        break
                    frame = camera.read(after=last_id, timeout=cfg["control"]["max_frame_age_s"])
                    last_id = frame["frame_id"]
                    action = human.poll()
                    input_stamp = time.monotonic()
                    if human.stop:
                        reason = "emergency_stop"
                        break
                    if human.finish:
                        status, reason = "completed", "operator_finished"
                        break
                    if policy:
                        # Warm the GRU with stationary observations. Reset on deadman release.
                        inference_started = time.monotonic()
                        raw_predicted = policy(frame["rgb"], previous)
                        inference_s = time.monotonic() - inference_started
                        adjusted, predicted = bounded_policy_action(
                            raw_predicted, policy_caps, policy_throttle_scale)
                        elapsed = time.monotonic() - started
                        if elapsed < policy_warmup_s:
                            action, policy_gate = (0., 0.), "warmup"
                            stop_streak = 0
                        elif not human.active:
                            action, policy_gate = (0., 0.), "deadman_released"
                            stop_streak = 0
                        else:
                            action, policy_gate = predicted, "enabled"
                            stop_probability = getattr(policy, "last_stop_probability", None)
                            if stop_probability is not None and elapsed >= policy_stop_min_s:
                                stop_streak = stop_streak + 1 if stop_probability >= policy_stop_threshold else 0
                                if stop_streak >= policy_stop_frames:
                                    action, policy_gate = (0., 0.), "policy_stop"
                                    stop_latched = True
                        if not human.active:
                            policy.reset()
                    frame_age = time.monotonic() - frame["frame_monotonic_s"]
                    if frame_age > cfg["control"]["max_frame_age_s"]:
                        detail = " age={:.3f}s limit={:.3f}s".format(
                            frame_age, cfg["control"]["max_frame_age_s"])
                        if policy:
                            detail += " inference={:.3f}s".format(inference_s)
                        raise RuntimeError("Frame too old after decision/inference:" + detail)
                    if stop_event is not None and stop_event.is_set():
                        reason = getattr(stop_event, "reason", "external_stop")
                        break
                    command = car.apply(action)
                    command["input_monotonic_s"] = input_stamp
                    missed = max(0, int((command["command_start_monotonic_s"] - deadline) / period))
                    if recorder:
                        recorder.append(frame, command, action, previous, human.active, missed)
                    previous = command["action"]
                    if step % max(1, int(cfg["control"]["hz"])) == 0:
                        timing = " inference={:.0f}ms".format(inference_s * 1000.) if policy else ""
                        network = (" raw_net=({:+.3f},{:+.3f}) throttle_x{:.2f}={:+.3f}"
                                   " capped=({:+.3f},{:+.3f}) gate={}".format(
                            raw_predicted[0], raw_predicted[1], policy_throttle_scale, adjusted[1],
                            predicted[0], predicted[1], policy_gate)) if policy else ""
                        if policy and getattr(policy, "last_stop_probability", None) is not None:
                            network += " stop_p={:.3f} stop_streak={}".format(
                                policy.last_stop_probability, stop_streak)
                        print(("t={:.1f}s step={} applied=({:+.3f},{:+.3f})" + network + timing).format(
                            time.monotonic() - started, step, *previous), flush=True)
                    if stop_latched:
                        print("Policy stop latched: probability={:.3f} for {} consecutive frames".format(
                            policy.last_stop_probability, stop_streak), flush=True)
                        status, reason = "completed", "policy_stop"
                        break
                    step += 1
                    deadline += period
                    if deadline < time.monotonic():
                        deadline = time.monotonic()  # no bursts of catch-up motor commands
                    delay = max(0, deadline - time.monotonic())
                    if stop_event is None:
                        time.sleep(delay)
                    else:
                        stop_event.wait(delay)
                else:
                    status, reason = "completed", "duration_limit"
                car.stop(reason)
                try:
                    final = camera.read(after=last_id, timeout=0.3)
                except RuntimeError:
                    pass
        except KeyboardInterrupt:
            status = "aborted"
            reason = "keyboard_interrupt"
        except BaseException as exc:
            status = "aborted"
            reason = repr(exc)
            raise
        finally:
            if recorder:
                recorder.finish(status, reason, car.events if car else [], final)
    print("Stopped:", reason, flush=True)
    return recorder.path if recorder else None
