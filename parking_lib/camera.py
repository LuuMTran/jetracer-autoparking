"""Latest-frame capture. Timestamps describe host delivery, not sensor exposure."""
import atexit
import os
import threading
import time
import numpy as np
from .core import canonical_config


# Some JetPack 4 OpenCV/GStreamer builds corrupt the allocator when a CSI
# VideoCapture is explicitly destroyed after TensorRT has run in the process.
# PPO can retain that native object until os._exit lets the kernel reclaim it.
_PROCESS_LIFETIME_DEVICES = []


def camera_kwargs(cfg):
    """Build JetCam arguments, with optional source-resolution overrides."""
    cfg = canonical_config(cfg)
    result = dict(cfg["camera"])
    source_width = os.environ.get("JETRACER_CAPTURE_WIDTH")
    source_height = os.environ.get("JETRACER_CAPTURE_HEIGHT")
    source_fps = os.environ.get("JETRACER_CAPTURE_FPS")
    if bool(source_width) != bool(source_height):
        raise ValueError("Set both JETRACER_CAPTURE_WIDTH and JETRACER_CAPTURE_HEIGHT")
    if source_width:
        width, height = int(source_width), int(source_height)
        if width < result["width"] or height < result["height"]:
            raise ValueError("Camera source dimensions cannot be smaller than output dimensions")
        result["capture_width"], result["capture_height"] = width, height
    if source_fps:
        fps = int(source_fps)
        if not 1 <= fps <= 240:
            raise ValueError("JETRACER_CAPTURE_FPS must be from 1 to 240")
        result["capture_fps"] = fps
    return cfg, result


class Camera:
    def __init__(self, cfg, real=False, threaded=True, release_on_close=True):
        self.cfg, device_kwargs = camera_kwargs(cfg)
        self.real = real
        self.threaded = bool(threaded)
        self.release_on_close = bool(release_on_close)
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.latest = None
        self.error = None
        self.device = None
        self.release_callback = None
        self.closed = False
        self.counter = 0
        if real:
            from jetcam.csi_camera import CSICamera

            class LatestCSI(CSICamera):
                def _gst_str(self):
                    return super(LatestCSI, self)._gst_str() + " max-buffers=1 drop=true sync=false"

            if "capture_width" in device_kwargs:
                print("Camera source {}x{}@{} -> output {}x{}".format(
                    device_kwargs["capture_width"], device_kwargs["capture_height"],
                    device_kwargs["capture_fps"], device_kwargs["width"],
                    device_kwargs["height"]), flush=True)
            self.device = LatestCSI(**device_kwargs)
            self.release_callback = self.device.cap.release
            # JetCam registers cap.release with atexit. This wrapper owns the
            # lifecycle, so remove that callback now and release exactly once.
            atexit.unregister(self.release_callback)
        self.thread = None
        if self.threaded:
            self.thread = threading.Thread(target=self._capture, daemon=True)
            self.thread.start()

    def _next_frame(self):
        if self.real:
            rgb = self.device.read()[:, :, ::-1].copy()  # JetCam BGR -> RGB
        else:
            c = self.cfg["camera"]
            rgb = np.zeros((c["height"], c["width"], 3), dtype=np.uint8)
            rgb[:, :, 0] = self.counter % 256
            rgb[:, (self.counter * 9) % c["width"]:, 1] = 100
        frame = {"rgb": rgb, "frame_id": self.counter,
                 "frame_monotonic_s": time.monotonic(), "frame_unix_s": time.time()}
        self.counter += 1
        return frame

    def _capture(self):
        try:
            while not self.done.is_set():
                frame = self._next_frame()
                with self.lock:
                    self.latest = frame
                if not self.real:
                    self.done.wait(1. / self.cfg["camera"]["capture_fps"])
        except Exception as exc:
            self.error = repr(exc)

    def read(self, after=-1, timeout=2.):
        if self.closed:
            raise RuntimeError("Camera is closed")
        if not self.threaded:
            try:
                frame = self._next_frame()
            except Exception as exc:
                raise RuntimeError("Camera failed: " + repr(exc))
            return frame
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.error:
                raise RuntimeError("Camera failed: " + self.error)
            with self.lock:
                frame = self.latest
            if frame is not None and frame["frame_id"] > after:
                if time.monotonic() - frame["frame_monotonic_s"] > self.cfg["control"]["max_frame_age_s"]:
                    raise RuntimeError("Stale camera frame")
                return dict(frame)
            time.sleep(0.002)
        raise RuntimeError("Camera did not deliver a new frame")

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.done.set()
        if self.thread is not None:
            self.thread.join(timeout=2.)
        thread_stopped = self.thread is None or not self.thread.is_alive()
        if self.release_callback is not None and thread_stopped:
            if self.release_on_close:
                self.release_callback()
            else:
                # Keep both the JetCam object and bound release callback alive;
                # 05_online_ppo.py exits with os._exit after other cleanup.
                _PROCESS_LIFETIME_DEVICES.append((self.device, self.release_callback))
            self.release_callback = None
        # A blocked driver read must not block motor shutdown; process exit releases it.

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
