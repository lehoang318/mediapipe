"""Frame sources (video file or camera) with fps handling.

* ``--video``: fps is read from the container; if it is missing or absurd the
  configurable ``video.fallback_fps`` is used.
* ``--camera``: fps comes from ``camera.fps`` in the config; the value is
  pushed to the device (best effort) and also used to pace the display.
"""

from __future__ import annotations

import math

import cv2


class FrameSource:
    def read(self):
        raise NotImplementedError

    @property
    def fps(self) -> float:
        raise NotImplementedError

    @property
    def delay_ms(self) -> int:
        return max(1, round(1000.0 / self.fps)) if self.fps else 1

    def describe(self) -> str:
        return ""

    def release(self) -> None:
        pass


class VideoFileSource(FrameSource):
    def __init__(self, path: str, fallback_fps: float = 30.0):
        self.path = str(path)
        self.fallback_fps = float(fallback_fps)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise SystemExit(f"error: cannot open video {self.path}")
        self._fps, self.used_fallback = _deduce_fps(self.cap, self.fallback_fps)

    @property
    def fps(self) -> float:
        return self._fps

    def read(self):
        return self.cap.read()

    def describe(self) -> str:
        fallback = f" (fallback {self.fallback_fps:g})" if self.used_fallback else ""
        return f"video {self.path}  fps {self.fps:g}{fallback}"

    def release(self) -> None:
        self.cap.release()


class CameraSource(FrameSource):
    def __init__(self, index: int, fps: float = 30.0):
        self.index = int(index)
        self._fps = float(fps)
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            raise SystemExit(f"error: cannot open camera {self.index}")
        self.cap.set(cv2.CAP_PROP_FPS, self._fps)
        self.device_fps = self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def fps(self) -> float:
        return self._fps

    def read(self):
        return self.cap.read()

    def describe(self) -> str:
        device = f"  device reports {self.device_fps:g} fps" if self.device_fps else ""
        return f"camera {self.index}  fps {self.fps:g} (config){device}"

    def release(self) -> None:
        self.cap.release()


def _deduce_fps(cap: cv2.VideoCapture, fallback: float):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or not math.isfinite(fps) or not 1.0 <= fps <= 1000.0:
        return float(fallback), True
    return float(fps), False


def open_source(video, camera, cfg) -> FrameSource:
    """Camera takes priority over video when both are supplied."""
    if camera is not None:
        return CameraSource(camera, cfg.camera.fps)
    return VideoFileSource(video, cfg.video.fallback_fps)
