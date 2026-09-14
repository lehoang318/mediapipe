"""Configuration loading for the face framework.

Reads ``config.json`` (repo root by default) and merges it over built-in
defaults, so a missing file or missing key never breaks a run.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

from . import REPO_ROOT

DEFAULTS = {
    "pretrained_dir": "pretrained",
    "video": {"fallback_fps": 30.0},
    "camera": {"fps": 30.0},
    "face": {"min_score": 0.7, "nms_iou": 0.5},
    "display": {"window": "face_framework (q quit | space pause)"},
}


@dataclass
class VideoConfig:
    fallback_fps: float = 30.0


@dataclass
class CameraConfig:
    fps: float = 30.0


@dataclass
class FaceConfig:
    min_score: float = 0.7
    nms_iou: float = 0.5


@dataclass
class DisplayConfig:
    window: str = "face_framework (q quit | space pause)"


@dataclass
class Config:
    pretrained_dir: str = "pretrained"
    video: VideoConfig = field(default_factory=VideoConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    loaded_from: Path | None = None

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        resolved = Path(path) if path else REPO_ROOT / "config.json"
        raw = json.loads(resolved.read_text()) if resolved.exists() else {}
        merged = _merge(copy.deepcopy(DEFAULTS), raw)
        return cls(
            pretrained_dir=str(merged["pretrained_dir"]),
            video=_build(VideoConfig, merged["video"]),
            camera=_build(CameraConfig, merged["camera"]),
            face=_build(FaceConfig, merged["face"]),
            display=_build(DisplayConfig, merged["display"]),
            loaded_from=resolved if resolved.exists() else None,
        )

    def pretrained_path(self) -> Path:
        p = Path(self.pretrained_dir)
        return p if p.is_absolute() else REPO_ROOT / p


def _merge(base: dict, override: dict) -> dict:
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _merge(base[key], value)
        else:
            base[key] = value
    return base


def _build(cls, values):
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in (values or {}).items() if k in names})
