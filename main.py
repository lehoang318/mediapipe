#!/usr/bin/env python3
"""Face framework entry point.

    python main.py --video PATH     # fps from the video, config fallback
    python main.py --camera INDEX   # fps from config.json (priority over --video)
    python main.py --video PATH --far   # use the far-range (full-range) detector

Reads frames, detects the biggest face with score >= face.min_score, extracts
the 478 landmarks, and displays annotated frames (eyes/irises/mouth).

The near-range detector (blaze_face_short_range) is used by default; --far
switches to the full-range detector (blaze_face_full_range), which is better
for faces beyond ~2 m.
"""

from __future__ import annotations

import os

# opencv-python's Qt GUI ships no fonts; point it at the system ones. This
# must be set before cv2 is imported.
if "QT_QPA_FONTDIR" not in os.environ and os.path.isdir("/usr/share/fonts"):
    os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"

import argparse
from pathlib import Path

from faceframework.app import run
from faceframework.config import Config
from faceframework.model_store import ensure_models
from faceframework.pipeline import BiggestFacePipeline
from faceframework.sources import open_source


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", metavar="PATH", help="input video file")
    parser.add_argument("--camera", type=int, metavar="INDEX",
                        help="camera device index (takes priority over --video)")
    parser.add_argument("--config", type=Path, default=None, metavar="PATH",
                        help="json config file (default: config.json at repo root)")
    parser.add_argument("--far", action="store_true",
                        help="use the far-range (full-range) detector for faces "
                             "beyond ~2 m instead of the default near-range one")
    args = parser.parse_args(argv)
    if args.video is None and args.camera is None:
        parser.error("one of --video or --camera is required")
    if args.video is not None and args.camera is not None:
        print("[framework] both --video and --camera given; using --camera (priority)")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = Config.load(args.config)
    if cfg.loaded_from:
        print(f"[framework] config: {cfg.loaded_from}")

    detectors, landmarker = ensure_models(cfg.pretrained_path())
    variant = "far" if args.far else "near"
    detector = detectors[variant]
    print(f"[framework] detector: {variant}-range ({detector.name})")
    pipeline = BiggestFacePipeline(detector, landmarker,
                                   min_score=cfg.face.min_score,
                                   nms_iou=cfg.face.nms_iou)
    source = open_source(args.video, args.camera, cfg)
    run(source, pipeline, cfg)


if __name__ == "__main__":
    main()
