#!/usr/bin/env python3
"""Download MediaPipe face models from Google storage and unpack them.

Downloads (official MediaPipe model bucket, storage.googleapis.com):
  * blaze_face_short_range.tflite  - face detector (BlazeFace near range)
  * blaze_face_full_range.tflite   - face detector (BlazeFace far range)
  * face_landmarker.task           - FaceMesh V2 task bundle (a zip archive),
                                     from which face_landmarks_detector.tflite
                                     is extracted.

Usage:
    python -m tools.download_models [--test-image]

Outputs (in pretrained/):
    pretrained/blaze_face_short_range.tflite
    pretrained/blaze_face_full_range.tflite
    pretrained/face_landmarks_detector.tflite
    pretrained/portrait.jpg   (only with --test-image)
"""

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://storage.googleapis.com"

DETECTOR_URL = (
    f"{BASE}/mediapipe-models/face_detector/blaze_face_short_range/"
    "float16/1/blaze_face_short_range.tflite"
)
FULL_RANGE_DETECTOR_URL = (
    f"{BASE}/mediapipe-models/face_detector/blaze_face_full_range/"
    "float16/1/blaze_face_full_range.tflite"
)
TASK_URL = (
    f"{BASE}/mediapipe-models/face_landmarker/face_landmarker/"
    "float16/1/face_landmarker.task"
)
TEST_IMAGE_URL = f"{BASE}/mediapipe-assets/portrait.jpg"

MODELS_DIR = Path(__file__).resolve().parent.parent / "pretrained"
LANDMARKS_MEMBER = "face_landmarks_detector.tflite"


def fetch(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"[skip] {dest.name} already present")
        return dest
    print(f"[down] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.rename(dest)
    print(f"       -> {dest} ({dest.stat().st_size} bytes)")
    return dest


def extract_landmark_model(task_path: Path, dest: Path) -> Path:
    if dest.exists():
        print(f"[skip] {dest.name} already present")
        return dest
    with zipfile.ZipFile(task_path) as zf:
        names = zf.namelist()
        if LANDMARKS_MEMBER not in names:
            sys.exit(f"error: {LANDMARKS_MEMBER} not found in {task_path.name}; "
                     f"members: {names}")
        dest.write_bytes(zf.read(LANDMARKS_MEMBER))
    print(f"[extr] {LANDMARKS_MEMBER} -> {dest} ({dest.stat().st_size} bytes)")
    return dest


def download_all(models_dir: Path, test_image: bool = False) -> None:
    """Download the model files into `models_dir` (skipping files already there)."""
    models_dir.mkdir(parents=True, exist_ok=True)
    fetch(DETECTOR_URL, models_dir / "blaze_face_short_range.tflite")
    fetch(FULL_RANGE_DETECTOR_URL, models_dir / "blaze_face_full_range.tflite")
    task = fetch(TASK_URL, models_dir / "face_landmarker.task")
    extract_landmark_model(task, models_dir / LANDMARKS_MEMBER)
    if test_image:
        fetch(TEST_IMAGE_URL, models_dir / "portrait.jpg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR,
                        help="target directory (default: pretrained/)")
    parser.add_argument("--test-image", action="store_true",
                        help="also download a sample face image (portrait.jpg)")
    args = parser.parse_args()
    download_all(args.models_dir, args.test_image)


if __name__ == "__main__":
    main()
