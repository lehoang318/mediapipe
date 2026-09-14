"""Ensure the converted ONNX models are present in the pretrained directory.

All three models (near detector, far detector, landmarker) are bootstrapped
together: if any ONNX file is missing the official MediaPipe tflite models are
downloaded into the same directory and converted with tf2onnx (heavy imports
stay lazy: they are only touched when a conversion is actually needed).
"""

from __future__ import annotations

from pathlib import Path

DETECTOR_ONNX = "blaze_face_short_range.onnx"
FULL_RANGE_DETECTOR_ONNX = "blaze_face_full_range.onnx"
LANDMARKER_ONNX = "face_landmarks_detector.onnx"

DETECTORS = {"near": DETECTOR_ONNX, "far": FULL_RANGE_DETECTOR_ONNX}


def ensure_models(pretrained_dir: str | Path, opset: int = 16, log=print):
    """Return (detectors, landmarker) as paths, downloading + converting if needed.

    ``detectors`` maps "near" -> short-range detector ONNX and "far" ->
    full-range detector ONNX.
    """
    directory = Path(pretrained_dir)
    detectors = {name: directory / file for name, file in DETECTORS.items()}
    landmarker = directory / LANDMARKER_ONNX

    if all(p.exists() for p in (*detectors.values(), landmarker)):
        log("[models] using " + ", ".join(
            str(p) for p in (*detectors.values(), landmarker)))
        return detectors, landmarker

    log(f"[models] ONNX files missing under {directory}; downloading + converting ...")
    directory.mkdir(parents=True, exist_ok=True)

    try:
        from tools import convert_to_onnx, download_models
    except ImportError as exc:
        raise SystemExit(f"error: cannot import download/convert helpers: {exc}")

    try:
        download_models.download_all(directory)
    except Exception as exc:
        raise SystemExit(
            f"error: failed to download models into {directory}: {exc}\n"
            f"       (network access to storage.googleapis.com is required)"
        )

    for tflite_name, onnx_name in convert_to_onnx.PAIRS:
        onnx_path = directory / onnx_name
        if onnx_path.exists():
            log(f"[models] {onnx_name} already present, skipping conversion")
            continue
        tflite_path = directory / tflite_name
        try:
            convert_to_onnx.convert_one(tflite_path, onnx_path, opset)
        except Exception as exc:
            raise SystemExit(
                f"error: failed to convert {tflite_name} -> {onnx_name}: {exc}\n"
                f"       (install the bootstrap deps: pip install -r requirements-dev.txt)"
            )
    return detectors, landmarker
