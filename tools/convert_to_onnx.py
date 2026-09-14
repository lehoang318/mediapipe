#!/usr/bin/env python3
"""Convert MediaPipe face tflite models to ONNX using tf2onnx.

Converts (in pretrained/):
    blaze_face_short_range.tflite   -> blaze_face_short_range.onnx
    blaze_face_full_range.tflite    -> blaze_face_full_range.onnx
    face_landmarks_detector.tflite  -> face_landmarks_detector.onnx

Both models contain only standard TFLite ops, so tf2onnx handles them
directly. The raw graph is converted as-is; SSD anchor decoding, NMS and
landmark projection are implemented in Python by faceframework/face_pipeline.py.

Usage:
    python -m tools.convert_to_onnx [--models-dir pretrained] [--opset 16]
"""

import argparse
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "pretrained"

PAIRS = [
    ("blaze_face_short_range.tflite", "blaze_face_short_range.onnx"),
    ("blaze_face_full_range.tflite", "blaze_face_full_range.onnx"),
    ("face_landmarks_detector.tflite", "face_landmarks_detector.onnx"),
]


def tflite_io(tflite_path: Path):
    """Return (input_specs, output_specs) of a tflite model via the interpreter."""
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()

    def spec(d):
        return {"name": d["name"], "shape": list(d["shape"]),
                "dtype": str(d["dtype"])}

    return ([spec(d) for d in interp.get_input_details()],
            [spec(d) for d in interp.get_output_details()])


def convert_one(tflite_path: Path, onnx_path: Path, opset: int) -> None:
    import onnx
    import onnx.shape_inference
    import tf2onnx

    ins, outs = tflite_io(tflite_path)
    print(f"\n== {tflite_path.name}")
    for s in ins:
        print(f"   in : {s['name']} {s['shape']} {s['dtype']}")
    for s in outs:
        print(f"   out: {s['name']} {s['shape']} {s['dtype']}")

    model, _ = tf2onnx.convert.from_tflite(str(tflite_path), opset=opset)
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as exc:  # shape inference is best-effort
        print(f"   (shape inference skipped: {exc})")
    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_path))
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"   -> {onnx_path.name} ({size_mb:.2f} MB, opset {model.opset_import[0].version})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR,
                        help="directory with the tflite models (default: pretrained/)")
    parser.add_argument("--opset", type=int, default=16)
    args = parser.parse_args()

    for tflite_name, onnx_name in PAIRS:
        tflite_path = args.models_dir / tflite_name
        onnx_path = args.models_dir / onnx_name
        if not tflite_path.exists():
            raise SystemExit(f"error: {tflite_path} not found "
                             f"(run `python -m tools.download_models` first)")
        convert_one(tflite_path, onnx_path, args.opset)


if __name__ == "__main__":
    main()
