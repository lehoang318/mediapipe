#!/usr/bin/env python3
"""Verify numeric parity between the MediaPipe tflite models and their
ONNX conversions.

For every model, identical input is fed to the TFLite interpreter and to
ONNX Runtime; every output tensor is compared element-wise.

Usage:
    python -m tools.verify_parity [--models-dir pretrained] [--atol 1e-4]
"""

import argparse
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "pretrained"

# (tflite, onnx, input lo, input hi, atol).
# The source models store fp16 weights, so TFLite (XNNPACK) and ONNX Runtime
# accumulate convolutions in different orders; ~1e-4..5e-4 max disagreement on
# the landmark regressor (~0.1 px in 256-space) is expected fp16 noise.
MODELS = [
    ("blaze_face_short_range.tflite", "blaze_face_short_range.onnx", -1.0, 1.0, 1e-4),
    ("blaze_face_full_range.tflite", "blaze_face_full_range.onnx", -1.0, 1.0, 1e-4),
    ("face_landmarks_detector.tflite", "face_landmarks_detector.onnx", 0.0, 1.0, 1e-3),
]


def tflite_run(tflite_path: Path, feeds: dict):
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    for d in interp.get_input_details():
        interp.set_tensor(d["index"], feeds[d["name"]])
    interp.invoke()
    return [interp.get_tensor(d["index"]) for d in interp.get_output_details()]


def onnx_run(onnx_path: Path, feeds: dict):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR,
                        help="directory with the tflite + onnx models "
                             "(default: pretrained/)")
    parser.add_argument("--atol", type=float, default=None,
                        help="override per-model absolute tolerance")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    failures = []

    for tflite_name, onnx_name, lo, hi, atol in MODELS:
        if args.atol is not None:
            atol = args.atol
        tflite_path = args.models_dir / tflite_name
        onnx_path = args.models_dir / onnx_name
        print(f"== {tflite_name}  vs  {onnx_name}  (atol={atol:g})")

        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=str(tflite_path))
        interp.allocate_tensors()
        feeds = {}
        for d in interp.get_input_details():
            shape = tuple(int(s) for s in d["shape"])
            feeds[d["name"]] = rng.uniform(lo, hi, size=shape).astype(np.float32)

        tfl_outs = tflite_run(tflite_path, feeds)
        ort_outs = onnx_run(onnx_path, feeds)

        if len(tfl_outs) != len(ort_outs):
            failures.append((tflite_name, "output count mismatch"))
            print(f"   FAIL: {len(tfl_outs)} tflite outputs vs "
                  f"{len(ort_outs)} onnx outputs")
            continue

        ok = True
        for i, (a, b) in enumerate(zip(tfl_outs, ort_outs)):
            a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
            if a.shape != b.shape:
                failures.append((tflite_name, f"output {i} shape {a.shape} != {b.shape}"))
                print(f"   output {i}: SHAPE MISMATCH {a.shape} vs {b.shape}")
                ok = False
                continue
            max_abs = float(np.max(np.abs(a - b)))
            denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-6)
            max_rel = float(np.max(np.abs(a - b) / denom))
            status = "OK " if max_abs <= atol else "FAIL"
            print(f"   output {i}: shape {a.shape}  max_abs={max_abs:.3e}  "
                  f"max_rel={max_rel:.3e}  [{status}]")
            if max_abs > atol:
                failures.append((tflite_name, f"output {i} max_abs={max_abs:.3e}"))
                ok = False
        if ok:
            print("   parity PASSED")

    print()
    if failures:
        for name, why in failures:
            print(f"PARITY FAILED: {name}: {why}")
        raise SystemExit(1)
    print("All models passed parity checks.")


if __name__ == "__main__":
    main()
