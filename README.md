# face-framework

This project is a reimplementation of the MediaPipe face pipeline. Models and
algorithm design belong to the MediaPipe project (Google LLC, Apache-2.0):

- MediaPipe Face Detection: <https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector>
- MediaPipe Face Landmarker: <https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker>
- MediaPipe repository: <https://github.com/google-ai-edge/mediapipe>
- BlazeFace paper: Bazarevsky et al., *BlazeFace: Sub-millisecond Neural Face
  Detection on Mobile GPUs*, arXiv:1907.05047

Model files under `pretrained/` are generated from the official MediaPipe model URLs and remain under their original (Apache-2.0) license.

## Layout

```
main.py                  entry point (video / camera / --far)
config.json              runtime config (model dir, fps, thresholds, window)
faceframework/           the library (engine, pipeline, overlay, sources, app)
tools/                   model bootstrap + verification scripts
pretrained/              downloaded tflite + converted onnx models (generated)
cpp/                     C/C++ port (reference implementation)
```

## Install

Runtime:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Bootstrap/verification dependencies (only if you need to download/convert or
re-verify the models):

```bash
pip install -r requirements-dev.txt
```

## Usage

On first run the models are fetched into `pretrained/` automatically (tflite +
tf2onnx conversion), which requires the dev dependencies above.

```bash
# webcam (device index)
python main.py --camera 0

# video file (FPS taken from the file, config fallback)
python main.py --video /path/to/clip.mp4

# faces farther than ~2 m: full-range detector
python main.py --video /path/to/clip.mp4 --far

# custom config
python main.py --video clip.mp4 --config myconfig.json
```

In the preview window: `q` quits, `space` pauses. `--camera` takes priority over
`--video` when both are given. Without `--far`, the near-range
(`blaze_face_short_range`) detector is used.

### Engine CLI (single image)

```bash
python -m faceframework.face_pipeline --image pretrained/portrait.jpg \
    --out annotated.jpg --json result.json [--show-check]
```

### Tools

```bash
python -m tools.download_models [--test-image]     # fetch tflite models
python -m tools.convert_to_onnx                    # tflite -> onnx
python -m tools.verify_parity                      # tflite vs onnx parity
```

## Dependencies

- `requirements.txt` is the minimum to **run** the framework: `onnxruntime-gpu`, `numpy` and `opencv-python`
- `requirements-dev.txt` adds the one-time **bootstrap/verification** stack (`tensorflow-cpu`, `tf2onnx`, `onnx`) plus everything from the runtime file via `-r requirements.txt`
  - The heavy bootstrap imports stay lazy, so a normal run never touches TensorFlow.

## License

MIT. See [LICENSE](LICENSE).
