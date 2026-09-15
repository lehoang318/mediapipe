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

Three standalone scripts handle the models. The download step needs no
TensorFlow; the other two need the dev dependencies
(`pip install -r requirements-dev.txt`).

1. Download the official MediaPipe models (tflite):

   ```bash
   python -m tools.download_models                # into pretrained/
   python -m tools.download_models --test-image   # also fetches portrait.jpg
   ```

   - `--models-dir DIR` — target directory (default `pretrained/`)
   - `--test-image` — additionally downloads the sample face image
   - Fetches the two BlazeFace detectors and `face_landmarker.task`, and
     extracts `face_landmarks_detector.tflite` from the task bundle;
     already-present files are skipped.

2. Convert tflite -> ONNX (requires tensorflow-cpu + tf2onnx):

   ```bash
   python -m tools.convert_to_onnx
   ```

   - `--models-dir DIR` — directory with the tflite models (default `pretrained/`)
   - `--opset N` — ONNX opset version (default 16)
   - Converts both detectors and the landmarker next to the tflite files; if a
     tflite file is missing it errors with a reminder to run the download
     script first.

3. Check ONNX numerically matches tflite (optional):

   ```bash
   python -m tools.verify_parity [--models-dir DIR] [--atol F] [--seed N]
   ```

The steps are independent commands; `python main.py` runs download +
conversion automatically on first run when ONNX files are missing.

## C++ port

`cpp/` is a C++17 port of the display app (`main.py` + `faceframework`): it
detects the biggest face with score >= the threshold, extracts the 478
landmarks and shows the same 14-point overlay, HUD, pause/resume and exit
summary. It shares the ONNX models in `pretrained/` with the Python app.

### Dependencies

The C++ port does not bundle its dependencies; install them beforehand:

- C++17 compiler and CMake >= 3.16
- OpenCV built with the `core`, `imgproc`, `imgcodecs`, `videoio` and
  `highgui` modules (GUI display needs a toolkit such as GTK)
- ONNX Runtime **GPU** (shared libraries, CUDA Execution Provider)
- CUDA + cuDNN runtime libraries for the GPU provider

By default CMake picks up OpenCV through `find_package` (point it elsewhere
with `-DOpenCV_DIR=...`) and ONNX Runtime from `$HOME/.local` (override with
`-DORT_ROOT=...`).

### Build and run

```bash
cmake -S cpp -B cpp/build && cmake --build cpp/build -j

cpp/build/face_video --pretrained pretrained --video /path/to/clip.mp4
cpp/build/face_video --pretrained pretrained --camera 0
cpp/build/face_video --pretrained pretrained --video clip.mp4 --far
```

`--pretrained` is required; pass `--video` or `--camera` (`--camera` wins when
both are given), and `--far` selects the full-range detector for faces beyond
~2 m. Controls are the same as the Python app: `q` quits, `space` pauses.
Thresholds are configurable via `--min-score`/`--nms-iou`; use `--cpu` to
force the CPU provider, `--no-display` for headless benchmarking and
`--max-frames N` to stop after N frames — the run then ends with per-stage
latency histograms and a summary.

## Dependencies

- `requirements.txt` is the minimum to **run** the framework: `onnxruntime-gpu`, `numpy` and `opencv-python`
- `requirements-dev.txt` adds the one-time **bootstrap/verification** stack (`tensorflow-cpu`, `tf2onnx`, `onnx`) plus everything from the runtime file via `-r requirements.txt`
  - The heavy bootstrap imports stay lazy, so a normal run never touches TensorFlow.

## License

MIT. See [LICENSE](LICENSE).
