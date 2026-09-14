"""Face framework: camera/video -> biggest face -> eyes/iris/mouth landmarks.

The verified ONNX port lives in ``face_pipeline.py`` inside this package. The
repo root is kept on ``sys.path`` so the sibling ``tools`` package (used only
for the one-time model download/convert bootstrap) stays importable.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
