"""Annotation drawing: box + score, four eye boundary points, one iris point
per eye, and four mouth boundary points (no connecting lines).

Landmark indices come from the 478-point MediaPipe face mesh. Per eye the four
boundary points are the outer corner, inner corner, upper lid and lower lid;
the iris is represented by its center point.
"""

from __future__ import annotations

import cv2
import numpy as np

from .face_pipeline import FaceResult

RIGHT_EYE = [33, 133, 159, 145]     # outer, inner, upper, lower
LEFT_EYE = [263, 362, 386, 374]     # outer, inner, upper, lower
RIGHT_IRIS = [468]                  # right iris center
LEFT_IRIS = [473]                   # left iris center
MOUTH = [61, 291, 0, 17]            # corners + top/bottom center

BOX_COLOR = (0, 200, 0)      # green
EYE_COLOR = (0, 200, 255)    # amber
RIGHT_IRIS_COLOR = (0, 0, 255)   # red
LEFT_IRIS_COLOR = (255, 0, 0)    # blue
LIP_COLOR = (255, 0, 200)        # pink


def _pts(landmarks: np.ndarray, indices) -> np.ndarray:
    pts = np.array([[landmarks[i][0], landmarks[i][1]] for i in indices], np.float32)
    return np.round(pts).astype(np.int32)


def draw(image_bgr: np.ndarray, res: FaceResult) -> np.ndarray:
    out = image_bgr.copy()
    x, y, w, h = (int(round(v)) for v in res.box)
    cv2.rectangle(out, (x, y), (x + w, y + h), BOX_COLOR, 2)
    cv2.putText(out, f"{res.score:.2f}", (x, max(12, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1)

    lms = res.landmarks
    if len(lms) >= 478:
        for group, color in ((RIGHT_EYE, EYE_COLOR), (LEFT_EYE, EYE_COLOR),
                             (MOUTH, LIP_COLOR), (RIGHT_IRIS, RIGHT_IRIS_COLOR),
                             (LEFT_IRIS, LEFT_IRIS_COLOR)):
            for p in _pts(lms, group):
                cv2.circle(out, tuple(p), 2, color, -1)
    return out
