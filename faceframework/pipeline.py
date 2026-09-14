"""Biggest-face detection followed by landmark extraction.

Only the largest detected face whose score clears ``min_score`` is processed;
the detector session is built with that same threshold so the gate matches
MediaPipe's ``min_detection_confidence`` semantics (sub-threshold candidates
never enter NMS).
"""

from __future__ import annotations

import cv2

from .face_pipeline import FacePipeline, FaceResult

DEFAULT_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


class BiggestFacePipeline:
    def __init__(self, detector_onnx, landmarker_onnx, min_score: float = 0.7,
                 nms_iou: float = 0.5, providers=None):
        self.min_score = float(min_score)
        self.fp = FacePipeline(
            detector_onnx, landmarker_onnx,
            min_detection_confidence=self.min_score,
            min_suppression_threshold=nms_iou,
            num_faces=None,
            providers=providers or DEFAULT_PROVIDERS,
        )

    @property
    def providers(self) -> list[str]:
        return self.fp.det.get_providers()

    def process(self, frame_bgr) -> FaceResult | None:
        """Return the landmark result for the biggest confident face, else None."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        candidates = [d for d in self.fp.detect(rgb) if d.score >= self.min_score]
        if not candidates:
            return None
        best = max(candidates, key=lambda d: d.box[2] * d.box[3])
        return self.fp.landmark_single(rgb, best)
