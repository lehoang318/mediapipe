#!/usr/bin/env python3
"""End-to-end face detection + 478-point face landmarks using the ONNX models.

Replicates the MediaPipe Tasks pipeline in pure Python + ONNX Runtime:

  Detector stage (blaze_face_short_range.onnx OR blaze_face_full_range.onnx):
    - short range: 128x128, strides 8/16/16/16  -> 896 anchors
    - full range:  192x192, stride 4 (48x48 grid) -> 2304 anchors
      (both: RGB in [-1, 1], fixed-size anchors, interpolated scale
       aspect ratio 1.0 / 0.0 respectively)
    - letterbox image to the model input size (keep aspect ratio, zero padding)
    - decode SSD regressors/classificators with fixed-size anchors
    - sigmoid scores, confidence threshold, weighted NMS (IoU)
    - project detections back to full-image coordinates

  Landmark stage (face_landmarks_detector.onnx, 256x256, RGB in [0, 1]):
    - rotated face ROI from the two eye keypoints (target angle 0),
      expanded by 1.5 in width and height (RectTransformationCalculator)
    - rotate + crop ROI, FIT-scale with zero padding into 256x256
    - outputs: 1434 = 478 x (x, y, z) landmark values, presence logit,
      auxiliary output (unused)
    - remove letterbox padding, then project landmarks from the rotated
      ROI back into full-image pixel coordinates
      (LandmarkLetterboxRemovalCalculator + LandmarkProjectionCalculator)

The detector model variant is inferred from the ONNX input shape, so passing
blaze_face_full_range.onnx automatically selects the full-range anchors.

Usage:
    python -m faceframework.face_pipeline --image pretrained/portrait.jpg \
        --out annotated.jpg --json result.json [--show-check]
    python -m faceframework.face_pipeline --image img.jpg \
        --detector pretrained/blaze_face_full_range.onnx
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

DET_SIZE = 128
MODELS_DIR = Path(__file__).resolve().parent.parent / "pretrained"
LMK_SIZE = 256

# Detector variants, keyed by square ONNX input size. Faithful ports of the
# MediaPipe SsdAnchorsCalculator options used by the Tasks face detector graph:
#   * near range / 128x128: num_layers=4, strides [8,16,16,16],
#     interpolated_scale_aspect_ratio=1.0  -> 896 anchors
#   * far range  / 192x192: num_layers=1, strides [4] (48x48 grid),
#     interpolated_scale_aspect_ratio=0.0  -> 2304 anchors
# Both use min_scale=0.1484375, max_scale=0.75, aspect_ratios=[1.0],
# fixed_anchor_size=true (so the scale values only matter through the anchor
# count; w = h = 1.0). See ConfigureSsdAnchorsCalculator in face_detector_graph.cc.
DETECTOR_CONFIGS = {
    128: {"num_layers": 4, "strides": [8, 16, 16, 16],
          "min_scale": 0.1484375, "max_scale": 0.75, "interp_ar": 1.0},
    192: {"num_layers": 1, "strides": [4],
          "min_scale": 0.1484375, "max_scale": 0.75, "interp_ar": 0.0},
}

# Default options of mediapipe.tasks FaceDetector / FaceLandmarker.
MIN_DETECTION_CONFIDENCE = 0.5
MIN_SUPPRESSION_THRESHOLD = 0.5

# RectTransformationCalculator options of the face detector graph.
ROI_SCALE_X = 1.5
ROI_SCALE_Y = 1.5

# DetectionsToRectsCalculator options of the face landmarker graph:
# rotation from detector keypoints 0 (left eye) and 1 (right eye),
# rotation_vector_target_angle_degrees = 0.
ROTATION_START_KEYPOINT = 0
ROTATION_END_KEYPOINT = 1
ROTATION_TARGET_ANGLE = 0.0

# Face mesh landmark indices for the self-check: outer eye corners,
# nose tip, inner lip center.
EYE_OUTER = {"left": 263, "right": 33}
NOSE_TIP = 1
LIP_CENTER = 14


def normalize_radians(angle: float) -> float:
    return angle - 2 * np.pi * np.floor((angle - (-np.pi)) / (2 * np.pi))


def gen_ssd_anchors(input_size: int = DET_SIZE, *,
                    num_layers: int | None = None,
                    strides: list | None = None,
                    min_scale: float = 0.1484375,
                    max_scale: float = 0.75,
                    interp_ar: float = 1.0) -> np.ndarray:
    """SSD anchors, faithful port of mediapipe SsdAnchorsCalculator.

    With the defaults this reproduces the short-range face detector options
    (num_layers=4, input 128, strides [8,16,16,16], aspect_ratios [1.0],
    fixed_anchor_size=true) and returns (896, 2) normalized anchor centers.
    For the full-range model use input_size=192, num_layers=1, strides=[4],
    interp_ar=0.0 -> (2304, 2). Fixed anchor size means w = h = 1.0, so the
    scale values do not enter the (x_center, y_center) output.
    """
    if strides is None:
        strides = [8, 16, 16, 16]
    if num_layers is None:
        num_layers = len(strides)
    aspect_ratios = [1.0]

    def calc_scale(i):
        if num_layers == 1:
            return (min_scale + max_scale) * 0.5
        return min_scale + (max_scale - min_scale) * i / (num_layers - 1)

    anchors = []
    layer_id = 0
    while layer_id < num_layers:
        scales, ratios = [], []
        last_same = layer_id
        while last_same < len(strides) and strides[last_same] == strides[layer_id]:
            scale = calc_scale(last_same)
            for ar in aspect_ratios:
                ratios.append(ar)
                scales.append(scale)
            if interp_ar > 0.0:
                scale_next = (1.0 if last_same == len(strides) - 1
                              else calc_scale(last_same + 1))
                scales.append(float(np.sqrt(scale * scale_next)))
                ratios.append(interp_ar)
            last_same += 1

        stride = strides[layer_id]
        fm_h = int(np.ceil(input_size / stride))
        fm_w = int(np.ceil(input_size / stride))

        for y in range(fm_h):
            for x in range(fm_w):
                x_center = (x + 0.5) / fm_w
                y_center = (y + 0.5) / fm_h
                for _ in scales:  # fixed_anchor_size: w = h = 1.0
                    anchors.append((x_center, y_center))
        layer_id = last_same

    return np.asarray(anchors, dtype=np.float32)


@dataclass
class Detection:
    score: float
    box: tuple  # (x, y, w, h), normalized to the full image
    keypoints: np.ndarray  # (6, 2), normalized to the full image


@dataclass
class FaceResult:
    score: float
    presence: float
    box: tuple  # detection box (x, y, w, h) in pixels
    keypoints: np.ndarray  # (6, 2) detector keypoints in pixels
    roi: dict  # rotated crop rect: x_center, y_center (px), width, height (px), rotation (rad)
    landmarks: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))


def sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def weighted_nms(boxes: np.ndarray, scores: np.ndarray,
                 iou_threshold: float) -> list:
    """Greedy weighted NMS, mediapipe NonMaxSuppressionCalculator
    (overlap_type=IoU, algorithm=WEIGHTED). boxes: (N, 4) as x0, y0, x1, y1.
    Returns (weighted_box, top_index, group_indices) per kept detection;
    the group is needed because mediapipe weight-averages keypoints too."""
    order = np.argsort(-scores)
    candidates = list(order)

    def iou(a, b):
        ix0, iy0 = np.maximum(a[:2], b[:2])
        ix1, iy1 = np.minimum(a[2:], b[2:])
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    kept = []
    while candidates:
        i = candidates.pop(0)
        overlapping = [i] + [j for j in candidates if iou(boxes[i], boxes[j]) > iou_threshold]
        for j in overlapping[1:]:
            candidates.remove(j)
        w = scores[overlapping]
        kept.append(((boxes[overlapping] * w[:, None]).sum(0) / w.sum(),
                     i, overlapping))
    return kept


class FacePipeline:
    def __init__(self, detector_onnx: str | Path, landmarker_onnx: str | Path,
                 min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
                 min_suppression_threshold: float = MIN_SUPPRESSION_THRESHOLD,
                 num_faces: int | None = None,
                 providers: list[str] | None = None):
        so = ort.SessionOptions()
        self.providers = list(providers) if providers else ["CPUExecutionProvider"]
        self.det = ort.InferenceSession(str(detector_onnx), so, self.providers)
        self.lmk = ort.InferenceSession(str(landmarker_onnx), so, self.providers)
        self.det_in = self.det.get_inputs()[0].name
        self.lmk_in = self.lmk.get_inputs()[0].name

        # Infer the detector variant from the ONNX input shape. The TensorFlow
        # Lite models declare a fixed square input, so [1, 128, 128, 3] selects
        # the near-range anchors and [1, 192, 192, 3] the far-range ones.
        in_shape = self.det.get_inputs()[0].shape
        height, width = int(in_shape[1]), int(in_shape[2])
        if height != width or height not in DETECTOR_CONFIGS:
            raise ValueError(
                f"unsupported detector input shape {in_shape} for "
                f"{detector_onnx!r}; known sizes: {sorted(DETECTOR_CONFIGS)}")
        self.det_size = height
        self.anchors = gen_ssd_anchors(self.det_size, **DETECTOR_CONFIGS[self.det_size])
        self.num_anchors = int(self.anchors.shape[0])
        self.min_detection_confidence = min_detection_confidence
        self.min_suppression_threshold = min_suppression_threshold
        self.num_faces = num_faces

    # ------------------------------------------------------------- detector

    def _letterbox(self, image: np.ndarray, size: int):
        """Whole-image letterbox: keep aspect ratio, zero padding, centered.

        Uses a float affine warp (like MediaPipe's ImageToTensorCalculator);
        rounding the content size to integers changes model outputs enough to
        alter scores and NMS results. Returns (tensor, scale, offset_x, offset_y)."""
        h, w = image.shape[:2]
        s = min(size / w, size / h)
        ox, oy = (size - w * s) / 2.0, (size - h * s) / 2.0
        m = np.array([[s, 0.0, ox], [0.0, s, oy]], dtype=np.float64)
        canvas = cv2.warpAffine(image, m, (size, size), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return canvas, s, ox, oy

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        h, w = image_rgb.shape[:2]
        tensor, s, ox, oy = self._letterbox(image_rgb, self.det_size)
        x = tensor.astype(np.float32) / 127.5 - 1.0
        x = x[None]
        regressors, classificators = self.det.run(
            None, {self.det_in: x})
        n = self.num_anchors
        if regressors.shape[1] != n:
            raise ValueError(
                f"detector produced {regressors.shape[1]} boxes but "
                f"{n} anchors were generated for a {self.det_size}px input")
        regressors = regressors.reshape(n, 16)
        scores = sigmoid(np.clip(classificators.reshape(n), -100.0, 100.0))

        # DecodeBoxes (reverse_output_order, no exponential box size,
        # x/y/w/h_scale = det_size, fixed anchor size 1.0):
        #   v = raw / scale * anchor_size + anchor_center
        ax, ay = self.anchors[:, 0:1], self.anchors[:, 1:2]
        xc = regressors[:, 0:1] / self.det_size + ax
        yc = regressors[:, 1:2] / self.det_size + ay
        bw = regressors[:, 2:3] / self.det_size
        bh = regressors[:, 3:4] / self.det_size
        # tensor-space boxes as x0, y0, x1, y1 (normalized [0, 1])
        boxes = np.concatenate(
            [xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2], axis=1)
        # keypoints: 6 x (x, y)
        kpts = regressors[:, 4:16].reshape(n, 6, 2).copy()
        kpts[:, :, 0:1] = kpts[:, :, 0:1] / self.det_size + ax[:, :, None]
        kpts[:, :, 1:2] = kpts[:, :, 1:2] / self.det_size + ay[:, :, None]

        keep = scores >= self.min_detection_confidence
        if not keep.any():
            return []
        boxes_t, kpts_t, scores_k = boxes[keep], kpts[keep], scores[keep]

        # Project tensor coordinates back to full-image normalized coords.
        def to_image(px_x, px_y):
            return (px_x - ox) / s / w, (px_y - oy) / s / h

        # Weighted NMS runs in tensor space (mediapipe: TensorsToDetections ->
        # NonMaxSuppression -> DetectionProjection), *before* the projection.
        nms_boxes = weighted_nms(boxes_t, scores_k,
                                 self.min_suppression_threshold)

        detections = []
        selected = nms_boxes[: self.num_faces] if self.num_faces else nms_boxes
        for box_t, idx, grp in selected:
            # box_t and the keypoints are the score-weighted averages of the
            # overlapping detections (mediapipe WEIGHTED NMS behaviour).
            wt = scores_k[grp]
            kpts_w = (kpts_t[grp] * wt[:, None, None]).sum(0) / wt.sum()
            x0, y0 = to_image(box_t[0] * self.det_size, box_t[1] * self.det_size)
            x1, y1 = to_image(box_t[2] * self.det_size, box_t[3] * self.det_size)
            kx, ky = to_image(kpts_w[:, 0] * self.det_size,
                              kpts_w[:, 1] * self.det_size)
            detections.append(Detection(
                score=float(scores_k[idx]),
                box=(float(x0), float(y0),
                     float(x1 - x0), float(y1 - y0)),
                keypoints=np.stack([kx, ky], axis=-1),
            ))
        return detections

    # ------------------------------------------------------------ landmarks

    def _detection_to_roi(self, det: Detection, image_w: int,
                          image_h: int) -> dict:
        """DetectionsToRectsCalculator + RectTransformationCalculator
        (scale_x = scale_y = 1.5, no shift, rotation from eye keypoints,
        target angle 0). The 1.5 expansion is essential: the landmarker model
        is trained on ROIs that are 1.5x the detection box. Rotation is
        computed in pixel space (atan2(dy_px, dx_px)), like mediapipe."""
        x, y, bw, bh = det.box
        start = det.keypoints[ROTATION_START_KEYPOINT]
        end = det.keypoints[ROTATION_END_KEYPOINT]
        rotation = normalize_radians(
            np.arctan2((end[1] - start[1]) * image_h,
                       (end[0] - start[0]) * image_w)
            - np.deg2rad(ROTATION_TARGET_ANGLE))
        return {
            "x_center": x + bw / 2,
            "y_center": y + bh / 2,
            "width": bw * ROI_SCALE_X,
            "height": bh * ROI_SCALE_Y,
            "rotation": float(rotation),
        }

    def _crop_roi(self, image_rgb: np.ndarray, roi: dict) -> np.ndarray:
        """Rotate + crop the ROI, FIT-scale with zero padding into 256x256.
        Returns the float32 tensor in [0, 1] and the letterbox paddings."""
        h, w = image_rgb.shape[:2]
        cx, cy = roi["x_center"] * w, roi["y_center"] * h
        rw, rh = roi["width"] * w, roi["height"] * h
        theta = roi["rotation"]

        s = min(LMK_SIZE / rw, LMK_SIZE / rh)
        left = (LMK_SIZE - rw * s) / 2.0
        top = (LMK_SIZE - rh * s) / 2.0

        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # Affine map: dst tensor pixel (tx, ty) -> src image pixel.
        #   u = (tx - left) / s - rw / 2 ; v = (ty - top) / s - rh / 2
        #   src = (cx, cy) + R(theta) @ (u, v)
        m = np.array([
            [cos_t / s, -sin_t / s,
             cx - left * cos_t / s + top * sin_t / s - cos_t * rw / 2 + sin_t * rh / 2],
            [sin_t / s, cos_t / s,
             cy - left * sin_t / s - top * cos_t / s - (sin_t * rw + cos_t * rh) / 2],
        ], dtype=np.float64)
        crop = cv2.warpAffine(image_rgb, m, (LMK_SIZE, LMK_SIZE),
                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return crop.astype(np.float32) / 255.0, left, top, s

    def _remove_letterbox(self, landmarks: np.ndarray, left: float,
                          top: float) -> np.ndarray:
        """LandmarkLetterboxRemovalCalculator: padding [left, top, right, bottom]
        in normalized tensor units, right = left, bottom = top. z scales as x.
        `left`/`top` arrive in tensor pixels, so normalize them first."""
        ln = left / LMK_SIZE
        tn = top / LMK_SIZE
        out = landmarks.copy()
        out[:, 0] = (landmarks[:, 0] - ln) / (1.0 - 2.0 * ln)
        out[:, 1] = (landmarks[:, 1] - tn) / (1.0 - 2.0 * tn)
        out[:, 2] = landmarks[:, 2] / (1.0 - 2.0 * ln)
        return out

    def _project_landmarks(self, landmarks: np.ndarray, roi: dict,
                           image_w: int, image_h: int) -> np.ndarray:
        """LandmarkProjectionCalculator (normalized-rect path) in pixel space."""
        h, w = image_h, image_w
        cx, cy = roi["x_center"] * w, roi["y_center"] * h
        rw, rh = roi["width"] * w, roi["height"] * h
        theta = roi["rotation"]
        x = landmarks[:, 0] - 0.5
        y = landmarks[:, 1] - 0.5
        px = (np.cos(theta) * x - np.sin(theta) * y) * rw + cx
        py = (np.sin(theta) * x + np.cos(theta) * y) * rh + cy
        pz = landmarks[:, 2] * rw  # scale z coordinate as x
        return np.stack([px, py, pz], axis=1)

    def landmark_single(self, image_rgb: np.ndarray, det: Detection) -> FaceResult:
        """Run the landmark stage for a single detection and project the
        478 landmarks back into full-image pixel coordinates."""
        h, w = image_rgb.shape[:2]
        roi = self._detection_to_roi(det, w, h)
        tensor, left, top, s = self._crop_roi(image_rgb, roi)
        outs = self.lmk.run(None, {self.lmk_in: tensor[None]})
        lmk = outs[0].reshape(-1, 3).astype(np.float64) / LMK_SIZE
        presence = float(sigmoid(float(outs[1].reshape(-1)[0])))
        lmk = self._remove_letterbox(lmk, left, top)
        pts = self._project_landmarks(lmk, roi, w, h)
        return FaceResult(
            score=det.score,
            presence=presence,
            box=(det.box[0] * w, det.box[1] * h,
                 det.box[2] * w, det.box[3] * h),
            keypoints=det.keypoints * np.array([w, h]),
            roi={"x_center": roi["x_center"] * w,
                 "y_center": roi["y_center"] * h,
                 "width": roi["width"] * w,
                 "height": roi["height"] * h,
                 "rotation": roi["rotation"]},
            landmarks=pts,
        )

    def landmarks(self, image_rgb: np.ndarray) -> list[FaceResult]:
        return [self.landmark_single(image_rgb, det)
                for det in self.detect(image_rgb)]


# ------------------------------------------------------------------ helpers

def draw_result(image_bgr: np.ndarray, res: FaceResult) -> np.ndarray:
    out = image_bgr.copy()
    x, y, bw, bh = (int(round(v)) for v in res.box)
    cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 200, 0), 2)
    cv2.putText(out, f"{res.score:.2f}", (x, max(12, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    for kx, ky in res.keypoints:
        cv2.circle(out, (int(round(kx)), int(round(ky))), 3, (0, 140, 255), -1)
    for lx, ly, _ in res.landmarks:
        cv2.circle(out, (int(round(lx)), int(round(ly))), 1, (0, 220, 255), -1)
    return out


def self_check(res: FaceResult) -> dict:
    """Compare detector keypoints with the corresponding mesh landmarks.
    BlazeFace keypoints: [right eye, left eye, nose, mouth, right ear, left ear]
    (subject-relative). Mesh: 33/263 outer eye corners, 1 nose tip, 14 inner
    lip center. The detector's eye keypoints sit slightly outside the mesh
    corners, so ~10 px on a 250 px face is the expected agreement."""
    lms = res.landmarks
    pairs = [(0, EYE_OUTER["right"]), (1, EYE_OUTER["left"]),
             (2, NOSE_TIP), (3, LIP_CENTER)]
    dists = {}
    for kp, lm in pairs:
        d = float(np.hypot(*(res.keypoints[kp][:2] - lms[lm][:2])))
        dists[f"kp{kp}_lm{lm}"] = round(d, 2)
    return dists


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", default=str(MODELS_DIR / "blaze_face_short_range.onnx"))
    parser.add_argument("--landmarker", default=str(MODELS_DIR / "face_landmarks_detector.onnx"))
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default=None, help="save annotated image here")
    parser.add_argument("--json", default=None, help="save results as json")
    parser.add_argument("--num-faces", type=int, default=None)
    parser.add_argument("--min-det-conf", type=float, default=MIN_DETECTION_CONFIDENCE)
    parser.add_argument("--nms-iou", type=float, default=MIN_SUPPRESSION_THRESHOLD)
    parser.add_argument("--show-check", action="store_true",
                        help="print detector-keypoint vs landmark distances")
    args = parser.parse_args()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise SystemExit(f"error: cannot read image {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    pipeline = FacePipeline(args.detector, args.landmarker,
                            min_detection_confidence=args.min_det_conf,
                            min_suppression_threshold=args.nms_iou,
                            num_faces=args.num_faces)
    results = pipeline.landmarks(image_rgb)

    print(f"{len(results)} face(s) in {args.image}")
    payload = []
    for i, res in enumerate(results):
        print(f"face {i}: score={res.score:.3f} presence={res.presence:.3f} "
              f"box={tuple(round(v, 1) for v in res.box)} "
              f"rotation={np.degrees(res.roi['rotation']):.1f} deg")
        entry = {
            "score": res.score,
            "presence": res.presence,
            "box_px": list(res.box),
            "roi_px": res.roi,
            "keypoints_px": res.keypoints.tolist(),
            "landmarks_px": res.landmarks.tolist(),
        }
        if args.show_check:
            check = self_check(res)
            entry["keypoint_landmark_check_px"] = check
            print("  det-keypoint vs mesh-landmark distance (px):", check)
        payload.append(entry)

    if args.out:
        annotated = image_bgr
        for res in results:
            annotated = draw_result(annotated, res)
        cv2.imwrite(args.out, annotated)
        print(f"annotated image -> {args.out}")
    if args.json:
        Path(args.json).write_text(json.dumps(payload))
        print(f"results -> {args.json}")


if __name__ == "__main__":
    main()
