#include "face_pipeline.h"

#include "anchors.h"
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

constexpr int kLmkSize = 256;
constexpr float kRoiScale = 1.5f;
constexpr double kPi = 3.14159265358979323846;

// SsdAnchorsCalculator options shared by both detector variants.
constexpr float kMinScale = 0.1484375f, kMaxScale = 0.75f;

float sigmoid(float x) {
  x = std::clamp(x, -60.0f, 60.0f);
  return 1.0f / (1.0f + std::exp(-x));
}

double sigmoid(double x) {
  x = std::clamp(x, -60.0, 60.0);
  return 1.0 / (1.0 + std::exp(-x));
}

// normalize_radians, same as mediapipe
double normalize_radians(double angle) {
  return angle - 2.0 * kPi * std::floor((angle - (-kPi)) / (2.0 * kPi));
}

template <class F>
double timed(F&& f) {
  const auto t0 = std::chrono::steady_clock::now();
  f();
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0)
      .count();
}

float iou(const std::array<float, 4>& a, const std::array<float, 4>& b) {
  const float ix0 = std::max(a[0], b[0]), iy0 = std::max(a[1], b[1]);
  const float ix1 = std::min(a[2], b[2]), iy1 = std::min(a[3], b[3]);
  const float inter = std::max(0.f, ix1 - ix0) * std::max(0.f, iy1 - iy0);
  const float area_a = std::max(0.f, a[2] - a[0]) * std::max(0.f, a[3] - a[1]);
  const float area_b = std::max(0.f, b[2] - b[0]) * std::max(0.f, b[3] - b[1]);
  const float uni = area_a + area_b - inter;
  return uni > 0.f ? inter / uni : 0.f;
}

struct Kept {
  std::array<float, 4> box;   // score-weighted average, tensor space
  size_t top;                 // index of the top-scoring detection
  std::vector<size_t> group;  // overlapping detections (incl. top)
};

// Greedy weighted NMS (mediapipe NonMaxSuppressionCalculator, IoU/WEIGHTED).
std::vector<Kept> weighted_nms(const std::vector<std::array<float, 4>>& boxes,
                               const std::vector<float>& scores, float thr) {
  std::vector<size_t> cand(boxes.size());
  std::iota(cand.begin(), cand.end(), 0);
  std::sort(cand.begin(), cand.end(),
            [&](size_t a, size_t b) { return scores[a] > scores[b]; });

  std::vector<Kept> kept;
  while (!cand.empty()) {
    const size_t i = cand.front();
    Kept k{{0.f, 0.f, 0.f, 0.f}, i, {i}};
    std::vector<size_t> rest;
    for (size_t c = 1; c < cand.size(); ++c) {
      const size_t j = cand[c];
      if (iou(boxes[i], boxes[j]) > thr)
        k.group.push_back(j);
      else
        rest.push_back(j);
    }
    float sw = 0.f;
    for (size_t j : k.group) {
      const float w = scores[j];
      sw += w;
      for (int t = 0; t < 4; ++t) k.box[t] += boxes[j][t] * w;
    }
    for (auto& v : k.box) v /= sw;
    kept.push_back(std::move(k));
    cand = std::move(rest);
  }
  return kept;
}

size_t elem_count(const Ort::Value& v) {
  return v.GetTensorTypeAndShapeInfo().GetElementCount();
}

}  // namespace

FacePipeline::FacePipeline(const std::string& detector_onnx,
                           const std::string& landmarker_onnx, float min_score,
                           float nms_iou, bool use_cpu)
    : env_(ORT_LOGGING_LEVEL_WARNING, "face_video"),
      min_score_(min_score),
      nms_iou_(nms_iou) {
  Ort::SessionOptions so;
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (use_cpu) {
    provider_desc_ = "CPU";
  } else {
    try {
      OrtCUDAProviderOptions opts{};  // device 0, arena defaults
      so.AppendExecutionProvider_CUDA(opts);
      provider_desc_ = "CUDA";
    } catch (const Ort::Exception& e) {
      provider_desc_ = std::string("CPU (CUDA unavailable: ") + e.what() + ")";
    }
  }

  det_session_ = Ort::Session(env_, detector_onnx.c_str(), so);
  lmk_session_ = Ort::Session(env_, landmarker_onnx.c_str(), so);
  mem_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // Infer the detector variant from the ONNX input shape. The TensorFlow Lite
  // models declare a fixed square input: [1, 128, 128, 3] selects the
  // near-range anchors and [1, 192, 192, 3] the far-range ones
  // (face_pipeline.py DETECTOR_CONFIGS).
  const auto in_shape =
      det_session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  if (in_shape.size() != 4 || in_shape[1] != in_shape[2]) {
    throw std::runtime_error("unsupported detector input shape for " +
                             detector_onnx);
  }
  det_size_ = static_cast<int>(in_shape[1]);
  int num_layers = 0;
  std::vector<int> strides;
  float interp_ar = 0.0f;
  if (det_size_ == 128) {
    num_layers = 4;
    strides = {8, 16, 16, 16};
    interp_ar = 1.0f;
  } else if (det_size_ == 192) {
    num_layers = 1;
    strides = {4};
    interp_ar = 0.0f;
  } else {
    std::ostringstream os;
    os << "unsupported detector input size " << det_size_ << " for "
       << detector_onnx << "; known sizes: 128, 192";
    throw std::runtime_error(os.str());
  }
  anchors_ = gen_ssd_anchors(det_size_, num_layers, strides.data(), kMinScale,
                             kMaxScale, interp_ar);
  num_anchors_ = anchors_.size() / 2;

  Ort::AllocatorWithDefaultOptions alloc;
  det_in_name_ = det_session_.GetInputNameAllocated(0, alloc).get();
  lmk_in_name_ = lmk_session_.GetInputNameAllocated(0, alloc).get();
  for (size_t i = 0; i < det_session_.GetOutputCount(); ++i)
    det_out_names_.emplace_back(det_session_.GetOutputNameAllocated(i, alloc).get());
  for (size_t i = 0; i < lmk_session_.GetOutputCount(); ++i)
    lmk_out_names_.emplace_back(lmk_session_.GetOutputNameAllocated(i, alloc).get());
}

std::vector<Detection> FacePipeline::detect(const cv::Mat& rgb, StageTimes& st) {
  const int w = rgb.cols, h = rgb.rows;
  const int size = det_size_;

  // Letterbox params (shared by pre-processing and post-projection).
  const float s = std::min(size / float(w), size / float(h));
  const float ox = (size - w * s) / 2.f;
  const float oy = (size - h * s) / 2.f;

  // ---- pre: float affine letterbox (zero padding), /127.5 - 1 ----
  cv::Mat f32;
  st.det_pre += timed([&] {
    cv::Mat canvas;
    const cv::Mat M = (cv::Mat_<double>(2, 3) << s, 0.0, ox, 0.0, s, oy);
    cv::warpAffine(rgb, canvas, M, {size, size}, cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar::all(0));
    canvas.convertTo(f32, CV_32F, 1.0 / 127.5, -1.0);  // x / 127.5 - 1
  });

  // ---- infer ----
  std::vector<Ort::Value> outs;
  st.det_infer += timed([&] {
    const std::array<int64_t, 4> shape{1, size, size, 3};
    Ort::Value in = Ort::Value::CreateTensor<float>(
        mem_info_, reinterpret_cast<float*>(f32.data),
        size_t(size) * size * 3, shape.data(), 4);
    const char* in_names[] = {det_in_name_.c_str()};
    std::vector<const char*> out_names;
    for (const auto& n : det_out_names_) out_names.push_back(n.c_str());
    outs = det_session_.Run(Ort::RunOptions{}, in_names, &in, 1, out_names.data(),
                            out_names.size());
  });

  if (outs.size() < 2 || elem_count(outs[0]) != num_anchors_ * 16 ||
      elem_count(outs[1]) != num_anchors_) {
    throw std::runtime_error("unexpected detector output layout");
  }
  const float* reg = outs[0].GetTensorData<float>();  // [num_anchors, 16]
  const float* cls = outs[1].GetTensorData<float>();  // [num_anchors, 1]

  // ---- post: decode, sigmoid, threshold, weighted NMS, project ----
  std::vector<Detection> dets;
  st.det_post += timed([&] {
    std::vector<std::array<float, 4>> boxes;
    std::vector<std::array<float, 12>> kpts;
    std::vector<float> scores;
    boxes.reserve(num_anchors_);
    kpts.reserve(num_anchors_);
    scores.reserve(num_anchors_);
    for (size_t a = 0; a < num_anchors_; ++a) {
      const float sc = sigmoid(cls[a]);
      if (sc < min_score_) continue;
      const float* r = reg + a * 16;
      const float ax = anchors_[2 * a], ay = anchors_[2 * a + 1];
      const float xc = r[0] / size + ax, yc = r[1] / size + ay;
      const float bw = r[2] / size, bh = r[3] / size;
      boxes.push_back({xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2});
      std::array<float, 12> kp;
      for (int k = 0; k < 6; ++k) {
        kp[2 * k] = r[4 + 2 * k] / size + ax;
        kp[2 * k + 1] = r[5 + 2 * k] / size + ay;
      }
      kpts.push_back(kp);
      scores.push_back(sc);
    }

    const auto kept = weighted_nms(boxes, scores, nms_iou_);
    dets.reserve(kept.size());
    auto to_x = [&](float px) { return (px - ox) / s / w; };
    auto to_y = [&](float py) { return (py - oy) / s / h; };
    for (const auto& k : kept) {
      // Weighted NMS weight-averages the keypoints too.
      float sw = 0.f;
      std::array<float, 12> kw{};
      for (size_t j : k.group) {
        const float wj = scores[j];
        sw += wj;
        for (int t = 0; t < 12; ++t) kw[t] += kpts[j][t] * wj;
      }
      for (auto& v : kw) v /= sw;

      Detection d;
      d.score = scores[k.top];
      const float x0 = to_x(k.box[0] * size), y0 = to_y(k.box[1] * size);
      const float x1 = to_x(k.box[2] * size), y1 = to_y(k.box[3] * size);
      d.box[0] = x0;
      d.box[1] = y0;
      d.box[2] = x1 - x0;
      d.box[3] = y1 - y0;
      d.keypoints.resize(6);
      for (int kk = 0; kk < 6; ++kk)
        d.keypoints[kk] = {to_x(kw[2 * kk] * size),
                           to_y(kw[2 * kk + 1] * size)};
      dets.push_back(std::move(d));
    }
  });
  return dets;
}

void FacePipeline::runLandmarker(const cv::Mat& rgb, const Detection& det,
                                 FaceResult& res, StageTimes& st) {
  const int w = rgb.cols, h = rgb.rows;

  // ---- pre: detection -> ROI (1.5x, eye-keypoint rotation) -> rotated crop ----
  double rotation = 0.0, cx = 0.0, cy = 0.0, rw = 0.0, rh = 0.0;
  cv::Mat f32;
  st.lmk_pre += timed([&] {
    const auto& p0 = det.keypoints[0];  // left eye
    const auto& p1 = det.keypoints[1];  // right eye
    rotation = normalize_radians(
        std::atan2((p1.y - p0.y) * h, (p1.x - p0.x) * w) - 0.0);
    const float x = det.box[0], y = det.box[1];
    const float bw = det.box[2], bh = det.box[3];
    cx = (x + bw / 2.0) * w;
    cy = (y + bh / 2.0) * h;
    rw = bw * kRoiScale * w;
    rh = bh * kRoiScale * h;

    const double s = std::min(kLmkSize / rw, kLmkSize / rh);
    const double left = (kLmkSize - rw * s) / 2.0;
    const double top = (kLmkSize - rh * s) / 2.0;
    const double cs = std::cos(rotation), sn = std::sin(rotation);
    // Affine map: dst tensor pixel (tx, ty) -> src image pixel.
    const cv::Mat M = (cv::Mat_<double>(2, 3) <<
        cs / s, -sn / s,
        cx - left * cs / s + top * sn / s - cs * rw / 2 + sn * rh / 2,
        sn / s, cs / s,
        cy - left * sn / s - top * cs / s - (sn * rw + cs * rh) / 2);
    cv::Mat crop;
    cv::warpAffine(rgb, crop, M, {kLmkSize, kLmkSize},
                   cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                   cv::BORDER_CONSTANT, cv::Scalar::all(0));
    crop.convertTo(f32, CV_32F, 1.0 / 255.0);
  });

  // ---- infer ----
  std::vector<Ort::Value> outs;
  st.lmk_infer += timed([&] {
    const std::array<int64_t, 4> shape{1, kLmkSize, kLmkSize, 3};
    Ort::Value in = Ort::Value::CreateTensor<float>(
        mem_info_, reinterpret_cast<float*>(f32.data),
        size_t(kLmkSize) * kLmkSize * 3, shape.data(), 4);
    const char* in_names[] = {lmk_in_name_.c_str()};
    std::vector<const char*> out_names;
    for (const auto& n : lmk_out_names_) out_names.push_back(n.c_str());
    outs = lmk_session_.Run(Ort::RunOptions{}, in_names, &in, 1, out_names.data(),
                            out_names.size());
  });

  if (outs.size() < 2 || elem_count(outs[0]) != 478 * 3) {
    throw std::runtime_error("unexpected landmarker output layout");
  }

  // ---- post: presence, letterbox removal, projection ----
  st.lmk_post += timed([&] {
    const float* lmk_raw = outs[0].GetTensorData<float>();   // [1434]
    const float* pres_raw = outs[1].GetTensorData<float>();  // [1]
    res.presence = static_cast<float>(sigmoid(double(pres_raw[0])));

    const double s = std::min(kLmkSize / rw, kLmkSize / rh);
    const double left = (kLmkSize - rw * s) / 2.0;
    const double top = (kLmkSize - rh * s) / 2.0;
    const double ln = left / kLmkSize, tn = top / kLmkSize;
    const double cs = std::cos(rotation), sn = std::sin(rotation);

    res.landmarks.resize(478);
    for (int i = 0; i < 478; ++i) {
      double x = double(lmk_raw[3 * i]) / kLmkSize;
      double y = double(lmk_raw[3 * i + 1]) / kLmkSize;
      double z = double(lmk_raw[3 * i + 2]) / kLmkSize;
      // LandmarkLetterboxRemovalCalculator
      x = (x - ln) / (1.0 - 2.0 * ln);
      y = (y - tn) / (1.0 - 2.0 * tn);
      z = z / (1.0 - 2.0 * ln);
      // LandmarkProjectionCalculator (z scales as x)
      const double xu = x - 0.5, yu = y - 0.5;
      const double px = (cs * xu - sn * yu) * rw + cx;
      const double py = (sn * xu + cs * yu) * rh + cy;
      const double pz = z * rw;
      res.landmarks[i] = {float(px), float(py), float(pz)};
    }
  });
}

std::optional<FaceResult> FacePipeline::process(const cv::Mat& bgr,
                                                StageTimes& st) {
  const int w = bgr.cols, h = bgr.rows;

  cv::Mat rgb;
  st.det_pre += timed([&] {
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  });

  const auto dets = detect(rgb, st);

  // Biggest face by box area (BiggestFacePipeline).
  const Detection* best = nullptr;
  for (const auto& d : dets) {
    if (!best || d.box[2] * d.box[3] > best->box[2] * best->box[3]) best = &d;
  }
  if (!best) return std::nullopt;

  FaceResult res;
  res.score = best->score;
  res.box = {best->box[0] * w, best->box[1] * h, best->box[2] * w,
             best->box[3] * h};
  runLandmarker(rgb, *best, res, st);
  return res;
}
