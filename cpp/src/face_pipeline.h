#pragma once
#include <onnxruntime_cxx_api.h>

#include <opencv2/core.hpp>

#include <optional>
#include <string>
#include <vector>

// Wall-clock duration (ms) of every pipeline stage for one frame.
struct StageTimes {
  double det_pre = 0.0;    // BGR->RGB, letterbox to det_size, /127.5-1
  double det_infer = 0.0;  // detector Run()
  double det_post = 0.0;   // anchor decode, sigmoid, threshold, weighted NMS, project
  double lmk_pre = 0.0;    // ROI + rotate/crop 256x256, /255
  double lmk_infer = 0.0;  // landmarker Run()
  double lmk_post = 0.0;   // presence, letterbox removal, projection
  double total = 0.0;      // filled by the caller (end-to-end wall time)
};

struct Detection {
  float score = 0.f;
  float box[4] = {0.f, 0.f, 0.f, 0.f};  // normalized x, y, w, h in the full image
  std::vector<cv::Point2f> keypoints;   // 6, normalized
};

struct FaceResult {
  float score = 0.f;
  float presence = 0.f;  // sigmoid of the landmarker presence logit
  cv::Vec4f box{0.f, 0.f, 0.f, 0.f};    // px x, y, w, h
  std::vector<cv::Point3f> landmarks;   // 478 x, y, z in px
};

// Biggest-face detection + 478-point landmarks, mirroring
// faceframework/pipeline.py (BiggestFacePipeline over face_pipeline.py).
// The detector variant (near 128 / far 192) is inferred from the ONNX input
// shape, which also selects the anchor set (896 / 2304).
class FacePipeline {
 public:
  FacePipeline(const std::string& detector_onnx, const std::string& landmarker_onnx,
               float min_score, float nms_iou, bool use_cpu);

  // Detects all faces with score >= min_score, keeps the largest by box area
  // and runs the landmarker on it. Returns nullopt when no face is found.
  std::optional<FaceResult> process(const cv::Mat& bgr, StageTimes& st);

  const std::string& providerDescription() const { return provider_desc_; }

 private:
  std::vector<Detection> detect(const cv::Mat& rgb, StageTimes& st);
  void runLandmarker(const cv::Mat& rgb, const Detection& det, FaceResult& res,
                     StageTimes& st);

  Ort::Env env_;
  Ort::Session det_session_{nullptr};
  Ort::Session lmk_session_{nullptr};
  std::string det_in_name_, lmk_in_name_;
  std::vector<std::string> det_out_names_, lmk_out_names_;
  Ort::MemoryInfo mem_info_{nullptr};

  int det_size_ = 0;            // 128 (near) or 192 (far), from the ONNX input
  size_t num_anchors_ = 0;      // 896 / 2304
  std::vector<float> anchors_;  // x_center, y_center pairs, normalized
  float min_score_, nms_iou_;
  std::string provider_desc_;
};
