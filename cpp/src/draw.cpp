#include "draw.h"

#include <opencv2/imgproc.hpp>

#include <cstdio>

namespace {

constexpr int kRightEye[] = {33, 133, 159, 145};   // outer, inner, upper, lower
constexpr int kLeftEye[] = {263, 362, 386, 374};   // outer, inner, upper, lower
constexpr int kRightIris[] = {468};                // right iris center
constexpr int kLeftIris[] = {473};                 // left iris center
constexpr int kMouth[] = {61, 291, 0, 17};         // corners + top/bottom center

const cv::Scalar kBoxColor(0, 200, 0);        // green
const cv::Scalar kEyeColor(0, 200, 255);      // amber
const cv::Scalar kRightIrisColor(0, 0, 255);  // red
const cv::Scalar kLeftIrisColor(255, 0, 0);   // blue
const cv::Scalar kLipColor(255, 0, 200);      // pink

void draw_points(cv::Mat& bgr, const FaceResult& r, const int* idx, size_t n,
                 const cv::Scalar& color) {
  for (size_t i = 0; i < n; ++i) {
    const auto& lm = r.landmarks.at(size_t(idx[i]));
    cv::circle(bgr, {cvRound(lm.x), cvRound(lm.y)}, 2, color, -1);
  }
}

}  // namespace

void draw_result(cv::Mat& bgr, const FaceResult& r) {
  const int x = cvRound(r.box[0]), y = cvRound(r.box[1]);
  const int bw = cvRound(r.box[2]), bh = cvRound(r.box[3]);
  cv::rectangle(bgr, {x, y}, {x + bw, y + bh}, kBoxColor, 2);

  char score[16];
  std::snprintf(score, sizeof score, "%.2f", r.score);
  cv::putText(bgr, score, {x, std::max(12, y - 6)}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
              kBoxColor, 1);

  if (r.landmarks.size() < 478) return;
  draw_points(bgr, r, kRightEye, 4, kEyeColor);
  draw_points(bgr, r, kLeftEye, 4, kEyeColor);
  draw_points(bgr, r, kMouth, 4, kLipColor);
  draw_points(bgr, r, kRightIris, 1, kRightIrisColor);
  draw_points(bgr, r, kLeftIris, 1, kLeftIrisColor);
}

void draw_hud(cv::Mat& bgr, const std::vector<std::string>& lines) {
  for (size_t i = 0; i < lines.size(); ++i) {
    const cv::Point org(8, 20 + static_cast<int>(i) * 18);
    cv::putText(bgr, lines[i], org, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 3);
    cv::putText(bgr, lines[i], org, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 255, 0}, 1);
  }
}
