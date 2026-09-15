#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "face_pipeline.h"

// Detection box + score + the 14 overlay points (eye boundaries, iris centers,
// mouth corners), mirroring faceframework/annotate.py.
void draw_result(cv::Mat& bgr, const FaceResult& r);

// Multi-line HUD, black outline + green text (app.py _draw_hud).
void draw_hud(cv::Mat& bgr, const std::vector<std::string>& lines);
