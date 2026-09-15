#pragma once
#include <vector>

// SSD anchor centers (x_center, y_center), normalized [0, 1].
// Faithful port of mediapipe SsdAnchorsCalculator (aspect_ratios=[1.0],
// fixed_anchor_size=true, so the scale values only matter through the anchor
// count; w = h = 1.0):
//   * near range / 128: num_layers=4, strides [8,16,16,16], interp_ar=1.0 -> 896
//   * far range  / 192: num_layers=1, strides [4],         interp_ar=0.0 -> 2304
// Both use min_scale=0.1484375, max_scale=0.75.
std::vector<float> gen_ssd_anchors(int input_size, int num_layers,
                                   const int* strides, float min_scale,
                                   float max_scale, float interp_ar);
