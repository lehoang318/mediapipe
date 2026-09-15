#include "anchors.h"

#include <cmath>

std::vector<float> gen_ssd_anchors(int input_size, int num_layers,
                                   const int* strides, float min_scale,
                                   float max_scale, float interp_ar) {
  auto calc_scale = [&](int i) {
    if (num_layers == 1) return (min_scale + max_scale) * 0.5f;
    return min_scale + (max_scale - min_scale) * i / (num_layers - 1);
  };

  std::vector<float> anchors;
  int layer_id = 0;
  while (layer_id < num_layers) {
    // Scales for this run of equal strides (aspect_ratios = [1.0]).
    std::vector<float> scales;
    int last_same = layer_id;
    while (last_same < num_layers && strides[last_same] == strides[layer_id]) {
      const float scale = calc_scale(last_same);
      scales.push_back(scale);
      if (interp_ar > 0.0f) {
        const float scale_next =
            (last_same == num_layers - 1) ? 1.0f : calc_scale(last_same + 1);
        scales.push_back(std::sqrt(scale * scale_next));
      }
      ++last_same;
    }

    const int stride = strides[layer_id];
    const int fm = static_cast<int>(std::ceil(input_size / float(stride)));
    for (int y = 0; y < fm; ++y) {
      for (int x = 0; x < fm; ++x) {
        const float xc = (x + 0.5f) / fm;
        const float yc = (y + 0.5f) / fm;
        for (size_t s = 0; s < scales.size(); ++s) {
          (void)s;  // fixed_anchor_size: w = h = 1.0, not stored
          anchors.push_back(xc);
          anchors.push_back(yc);
        }
      }
    }
    layer_id = last_same;
  }
  return anchors;
}
