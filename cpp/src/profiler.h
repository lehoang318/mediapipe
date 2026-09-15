#pragma once
#include <string>
#include <vector>

#include "face_pipeline.h"

// Records per-frame stage timings and reports warm-up timing plus 10-bin
// histograms of the stable phase.
class Profiler {
 public:
  void add(int frame_idx, bool warmup, const StageTimes& st, int n_faces,
           float best_score, float presence);

  // Prints warm-up timing and per-stage 10-bin histograms of the stable phase.
  void report() const;

 private:
  struct Row {
    bool warmup;
    int frame;
    StageTimes st;
    int n_faces;
    float score;
    float presence;
  };

  std::vector<Row> rows_;
};
