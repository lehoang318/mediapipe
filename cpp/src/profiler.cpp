#include "profiler.h"

#include <algorithm>
#include <iomanip>
#include <iostream>

namespace {

enum Stage { kDetPre, kDetInfer, kDetPost, kLmkPre, kLmkInfer, kLmkPost, kTotal, kNumStages };

const char* kStageNames[kNumStages] = {
    "det_pre", "det_infer", "det_post", "lmk_pre", "lmk_infer", "lmk_post", "total"};

double get(const StageTimes& st, int stage) {
  switch (stage) {
    case kDetPre: return st.det_pre;
    case kDetInfer: return st.det_infer;
    case kDetPost: return st.det_post;
    case kLmkPre: return st.lmk_pre;
    case kLmkInfer: return st.lmk_infer;
    case kLmkPost: return st.lmk_post;
    default: return st.total;
  }
}

}  // namespace

void Profiler::add(int frame_idx, bool warmup, const StageTimes& st, int n_faces,
                   float best_score, float presence) {
  rows_.push_back({warmup, frame_idx, st, n_faces, best_score, presence});
}

void Profiler::report() const {
  std::vector<Row> warmup, stable;
  for (const auto& r : rows_) (r.warmup ? warmup : stable).push_back(r);

  std::cout << "\n== warm-up (" << warmup.size() << " frame(s))\n";
  for (const auto& r : warmup) {
    std::cout << "  frame " << r.frame << ":";
    for (int s = 0; s < kNumStages; ++s)
      std::cout << ' ' << kStageNames[s] << ' ' << std::fixed << std::setprecision(2)
                << get(r.st, s) << " ms";
    std::cout << '\n';
  }
  if (warmup.empty()) std::cout << "  (none)\n";

  std::cout << "\n== stable phase (" << stable.size() << " frames), per-stage latency\n";
  if (stable.empty()) {
    std::cout << "  (no stable frames)\n";
    return;
  }
  std::cout << "  " << std::left << std::setw(10) << "stage" << std::right
            << std::setw(7) << "n" << std::setw(9) << "mean" << std::setw(9) << "p50"
            << std::setw(9) << "p95" << std::setw(9) << "min" << std::setw(9) << "max"
            << "   (ms)\n";

  for (int s = 0; s < kNumStages; ++s) {
    std::vector<double> v;
    v.reserve(stable.size());
    double sum = 0.0;
    for (const auto& r : stable) {
      v.push_back(get(r.st, s));
      sum += v.back();
    }
    std::sort(v.begin(), v.end());
    const auto pct = [&](double p) {
      const size_t i = std::min(v.size() - 1, size_t(p * double(v.size())));
      return v[i];
    };
    const double lo = v.front(), hi = v.back();
    std::cout << "  " << std::left << std::setw(10) << kStageNames[s] << std::right
              << std::setw(7) << v.size() << std::setw(9) << std::fixed
              << std::setprecision(3) << sum / v.size() << std::setw(9) << pct(0.50)
              << std::setw(9) << pct(0.95) << std::setw(9) << lo << std::setw(9) << hi
              << '\n';

    // 10-bin histogram between min and max of the stable phase.
    const double width = (hi - lo) / 10.0;
    int bins[10] = {};
    for (double x : v) {
      int b = width > 0 ? int((x - lo) / width) : 0;
      b = std::clamp(b, 0, 9);
      ++bins[b];
    }
    std::cout << "    bins(ms): ";
    const int prec = width < 0.05 ? 3 : 2;
    for (int b = 0; b < 10; ++b)
      std::cout << (b ? " | " : "") << std::fixed << std::setprecision(prec)
                << lo + b * width << '-' << lo + (b + 1) * width << ':' << bins[b];
    std::cout << '\n';
  }
}
