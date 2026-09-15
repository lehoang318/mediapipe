// Face framework C++ port: biggest-face detection + 478-point landmarks on a
// video file or camera with ONNX Runtime (GPU when available). Mirrors
// main.py + faceframework (app / pipeline / sources / annotate).
//
//   face_video --pretrained DIR (--video PATH | --camera INDEX) [--far]
//       [--min-score F] [--nms-iou F] [--window TEXT]
//       [--cpu] [--no-display] [--max-frames N]
//
// The literal first frame is the warm-up run; all later frames form the
// stable phase. On exit the warm-up timing and a 10-bin histogram per stage
// are printed, followed by the run summary.

#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "draw.h"
#include "face_pipeline.h"
#include "profiler.h"

namespace {

// Default fps handling, matching config.json defaults: video files fall back
// to 30 fps when the container fps is missing/absurd, cameras are paced at
// 30 fps (pushed to the device best effort).
constexpr double kFallbackFps = 30.0;
constexpr double kCameraFps = 30.0;

struct Options {
  std::string pretrained;
  std::string video;
  int camera = -1;
  bool have_camera = false;
  bool far = false;
  float min_score = 0.7f;
  float nms_iou = 0.5f;
  std::string window = "face_framework (q quit | space pause)";
  bool use_cpu = false;
  bool no_display = false;
  long max_frames = 0;
};

void usage(const char* argv0) {
  std::cerr
      << "usage: " << argv0
      << " --pretrained DIR (--video PATH | --camera INDEX) [options]\n"
      << "  --pretrained DIR    directory with the onnx models (required)\n"
      << "  --video PATH        input video file\n"
      << "  --camera INDEX      camera device index (takes priority over --video)\n"
      << "  --far               use the far-range (full-range) detector\n"
      << "  --min-score F       detector score threshold (default 0.7)\n"
      << "  --nms-iou F         NMS IoU threshold (default 0.5)\n"
      << "  --window TEXT       preview window title\n"
      << "  --cpu               use CPUExecutionProvider even if CUDA is available\n"
      << "  --no-display        run without a window (headless benchmark)\n"
      << "  --max-frames N      stop after N frames (0 = whole video)\n";
}

// Frame source abstraction mirroring faceframework/sources.py.
class Source {
 public:
  virtual ~Source() = default;
  virtual bool read(cv::Mat& frame) = 0;
  virtual double fps() const = 0;
  virtual std::string describe() const = 0;
  virtual void release() = 0;

  int delayMs() const {
    const double f = fps();
    if (f <= 0.0) return 1;
    return std::max(1, static_cast<int>(std::lround(1000.0 / f)));
  }
};

double deduce_fps(const cv::VideoCapture& cap, double fallback, bool& used_fallback) {
  const double fps = cap.get(cv::CAP_PROP_FPS);
  const bool ok = std::isfinite(fps) && 1.0 <= fps && fps <= 1000.0;
  used_fallback = !ok;
  return ok ? fps : fallback;
}

class VideoFileSource : public Source {
 public:
  VideoFileSource(const std::string& path, double fallback_fps) : path_(path) {
    cap_.open(path);
    if (!cap_.isOpened()) throw std::runtime_error("cannot open video " + path);
    fps_ = deduce_fps(cap_, fallback_fps, used_fallback_);
    fallback_ = fallback_fps;
  }

  bool read(cv::Mat& frame) override { return cap_.read(frame); }
  double fps() const override { return fps_; }
  void release() override { cap_.release(); }

  std::string describe() const override {
    std::ostringstream os;
    os << "video " << path_ << "  fps " << fps_;
    if (used_fallback_) os << " (fallback " << fallback_ << ")";
    return os.str();
  }

 private:
  std::string path_;
  cv::VideoCapture cap_;
  double fps_ = 0.0, fallback_ = 0.0;
  bool used_fallback_ = false;
};

class CameraSource : public Source {
 public:
  CameraSource(int index, double fps) : index_(index), fps_(fps) {
    cap_.open(index);
    if (!cap_.isOpened())
      throw std::runtime_error("cannot open camera " + std::to_string(index));
    cap_.set(cv::CAP_PROP_FPS, fps);
    device_fps_ = cap_.get(cv::CAP_PROP_FPS);
  }

  bool read(cv::Mat& frame) override { return cap_.read(frame); }
  double fps() const override { return fps_; }
  void release() override { cap_.release(); }

  std::string describe() const override {
    std::ostringstream os;
    os << "camera " << index_ << "  fps " << fps_ << " (config)";
    if (device_fps_) os << "  device reports " << device_fps_ << " fps";
    return os.str();
  }

 private:
  int index_;
  cv::VideoCapture cap_;
  double fps_, device_fps_ = 0.0;
};

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need_value = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "error: " << name << " requires a value\n";
        std::exit(1);
      }
      return argv[++i];
    };
    if (a == "--pretrained") opt.pretrained = need_value("--pretrained");
    else if (a == "--video") opt.video = need_value("--video");
    else if (a == "--camera") {
      opt.camera = std::stoi(need_value("--camera"));
      opt.have_camera = true;
    } else if (a == "--far") opt.far = true;
    else if (a == "--min-score") opt.min_score = std::stof(need_value("--min-score"));
    else if (a == "--nms-iou") opt.nms_iou = std::stof(need_value("--nms-iou"));
    else if (a == "--window") opt.window = need_value("--window");
    else if (a == "--cpu") opt.use_cpu = true;
    else if (a == "--no-display") opt.no_display = true;
    else if (a == "--max-frames") opt.max_frames = std::stol(need_value("--max-frames"));
    else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
    else { std::cerr << "error: unknown option " << a << "\n"; usage(argv[0]); return 1; }
  }
  if (opt.pretrained.empty()) {
    std::cerr << "error: --pretrained is required\n";
    usage(argv[0]);
    return 1;
  }
  if (!opt.have_camera && opt.video.empty()) {
    std::cerr << "error: one of --video or --camera is required\n";
    usage(argv[0]);
    return 1;
  }
  if (opt.have_camera && !opt.video.empty()) {
    std::cout << "[framework] both --video and --camera given; using --camera (priority)\n";
  }

  try {
    const char* det_name =
        opt.far ? "blaze_face_full_range.onnx" : "blaze_face_short_range.onnx";
    FacePipeline pipeline(opt.pretrained + "/" + det_name,
                          opt.pretrained + "/face_landmarks_detector.onnx",
                          opt.min_score, opt.nms_iou, opt.use_cpu);
    std::cout << "[framework] detector: " << (opt.far ? "far" : "near")
              << "-range (" << det_name << ")\n";

    std::unique_ptr<Source> source;
    if (opt.have_camera)
      source = std::make_unique<CameraSource>(opt.camera, kCameraFps);
    else
      source = std::make_unique<VideoFileSource>(opt.video, kFallbackFps);
    std::cout << "[framework] source: " << source->describe() << "\n";
    std::cout << "[framework] onnxruntime providers: "
              << pipeline.providerDescription() << "\n";

    Profiler profiler;
    long frame_idx = 0, n_with_face = 0;
    double score_sum = 0.0, proc_fps = 0.0;
    bool paused = false, quit = false, have_last = false, last_has_face = false;
    float last_score = 0.f;
    cv::Mat last;

    while (!quit) {
      if (!paused) {
        cv::Mat frame;
        if (!source->read(frame)) break;
        ++frame_idx;
        const auto t0 = std::chrono::steady_clock::now();
        StageTimes st;
        const auto res = pipeline.process(frame, st);
        st.total = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t0)
                       .count();

        if (res) {
          ++n_with_face;
          score_sum += res->score;
          last_has_face = true;
          last_score = res->score;
          draw_result(frame, *res);
        } else {
          last_has_face = false;
        }
        // The capture buffer is reused on the next read, so keep our own copy
        // for the (possibly paused) display.
        frame.copyTo(last);
        have_last = true;

        if (st.total > 0) {
          const double inst = 1000.0 / st.total;
          proc_fps = proc_fps == 0.0 ? inst : 0.9 * proc_fps + 0.1 * inst;
        }

        profiler.add(int(frame_idx), frame_idx == 1, st, res ? 1 : 0,
                     res ? res->score : 0.f, res ? res->presence : 0.f);

        if (opt.max_frames && frame_idx >= opt.max_frames) break;
      }

      if (!opt.no_display && have_last) {
        std::ostringstream line1;
        line1 << std::fixed << std::setprecision(1) << "src " << source->fps()
              << " fps | proc " << proc_fps << " fps | frame " << frame_idx;
        std::vector<std::string> lines;
        lines.push_back(line1.str());
        if (last_has_face) {
          char buf[32];
          std::snprintf(buf, sizeof buf, "face score %.2f", last_score);
          lines.push_back(buf);
        } else {
          lines.push_back("no face");
        }
        if (paused) lines.push_back("PAUSED (space to resume)");
        draw_hud(last, lines);
        cv::imshow(opt.window, last);

        // app.py _handle_keys: block while paused, q/ESC quits, space toggles.
        while (true) {
          const int key = cv::waitKey(source->delayMs()) & 0xff;
          if (key == 'q' || key == 27) { quit = true; break; }
          if (key == ' ') paused = !paused;
          if (!paused) break;
        }
      }
    }

    profiler.report();
    const double mean_score = n_with_face ? score_sum / double(n_with_face) : 0.0;
    std::cout << "\n[framework] frames " << frame_idx << ", with face "
              << n_with_face << ", avg " << std::fixed << std::setprecision(1)
              << proc_fps << " proc fps, mean det score "
              << std::setprecision(3) << mean_score << "\n";

    source->release();
    cv::destroyAllWindows();
  } catch (const Ort::Exception& e) {
    std::cerr << "onnxruntime error: " << e.what() << "\n";
    return 1;
  } catch (const cv::Exception& e) {
    std::cerr << "opencv error: " << e.what() << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
