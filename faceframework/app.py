"""Display loop: read frame -> biggest face + landmarks -> annotate -> show.

Controls: q / ESC quit, space pause/resume.
"""

from __future__ import annotations

import time

import cv2

from .annotate import draw


def _draw_hud(image, src_fps, proc_fps, score, frame_idx, paused):
    lines = [
        f"src {src_fps:.1f} fps | proc {proc_fps:4.1f} fps | frame {frame_idx}",
        f"face score {score:.2f}" if score is not None else "no face",
    ]
    if paused:
        lines.append("PAUSED (space to resume)")
    for i, text in enumerate(lines):
        org = (8, 20 + i * 18)
        cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def _handle_keys(delay_ms, state):
    """Return False to quit. Blocks while paused so the loop stays frozen."""
    while True:
        key = cv2.waitKey(max(1, delay_ms)) & 0xFF
        if key in (ord("q"), 27):
            return False
        if key == ord(" "):
            state["paused"] = not state["paused"]
        if not state["paused"]:
            return True


def run(source, pipeline, cfg) -> None:
    window = cfg.display.window
    print(f"[framework] source: {source.describe()}")
    print(f"[framework] onnxruntime providers: {pipeline.providers}")

    state = {"paused": False}
    frame_idx = 0
    n_with_face = 0
    score_sum = 0.0
    proc_fps = 0.0
    last = None
    last_score = None

    try:
        while True:
            if not state["paused"]:
                ok, frame = source.read()
                if not ok:
                    break
                frame_idx += 1
                t0 = time.perf_counter()
                res = pipeline.process(frame)
                dt = time.perf_counter() - t0
                inst = 1.0 / dt if dt > 0 else 0.0
                proc_fps = inst if proc_fps == 0.0 else 0.9 * proc_fps + 0.1 * inst
                if res is not None:
                    n_with_face += 1
                    score_sum += res.score
                    last_score = res.score
                    last = draw(frame, res)
                else:
                    last_score = None
                    last = frame

            if last is not None:
                _draw_hud(last, source.fps, proc_fps, last_score, frame_idx,
                          state["paused"])
                cv2.imshow(window, last)
                if not _handle_keys(source.delay_ms, state):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        source.release()
        cv2.destroyAllWindows()

    mean_score = score_sum / n_with_face if n_with_face else 0.0
    print(f"[framework] frames {frame_idx}, with face {n_with_face}, "
          f"avg {proc_fps:.1f} proc fps, mean det score {mean_score:.3f}")
