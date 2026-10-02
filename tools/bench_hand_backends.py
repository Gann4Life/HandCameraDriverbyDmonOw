"""
Benchmark hand-tracking backends (MediaPipe vs WiLoR) on the same recorded clip.

    # 1. Record a clip with the live camera (same capture path and config as Camera.py)
    python tools/bench_hand_backends.py --record 20

    # 2. Run the backends on it; label the run with the conditions
    python tools/bench_hand_backends.py --label game_off
    python tools/bench_hand_backends.py --label game_vr

WiLoR needs the .venv-wilor environment (see requirements-wilor.txt); in the
plain .venv only MediaPipe runs.

Jitter is the RMS of the second-difference residual p[t] - (p[t-1] + p[t+1]) / 2,
which smooth hand motion barely moves, so it isolates frame-to-frame noise
without needing the hand to be perfectly still.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
import types

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from utils.camera_utils import CameraCapture  # noqa: E402

DEFAULT_CLIP = os.path.join(REPO, "tools", "bench_clip.mp4")
WARMUP_FRAMES = 10
FINGERTIPS = (4, 8, 12, 16, 20)


def load_config():
    with open(os.path.join(REPO, "config.json")) as f:
        return json.load(f)


def record(seconds: float, clip_path: str, config: dict):
    cam_cfg = config["camera"]
    cam = CameraCapture(device_id=cam_cfg["device_id"], width=cam_cfg["width"], height=cam_cfg["height"],
                        fps=cam_cfg["fps"], flip_horizontal=cam_cfg["flip_horizontal"],
                        backend=cam_cfg.get("backend", "auto"), rotate_180=cam_cfg.get("rotate_180", False))
    if not cam.start():
        sys.exit("camera failed to start")
    print("Recording. Suggested: hands still ~5 s, then move them around, then still again. 'q' stops early.")
    frames, start = [], time.time()
    while time.time() - start < seconds:
        ok, frame = cam.read_frame()
        if not ok:
            continue
        frames.append(frame)
        preview = frame.copy()
        cv2.putText(preview, f"REC {time.time() - start:4.1f}/{seconds:.0f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("bench record", preview)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
    fps = len(frames) / max(time.time() - start, 1e-6)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f"Saved {len(frames)} frames ({fps:.1f} fps, {w}x{h}) to {clip_path}")


def read_clip(clip_path: str):
    cap = cv2.VideoCapture(clip_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        sys.exit(f"no frames in {clip_path}; record one first with --record")
    return frames


# ----- Backends: each returns a list of hands per frame, as
# {"kp2d": (21, 2) px, "kp3d": (21, 3) metres, hand-relative, MediaPipe axes}

class MediaPipeBench:
    name = "mediapipe"

    def __init__(self, config):
        import mediapipe as mp
        t = config["tracking"]
        self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=t["max_hands"],
                                              min_detection_confidence=t["detection_confidence"],
                                              min_tracking_confidence=t["tracking_confidence"],
                                              model_complexity=t["model_complexity"])

    def process(self, rgb):
        h, w = rgb.shape[:2]
        res = self.hands.process(rgb)
        out = []
        if res.multi_hand_landmarks and res.multi_hand_world_landmarks:
            for lm, wl in zip(res.multi_hand_landmarks, res.multi_hand_world_landmarks):
                out.append({"kp2d": np.array([(p.x * w, p.y * h) for p in lm.landmark]),
                            "kp3d": np.array([(p.x, p.y, p.z) for p in wl.landmark])})
        return out

    def vram_mb(self):
        return None


class WiLoRBench:
    name = "wilor"

    def __init__(self, config):
        import warnings
        import torch
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
        warnings.filterwarnings("ignore", category=FutureWarning)
        if not torch.cuda.is_available():
            sys.exit("WiLoR benchmark needs CUDA (torch.cuda.is_available() is False)")
        self.torch = torch
        # verbose=False: otherwise YOLO prints a line per frame and the
        # pipeline warns on every empty one, which drowns the console and costs time
        self.pipe = WiLorHandPose3dEstimationPipeline(device=torch.device("cuda"), dtype=torch.float16, verbose=False)
        torch.cuda.reset_peak_memory_stats()

    def process(self, rgb):
        out = []
        for hand in self.pipe.predict(rgb):
            preds = hand.get("wilor_preds")
            if preds is None:
                continue
            kp3d = np.asarray(preds["pred_keypoints_3d"][0], dtype=float)
            out.append({"kp2d": np.asarray(preds["pred_keypoints_2d"][0], dtype=float),
                        "kp3d": kp3d - kp3d.mean(axis=0)})
        self.torch.cuda.synchronize()
        return out

    def vram_mb(self):
        return self.torch.cuda.max_memory_allocated() / 2**20


# ----- Metrics

def make_depth_solver(config, w, h):
    """Reuse Camera.py's solvePnP wrist estimate so depth jitter is measured exactly as tracking sees it."""
    import Camera
    tracker = Camera.HandTracker.__new__(Camera.HandTracker)
    tracker.hfov_deg = float(config["camera"].get("hfov_deg", 70.0))
    tracker.hand_scale = float(config.get("calibration", {}).get("hand_scale", 1.0))
    tracker.mirror_x = False
    tracker.steady_hand_size = False  # raw model output, to compare backends as they are

    def depth(hand):
        lms = types.SimpleNamespace(landmark=[types.SimpleNamespace(x=u / w, y=v / h) for u, v in hand["kp2d"]])
        pos = tracker.estimate_wrist_position(lms, hand["kp3d"].tolist(), w, h)
        return None if pos is None else -pos[2]
    return depth


def link_tracks(per_frame):
    """Follow up to two hands across frames by nearest wrist, so jitter is per hand."""
    tracks = [[None] * len(per_frame) for _ in range(2)]
    last = [None, None]
    for t, hands in enumerate(per_frame):
        free = list(range(len(hands)))
        for k in range(2):
            if not free:
                break
            if last[k] is None:
                j = free[0]
            else:
                j = min(free, key=lambda i: np.linalg.norm(hands[i]["kp2d"][0] - last[k]))
                if np.linalg.norm(hands[j]["kp2d"][0] - last[k]) > 80:  # px: a different hand
                    continue
            tracks[k][t] = hands[j]
            last[k] = hands[j]["kp2d"][0]
            free.remove(j)
    return tracks


def second_diff_rms(series):
    """RMS of x[t] - (x[t-1] + x[t+1]) / 2 over consecutive valid triples."""
    residuals = []
    for a, b, c in zip(series, series[1:], series[2:]):
        if a is None or b is None or c is None:
            continue
        residuals.append(np.linalg.norm(np.asarray(b) - (np.asarray(a) + np.asarray(c)) / 2.0))
    return (float(np.sqrt(np.mean(np.square(residuals)))), len(residuals)) if residuals else (float("nan"), 0)


def gpu_snapshot():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        util, mem = (v.strip() for v in out.stdout.strip().split(","))
        return {"gpu_util_pct": int(util), "gpu_mem_used_mb": int(mem)}
    except Exception:
        return {}


def run_backend(bench, frames, config):
    h, w = frames[0].shape[:2]
    depth = make_depth_solver(config, w, h)
    times, per_frame, gpu_mid = [], [], {}
    for i, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        hands = bench.process(rgb)
        dt = time.perf_counter() - t0
        if i >= WARMUP_FRAMES:
            times.append(dt * 1000)
        per_frame.append(hands)
        if i == len(frames) // 2:
            gpu_mid = gpu_snapshot()

    tracks = link_tracks(per_frame)
    wrist_px, tips_px, depth_mm, samples = [], [], [], 0
    for track in tracks:
        j, n = second_diff_rms([None if h is None else h["kp2d"][0] for h in track])
        if n:
            wrist_px.append(j)
            samples += n
        j, n = second_diff_rms([None if h is None else h["kp2d"][list(FINGERTIPS)].ravel() / math.sqrt(len(FINGERTIPS))
                                for h in track])
        if n:
            tips_px.append(j)
        d = [None if h is None else depth(h) for h in track]
        j, n = second_diff_rms([None if v is None else [v * 1000] for v in d])
        if n:
            depth_mm.append(j)

    t = np.array(times)
    return {
        "backend": bench.name,
        "frames": len(frames),
        "ms_mean": float(t.mean()), "ms_p95": float(np.percentile(t, 95)), "ms_max": float(t.max()),
        "fps": float(1000.0 / t.mean()),
        "detected_pct": 100.0 * sum(1 for f in per_frame if f) / len(per_frame),
        "jitter_wrist_px": float(np.mean(wrist_px)) if wrist_px else float("nan"),
        "jitter_tips_px": float(np.mean(tips_px)) if tips_px else float("nan"),
        "jitter_depth_mm": float(np.mean(depth_mm)) if depth_mm else float("nan"),
        "jitter_samples": samples,
        "torch_peak_vram_mb": bench.vram_mb(),
        **gpu_mid,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", type=float, metavar="SECONDS", help="record a new clip from the camera")
    parser.add_argument("--clip", default=DEFAULT_CLIP)
    parser.add_argument("--backends", default="mediapipe,wilor")
    parser.add_argument("--label", default="run", help="tag for this run's conditions, e.g. game_off / game_vr")
    args = parser.parse_args()
    config = load_config()

    if args.record:
        record(args.record, args.clip, config)
        return

    frames = read_clip(args.clip)
    print(f"Clip: {len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]}  |  label: {args.label}")
    print(f"GPU before: {gpu_snapshot()}")
    results = []
    for name in args.backends.split(","):
        cls = {"mediapipe": MediaPipeBench, "wilor": WiLoRBench}[name.strip()]
        try:
            bench = cls(config)
        except ImportError as e:
            print(f"[{name}] skipped: {e}")
            continue
        print(f"[{name}] running...")
        results.append(run_backend(bench, frames, config))

    cols = [("backend", "{:>10}"), ("fps", "{:7.1f}"), ("ms_mean", "{:8.1f}"), ("ms_p95", "{:7.1f}"),
            ("ms_max", "{:7.1f}"), ("detected_pct", "{:8.0f}%"), ("jitter_wrist_px", "{:9.2f}"),
            ("jitter_tips_px", "{:9.2f}"), ("jitter_depth_mm", "{:9.1f}"), ("torch_peak_vram_mb", "{:>9}"),
            ("gpu_util_pct", "{:>6}")]
    print("\n" + " ".join(f"{c:>{max(7, len(c))}}" for c, _ in cols))
    for r in results:
        cells = []
        for c, fmt in cols:
            v = r.get(c)
            if v is None:
                cells.append(f"{'-':>{max(7, len(c))}}")
            elif isinstance(v, float) and c == "torch_peak_vram_mb":
                cells.append(f"{v:>{max(7, len(c))}.0f}")
            else:
                cells.append(f"{fmt.format(v):>{max(7, len(c))}}")
        print(" ".join(cells))

    out = os.path.join(REPO, "tools", f"bench_results_{args.label}.json")
    with open(out, "w") as f:
        json.dump({"label": args.label, "results": results}, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
