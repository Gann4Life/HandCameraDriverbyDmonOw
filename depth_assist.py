"""
Asynchronous WiLoR depth for the MediaPipe pipeline.

WiLoR's 3D hand mesh gives far steadier depth than MediaPipe, but on a mid-range
GPU it runs at ~9 fps, too slow to drive tracking on its own. Here it runs on a
background thread over the newest frame it can get, and only its per-hand
wrist distance is used; MediaPipe keeps supplying everything else at full rate.

EXPERIMENTAL / NON-COMMERCIAL: WiLoR (CC BY-NC-ND), the MANO hand model
(non-commercial, non-redistributable) and the ultralytics YOLO detector
(AGPL-3.0) are pulled in here. See requirements-wilor.txt.
"""
import threading
import time
import types
from typing import Callable, List, Optional, Tuple

import numpy as np

# (normalised wrist x, y, distance from the camera in metres)
DepthResult = Tuple[float, float, float]


class WiLoRDepthAssist:
    """Background WiLoR worker that keeps the newest per-hand wrist distances."""

    def __init__(self, estimate_wrist: Callable, max_rate_hz: float = 10.0):
        """
        Args:
            estimate_wrist: Camera.HandTracker.estimate_wrist_position, used so
                WiLoR depth comes from the same solvePnP and camera calibration
                as MediaPipe's
            max_rate_hz: Upper bound on WiLoR runs per second, to leave GPU time
                for the VR game
        """
        self.estimate_wrist = estimate_wrist
        self.min_interval = 1.0 / max(0.5, max_rate_hz)
        self.available = False
        self.error: Optional[str] = None
        self.rate_hz = 0.0
        self._pipe = None
        self._frame = None
        self._frame_time = 0.0
        self._results: List[DepthResult] = []
        self._results_time = 0.0
        self._wake = threading.Condition()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """
        Load WiLoR (on the GPU, fp16) and start the worker.

        Returns:
            True if WiLoR is running; otherwise self.error says why
        """
        if self._running:
            return True
        try:
            import warnings
            import torch
            from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
            warnings.filterwarnings("ignore", category=FutureWarning)
            if not torch.cuda.is_available():
                self.error = "CUDA is not available"
                return False
            self._torch = torch
            self._pipe = WiLorHandPose3dEstimationPipeline(device=torch.device("cuda"), dtype=torch.float16,
                                                           verbose=False)
        except ImportError as e:
            self.error = f"WiLoR is not installed in this environment ({e}); run from .venv-wilor"
            return False
        except Exception as e:  # model download / load failures
            self.error = f"WiLoR failed to load: {e}"
            return False
        self.available = True
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        with self._wake:
            self._wake.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def submit(self, frame_rgb: np.ndarray, t: float):
        """Offer the current frame; the worker takes the newest one when it is free."""
        with self._wake:
            self._frame, self._frame_time = frame_rgb, t
            self._wake.notify()

    def latest(self) -> Tuple[float, List[DepthResult]]:
        """(time of the frame they came from, per-hand results)."""
        with self._wake:
            return self._results_time, list(self._results)

    def _loop(self):
        last_run = 0.0
        window_start, window_runs = time.perf_counter(), 0
        while self._running:
            with self._wake:
                while self._running and self._frame is None:
                    self._wake.wait(0.5)
                if not self._running:
                    break
                frame, frame_time = self._frame, self._frame_time
                self._frame = None

            # Throttle to max_rate_hz so the VR game keeps its GPU headroom
            wait = self.min_interval - (time.perf_counter() - last_run)
            if wait > 0:
                time.sleep(wait)
            last_run = time.perf_counter()

            try:
                results = self._process(frame)
            except Exception as e:
                self.error = f"WiLoR inference failed: {e}"
                continue
            with self._wake:
                self._results, self._results_time = results, frame_time

            window_runs += 1
            elapsed = time.perf_counter() - window_start
            if elapsed > 1.0:
                self.rate_hz = window_runs / elapsed
                window_start, window_runs = time.perf_counter(), 0

    def _process(self, frame_rgb: np.ndarray) -> List[DepthResult]:
        h, w = frame_rgb.shape[:2]
        results = []
        for hand in self._pipe.predict(frame_rgb):
            preds = hand.get("wilor_preds")
            if preds is None:
                continue
            kp2d = np.asarray(preds["pred_keypoints_2d"][0], dtype=float)
            kp3d = np.asarray(preds["pred_keypoints_3d"][0], dtype=float)
            kp3d -= kp3d.mean(axis=0)
            landmarks = types.SimpleNamespace(
                landmark=[types.SimpleNamespace(x=u / w, y=v / h) for u, v in kp2d])
            position = self.estimate_wrist(landmarks, kp3d.tolist(), w, h)
            if position is not None:
                results.append((kp2d[0][0] / w, kp2d[0][1] / h, -position[2]))
        return results
