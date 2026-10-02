"""
Runs the HandTracker on its own thread, so the interface never slows tracking
and tracking never freezes the interface.
"""
import copy
import queue
import threading
import time
import traceback
from typing import Any, Dict, Optional, Tuple

from PySide6.QtCore import QThread, Signal

from Camera import CameraLostError, HandTracker
from hand_data import TrackingFrame


class TrackerWorker(QThread):
    """
    Owns one tracking session. The GUI polls latest() for the newest frame and
    sends setting changes with apply_settings(); they run between frames.
    """

    tracking_started = Signal()
    tracking_stopped = Signal(str)  # why it stopped, empty when asked to
    settings_applied = Signal(dict)  # values in effect after a change

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config = copy.deepcopy(config)
        self._changes: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._running = False
        self._lock = threading.Lock()
        self._latest: Optional[TrackingFrame] = None
        self._sequence = 0

    def latest(self) -> Tuple[int, Optional[TrackingFrame]]:
        """The newest frame and its sequence number, to skip frames already shown."""
        with self._lock:
            return self._sequence, self._latest

    def apply_settings(self, changes: Dict[str, Any]):
        self._changes.put(dict(changes))

    def request_stop(self):
        self._running = False

    def run(self):
        self._running = True
        try:
            tracker = HandTracker(config=self._config)
        except Exception as e:
            traceback.print_exc()
            self.tracking_stopped.emit(f"Could not start tracking: {e}")
            return
        if not tracker.start():
            tracker.stop()
            self.tracking_stopped.emit("Could not open the camera. Check Camera index in Settings.")
            return

        self.tracking_started.emit()
        reason = ""
        try:
            while self._running:
                self._apply_pending(tracker)
                frame = tracker.step()
                if frame is None:
                    continue
                with self._lock:
                    self._latest = frame
                    self._sequence += 1
                tracker.report_slow_frame(*tracker.last_timings, time.perf_counter())
        except CameraLostError as e:
            reason = str(e)
        except Exception as e:
            traceback.print_exc()
            reason = f"Tracking stopped by an error: {e}"
        finally:
            tracker.stop()
        self.tracking_stopped.emit(reason)

    def _apply_pending(self, tracker: HandTracker):
        changes = {}
        while True:
            try:
                changes.update(self._changes.get_nowait())
            except queue.Empty:
                break
        if not changes:
            return
        try:
            self.settings_applied.emit(tracker.apply_settings(changes))
        except Exception as e:
            traceback.print_exc()
            print(f"Could not apply {', '.join(changes)}: {e}")
