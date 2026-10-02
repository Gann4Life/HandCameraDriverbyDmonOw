"""
Gesture calibration: the user holds their hands open, then closed into fists,
while tracking runs. How far each finger bends in either pose, as the camera
sees it from where it is, becomes the 0 and 1 of that finger's curl, so
trigger, grip and gestures fit the user's hands and the active preset's view.
"""
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QLabel, QProgressBar, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout)

from hand_data import TrackingFrame
from hand_features import FINGERS

READY_SECONDS = 2.0
RECORD_SECONDS = 3.0
MIN_SAMPLES = 30          # hand-frames per pose
OPEN_PERCENTILE = 90      # most open frames read as fully open...
OPEN_MARGIN_DEG = 3.0
FIST_PERCENTILE = 40      # ...and most fist frames as fully curled
MIN_RANGE_DEG = 25.0      # less than this between open and fist: the view can't tell them apart

POSES = (
    ("open", "Hold both hands open, fingers straight and relaxed, where you normally use them."),
    ("fist", "Now close both hands into fists, and keep them there."),
)


def curl_range(open_samples: Sequence[Sequence[float]], fist_samples: Sequence[Sequence[float]]
               ) -> Tuple[List[float], List[float], List[str]]:
    """
    Open and fully curled bend per finger from the recorded poses.

    Args:
        open_samples: curl_deg per hand-frame with the hands open
        fist_samples: curl_deg per hand-frame with fists

    Returns:
        (open_deg, full_deg, fingers whose range was too small and was widened)
    """
    open_arr, fist_arr = np.asarray(open_samples, dtype=float), np.asarray(fist_samples, dtype=float)
    open_deg = np.percentile(open_arr, OPEN_PERCENTILE, axis=0) + OPEN_MARGIN_DEG
    full_deg = np.percentile(fist_arr, FIST_PERCENTILE, axis=0)
    narrow = [finger for finger, o, f in zip(FINGERS, open_deg, full_deg) if f - o < MIN_RANGE_DEG]
    full_deg = np.maximum(full_deg, open_deg + MIN_RANGE_DEG)
    return [round(float(v), 1) for v in open_deg], [round(float(v), 1) for v in full_deg], narrow


class GestureCalibrationDialog(QDialog):
    def __init__(self, latest: Callable[[], Tuple[int, Optional[TrackingFrame]]],
                 current: Tuple[Sequence[float], Sequence[float]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibrate gestures")
        self._latest = latest
        self._current = current
        self._samples: Dict[str, List[Tuple[float, ...]]] = {pose: [] for pose, _ in POSES}
        self._step = 0
        self._elapsed = 0.0
        self._recording = False
        self._last_sequence = -1
        self.changes: Dict[str, List[float]] = {}

        self.instructions = QLabel()
        self.instructions.setWordWrap(True)
        self.status = QLabel()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.table = QTableWidget(len(FINGERS), 4)
        self.table.setHorizontalHeaderLabels(["Open now", "Open new", "Curled now", "Curled new"])
        self.table.setVerticalHeaderLabels([f.capitalize() for f in FINGERS])
        self.table.setVisible(False)
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self._start_pose)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Apply).setEnabled(False)
        self.buttons.button(QDialogButtonBox.Apply).clicked.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Keep the camera where you use it. Each pose takes a few seconds; "
                                "the result goes into the active preset."))
        for widget in (self.instructions, self.start_button, self.progress, self.status, self.table, self.buttons):
            layout.addWidget(widget)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._show_step()

    def _show_step(self):
        pose, text = POSES[self._step]
        self.instructions.setText(f"<b>{self._step + 1}/{len(POSES)}</b> {text}")
        self.progress.setValue(0)
        self.start_button.setText("Start")
        self.start_button.setEnabled(True)

    def _start_pose(self):
        self._samples[POSES[self._step][0]] = []
        self._elapsed = 0.0
        self._recording = False
        self.start_button.setEnabled(False)
        self._timer.start(33)

    def _tick(self):
        self._elapsed += self._timer.interval() / 1000.0
        pose = POSES[self._step][0]
        if self._elapsed < READY_SECONDS:
            self.status.setText(f"Get ready... {READY_SECONDS - self._elapsed:.0f}")
            return
        recorded = self._elapsed - READY_SECONDS
        sequence, frame = self._latest()
        if frame is not None and sequence != self._last_sequence:
            self._last_sequence = sequence
            for hand in frame.hands:
                if hand.features is not None:
                    self._samples[pose].append(tuple(hand.features.curl_deg))
        count = len(self._samples[pose])
        self.progress.setValue(min(100, int(recorded / RECORD_SECONDS * 100)))
        self.status.setText(f"Recording... {count} hand frames")
        if recorded < RECORD_SECONDS:
            return
        self._timer.stop()
        if count < MIN_SAMPLES:
            self.status.setText(f"Only {count} hand frames: keep your hands in view of the camera and try again.")
            self.start_button.setText("Retry")
            self.start_button.setEnabled(True)
            return
        if self._step + 1 < len(POSES):
            self._step += 1
            self.status.setText("")
            self._show_step()
        else:
            self._finish()

    def _finish(self):
        open_deg, full_deg, narrow = curl_range(self._samples["open"], self._samples["fist"])
        self.changes = {"gestures.curl_open_deg": open_deg, "gestures.curl_full_deg": full_deg}
        for row in range(len(FINGERS)):
            for col, value in enumerate((self._current[0][row], open_deg[row], self._current[1][row], full_deg[row])):
                self.table.setItem(row, col, QTableWidgetItem(f"{float(value):.0f}°"))
        self.table.setVisible(True)
        message = "Done. Apply puts these in use; Save keeps them in the preset."
        if narrow:
            message += (f"\nThe camera barely sees {', '.join(narrow)} bend from here, so "
                        f"{'it stays' if len(narrow) == 1 else 'they stay'} less reliable.")
        self.status.setText(message)
        self.instructions.setText("Calibration recorded.")
        self.start_button.setText("Start over")
        self.start_button.setEnabled(True)
        self.start_button.clicked.disconnect()
        self.start_button.clicked.connect(self._restart)
        self.buttons.button(QDialogButtonBox.Apply).setEnabled(True)

    def _restart(self):
        self.start_button.clicked.disconnect()
        self.start_button.clicked.connect(self._start_pose)
        self.table.setVisible(False)
        self.buttons.button(QDialogButtonBox.Apply).setEnabled(False)
        self.changes = {}
        self._step = 0
        self.status.setText("")
        self._show_step()
