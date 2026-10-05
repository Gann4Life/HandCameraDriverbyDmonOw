"""Live readout of what each hand sends to the driver, and the finger features behind it."""
from PySide6.QtWidgets import (QFormLayout, QGroupBox, QHBoxLayout, QLabel, QProgressBar, QScrollArea,
                               QVBoxLayout, QWidget)

from gui.style import HAND_COLORS, MUTED_COLOR
from hand_data import TrackingFrame
from gesture_scores import GESTURES
from hand_features import FINGERS

PINCH_FINGERS = ("index", "middle", "ring", "pinky")


class _HandReadout(QGroupBox):
    def __init__(self, side: str):
        super().__init__(f"{side.capitalize()} hand")
        color = HAND_COLORS[side].name()
        self.setStyleSheet(f"QGroupBox {{ color: {color}; font-weight: bold; }}")
        form = QFormLayout(self)
        self.state = QLabel()
        self.gesture = QLabel()
        self.depth = QLabel()
        self.position = QLabel()
        self.trigger = self._bar(color)
        self.grip = self._bar(color)
        form.addRow("Tracked", self.state)
        form.addRow("Gesture", self.gesture)
        form.addRow("Distance", self.depth)
        form.addRow("Position", self.position)
        form.addRow("Trigger", self.trigger)
        form.addRow("Grip", self.grip)

        form.addRow(self._heading("Gesture scores"))
        self.scores = {name: self._bar(color) for name in GESTURES}
        for name, bar in self.scores.items():
            form.addRow(name.replace("_", " ").capitalize(), bar)

        form.addRow(self._heading("Finger curl"))
        self.curls = {finger: self._bar(color) for finger in FINGERS}
        for finger, bar in self.curls.items():
            form.addRow(finger.capitalize(), bar)

        form.addRow(self._heading("Thumb pinch"))
        self.pinches = {finger: self._bar(color) for finger in PINCH_FINGERS}
        for finger, bar in self.pinches.items():
            form.addRow(f"to {finger}", bar)
        self.index_palm = QLabel()
        form.addRow("Index tip to palm", self.index_palm)
        self.splay = QLabel()
        form.addRow("Splay", self.splay)
        self.show_hand(None)

    @staticmethod
    def _heading(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(f"color: {MUTED_COLOR.name()}; margin-top: 8px;")
        return label

    @staticmethod
    def _bar(color: str) -> QProgressBar:
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setTextVisible(True)
        bar.setStyleSheet(f"QProgressBar::chunk {{ background: {color}; }}")
        return bar

    def show_hand(self, hand):
        bars = [self.trigger, self.grip, *self.scores.values(), *self.curls.values(), *self.pinches.values()]
        if hand is None:
            self.state.setText("no")
            for label in (self.gesture, self.depth, self.position, self.index_palm, self.splay):
                label.setText("-")
            for bar in bars:
                bar.setValue(0)
                bar.setFormat("%p%")
            return
        data = hand.data
        self.state.setText("predicted (lost)" if hand.predicted else "yes")
        self.gesture.setText(data.gesture)
        self.depth.setText(f"{-hand.camera_position[2] * 100:.0f} cm from the camera")
        self.position.setText("  ".join(f"{axis} {v:+.2f}" for axis, v in zip("XYZ", data.position)) + " m")
        self.trigger.setValue(round(data.trigger_value * 100))
        self.grip.setValue(round(data.grip_value * 100))

        features = hand.features
        if features is None:
            for bar in bars[2:]:
                bar.setValue(0)
            for label in (self.index_palm, self.splay):
                label.setText("-")
            return
        for name, bar in self.scores.items():
            bar.setValue(round(hand.gesture_scores.get(name, 0.0) * 100))
            bar.setFormat("%p%   ◀ active" if name == data.gesture else "%p%")
        for i, finger in enumerate(FINGERS):
            bar = self.curls[finger]
            bar.setValue(round(features.curl[i] * 100))
            bar.setFormat(f"%p%   {features.curl_deg[i]:.0f}°")
        for i, finger in enumerate(PINCH_FINGERS):
            bar = self.pinches[finger]
            bar.setValue(round(features.pinch[i] * 100))
            bar.setFormat(f"%p%   {features.pinch_distance[i]:.2f} palms")
        self.index_palm.setText(f"{features.index_tip_to_palm:.2f} palms")
        self.splay.setText("  ".join("-" if v != v else f"{v:.0f}°" for v in features.splay_deg))


class LivePanel(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        row = QHBoxLayout()
        self._readouts = {side: _HandReadout(side) for side in ("left", "right")}
        for readout in self._readouts.values():
            row.addWidget(readout)
        layout.addLayout(row)
        layout.addStretch(1)
        self.setWidget(content)

    def set_frame(self, frame: TrackingFrame):
        hands = {hand.data.hand_type: hand for hand in frame.hands}
        for side, readout in self._readouts.items():
            readout.show_hand(hands.get(side))

    def clear(self):
        for readout in self._readouts.values():
            readout.show_hand(None)
