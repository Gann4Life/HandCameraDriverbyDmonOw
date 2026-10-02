"""Live readout of what each hand sends to the driver."""
from PySide6.QtWidgets import QFormLayout, QGroupBox, QLabel, QProgressBar, QVBoxLayout, QWidget

from gui.style import HAND_COLORS
from hand_data import TrackingFrame


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
        self.show_hand(None)

    @staticmethod
    def _bar(color: str) -> QProgressBar:
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setTextVisible(True)
        bar.setStyleSheet(f"QProgressBar::chunk {{ background: {color}; }}")
        return bar

    def show_hand(self, hand):
        if hand is None:
            self.state.setText("no")
            for label in (self.gesture, self.depth, self.position):
                label.setText("-")
            self.trigger.setValue(0)
            self.grip.setValue(0)
            return
        data = hand.data
        self.state.setText("yes")
        self.gesture.setText(data.gesture)
        self.depth.setText(f"{-hand.camera_position[2] * 100:.0f} cm from the camera")
        self.position.setText("  ".join(f"{axis} {v:+.2f}" for axis, v in zip("XYZ", data.position)) + " m")
        self.trigger.setValue(round(data.trigger_value * 100))
        self.grip.setValue(round(data.grip_value * 100))


class LivePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._readouts = {side: _HandReadout(side) for side in ("left", "right")}
        for readout in self._readouts.values():
            layout.addWidget(readout)
        layout.addStretch(1)

    def set_frame(self, frame: TrackingFrame):
        hands = {hand.data.hand_type: hand for hand in frame.hands}
        for side, readout in self._readouts.items():
            readout.show_hand(hands.get(side))

    def clear(self):
        for readout in self._readouts.values():
            readout.show_hand(None)
