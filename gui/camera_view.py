"""The camera picture with each tracked hand drawn over it."""
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from gui.style import HAND_COLORS, MUTED_COLOR
from hand_data import HAND_CONNECTIONS, TrackingFrame


class CameraView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.show_landmarks = True
        self._image: Optional[QImage] = None
        self._frame: Optional[TrackingFrame] = None
        self._message = "Starting..."

    def set_frame(self, frame: TrackingFrame):
        rgb = frame.frame_rgb
        height, width = rgb.shape[:2]
        self._image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888).copy()
        self._frame = frame
        self._message = ""
        self.update()

    def set_message(self, message: str):
        """Show a message instead of the picture (camera stopped, errors)."""
        self._image = None
        self._frame = None
        self._message = message
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(18, 18, 20))
        if self._image is None:
            painter.setPen(MUTED_COLOR)
            painter.setFont(QFont(self.font().family(), 12))
            painter.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, self._message)
            return

        target = self._fit(self._image.width(), self._image.height())
        painter.drawImage(target, self._image)
        if self.show_landmarks:
            for hand in self._frame.hands:
                self._draw_hand(painter, target, hand)

    def _fit(self, width: int, height: int) -> QRectF:
        scale = min(self.width() / width, self.height() / height)
        w, h = width * scale, height * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def _draw_hand(self, painter: QPainter, target: QRectF, hand):
        color = HAND_COLORS.get(hand.data.hand_type, QColor(255, 255, 255))
        points = [QPointF(target.x() + x * target.width(), target.y() + y * target.height())
                  for x, y, _ in hand.data.landmarks]
        painter.setPen(QPen(color, 2.5))
        for a, b in HAND_CONNECTIONS:
            painter.drawLine(points[a], points[b])
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255))
        for point in points:
            painter.drawEllipse(point, 3, 3)

        depth_cm = -hand.camera_position[2] * 100.0
        label = f"{hand.data.hand_type.upper()}  {depth_cm:.0f} cm  {hand.data.gesture}"
        wrist = points[0]
        painter.setFont(QFont(self.font().family(), 10, QFont.Bold))
        metrics = painter.fontMetrics()
        box = QRectF(wrist.x() - metrics.horizontalAdvance(label) / 2 - 6, wrist.y() + 10,
                     metrics.horizontalAdvance(label) + 12, metrics.height() + 4)
        painter.setBrush(QColor(0, 0, 0, 170))
        painter.drawRoundedRect(box, 4, 4)
        painter.setPen(color)
        painter.drawText(box, Qt.AlignCenter, label)
