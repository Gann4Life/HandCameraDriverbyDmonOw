"""
3D debug view of what the tracker sends: each hand's skeleton where the
tracker places it relative to the camera, and the depth along the camera ray.
"""
import math
from typing import Dict

import numpy as np
from PySide6.QtGui import QColor, QVector3D
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from gui.style import HAND_COLORS, MUTED_COLOR
from hand_data import HAND_CONNECTIONS, TrackingFrame

FRUSTUM_DEPTH = 0.25


def to_gl(points) -> np.ndarray:
    """OpenVR camera space (x right, y up, z back) to the GL view's (x right, y forward, z up)."""
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    return np.column_stack((p[:, 0], -p[:, 2], p[:, 1]))


def rgba(color: QColor, alpha: float = 1.0):
    return (color.redF(), color.greenF(), color.blueF(), alpha)


class HandView3D(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setMinimumSize(320, 240)
        self._hfov = None
        self._aspect = None
        try:
            import pyqtgraph.opengl as gl
            self._gl = gl
            self.view = gl.GLViewWidget()
        except Exception as e:  # no OpenGL driver or pyqtgraph
            self.view = None
            label = QLabel(f"3D view unavailable: {e}")
            label.setWordWrap(True)
            layout.addWidget(label)
            return
        layout.addWidget(self.view)
        self.view.setBackgroundColor((18, 18, 20))
        # From behind and above the camera, looking out the way it looks
        self.view.setCameraPosition(pos=QVector3D(0.0, 0.35, -0.1), distance=1.4, elevation=25, azimuth=-100)

        grid = gl.GLGridItem(color=(255, 255, 255, 40))
        grid.setSize(1.6, 1.6)
        grid.setSpacing(0.1, 0.1)
        grid.translate(0.0, 0.6, -0.45)
        self.view.addItem(grid)

        self._frustum = gl.GLLinePlotItem(mode='lines', color=rgba(MUTED_COLOR), width=1.5, antialias=True)
        self.view.addItem(self._frustum)

        self._hands: Dict[str, dict] = {}
        for side, color in HAND_COLORS.items():
            items = {
                'bones': gl.GLLinePlotItem(mode='lines', color=rgba(color), width=3, antialias=True),
                'joints': gl.GLScatterPlotItem(color=(1, 1, 1, 1), size=5),
                'ray': gl.GLLinePlotItem(mode='lines', color=rgba(color, 0.35), width=1, antialias=True),
                'label': gl.GLTextItem(color=color),
            }
            for item in items.values():
                item.setVisible(False)
                self.view.addItem(item)
            self._hands[side] = items

    def set_frame(self, frame: TrackingFrame):
        if self.view is None or not self.isVisible():
            return
        height, width = frame.frame_rgb.shape[:2]
        self._update_frustum(frame.hfov_deg, height / width)

        seen = set()
        for hand in frame.hands:
            side = hand.data.hand_type
            items = self._hands.get(side)
            if items is None:
                continue
            seen.add(side)
            wrist = to_gl(hand.camera_position)[0]
            if hand.camera_points is not None:
                points = to_gl(hand.camera_points)
                bones = np.array([points[i] for pair in HAND_CONNECTIONS for i in pair])
                items['bones'].setData(pos=bones)
                items['joints'].setData(pos=points)
                items['bones'].setVisible(True)
            else:
                items['joints'].setData(pos=wrist.reshape(1, 3))
                items['bones'].setVisible(False)
            items['joints'].setVisible(True)
            items['ray'].setData(pos=np.array([[0.0, 0.0, 0.0], wrist]))
            items['ray'].setVisible(True)
            items['label'].setData(pos=wrist + np.array([0.0, 0.0, -0.06]),
                                   text=f"{-hand.camera_position[2] * 100:.0f} cm")
            items['label'].setVisible(True)
        for side, items in self._hands.items():
            if side not in seen:
                for item in items.values():
                    item.setVisible(False)

    def clear(self):
        if self.view is None:
            return
        for items in self._hands.values():
            for item in items.values():
                item.setVisible(False)

    def _update_frustum(self, hfov_deg: float, aspect: float):
        """Draw the camera as a pyramid matching its field of view."""
        if (hfov_deg, aspect) == (self._hfov, self._aspect):
            return
        self._hfov, self._aspect = hfov_deg, aspect
        half_w = FRUSTUM_DEPTH * math.tan(math.radians(hfov_deg) / 2.0)
        half_h = half_w * aspect
        corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
        far = [(x, y, -FRUSTUM_DEPTH) for x, y in corners]
        segments = []
        for i, corner in enumerate(far):
            segments += [(0.0, 0.0, 0.0), corner, corner, far[(i + 1) % 4]]
        self._frustum.setData(pos=to_gl(segments))
