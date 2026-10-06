"""
Follows a lost hand's pixels with optical flow.

When MediaPipe loses a hand on a fast move, the hand is still in the picture,
only blurred. Its last tracked frames can't tell a swing that is just starting
from a hand at rest, but the picture can: optical flow (OpenCV's DIS) between
consecutive frames shows where the blurred hand went. hand_motion moves the
lost hand by what this measures across the picture; depth still comes from
the hand's last tracked frames.

The flow is only computed on frames where a hand is lost, on a small grey copy
of the picture.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import cv2
import numpy as np

# The picture is shrunk to this fraction of its size first: a hand is still
# tens of pixels wide, and the flow costs about 1.5 ms a frame at 640 x 480
SCALE = 0.25
# The hand's box is its landmarks' bounding box grown by this fraction of its
# size on each side, so a fast hand is still inside it on the next frame
BOX_PADDING = 0.3
# The pixels of the box that move most unlike the background (this fraction of
# them) are the hand: the rest is background, or the arm's slower end. On the
# recorded sessions 0.1 to 0.5, and padding 0.15 to 0.5, ended about as close
# to where lost hands were found; scaling the flow up (it under-reads a
# blurred hand) ended further.
HAND_SHARE = 0.25
# A box with less than this fraction of its area inside the picture has left it
MIN_VISIBLE = 0.5


@dataclass
class _Lost:
    box: np.ndarray                    # x0, y0, x1, y1 in normalised image coordinates
    offset: np.ndarray                 # how far the hand moved since its last tracked frame, normalised
    following: bool = True             # False once the box left the picture: the offset holds


class HandFlow:
    """Per hand: the box of its last tracked frame, followed through the picture while the hand is lost."""

    def __init__(self):
        self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self._previous: Optional[np.ndarray] = None
        self._current: Optional[np.ndarray] = None
        self._flow: Optional[np.ndarray] = None
        self._background = np.zeros(2)
        self._boxes: Dict[str, np.ndarray] = {}
        self._lost: Dict[str, _Lost] = {}

    def next_frame(self, frame_bgr: np.ndarray) -> None:
        """Take the newest camera frame (before any hand of it is tracked or followed)."""
        small = cv2.resize(frame_bgr, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        if self._current is not None and self._current.shape != gray.shape:
            self._current = None       # the camera changed: nothing to compare with
        self._previous, self._current, self._flow = self._current, gray, None

    def tracked(self, hand: str, image_points: Sequence[Sequence[float]]) -> None:
        """A tracked hand's landmarks (normalised image coordinates) on the newest frame."""
        points = np.asarray(image_points, dtype=float)[:, :2]
        low, high = points.min(axis=0), points.max(axis=0)
        pad = BOX_PADDING * (high - low)
        self._boxes[hand] = np.concatenate((low - pad, high + pad))
        self._lost.pop(hand, None)

    def forget(self, hand: str) -> None:
        """Drop a hand: the next time it is lost there is nothing to follow."""
        self._boxes.pop(hand, None)
        self._lost.pop(hand, None)

    def follow(self, hand: str) -> Optional[np.ndarray]:
        """
        A lost hand's movement across the picture since its last tracked frame,
        in normalised image coordinates, followed up to the newest frame. Call
        it once per frame while the hand is lost. None when there is nothing to
        follow (no tracked frame, or no previous frame to compare with).
        """
        lost = self._lost.get(hand)
        if lost is None:
            box = self._boxes.get(hand)
            if box is None:
                return None
            lost = self._lost[hand] = _Lost(box.copy(), np.zeros(2))
        if lost.following:
            shift = self._shift(lost.box)
            if shift is None:
                lost.following = False
            else:
                lost.box += np.tile(shift, 2)
                lost.offset = lost.offset + shift
        return lost.offset.copy()

    def _shift(self, box: np.ndarray) -> Optional[np.ndarray]:
        """How far the pixels in box moved between the last two frames (normalised), or None if it can't tell."""
        if self._previous is None or self._current is None:
            return None
        height, width = self._current.shape
        if self._flow is None:
            self._flow = self._dis.calc(self._previous, self._current, None)
            # What most of the picture does: a turning head moves everything
            self._background = np.median(self._flow.reshape(-1, 2), axis=0)
        x0, y0, x1, y1 = box * (width, height, width, height)
        area = max((x1 - x0) * (y1 - y0), 1e-9)
        cx0, cy0, cx1, cy1 = (int(round(min(max(v, 0.0), limit)))
                              for v, limit in zip((x0, y0, x1, y1), (width, height, width, height)))
        if (cx1 - cx0) * (cy1 - cy0) < MIN_VISIBLE * area or cx1 - cx0 < 2 or cy1 - cy0 < 2:
            return None
        vectors = self._flow[cy0:cy1, cx0:cx1].reshape(-1, 2)
        unlike = np.linalg.norm(vectors - self._background, axis=1)
        hand = vectors[unlike >= np.quantile(unlike, 1.0 - HAND_SHARE)]
        return np.median(hand, axis=0) / (width, height)
