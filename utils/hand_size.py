"""
Steadier depth from a steadier hand size. solvePnP's distance scales with
the size of the hand model it fits, and MediaPipe's world landmarks change
that size by a few percent from frame to frame, which shows up as the hand
moving toward and away from the camera. A real hand keeps its size, so each
hand's model is rescaled to the median size of its recent frames.
"""
from collections import deque
from typing import Deque, Dict, Sequence, Tuple

import numpy as np

# Palm edges and the wrist-to-middle-knuckle line: rigid, unlike the fingers
PALM_EDGES = ((0, 5), (5, 17), (17, 0), (0, 9))


def palm_size(world: Sequence[Tuple[float, float, float]]) -> float:
    points = np.asarray(world, dtype=np.float64)
    return float(sum(np.linalg.norm(points[a] - points[b]) for a, b in PALM_EDGES))


class HandSizeStabilizer:
    """Per-hand factor that brings each frame's hand model to its median size."""

    def __init__(self, window: int = 90):
        self.window = max(1, int(window))
        self.sizes: Dict[str, Deque[float]] = {}

    def __call__(self, hand: str, world: Sequence[Tuple[float, float, float]]) -> float:
        size = palm_size(world)
        if size <= 1e-6:
            return 1.0
        history = self.sizes.setdefault(hand, deque(maxlen=self.window))
        history.append(size)
        return float(np.median(history)) / size

    def reset(self):
        self.sizes.clear()
