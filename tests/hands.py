"""
Synthetic MediaPipe world landmarks (metres) for tests: a flat right hand in
the z = 0 plane, fingers pointing +y, each finger bent by a chosen angle at
every joint, towards -z (the palm side).
"""
from typing import Dict, List, Optional

import numpy as np

# Knuckle (MCP, or CMC for the thumb) positions, relative to the wrist at the origin
_BASES = {
    "thumb": (0.025, 0.025, 0.0),
    "index": (0.025, 0.090, 0.0),
    "middle": (0.000, 0.095, 0.0),
    "ring": (-0.020, 0.090, 0.0),
    "pinky": (-0.040, 0.080, 0.0),
}
_FIRST = {"thumb": 1, "index": 5, "middle": 9, "ring": 13, "pinky": 17}
_BONE_M = 0.03
_PALM_SIDE = np.array([0.0, 0.0, -1.0])


def make_hand(bend_deg: Optional[Dict[str, float]] = None) -> List[List[float]]:
    """
    Args:
        bend_deg: Bend at each joint, per finger name; 0 (straight) when missing.
            Fingers bend at the knuckle and the two joints after it; the thumb
            at its CMC, MCP and IP.

    Returns:
        21 landmarks as [x, y, z] in metres
    """
    bend_deg = bend_deg or {}
    pts = np.zeros((21, 3))
    for finger, base in _BASES.items():
        first = _FIRST[finger]
        base = np.array(base)
        pts[first] = base
        # Straight fingers continue the wrist-to-knuckle line
        straight = base / np.linalg.norm(base)
        bend = np.radians(bend_deg.get(finger, 0.0))
        for i in range(1, 4):
            # Each joint adds the bend, in the plane of the finger and the palm normal
            direction = np.cos(i * bend) * straight + np.sin(i * bend) * _PALM_SIDE
            pts[first + i] = pts[first + i - 1] + _BONE_M * direction
    return pts.tolist()
