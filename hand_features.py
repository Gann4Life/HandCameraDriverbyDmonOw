"""
Continuous hand features from MediaPipe world landmarks (metres): how much
each finger curls, how far the fingers spread, and how close the thumb is to
each fingertip. Gestures, trigger/grip and (later) Index finger curls are all
built from these, instead of from on/off "finger extended" tests.

Everything here is scale-free (angles, or distances divided by palm size), so
it does not depend on hand size or distance to the camera.
"""
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")

# Landmark chains, base to tip. The thumb starts at its CMC joint (1); the
# other fingers start at the wrist so the MCP bend is measured against the palm.
_CHAINS = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "pinky": (0, 17, 18, 19, 20),
}
_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
_PALM = (0, 5, 9, 13, 17)

# Summed joint bend, in degrees, of an open hand and of a fully curled finger.
# An open hand is not 0: the wrist-to-MCP line is not in line with the finger,
# more so for ring and pinky. Taken from a recorded POV clip (5th percentile and
# max) until the calibration wizard records each user's own range.
OPEN_CURL_DEG = {"thumb": 10.0, "index": 45.0, "middle": 35.0, "ring": 50.0, "pinky": 55.0}
FULL_CURL_DEG = {"thumb": 100.0, "index": 210.0, "middle": 210.0, "ring": 205.0, "pinky": 215.0}

# Thumb-tip to fingertip distance, in palm sizes, for no pinch and a full pinch
PINCH_OPEN = 0.55
PINCH_CLOSED = 0.20


@dataclass
class HandFeatures:
    curl: Tuple[float, ...]       # per finger in FINGERS order, 0 straight .. 1 fully curled
    curl_deg: Tuple[float, ...]   # the raw summed bend behind each curl
    splay_deg: Tuple[float, ...]  # thumb-index, index-middle, middle-ring, ring-pinky, in the palm plane
    pinch: Tuple[float, ...]      # thumb to index/middle/ring/pinky tip, 0 apart .. 1 touching
    pinch_distance: Tuple[float, ...]  # the same distances, in palm sizes
    index_tip_to_palm: float      # in palm sizes; small in a fist, large in a pinch
    palm_size_m: float            # wrist to middle MCP


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def _ramp(value: float, zero_at: float, one_at: float) -> float:
    """0 at zero_at, 1 at one_at, linear in between and clamped (either direction)."""
    return float(np.clip((value - zero_at) / (one_at - zero_at), 0.0, 1.0))


def compute_features(world_landmarks: Sequence[Sequence[float]]) -> HandFeatures:
    """
    Args:
        world_landmarks: 21 MediaPipe world landmarks, in metres

    Returns:
        The hand's features
    """
    pts = np.asarray(world_landmarks, dtype=float)
    palm_size = float(np.linalg.norm(pts[9] - pts[0])) or 1e-6

    curls, curl_degs = [], []
    for finger in FINGERS:
        chain = _CHAINS[finger]
        bones = [pts[chain[i + 1]] - pts[chain[i]] for i in range(len(chain) - 1)]
        # Bend at each joint = angle between consecutive bones. The thumb's
        # CMC is skipped: it mostly moves the thumb across the palm, not curls it.
        joints = range(1, len(bones) - 1) if finger == "thumb" else range(len(bones) - 1)
        bend = sum(_angle_deg(bones[j], bones[j + 1]) for j in joints)
        curl_degs.append(bend)
        curls.append(_ramp(bend, OPEN_CURL_DEG[finger], FULL_CURL_DEG[finger]))

    # Splay: angle between neighbouring proximal bones, projected onto the palm plane
    normal = np.cross(pts[5] - pts[0], pts[17] - pts[0])
    normal /= np.linalg.norm(normal) or 1.0

    def in_palm(v: np.ndarray) -> np.ndarray:
        return v - np.dot(v, normal) * normal

    proximal = [in_palm(pts[2] - pts[1])] + [in_palm(pts[m + 1] - pts[m]) for m in (5, 9, 13, 17)]
    splays = tuple(_angle_deg(proximal[i], proximal[i + 1]) for i in range(4))

    distances = tuple(float(np.linalg.norm(pts[tip] - pts[4])) / palm_size for tip in _TIPS.values())
    pinches = tuple(_ramp(d, PINCH_OPEN, PINCH_CLOSED) for d in distances)

    palm_center = pts[list(_PALM)].mean(axis=0)
    index_to_palm = float(np.linalg.norm(pts[8] - palm_center)) / palm_size

    return HandFeatures(curl=tuple(curls), curl_deg=tuple(curl_degs), splay_deg=splays,
                        pinch=pinches, pinch_distance=distances,
                        index_tip_to_palm=index_to_palm, palm_size_m=palm_size)


def legacy_extended(detector, landmarks: List[Tuple[float, float, float]]) -> Tuple[bool, ...]:
    """
    What the old gesture detector thinks is extended, per finger, from the
    normalised image landmarks it uses. Shown next to the curls to see where
    the two disagree. On a recorded POV clip it said "extended" for every
    finger in every frame: tip-to-wrist rarely drops below 0.6x MCP-to-wrist,
    even in a fist, which is why FIST and POINT almost never fired.
    """
    return tuple(detector.is_finger_extended(landmarks, finger) for finger in FINGERS)
