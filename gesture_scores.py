"""
Named gestures as weighted scores over hand features, instead of on/off
finger tests. Every gesture gets a 0..1 score each frame; the one shown (and
later bound to buttons) changes only with hysteresis, so it does not flicker
between two gestures that score about the same.
"""
from typing import Dict, Optional, Tuple

from hand_features import HandFeatures, ramp

GESTURES = ("OPEN", "FIST", "POINT", "PINCH", "THUMBS_UP", "PEACE")
UNKNOWN = "UNKNOWN"

DEFAULTS = {
    "gesture_enter": 0.6,          # a gesture needs this score to be picked
    "gesture_exit": 0.4,           # ... and is kept until it drops below this
    "gesture_confirm_frames": 3,   # frames a new gesture must lead before it is picked
    "finger_closed_start": 0.25,   # finger curl where a finger starts to count as closed
    "finger_closed_full": 0.65,    # ... and fully counts as closed
    "thumb_closed_start": 0.2,     # thumb curl where it starts to count as tucked in (fist, not thumbs up)
    "thumb_closed_full": 0.5,
}

# While the current gesture still scores above gesture_exit, another one must
# beat it by this much to take over
SWITCH_MARGIN = 0.15


def score_gestures(features: HandFeatures, pinch: float, config: Dict) -> Dict[str, float]:
    """
    Args:
        features: The hand's features
        pinch: Thumb-index pinch strength, already faded out in a fist (ControlMapper.pinch_strength)
        config: The "gestures" config section

    Returns:
        Score per gesture name, 0..1
    """
    # How closed each finger is, as a soft yes/no; a gesture scores as well as
    # its worst-matching finger (fuzzy AND), so "open" needs every finger open,
    # not just most of them on average
    start, full = float(config["finger_closed_start"]), float(config["finger_closed_full"])
    index, middle, ring, pinky = (ramp(c, start, full) for c in features.curl[1:])
    thumb = ramp(features.curl[0], float(config["thumb_closed_start"]), float(config["thumb_closed_full"]))
    return {
        "OPEN": min(1 - index, 1 - middle, 1 - ring, 1 - pinky, 1 - pinch),
        "FIST": min(index, middle, ring, pinky, thumb),
        "POINT": min(1 - index, middle, ring, pinky),
        "PINCH": pinch,
        "THUMBS_UP": min(index, middle, ring, pinky, 1 - thumb),
        "PEACE": min(1 - index, 1 - middle, ring, pinky),
    }


class GestureClassifier:
    """Per-hand gesture name from scores, with hysteresis and a short confirmation."""

    def __init__(self, config: Dict):
        self.config = {key: config.get(key, default) for key, default in DEFAULTS.items()}
        self._current: Dict[str, str] = {}
        self._candidate: Dict[str, Tuple[Optional[str], int]] = {}

    def reset(self, hand_type: str):
        """Forget one hand's gesture, for a hand that is new on this side."""
        self._current.pop(hand_type, None)
        self._candidate.pop(hand_type, None)

    def __call__(self, hand_type: str, scores: Dict[str, float]) -> str:
        enter = float(self.config["gesture_enter"])
        exit_ = float(self.config["gesture_exit"])
        confirm = max(1, int(self.config["gesture_confirm_frames"]))

        current = self._current.get(hand_type, UNKNOWN)
        best = max(scores, key=scores.get)
        keep = current != UNKNOWN and scores.get(current, 0.0) >= exit_
        if keep and (best == current or scores[best] < max(enter, scores[current] + SWITCH_MARGIN)):
            wanted = current
        elif scores[best] >= enter:
            wanted = best
        else:
            wanted = current if keep else UNKNOWN

        if wanted == current:
            self._candidate[hand_type] = (None, 0)
            return current
        candidate, frames = self._candidate.get(hand_type, (None, 0))
        frames = frames + 1 if candidate == wanted else 1
        if frames >= confirm:
            self._current[hand_type] = wanted
            self._candidate[hand_type] = (None, 0)
            return wanted
        self._candidate[hand_type] = (wanted, frames)
        return current
