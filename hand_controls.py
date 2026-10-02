"""
Analog trigger and grip from hand features, like a Valve Index controller
reads them: grip is how far middle, ring and pinky close, trigger is how far
the index closes or how hard thumb and index pinch. Both can be on at once
(a pinch with the other fingers closed), so nothing has to choose between them.
"""
from typing import Dict, Tuple

from hand_features import HandFeatures, ramp
from utils.one_euro import OneEuroFilter

# Config keys under "gestures" and their defaults; load_config fills in
# whichever are missing so older config files keep working
DEFAULTS = {
    "pinch_open": 0.55,         # thumb to index tip, in palm sizes: no pinch at or above this
    "pinch_closed": 0.20,       # ... full pinch at or below this
    "pinch_palm_gate": 0.6,     # index tip to palm, in palm sizes: below this it is a fist, not a pinch
    "trigger_from_pinch": True,
    "trigger_curl_start": 0.15,  # index curl where the trigger starts moving
    "trigger_curl_full": 0.75,   # ... and where it is fully pressed
    "grip_curl_start": 0.20,     # same for the mean curl of middle, ring and pinky
    "grip_curl_full": 0.80,
    "controls_min_cutoff": 2.0,  # One Euro smoothing of trigger and grip
    "controls_beta": 5.0,
}

# How far below the gate the pinch fades out completely
PINCH_GATE_FADE = 0.15


class ControlMapper:
    """Per-hand trigger and grip from features, smoothed."""

    def __init__(self, config: Dict):
        self.config = {key: config.get(key, default) for key, default in DEFAULTS.items()}
        self._filters: Dict[str, OneEuroFilter] = {}

    def pinch_strength(self, features: HandFeatures) -> float:
        """Thumb-index pinch, 0..1, faded out when the index tip is in the palm (a fist)."""
        c = self.config
        closeness = ramp(features.pinch_distance[0], float(c["pinch_open"]), float(c["pinch_closed"]))
        gate = float(c["pinch_palm_gate"])
        return closeness * ramp(features.index_tip_to_palm, gate - PINCH_GATE_FADE, gate)

    def raw(self, features: HandFeatures) -> Tuple[float, float]:
        """(trigger, grip) for one frame, unsmoothed."""
        c = self.config
        trigger = ramp(features.curl[1], float(c["trigger_curl_start"]), float(c["trigger_curl_full"]))
        if c["trigger_from_pinch"]:
            trigger = max(trigger, self.pinch_strength(features))
        grip_curl = sum(features.curl[2:5]) / 3.0
        grip = ramp(grip_curl, float(c["grip_curl_start"]), float(c["grip_curl_full"]))
        return trigger, grip

    def __call__(self, hand_type: str, features: HandFeatures, t: float) -> Tuple[float, float]:
        """Smoothed (trigger, grip) for one hand."""
        smoother = self._filters.get(hand_type)
        if smoother is None:
            smoother = OneEuroFilter(float(self.config["controls_min_cutoff"]), float(self.config["controls_beta"]))
            self._filters[hand_type] = smoother
        trigger, grip = smoother(self.raw(features), t)
        return min(1.0, max(0.0, trigger)), min(1.0, max(0.0, grip))

    def finger_curls(self, hand_type: str, features: HandFeatures, t: float) -> Tuple[float, ...]:
        """Smoothed curl of each finger, thumb to pinky, for the Index finger inputs and skeleton."""
        key = hand_type + ".curls"
        smoother = self._filters.get(key)
        if smoother is None:
            smoother = OneEuroFilter(float(self.config["controls_min_cutoff"]), float(self.config["controls_beta"]))
            self._filters[key] = smoother
        return tuple(min(1.0, max(0.0, c)) for c in smoother(features.curl, t))
