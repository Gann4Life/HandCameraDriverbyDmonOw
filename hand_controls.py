"""
Analog trigger and grip from hand features, like a Valve Index controller
reads them: grip is how far middle, ring and pinky close, trigger is how far
the index closes or how hard thumb and index pinch. Both can be on at once
(a pinch with the other fingers closed), so nothing has to choose between them.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from hand_features import FINGERS, FULL_CURL_DEG, OPEN_CURL_DEG, HandFeatures, ramp
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
    # A grip that reaches latch_on is held at its peak until it stays below
    # latch_off for GRIP_RELEASE_SECONDS, so a value dipping near the game's own
    # threshold doesn't drop what the hand holds. It stays analog (no snap to 0/1).
    "latch_on": 0.6,
    "latch_off": 0.35,
    # Summed bend of each finger (thumb..pinky), open and fully curled, in
    # degrees; Calibrate gestures measures them for the user and the view
    "curl_open_deg": [OPEN_CURL_DEG[f] for f in FINGERS],
    "curl_full_deg": [FULL_CURL_DEG[f] for f in FINGERS],
}

# How far below the gate the pinch fades out completely
PINCH_GATE_FADE = 0.15

# How long a held grip must stay below latch_off before it lets go: longer than
# a few misread frames, short enough that opening the hand feels immediate.
# The "Grip lets go below" help in gui/settings_schema.py names this time.
GRIP_RELEASE_SECONDS = 0.1


@dataclass
class _GripHold:
    peak: float
    below_since: Optional[float] = None


def _curls_key(hand_type: str) -> str:
    """ControlMapper's filter key for a hand's finger curls (its controls use the bare hand_type)."""
    return hand_type + ".curls"


class ControlMapper:
    """Per-hand trigger and grip from features, smoothed."""

    def __init__(self, config: Dict):
        self.config = {key: config.get(key, default) for key, default in DEFAULTS.items()}
        self._filters: Dict[str, OneEuroFilter] = {}
        self._grip_holds: Dict[str, _GripHold] = {}

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
        trigger, grip = min(1.0, max(0.0, trigger)), min(1.0, max(0.0, grip))
        return trigger, self._hold_grip(hand_type, grip, t)

    def reset(self, hand_type: str):
        """Forget one hand's smoothing and held grip, for a hand that is new on this side."""
        for key in (hand_type, _curls_key(hand_type)):
            self._filters.pop(key, None)
        self._grip_holds.pop(hand_type, None)

    def _hold_grip(self, hand_type: str, grip: float, t: float) -> float:
        """The grip to send: its peak while held, the value itself otherwise."""
        hold = self._grip_holds.get(hand_type)
        if hold is None:
            if grip >= float(self.config["latch_on"]):
                self._grip_holds[hand_type] = _GripHold(grip)
            return grip
        hold.peak = max(hold.peak, grip)
        # A release value set above the hold value would grab and drop over and over
        if grip >= min(float(self.config["latch_off"]), float(self.config["latch_on"])):
            hold.below_since = None
        elif hold.below_since is None:
            hold.below_since = t
        elif t - hold.below_since >= GRIP_RELEASE_SECONDS:
            del self._grip_holds[hand_type]
            return grip
        return hold.peak

    def finger_curls(self, hand_type: str, features: HandFeatures, t: float) -> Tuple[float, ...]:
        """Smoothed curl of each finger, thumb to pinky, for the Index finger inputs and skeleton."""
        key = _curls_key(hand_type)
        smoother = self._filters.get(key)
        if smoother is None:
            smoother = OneEuroFilter(float(self.config["controls_min_cutoff"]), float(self.config["controls_beta"]))
            self._filters[key] = smoother
        return tuple(min(1.0, max(0.0, c)) for c in smoother(features.curl, t))
