"""
Keeps each tracked hand's left/right identity stable over time.

Per-frame handedness (hand geometry, or MediaPipe's label) is noisy: near
edge-on poses it flips for a frame or two, and MediaPipe occasionally reports
the same hand twice. Deciding every frame from scratch makes a hand teleport to
the other side. Here identity follows continuity first, and per-frame
evidence only overrides it when it persists.
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class HandDetection:
    """One hand MediaPipe found in the current frame."""
    index: int                    # position in MediaPipe's result lists
    wrist: Tuple[float, float]    # normalised image coordinates
    evidence: float               # handedness vote in [-1, 1]; > 0 means left
    score: float                  # MediaPipe's detection confidence


@dataclass
class _Track:
    wrist: Tuple[float, float]
    time: float
    contradictions: int = 0


class HandIdentityTracker:
    """Assigns "left"/"right" to each frame's detections."""

    def __init__(self, continuity_radius: float = 0.15, memory_seconds: float = 0.4,
                 switch_frames: int = 6, duplicate_radius: float = 0.05,
                 strong_evidence: float = 0.5, user_left_is_image_left: bool = True):
        """
        Args:
            continuity_radius: Max wrist movement between frames (fraction of the
                image) for a detection to count as the same hand
            memory_seconds: How long a lost hand's last position is remembered
            switch_frames: Consecutive frames of strong contrary evidence needed
                before a continuing hand is allowed to change side
            duplicate_radius: Detections closer than this are one hand seen twice
            strong_evidence: |evidence| above which a vote counts as a contradiction
            user_left_is_image_left: Which image side the user's left hand is on,
                used only when nothing else can decide
        """
        self.continuity_radius = continuity_radius
        self.memory_seconds = memory_seconds
        self.switch_frames = switch_frames
        self.duplicate_radius = duplicate_radius
        self.strong_evidence = strong_evidence
        self.user_left_is_image_left = user_left_is_image_left
        self._tracks: Dict[str, _Track] = {}

    @staticmethod
    def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _recent_tracks(self, now: float) -> Dict[str, _Track]:
        return {side: t for side, t in self._tracks.items() if now - t.time <= self.memory_seconds}

    def _side_from_evidence(self, det: HandDetection) -> str:
        if det.evidence > 0:
            return "left"
        if det.evidence < 0:
            return "right"
        on_image_left = det.wrist[0] < 0.5
        return "left" if on_image_left == self.user_left_is_image_left else "right"

    def _assign_single(self, det: HandDetection, recent: Dict[str, _Track]) -> str:
        nearest = None
        if recent:
            side, track = min(recent.items(), key=lambda kv: self._distance(det.wrist, kv[1].wrist))
            if self._distance(det.wrist, track.wrist) <= self.continuity_radius:
                nearest = side
        if nearest is None:
            return self._side_from_evidence(det)

        track = recent[nearest]
        contrary = det.evidence > self.strong_evidence if nearest == "right" else det.evidence < -self.strong_evidence
        track.contradictions = track.contradictions + 1 if contrary else 0
        if track.contradictions >= self.switch_frames:
            track.contradictions = 0
            return "right" if nearest == "left" else "left"
        return nearest

    def _assign_pair(self, a: HandDetection, b: HandDetection, recent: Dict[str, _Track]) -> Tuple[str, str]:
        def cost(det: HandDetection, side: str) -> float:
            track = recent.get(side)
            # Continuity dominates; an unknown track costs a neutral amount
            continuity = min(self._distance(det.wrist, track.wrist), 0.5) if track else 0.25
            vote = -det.evidence if side == "left" else det.evidence
            return continuity + 0.15 * vote
        straight = cost(a, "left") + cost(b, "right")
        swapped = cost(a, "right") + cost(b, "left")
        return ("left", "right") if straight <= swapped else ("right", "left")

    def assign(self, detections: List[HandDetection], now: float) -> Dict[int, str]:
        """
        Decide which side each detection is.

        Args:
            detections: This frame's hands
            now: Timestamp in seconds

        Returns:
            {detection index: "left" | "right"} for the detections kept;
            duplicates of an already-kept hand are left out
        """
        # One hand seen twice: keep the more confident copy
        kept: List[HandDetection] = []
        for det in sorted(detections, key=lambda d: d.score, reverse=True):
            if all(self._distance(det.wrist, k.wrist) > self.duplicate_radius for k in kept):
                kept.append(det)
        kept = kept[:2]

        recent = self._recent_tracks(now)
        if len(kept) == 1:
            sides = {kept[0].index: self._assign_single(kept[0], recent)}
        elif len(kept) == 2:
            side_a, side_b = self._assign_pair(kept[0], kept[1], recent)
            sides = {kept[0].index: side_a, kept[1].index: side_b}
        else:
            sides = {}

        for det in kept:
            side = sides[det.index]
            previous = self._tracks.get(side)
            self._tracks[side] = _Track(det.wrist, now, previous.contradictions if previous else 0)
            # A single hand that just changed side leaves its old track sitting
            # on top of it; drop that track so it cannot pull the hand back
            other = self._tracks.get("right" if side == "left" else "left")
            if len(kept) == 1 and other is not None and \
                    self._distance(det.wrist, other.wrist) <= self.continuity_radius and other.time < now:
                del self._tracks["right" if side == "left" else "left"]
        return sides
