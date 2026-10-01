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
                 strong_evidence: float = 0.5, user_left_is_image_left: bool = True,
                 order_weight: float = 1.5, order_full_separation: float = 0.2):
        """
        Args:
            continuity_radius: Max wrist movement between frames (fraction of the
                image) for a detection to count as the same hand
            memory_seconds: How long a lost hand's last position is remembered
            switch_frames: Consecutive frames of strong contrary evidence needed
                before a continuing hand is allowed to change side
            duplicate_radius: Detections closer than this are one hand seen twice
            strong_evidence: |evidence| above which a vote counts as a contradiction
            user_left_is_image_left: Which image side the user's left hand is on
            order_weight: How strongly two separated hands are assigned by
                left-right order (0 disables it)
            order_full_separation: Horizontal gap (fraction of the image) at
                which the order prior reaches full weight
        """
        self.continuity_radius = continuity_radius
        self.memory_seconds = memory_seconds
        self.switch_frames = switch_frames
        self.duplicate_radius = duplicate_radius
        self.strong_evidence = strong_evidence
        self.user_left_is_image_left = user_left_is_image_left
        self.order_weight = order_weight
        self.order_full_separation = order_full_separation
        self._tracks: Dict[str, _Track] = {}

    @staticmethod
    def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _recent_tracks(self, now: float) -> Dict[str, _Track]:
        return {side: t for side, t in self._tracks.items() if now - t.time <= self.memory_seconds}

    # Below this, a newly appearing hand is placed by image side instead: a weak
    # vote is typically a flat hand whose secondary signal may be inverted
    WEAK_EVIDENCE = 0.3

    def _side_from_evidence(self, det: HandDetection) -> str:
        if det.evidence >= self.WEAK_EVIDENCE:
            return "left"
        if det.evidence <= -self.WEAK_EVIDENCE:
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
            # An unknown track costs a neutral amount
            continuity = min(self._distance(det.wrist, track.wrist), 0.5) if track else 0.25
            vote = -det.evidence if side == "left" else det.evidence
            return continuity + 0.3 * vote

        def order_cost(left: HandDetection, right: HandDetection) -> float:
            # With both hands in view and clearly apart, the left hand is on
            # the user's left. This is what lets a pair that got swapped
            # correct itself: continuity alone would keep the swap forever.
            # Crossed arms held apart are the price.
            dx = right.wrist[0] - left.wrist[0]
            if not self.user_left_is_image_left:
                dx = -dx
            separation = min(abs(dx) / self.order_full_separation, 1.0)
            return self.order_weight * separation if dx < 0 else 0.0

        straight = cost(a, "left") + cost(b, "right") + order_cost(a, b)
        swapped = cost(a, "right") + cost(b, "left") + order_cost(b, a)
        return ("left", "right") if straight <= swapped else ("right", "left")

    def swap(self):
        """Exchange the two identities, as a manual correction."""
        left, right = self._tracks.get("left"), self._tracks.get("right")
        self._tracks = {side: track for side, track in (("left", right), ("right", left)) if track is not None}
        for track in self._tracks.values():
            track.contradictions = 0

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
