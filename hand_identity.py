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
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# Skeleton distance (see skeleton_distance) under which two detections are one hand
# seen twice, whatever came before. On the recorded sessions every pair this
# close was one blurred hand fitted twice; two real hands held together (palms
# side by side, fists touching) started at about 0.14.
SAME_HAND_DISTANCE = 0.12
# Under this, two detections lie on top of each other: one hand, or two hands
# held together or crossing. Telling them apart takes the tracks (see
# HandIdentityTracker._copy_on_other_hand).
ON_TOP_DISTANCE = 0.35


@dataclass
class HandDetection:
    """One hand MediaPipe found in the current frame."""
    index: int                    # position in MediaPipe's result lists
    wrist: Tuple[float, float]    # normalised image coordinates
    evidence: float               # handedness vote in [-1, 1]; > 0 means left
    score: float                  # MediaPipe's detection confidence
    # All its landmarks in normalised image coordinates (x, y), wrist first
    # (points[0] is wrist); None compares wrists only
    points: Optional[Sequence[Sequence[float]]] = None


def skeleton_distance(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    """
    How far apart two hands' landmarks lie, in hand sizes: the mean distance
    from each point to the nearest point of the other hand, both ways. Any
    point may match any other, so a copy fitted as the other hand (mirrored,
    or with its fingers in the wrong order) still counts as on top. 0 is the
    same skeleton; two hands side by side are about 1 apart.
    """
    a, b = np.asarray(a, dtype=float)[:, :2], np.asarray(b, dtype=float)[:, :2]
    size = max(float(np.ptp(a, axis=0).max()), float(np.ptp(b, axis=0).max()), 1e-6)
    distances = np.linalg.norm(a[:, None] - b[None], axis=2)
    return 0.5 * float(distances.min(axis=1).mean() + distances.min(axis=0).mean()) / size


@dataclass
class _Track:
    wrist: Tuple[float, float]
    time: float
    contradictions: int = 0


def other_side(side: str) -> str:
    return "right" if side == "left" else "left"


# A curl vote this clear counts as an opinion; below it, it is only noise of a flat hand
CLEAR_CURL_VOTE = 0.2


def combine_handedness_votes(curl: float, secondary: float, secondary_ignores_depth: bool) -> float:
    """
    One frame's left/right vote, in [-1, 1] (> 0 is left), from the finger-curl
    vote and a secondary cue.

    Curl leads: fingers only bend toward the palm, whatever the rotation. But it
    reads MediaPipe's depth, which flips when MediaPipe sees the back of the hand,
    and a flipped depth reverses the vote. The palm-away cue's sign uses only the
    landmarks' x and y, so it cannot flip that way: when the two clearly disagree
    the frame says nothing and the hand keeps its side. The price: in POV a palm
    turned toward the camera also says nothing, so a hand on the wrong side cannot
    correct itself while held that way.
    """
    if secondary_ignores_depth and abs(curl) >= CLEAR_CURL_VOTE and secondary * curl < 0:
        return 0.0
    return max(-1.0, min(1.0, 0.8 * curl + 0.2 * secondary))


class HandIdentityTracker:
    """Assigns "left"/"right" to each frame's detections."""

    def __init__(self, continuity_radius: float = 0.15, memory_seconds: float = 0.4,
                 switch_frames: int = 15, duplicate_radius: float = 0.05,
                 strong_evidence: float = 0.5, user_left_is_image_left: bool = True,
                 order_weight: float = 1.5, order_full_separation: float = 0.2,
                 lone_memory_seconds: float = 3.0):
        """
        Args:
            continuity_radius: Max wrist movement between frames (fraction of the
                image) for a detection to count as the same hand
            memory_seconds: How long a lost hand's last position is remembered
            switch_frames: Consecutive frames of strong contrary evidence needed
                before a continuing hand is allowed to change side
            duplicate_radius: Detections whose wrists are closer than this are
                one hand seen twice (and so are skeletons that overlap, see
                SAME_HAND_DISTANCE)
            strong_evidence: |evidence| above which a vote counts as a contradiction
            user_left_is_image_left: Which image side the user's left hand is on
            order_weight: How strongly two separated hands are assigned by
                left-right order (0 disables it)
            order_full_separation: Horizontal gap (fraction of the image) at
                which the order prior reaches full weight
            lone_memory_seconds: For this long after only one hand was seen, a
                hand showing up far away is taken to be that hand when it shows
                up within FAST_MOVE_SECONDS, or its vote is not strong
        """
        self.continuity_radius = continuity_radius
        self.memory_seconds = memory_seconds
        self.switch_frames = switch_frames
        self.duplicate_radius = duplicate_radius
        self.strong_evidence = strong_evidence
        self.user_left_is_image_left = user_left_is_image_left
        self.order_weight = order_weight
        self.order_full_separation = order_full_separation
        self.lone_memory_seconds = lone_memory_seconds
        self._tracks: Dict[str, _Track] = {}
        # (side, time) of the last frame with exactly one hand; None after two
        self._lone: Optional[Tuple[str, float]] = None
        # Sides whose hand in the last assign() is new there: it appeared, came
        # back after memory_seconds, or changed side. Any per-hand history kept
        # for the side belongs to another moment or another hand
        self.new_sides: Set[str] = set()

    @staticmethod
    def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _recent_tracks(self, now: float) -> Dict[str, _Track]:
        return {side: t for side, t in self._tracks.items() if now - t.time <= self.memory_seconds}

    # Below this, a newly appearing hand is placed by image side instead: a weak
    # vote is typically a flat hand whose secondary signal may be inverted
    WEAK_EVIDENCE = 0.3
    # A lone hand seen this recently is still the same hand, however far it jumped:
    # a few dropped frames, far too short to swap one hand for the other
    FAST_MOVE_SECONDS = 0.15

    def _side_from_evidence(self, det: HandDetection) -> str:
        if det.evidence >= self.WEAK_EVIDENCE:
            return "left"
        if det.evidence <= -self.WEAK_EVIDENCE:
            return "right"
        on_image_left = det.wrist[0] < 0.5
        return "left" if on_image_left == self.user_left_is_image_left else "right"

    def _assign_single(self, det: HandDetection, recent: Dict[str, _Track], now: float) -> str:
        nearest = None
        if recent:
            side, track = min(recent.items(), key=lambda kv: self._distance(det.wrist, kv[1].wrist))
            if self._distance(det.wrist, track.wrist) <= self.continuity_radius:
                nearest = side
        if nearest is None:
            lone = self._lone_side(recent, now)
            # Seen moments ago, it is the same hand moved fast. Back after a
            # loss it may be the other hand, so a strong vote decides; anything
            # less (a flat hand, one misread frame) defers to memory
            just_seen = lone is not None and now - self._lone[1] <= self.FAST_MOVE_SECONDS
            if lone is not None and (just_seen or abs(det.evidence) <= self.strong_evidence):
                nearest = lone
        if nearest is None:
            return self._side_from_evidence(det)

        track = self._tracks[nearest]
        contrary = det.evidence > self.strong_evidence if nearest == "right" else det.evidence < -self.strong_evidence
        track.contradictions = track.contradictions + 1 if contrary else 0
        if track.contradictions >= self.switch_frames:
            track.contradictions = 0
            return other_side(nearest)
        return nearest

    def _lone_side(self, recent: Dict[str, _Track], now: float) -> Optional[str]:
        """
        The side of the only hand seen lately, when no other hand is around.
        With one hand in view a turning camera sweeps it across the image
        faster than continuity_radius; this keeps it the same hand.
        """
        if self._lone is None or now - self._lone[1] > self.lone_memory_seconds:
            return None
        side = self._lone[0]
        return side if other_side(side) not in recent and side in self._tracks else None

    def _continuing(self, det: HandDetection, recent: Dict[str, _Track]) -> Optional[str]:
        """The side whose recent track this detection continues, if any."""
        near = [(self._distance(det.wrist, t.wrist), side) for side, t in recent.items()]
        near = [n for n in near if n[0] <= self.continuity_radius]
        return min(near)[1] if near else None

    def _assign_pair(self, a: HandDetection, b: HandDetection, recent: Dict[str, _Track]) -> Tuple[str, str]:
        # A hand already being tracked keeps its side when another one shows
        # up: a newcomer (often a partial hand at the edge) must not push it
        # over through the left-right order prior
        side_a, side_b = self._continuing(a, recent), self._continuing(b, recent)
        if side_a is not None and side_b is None:
            return side_a, other_side(side_a)
        if side_b is not None and side_a is None:
            return other_side(side_b), side_b

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

    def _update_new_sides(self, kept: List[HandDetection], sides: Dict[int, str], recent: Dict[str, _Track]):
        # A fast move alone does not make a hand new: resetting a held grip
        # mid-swing would drop what it holds
        def is_new(det: HandDetection) -> bool:
            side = sides[det.index]
            track = recent.get(side)
            if track is None:
                return True  # even the same hand: what it held was let go while out of view
            if self._distance(det.wrist, track.wrist) <= self.continuity_radius:
                return False
            # Far from its side's track: new if it came from the other side's
            # track, or another hand is the one continuing this side's
            other = recent.get(other_side(side))
            came_from_other = other is not None and self._distance(det.wrist, other.wrist) <= self.continuity_radius
            taken = any(self._distance(k.wrist, track.wrist) <= self.continuity_radius for k in kept if k is not det)
            return came_from_other or taken
        self.new_sides = {sides[det.index] for det in kept if is_new(det)}

    def follow(self, side: str, wrist: Tuple[float, float]) -> None:
        """
        Move a lost hand's track to where the hand is predicted to be (see
        hand_motion), so the hand found there again is the same one. The track's
        time stays its last sighting: how long it is remembered doesn't change.
        A prediction running onto the other hand, still in view, leaves the track
        where it was: the visible hand must not be taken for the lost one.
        """
        track = self._tracks.get(side)
        if track is None:
            return
        wrist = (float(wrist[0]), float(wrist[1]))
        other = self._tracks.get(other_side(side))
        if other is not None and other.time > track.time and self._distance(wrist, other.wrist) <= self.continuity_radius:
            return
        track.wrist = wrist

    def has_track(self, side: str) -> bool:
        """
        Whether a hand is still remembered on side. A single hand that changed
        side drops its old track, so its prediction there must stop too.
        """
        return side in self._tracks

    def swap(self):
        """Exchange the two identities, as a manual correction."""
        left, right = self._tracks.get("left"), self._tracks.get("right")
        self._tracks = {side: track for side, track in (("left", right), ("right", left)) if track is not None}
        for track in self._tracks.values():
            track.contradictions = 0

    def _same_hand(self, a: HandDetection, b: HandDetection) -> bool:
        if self._distance(a.wrist, b.wrist) <= self.duplicate_radius:
            return True
        return a.points is not None and b.points is not None and \
            skeleton_distance(a.points, b.points) < SAME_HAND_DISTANCE

    def _without_copies(self, detections: List[HandDetection], recent: Dict[str, _Track]) -> List[HandDetection]:
        """This frame's hands without the copies of another: at most two, most confident first."""
        kept: List[HandDetection] = []
        for det in sorted(detections, key=lambda d: d.score, reverse=True):
            if not any(self._same_hand(det, k) for k in kept):
                kept.append(det)
        kept = kept[:2]
        if len(kept) == 2:
            copy = self._copy_on_other_hand(kept[0], kept[1], recent)
            if copy is not None:
                kept.remove(copy)
        return kept

    def _copy_on_other_hand(self, a: HandDetection, b: HandDetection,
                            recent: Dict[str, _Track]) -> Optional[HandDetection]:
        """
        When two hands cross or touch, MediaPipe can lose the hidden one and fit
        its slot onto the hand in front: that skeleton jumps onto the other hand
        in a frame. Two skeletons on top of each other that both continue the
        same track, while the other track was elsewhere, are such a copy. The
        one further from that track is it; the hidden hand is lost instead (and
        predicted). Two hands that came together each continue their own track.
        """
        if a.points is None or b.points is None or len(recent) < 2:
            return None
        if skeleton_distance(a.points, b.points) >= ON_TOP_DISTANCE:
            return None
        def nearest(det: HandDetection) -> str:
            return min(recent, key=lambda side: self._distance(det.wrist, recent[side].wrist))
        side = nearest(a)
        if nearest(b) != side:
            return None
        copy = max((a, b), key=lambda det: self._distance(det.wrist, recent[side].wrist))
        # Still within reach of the other track: a hand moving fast toward the
        # other one (a clap, both hands on one object), not a copy
        if self._distance(copy.wrist, recent[other_side(side)].wrist) <= self.continuity_radius:
            return None
        return copy

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
        recent = self._recent_tracks(now)
        kept = self._without_copies(detections, recent)
        if len(kept) == 1:
            sides = {kept[0].index: self._assign_single(kept[0], recent, now)}
        elif len(kept) == 2:
            side_a, side_b = self._assign_pair(kept[0], kept[1], recent)
            sides = {kept[0].index: side_a, kept[1].index: side_b}
        else:
            sides = {}

        self._update_new_sides(kept, sides, recent)
        if len(kept) == 1:
            self._lone = (sides[kept[0].index], now)
        elif len(kept) == 2:
            self._lone = None
        for det in kept:
            side = sides[det.index]
            previous = self._tracks.get(side)
            # Contrary frames only count while one hand stays in view without a break
            keep_count = previous is not None and len(kept) == 1 and side not in self.new_sides
            self._tracks[side] = _Track(det.wrist, now, previous.contradictions if keep_count else 0)
            # A single hand that just changed side leaves its old track sitting
            # on top of it; drop that track so it cannot pull the hand back
            other = self._tracks.get(other_side(side))
            if len(kept) == 1 and other is not None and \
                    self._distance(det.wrist, other.wrist) <= self.continuity_radius and other.time < now:
                del self._tracks[other_side(side)]
        return sides
