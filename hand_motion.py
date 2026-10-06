"""
Keeps a hand moving through short tracking gaps instead of freezing it.

On a fast move motion blur makes MediaPipe lose the hand for a few frames: the
hand would stop where it was last seen and jump when it is found again. While
a hand is lost, its pose follows the trajectory of its last tracked frames
for at most prediction_seconds. A hand that was leaving the picture eases to
a stop instead, and one that was barely moving holds still. When the hand is
found again, the pose blends from where it was predicted to where it is
measured.

A lost hand carries on along a curve, not a straight line: the heading and
speed of each step between its tracked frames are fitted against time, so a
swing keeps bending as it was (an arc around the elbow keeps its radius) and
a hand that was slowing down keeps slowing down. It keeps its full speed for
MOMENTUM_SECONDS, then eases off, so a prediction always comes to rest.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from gesture_detector import Quaternion, quat_multiply, quat_slerp

DEFAULTS = {
    "prediction_seconds": 0.3,  # how long a lost hand keeps moving; 0 turns prediction off
}
MAX_PREDICTION_SECONDS = 1.0

# Tracked frames the trajectory is fitted to: the newest ones within this time
HISTORY_SECONDS = 0.15
HISTORY_SAMPLES = 8
# A lost hand keeps its full speed (and turn) this long, then slows down with
# these time constants. Depth only moves at all when its frames agree (see
# DEPTH_NOISE_M); a hand leaving the picture eases to a stop at once. Hiding
# frames of tracked fast moves in recorded sessions, holding 0.1 s overshot
# where the hand really went; 0.05 s ended closer than slowing down at once.
MOMENTUM_SECONDS = 0.05
POSITION_DAMPING_SECONDS = 0.2
DEPTH_DAMPING_SECONDS = 0.2
ROTATION_DAMPING_SECONDS = 0.2
LEAVING_DAMPING_SECONDS = 0.05
# The heading of a lost hand's path turns at most this fast when it is lost,
# and this far in all: a little noise in the headings of short steps is a
# sharp turn
MAX_PATH_TURN_RATE = 4.0 * math.pi
MAX_PATH_TURN_DEG = 180.0
# A lost hand speeds up to at most this many times the speed it was lost at.
# A hand slowing down keeps slowing down, but speeding up on past the last
# frames overshot on the recorded sessions: a swing is fastest mid-arc.
MAX_SPEED_GAIN = 1.0
# Time step of the predicted trajectory, computed once per gap
TRAJECTORY_STEP = 1.0 / 240.0
# A hand lost while its wrist crosses the picture slower than this (in picture
# sizes per second: widths across, heights down) holds still, turning included:
# its speed would be mostly noise, and on the
# recorded clips moving it ended further from where it was found than holding.
MIN_IMAGE_SPEED = 0.5
# Faster than a hand moves through a gap: a depth misread, not motion
MAX_SPEED_M_S = 4.0
# Depth keeps moving through a gap only when the tracked frames agree on it:
# every step at least DEPTH_NOISE_M goes the same way, and no single step makes
# most of the move. One camera's depth jumps by centimetres when the wrist turns
# or a finger hides, and on the recorded clips extrapolating such a jump sent
# lost hands 8-14 cm away within a frame or two.
DEPTH_NOISE_M = 0.005
# Three: a hand turning or curling changes one camera's depth (read from the
# hand's size) by over 10 cm a frame for two frames in a row
DEPTH_MIN_STEPS = 3
DEPTH_MAX_STEP_SHARE = 0.6
# A tracked wrist that jumps this many times further than the frames before it
# moved, and at least JUMP_MIN_IMAGE of the picture, is a misdetection or a
# hand mistaken for the other: the fit leaves it out. After frames of rest a
# big step is the hand setting off, unless it is at least JUMP_FROM_REST_IMAGE
# (4.5 picture widths a second at 30 fps): too far for a hand starting to move.
JUMP_RATIO = 3.0
JUMP_MIN_IMAGE = 0.05
JUMP_FROM_REST_IMAGE = 0.15
# A bad rotation estimate can't spin the hand further than this
MAX_EXTRA_TURN_DEG = 90.0
# A wrist this close to the picture's edge (fraction of the image), moving toward it, is leaving
EDGE_MARGIN = 0.05
# ...or one about to cross an edge within this long. Not the whole horizon: a
# fast swing heads past an edge within a few tenths of a second wherever it is,
# and treating every fast hand as leaving stopped it dead.
LEAVING_LOOKAHEAD_SECONDS = 0.05
# How long a returning hand takes to go from its predicted pose to its measured one
BLEND_SECONDS = 0.1
# Points along the predicted path drawn by the preview
PATH_POINTS = 12


@dataclass
class Prediction:
    """Where a lost hand is now, and the path it is predicted along."""
    camera_position: np.ndarray        # wrist, the same space as observe()'s camera_position
    rotation: Quaternion
    image_wrist: np.ndarray            # normalised image coordinates
    image_offset: np.ndarray           # image_wrist minus the last tracked one
    leaving: bool                      # the hand was leaving the picture: it eases to a stop
    path: np.ndarray                   # N x 2 image points: tracked frames, then the predicted path to its end


@dataclass
class _Sample:
    time: float
    camera_position: np.ndarray
    rotation: Quaternion
    image_wrist: np.ndarray


@dataclass
class _Motion:
    """A lost hand's fitted motion, from its last tracked sample."""
    start: _Sample                     # the last measured frame
    position: np.ndarray               # the pose sent for it, where the motion starts
    rotation: Quaternion
    step: float                        # seconds between the rows of table, from start.time
    # One row per step: camera offset (metres from position, 3), image offset
    # (from start.image_wrist, 2), extra rotation (a rotation vector, 3)
    table: np.ndarray
    path_ahead: np.ndarray             # PATH_POINTS x 2 image points out to the horizon
    leaving: bool

    def at(self, tau: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(camera offset, image offset, turn) tau seconds into the gap, between the table's rows."""
        position = min(max(tau / self.step, 0.0), len(self.table) - 1.0)
        row = min(int(position), len(self.table) - 2)
        fraction = position - row
        values = self.table[row] * (1.0 - fraction) + self.table[row + 1] * fraction
        return values[:3], values[3:5], values[5:]


@dataclass
class _Blend:
    """A returning hand's offset from its measured pose, faded out over BLEND_SECONDS."""
    start: float
    position_offset: np.ndarray
    rotation_offset: Quaternion        # applied on top of the measured rotation


@dataclass
class _HandState:
    history: List[_Sample] = field(default_factory=list)  # tracked frames, as measured
    # The pose observe() last returned: a gap starts from what was sent, which differs
    # from the measurement while a returning hand blends in
    sent: Optional[Tuple[np.ndarray, Quaternion]] = None
    motion: Optional[_Motion] = None
    # The optical flow the last prediction followed, if any (hand_flow): see predict()
    flow: Optional[Tuple[np.ndarray, np.ndarray]] = None
    last_prediction: Optional[Prediction] = None
    blend: Optional[_Blend] = None


def _conjugate(q: Quaternion) -> Quaternion:
    return (q[0], -q[1], -q[2], -q[3])


def rotation_vector(q: Quaternion) -> np.ndarray:
    """The rotation of unit quaternion q as axis * angle (radians), along the short arc."""
    w, x, y, z = q
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    sin_half = math.sqrt(x * x + y * y + z * z)
    if sin_half < 1e-9:
        return np.zeros(3)
    angle = 2.0 * math.atan2(sin_half, w)
    return np.array((x, y, z)) * (angle / sin_half)


def quat_from_rotation_vector(v: np.ndarray) -> Quaternion:
    angle = float(np.linalg.norm(v))
    if angle < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    axis = v / angle
    s = math.sin(angle / 2.0)
    return (math.cos(angle / 2.0), float(axis[0] * s), float(axis[1] * s), float(axis[2] * s))


def speed_kept(s: np.ndarray, hold_seconds: float, damping_seconds: float) -> np.ndarray:
    """The fraction of its speed a lost hand keeps s seconds in: all of it for hold_seconds, then decaying."""
    s = np.asarray(s, dtype=float)
    return np.exp(-np.maximum(s - hold_seconds, 0.0) / damping_seconds)


def _integrate(s: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """Offsets from the start, by the trapezoid rule over the times s."""
    steps = 0.5 * (velocities[1:] + velocities[:-1]) * np.diff(s)[:, None]
    return np.vstack((np.zeros((1, velocities.shape[1])), np.cumsum(steps, axis=0)))


def _changing_speed(s: np.ndarray, speed: float, speed_change: float) -> np.ndarray:
    """The speed the fitted change of speed leads to, never backwards nor past MAX_SPEED_GAIN times the start."""
    return np.clip(speed + speed_change * s, 0.0, MAX_SPEED_GAIN * speed)


@dataclass
class Curve:
    """Motion in a plane at the last sample: how fast it was turning and changing speed there."""
    velocity: np.ndarray               # 2D, per second
    turn_rate: float                   # rad/s at the last sample, counterclockwise; the path keeps its bend
    speed_change: float                # per second squared


def fit_curve(times: Sequence[float], points: np.ndarray) -> Curve:
    """
    The curve 2D samples were moving along at the last one: a straight line
    is fitted to the heading of each step between samples, and another to
    their speed, against time, using only the steps since the hand last turned
    back. Unlike a quadratic fit to the points, this follows a swing's arc
    however far it has turned. With fewer than three steps it goes straight on.

    Args:
        times: Sample times in seconds, oldest first
        points: N x 2
    """
    points = np.asarray(points, dtype=float)
    times = np.asarray(times, dtype=float)
    still = Curve(np.zeros(2), 0.0, 0.0)
    if len(times) < 2:
        return still
    dt = np.diff(times)
    if np.any(dt <= 1e-6):
        return still
    middle = 0.5 * (times[1:] + times[:-1]) - times[-1]   # each step's time, from the last sample
    steps = np.diff(points, axis=0)
    # Only the steps since the hand last turned back: across a reversal (a punch
    # pulled back, a shake) the heading flips half a turn, which is no turn rate
    back = np.nonzero(steps @ steps[-1] < 0.0)[0]
    if len(back):
        steps, dt, middle = steps[back[-1] + 1:], dt[back[-1] + 1:], middle[back[-1] + 1:]
    speeds = np.linalg.norm(steps, axis=1) / dt
    if speeds[-1] < 1e-9:
        return still
    headings = np.unwrap(np.arctan2(steps[:, 1], steps[:, 0]))
    if len(steps) < 3:
        # Too few steps to tell a turn or a change of speed from noise: straight on, at their speed
        heading = headings[-1]
        return Curve(float(np.mean(speeds)) * np.array((math.cos(heading), math.sin(heading))), 0.0, 0.0)
    # Long steps tell their heading best: a step of noise points anywhere
    turn_rate, heading = np.polyfit(middle, headings, 1, w=np.maximum(speeds, 1e-9))
    speed_change, speed = np.polyfit(middle, speeds, 1)
    speed = max(float(speed), 0.0)
    return Curve(speed * np.array((math.cos(heading), math.sin(heading))), float(turn_rate), float(speed_change))


def curve_offsets(s: np.ndarray, curve: Curve, hold_seconds: float, damping_seconds: float) -> np.ndarray:
    """
    Offsets along a curve at the times s (from 0): its speed keeps changing at
    its rate and eases off after hold_seconds, and it keeps bending as it was
    (the heading turns with the distance travelled, so an arc keeps its radius
    as the hand slows down instead of spiralling in).

    Returns:
        len(s) x 2 offsets from the start
    """
    speed = float(np.linalg.norm(curve.velocity))
    if speed < 1e-9:
        return np.zeros((len(s), 2))
    speeds = _changing_speed(s, speed, curve.speed_change) * speed_kept(s, hold_seconds, damping_seconds)
    curvature = float(np.clip(curve.turn_rate, -MAX_PATH_TURN_RATE, MAX_PATH_TURN_RATE)) / speed
    limit = math.radians(MAX_PATH_TURN_DEG)
    travelled = _integrate(s, speeds[:, None])[:, 0]
    angle = math.atan2(curve.velocity[1], curve.velocity[0]) + np.clip(curvature * travelled, -limit, limit)
    return _integrate(s, speeds[:, None] * np.column_stack((np.cos(angle), np.sin(angle))))


def line_offsets(s: np.ndarray, velocity: float, acceleration: float,
                 hold_seconds: float, damping_seconds: float) -> np.ndarray:
    """curve_offsets along one axis: len(s) x 1."""
    speed = abs(velocity)
    if speed < 1e-9:
        return np.zeros((len(s), 1))
    speed_change = acceleration if velocity > 0.0 else -acceleration
    speeds = _changing_speed(s, speed, speed_change) * speed_kept(s, hold_seconds, damping_seconds)
    return _integrate(s, math.copysign(1.0, velocity) * speeds[:, None])


def fit_motion(times: Sequence[float], points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Velocity and acceleration at the last sample, from a quadratic fitted to
    the samples by least squares (a straight line with two, nothing with one).

    Args:
        times: Sample times in seconds, oldest first
        points: One row per sample

    Returns:
        (velocity, acceleration) per second and per second squared
    """
    points = np.asarray(points, dtype=float)
    zero = np.zeros(points.shape[1])
    s = np.asarray(times, dtype=float) - times[-1]
    if len(s) < 2 or s[0] > -1e-6:
        return zero, zero
    if len(s) == 2:
        return (points[1] - points[0]) / (s[1] - s[0]), zero
    design = np.column_stack((np.ones_like(s), s, s * s))
    coefficients, *_ = np.linalg.lstsq(design, points, rcond=None)
    return coefficients[1], 2.0 * coefficients[2]


def prediction_horizon(prediction_seconds: float) -> float:
    """The prediction_seconds setting as used: 0 (off) to MAX_PREDICTION_SECONDS."""
    return min(max(float(prediction_seconds), 0.0), MAX_PREDICTION_SECONDS)


class MotionPredictor:
    """Per hand: records tracked poses, predicts the pose while lost, blends it back on return."""

    def __init__(self, prediction_seconds: float = DEFAULTS["prediction_seconds"]):
        """
        Args:
            prediction_seconds: How long a lost hand keeps moving (0 turns
                prediction off; at most MAX_PREDICTION_SECONDS). Past it the
                hand holds its last predicted pose.
        """
        self.prediction_seconds = prediction_horizon(prediction_seconds)
        self._hands: Dict[str, _HandState] = {}

    def forget(self, hand: str) -> None:
        """Drop a hand's history: the next tracked frame is a new hand, nothing to predict or blend from."""
        self._hands.pop(hand, None)

    def observe(self, hand: str, t: float, camera_position: Sequence[float], rotation: Quaternion,
                image_wrist: Sequence[float]) -> Tuple[Tuple[float, float, float], Quaternion]:
        """
        Record a tracked pose.

        Args:
            hand: "left" or "right"
            t: Time in seconds
            camera_position: Wrist in metres, camera space: x and y across the
                picture, z along the camera's view (depth, the noisiest)
            rotation: Hand rotation, unit quaternion (qw, qx, qy, qz)
            image_wrist: Wrist in normalised image coordinates

        Returns:
            (camera_position, rotation) to use: the measured pose, or, for a
            hand found again after a prediction, a blend from the predicted one
        """
        position = np.asarray(camera_position, dtype=float)
        sample = _Sample(t, position, tuple(float(c) for c in rotation), np.asarray(image_wrist, dtype=float)[:2])
        if not (np.all(np.isfinite(position)) and all(math.isfinite(c) for c in sample.rotation)
                and np.all(np.isfinite(sample.image_wrist))):
            # Never part of a trajectory; the caller skips sending such a pose
            return tuple(float(v) for v in position), sample.rotation
        state = self._hands.setdefault(hand, _HandState())
        if state.last_prediction is not None:
            # Blend in from where the hand would be shown now, so the first frame back doesn't stall
            shown = self._evaluate(state, state.motion,
                                   min(t - state.history[-1].time, self.prediction_seconds), state.flow)
            state.blend = _Blend(t, shown.camera_position - position,
                                 quat_multiply(shown.rotation, _conjugate(sample.rotation)))
        state.motion = None
        state.flow = None
        state.last_prediction = None
        # Measured frames only, gap or not: a fit to what was sent would take the
        # blend's own correction for motion, and errors would build up over
        # repeated dropouts. Frames from before a gap still count while recent,
        # so a hand lost again right after it is found keeps its speed.
        state.history = [s for s in state.history if t - s.time <= HISTORY_SECONDS][-(HISTORY_SAMPLES - 1):]
        state.history.append(sample)
        position, rotation = self._blended(state, sample)
        state.sent = (position, rotation)
        return (float(position[0]), float(position[1]), float(position[2])), rotation

    @staticmethod
    def _blended(state: _HandState, sample: _Sample) -> Tuple[np.ndarray, Quaternion]:
        position, rotation = sample.camera_position, sample.rotation
        blend = state.blend
        if blend is not None:
            remaining = 1.0 - (sample.time - blend.start) / BLEND_SECONDS
            if remaining <= 0.0:
                state.blend = None
            else:
                weight = remaining * remaining * (3.0 - 2.0 * remaining)  # smoothstep: no kink at either end
                position = position + weight * blend.position_offset
                rotation = quat_multiply(quat_slerp((1.0, 0.0, 0.0, 0.0), blend.rotation_offset, weight), rotation)
        return position, rotation

    def predict(self, hand: str, t: float, flow_offset: Optional[Sequence[float]] = None,
                image_span: Optional[Sequence[float]] = None) -> Optional[Prediction]:
        """
        A lost hand's pose at time t, or None when there is nothing to predict:
        prediction is off, the hand has no history, or its last tracked frame is
        more than prediction_seconds old (it holds where it was last sent).

        Args:
            hand: "left" or "right"
            t: Time in seconds
            flow_offset: How far the hand's pixels moved across the picture since
                its last tracked frame, measured by optical flow (hand_flow), in
                normalised image coordinates. Given, the hand goes there instead
                of along its fitted path; its depth and rotation still follow
                the fit. None predicts from the tracked frames alone.
            image_span: With flow_offset: camera x and y, in metres, across a
                whole picture width and height one metre from the camera
                (signed, as the camera maps the picture)
        """
        state = self._hands.get(hand)
        if state is None or not state.history or self.prediction_seconds <= 0.0:
            return None
        start = state.history[-1]
        tau = t - start.time
        if not 0.0 < tau <= self.prediction_seconds:
            return None
        if state.motion is None:
            state.motion = self._fit(state.history, state.sent, self.prediction_seconds)
        if flow_offset is not None and image_span is not None:
            state.flow = (np.asarray(flow_offset, dtype=float)[:2], np.asarray(image_span, dtype=float)[:2])
        prediction = self._evaluate(state, state.motion, tau, state.flow)
        state.last_prediction = prediction
        return prediction

    @staticmethod
    def _fit(history: List[_Sample], sent: Tuple[np.ndarray, Quaternion], horizon: float) -> _Motion:
        # The motion starts from the last frame, as sent, even when the fit leaves it out as a jump
        start = history[-1]
        fitted = _without_jump(history)
        times = [s.time for s in fitted]
        positions = np.array([s.camera_position for s in fitted])
        # Across the picture the hand follows a curve; depth, a line (and only when its frames agree)
        across = fit_curve(times, positions[:, :2])
        image = fit_curve(times, np.array([s.image_wrist for s in fitted]))
        depth_velocity, depth_acceleration = (float(v[0]) for v in fit_motion(times, positions[:, 2:]))
        if not _depth_agrees(positions[:, 2]):
            depth_velocity = depth_acceleration = 0.0

        # Average turn rate over the history, so one noisy frame doesn't decide it
        angular_velocity = np.zeros(3)
        if len(fitted) >= 2 and times[-1] > times[0]:
            turn = sum((rotation_vector(quat_multiply(b.rotation, _conjugate(a.rotation)))
                        for a, b in zip(fitted, fitted[1:])), np.zeros(3))
            angular_velocity = turn / (times[-1] - times[0])

        if np.linalg.norm(image.velocity) < MIN_IMAGE_SPEED:
            # Barely moving across the picture. A punch straight at or away from
            # the camera barely crosses it either, so steady depth still moves.
            across = image = Curve(np.zeros(2), 0.0, 0.0)
            angular_velocity = np.zeros(3)
        # Faster is a misread: the cap scales the change of speed too
        speed = math.hypot(float(np.linalg.norm(across.velocity)), depth_velocity)
        scale = min(1.0, MAX_SPEED_M_S / max(speed, 1e-9))
        across = Curve(across.velocity * scale, across.turn_rate, across.speed_change * scale)
        depth_velocity, depth_acceleration = depth_velocity * scale, depth_acceleration * scale

        hold, damping = MOMENTUM_SECONDS, POSITION_DAMPING_SECONDS
        leaving = _is_leaving(start.image_wrist, image.velocity,
                              start.image_wrist + image.velocity * LEAVING_LOOKAHEAD_SECONDS)
        if leaving:
            hold, damping = 0.0, LEAVING_DAMPING_SECONDS

        s = np.linspace(0.0, horizon, max(1, math.ceil(horizon / TRAJECTORY_STEP)) + 1)
        turn_damping = min(damping, ROTATION_DAMPING_SECONDS)
        turns = angular_velocity * _integrate(s, speed_kept(s, hold, turn_damping)[:, None])
        limit = math.radians(MAX_EXTRA_TURN_DEG)
        turns = turns * np.minimum(1.0, limit / np.maximum(np.linalg.norm(turns, axis=1, keepdims=True), 1e-12))
        image_offsets = curve_offsets(s, image, hold, damping)
        table = np.column_stack((
            curve_offsets(s, across, hold, damping),
            line_offsets(s, depth_velocity, depth_acceleration, hold, min(damping, DEPTH_DAMPING_SECONDS)),
            image_offsets, turns))
        # The path ahead is the same on every frame of the gap: drawn from here
        ahead = np.linspace(0, len(s) - 1, PATH_POINTS + 1).round().astype(int)[1:]
        return _Motion(start=start, position=sent[0], rotation=sent[1], step=float(s[1] - s[0]), table=table,
                       path_ahead=start.image_wrist + image_offsets[ahead], leaving=leaving)

    def _evaluate(self, state: _HandState, motion: _Motion, tau: float,
                  flow: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Prediction:
        camera_offset, image_offset, turn = motion.at(tau)
        rotation = quat_multiply(quat_from_rotation_vector(turn), motion.rotation)
        position = motion.position + camera_offset
        history = np.array([s.image_wrist for s in state.history])
        if flow is None:
            path = np.vstack([history, motion.path_ahead])
        else:
            image_offset, span = flow
            position = _moved_across(motion.position, position[2], image_offset, span)
            path = np.vstack([history, motion.start.image_wrist + image_offset])
        return Prediction(camera_position=position, rotation=rotation,
                          image_wrist=motion.start.image_wrist + image_offset, image_offset=image_offset,
                          leaving=motion.leaving, path=path)


def _moved_across(start: np.ndarray, depth: float, image_offset: np.ndarray, span: np.ndarray) -> np.ndarray:
    """
    Camera position of a wrist that started at start, now at camera z depth,
    whose picture moved by image_offset: x and y keep their place in the
    picture as the depth changes, plus the offset at the new depth.
    """
    if start[2] > -1e-3 or depth > -1e-3:
        return np.array((start[0], start[1], depth))   # at or behind the camera: no picture to follow
    distance = -depth
    across = start[:2] * (depth / start[2]) + image_offset * span * distance
    return np.array((across[0], across[1], depth))


def _without_jump(history: List[_Sample]) -> List[_Sample]:
    """The history without its newest frame when that frame's wrist jumped far beyond how the hand was moving."""
    if len(history) < 3:
        return history
    wrists = np.array([s.image_wrist for s in history])
    times = np.array([s.time for s in history])
    steps = np.linalg.norm(np.diff(wrists, axis=0), axis=1)
    speeds = steps / np.maximum(np.diff(times), 1e-6)
    usual = float(np.median(speeds[:-1]))
    if usual < MIN_IMAGE_SPEED:
        jumped = steps[-1] >= JUMP_FROM_REST_IMAGE
    else:
        jumped = steps[-1] >= JUMP_MIN_IMAGE and speeds[-1] > JUMP_RATIO * usual
    return history[:-1] if jumped else history


def _depth_agrees(depths: np.ndarray) -> bool:
    """Whether the tracked frames move steadily in depth: enough steps, all one way, none making most of the move."""
    steps = np.diff(depths)
    steps = steps[np.abs(steps) >= DEPTH_NOISE_M]
    if len(steps) < max(DEPTH_MIN_STEPS, 1) or not (np.all(steps > 0.0) or np.all(steps < 0.0)):
        return False
    return float(np.max(np.abs(steps))) <= DEPTH_MAX_STEP_SHARE * abs(float(depths[-1] - depths[0]))


def _is_leaving(wrist: np.ndarray, velocity: np.ndarray, end: np.ndarray) -> bool:
    """Whether a wrist was on its way out of the picture: near an edge and moving toward it, or headed past one."""
    if np.any(end < 0.0) or np.any(end > 1.0):
        return True
    near_low = (wrist < EDGE_MARGIN) & (velocity < 0.0)
    near_high = (wrist > 1.0 - EDGE_MARGIN) & (velocity > 0.0)
    return bool(np.any(near_low | near_high))
