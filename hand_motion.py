"""
Keeps a hand moving through short tracking gaps instead of freezing it.

On a fast move motion blur makes MediaPipe lose the hand for a few frames: the
hand would stop where it was last seen and jump when it is found again. While
a hand is lost, its pose follows the trajectory of its last tracked frames,
slowing down over time, for at most prediction_seconds. A hand that was
leaving the picture eases to a stop instead, and one that was barely moving
holds still. When the hand is found again, the pose blends from where it was
predicted to where it is measured.

Each lost hand moves with a velocity that decays exponentially:
v(s) = (v0 + a0 * s) * exp(-s / T), whose integral has a closed form and a
finite end point, so a prediction always comes to rest.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from gesture_detector import Quaternion, quat_multiply, quat_slerp

DEFAULTS = {
    "prediction_seconds": 0.3,  # how long a lost hand keeps moving; 0 turns prediction off
}
MAX_PREDICTION_SECONDS = 1.0

# Tracked frames the trajectory is fitted to: the newest ones within this time
HISTORY_SECONDS = 0.15
HISTORY_SAMPLES = 8
# Time constants of the velocity's decay while lost. Depth from one camera is
# far noisier than the picture's axes (a turning wrist alone changes it), and
# rotation is noisier too, so both slow down sooner; a hand leaving the picture
# stops sooner still. Depth only moves at all when its frames agree (see
# DEPTH_NOISE_M); on the recorded sessions such depth motion, damped over 0.05
# to 0.2 s, ended as far from where the hand was found either way, and 0.1 s
# lets a punch carry on instead of stopping a few centimetres in.
POSITION_DAMPING_SECONDS = 0.2
DEPTH_DAMPING_SECONDS = 0.1
ROTATION_DAMPING_SECONDS = 0.1
LEAVING_DAMPING_SECONDS = 0.05
# A hand lost while its wrist crosses the picture slower than this (in picture
# sizes per second: widths across, heights down) holds still, turning included:
# its speed would be mostly noise, and on the
# recorded clips moving it ended further from where it was found than holding.
MIN_IMAGE_SPEED = 0.5
# The fastest a lost hand is moved. Hands do swing faster, but the speed
# measured just before a gap is mostly noise above this: on a recorded VR
# session it cut the furthest a lost hand travelled from 18 to 12 cm, and
# lost hands ended as close to where they were found as with 4 m/s.
MAX_SPEED_M_S = 1.5
# Depth keeps moving through a gap only when the tracked frames agree on it:
# every step at least DEPTH_NOISE_M goes the same way, and no single step makes
# most of the move. One camera's depth jumps by centimetres when the wrist turns
# or a finger hides, and on the recorded clips extrapolating such a jump sent
# lost hands 8-14 cm away within a frame or two.
DEPTH_NOISE_M = 0.005
DEPTH_MIN_STEPS = 2
DEPTH_MAX_STEP_SHARE = 0.6
# A tracked wrist that jumps this many times further than the frames before it
# moved, and at least JUMP_MIN_IMAGE of the picture, is a misdetection or a
# hand mistaken for the other: the fit leaves it out
JUMP_RATIO = 3.0
JUMP_MIN_IMAGE = 0.05
# A bad rotation estimate can't spin the hand further than this
MAX_EXTRA_TURN_DEG = 45.0
# A wrist this close to the picture's edge (fraction of the image), moving toward it, is leaving
EDGE_MARGIN = 0.05
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
    velocity: np.ndarray               # m/s
    acceleration: np.ndarray           # m/s^2
    image_velocity: np.ndarray         # image widths/heights per second
    image_acceleration: np.ndarray
    angular_velocity: np.ndarray       # rad/s, as a rotation vector
    damping_seconds: np.ndarray        # per camera axis: x, y, then depth
    image_damping_seconds: float
    leaving: bool


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


def damped_distance(tau: float, damping_seconds: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    How far a unit velocity and a unit acceleration carry in tau seconds when
    both decay with exp(-s / damping_seconds): the integrals of exp(-s/T) and
    s * exp(-s/T) from 0 to tau. damping_seconds may be one value per axis.
    """
    k = 1.0 / np.asarray(damping_seconds, dtype=float)
    decay = np.exp(-k * tau)
    return (1.0 - decay) / k, (1.0 - decay * (1.0 + k * tau)) / (k * k)


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


def _limit_acceleration(velocity: np.ndarray, acceleration: np.ndarray, damping_seconds: float) -> np.ndarray:
    """
    At most |velocity| / damping_seconds, so the acceleration can bend the path
    or slow it down, but the hand never ends up behind where it was lost (it
    may turn back after damping_seconds), nor further than twice where its
    velocity alone would take it.
    """
    limit = float(np.linalg.norm(velocity)) / damping_seconds
    size = float(np.linalg.norm(acceleration))
    return acceleration * (limit / size) if size > limit else acceleration


def prediction_horizon(prediction_seconds: float) -> float:
    """The prediction_seconds setting as used: 0 (off) to MAX_PREDICTION_SECONDS."""
    return min(max(float(prediction_seconds), 0.0), MAX_PREDICTION_SECONDS)


def _clamp_norm(v: np.ndarray, limit: float) -> np.ndarray:
    size = float(np.linalg.norm(v))
    return v * (limit / size) if size > limit else v


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
                                   min(t - state.history[-1].time, self.prediction_seconds))
            state.blend = _Blend(t, shown.camera_position - position,
                                 quat_multiply(shown.rotation, _conjugate(sample.rotation)))
        state.motion = None
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

    def predict(self, hand: str, t: float) -> Optional[Prediction]:
        """
        A lost hand's pose at time t, or None when there is nothing to predict:
        prediction is off, the hand has no history, or its last tracked frame is
        more than prediction_seconds old (it holds where it was last sent).
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
        prediction = self._evaluate(state, state.motion, tau)
        state.last_prediction = prediction
        return prediction

    @staticmethod
    def _fit(history: List[_Sample], sent: Tuple[np.ndarray, Quaternion], horizon: float) -> _Motion:
        start = history[-1]
        history = _without_jump(history)
        times = [s.time for s in history]
        positions = np.array([s.camera_position for s in history])
        velocity, acceleration = fit_motion(times, positions)
        image_velocity, image_acceleration = fit_motion(times, np.array([s.image_wrist for s in history]))
        if not _depth_agrees(positions[:, 2]):
            velocity[2] = acceleration[2] = 0.0

        # Average turn rate over the history, so one noisy frame doesn't decide it
        angular_velocity = np.zeros(3)
        if len(history) >= 2 and times[-1] > times[0]:
            turn = sum((rotation_vector(quat_multiply(b.rotation, _conjugate(a.rotation)))
                        for a, b in zip(history, history[1:])), np.zeros(3))
            angular_velocity = turn / (times[-1] - times[0])

        if np.linalg.norm(image_velocity) < MIN_IMAGE_SPEED:
            velocity, acceleration = np.zeros(3), np.zeros(3)
            image_velocity, image_acceleration = np.zeros(2), np.zeros(2)
            angular_velocity = np.zeros(3)

        damping = POSITION_DAMPING_SECONDS
        leaving = _is_leaving(start.image_wrist, image_velocity,
                              start.image_wrist + image_velocity * damped_distance(horizon, damping)[0])
        if leaving:
            damping = LEAVING_DAMPING_SECONDS
        velocity = _clamp_norm(velocity, MAX_SPEED_M_S)
        return _Motion(start=start, position=sent[0], rotation=sent[1], velocity=velocity,
                       acceleration=_limit_acceleration(velocity, acceleration, damping),
                       image_velocity=image_velocity,
                       image_acceleration=_limit_acceleration(image_velocity, image_acceleration, damping),
                       angular_velocity=angular_velocity,
                       damping_seconds=np.array((damping, damping, min(damping, DEPTH_DAMPING_SECONDS))),
                       image_damping_seconds=damping, leaving=leaving)

    def _evaluate(self, state: _HandState, motion: _Motion, tau: float) -> Prediction:
        start = motion.start
        by_velocity, by_acceleration = damped_distance(tau, motion.damping_seconds)
        camera_offset = motion.velocity * by_velocity + motion.acceleration * by_acceleration
        by_velocity, by_acceleration = damped_distance(tau, motion.image_damping_seconds)
        image_offset = motion.image_velocity * by_velocity + motion.image_acceleration * by_acceleration

        turn_damping = min(ROTATION_DAMPING_SECONDS, motion.image_damping_seconds)
        turn = motion.angular_velocity * damped_distance(tau, turn_damping)[0]
        turn = _clamp_norm(turn, math.radians(MAX_EXTRA_TURN_DEG))
        rotation = quat_multiply(quat_from_rotation_vector(turn), motion.rotation)

        image_wrist = start.image_wrist + image_offset
        return Prediction(camera_position=motion.position + camera_offset, rotation=rotation,
                          image_wrist=image_wrist, image_offset=image_offset,
                          leaving=motion.leaving, path=self._path(state.history, motion))

    def _path(self, history: List[_Sample], motion: _Motion) -> np.ndarray:
        """Tracked wrists, then the predicted path out to prediction_seconds."""
        ahead = []
        for i in range(1, PATH_POINTS + 1):
            by_velocity, by_acceleration = damped_distance(self.prediction_seconds * i / PATH_POINTS,
                                                           motion.image_damping_seconds)
            ahead.append(motion.start.image_wrist + motion.image_velocity * by_velocity
                         + motion.image_acceleration * by_acceleration)
        return np.array([s.image_wrist for s in history] + ahead)


def _without_jump(history: List[_Sample]) -> List[_Sample]:
    """The history without its newest frame when that frame's wrist jumped far beyond how the hand was moving."""
    if len(history) < 3:
        return history
    wrists = np.array([s.image_wrist for s in history])
    times = np.array([s.time for s in history])
    steps = np.linalg.norm(np.diff(wrists, axis=0), axis=1)
    speeds = steps / np.maximum(np.diff(times), 1e-6)
    if steps[-1] >= JUMP_MIN_IMAGE and speeds[-1] > JUMP_RATIO * float(np.median(speeds[:-1])):
        return history[:-1]
    return history


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
