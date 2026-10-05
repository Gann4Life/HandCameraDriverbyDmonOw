"""
Rebuild a hand in 3D from MediaPipe's 2D landmarks alone.

Seen from behind (POV), MediaPipe places the 2D points well but gets the 3D
shape of the hand wrong: knuckle depths flip and the hand folds onto itself.
Everything derived from that shape (distance, rotation, finger curl) inherits
the error. Here a hand model of fixed size, whose fingers only bend toward the
palm, is fitted to the 2D points instead, so the only 3D shapes on offer are
ones a real hand can make.

Axes: OpenCV camera space, metres (x right, y down, z forward), like
MediaPipe's world landmarks. Model space is the palm frame of a right hand:
x across the knuckles (pinky to index), y forward (wrist to knuckles), z out of
the palm. A left hand is its mirror image (z reflected).
"""
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

WRIST = 0
PALM = [0, 1, 5, 9, 13, 17]
KNUCKLES = [5, 9, 13, 17]
THUMB = [1, 2, 3, 4]
FINGERS = [[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]

# Parameters: rotation vector (3), translation (3), thumb [swing, turn, bend, bend],
# then each finger [knuckle bend, spread, middle bend, last bend]
N_PARAMS = 6 + 4 + 4 * 4
LOWER = np.r_[[-np.inf] * 5, 0.08, -0.6, -0.8, -0.3, -0.3, [-0.35, -0.35, -0.05, -0.05] * 4]
UPPER = np.r_[[np.inf] * 5, 2.0, 1.0, 0.8, 1.2, 1.4, [1.65, 0.35, 1.9, 1.5] * 4]
# Typical step size of each parameter, so the solver treats them alike
SCALE = np.r_[[0.3] * 3, 0.05, 0.05, 0.1, [0.3] * 20]


# MediaPipe's average hand, right-handed, in the palm frame (metres): the median
# of its world landmarks over two thousand tracked hands. Only the palm, the
# bone lengths and the thumb's first bone are used, so the fingers' bend here
# does not matter. Same size as MediaPipe's model, so hand_scale keeps meaning
# what it did.
DEFAULT_HAND = np.array([
    [0.0000, 0.0000, 0.0000],
    [0.0282, 0.0235, 0.0119],
    [0.0491, 0.0485, 0.0230],
    [0.0649, 0.0798, 0.0277],
    [0.0723, 0.1078, 0.0250],
    [0.0327, 0.0897, 0.0021],
    [0.0307, 0.1110, 0.0147],
    [0.0280, 0.1219, 0.0330],
    [0.0169, 0.1291, 0.0509],
    [0.0088, 0.0906, -0.0024],
    [0.0033, 0.1162, 0.0176],
    [0.0013, 0.1214, 0.0400],
    [-0.0015, 0.1440, 0.0376],
    [-0.0123, 0.0860, -0.0010],
    [-0.0167, 0.1059, 0.0169],
    [-0.0227, 0.1065, 0.0382],
    [-0.0211, 0.1266, 0.0333],
    [-0.0289, 0.0701, 0.0021],
    [-0.0334, 0.0890, 0.0075],
    [-0.0419, 0.0832, 0.0233],
    [-0.0339, 0.0968, 0.0199],
])


def palm_frame(points: np.ndarray) -> np.ndarray:
    """Rotation whose columns are the palm axes of a hand's 21 points (see module docstring)."""
    forward = points[KNUCKLES].mean(0) - points[WRIST]
    y = forward / np.linalg.norm(forward)
    across = points[5] - points[17]
    x = across - (across @ y) * y
    x /= np.linalg.norm(x)
    return np.column_stack((x, y, np.cross(x, y)))


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """np.cross for (..., 3) arrays, without its overhead (this runs thousands of times a second)."""
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    return np.stack((a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0), axis=-1)


def _rotation_matrices(rotvecs: np.ndarray) -> np.ndarray:
    """(B, 3) rotation vectors -> (B, 3, 3) matrices (Rodrigues)."""
    angle = np.linalg.norm(rotvecs, axis=1)
    k = rotvecs / np.maximum(angle, 1e-12)[:, None]
    s, c = np.sin(angle), np.cos(angle)
    x, y, z = k[:, 0], k[:, 1], k[:, 2]
    t = 1 - c
    return np.stack((
        np.stack((c + x * x * t, x * y * t - z * s, x * z * t + y * s), axis=-1),
        np.stack((y * x * t + z * s, c + y * y * t, y * z * t - x * s), axis=-1),
        np.stack((z * x * t - y * s, z * y * t + x * s, c + z * z * t), axis=-1)), axis=1)


class HandShape:
    """A hand's fixed proportions: palm points and bone lengths, in the right-hand palm frame."""

    # Which way the thumb bends: from into the palm (0) toward the little finger (90)
    THUMB_BEND_TILT_DEG = 45.0

    def __init__(self, points: np.ndarray):
        """
        Args:
            points: 21 x 3 joints of a right hand in its palm frame, wrist at
                the origin, palm side toward +z (see from_samples)
        """
        self.points = np.asarray(points, dtype=float)
        p = self.points
        self.palm = p[PALM]
        n = np.array([0.0, 0.0, 1.0])
        self.normal = n
        # Thumb: rest direction of its first bone as measured; it swings about
        # the axis across it, turns about the palm normal and bends across the palm
        self.thumb_root = p[1]
        self.thumb_rest = _unit(p[2] - p[1])
        self.thumb_swing_axis = _unit(np.cross(self.thumb_rest, n))
        self._thumb_swing_dir = np.cross(self.thumb_swing_axis, self.thumb_rest)
        tilt = np.radians(self.THUMB_BEND_TILT_DEG)
        self._thumb_bend_toward = np.array([-np.sin(tilt), 0.0, np.cos(tilt)])  # -x is toward the little finger
        self.thumb_lengths = np.linalg.norm(np.diff(p[THUMB], axis=0), axis=1)
        # Fingers: straight along their knuckle's direction, flat on the palm plane
        roots = np.array([p[chain[0]] for chain in FINGERS])
        rest = roots - p[WRIST]
        rest[:, 2] = 0.0
        self.finger_roots = roots
        self.finger_rest = _unit(rest)
        self.finger_side = np.cross(n, self.finger_rest)  # where a spread turns them
        self.finger_lengths = np.array([np.linalg.norm(np.diff(p[chain], axis=0), axis=1) for chain in FINGERS])

    @classmethod
    def default(cls, scale: float = 1.0) -> "HandShape":
        """MediaPipe's average hand, scaled (calibration.hand_scale)."""
        return cls(DEFAULT_HAND * scale)

    @classmethod
    def from_samples(cls, samples: Sequence[np.ndarray], right: Sequence[bool]) -> "HandShape":
        """
        Average hand from many 3D hands (e.g. MediaPipe world landmarks), each
        given with whether it is a right hand. Medians, so a few bad frames do
        not matter; left hands are mirrored into right ones.
        """
        frames = []
        for pts, is_right in zip(samples, right):
            pts = np.asarray(pts, dtype=float) - pts[WRIST]
            local = pts @ palm_frame(pts)
            if not is_right:
                local[:, 2] = -local[:, 2]
            frames.append(local)
        frames = np.array(frames)
        # Which side the fingers curl to tells the palm side; keep it on +z
        curl = np.median(frames[:, [8, 12, 16, 20], 2] - frames[:, KNUCKLES, 2])
        if curl < 0:
            frames[:, :, 2] = -frames[:, :, 2]
        median = np.median(frames, axis=0)
        # Bone lengths as medians too, laid along the median directions
        for chain in [THUMB] + FINGERS:
            for a, b in zip(chain, chain[1:]):
                length = np.median(np.linalg.norm(frames[:, b] - frames[:, a], axis=1))
                direction = _unit(median[b] - median[a])
                median[b] = median[a] + length * direction
        return cls(median)

    def local_points(self, joints: np.ndarray) -> np.ndarray:
        """Joints of a right hand in its palm frame, for B sets of 20 joint angles: (B, 20) -> (B, 21, 3)."""
        batch = joints.shape[0]
        out = np.empty((batch, 21, 3))
        out[:, PALM] = self.palm
        n = self.normal
        # Thumb: swing its first bone across its rest direction, turn it about
        # the palm normal (z), then bend the next two joints the same way
        swing, turn = joints[:, 0, None], joints[:, 1]
        d = self.thumb_rest * np.cos(swing) + self._thumb_swing_dir * np.sin(swing)
        st, ct = np.sin(turn), np.cos(turn)
        d = np.stack((d[:, 0] * ct - d[:, 1] * st, d[:, 0] * st + d[:, 1] * ct, d[:, 2]), axis=-1)
        # The thumb bends across the palm toward the little finger, not
        # straight into it like the fingers do
        bend_dir = self._thumb_bend_toward - np.sum(self._thumb_bend_toward * d, axis=1, keepdims=True) * d
        bend_dir /= np.maximum(np.linalg.norm(bend_dir, axis=1, keepdims=True), 1e-9)
        angles = np.stack((np.zeros(batch), joints[:, 2], joints[:, 2] + joints[:, 3]), axis=-1)[..., None]
        bones = (d[:, None] * np.cos(angles) + bend_dir[:, None] * np.sin(angles)) * self.thumb_lengths[:, None]
        out[:, THUMB[1:]] = self.thumb_root + np.cumsum(bones, axis=1)
        # Fingers, all four at once. Each stays in one plane: spread turns it
        # about the palm normal, then its joints bend it toward the palm.
        q = joints[:, 4:].reshape(batch, 4, 4)
        spread = q[:, :, 1, None]
        d = self.finger_rest * np.cos(spread) + self.finger_side * np.sin(spread)  # (B, 4, 3)
        angles = np.cumsum(np.stack((q[:, :, 0], q[:, :, 2], q[:, :, 3]), axis=-1), axis=-1)  # (B, 4, 3)
        bones = (d[:, :, None, :] * np.cos(angles)[..., None] + n * np.sin(angles)[..., None]
                 ) * self.finger_lengths[None, :, :, None]
        tips = self.finger_roots[None, :, None, :] + np.cumsum(bones, axis=2)  # (B, 4, 3 joints, 3)
        for f, chain in enumerate(FINGERS):
            out[:, chain[1:]] = tips[:, f]
        return out


def bend_angles(points: np.ndarray) -> np.ndarray:
    """
    Joint angles (as fitted, 20 values) read from a 3D hand, e.g. MediaPipe's
    world landmarks. Only how far each finger joint bends is read, which
    MediaPipe gets about right even when the hand's overall 3D shape is off;
    spread and the thumb's base stay at rest.
    """
    p = np.asarray(points, dtype=float)
    joints = np.zeros(20)
    thumb = _unit(np.diff(p[THUMB], axis=0))
    joints[2:4] = np.arccos(np.clip(np.sum(thumb[:-1] * thumb[1:], axis=1), -1.0, 1.0))
    for f, chain in enumerate(FINGERS):
        bones = [p[chain[0]] - p[WRIST]] + [p[b] - p[a] for a, b in zip(chain, chain[1:])]
        bones = _unit(np.array(bones))
        bends = np.arccos(np.clip(np.sum(bones[:-1] * bones[1:], axis=1), -1.0, 1.0))
        joints[4 + 4 * f + 0] = bends[0]
        joints[4 + 4 * f + 2] = bends[1]
        joints[4 + 4 * f + 3] = bends[2]
    return np.clip(joints, LOWER[6:], UPPER[6:])


# Parameters bend_angles reads: the thumb's two outer joints and each finger's three
BENDS = np.r_[8, 9, [10 + 4 * f + j for f in range(4) for j in (0, 2, 3)]]

# A closed hand, to start from when an open one does not fit
FIST = np.r_[0.6, 0.4, 0.6, 0.6, [1.3, 0.0, 1.5, 1.0] * 4]


@dataclass
class HandFit:
    params: np.ndarray       # N_PARAMS
    points: np.ndarray       # 21 x 3, camera space (OpenCV axes), metres
    rms_px: float            # how far the model's joints land from MediaPipe's, in pixels
    palm_to_camera: float    # cosine: > 0 when the palm faces the camera
    refreshed: bool          # whether this fit started over from fresh poses


class HandFitter:
    """Fits a HandShape to one hand's 2D landmarks, frame after frame."""

    # Pixel noise of MediaPipe's 2D landmarks
    PIXEL_SIGMA = 4.0
    # How much a parameter may move between frames before it costs as much as
    # a pixel sigma of misfit
    MOTION_SIGMA = np.r_[[0.25] * 3, 0.03, 0.03, 0.06, [0.35] * 20]
    # How far the fit may bend a joint away from the bends MediaPipe read
    # before that costs as much as a pixel sigma of misfit. Loose: they only
    # settle what the 2D points cannot, like a thumb tucked in or pointing
    # at the camera.
    BEND_SIGMA = np.r_[0.25, 0.25, [0.5] * 12]  # tighter on the thumb, often hidden behind the fist
    # Seen from behind, a palm facing the camera needs extra evidence
    PALM_AWAY_WEIGHT = 30.0
    # A warm start fitting worse than this (RMS pixels) also tries a fresh start
    REFIT_RMS_PX = 8.0
    # A warm start whose joints are further than this (radians) from the bends
    # MediaPipe read also tries last frame's pose with MediaPipe's bends. A
    # finger curled toward the camera looks almost like a straight one, so a
    # warm start can stay curled after the finger opens, with a small pixel error.
    BENDS_DISAGREE = 0.8
    ITERATIONS = 4
    FRESH_ITERATIONS = 8
    SCOUT_ITERATIONS = 2
    FINISHED_STARTS = 2
    DAMPING_TRIES = (1.0, 10.0, 100.0)
    STEP = 1e-5

    def __init__(self, shape: HandShape, focal: float, centre: Tuple[float, float]):
        self.shape = shape
        self.focal = focal
        self.centre = np.asarray(centre, dtype=float)
        self.camera_matrix = np.array([[focal, 0, centre[0]], [0, focal, centre[1]], [0, 0, 1.0]])
        self._bends: Optional[np.ndarray] = None  # this fit's bend_angles, at BENDS

    # ---- model
    def camera_points(self, params: np.ndarray, is_right: bool) -> np.ndarray:
        """(B, N_PARAMS) -> (B, 21, 3) joints in camera space."""
        pts = self.shape.local_points(params[:, 6:])
        if not is_right:
            pts = pts * np.array([1.0, 1.0, -1.0])
        return pts @ _rotation_matrices(params[:, :3]).transpose(0, 2, 1) + params[:, None, 3:6]

    def project(self, points: np.ndarray) -> np.ndarray:
        return self.focal * points[..., :2] / points[..., 2:3] + self.centre

    def _residuals(self, params: np.ndarray, observed: np.ndarray, is_right: bool,
                   previous: Optional[np.ndarray]) -> np.ndarray:
        """(B, N_PARAMS) -> (B, M) weighted residuals."""
        batch = params.shape[0]
        parts = [((self.project(self.camera_points(params, is_right)) - observed) / self.PIXEL_SIGMA
                  ).reshape(batch, -1)]
        q = params[:, 10:].reshape(batch, 4, 4)
        parts.append((q[:, :, 3] - 0.7 * q[:, :, 2]) / 0.3)  # a finger's last joint follows its middle one
        if previous is not None:
            parts.append((params - previous) / self.MOTION_SIGMA)
        if self._bends is not None:
            parts.append((params[:, BENDS] - self._bends) / self.BEND_SIGMA)
        return np.concatenate(parts, axis=1)

    def palm_to_camera(self, params: np.ndarray, is_right: bool) -> float:
        rotation, _ = cv2.Rodrigues(params[:3])
        normal = rotation @ (self.shape.normal * (1.0 if is_right else -1.0))
        centre = rotation @ self.shape.palm.mean(0) + params[3:6]
        return float(normal @ (-centre / np.linalg.norm(centre)))

    # ---- solver
    def _solve(self, start: np.ndarray, observed: np.ndarray, is_right: bool,
               previous: Optional[np.ndarray], iterations: int) -> Tuple[np.ndarray, float]:
        """Levenberg-Marquardt within the joint limits, forward-difference Jacobian in one batch."""
        x = np.clip(start, LOWER, UPPER)
        probes = np.eye(N_PARAMS) * (self.STEP * SCALE)
        damping = 1e-2
        r = self._residuals(x[None], observed, is_right, previous)[0]
        cost = r @ r
        for _ in range(iterations):
            batch = self._residuals(np.vstack((x, x + probes)), observed, is_right, previous)
            jac = (batch[1:] - batch[0]).T / (self.STEP * SCALE)  # (M, N)
            js = jac * SCALE
            g = js.T @ batch[0]
            h = js.T @ js
            # Try three dampings at once: one batch costs about what one pose does
            diag = np.diag(np.diag(h) + 1e-6)
            trials = []
            for factor in self.DAMPING_TRIES:
                # OpenCV's solver: numpy's spins up threads, 30x slower on a matrix this small
                _, step = cv2.solve(h + damping * factor * diag, -g, flags=cv2.DECOMP_LU)
                trials.append(np.clip(x + step.ravel() * SCALE, LOWER, UPPER))
            trials = np.array(trials)
            rt = self._residuals(trials, observed, is_right, previous)
            costs = np.einsum("ij,ij->i", rt, rt)
            best = int(np.argmin(costs))
            if costs[best] >= cost:
                damping *= 100
                continue
            moved = np.max(np.abs((trials[best] - x) / SCALE))
            x, cost = trials[best], float(costs[best])
            damping = max(damping * self.DAMPING_TRIES[best] / 3, 1e-6)
            if moved < 1e-3:
                break
        return x, float(cost)

    def _starts(self, observed: np.ndarray, is_right: bool, guesses):
        """
        Fresh starting poses: each guessed set of joint angles with the palm
        placed by IPPE (both of its answers), plus the open hand placed whole by SQPnP.
        """
        placements = self._placements(observed, is_right, guesses[0], whole_hand=False)
        starts = [np.r_[rt, joints] for joints in guesses for rt in placements]
        return starts + [np.r_[rt, guesses[0]] for rt in self._placements(observed, is_right, guesses[0], palm=False)]

    def _placements(self, observed: np.ndarray, is_right: bool, joints: np.ndarray,
                    palm: bool = True, whole_hand: bool = True):
        """Candidate [rotation vector, translation] for the hand, from the palm alone and/or the whole hand."""
        starts = []
        rvecs, tvecs = [], []
        palm_points = self.shape.palm[[0, 2, 3, 4, 5]].copy()  # wrist and the four knuckles
        if not is_right:
            palm_points[:, 2] = -palm_points[:, 2]
        palm_points[:, 2] = 0.0  # IPPE wants a flat object; the palm nearly is
        if palm:
            try:
                _, found_r, found_t, _ = cv2.solvePnPGeneric(palm_points, observed[[0] + KNUCKLES].astype(np.float64),
                                                             self.camera_matrix, None, flags=cv2.SOLVEPNP_IPPE)
                rvecs, tvecs = list(found_r), list(found_t)
            except cv2.error:
                pass
        if whole_hand:
            rest = self.camera_points(np.r_[0, 0, 0, 0, 0, 1.0, joints][None], is_right)[0] - [0, 0, 1.0]
            try:
                ok, rvec, tvec = cv2.solvePnP(rest, observed.astype(np.float64), self.camera_matrix, None,
                                              flags=cv2.SOLVEPNP_SQPNP)
                if ok:
                    rvecs.append(rvec)
                    tvecs.append(tvec)
            except cv2.error:
                pass
        for rvec, tvec in zip(rvecs, tvecs):
            if tvec[2, 0] > 0.08:
                starts.append(np.r_[rvec.ravel(), tvec.ravel()])
        return starts

    def fit(self, observed: np.ndarray, is_right: bool, previous: Optional[np.ndarray] = None,
            palm_away: bool = True, bends: Optional[np.ndarray] = None,
            allow_refresh: bool = True) -> Optional[HandFit]:
        """
        Args:
            observed: 21 x 2 landmark positions in pixels
            is_right: Whether this is the user's right hand
            previous: This hand's parameters last frame, to start from and stay near
            palm_away: Prefer the back of the hand toward the camera when two
                poses fit about as well (POV)
            bends: Joint angles read from MediaPipe's 3D hand (bend_angles): a
                starting guess, and a loose pull where the 2D points are ambiguous
            allow_refresh: Whether a poor fit may start over from fresh poses
                (several times slower); the first fit always does

        Returns:
            The fit, or None if no starting pose could be found
        """
        observed = np.asarray(observed, dtype=float)
        self._bends = None if bends is None else np.asarray(bends, dtype=float)[BENDS - 6]

        def score(params: np.ndarray, cost: float) -> float:
            if palm_away:
                cost += self.PALM_AWAY_WEIGHT * max(0.0, self.palm_to_camera(params, is_right)) ** 2
            return cost

        candidates = []
        if previous is not None:
            params, cost = self._solve(previous, observed, is_right, previous, self.ITERATIONS)
            candidates.append((score(params, cost), params))
            if self._bends is not None and np.max(np.abs(params[BENDS] - self._bends)) > self.BENDS_DISAGREE:
                start = previous.copy()
                start[BENDS] = self._bends
                params, cost = self._solve(start, observed, is_right, previous, self.ITERATIONS)
                candidates.append((score(params, cost), params))
        warm = min(candidates, key=lambda c: c[0])[1] if candidates else None
        refreshed = previous is None or (allow_refresh and self._rms(warm, observed, is_right) > self.REFIT_RMS_PX)
        if refreshed:
            # Start fresh from an open hand, a fist and MediaPipe's own bends, each
            # placed every way the palm fits. A short look at all of them, then
            # only the most promising are finished.
            guesses = [np.zeros(20), FIST] + ([bends] if bends is not None else [])
            starts = self._starts(observed, is_right, guesses)
            rough = []
            for i, start in enumerate(starts):
                params, cost = self._solve(start, observed, is_right, previous, self.SCOUT_ITERATIONS)
                rough.append((score(params, cost), i, params))
            for _, _, start in sorted(rough)[:self.FINISHED_STARTS]:
                params, cost = self._solve(start, observed, is_right, previous, self.FRESH_ITERATIONS)
                candidates.append((score(params, cost), params))
        if not candidates:
            return None
        params = min(candidates, key=lambda c: c[0])[1]
        return HandFit(params, self.camera_points(params[None], is_right)[0],
                       self._rms(params, observed, is_right), self.palm_to_camera(params, is_right), refreshed)

    def _rms(self, params: np.ndarray, observed: np.ndarray, is_right: bool) -> float:
        error = self.project(self.camera_points(params[None], is_right)[0]) - observed
        return float(np.sqrt(np.mean(np.sum(error ** 2, axis=1))))
