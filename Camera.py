"""
Hand tracking camera capture and processing for SteamVR.
Captures video, detects hands using MediaPipe, and sends data to SteamVR driver.
"""
import cv2
import mediapipe as mp
import argparse
import copy
import dataclasses
import json
import math
import sys
import time
import numpy as np
from typing import Any, Callable, Dict, List, Set, Tuple, Optional
from hand_data import HAND_CONNECTIONS, HandData, TrackedHand, TrackingFrame, protocol_greeting
from hand_features import compute_features
from hand_controls import ControlMapper
from gesture_scores import GestureClassifier, score_gestures
from config_defaults import DEFAULT_CONFIG, DEFAULT_ROTATION_OFFSET_DEG, fill_defaults
import presets
from hand_identity import HandDetection, HandIdentityTracker, combine_handedness_votes, other_side
from hand_flow import HandFlow
from hand_motion import MotionPredictor, image_span, prediction_horizon
from depth_assist import WiLoRDepthAssist
from hand_fit import PALM, HandFitter, HandShape, bend_angles
from gesture_detector import GestureDetector, quat_from_euler_deg, quat_multiply, quat_rotate
from utils.config_utils import get_value, set_value
from utils.hand_size import HandSizeStabilizer
from utils.one_euro import (OneEuroFilter, QuaternionOneEuroFilter, ExponentialFilter,
                            QuaternionExponentialFilter, PassThroughFilter)
from utils.win_process import keep_running_in_background
from utils.camera_utils import CameraCapture
from utils.socket_client import SocketClient


# "facing": camera in front of the user, looking back at them (selfie view).
# "pov":    camera looks the same way the user does (head/chest mounted).
VIEW_MODES = ("facing", "pov")

# Smoothing modes the 'f' key cycles through
FILTER_MODES = ("one_euro", "ema", "none")
FILTER_LABELS = {"one_euro": "One Euro", "ema": "EMA (previous)", "none": "none"}

# Settings that only take effect by reopening the camera
CAMERA_DEVICE_KEYS = ("camera.device_id", "camera.width", "camera.height", "camera.fps", "camera.backend")
# Settings that only take effect by recreating the MediaPipe model
HANDS_MODEL_KEYS = ("tracking.max_hands", "tracking.detection_confidence",
                    "tracking.tracking_confidence", "tracking.model_complexity")
# Settings that change how image axes map to the user's left/right and depth
VIEW_KEYS = ("camera.", "tracking.view_mode", "tracking.hands_follow", "tracking.palm_facing")


class CameraLostError(RuntimeError):
    """The camera stopped delivering frames and did not come back in time."""


class HandTracker:
    """Main hand tracking system."""

    def __init__(self, config_path: str = "config.json",
                 view_mode: Optional[str] = None,
                 swap_hands: Optional[bool] = None,
                 rotate_180: Optional[bool] = None,
                 depth_source: Optional[str] = None,
                 config: Optional[dict] = None,
                 preset: Optional[str] = None):
        """
        Initialize hand tracker with configuration.

        Args:
            config_path: Path to configuration JSON file, used when config is None
            view_mode: "facing" or "pov"; switches to that mode's built-in preset
                unless preset is given, and overrides tracking.view_mode
            swap_hands: Extra left/right swap; overrides tracking.swap_hands
            rotate_180: Camera mounted upside down; overrides camera.rotate_180
            depth_source: "mediapipe" or "wilor"; overrides tracking.depth_source
            config: Configuration already loaded (e.g. by the GUI); copied, not shared
            preset: Preset to switch to (see presets.py), e.g. "POV"
        """
        self.config = copy.deepcopy(config) if config is not None else self.load_config(config_path)
        if preset is None and view_mode is not None:
            preset = presets.MODE_PRESETS.get(view_mode)
        if preset is not None:
            if preset not in presets.names(self.config):
                raise ValueError(f"Unknown preset '{preset}'. Presets: {', '.join(presets.names(self.config))}")
            presets.apply(self.config, preset)
        for key, value in (("tracking.view_mode", view_mode), ("tracking.swap_hands", swap_hands),
                           ("camera.rotate_180", rotate_180), ("tracking.depth_source", depth_source)):
            if value is not None:
                set_value(self.config, key, value)

        # Keep full speed while the VR game has focus: a CPU-heavy foreground
        # app otherwise starves a normal-priority background process
        process_config = self.config.get('process', {})
        for line in keep_running_in_background(process_config.get('priority', 'above_normal'),
                                               process_config.get('disable_power_throttling', True)):
            print(line)
        self._last_slow_report = 0.0
        self.last_timings = (0.0, 0.0, 0.0)
        self.last_camera_position = {}
        # Config values the tracker changed on its own (e.g. WiLoR failing to
        # load), for a GUI to pick up and clear
        self.changed_by_tracker: Dict[str, Any] = {}
        # Set from another thread: tell the driver the user is facing the camera now
        self.recenter_requested = False
        # Whether step()'s caller shows an OpenCV window that needs pumping
        self.cv_preview = False
        self.camera = None
        self.hands = None
        self.socket_client = None
        self.depth_assist = None
        self.depth_source = 'mediapipe'
        # The newest frame's size, as delivered
        self.frame_size = (int(self.config['camera']['width']), int(self.config['camera']['height']))

        self.configure_view()
        self.configure_calibration()
        self.create_identity()
        self.configure_prediction()
        self.create_hands_model()
        self.create_camera()
        self.create_gesture_detector()
        self.create_socket_client()
        self.debug = self.config['debug']

        # Depth source; 'b' in the preview toggles it. WiLoR is experimental,
        # needs .venv-wilor and is research/non-commercial only.
        self.configure_depth_assist()
        if str(self.config['tracking'].get('depth_source', 'mediapipe')).lower() == 'wilor':
            self.set_depth_source('wilor')

        cam_config = self.config['camera']
        print("HandTracker initialized")
        print(f"Camera: {cam_config['width']}x{cam_config['height']} @ {cam_config['fps']}fps")
        print(f"Tracking: max {self.config['tracking']['max_hands']} hands")
        print(f"View: {self.view_mode}, hands {'swapped' if self.user_swap else 'as detected'}"
              f"{', rotated 180' if self.camera.rotate_180 else ''}")
        print(f"Network: {self.config['network']['host']}:{self.config['network']['port']}")

    def configure_view(self):
        """Derive the view mode and image-to-user axis mapping from the config."""
        tracking_config = self.config['tracking']
        cam_config = self.config['camera']
        mode = str(tracking_config.get('view_mode', 'facing')).lower()
        if mode not in VIEW_MODES:
            print(f"Warning: unknown view_mode '{mode}', using facing")
            mode = "facing"
        self.view_mode = mode
        # Whether the frame MediaPipe sees is a mirror image: the source may
        # already be mirrored (some phone-camera apps do it) and flip_horizontal
        # mirrors it again.
        mirrored = bool(cam_config.get('source_mirrored', False)) != bool(cam_config['flip_horizontal'])
        # Mirroring reverses a hand's chirality in the image: a right hand
        # looks like a left one. Every handedness cue read from the image has
        # to be flipped back for a mirrored frame.
        self.frame_mirrored = mirrored
        # The user's manual correction, applied to the final left/right labels
        self.user_swap = bool(tracking_config.get('swap_hands', False))
        # Image +X points to the user's right for POV, and to their left for an
        # unmirrored camera facing them; mirroring reverses either.
        self.mirror_x = (mode == "facing") != mirrored
        # Which way the palms face: "away" from the camera (back of the hand
        # visible, the POV default) or "auto" from MediaPipe's handedness.
        palm_facing = str(tracking_config.get('palm_facing', 'mode')).lower()
        if palm_facing not in ('away', 'auto'):
            # "mode": hands seen from behind in POV, palms free to face a camera in front
            palm_facing = 'away' if mode == 'pov' else 'auto'
        self.palm_away = palm_facing == 'away'
        # A camera facing the user looks back toward them, so its depth axis
        # runs opposite to the headset's: Z is reflected, and the hands are
        # placed relative to the camera's distance in front of the user.
        self.flip_z = mode == 'facing'
        # A camera in front of the user can stay put in the room while the
        # headset looks around; a camera on the head always turns with it
        self.room_anchor = mode == 'facing' and str(tracking_config.get('hands_follow', 'head')).lower() == 'room'
        self.facing_distance = float(cam_config.get('facing_distance', 0.8))
        # Metric position: horizontal field of view of the camera
        self.hfov_deg = float(cam_config.get('hfov_deg', 70.0))
        self.stall_timeout = float(cam_config.get('stall_timeout', 2.0))
        self.reconnect_timeout = float(cam_config.get('reconnect_timeout', 30.0))
        if self.camera is not None:
            self.camera.flip_horizontal = bool(cam_config['flip_horizontal'])
            self.camera.rotate_180 = bool(cam_config.get('rotate_180', False))

    def configure_calibration(self):
        """Derive offsets, rotations and filters from the calibration settings."""
        self.calibration = self.config['calibration']
        # Camera mounting relative to the HMD, as [pitch, yaw, roll] degrees
        # (e.g. a head-mounted camera tilted down toward the hands: negative pitch)
        self.camera_rotation = quat_from_euler_deg(*self.calibration.get('camera_rotation_deg', [0.0, 0.0, 0.0]))
        # Per-hand correction so the controller model sits like the real hand
        offsets = self.calibration.get('rotation_offset_deg', DEFAULT_ROTATION_OFFSET_DEG)
        self.rotation_offsets = {
            hand: quat_from_euler_deg(*offsets.get(hand, [0.0, 0.0, 0.0]))
            for hand in ('left', 'right')
        }
        # How the user's hand compares to MediaPipe's average-sized hand model
        self.hand_scale = float(self.calibration.get('hand_scale', 1.0))
        # Hold each hand's model at its recent median size, so MediaPipe's
        # size wobble doesn't become depth jitter
        self.steady_hand_size = bool(self.calibration.get('steady_hand_size', False))
        self.hand_size = HandSizeStabilizer(int(self.calibration.get('hand_size_window', 90)))
        # Rebuild each hand in 3D from MediaPipe's 2D points with a hand model
        # of fixed size (hand_fit), instead of using MediaPipe's 3D shape
        self.rebuild_hand = bool(self.calibration.get('rebuild_hand', False))
        self.hand_fitter = None  # made for the frame size on first use
        self.hand_fitter_key = None
        # Per side: (time, fit parameters, wrist in pixels, time of the last fresh start)
        self.hand_fits: Dict[str, Tuple[float, np.ndarray, np.ndarray, float]] = {}
        # Smoothing, per hand: image-plane position, depth (noisier, so
        # filtered on its own) and rotation. 'f' in the preview cycles modes.
        self.filter_config = self.calibration.get('filter', {})
        mode = str(self.filter_config.get('mode', 'one_euro')).lower()
        self.filter_mode = mode if mode in FILTER_MODES else 'one_euro'
        self.build_filters()

    def create_identity(self):
        """(Re)create the left/right identity tracker for the current view."""
        identity_config = self.config['tracking'].get('identity', {})
        self.hand_identity = HandIdentityTracker(
            continuity_radius=float(identity_config.get('continuity_radius', 0.15)),
            memory_seconds=self.identity_memory_seconds(),
            switch_frames=int(identity_config.get('switch_frames', 15)),
            duplicate_radius=float(identity_config.get('duplicate_radius', 0.05)),
            user_left_is_image_left=not self.mirror_x,
            order_weight=float(identity_config.get('order_weight', 1.5)))

    def identity_memory_seconds(self) -> float:
        """
        How long a lost hand keeps its side: at least as long as it is
        predicted, so the hand found again is the one that was predicted.
        """
        memory = float(self.config['tracking'].get('identity', {}).get('memory_seconds', 0.4))
        return max(memory, prediction_horizon(self.config['calibration']['prediction_seconds']))

    def configure_prediction(self) -> None:
        """(Re)create the motion predictor that keeps lost hands moving (hand_motion)."""
        self.motion = MotionPredictor(self.config['calibration']['prediction_seconds'])
        # Lost hands follow their pixels through the picture, when on
        self.hand_flow = HandFlow() if self.config['tracking']['optical_flow'] and self.motion.prediction_seconds > 0 \
            else None
        # The last tracked frame of each hand, whose fingers and controls a predicted hand keeps
        self.last_tracked: Dict[str, TrackedHand] = {}
        self.hand_identity.memory_seconds = self.identity_memory_seconds()

    def create_hands_model(self):
        """(Re)create the MediaPipe Hands model from the tracking settings."""
        if self.hands is not None:
            self.hands.close()
        tracking_config = self.config['tracking']
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=int(tracking_config['max_hands']),
            min_detection_confidence=float(tracking_config['detection_confidence']),
            min_tracking_confidence=float(tracking_config['tracking_confidence']),
            model_complexity=int(tracking_config['model_complexity'])
        )

    def create_camera(self):
        """Create the capture device from the camera settings (not started)."""
        cam_config = self.config['camera']
        self.camera = CameraCapture(
            device_id=int(cam_config['device_id']),
            width=int(cam_config['width']),
            height=int(cam_config['height']),
            fps=int(cam_config['fps']),
            flip_horizontal=bool(cam_config['flip_horizontal']),
            backend=cam_config.get('backend', 'auto'),
            rotate_180=bool(cam_config.get('rotate_180', False))
        )

    def restart_camera(self) -> bool:
        """Reopen the camera with the current settings."""
        if self.camera is not None:
            self.camera.release()
        self.create_camera()
        return self.camera.start()

    def create_gesture_detector(self):
        gesture_config = self.config['gestures']
        self.gesture_detector = GestureDetector(
            pinch_threshold=float(gesture_config['pinch_threshold']),
            finger_extended_threshold=float(gesture_config['finger_extended_threshold'])
        )
        self.control_mapper = ControlMapper(gesture_config)
        self.gesture_classifier = GestureClassifier(gesture_config)

    def create_socket_client(self):
        """(Re)create the driver connection; it connects in the background."""
        if self.socket_client is not None:
            self.socket_client.close()
        net_config = self.config['network']
        self.socket_client = SocketClient(host=net_config['host'], port=int(net_config['port']),
                                          greeting=lambda: protocol_greeting(self.controller_type()))

    def controller_type(self) -> str:
        """What the driver presents the hands as, "touch" or "index"."""
        return str(self.config['network'].get('controller_type', 'touch'))

    def configure_depth_assist(self):
        self.depth_assist_config = self.config['tracking'].get('depth_assist', {})
        self.depth_assist_max_age = float(self.depth_assist_config.get('max_age', 0.5))
        self.depth_assist_match_radius = float(self.depth_assist_config.get('match_radius', 0.12))
        self.depth_assist_scale = float(self.depth_assist_config.get('scale', 1.0))
        if self.depth_assist is not None:
            self.depth_assist.min_interval = 1.0 / max(0.5, float(self.depth_assist_config.get('max_rate_hz', 10.0)))

    def apply_settings(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Change config values while tracking and apply them, rebuilding only
        what depends on them, once per batch. process.* and debug.* apply on
        the next start.

        Args:
            changes: Dotted config keys (e.g. "calibration.filter.depth.beta") to new values

        Returns:
            The values in effect afterwards (they can differ, e.g. WiLoR failing to load)
        """
        for key, value in changes.items():
            set_value(self.config, key, value)
        actions = []
        for key in changes:
            action = self.setting_action(key)
            if action is not None and action not in actions:
                actions.append(action)
        for action in actions:
            action()
        return {key: get_value(self.config, key) for key in changes}

    def apply_setting(self, key: str, value: Any) -> Any:
        """Single-key apply_settings; returns the value in effect."""
        return self.apply_settings({key: value})[key]

    def setting_action(self, key: str) -> Optional[Callable[[], Any]]:
        """What has to be rebuilt for a changed config key, or None if nothing live."""
        if key in CAMERA_DEVICE_KEYS:
            return self.restart_camera
        if key == 'tracking.swap_hands':
            return self.swap_identities
        if key.startswith(VIEW_KEYS):
            return self.reconfigure_view
        if key in ('calibration.prediction_seconds', 'tracking.optical_flow'):
            return self.configure_prediction
        if key.startswith('calibration.'):
            return self.configure_calibration
        if key.startswith('tracking.identity.'):
            return self.create_identity
        if key in HANDS_MODEL_KEYS:
            return self.create_hands_model
        if key == 'tracking.depth_source':
            return self.apply_depth_source
        if key.startswith('tracking.depth_assist.'):
            return self.configure_depth_assist
        if key.startswith('gestures.'):
            return self.create_gesture_detector
        if key in ('network.controller_type', 'network.index_offset', 'network.index_rotation_deg'):
            return None  # read every frame; the driver only switches type after a SteamVR restart
        if key.startswith('network.'):
            return self.create_socket_client
        return None

    def reconfigure_view(self):
        """Apply a change to the camera mounting or view mode."""
        self.configure_view()
        self.configure_calibration()
        self.create_identity()

    def apply_depth_source(self):
        self.set_depth_source(str(self.config['tracking'].get('depth_source', 'mediapipe')).lower())

    def swap_identities(self):
        """
        Manual correction: swap which side each tracked hand is reported as.
        The identity tracker keeps its own, unswapped, view. Expects
        tracking.swap_hands to be toggled already.
        """
        self.configure_view()
        self.build_filters()  # each hand's history belonged to the other
        self.hand_size.reset()
        self.hand_fits = {}
        self.configure_prediction()
        print("Hands swapped")

    def reset_hand(self, hand_type: str) -> None:
        """
        Forget one side's history when a hand is new on it (just appeared, came
        back, or changed side), so it starts from its own first frame instead of
        blending with whatever that side held before. The hand size history is
        kept: it is the same person, and both hands are about the same size.
        """
        for filters in (self.position_filters, self.depth_filters, self.rotation_filters):
            filters[hand_type].reset()
        self.hand_fits.pop(hand_type, None)
        self.last_camera_position.pop(hand_type, None)
        self.control_mapper.reset(hand_type)
        self.gesture_classifier.reset(hand_type)
        self.motion.forget(hand_type)
        if self.hand_flow is not None:
            self.hand_flow.forget(hand_type)
        self.last_tracked.pop(hand_type, None)

    def fit_hand(self, hand_landmarks, world: List[Tuple[float, float, float]],
                 frame_width: int, frame_height: int, hand_type: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Locate the hand in 3D by fitting MediaPipe's metric hand model to where
        its landmarks appear in the image (perspective-n-point).

        MediaPipe world landmarks use OpenCV camera axes (x right, y down,
        z forward), so solvePnP's pose places them directly in camera space.

        Args:
            hand_landmarks: Normalised image landmarks
            world: World landmarks in metres, centred on the hand
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            hand_type: "left" or "right", for the steady hand size (None skips it)

        Returns:
            21 x 3 joints in OpenVR camera space (y up, -z forward), wrist
            first, or None if the fit failed
        """
        scale = self.hand_scale
        if self.steady_hand_size and hand_type is not None:
            scale *= self.hand_size(hand_type, world)
        object_points = np.array(world, dtype=np.float64) * scale
        image_points = np.array([(lm.x * frame_width, lm.y * frame_height)
                                 for lm in hand_landmarks.landmark], dtype=np.float64)
        focal = (frame_width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)
        camera_matrix = np.array([[focal, 0.0, frame_width / 2.0],
                                  [0.0, focal, frame_height / 2.0],
                                  [0.0, 0.0, 1.0]])
        try:
            ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, None,
                                          flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error:
            return None
        if not ok:
            return None
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        points = object_points @ rotation_matrix.T + tvec.reshape(3)
        # Behind the camera or implausibly far means the fit went wrong
        if not 0.05 < points[0, 2] < 3.0:
            return None
        return points * np.array([-1.0 if self.mirror_x else 1.0, -1.0, -1.0])

    # Starting over from fresh poses is several times slower than following
    # last frame's fit, so a hand does it at most this often
    REBUILD_REFRESH_SECONDS = 0.2
    # Last frame's fit is only a starting point while the hand is still near
    # it: wrist movement as a fraction of the frame width, and time
    REBUILD_CONTINUITY = 0.08
    REBUILD_MEMORY_SECONDS = 0.1

    def rebuild_hand_points(self, landmarks: List[Tuple[float, float, float]], world: List[Tuple[float, float, float]],
                            hand_type: str, frame_width: int, frame_height: int, now: float):
        """
        Rebuild a hand in 3D from its 2D landmarks with hand_fit's model, for
        when MediaPipe's own 3D shape is unreliable (seen from behind).

        Returns:
            (world, points): the hand as MediaPipe-style world landmarks
            (camera axes, centred on the palm) and as 21 x 3 joints in OpenVR
            camera space, like fit_hand; or None if it could not be fitted
        """
        key = (frame_width, frame_height, self.hfov_deg, self.hand_scale)
        if self.hand_fitter is None or self.hand_fitter_key != key:
            focal = (frame_width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)
            self.hand_fitter = HandFitter(HandShape.default(self.hand_scale), focal,
                                          (frame_width / 2.0, frame_height / 2.0))
            self.hand_fitter_key = key
            self.hand_fits = {}
        observed = np.array([(x * frame_width, y * frame_height) for x, y, _ in landmarks])
        # The model is fitted to the picture, where a mirrored frame shows a
        # right hand as a left one
        image_right = (hand_type == "right") != self.frame_mirrored
        previous, refreshed_at = None, -math.inf
        state = self.hand_fits.get(hand_type)
        if state is not None:
            then, params, wrist, refreshed_at = state
            if (now - then <= self.REBUILD_MEMORY_SECONDS
                    and np.linalg.norm(observed[0] - wrist) <= self.REBUILD_CONTINUITY * frame_width):
                previous = params
        result = self.hand_fitter.fit(observed, image_right, previous, palm_away=self.palm_away,
                                      bends=bend_angles(world),
                                      allow_refresh=now - refreshed_at >= self.REBUILD_REFRESH_SECONDS)
        if result is None:
            return None
        self.hand_fits[hand_type] = (now, result.params, observed[0], now if result.refreshed else refreshed_at)
        points = result.points
        centred = points - points[PALM].mean(axis=0)
        return ([tuple(float(v) for v in p) for p in centred],
                points * np.array([-1.0 if self.mirror_x else 1.0, -1.0, -1.0]))

    def estimate_wrist_position(self, hand_landmarks, world: List[Tuple[float, float, float]],
                                frame_width: int, frame_height: int) -> Optional[Tuple[float, float, float]]:
        """Wrist (x, y, z) in OpenVR camera space from fit_hand, or None if the fit failed."""
        points = self.fit_hand(hand_landmarks, world, frame_width, frame_height)
        if points is None:
            return None
        return tuple(float(v) for v in points[0])

    def estimate_wrist_position_fallback(self, landmarks: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        """
        Rough wrist position when no metric fit is available: image position
        projected at a fixed arm's-length depth.

        Args:
            landmarks: Normalised image landmarks

        Returns:
            Wrist (x, y, z) in OpenVR camera space
        """
        depth = 0.45
        half_width = depth * math.tan(math.radians(self.hfov_deg) / 2.0)
        x = (landmarks[0][0] - 0.5) * 2.0 * half_width
        y = -(landmarks[0][1] - 0.5) * 2.0 * half_width * (self.camera.height / self.camera.width)
        return (-x if self.mirror_x else x, y, -depth)

    def smooth_camera_position(self, hand_type: str, position: Tuple[float, float, float],
                               t: float) -> Tuple[float, float, float]:
        """
        Smooth a camera-space position. Depth is far noisier than the
        image-plane axes, so it gets its own filter.

        Args:
            hand_type: "left" or "right"
            position: New camera-space sample
            t: Sample time in seconds

        Returns:
            Smoothed position (also stored for the next frame)
        """
        x, y = self.position_filters[hand_type](position[:2], t)
        (z,) = self.depth_filters[hand_type]((position[2],), t)
        position = (x, y, z)
        if all(math.isfinite(v) for v in position):
            self.last_camera_position[hand_type] = position
        return position

    def set_depth_source(self, source: str):
        """
        Switch where per-hand depth comes from: "mediapipe" (solvePnP on its
        landmarks) or "wilor" (asynchronous WiLoR mesh, experimental).
        Falls back to MediaPipe if WiLoR cannot be loaded.
        """
        if source == 'wilor':
            if self.depth_assist is None:
                print("Loading WiLoR for depth in the background (first run downloads the models)...")
                self.depth_assist = WiLoRDepthAssist(self.estimate_wrist_position,
                                                     float(self.depth_assist_config.get('max_rate_hz', 10.0)))
            if not self.depth_assist.start():
                print(f"WiLoR depth unavailable: {self.depth_assist.error}. Staying on MediaPipe.")
                source = 'mediapipe'
        self.depth_source = source
        self.config['tracking']['depth_source'] = source
        self.build_filters()  # depth history from the other source no longer applies
        print(f"Depth: {source}")

    def depth_label(self) -> str:
        if self.depth_source != 'wilor':
            return "MediaPipe"
        if self.depth_assist.loading:
            return "WiLoR loading..."
        label = f"WiLoR {self.depth_assist.rate_hz:.0f} Hz"
        torch = getattr(self.depth_assist, '_torch', None)
        if torch is not None:
            label += f", {torch.cuda.memory_reserved() / 1e9:.1f} GB VRAM"
        return label

    def apply_assisted_depth(self, wrist: Tuple[float, float], position: Tuple[float, float, float],
                             now: float) -> Tuple[float, float, float]:
        """
        Replace the distance of a MediaPipe camera-space position with WiLoR's,
        keeping it on MediaPipe's current line of sight: the hand stays where
        MediaPipe sees it in the image, only how far along that ray changes.
        WiLoR's result comes from an older frame, so it is matched to this hand
        by image position and dropped once too old.
        """
        results_time, results = self.depth_assist.latest()
        if not results or now - results_time > self.depth_assist_max_age:
            return position
        nearest = min(results, key=lambda r: math.hypot(r[0] - wrist[0], r[1] - wrist[1]))
        if math.hypot(nearest[0] - wrist[0], nearest[1] - wrist[1]) > self.depth_assist_match_radius:
            return position
        mediapipe_distance = -position[2]
        if mediapipe_distance <= 1e-3:
            return position
        # WiLoR's hand model is not sized like MediaPipe's, so its distances
        # get their own calibration factor
        scale = nearest[2] * self.depth_assist_scale / mediapipe_distance
        return (position[0] * scale, position[1] * scale, position[2] * scale)

    def build_filters(self):
        """(Re)create the per-hand filters for the current filter_mode."""
        config = self.filter_config
        hands = ('left', 'right')
        if self.filter_mode == 'one_euro':
            def make(kind, cls, defaults):
                s = dict(defaults, **config.get(kind, {}))
                return {h: cls(s['min_cutoff'], s['beta'], config.get('d_cutoff', 3.0)) for h in hands}
            self.position_filters = make('position', OneEuroFilter, {'min_cutoff': 1.0, 'beta': 1.5})
            self.depth_filters = make('depth', OneEuroFilter, {'min_cutoff': 0.3, 'beta': 4.0})
            self.rotation_filters = make('rotation', QuaternionOneEuroFilter, {'min_cutoff': 1.0, 'beta': 0.5})
        elif self.filter_mode == 'ema':
            follow = dict({'position': 0.7, 'depth': 0.35, 'rotation': 0.5}, **config.get('ema', {}))
            self.position_filters = {h: ExponentialFilter(follow['position']) for h in hands}
            self.depth_filters = {h: ExponentialFilter(follow['depth']) for h in hands}
            self.rotation_filters = {h: QuaternionExponentialFilter(follow['rotation']) for h in hands}
        else:
            self.position_filters = {h: PassThroughFilter() for h in hands}
            self.depth_filters = {h: PassThroughFilter() for h in hands}
            self.rotation_filters = {h: PassThroughFilter() for h in hands}

    def cycle_filter_mode(self):
        """Switch to the next smoothing mode, for live A/B comparison."""
        mode = FILTER_MODES[(FILTER_MODES.index(self.filter_mode) + 1) % len(FILTER_MODES)]
        self.apply_setting('calibration.filter.mode', mode)
        print(f"Filter: {FILTER_LABELS[self.filter_mode]}")

    SLOW_FRAME_SECONDS = 0.25

    def report_slow_frame(self, t_frame: float, t_tracked: float, t_sent: float, t_done: float):
        """
        Print where the time went when one frame took far too long, at most
        once per second. A frame normally takes ~20-40 ms; this is the
        evidence for telling CPU starvation (tracking slow), a stuck driver
        connection (sending slow) or a blocked preview window (display slow)
        apart.
        """
        if t_done - t_frame < self.SLOW_FRAME_SECONDS or t_done - self._last_slow_report < 1.0:
            return
        self._last_slow_report = t_done
        print(f"Slow frame: {(t_done - t_frame) * 1000:.0f} ms "
              f"(tracking {(t_tracked - t_frame) * 1000:.0f}, "
              f"sending {(t_sent - t_tracked) * 1000:.0f}, "
              f"display {(t_done - t_sent) * 1000:.0f})")

    def reconnect_camera(self) -> bool:
        """
        Keep reopening the camera until it delivers frames or reconnect_timeout
        runs out. Network cameras (phone-as-webcam apps) drop out routinely.

        Returns:
            True if the camera is back
        """
        deadline = time.time() + self.reconnect_timeout
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            if self.camera.reopen():
                print(f"Camera reconnected (attempt {attempt})")
                return True
            # Keep the preview window responsive while waiting
            if self.cv_preview:
                cv2.waitKey(1)
            time.sleep(1.0)
        return False

    @staticmethod
    def load_config(config_path: str) -> dict:
        """
        Load configuration from JSON file.

        Args:
            config_path: Path to config file

        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Configuration loaded from {config_path}")
        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using default configuration")
            config = {}
        calibration = config.setdefault('calibration', {})
        if 'rotation_offset_deg' not in calibration:
            # Older files kept it per view mode, though it does not depend
            # on the camera; the POV one is the one that was tuned
            modes = [calibration.get(mode) or {} for mode in VIEW_MODES]
            tuned = (calibration.get('pov') or {}).get('rotation_offset_deg')
            calibration['rotation_offset_deg'] = copy.deepcopy(tuned or DEFAULT_ROTATION_OFFSET_DEG)
            for section in modes:
                section.pop('rotation_offset_deg', None)
        # Settings added after this file was written start at the active
        # preset's value, or else at their defaults
        presets.fill_new_settings(config)
        fill_defaults(config, DEFAULT_CONFIG)
        # Placement used to be kept per view mode; it now belongs to presets
        presets.migrate(config)
        return config

    def handedness_evidence(self, hand_world_landmarks, handedness) -> float:
        """
        This frame's left/right vote for one hand, in [-1, 1] (> 0 is left).
        Only a vote: HandIdentityTracker weighs it against continuity.

        Finger curl leads: fingers only bend toward the palm, so it holds for
        any hand rotation. The secondary signal assumes which side faces the
        camera (palm-away geometry in POV, MediaPipe's selfie label otherwise),
        so it inverts when the hand turns over; it only gets a small weight,
        enough to decide a flat hand. See combine_handedness_votes for when the
        two disagree.
        """
        classification = handedness.classification[0]
        # MediaPipe labels handedness assuming a mirrored (selfie) image, so
        # its label is only right as-is for a mirrored frame
        mp_is_left = (classification.label == "Left") == self.frame_mirrored
        mp_vote = classification.score if mp_is_left else -classification.score
        if hand_world_landmarks is None:
            return mp_vote
        world = [(lm.x, lm.y, lm.z) for lm in hand_world_landmarks.landmark]
        # The curl vote reads the chirality the image shows, true only unmirrored
        curl = self.gesture_detector.handedness_evidence_curl(world)
        if self.frame_mirrored:
            curl = -curl
        if self.palm_away:
            secondary = self.gesture_detector.handedness_evidence_palm_away(world, self.mirror_x, self.flip_z)
        else:
            secondary = mp_vote
        return combine_handedness_votes(curl, secondary, secondary_ignores_depth=self.palm_away)

    def process_hand_landmarks(self, hand_landmarks, hand_world_landmarks, hand_type: str,
                               frame_width: int, frame_height: int) -> TrackedHand:
        """
        Process MediaPipe hand landmarks into the data sent to the driver.

        Args:
            hand_landmarks: MediaPipe hand landmarks (normalised image coordinates)
            hand_world_landmarks: MediaPipe world landmarks (metres), or None
            hand_type: "left" or "right", as decided by HandIdentityTracker
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels

        Returns:
            The hand's HandData plus its camera-space pose for the previews
        """
        # Extract landmarks as list of tuples
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z))

        is_left = hand_type == "left"
        world = None
        if hand_world_landmarks is not None:
            world = [(lm.x, lm.y, lm.z) for lm in hand_world_landmarks.landmark]

        # Wrist position in OpenVR camera space, in metres
        now = time.perf_counter()
        fitted = None
        if world is not None and self.rebuild_hand:
            rebuilt = self.rebuild_hand_points(landmarks, world, hand_type, frame_width, frame_height, now)
            if rebuilt is not None:
                # The rebuilt hand stands in for MediaPipe's from here on:
                # rotation and finger curl are read from it too
                world, fitted = rebuilt
        if world is not None and fitted is None:
            fitted = self.fit_hand(hand_landmarks, world, frame_width, frame_height, hand_type)
        camera_position = tuple(float(v) for v in fitted[0]) if fitted is not None else None
        if camera_position is None:
            camera_position = self.last_camera_position.get(hand_type)
        if camera_position is None:
            camera_position = self.estimate_wrist_position_fallback(landmarks)
        if self.depth_source == 'wilor':
            camera_position = self.apply_assisted_depth(landmarks[0][:2], camera_position, now)
        camera_position = self.smooth_camera_position(hand_type, camera_position, now)

        # Calculate hand orientation from the metric world landmarks; the
        # normalised ones have a squashed, image-relative Z and unequal X/Y scales.
        # The palm side comes from the hand's chirality (is_left, now stable
        # over time), not from assuming which side faces the camera, so a hand
        # turned palm-to-camera keeps the right orientation.
        if world is not None:
            rotation = self.gesture_detector.calculate_hand_orientation(world, is_left, self.mirror_x,
                                                                        palm_away=False, flip_z=self.flip_z)
            rotation = quat_multiply(self.camera_rotation, rotation)
            rotation = quat_multiply(rotation, self.rotation_offsets[hand_type])
            rotation = self.rotation_filters[hand_type](rotation, now)
        else:
            rotation = (1.0, 0.0, 0.0, 0.0)
        if world is not None:
            # A hand found again after a prediction blends in from where it was predicted.
            # Without world landmarks the rotation is a placeholder, not part of any motion.
            camera_position, rotation = self.motion.observe(hand_type, now, camera_position, rotation,
                                                            landmarks[0][:2])
            if self.hand_flow is not None:
                # Followed from the same frame the motion starts from
                self.hand_flow.tracked(hand_type, landmarks)
        # The joints follow the filtered wrist, so the 3D preview shows what is sent
        camera_points = fitted - fitted[0] + np.array(camera_position) if fitted is not None else None
        position, rotation = self.driver_pose(hand_type, camera_position, rotation)

        # Trigger, grip and the gesture name, from finger curl and pinch. Without
        # world landmarks there are no features, so the old detector is the fallback.
        gesture_config = self.config['gestures']
        features = None
        finger_curls: Tuple[float, ...] = ()
        scores: Dict[str, float] = {}
        if world is not None:
            features = compute_features(world, float(gesture_config['pinch_open']),
                                        float(gesture_config['pinch_closed']),
                                        gesture_config.get('curl_open_deg'), gesture_config.get('curl_full_deg'))
            trigger_value, grip_value = self.control_mapper(hand_type, features, now)
            finger_curls = self.control_mapper.finger_curls(hand_type, features, now)
            scores = score_gestures(features, self.control_mapper.pinch_strength(features), gesture_config)
            gesture = self.gesture_classifier(hand_type, scores)
        else:
            gesture = self.gesture_detector.detect_gesture(landmarks)
            trigger_value = self.gesture_detector.get_trigger_value(gesture)
            grip_value = self.gesture_detector.get_grip_value(gesture)

        hand_data = HandData(
            hand_type=hand_type,
            position=position,
            rotation=rotation,
            gesture=gesture,
            trigger_value=trigger_value,
            grip_value=grip_value,
            landmarks=landmarks,
            is_detected=True,
            finger_curls=finger_curls,
        )
        return TrackedHand(data=hand_data, camera_position=camera_position, camera_points=camera_points,
                           features=features, gesture_scores=scores)

    def driver_pose(self, hand_type: str, camera_position: Tuple[float, float, float],
                    rotation: Tuple[float, float, float, float]
                    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        """
        The pose sent to the driver for a camera-space wrist and a hand rotation
        already in headset space: the calibration, then the Index adjustment.
        """
        # Scale in camera space, tilt into HMD space, then offset
        scale = self.calibration['scale']
        offset = self.calibration['position_offset']
        x, y, z = camera_position
        if self.flip_z:
            # Distance from a camera in front of the user becomes distance
            # forward of the user: a hand nearer the camera is further out
            z = -z - self.facing_distance
        cx, cy, cz = quat_rotate(self.camera_rotation, (x * scale, y * scale, z * scale))
        position = (cx + offset[0], cy + offset[1], cz + offset[2])
        return self.index_adjustment(hand_type, position, rotation)

    def predicted_hand(self, hand_type: str, now: float) -> Optional[TrackedHand]:
        """
        A lost hand moved to where it is predicted to be (hand_motion), with the
        fingers, controls and gesture of its last tracked frame. None when it
        isn't predicted.
        """
        last = self.last_tracked.get(hand_type)
        if last is None or not self.motion.can_predict(hand_type, now):
            return None
        flow = None
        if self.hand_flow is not None:
            offset = self.hand_flow.follow(hand_type)
            if offset is not None:
                flow = (offset, image_span(self.hfov_deg, *self.frame_size, self.mirror_x))
        prediction = self.motion.predict(hand_type, now, flow)
        if prediction is None:
            return None
        camera_position = tuple(float(v) for v in prediction.camera_position)
        position, rotation = self.driver_pose(hand_type, camera_position, prediction.rotation)
        du, dv = prediction.image_offset
        landmarks = [(x + du, y + dv, z) for x, y, z in last.data.landmarks]
        data = dataclasses.replace(last.data, position=position, rotation=rotation, landmarks=landmarks,
                                   is_detected=False)
        camera_points = None
        if last.camera_points is not None:
            camera_points = last.camera_points - np.array(last.camera_position) + prediction.camera_position
        return dataclasses.replace(last, data=data, camera_position=camera_position, camera_points=camera_points,
                                   predicted=True, path=prediction.path)

    def index_adjustment(self, hand_type: str, position: Tuple[float, float, float],
                         rotation: Tuple[float, float, float, float]):
        """
        Shown as Index controllers, move each hand from where a Touch sits in it
        to where an Index does, in the controller's own frame. The settings are
        for the left hand; the right one is its mirror image.
        """
        network = self.config['network']
        if str(network.get('controller_type', 'touch')) != 'index':
            return position, rotation
        side = 1.0 if hand_type == 'left' else -1.0
        ox, oy, oz = network.get('index_offset', (0.0, 0.0, 0.0))
        pitch, yaw, roll = network.get('index_rotation_deg', (0.0, 0.0, 0.0))
        dx, dy, dz = quat_rotate(rotation, (side * ox, oy, oz))
        position = (position[0] + dx, position[1] + dy, position[2] + dz)
        rotation = quat_multiply(rotation, quat_from_euler_deg(pitch, side * yaw, side * roll))
        return position, rotation

    def calculate_palm_size(self, landmarks: List[Tuple[float, float, float]]) -> float:
        """
        Calculate palm size for depth estimation.

        Args:
            landmarks: List of hand landmarks

        Returns:
            Palm size as average of key distances
        """
        if len(landmarks) < 21:
            return 0.1

        # Calculate distances between key palm points
        wrist = landmarks[0]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]

        dist1 = self.gesture_detector.calculate_distance(wrist, index_mcp)
        dist2 = self.gesture_detector.calculate_distance(wrist, pinky_mcp)
        dist3 = self.gesture_detector.calculate_distance(index_mcp, pinky_mcp)

        return (dist1 + dist2 + dist3) / 3.0

    def draw_landmarks(self, frame, landmarks: List[Tuple[float, float, float]],
                       color: Tuple[int, int, int] = (255, 0, 0)):
        """
        Draw a hand's skeleton on the frame.

        Args:
            frame: OpenCV frame
            landmarks: Normalised image landmarks
            color: BGR colour of the bones
        """
        h, w = frame.shape[:2]
        points = [(int(x * w), int(y * h)) for x, y, _ in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, points[a], points[b], color, 2)
        for point in points:
            cv2.circle(frame, point, 2, (0, 255, 0), 2)

    def draw_info(self, frame, hands_data: List[HandData], fps: float):
        """
        Draw information overlay on frame.

        Args:
            frame: OpenCV frame
            hands_data: List of detected hands
            fps: Current FPS
        """
        # Draw FPS
        if self.debug['show_fps']:
            cv2.putText(frame, f"FPS: {fps:.1f} (camera {self.camera.capture_fps:.0f})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Filter [f]: {FILTER_LABELS[self.filter_mode]}", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(frame, f"Depth [b]: {self.depth_label()}", (10, frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # Draw hand information
        y_offset = 60
        for hand in hands_data:
            if hand.is_detected:
                depth_cm = -self.last_camera_position.get(hand.hand_type, (0.0, 0.0, 0.0))[2] * 100.0
                info_text = f"{hand.hand_type.upper()}: {hand.gesture}  {depth_cm:.0f} cm"
                cv2.putText(frame, info_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_offset += 30

    def start(self) -> bool:
        """
        Open the camera and start connecting to the driver.

        Returns:
            False if the camera could not be started
        """
        if not self.camera.start():
            print("Failed to start camera.")
            return False
        print("Connecting to SteamVR driver...")
        if not self.socket_client.connect():
            print("Warning: Could not connect to driver. Will keep trying...")
        return True

    def step(self) -> Optional[TrackingFrame]:
        """
        Track the newest camera frame and send the hands to the driver.

        Returns:
            The frame's results, or None if no new frame arrived in time

        Raises:
            CameraLostError: The camera stopped and did not come back
        """
        ret, frame = self.camera.read_frame()
        if not ret:
            if self.camera.is_stalled(self.stall_timeout):
                print(f"Camera stopped delivering frames for {self.stall_timeout:.1f}s. Reconnecting...")
                if not self.reconnect_camera():
                    raise CameraLostError(f"Camera did not come back within {self.reconnect_timeout:.0f}s.")
            # read_frame already waited for a frame; nothing more to wait for
            return None

        t_frame = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # As delivered: a camera may not give the size asked for
        self.frame_size = (frame.shape[1], frame.shape[0])
        if self.hand_flow is not None:
            self.hand_flow.next_frame(frame)
        results = self.hands.process(frame_rgb)
        t_tracked = time.perf_counter()
        if self.depth_source == 'wilor':
            if self.depth_assist.failed:
                # It loads in the background, so a failure only shows up now
                print(f"{self.depth_assist.error}. Back to MediaPipe depth.")
                self.set_depth_source('mediapipe')
                self.changed_by_tracker['tracking.depth_source'] = 'mediapipe'
            else:
                self.depth_assist.submit(frame_rgb, t_frame)

        tracked = []
        if results.multi_hand_landmarks and results.multi_handedness:
            world_list = results.multi_hand_world_landmarks or [None] * len(results.multi_hand_landmarks)

            # Decide left/right for the whole frame at once, with memory
            # of previous frames, so a one-frame misread or a duplicate
            # detection cannot teleport a hand to the other side
            detections = [
                HandDetection(i, (lm.landmark[0].x, lm.landmark[0].y),
                              self.handedness_evidence(world, handedness),
                              handedness.classification[0].score)
                for i, (lm, world, handedness) in enumerate(zip(results.multi_hand_landmarks, world_list,
                                                                results.multi_handedness))
            ]
            sides = self.hand_identity.assign(detections, t_tracked)
            new_sides = self.hand_identity.new_sides
            if self.user_swap:
                sides = {i: other_side(side) for i, side in sides.items()}
                new_sides = {other_side(side) for side in new_sides}
            for side in new_sides:
                self.reset_hand(side)

            for i, (hand_landmarks, hand_world) in enumerate(zip(results.multi_hand_landmarks, world_list)):
                if i not in sides:
                    continue  # duplicate of a hand already kept
                hand = self.process_hand_landmarks(hand_landmarks, hand_world, sides[i],
                                                   frame.shape[1], frame.shape[0])
                tracked.append(hand)
                if hand.features is not None and hand.data.is_sendable_pose():
                    # What a prediction starts from, if the hand is lost next
                    self.last_tracked[sides[i]] = hand
                if self.debug['log_gestures']:
                    print(f"{hand.data.hand_type}: {hand.data.gesture} "
                          f"T:{hand.data.trigger_value:.2f} G:{hand.data.grip_value:.2f}")
        tracked += self.predict_lost_hands({hand.data.hand_type for hand in tracked}, time.perf_counter())

        if self.recenter_requested:
            self.recenter_requested = False
            # A camera fixed in the room is where the headset looks now
            self.socket_client.send("RECENTER:1")
        controller_type = self.controller_type()
        for hand in tracked:
            if not hand.data.is_sendable_pose():
                continue  # the driver keeps the hand's last good pose
            self.socket_client.send(hand.data.to_protocol_string(controller_type, self.room_anchor))
        # With no hands in view nothing else is sent, and the driver drops silent connections
        self.socket_client.keepalive()
        t_sent = time.perf_counter()
        self.last_timings = (t_frame, t_tracked, t_sent)

        return TrackingFrame(
            frame_rgb=frame_rgb,
            frame_bgr=frame,
            hands=tracked,
            timings_ms={'tracking': (t_tracked - t_frame) * 1000.0, 'sending': (t_sent - t_tracked) * 1000.0},
            tracking_fps=self.camera.get_fps(),
            camera_fps=self.camera.capture_fps,
            driver_connected=self.socket_client.connected,
            depth_label=self.depth_label(),
            hfov_deg=self.hfov_deg,
        )

    def predict_lost_hands(self, seen: Set[str], now: float) -> List[TrackedHand]:
        """
        Keep each hand lost this frame moving along its path for a while
        (hand_motion), instead of freezing it until it is found again.

        Args:
            seen: The sides tracked this frame
            now: Time in seconds

        Returns:
            The predicted hands
        """
        predicted = []
        for side in ('left', 'right'):
            if side in seen or side not in self.last_tracked:
                continue
            # The identity tracker keeps its own, unswapped, sides
            identity_side = other_side(side) if self.user_swap else side
            if not self.hand_identity.has_track(identity_side):
                # The hand changed side: it is tracked as the other hand now
                self.reset_hand(side)
                continue
            hand = self.predicted_hand(side, now)
            if hand is None:
                continue
            predicted.append(hand)
            # Look for the hand where it is predicted to be
            self.hand_identity.follow(identity_side, hand.data.landmarks[0][:2])
        return predicted

    def stop(self):
        """Release the camera, the driver connection and the models."""
        print("\nCleaning up...")
        if self.depth_assist is not None:
            self.depth_assist.stop()
        self.camera.release()
        self.socket_client.close()
        self.hands.close()
        print("Cleanup complete")

    def show_preview(self, tracking_frame: TrackingFrame) -> bool:
        """
        Show the OpenCV preview window and handle its keys.

        Returns:
            False when the user asked to quit
        """
        frame = tracking_frame.frame_bgr
        if self.debug['show_landmarks']:
            for hand in tracking_frame.hands:
                self.draw_landmarks(frame, hand.data.landmarks, (160, 160, 160) if hand.predicted else (255, 0, 0))
        self.draw_info(frame, [hand.data for hand in tracking_frame.hands], tracking_frame.tracking_fps)
        cv2.imshow('Hand Tracking', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting...")
            return False
        if key == ord('s'):
            self.apply_setting('tracking.swap_hands', not self.config['tracking'].get('swap_hands', False))
        if key == ord('f'):
            self.cycle_filter_mode()
        if key == ord('b'):
            self.set_depth_source('mediapipe' if self.depth_source == 'wilor' else 'wilor')
        return True

    def run(self):
        """Main tracking loop with the OpenCV preview (command-line mode)."""
        print("\n=== Starting Hand Tracking ===")
        if not self.start():
            print("Exiting.")
            return
        self.cv_preview = bool(self.debug['show_video'])

        print("\nHand tracking active!")
        print("Press 'q' to quit, 's' to swap left/right, 'f' to cycle smoothing, 'b' to toggle WiLoR depth\n")

        try:
            while True:
                tracking_frame = self.step()
                if tracking_frame is None:
                    continue
                if self.cv_preview and not self.show_preview(tracking_frame):
                    break
                self.report_slow_frame(*self.last_timings, time.perf_counter())

        except CameraLostError as e:
            print(f"{e} Exiting.")
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"\nError in tracking loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
            cv2.destroyAllWindows()


def main():
    """Main entry point."""
    print("=" * 50)
    print("Hand Camera Driver for SteamVR")
    print("=" * 50)

    parser = argparse.ArgumentParser(description="Hand Camera Driver for SteamVR")
    parser.add_argument("config", nargs="?", default="config.json",
                        help="Path to configuration JSON file")
    parser.add_argument("--preset",
                        help="Preset to use, e.g. POV or Facing (default: the one last used)")
    parser.add_argument("--mode", choices=VIEW_MODES,
                        help="facing: camera in front of you; pov: camera looks where you look. "
                             "Uses that mode's built-in preset unless --preset is given")
    parser.add_argument("--swap-hands", action="store_true", default=None,
                        help="Swap left/right on top of what the mode implies")
    parser.add_argument("--rotate-180", action="store_true", default=None,
                        help="Camera is mounted upside down")
    parser.add_argument("--depth", choices=["mediapipe", "wilor"],
                        help="Depth source; wilor is experimental and needs its own environment")
    args = parser.parse_args()

    # Windows' default 15.6 ms timer tick puts a floor under every
    # cv2.waitKey(1) and sleep; that alone costs ~15 ms per frame.
    timer_raised = False
    if sys.platform == "win32":
        try:
            import ctypes
            timer_raised = ctypes.windll.winmm.timeBeginPeriod(1) == 0
        except (OSError, AttributeError):
            pass

    try:
        # Create and run tracker
        tracker = HandTracker(args.config, view_mode=args.mode,
                              swap_hands=args.swap_hands, rotate_180=args.rotate_180,
                              depth_source=args.depth, preset=args.preset)
        tracker.run()
    finally:
        if timer_raised:
            ctypes.windll.winmm.timeEndPeriod(1)


if __name__ == "__main__":
    main()
