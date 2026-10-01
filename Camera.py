"""
Hand tracking camera capture and processing for SteamVR.
Captures video, detects hands using MediaPipe, and sends data to SteamVR driver.
"""
import cv2
import mediapipe as mp
import argparse
import json
import math
import sys
import time
import numpy as np
from typing import List, Tuple, Optional
from hand_data import HandData
from hand_identity import HandDetection, HandIdentityTracker
from gesture_detector import GestureDetector, quat_from_euler_deg, quat_multiply, quat_rotate
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


class HandTracker:
    """Main hand tracking system."""

    def __init__(self, config_path: str = "config.json",
                 view_mode: Optional[str] = None,
                 swap_hands: Optional[bool] = None,
                 rotate_180: Optional[bool] = None):
        """
        Initialize hand tracker with configuration.

        Args:
            config_path: Path to configuration JSON file
            view_mode: "facing" or "pov"; overrides tracking.view_mode
            swap_hands: Extra left/right swap; overrides tracking.swap_hands
            rotate_180: Camera mounted upside down; overrides camera.rotate_180
        """
        # Load configuration
        self.config = self.load_config(config_path)

        # Keep full speed while the VR game has focus: a CPU-heavy foreground
        # app otherwise starves a normal-priority background process
        process_config = self.config.get('process', {})
        for line in keep_running_in_background(process_config.get('priority', 'above_normal'),
                                               process_config.get('disable_power_throttling', True)):
            print(line)
        self._last_slow_report = 0.0

        # Resolve view mode and handedness mapping
        tracking_config = self.config['tracking']
        mode = str(view_mode or tracking_config.get('view_mode', 'facing')).lower()
        if mode not in VIEW_MODES:
            print(f"Warning: unknown view_mode '{mode}', using facing")
            mode = "facing"
        self.view_mode = mode
        if swap_hands is None:
            swap_hands = tracking_config.get('swap_hands', False)
        # Whether the frame MediaPipe sees is a mirror image: the source may
        # already be mirrored (some phone-camera apps do it) and flip_horizontal
        # mirrors it again.
        cam_config = self.config['camera']
        mirrored = bool(cam_config.get('source_mirrored', False)) != bool(cam_config['flip_horizontal'])
        # MediaPipe labels handedness assuming a mirrored (selfie) image, so an
        # unmirrored frame comes out with Left/Right swapped, whatever the mode.
        self.swap_hands = (not mirrored) != bool(swap_hands)
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
        self.facing_distance = float(cam_config.get('facing_distance', 0.8))
        identity_config = tracking_config.get('identity', {})
        self.hand_identity = HandIdentityTracker(
            continuity_radius=float(identity_config.get('continuity_radius', 0.15)),
            memory_seconds=float(identity_config.get('memory_seconds', 0.4)),
            switch_frames=int(identity_config.get('switch_frames', 6)),
            duplicate_radius=float(identity_config.get('duplicate_radius', 0.05)),
            user_left_is_image_left=not self.mirror_x,
            order_weight=float(identity_config.get('order_weight', 1.5)))

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config['tracking']['max_hands'],
            min_detection_confidence=self.config['tracking']['detection_confidence'],
            min_tracking_confidence=self.config['tracking']['tracking_confidence'],
            model_complexity=self.config['tracking']['model_complexity']
        )
        
        # Initialize camera
        if rotate_180 is None:
            rotate_180 = cam_config.get('rotate_180', False)
        self.stall_timeout = float(cam_config.get('stall_timeout', 2.0))
        self.reconnect_timeout = float(cam_config.get('reconnect_timeout', 30.0))
        self.camera = CameraCapture(
            device_id=cam_config['device_id'],
            width=cam_config['width'],
            height=cam_config['height'],
            fps=cam_config['fps'],
            flip_horizontal=cam_config['flip_horizontal'],
            backend=cam_config.get('backend', 'auto'),
            rotate_180=bool(rotate_180)
        )
        
        # Initialize gesture detector
        gesture_config = self.config['gestures']
        self.gesture_detector = GestureDetector(
            pinch_threshold=gesture_config['pinch_threshold'],
            finger_extended_threshold=gesture_config['finger_extended_threshold']
        )
        
        # Initialize socket client
        net_config = self.config['network']
        self.socket_client = SocketClient(
            host=net_config['host'],
            port=net_config['port']
        )
        
        # Debug settings
        self.debug = self.config['debug']
        
        # Calibration
        # Shared values at the top level; the active mode's section (e.g.
        # calibration.pov) overrides them, so each mode keeps its own offsets
        base_calibration = {'position_offset': [0.0, 0.0, 0.0], 'scale': 1.0}
        base_calibration.update(self.config.get('calibration', {}))
        self.calibration = {k: v for k, v in base_calibration.items() if k not in VIEW_MODES}
        self.calibration.update(base_calibration.get(self.view_mode, {}))
        # Camera mounting relative to the HMD, as [pitch, yaw, roll] degrees
        # (e.g. a head-mounted camera tilted down toward the hands: negative pitch)
        self.camera_rotation = quat_from_euler_deg(*self.calibration.get('camera_rotation_deg', [0.0, 0.0, 0.0]))
        # Per-hand correction so the controller model sits like the real hand
        offsets = self.calibration.get('rotation_offset_deg', {})
        self.rotation_offsets = {
            hand: quat_from_euler_deg(*offsets.get(hand, [0.0, 0.0, 0.0]))
            for hand in ('left', 'right')
        }
        # Smoothing, per hand: image-plane position, depth (noisier, so
        # filtered on its own) and rotation. 'f' in the preview cycles modes.
        self.filter_config = self.calibration.get('filter', {})
        mode = str(self.filter_config.get('mode', 'one_euro')).lower()
        self.filter_mode = mode if mode in FILTER_MODES else 'one_euro'
        self.build_filters()
        # Metric position: horizontal field of view of the camera, and how the
        # user's hand compares to MediaPipe's average-sized hand model
        self.hfov_deg = float(cam_config.get('hfov_deg', 70.0))
        self.hand_scale = float(self.calibration.get('hand_scale', 1.0))
        self.last_camera_position = {}

        print("HandTracker initialized")
        print(f"Camera: {cam_config['width']}x{cam_config['height']} @ {cam_config['fps']}fps")
        print(f"Tracking: max {self.config['tracking']['max_hands']} hands")
        print(f"View: {self.view_mode}, hands {'swapped' if self.swap_hands else 'as detected'}"
              f"{', rotated 180' if self.camera.rotate_180 else ''}")
        print(f"Network: {net_config['host']}:{net_config['port']}")
    
    def estimate_wrist_position(self, hand_landmarks, world: List[Tuple[float, float, float]],
                                frame_width: int, frame_height: int) -> Optional[Tuple[float, float, float]]:
        """
        Locate the wrist in 3D by fitting MediaPipe's metric hand model to where
        its landmarks appear in the image (perspective-n-point).

        MediaPipe world landmarks use OpenCV camera axes (x right, y down,
        z forward), so solvePnP's translation is directly the hand's offset
        from the camera.

        Args:
            hand_landmarks: Normalised image landmarks
            world: World landmarks in metres, centred on the hand
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels

        Returns:
            Wrist (x, y, z) in OpenVR camera space (y up, -z forward), or None
            if the fit failed
        """
        object_points = np.array(world, dtype=np.float64) * self.hand_scale
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
        wrist = rotation_matrix @ object_points[0] + tvec.reshape(3)
        # Behind the camera or implausibly far means the fit went wrong
        if not 0.05 < wrist[2] < 3.0:
            return None
        x = -wrist[0] if self.mirror_x else wrist[0]
        return (float(x), float(-wrist[1]), float(-wrist[2]))

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
        self.last_camera_position[hand_type] = position
        return position

    def build_filters(self):
        """(Re)create the per-hand filters for the current filter_mode."""
        config = self.filter_config
        hands = ('left', 'right')
        if self.filter_mode == 'one_euro':
            def make(kind, cls, defaults):
                s = dict(defaults, **config.get(kind, {}))
                return {h: cls(s['min_cutoff'], s['beta'], config.get('d_cutoff', 1.0)) for h in hands}
            self.position_filters = make('position', OneEuroFilter, {'min_cutoff': 1.0, 'beta': 1.5})
            self.depth_filters = make('depth', OneEuroFilter, {'min_cutoff': 0.3, 'beta': 2.0})
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
        self.filter_mode = FILTER_MODES[(FILTER_MODES.index(self.filter_mode) + 1) % len(FILTER_MODES)]
        self.build_filters()
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
            if self.debug['show_video']:
                cv2.waitKey(1)
            time.sleep(1.0)
        return False

    def load_config(self, config_path: str) -> dict:
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
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using default configuration")
            # Return default config
            return {
                "camera": {"device_id": 0, "width": 640, "height": 480, "fps": 30, "flip_horizontal": True, "source_mirrored": False, "backend": "auto", "rotate_180": False},
                "tracking": {"max_hands": 2, "detection_confidence": 0.7, "tracking_confidence": 0.5, "model_complexity": 1, "view_mode": "facing", "swap_hands": False},
                "network": {"host": "127.0.0.1", "port": 65432},
                "gestures": {"pinch_threshold": 0.05, "finger_extended_threshold": 0.6},
                "calibration": {"position_offset": [0.0, 0.0, 0.0], "scale": 1.0},
                "debug": {"show_video": True, "show_landmarks": True, "show_fps": True, "log_gestures": False}
            }
    
    def handedness_evidence(self, hand_world_landmarks, handedness) -> float:
        """
        This frame's left/right vote for one hand, in [-1, 1] (> 0 is left).
        Only a vote: HandIdentityTracker weighs it against continuity.

        Finger curl leads: fingers only bend toward the palm, so it holds for
        any hand rotation. The secondary signal assumes which side faces the
        camera (palm-away geometry in POV, MediaPipe's selfie label otherwise),
        so it inverts when the hand turns over; it only gets a small weight,
        enough to decide a flat hand but never to outvote a clear curl.
        """
        classification = handedness.classification[0]
        mp_is_left = (classification.label == "Left") != self.swap_hands
        mp_vote = classification.score if mp_is_left else -classification.score
        if hand_world_landmarks is None:
            return mp_vote
        world = [(lm.x, lm.y, lm.z) for lm in hand_world_landmarks.landmark]
        curl = self.gesture_detector.handedness_evidence_curl(world)
        if self.palm_away:
            secondary = self.gesture_detector.handedness_evidence_palm_away(world, self.mirror_x, self.flip_z)
        else:
            secondary = mp_vote
        return max(-1.0, min(1.0, 0.8 * curl + 0.2 * secondary))

    def process_hand_landmarks(self, hand_landmarks, hand_world_landmarks, hand_type: str,
                               frame_width: int, frame_height: int) -> HandData:
        """
        Process MediaPipe hand landmarks into HandData object.

        Args:
            hand_landmarks: MediaPipe hand landmarks (normalised image coordinates)
            hand_world_landmarks: MediaPipe world landmarks (metres), or None
            hand_type: "left" or "right", as decided by HandIdentityTracker
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels

        Returns:
            HandData object with processed hand information
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
        camera_position = None
        if world is not None:
            camera_position = self.estimate_wrist_position(hand_landmarks, world, frame_width, frame_height)
        if camera_position is None:
            camera_position = self.last_camera_position.get(hand_type)
        if camera_position is None:
            camera_position = self.estimate_wrist_position_fallback(landmarks)
        now = time.perf_counter()
        camera_position = self.smooth_camera_position(hand_type, camera_position, now)

        # Apply calibration: scale in camera space, tilt into HMD space, then offset
        scale = self.calibration['scale']
        offset = self.calibration['position_offset']
        x, y, z = camera_position
        if self.flip_z:
            # Distance from a camera in front of the user becomes distance
            # forward of the user: a hand nearer the camera is further out
            z = -z - self.facing_distance
        cx, cy, cz = quat_rotate(self.camera_rotation, (x * scale, y * scale, z * scale))
        position = (cx + offset[0], cy + offset[1], cz + offset[2])

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

        # Detect gesture
        gesture = self.gesture_detector.detect_gesture(landmarks)

        # Calculate trigger and grip values
        trigger_value = self.gesture_detector.get_trigger_value(gesture)
        grip_value = self.gesture_detector.get_grip_value(gesture)

        return HandData(
            hand_type=hand_type,
            position=position,
            rotation=rotation,
            gesture=gesture,
            trigger_value=trigger_value,
            grip_value=grip_value,
            landmarks=landmarks,
            is_detected=True
        )
    
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
    
    def draw_landmarks(self, frame, hand_landmarks):
        """
        Draw hand landmarks on frame.
        
        Args:
            frame: OpenCV frame
            hand_landmarks: MediaPipe hand landmarks
        """
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )
    
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
        
        # Draw hand information
        y_offset = 60
        for hand in hands_data:
            if hand.is_detected:
                depth_cm = -self.last_camera_position.get(hand.hand_type, (0.0, 0.0, 0.0))[2] * 100.0
                info_text = f"{hand.hand_type.upper()}: {hand.gesture}  {depth_cm:.0f} cm"
                cv2.putText(frame, info_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_offset += 30
    
    def run(self):
        """Main tracking loop."""
        print("\n=== Starting Hand Tracking ===")
        
        # Start camera
        if not self.camera.start():
            print("Failed to start camera. Exiting.")
            return
        
        # Connect to driver
        print("Connecting to SteamVR driver...")
        if not self.socket_client.connect():
            print("Warning: Could not connect to driver. Will keep trying...")
        
        print("\nHand tracking active!")
        print("Press 'q' to quit, 's' to swap left/right, 'f' to cycle smoothing\n")
        
        try:
            while True:
                # Read frame
                ret, frame = self.camera.read_frame()
                if not ret:
                    if self.camera.is_stalled(self.stall_timeout):
                        print(f"Camera stopped delivering frames for {self.stall_timeout:.1f}s. Reconnecting...")
                        if not self.reconnect_camera():
                            print(f"Camera did not come back within {self.reconnect_timeout:.0f}s. Exiting.")
                            break
                        continue
                    # read_frame already waited for a frame; nothing more to wait for
                    continue
                
                t_frame = time.perf_counter()

                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process with MediaPipe
                results = self.hands.process(frame_rgb)
                t_tracked = time.perf_counter()

                # Prepare hand data
                hands_data = []
                
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

                    for i, (hand_landmarks, hand_world) in enumerate(zip(results.multi_hand_landmarks, world_list)):
                        if i not in sides:
                            continue  # duplicate of a hand already kept

                        # Process hand
                        hand_data = self.process_hand_landmarks(
                            hand_landmarks, hand_world, sides[i],
                            frame.shape[1], frame.shape[0]
                        )
                        hands_data.append(hand_data)
                        
                        # Draw landmarks if enabled
                        if self.debug['show_landmarks']:
                            self.draw_landmarks(frame, hand_landmarks)
                        
                        # Log gesture if enabled
                        if self.debug['log_gestures']:
                            print(f"{hand_data.hand_type}: {hand_data.gesture} "
                                  f"T:{hand_data.trigger_value:.2f} G:{hand_data.grip_value:.2f}")
                
                # Send data to driver
                for hand_data in hands_data:
                    protocol_string = hand_data.to_protocol_string()
                    self.socket_client.send(protocol_string)
                t_sent = time.perf_counter()

                # Draw info overlay
                if self.debug['show_video']:
                    self.draw_info(frame, hands_data, self.camera.get_fps())
                    cv2.imshow('Hand Tracking', frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nQuitting...")
                        break
                    if key == ord('s'):
                        # Manual correction: exchange the current identities now
                        # (and flip MediaPipe's label vote, its only other input)
                        self.swap_hands = not self.swap_hands
                        self.hand_identity.swap()
                        self.build_filters()  # each hand's history belonged to the other
                        print("Hands swapped")
                    if key == ord('f'):
                        self.cycle_filter_mode()

                self.report_slow_frame(t_frame, t_tracked, t_sent, time.perf_counter())

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"\nError in tracking loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            print("\nCleaning up...")
            self.camera.release()
            self.socket_client.close()
            self.hands.close()
            cv2.destroyAllWindows()
            print("Cleanup complete")


def main():
    """Main entry point."""
    print("=" * 50)
    print("Hand Camera Driver for SteamVR")
    print("=" * 50)
    
    parser = argparse.ArgumentParser(description="Hand Camera Driver for SteamVR")
    parser.add_argument("config", nargs="?", default="config.json",
                        help="Path to configuration JSON file")
    parser.add_argument("--mode", choices=VIEW_MODES,
                        help="facing: camera in front of you; pov: camera looks where you look")
    parser.add_argument("--swap-hands", action="store_true", default=None,
                        help="Swap left/right on top of what the mode implies")
    parser.add_argument("--rotate-180", action="store_true", default=None,
                        help="Camera is mounted upside down")
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
                              swap_hands=args.swap_hands, rotate_180=args.rotate_180)
        tracker.run()
    finally:
        if timer_raised:
            ctypes.windll.winmm.timeEndPeriod(1)


if __name__ == "__main__":
    main()