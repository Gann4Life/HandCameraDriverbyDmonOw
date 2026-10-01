"""
Hand tracking camera capture and processing for SteamVR.
Captures video, detects hands using MediaPipe, and sends data to SteamVR driver.
"""
import cv2
import mediapipe as mp
import argparse
import json
import time
from typing import List, Tuple, Optional
from hand_data import HandData
from gesture_detector import GestureDetector, quat_from_euler_deg, quat_multiply, quat_rotate, quat_slerp
from utils.camera_utils import CameraCapture
from utils.socket_client import SocketClient


# "facing": camera in front of the user, looking back at them (selfie view).
# "pov":    camera looks the same way the user does (head/chest mounted).
VIEW_MODES = ("facing", "pov")


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
        palm_facing = str(tracking_config.get('palm_facing', 'away' if mode == 'pov' else 'auto')).lower()
        self.palm_away = palm_facing == 'away'

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
        self.calibration = self.config.get('calibration', {
            'position_offset': [0.0, 0.0, 0.0],
            'scale': 1.0
        })
        # Camera mounting relative to the HMD, as [pitch, yaw, roll] degrees
        # (e.g. a head-mounted camera tilted down toward the hands: negative pitch)
        self.camera_rotation = quat_from_euler_deg(*self.calibration.get('camera_rotation_deg', [0.0, 0.0, 0.0]))
        # Per-hand correction so the controller model sits like the real hand
        offsets = self.calibration.get('rotation_offset_deg', {})
        self.rotation_offsets = {
            hand: quat_from_euler_deg(*offsets.get(hand, [0.0, 0.0, 0.0]))
            for hand in ('left', 'right')
        }
        # Fraction of each new orientation sample to follow (1.0 = no smoothing)
        self.rotation_follow = min(1.0, max(0.05, float(self.calibration.get('rotation_follow', 0.5))))
        self.last_rotation = {}

        print("HandTracker initialized")
        print(f"Camera: {cam_config['width']}x{cam_config['height']} @ {cam_config['fps']}fps")
        print(f"Tracking: max {self.config['tracking']['max_hands']} hands")
        print(f"View: {self.view_mode}, hands {'swapped' if self.swap_hands else 'as detected'}"
              f"{', rotated 180' if self.camera.rotate_180 else ''}")
        print(f"Network: {net_config['host']}:{net_config['port']}")
    
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
    
    def process_hand_landmarks(self, hand_landmarks, hand_world_landmarks, hand_label: str,
                               frame_width: int, frame_height: int) -> HandData:
        """
        Process MediaPipe hand landmarks into HandData object.

        Args:
            hand_landmarks: MediaPipe hand landmarks (normalised image coordinates)
            hand_world_landmarks: MediaPipe world landmarks (metres), or None
            hand_label: "Left" or "Right"
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels

        Returns:
            HandData object with processed hand information
        """
        # Extract landmarks as list of tuples
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z))

        # Determine hand type
        is_left = hand_label == "Left"
        if self.swap_hands:
            is_left = not is_left
        world = None
        if hand_world_landmarks is not None:
            world = [(lm.x, lm.y, lm.z) for lm in hand_world_landmarks.landmark]
            if self.palm_away:
                # Seen from behind, the hand's own shape is a better handedness
                # signal than MediaPipe's palm-view label
                inferred = self.gesture_detector.infer_is_left_palm_away(world, self.mirror_x)
                if inferred is not None:
                    is_left = inferred
        hand_type = "left" if is_left else "right"

        # Calculate hand position (using wrist position)
        wrist = landmarks[0]
        
        # Convert normalized coordinates to world coordinates
        # Center the coordinates around 0 and scale appropriately
        x = (wrist[0] - 0.5) * 2.0  # Range: -1.0 to 1.0
        if self.mirror_x:
            x = -x
        y = -(wrist[1] - 0.5) * 2.0  # Range: -1.0 to 1.0, inverted
        
        # Estimate Z based on hand size (larger hand = closer to camera = more negative Z)
        palm_size = self.calculate_palm_size(landmarks)
        z = -0.5 - (palm_size * 2.0)  # Approximate depth
        
        # Apply calibration: scale in camera space, tilt into HMD space, then offset
        scale = self.calibration['scale']
        offset = self.calibration['position_offset']
        cx, cy, cz = quat_rotate(self.camera_rotation, (x * scale, y * scale, z * scale))
        position = (cx + offset[0], cy + offset[1], cz + offset[2])

        # Calculate hand orientation from the metric world landmarks; the
        # normalised ones have a squashed, image-relative Z and unequal X/Y scales.
        if world is not None:
            rotation = self.gesture_detector.calculate_hand_orientation(world, is_left, self.mirror_x,
                                                                        self.palm_away)
            rotation = quat_multiply(self.camera_rotation, rotation)
            rotation = quat_multiply(rotation, self.rotation_offsets[hand_type])
            previous = self.last_rotation.get(hand_type)
            if previous is not None:
                rotation = quat_slerp(previous, rotation, self.rotation_follow)
            self.last_rotation[hand_type] = rotation
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
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw hand information
        y_offset = 60
        for hand in hands_data:
            if hand.is_detected:
                info_text = f"{hand.hand_type.upper()}: {hand.gesture}"
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
        print("Press 'q' to quit, 's' to swap left/right\n")
        
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
                    time.sleep(0.01)
                    continue
                
                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = self.hands.process(frame_rgb)
                
                # Prepare hand data
                hands_data = []
                
                if results.multi_hand_landmarks and results.multi_handedness:
                    world_list = results.multi_hand_world_landmarks or [None] * len(results.multi_hand_landmarks)
                    for hand_landmarks, hand_world, handedness in zip(results.multi_hand_landmarks, world_list,
                                                                       results.multi_handedness):
                        # Get hand label
                        hand_label = handedness.classification[0].label

                        # Process hand
                        hand_data = self.process_hand_landmarks(
                            hand_landmarks, hand_world, hand_label,
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
                
                # Two hands can never be the same side; if classification said
                # so, the one further to the user's left is the left hand
                if len(hands_data) == 2 and hands_data[0].hand_type == hands_data[1].hand_type:
                    ordered = sorted(hands_data, key=lambda h: h.position[0])
                    ordered[0].hand_type, ordered[1].hand_type = "left", "right"

                # Send data to driver
                for hand_data in hands_data:
                    protocol_string = hand_data.to_protocol_string()
                    self.socket_client.send(protocol_string)
                
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
                        self.swap_hands = not self.swap_hands
                        print(f"Hands {'swapped' if self.swap_hands else 'as detected'}")
                
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

    # Create and run tracker
    tracker = HandTracker(args.config, view_mode=args.mode,
                          swap_hands=args.swap_hands, rotate_180=args.rotate_180)
    tracker.run()


if __name__ == "__main__":
    main()