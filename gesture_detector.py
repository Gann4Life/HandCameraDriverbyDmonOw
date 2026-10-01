"""
Gesture detection module for hand tracking.
Detects various hand gestures based on MediaPipe landmarks.
"""
import math
from typing import List, Tuple, Optional
import numpy as np


Quaternion = Tuple[float, float, float, float]  # (qw, qx, qy, qz)


def quat_from_matrix(m: np.ndarray) -> Quaternion:
    """
    Convert a 3x3 rotation matrix (basis vectors as columns) to a quaternion.

    Args:
        m: Proper rotation matrix

    Returns:
        Unit quaternion (qw, qx, qy, qz)
    """
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)
    n = math.sqrt(sum(c * c for c in q))
    return tuple(c / n for c in q)


def quat_multiply(a: Quaternion, b: Quaternion) -> Quaternion:
    """Hamilton product a * b (apply b, then a)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_from_euler_deg(pitch: float, yaw: float, roll: float) -> Quaternion:
    """
    Build a quaternion from degrees about X (pitch), Y (yaw) and Z (roll),
    applied as yaw, then pitch, then roll.
    """
    def axis(angle_deg, ix):
        half = math.radians(angle_deg) / 2.0
        q = [math.cos(half), 0.0, 0.0, 0.0]
        q[ix] = math.sin(half)
        return tuple(q)
    return quat_multiply(quat_multiply(axis(yaw, 2), axis(pitch, 1)), axis(roll, 3))


def quat_rotate(q: Quaternion, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Rotate vector v by unit quaternion q."""
    w, x, y, z = q
    r = quat_multiply(quat_multiply(q, (0.0, *v)), (w, -x, -y, -z))
    return (r[1], r[2], r[3])


def quat_slerp(a: Quaternion, b: Quaternion, t: float) -> Quaternion:
    """Spherical interpolation from a (t=0) to b (t=1), along the short arc."""
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0.0:
        b = tuple(-c for c in b)
        dot = -dot
    if dot > 0.9995:
        q = tuple(x + t * (y - x) for x, y in zip(a, b))
    else:
        theta = math.acos(dot)
        sa = math.sin((1.0 - t) * theta) / math.sin(theta)
        sb = math.sin(t * theta) / math.sin(theta)
        q = tuple(sa * x + sb * y for x, y in zip(a, b))
    n = math.sqrt(sum(c * c for c in q))
    return tuple(c / n for c in q)


class GestureDetector:
    """Detects hand gestures from MediaPipe landmarks."""
    
    # MediaPipe hand landmark indices
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20
    
    def __init__(self, pinch_threshold: float = 0.05, finger_extended_threshold: float = 0.6):
        """
        Initialize gesture detector.
        
        Args:
            pinch_threshold: Distance threshold for pinch detection
            finger_extended_threshold: Ratio threshold for finger extension detection
        """
        self.pinch_threshold = pinch_threshold
        self.finger_extended_threshold = finger_extended_threshold
        self._last_dorsal = {}  # is_left -> back-of-hand normal from the previous frame
    
    @staticmethod
    def calculate_distance(point1: Tuple[float, float, float], 
                          point2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two 3D points."""
        return math.sqrt(
            (point1[0] - point2[0]) ** 2 +
            (point1[1] - point2[1]) ** 2 +
            (point1[2] - point2[2]) ** 2
        )
    
    def is_finger_extended(self, landmarks: List[Tuple[float, float, float]], 
                          finger_name: str) -> bool:
        """
        Check if a finger is extended.
        
        Args:
            landmarks: List of 21 hand landmarks
            finger_name: One of 'thumb', 'index', 'middle', 'ring', 'pinky'
        
        Returns:
            True if finger is extended, False otherwise
        """
        if len(landmarks) < 21:
            return False
        
        finger_tips = {
            'thumb': self.THUMB_TIP,
            'index': self.INDEX_FINGER_TIP,
            'middle': self.MIDDLE_FINGER_TIP,
            'ring': self.RING_FINGER_TIP,
            'pinky': self.PINKY_TIP
        }
        
        finger_mcp = {
            'thumb': self.THUMB_MCP,
            'index': self.INDEX_FINGER_MCP,
            'middle': self.MIDDLE_FINGER_MCP,
            'ring': self.RING_FINGER_MCP,
            'pinky': self.PINKY_MCP
        }
        
        if finger_name not in finger_tips:
            return False
        
        tip_idx = finger_tips[finger_name]
        mcp_idx = finger_mcp[finger_name]
        
        # For thumb, use different logic (check distance from wrist)
        if finger_name == 'thumb':
            dist_tip_to_wrist = self.calculate_distance(landmarks[tip_idx], landmarks[self.WRIST])
            dist_mcp_to_wrist = self.calculate_distance(landmarks[mcp_idx], landmarks[self.WRIST])
            return dist_tip_to_wrist > dist_mcp_to_wrist * 1.2
        
        # For other fingers, check if tip is farther from wrist than MCP
        dist_tip_to_wrist = self.calculate_distance(landmarks[tip_idx], landmarks[self.WRIST])
        dist_mcp_to_wrist = self.calculate_distance(landmarks[mcp_idx], landmarks[self.WRIST])
        
        # Finger is extended if tip is significantly farther from wrist than MCP
        return dist_tip_to_wrist > dist_mcp_to_wrist * self.finger_extended_threshold
    
    def detect_pinch(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """
        Detect pinch gesture (thumb and index finger touching).
        
        Args:
            landmarks: List of 21 hand landmarks
        
        Returns:
            True if pinch is detected, False otherwise
        """
        if len(landmarks) < 21:
            return False
        
        thumb_tip = landmarks[self.THUMB_TIP]
        index_tip = landmarks[self.INDEX_FINGER_TIP]
        
        distance = self.calculate_distance(thumb_tip, index_tip)
        return distance < self.pinch_threshold
    
    def detect_gesture(self, landmarks: List[Tuple[float, float, float]]) -> str:
        """
        Detect the current gesture from hand landmarks.
        
        Args:
            landmarks: List of 21 hand landmarks
        
        Returns:
            Gesture name: 'FIST', 'POINT', 'OPEN', 'THUMBS_UP', 'PEACE', 'PINCH', 'UNKNOWN'
        """
        if len(landmarks) < 21:
            return 'UNKNOWN'
        
        # Check finger extension states
        thumb_extended = self.is_finger_extended(landmarks, 'thumb')
        index_extended = self.is_finger_extended(landmarks, 'index')
        middle_extended = self.is_finger_extended(landmarks, 'middle')
        ring_extended = self.is_finger_extended(landmarks, 'ring')
        pinky_extended = self.is_finger_extended(landmarks, 'pinky')
        
        # Count extended fingers
        extended_count = sum([
            thumb_extended, index_extended, middle_extended, 
            ring_extended, pinky_extended
        ])
        
        # Detect pinch first (highest priority)
        if self.detect_pinch(landmarks):
            return 'PINCH'
        
        # Fist: all fingers closed
        if extended_count == 0:
            return 'FIST'
        
        # Point: only index finger extended
        if index_extended and not middle_extended and not ring_extended and not pinky_extended and not thumb_extended:
            return 'POINT'
        
        # Peace sign: index and middle fingers extended
        if index_extended and middle_extended and not ring_extended and not pinky_extended:
            return 'PEACE'
        
        # Thumbs up: only thumb extended
        if thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return 'THUMBS_UP'
        
        # Open hand: all fingers extended
        if extended_count >= 4:
            return 'OPEN'
        
        return 'UNKNOWN'
    
    # In "away" mode the back of the hand is taken to face the camera. When the
    # hand is closer to edge-on than this (|cos| of the back-of-hand normal
    # against the camera axis), the previous frame's side is kept instead,
    # so the palm does not flicker between sides.
    PALM_EDGE_ON_THRESHOLD = 0.25

    @staticmethod
    def _user_frame_points(world_landmarks: List[Tuple[float, float, float]],
                           mirror_x: bool, flip_z: bool) -> Tuple[np.ndarray, float]:
        """
        Convert MediaPipe world landmarks into the user's OpenVR frame.

        Reflections are applied to the points before any frame is built from
        them, which keeps the result a proper rotation; reflecting a finished
        quaternion would not.

        Args:
            world_landmarks: 21 MediaPipe world landmarks (x right, y down,
                z away from the camera)
            mirror_x: Reflect X (image mirroring and/or a camera facing the user)
            flip_z: Reflect Z (camera facing the user, looking back toward them)

        Returns:
            (points, toward_camera_z): the converted points, and the sign of Z
            that points from the hands toward the camera in that frame
        """
        # MediaPipe camera axes -> OpenVR axes (y up, z toward the viewer)
        pts = np.array(world_landmarks, dtype=float) * np.array([1.0, -1.0, -1.0])
        if mirror_x:
            pts[:, 0] = -pts[:, 0]
        if flip_z:
            pts[:, 2] = -pts[:, 2]
        # A head-mounted camera sits behind the hands (+Z); one facing the
        # user sits in front of them (-Z)
        return pts, (-1.0 if flip_z else 1.0)

    def infer_is_left_palm_away(self, world_landmarks: List[Tuple[float, float, float]],
                                mirror_x: bool = False, flip_z: bool = False) -> Optional[bool]:
        """
        Tell left from right by hand geometry, given the back of the hand faces
        the camera. MediaPipe's own label assumes a palm-side selfie view and is
        unreliable when it sees the back of the hand.

        Args:
            world_landmarks: 21 MediaPipe world landmarks
            mirror_x: Same reflection as used for orientation
            flip_z: Same reflection as used for orientation

        Returns:
            True for left, False for right, None when the hand is too close to
            edge-on for the geometry to decide
        """
        if len(world_landmarks) < 21:
            return None
        pts, toward_camera_z = self._user_frame_points(world_landmarks, mirror_x, flip_z)
        forward = (pts[self.INDEX_FINGER_MCP] + pts[self.MIDDLE_FINGER_MCP] +
                   pts[self.RING_FINGER_MCP] + pts[self.PINKY_MCP]) / 4.0 - pts[self.WRIST]
        across = pts[self.INDEX_FINGER_MCP] - pts[self.PINKY_MCP]
        # Back-of-hand normal if this were a right hand
        dorsal_if_right = np.cross(forward, across)
        norm = np.linalg.norm(dorsal_if_right)
        if norm < 1e-9:
            return None
        facing = toward_camera_z * dorsal_if_right[2] / norm
        if abs(facing) < self.PALM_EDGE_ON_THRESHOLD:
            return None
        return facing < 0

    def calculate_hand_orientation(self, world_landmarks: List[Tuple[float, float, float]],
                                   is_left: bool, mirror_x: bool = False,
                                   palm_away: bool = False,
                                   flip_z: bool = False) -> Tuple[float, float, float, float]:
        """
        Calculate hand orientation as a quaternion in OpenVR camera space.

        The resulting frame follows the OpenVR controller convention: -Z points
        along the fingers (wrist to middle knuckle), +Y out of the back of the
        hand, +X completes a right-handed frame. A flat hand, palm down, fingers
        pointing away from the camera, is the identity rotation.

        Args:
            world_landmarks: 21 MediaPipe world landmarks (metres; x right,
                y down, z away from the camera)
            is_left: Whether this is the user's left hand
            mirror_x: Reflect the X axis to match a mirrored position mapping
            palm_away: Decide the palm side from the camera (back of the hand
                toward it, as in a first-person view) instead of from is_left
            flip_z: Reflect Z, for a camera facing the user

        Returns:
            Quaternion (qw, qx, qy, qz) representing hand orientation
        """
        if len(world_landmarks) < 21:
            return (1.0, 0.0, 0.0, 0.0)  # Identity quaternion

        pts, toward_camera_z = self._user_frame_points(world_landmarks, mirror_x, flip_z)

        wrist = pts[self.WRIST]
        knuckles = (pts[self.INDEX_FINGER_MCP] + pts[self.MIDDLE_FINGER_MCP] +
                    pts[self.RING_FINGER_MCP] + pts[self.PINKY_MCP]) / 4.0
        forward = knuckles - wrist
        # Across the palm, pinky side to thumb side
        across = pts[self.INDEX_FINGER_MCP] - pts[self.PINKY_MCP]

        # Back-of-hand normal. The thumb is on opposite sides for each hand,
        # so the cross product has to be taken in opposite orders.
        dorsal = np.cross(across, forward) if is_left else np.cross(forward, across)

        if palm_away:
            norm = np.linalg.norm(dorsal)
            facing = toward_camera_z * dorsal[2] / norm if norm > 1e-9 else 0.0
            previous = self._last_dorsal.get(is_left)
            if abs(facing) >= self.PALM_EDGE_ON_THRESHOLD:
                if facing < 0:
                    dorsal = -dorsal
            elif previous is not None and np.dot(dorsal, previous) < 0:
                dorsal = -dorsal
            self._last_dorsal[is_left] = dorsal

        z_axis = -forward
        z_norm = np.linalg.norm(z_axis)
        if z_norm < 1e-6:
            return (1.0, 0.0, 0.0, 0.0)
        z_axis /= z_norm

        y_axis = dorsal - np.dot(dorsal, z_axis) * z_axis
        y_norm = np.linalg.norm(y_axis)
        if y_norm < 1e-6:
            return (1.0, 0.0, 0.0, 0.0)
        y_axis /= y_norm

        x_axis = np.cross(y_axis, z_axis)

        return quat_from_matrix(np.column_stack((x_axis, y_axis, z_axis)))
    
    def get_trigger_value(self, gesture: str) -> float:
        """
        Get trigger value based on gesture.
        
        Args:
            gesture: Current gesture name
        
        Returns:
            Trigger value between 0.0 and 1.0
        """
        if gesture == 'POINT':
            return 0.8
        elif gesture == 'PINCH':
            return 1.0
        else:
            return 0.0
    
    def get_grip_value(self, gesture: str) -> float:
        """
        Get grip value based on gesture.
        
        Args:
            gesture: Current gesture name
        
        Returns:
            Grip value between 0.0 and 1.0
        """
        if gesture == 'FIST':
            return 1.0
        elif gesture == 'PINCH':
            return 0.5
        else:
            return 0.0