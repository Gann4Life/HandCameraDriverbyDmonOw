"""
Data class for hand tracking information.
"""
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field

import numpy as np

from hand_features import HandFeatures

# Landmark index pairs that form the hand skeleton (MediaPipe's HAND_CONNECTIONS)
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


@dataclass
class HandData:
    """Encapsulates all data for a tracked hand."""
    
    hand_type: str  # "left" or "right"
    position: Tuple[float, float, float]  # x, y, z in world coordinates
    rotation: Tuple[float, float, float, float]  # qw, qx, qy, qz quaternion
    gesture: str  # Current gesture name
    trigger_value: float  # 0.0-1.0
    grip_value: float  # 0.0-1.0
    landmarks: List[Tuple[float, float, float]]  # 21 hand landmarks
    is_detected: bool = True
    
    def to_protocol_string(self) -> str:
        """
        Convert hand data to protocol string for socket transmission.
        Format: HAND:LEFT,X:0.5,Y:0.3,Z:-0.2,QW:1.0,QX:0.0,QY:0.0,QZ:0.0,TRIGGER:0.8,GRIP:0.0,GESTURE:POINT
        """
        return (
            f"HAND:{self.hand_type.upper()},"
            f"X:{self.position[0]:.4f},"
            f"Y:{self.position[1]:.4f},"
            f"Z:{self.position[2]:.4f},"
            f"QW:{self.rotation[0]:.4f},"
            f"QX:{self.rotation[1]:.4f},"
            f"QY:{self.rotation[2]:.4f},"
            f"QZ:{self.rotation[3]:.4f},"
            f"TRIGGER:{self.trigger_value:.2f},"
            f"GRIP:{self.grip_value:.2f},"
            f"GESTURE:{self.gesture}"
        )
    
    @staticmethod
    def create_default(hand_type: str) -> 'HandData':
        """Create a default HandData object with neutral values."""
        return HandData(
            hand_type=hand_type,
            position=(0.0, 0.0, 0.0),
            rotation=(1.0, 0.0, 0.0, 0.0),  # Identity quaternion
            gesture="OPEN",
            trigger_value=0.0,
            grip_value=0.0,
            landmarks=[],
            is_detected=False
        )


@dataclass
class TrackedHand:
    """One hand in a TrackingFrame, with what the previews need to draw it."""

    data: HandData
    camera_position: Tuple[float, float, float]  # filtered wrist, OpenVR camera space (y up, -z forward)
    camera_points: Optional[np.ndarray] = None   # 21 x 3 joints in the same space, or None without a metric fit
    features: Optional["HandFeatures"] = None     # curls, splay, pinch; None without world landmarks
    gesture_scores: Dict[str, float] = field(default_factory=dict)  # 0..1 per gesture name


@dataclass
class TrackingFrame:
    """Everything one tracking step produced, as an immutable snapshot for display."""

    frame_rgb: np.ndarray
    frame_bgr: np.ndarray
    hands: List[TrackedHand]
    timings_ms: Dict[str, float] = field(default_factory=dict)
    tracking_fps: float = 0.0
    camera_fps: float = 0.0
    driver_connected: bool = False
    depth_label: str = "MediaPipe"
    hfov_deg: float = 70.0
