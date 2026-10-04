"""
Data class for hand tracking information.
"""
import math
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

# The driver drops a line with a coordinate beyond this (kMaxPositionM in hand_message.cpp)
MAX_POSITION_M = 10.0

# docs/PROTOCOL.md version; the driver refuses a tracker that greets with another (kProtocolVersion)
PROTOCOL_VERSION = 1


def protocol_greeting(controller_type: str = "touch") -> str:
    """
    The first line on every connection to the driver, which closes connections without it.
    TYPE is repeated from the hand lines because drivers older than the greeting read this
    line as a hand message, and the first one they read fixes the controller type.
    """
    return f"HELLO:HANDCAM,VERSION:{PROTOCOL_VERSION},{_type_field(controller_type)}"


def _type_field(controller_type: str) -> str:
    return f"TYPE:{controller_type.upper()}"


def _unit(value: float) -> float:
    """value clamped to 0..1, as the protocol defines trigger, grip and curls; NaN and infinity become 0."""
    return min(1.0, max(0.0, value)) if math.isfinite(value) else 0.0


@dataclass
class HandData:
    """Encapsulates all data for a tracked hand."""
    
    hand_type: str  # "left" or "right"
    position: Tuple[float, float, float]  # metres from the headset, OpenVR axes (see docs/PROTOCOL.md)
    rotation: Tuple[float, float, float, float]  # qw, qx, qy, qz quaternion
    gesture: str  # Current gesture name
    trigger_value: float  # 0.0-1.0
    grip_value: float  # 0.0-1.0
    landmarks: List[Tuple[float, float, float]]  # 21 hand landmarks
    is_detected: bool = True
    finger_curls: Tuple[float, ...] = ()  # thumb..pinky, 0 straight .. 1 curled; empty without features

    def is_sendable_pose(self) -> bool:
        """
        False when the driver would drop the whole line for its pose (docs/PROTOCOL.md): a NaN or
        infinite value, a coordinate beyond MAX_POSITION_M, or a quaternion near zero (stricter than
        the driver, which drops only what rounds to zero; a real rotation has length 1).
        """
        if not all(math.isfinite(v) for v in (*self.position, *self.rotation)):
            return False
        return (all(abs(v) <= MAX_POSITION_M for v in self.position)
                and math.sqrt(sum(c * c for c in self.rotation)) >= 1e-3)

    def to_protocol_string(self, controller_type: str = "touch", room_anchor: bool = False) -> str:
        """
        Convert hand data to protocol string for socket transmission.
        Format: HAND:LEFT,X:0.5,Y:0.3,Z:-0.2,QW:1.0,QX:0.0,QY:0.0,QZ:0.0,TRIGGER:0.8,GRIP:0.0,GESTURE:POINT,
        TYPE:INDEX,CURL:0.10;0.20;0.30;0.40;0.50,ANCHOR:ROOM

        Args:
            controller_type: What the driver presents the hands as, "touch" or "index"
            room_anchor: The camera is fixed in the room: the driver places the hands
                facing the tracking space's forward instead of turning them with the headset
        """
        curls = f",CURL:{';'.join(f'{_unit(c):.2f}' for c in self.finger_curls)}" if self.finger_curls else ""
        anchor = ",ANCHOR:ROOM" if room_anchor else ""
        return (
            f"HAND:{self.hand_type.upper()},"
            f"X:{self.position[0]:.4f},"
            f"Y:{self.position[1]:.4f},"
            f"Z:{self.position[2]:.4f},"
            f"QW:{self.rotation[0]:.4f},"
            f"QX:{self.rotation[1]:.4f},"
            f"QY:{self.rotation[2]:.4f},"
            f"QZ:{self.rotation[3]:.4f},"
            f"TRIGGER:{_unit(self.trigger_value):.2f},"
            f"GRIP:{_unit(self.grip_value):.2f},"
            f"GESTURE:{self.gesture},"
            f"{_type_field(controller_type)}"
            f"{curls}"
            f"{anchor}"
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
