"""
The settings the GUI exposes, declared as data. Adding a setting is one entry
in SETTINGS: the settings panel builds its editor from it and the tracker
applies the change live (see HandTracker.setting_action).
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class Apply(Enum):
    """When a change takes effect, shown next to the setting."""

    LIVE = ""
    CAMERA = "reopens the camera"
    MODEL = "reloads the hand model"
    NEXT_START = "next start"
    RECONNECT = "reconnects to the driver"


@dataclass(frozen=True)
class Setting:
    key: str  # dotted config key; "{mode}" stands for the active view mode
    label: str
    kind: str  # "float", "int", "bool", "choice", "vec3", "text" or "resolution"
    section: str
    help: str = ""
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.01
    decimals: int = 2
    unit: str = ""
    choices: Tuple[Tuple[Any, str], ...] = ()
    apply: Apply = Apply.LIVE
    advanced: bool = False
    # (value, message): switching to value asks the user to confirm first
    confirm: Optional[Tuple[Any, str]] = None

    def resolve(self, view_mode: str) -> str:
        return self.key.replace("{mode}", view_mode)

    def changes(self, value: Any, view_mode: str) -> Dict[str, Any]:
        """Config keys and values one edit of this setting writes."""
        if self.kind == "resolution":
            width, height = (int(v) for v in str(value).split("x"))
            return {"camera.width": width, "camera.height": height}
        return {self.resolve(view_mode): value}


RESOLUTIONS = ((640, 480), (800, 600), (1280, 720), (1920, 1080))

WILOR_WARNING = (
    "WiLoR runs a large 3D hand model on the GPU next to your VR game.\n\n"
    "• High GPU load and several GB of video memory: the game may lose frame rate.\n"
    "• About 200 ms of extra lag on depth.\n"
    "• NVIDIA GPU only, and a separate install (install-depth.bat, about 3 GB).\n"
    "• Personal, non-commercial use only (WiLoR CC BY-NC-ND, MANO, Ultralytics AGPL).\n\n"
    "Turn it on?"
)

SECTIONS = ("Camera", "View", "Placement", "Depth", "Smoothing", "Gestures", "Tracking model",
            "Hand identity", "Driver connection", "Performance")

SETTINGS = (
    # Camera
    Setting("camera.device_id", "Camera index", "int", "Camera", maximum=20, step=1,
            help="0 is usually the built-in webcam, 1 the next camera, and so on.", apply=Apply.CAMERA),
    Setting("camera.resolution", "Resolution", "resolution", "Camera",
            help="Higher is not always better: 640x480 is fastest.", apply=Apply.CAMERA),
    Setting("camera.fps", "Frame rate", "choice", "Camera", choices=((30, "30 fps"), (60, "60 fps")),
            help="60 fps lowers lag, if your camera supports it.", apply=Apply.CAMERA),
    Setting("camera.flip_horizontal", "Mirror image", "bool", "Camera",
            help="Flip the picture left to right."),
    Setting("camera.rotate_180", "Upside down", "bool", "Camera",
            help="The camera is mounted upside down."),
    Setting("camera.hfov_deg", "Field of view", "float", "Camera", minimum=30, maximum=130, step=1,
            decimals=0, unit="°", help="Horizontal field of view of the camera. Wrong values make hands "
                                       "look nearer or farther than they are."),
    Setting("camera.source_mirrored", "Source already mirrored", "bool", "Camera",
            help="Phone front cameras often send a mirrored picture. To check, turn Mirror image off, face "
                 "the camera and raise your right hand: if it shows on the right of the preview, turn this on."),
    Setting("camera.backend", "Capture backend", "choice", "Camera", advanced=True, apply=Apply.CAMERA,
            choices=(("auto", "Automatic"), ("msmf", "Media Foundation"), ("dshow", "DirectShow"))),
    Setting("camera.stall_timeout", "Reconnect after", "float", "Camera", advanced=True, minimum=0.5,
            maximum=10, step=0.5, decimals=1, unit=" s", help="Seconds without frames before reconnecting."),

    # View
    Setting("tracking.view_mode", "Camera position", "choice", "View",
            choices=(("pov", "On my head or chest, looking where I look"),
                     ("facing", "In front of me, looking at me")),
            help="Each position keeps its own placement settings."),
    Setting("camera.facing_distance", "Distance to camera", "float", "View", minimum=0.2, maximum=3.0,
            step=0.05, unit=" m", help="Only for a camera in front of you."),
    Setting("tracking.swap_hands", "Swap left and right", "bool", "View",
            help="Use it if your hands come out the wrong way round. If they also move the wrong way, "
                 "check Source already mirrored instead."),
    Setting("tracking.palm_facing", "Palms face", "choice", "View", advanced=True,
            choices=(("mode", "Depends on camera position"), ("away", "Away from the camera"),
                     ("auto", "Detect automatically"))),

    # Placement (per view mode)
    Setting("calibration.{mode}.position_offset", "Position offset", "vec3", "Placement", minimum=-1.0,
            maximum=1.0, step=0.01, unit=" m", help="Moves both hands: X right, Y up, Z back."),
    Setting("calibration.{mode}.camera_rotation_deg", "Camera tilt", "vec3", "Placement", minimum=-180,
            maximum=180, step=1, decimals=0, unit="°", help="Pitch, yaw, roll of the camera relative to your head."),
    Setting("calibration.hand_scale", "Hand size", "float", "Placement", minimum=0.6, maximum=1.4,
            step=0.01, help="1 is an average hand. Raise it if your hands look too close."),
    Setting("calibration.scale", "Movement scale", "float", "Placement", minimum=0.2, maximum=3.0,
            step=0.05, help="Multiplies how far the hands move."),
    Setting("calibration.rotation_offset_deg.left", "Left hand rotation", "vec3", "Placement",
            minimum=-180, maximum=180, step=1, decimals=0, unit="°", advanced=True,
            help="How the controller sits in your hand. Shared by every camera position."),
    Setting("calibration.rotation_offset_deg.right", "Right hand rotation", "vec3", "Placement",
            minimum=-180, maximum=180, step=1, decimals=0, unit="°", advanced=True,
            help="How the controller sits in your hand. Shared by every camera position."),

    # Depth
    Setting("tracking.depth_source", "Depth source", "choice", "Depth",
            choices=(("mediapipe", "Standard (light)"), ("wilor", "WiLoR 3D model (heavy, experimental)")),
            help="WiLoR gives steadier distance but uses a lot of GPU.", confirm=("wilor", WILOR_WARNING)),
    Setting("tracking.depth_assist.max_rate_hz", "WiLoR max rate", "float", "Depth", minimum=1, maximum=30,
            step=1, decimals=0, unit=" Hz", advanced=True, help="Lower leaves more GPU for the game."),
    Setting("tracking.depth_assist.scale", "WiLoR depth scale", "float", "Depth", minimum=0.5, maximum=1.5,
            step=0.01, advanced=True),
    Setting("tracking.depth_assist.max_age", "WiLoR max age", "float", "Depth", minimum=0.1, maximum=2.0,
            step=0.05, unit=" s", advanced=True),
    Setting("tracking.depth_assist.match_radius", "WiLoR match radius", "float", "Depth", minimum=0.02,
            maximum=0.5, step=0.01, advanced=True),

    # Smoothing
    Setting("calibration.filter.mode", "Smoothing", "choice", "Smoothing",
            choices=(("one_euro", "Adaptive (One Euro)"), ("ema", "Simple (EMA)"), ("none", "Off"))),
    Setting("calibration.filter.position.min_cutoff", "Position steadiness", "float", "Smoothing",
            minimum=0.05, maximum=5.0, step=0.05, help="Lower is steadier when still, but laggier."),
    Setting("calibration.filter.position.beta", "Position responsiveness", "float", "Smoothing",
            minimum=0.0, maximum=10.0, step=0.1, help="Higher follows fast moves with less lag."),
    Setting("calibration.filter.depth.min_cutoff", "Depth steadiness", "float", "Smoothing",
            minimum=0.05, maximum=5.0, step=0.05),
    Setting("calibration.filter.depth.beta", "Depth responsiveness", "float", "Smoothing",
            minimum=0.0, maximum=10.0, step=0.1),
    Setting("calibration.filter.rotation.min_cutoff", "Rotation steadiness", "float", "Smoothing",
            minimum=0.05, maximum=5.0, step=0.05, advanced=True),
    Setting("calibration.filter.rotation.beta", "Rotation responsiveness", "float", "Smoothing",
            minimum=0.0, maximum=10.0, step=0.1, advanced=True),
    Setting("calibration.filter.d_cutoff", "Speed cutoff", "float", "Smoothing", minimum=0.1, maximum=5.0,
            step=0.1, advanced=True),

    # Gestures
    # Curl is 0 for a straight finger and 1 for a fully curled one (see the Hands tab)
    Setting("gestures.grip_curl_start", "Grip starts at", "float", "Gestures",
            help="How far middle, ring and pinky must curl before grip starts to close."),
    Setting("gestures.grip_curl_full", "Grip full at", "float", "Gestures",
            help="Curl of middle, ring and pinky that counts as a full grip."),
    Setting("gestures.trigger_curl_start", "Trigger starts at", "float", "Gestures",
            help="How far the index must curl before the trigger starts to move."),
    Setting("gestures.trigger_curl_full", "Trigger full at", "float", "Gestures",
            help="Index curl that counts as a fully pressed trigger."),
    Setting("gestures.trigger_from_pinch", "Pinch pulls the trigger", "bool", "Gestures",
            help="Touching thumb and index tips also presses the trigger, like grabbing small things."),
    Setting("gestures.pinch_open", "Pinch starts at", "float", "Gestures", minimum=0.1, maximum=1.5,
            unit=" palms", help="Thumb-to-index-tip distance, in palm lengths, where a pinch begins."),
    Setting("gestures.pinch_closed", "Full pinch at", "float", "Gestures", minimum=0.0, maximum=1.0,
            unit=" palms", help="Thumb-to-index-tip distance, in palm lengths, of a full pinch."),
    Setting("gestures.pinch_palm_gate", "Fist, not pinch, below", "float", "Gestures", minimum=0.2,
            maximum=1.2, unit=" palms", advanced=True,
            help="In a fist the thumb also touches the index. When the index tip is closer than this to the "
                 "palm it is a fist, and the pinch does not count."),
    Setting("gestures.controls_min_cutoff", "Trigger/grip steadiness", "float", "Gestures", minimum=0.1,
            maximum=10.0, step=0.1, advanced=True, help="Lower is steadier when still, but laggier."),
    Setting("gestures.controls_beta", "Trigger/grip responsiveness", "float", "Gestures", minimum=0.0,
            maximum=20.0, step=0.5, advanced=True, help="Higher follows quick squeezes with less lag."),
    Setting("gestures.gesture_enter", "Gesture picked at", "float", "Gestures", advanced=True,
            help="Score a gesture needs to be picked (see the scores in the Hands tab)."),
    Setting("gestures.gesture_exit", "Gesture kept above", "float", "Gestures", advanced=True,
            help="The current gesture is kept until its score drops below this. Keep it under "
                 "\"Gesture picked at\", so gestures do not flicker."),
    Setting("gestures.gesture_confirm_frames", "Gesture confirm frames", "int", "Gestures", minimum=1,
            maximum=15, step=1, advanced=True, help="Frames a new gesture must lead before it is picked."),
    Setting("gestures.finger_closed_start", "Finger closed from", "float", "Gestures", advanced=True,
            help="Curl where a finger starts to count as closed for gestures. Lower it if a gesture needs "
                 "fingers closed and you cannot curl them far enough."),
    Setting("gestures.finger_closed_full", "Finger closed at", "float", "Gestures", advanced=True,
            help="Curl where a finger fully counts as closed for gestures."),
    Setting("gestures.thumb_closed_start", "Thumb tucked from", "float", "Gestures", advanced=True,
            help="Thumb curl where a closed hand starts to count as a fist rather than a thumbs up."),
    Setting("gestures.thumb_closed_full", "Thumb tucked at", "float", "Gestures", advanced=True),
    Setting("gestures.pinch_threshold", "Fallback pinch distance", "float", "Gestures", minimum=0.01,
            maximum=0.2, step=0.005, decimals=3, advanced=True,
            help="Old detector, only used when the model gives no 3D points."),
    Setting("gestures.finger_extended_threshold", "Fallback finger extended at", "float", "Gestures",
            minimum=0.2, maximum=1.5, step=0.05, advanced=True,
            help="Old detector, only used when the model gives no 3D points."),

    # Tracking model
    Setting("tracking.model_complexity", "Model", "choice", "Tracking model", apply=Apply.MODEL,
            choices=((0, "Fast"), (1, "Accurate"))),
    Setting("tracking.max_hands", "Hands", "int", "Tracking model", minimum=1, maximum=2, step=1,
            apply=Apply.MODEL, advanced=True),
    Setting("tracking.detection_confidence", "Detection confidence", "float", "Tracking model",
            minimum=0.1, maximum=1.0, step=0.05, apply=Apply.MODEL, advanced=True),
    Setting("tracking.tracking_confidence", "Tracking confidence", "float", "Tracking model",
            minimum=0.1, maximum=1.0, step=0.05, apply=Apply.MODEL, advanced=True),

    # Hand identity
    Setting("tracking.identity.continuity_radius", "Continuity radius", "float", "Hand identity",
            minimum=0.01, maximum=0.5, step=0.01, advanced=True),
    Setting("tracking.identity.memory_seconds", "Memory", "float", "Hand identity", minimum=0.0,
            maximum=2.0, step=0.05, unit=" s", advanced=True),
    Setting("tracking.identity.switch_frames", "Frames to switch", "int", "Hand identity", minimum=1,
            maximum=30, step=1, advanced=True),
    Setting("tracking.identity.duplicate_radius", "Duplicate radius", "float", "Hand identity",
            minimum=0.0, maximum=0.3, step=0.01, advanced=True),
    Setting("tracking.identity.order_weight", "Left/right order weight", "float", "Hand identity",
            minimum=0.0, maximum=5.0, step=0.1, advanced=True),

    # Driver connection
    Setting("network.host", "Host", "text", "Driver connection", apply=Apply.RECONNECT, advanced=True),
    Setting("network.port", "Port", "int", "Driver connection", minimum=1024, maximum=65535, step=1,
            apply=Apply.RECONNECT, advanced=True),

    # Performance
    Setting("process.priority", "Process priority", "choice", "Performance", apply=Apply.NEXT_START,
            advanced=True, choices=(("normal", "Normal"), ("above_normal", "Above normal"), ("high", "High"))),
    Setting("process.disable_power_throttling", "Disable power throttling", "bool", "Performance",
            apply=Apply.NEXT_START, advanced=True),
)
