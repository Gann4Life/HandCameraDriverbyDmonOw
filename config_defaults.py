"""
Factory defaults for every config value. load_config fills in whatever a
config file lacks from here, so older files keep working, and presets reset
to these values.
"""
import copy
from typing import Any, Dict

from gesture_scores import DEFAULTS as GESTURE_DEFAULTS
from hand_controls import DEFAULTS as CONTROL_DEFAULTS

# How a Touch controller sits in the hand: rolled from the flat-hand frame
# toward the thumb, [pitch, yaw, roll] degrees. Each preset can tune its own.
DEFAULT_ROTATION_OFFSET_DEG = {"left": [0.0, 0.0, -90.0], "right": [0.0, 0.0, 90.0]}

DEFAULT_CONFIG: Dict[str, Any] = {
    "camera": {
        "device_id": 0, "width": 640, "height": 480, "fps": 30,
        "flip_horizontal": False, "source_mirrored": False, "backend": "auto", "rotate_180": False,
        "hfov_deg": 70.0, "facing_distance": 0.8, "stall_timeout": 2.0, "reconnect_timeout": 30.0,
    },
    "tracking": {
        "max_hands": 2, "detection_confidence": 0.7, "tracking_confidence": 0.5, "model_complexity": 1,
        "view_mode": "facing", "hands_follow": "head", "palm_facing": "mode", "swap_hands": False, "depth_source": "mediapipe",
        "depth_assist": {"max_rate_hz": 10.0, "max_age": 0.5, "match_radius": 0.12, "scale": 1.0},
        "identity": {"continuity_radius": 0.15, "memory_seconds": 0.4, "switch_frames": 6,
                     "duplicate_radius": 0.05, "order_weight": 1.5},
    },
    "process": {"priority": "above_normal", "disable_power_throttling": True},
    "network": {"host": "127.0.0.1", "port": 65432, "controller_type": "touch",
                # Tuned live in SteamVR Home against the Touch hands
                "index_offset": [0.0, 0.0, -0.1], "index_rotation_deg": [45.0, 0.0, 0.0]},
    "gestures": {"pinch_threshold": 0.05, "finger_extended_threshold": 0.6,
                 **CONTROL_DEFAULTS, **GESTURE_DEFAULTS},
    "calibration": {
        "scale": 1.0,
        "hand_scale": 1.0,
        "steady_hand_size": False,
        "rebuild_hand": False,
        "hand_size_window": 90,
        "position_offset": [0.0, 0.0, 0.0],
        "camera_rotation_deg": [0.0, 0.0, 0.0],
        "rotation_offset_deg": copy.deepcopy(DEFAULT_ROTATION_OFFSET_DEG),
        "filter": {
            "mode": "one_euro",
            "position": {"min_cutoff": 1.0, "beta": 1.5},
            "depth": {"min_cutoff": 0.3, "beta": 2.0},
            "rotation": {"min_cutoff": 1.0, "beta": 0.5},
            "d_cutoff": 1.0,
            "ema": {"position": 0.7, "depth": 0.35, "rotation": 0.5},
        },
    },
    "debug": {"show_video": True, "show_landmarks": True, "show_fps": True, "log_gestures": False},
    # The app checks the SteamVR driver when it starts and offers to install or update it
    "addons": {"check_at_startup": True},
}


def fill_defaults(config: Dict[str, Any], defaults: Dict[str, Any] = DEFAULT_CONFIG) -> Dict[str, Any]:
    """Add every value of defaults that config lacks, recursively; existing values stay."""
    for key, value in defaults.items():
        if key not in config:
            config[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(config[key], dict):
            fill_defaults(config[key], value)
    return config
