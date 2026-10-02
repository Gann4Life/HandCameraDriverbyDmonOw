"""
Presets: named sets of the settings that depend on where the camera is
(view, mirroring, placement, smoothing, gestures, hand identity), so
switching camera positions does not mean retuning everything.

The config's normal sections always hold the values in use. "preset" names
the active preset and "presets" stores the user's presets, plus any built-in
one the user changed, as {dotted key: value}. Settings that belong to the
hardware or the whole app (camera device, tracking model, depth source,
controller rotation, driver connection...) are shared by every preset.
"""
import copy
from typing import Any, Dict, List, Optional

from config_defaults import DEFAULT_CONFIG
from utils.config_utils import get_value, set_value

# Built-in presets, as changes to the factory defaults
BUILTIN_PRESETS: Dict[str, Dict[str, Any]] = {
    "POV": {"tracking.view_mode": "pov"},
    # Tuned live with a webcam facing the user, which shows a mirrored image
    "Facing": {
        "tracking.view_mode": "facing",
        "camera.source_mirrored": True,
        "calibration.filter.position.beta": 1.41,
        "calibration.position_offset": [0.0, 0.0, -0.2],
    },
}
# The built-in preset for each view mode, for older configs and --mode
MODE_PRESETS = {"pov": "POV", "facing": "Facing"}
DEFAULT_PRESET = "POV"

# Keys (and key prefixes, ending in ".") every preset shares
SHARED_KEYS = (
    "camera.device_id", "camera.width", "camera.height", "camera.fps", "camera.backend",
    "camera.hfov_deg", "camera.stall_timeout", "camera.reconnect_timeout",
    "tracking.max_hands", "tracking.detection_confidence", "tracking.tracking_confidence",
    "tracking.model_complexity", "tracking.depth_source", "tracking.depth_assist.",
    "calibration.rotation_offset_deg.", "network.", "process.", "debug.",
)
# Top-level entries that are not settings
META_KEYS = ("preset", "presets")


def is_preset_key(key: str) -> bool:
    """Whether a dotted config key is stored per preset."""
    if key.split(".", 1)[0] in META_KEYS:
        return False
    return not any(key == shared or (shared.endswith(".") and key.startswith(shared)) for shared in SHARED_KEYS)


def _flatten(node: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat = {}
    for key, value in node.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, dotted + "."))
        else:
            flat[dotted] = copy.deepcopy(value)
    return flat


def snapshot(config: Dict[str, Any]) -> Dict[str, Any]:
    """The preset settings currently in use, as {dotted key: value}."""
    return {key: value for key, value in _flatten(config).items() if is_preset_key(key)}


def stored(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return config.setdefault("presets", {})


def is_builtin(name: str) -> bool:
    return name in BUILTIN_PRESETS


def names(config: Dict[str, Any]) -> List[str]:
    """Built-in presets first, then the user's in alphabetical order."""
    user = sorted((n for n in stored(config) if not is_builtin(n)), key=str.lower)
    return list(BUILTIN_PRESETS) + user


def active(config: Dict[str, Any]) -> str:
    name = config.get("preset")
    return name if name in names(config) else DEFAULT_PRESET


def default_values(name: str) -> Dict[str, Any]:
    """A built-in preset as shipped (factory defaults for a user preset)."""
    values = snapshot(DEFAULT_CONFIG)
    values.update(copy.deepcopy(BUILTIN_PRESETS.get(name, {})))
    return values


def values(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    """
    A preset's settings. Settings added after it was saved come from the
    factory defaults (and the built-in preset's own values).
    """
    result = default_values(name)
    result.update(copy.deepcopy(stored(config).get(name, {})))
    return result


def is_modified(config: Dict[str, Any]) -> bool:
    """Whether the settings in use differ from the active preset."""
    return snapshot(config) != values(config, active(config))


def has_changes_from_default(config: Dict[str, Any], name: str) -> bool:
    return is_builtin(name) and name in stored(config)


def apply(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    """
    Make name the active preset and put its values in use.

    Returns:
        The changed dotted keys and their new values, to apply live
    """
    changes = {key: value for key, value in values(config, name).items() if get_value(config, key) != value}
    for key, value in changes.items():
        set_value(config, key, copy.deepcopy(value))
    config["preset"] = name
    return changes


def store(config: Dict[str, Any], name: Optional[str] = None) -> None:
    """
    Save the settings in use into a preset (the active one by default). A
    built-in preset left as shipped is not stored, so it keeps following
    future defaults.
    """
    name = name or active(config)
    current = snapshot(config)
    if is_builtin(name) and current == default_values(name):
        stored(config).pop(name, None)
    else:
        stored(config)[name] = current


def create(config: Dict[str, Any], name: str) -> None:
    """New preset from the settings in use, made active."""
    store(config, name)
    config["preset"] = name


def rename(config: Dict[str, Any], old: str, new: str) -> None:
    presets = stored(config)
    presets[new] = presets.pop(old, snapshot(config))
    if config.get("preset") == old:
        config["preset"] = new


def delete(config: Dict[str, Any], name: str) -> None:
    """Remove a user preset. Built-in presets can only be restored to default."""
    if is_builtin(name):
        raise ValueError(f"{name} is built in")
    stored(config).pop(name, None)


def restore_default(config: Dict[str, Any], name: str) -> None:
    stored(config).pop(name, None)


def validate_name(config: Dict[str, Any], name: str, current: Optional[str] = None) -> Optional[str]:
    """Why a preset name cannot be used, or None if it can."""
    name = name.strip()
    if not name:
        return "The name is empty."
    taken = {n.lower() for n in names(config) if n != current}
    if name.lower() in taken:
        return f"There is already a preset called {name}."
    return None


def migrate(config: Dict[str, Any]) -> None:
    """
    Older configs kept placement per view mode (calibration.pov,
    calibration.facing) and everything else shared. Each mode becomes its
    built-in preset: its own defaults, plus the shared values the user changed
    from the factory defaults, plus that mode's placement.
    """
    if "preset" in config:
        return
    calibration = config.setdefault("calibration", {})
    sections = {mode: calibration.pop(mode, None) for mode in MODE_PRESETS}
    current_mode = str(get_value(config, "tracking.view_mode") or "pov").lower()
    if current_mode not in MODE_PRESETS:
        current_mode = "pov"
    factory = snapshot(DEFAULT_CONFIG)
    changed = {key: value for key, value in snapshot(config).items() if factory.get(key) != value}
    changed.pop("tracking.view_mode", None)
    for mode, name in MODE_PRESETS.items():
        preset = copy.deepcopy(config)
        for key, value in {**default_values(name), **changed}.items():
            set_value(preset, key, copy.deepcopy(value))
        for key, value in (sections[mode] or {}).items():
            if factory.get(f"calibration.{key}") != value:
                set_value(preset, f"calibration.{key}", copy.deepcopy(value))
        store(preset, name)
        if name in stored(preset):
            stored(config)[name] = stored(preset)[name]
    apply(config, MODE_PRESETS[current_mode])
