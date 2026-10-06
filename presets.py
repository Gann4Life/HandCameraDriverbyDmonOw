"""
Presets: named sets of the settings that depend on where the camera is
(view, mirroring, placement, smoothing, gestures, hand identity), so
switching camera positions does not mean retuning everything.

The config's normal sections always hold the values in use. "preset" names
the active preset and "presets" stores the user's presets, plus any built-in
one the user changed, as {dotted key: value}. Settings that belong to the
hardware or the whole app (camera device, tracking model, depth source,
driver connection...) are shared by every preset.
"""
import copy
from typing import Any, Dict, List, Optional

from builtin_presets import BUILTIN_PRESETS
from config_defaults import DEFAULT_CONFIG
from utils.config_utils import get_value, remove_value, set_value

# The built-in presets (BUILTIN_PRESETS) are generated from the maintainer's
# saved presets by tools/publish_presets.py.
# The built-in preset for each view mode, for older configs and --mode
MODE_PRESETS = {"pov": "POV", "facing": "Facing"}
DEFAULT_PRESET = "Facing"
# Built-in values of older releases. One still in a built-in preset, or in use
# with one, was never tuned, so it moves to the current value.
RETIRED_DEFAULTS = {
    "calibration.rotation_offset_deg.left": [0.0, 0.0, -127.0],
    "calibration.rotation_offset_deg.right": [0.0, 0.0, 127.0],
}
# Old built-in values that move to the current default in every preset, the
# user's too: presets store every setting, so these were saved, never chosen.
# Keyed by the format that retires them.
RETIRED_EVERYWHERE = {
    # switch_frames 6 is too short for a depth misread in POV, which lasts a few frames
    4: {"tracking.identity.switch_frames": 6},
    # The old One Euro values lagged a punch's depth by centimetres more (see config_defaults)
    6: {"calibration.filter.depth.beta": 2.0, "calibration.filter.d_cutoff": 1.0},
}
# Settings that no longer exist; format 5 removes them from the config and every
# preset. Steady grip/trigger snapped to 0/1; grip now holds by itself.
REMOVED_KEYS = ("gestures.grip_latch", "gestures.trigger_latch")

# Keys (and key prefixes, ending in ".") every preset shares
SHARED_KEYS = (
    "camera.device_id", "camera.width", "camera.height", "camera.fps", "camera.backend",
    "camera.hfov_deg", "camera.stall_timeout", "camera.reconnect_timeout",
    "tracking.max_hands", "tracking.detection_confidence", "tracking.tracking_confidence",
    "tracking.model_complexity", "tracking.depth_source", "tracking.depth_assist.",
    "network.", "process.", "debug.", "addons.", "recording.",
)
# Settings that used to be shared: presets saved before take the value in use
FORMERLY_SHARED = ("calibration.rotation_offset_deg.left", "calibration.rotation_offset_deg.right")
FORMAT = 6
# Top-level entries that are not settings
META_KEYS = ("preset", "presets", "preset_format")


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


def without_unsaved(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    A copy of config with the active preset's settings as last saved: what to
    write to the file when only the shared settings are being saved.
    """
    result = copy.deepcopy(config)
    for key, value in values(config, active(config)).items():
        set_value(result, key, value)
    return result


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


_MISSING = object()


def fill_new_settings(config: Dict[str, Any]) -> None:
    """
    A setting added after the file was written starts at the active preset's
    value, so a built-in preset that turns it on does not open with unsaved
    changes. Run before the factory defaults fill in the rest.
    """
    if "preset" not in config:
        return  # migrate() builds the presets of an older file
    for key, value in values(config, active(config)).items():
        if get_value(config, key, _MISSING) is _MISSING:
            set_value(config, key, copy.deepcopy(value))


def migrate(config: Dict[str, Any]) -> None:
    """
    Older configs kept placement per view mode (calibration.pov,
    calibration.facing) and everything else shared. Each mode becomes its
    built-in preset: its own defaults, plus the shared values the user changed
    from the factory defaults, plus that mode's placement.
    """
    if "preset" in config:
        version = config.get("preset_format", 1)
        if version < 2:
            _adopt_formerly_shared(config)
        if version < 3:
            _forget_retired_defaults(config)
        if version < 4:
            _retire_everywhere(config, RETIRED_EVERYWHERE[4])
        if version < 5:
            _remove_keys(config)
        if version < 6:
            _retire_everywhere(config, RETIRED_EVERYWHERE[6])
            _drop_unchanged_builtins(config)
        config["preset_format"] = FORMAT
        return
    config["preset_format"] = FORMAT
    for retired in RETIRED_EVERYWHERE.values():
        _retire_everywhere(config, retired)
    _remove_keys(config)
    for key, old in RETIRED_DEFAULTS.items():
        if get_value(config, key) == old:
            set_value(config, key, get_value(DEFAULT_CONFIG, key))
    calibration = config.setdefault("calibration", {})
    sections = {mode: calibration.pop(mode, None) for mode in MODE_PRESETS}
    current_mode = str(get_value(config, "tracking.view_mode") or "facing").lower()
    if current_mode not in MODE_PRESETS:
        current_mode = "facing"
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


def _adopt_formerly_shared(config: Dict[str, Any]) -> None:
    """Every preset keeps the value it was using while the setting was shared."""
    current = snapshot(config)
    for name in names(config):
        preset = values(config, name)
        saved = stored(config).get(name, {})
        for key in FORMERLY_SHARED:
            if key not in saved:
                preset[key] = copy.deepcopy(current[key])
        if is_builtin(name) and preset == default_values(name):
            stored(config).pop(name, None)
        else:
            stored(config)[name] = preset


def _retire_everywhere(config: Dict[str, Any], retired: Dict[str, Any]) -> None:
    """Retired values (one format's RETIRED_EVERYWHERE) move to the current default, in use and in every saved preset."""
    for key, old in retired.items():
        new = get_value(DEFAULT_CONFIG, key)
        if get_value(config, key) == old:
            set_value(config, key, new)
        for saved in stored(config).values():
            if saved.get(key) == old:
                saved[key] = new


def _remove_keys(config: Dict[str, Any]) -> None:
    """
    Format 5: REMOVED_KEYS go from the config in use and from every saved
    preset; a built-in preset left as shipped is then no longer stored.
    """
    for key in REMOVED_KEYS:
        remove_value(config, key)
        for saved in stored(config).values():
            saved.pop(key, None)
    _drop_unchanged_builtins(config)


def _drop_unchanged_builtins(config: Dict[str, Any]) -> None:
    """Stop storing built-in presets that match their defaults, so they follow future defaults (see store)."""
    for builtin in BUILTIN_PRESETS:
        if stored(config).get(builtin) == default_values(builtin):
            stored(config).pop(builtin)


def _forget_retired_defaults(config: Dict[str, Any]) -> None:
    """Built-in presets drop the old defaults they were saved with; user presets keep theirs."""
    for builtin in BUILTIN_PRESETS:
        saved = stored(config).get(builtin)
        if saved is None:
            continue
        for key, old in RETIRED_DEFAULTS.items():
            if saved.get(key) == old:
                saved[key] = default_values(builtin)[key]
    _drop_unchanged_builtins(config)
    name = active(config)
    if is_builtin(name):
        for key, old in RETIRED_DEFAULTS.items():
            if get_value(config, key) == old:
                set_value(config, key, values(config, name)[key])
