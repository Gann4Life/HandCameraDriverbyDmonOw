"""
Access to nested config values by dotted key, e.g. "calibration.filter.depth.beta".
"""
from typing import Any


def get_value(config: dict, key: str, default: Any = None) -> Any:
    """Value at a dotted key, or default if any part of the path is missing."""
    node = config
    for part in key.split('.'):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def set_value(config: dict, key: str, value: Any) -> None:
    """Set the value at a dotted key, creating missing sections."""
    *path, leaf = key.split('.')
    node = config
    for part in path:
        node = node.setdefault(part, {})
    node[leaf] = value


def remove_value(config: dict, key: str) -> None:
    """Remove the value at a dotted key, if it is there."""
    *path, leaf = key.split('.')
    node = get_value(config, '.'.join(path)) if path else config
    if isinstance(node, dict):
        node.pop(leaf, None)
