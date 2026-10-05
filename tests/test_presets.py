"""presets.migrate on configs written by older versions."""
import copy

import presets
from config_defaults import DEFAULT_CONFIG
from utils.config_utils import get_value, set_value

KEY = "tracking.identity.switch_frames"


def format_3_config(switch_frames: int) -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    set_value(config, KEY, switch_frames)
    config.update(preset="Mine", preset_format=3,
                  presets={"Mine": {KEY: switch_frames}, "POV": {KEY: switch_frames}})
    return config


def test_format_4_moves_the_old_switch_frames_everywhere_even_in_user_presets():
    config = format_3_config(6)
    presets.migrate(config)
    new = get_value(DEFAULT_CONFIG, KEY)
    assert new != 6
    assert get_value(config, KEY) == new
    assert {name: saved[KEY] for name, saved in config["presets"].items()} == {"Mine": new, "POV": new}
    assert config["preset_format"] == presets.FORMAT


def test_format_4_keeps_a_value_the_user_chose():
    config = format_3_config(9)
    presets.migrate(config)
    assert get_value(config, KEY) == 9
    assert config["presets"]["Mine"][KEY] == 9

def test_a_config_from_before_presets_also_moves_the_old_switch_frames():
    config = copy.deepcopy(DEFAULT_CONFIG)
    set_value(config, KEY, 6)
    config.pop("preset", None)
    config.pop("presets", None)
    presets.migrate(config)
    assert get_value(config, KEY) == get_value(DEFAULT_CONFIG, KEY)
