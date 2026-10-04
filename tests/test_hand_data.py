"""HandData.to_protocol_string against the hand message in docs/PROTOCOL.md."""
import re

import pytest

from hand_data import HandData

GESTURES = {"OPEN", "FIST", "POINT", "PINCH", "THUMBS_UP", "PEACE", "UNKNOWN"}
FOUR_DECIMALS = re.compile(r"^-?\d+\.\d{4}$")
TWO_DECIMALS = re.compile(r"^\d\.\d{2}$")


def hand(hand_type="left", curls=()) -> HandData:
    return HandData(hand_type=hand_type, position=(0.12, -0.3, -0.4), rotation=(1.0, 0.0, 0.0, 0.0),
                    gesture="POINT", trigger_value=0.8, grip_value=0.0, landmarks=[], finger_curls=curls)


def fields(line: str) -> dict:
    pairs = [pair.split(":", 1) for pair in line.split(",")]
    assert all(len(pair) == 2 for pair in pairs), line
    keys = [key for key, _ in pairs]
    assert len(keys) == len(set(keys)), f"duplicate key in {line}"
    return dict(pairs)


def test_matches_the_example_in_protocol_md():
    line = hand(curls=(0.1, 0.2, 0.3, 0.4, 0.5)).to_protocol_string("index", room_anchor=True)
    assert line == ("HAND:LEFT,X:0.1200,Y:-0.3000,Z:-0.4000,QW:1.0000,QX:0.0000,QY:0.0000,QZ:0.0000,"
                    "TRIGGER:0.80,GRIP:0.00,GESTURE:POINT,TYPE:INDEX,CURL:0.10;0.20;0.30;0.40;0.50,ANCHOR:ROOM")


@pytest.mark.parametrize("hand_type", ["left", "right"])
def test_always_sent_fields(hand_type):
    sent = fields(hand(hand_type).to_protocol_string())
    assert sent["HAND"] == hand_type.upper()
    for key in ("X", "Y", "Z", "QW", "QX", "QY", "QZ"):
        assert FOUR_DECIMALS.match(sent[key]), (key, sent[key])
    for key in ("TRIGGER", "GRIP"):
        assert TWO_DECIMALS.match(sent[key]) and 0.0 <= float(sent[key]) <= 1.0, (key, sent[key])
    assert sent["GESTURE"] in GESTURES


@pytest.mark.parametrize("controller, expected", [("touch", "TOUCH"), ("index", "INDEX")])
def test_type_is_the_controller_profile(controller, expected):
    assert fields(hand().to_protocol_string(controller))["TYPE"] == expected


def test_curl_is_five_values_thumb_to_pinky_only_with_features():
    assert "CURL" not in fields(hand().to_protocol_string())
    curl = fields(hand(curls=(0.0, 0.25, 0.5, 0.75, 1.0)).to_protocol_string())["CURL"]
    assert curl.split(";") == ["0.00", "0.25", "0.50", "0.75", "1.00"]


def test_anchor_room_only_with_a_fixed_camera():
    assert "ANCHOR" not in fields(hand().to_protocol_string())
    assert fields(hand().to_protocol_string(room_anchor=True))["ANCHOR"] == "ROOM"


def test_one_line_without_the_terminator():
    # The socket client adds the "\n" that ends each message
    assert "\n" not in hand(curls=(0.1,) * 5).to_protocol_string("index", room_anchor=True)
