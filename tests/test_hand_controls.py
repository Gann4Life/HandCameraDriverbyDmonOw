import pytest

from hand_controls import DEFAULTS, ControlMapper
from hands import make_features as features

# More than OneEuroFilter's reset_after: every sample starts the filter over,
# so the tests see raw values without smoothing
NO_SMOOTHING_STEP_S = 1.0


def index_curl_for_trigger(value: float) -> float:
    """Index curl that gives this trigger value with the default ramp."""
    start, full = DEFAULTS["trigger_curl_start"], DEFAULTS["trigger_curl_full"]
    return start + value * (full - start)


def test_open_hand_presses_nothing():
    assert ControlMapper({}).raw(features()) == (0.0, 0.0)


def test_fist_presses_trigger_and_grip():
    assert ControlMapper({}).raw(features(curl=(1.0,) * 5)) == (1.0, 1.0)


def test_grip_is_the_mean_of_middle_ring_and_pinky():
    start, full = DEFAULTS["grip_curl_start"], DEFAULTS["grip_curl_full"]
    _, grip = ControlMapper({}).raw(features(curl=(0.0, 0.0, 1.0, full, start)))
    mean = (1.0 + full + start) / 3.0
    assert grip == pytest.approx((mean - start) / (full - start))


def test_pinch_presses_the_trigger():
    trigger, grip = ControlMapper({}).raw(features(pinch_distance=DEFAULTS["pinch_closed"]))
    assert (trigger, grip) == (1.0, 0.0)


def test_pinch_can_be_turned_off():
    mapper = ControlMapper({"trigger_from_pinch": False})
    assert mapper.raw(features(pinch_distance=DEFAULTS["pinch_closed"])) == (0.0, 0.0)


def test_pinch_fades_out_when_the_index_tip_is_in_the_palm():
    mapper = ControlMapper({})
    in_palm = features(pinch_distance=DEFAULTS["pinch_closed"], index_tip_to_palm=0.1)
    assert mapper.pinch_strength(in_palm) == 0.0


def test_latched_trigger_needs_a_clear_gap_to_press_and_release():
    mapper = ControlMapper({"trigger_latch": True, "trigger_from_pinch": False})
    on, off = DEFAULTS["latch_on"], DEFAULTS["latch_off"]
    steps = [
        ((on + off) / 2, 0.0),  # between the thresholds: not pressed yet
        (on + 0.05, 1.0),        # above latch_on: pressed
        ((on + off) / 2, 1.0),  # back between them: still pressed
        (off - 0.05, 0.0),       # below latch_off: released
    ]
    for i, (value, expected) in enumerate(steps):
        curl = (0.0, index_curl_for_trigger(value), 0.0, 0.0, 0.0)
        trigger, _ = mapper("right", features(curl=curl), i * NO_SMOOTHING_STEP_S)
        assert trigger == expected, f"step {i}: trigger {value:.2f}"


def test_latch_state_is_per_hand():
    mapper = ControlMapper({"grip_latch": True})
    between = (DEFAULTS["latch_on"] + DEFAULTS["latch_off"]) / 2
    start, full = DEFAULTS["grip_curl_start"], DEFAULTS["grip_curl_full"]
    between_curl = start + between * (full - start)
    _, left = mapper("left", features(curl=(0.0, 0.0, 1.0, 1.0, 1.0)), 0.0)
    _, left = mapper("left", features(curl=(0.0, 0.0) + (between_curl,) * 3), NO_SMOOTHING_STEP_S)
    _, right = mapper("right", features(curl=(0.0, 0.0) + (between_curl,) * 3), NO_SMOOTHING_STEP_S)
    assert left == 1.0   # held from the earlier press
    assert right == 0.0  # never pressed


def test_smoothing_is_per_hand():
    mapper = ControlMapper({})
    mapper("left", features(curl=(1.0,) * 5), 0.0)
    # The right hand's first sample isn't pulled towards the left hand's fist
    assert mapper("right", features(), 0.01) == (0.0, 0.0)


def test_finger_curls_are_clamped_and_per_hand():
    mapper = ControlMapper({})
    assert mapper.finger_curls("left", features(curl=(1.2, 0.5, -0.1, 0.0, 1.0)), 0.0) == (1.0, 0.5, 0.0, 0.0, 1.0)
    assert mapper.finger_curls("right", features(), 0.01) == (0.0,) * 5


def test_finger_curls_are_smoothed_apart_from_trigger_and_grip():
    mapper = ControlMapper({})
    mapper("left", features(curl=(1.0,) * 5), 0.0)
    # The same hand's first curls sample isn't blended with its trigger/grip history
    assert mapper.finger_curls("left", features(), 0.01) == (0.0,) * 5
