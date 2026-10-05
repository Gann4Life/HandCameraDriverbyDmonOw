import pytest

from hand_controls import DEFAULTS, GRIP_RELEASE_SECONDS, ControlMapper
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


def grip_curls(value: float) -> tuple:
    """Curls that give this grip value with the default ramp."""
    start, full = DEFAULTS["grip_curl_start"], DEFAULTS["grip_curl_full"]
    return (0.0, 0.0) + (start + value * (full - start),) * 3


def test_grip_holds_its_peak_until_it_stays_below_the_release_value():
    mapper = ControlMapper({})
    on, off = DEFAULTS["latch_on"], DEFAULTS["latch_off"]
    between = (on + off) / 2
    steps = [
        (between, between),  # below latch_on: analog, nothing held
        (on + 0.1, on + 0.1),  # holds from here
        (between, on + 0.1),  # back between the thresholds: still the peak
        (0.0, on + 0.1),  # one misread frame: still held
        (0.0, 0.0),  # below latch_off for longer: lets go
    ]
    # Apart enough for no smoothing, except the last: the same value again, a release wait later
    times = [0.0, 1.0, 2.0, 3.0, 3.0 + GRIP_RELEASE_SECONDS]
    for i, ((value, expected), t) in enumerate(zip(steps, times)):
        _, grip = mapper("right", features(curl=grip_curls(value)), t * NO_SMOOTHING_STEP_S)
        assert grip == pytest.approx(expected), f"step {i}: grip {value:.2f}"


def test_a_short_dip_does_not_restart_the_release_wait():
    mapper = ControlMapper({})
    mapper("right", features(curl=grip_curls(1.0)), 0.0)
    mapper("right", features(curl=grip_curls(0.0)), 1.0)
    mapper("right", features(curl=grip_curls(1.0)), 2.0)
    # Below again: the wait starts over, so a release needs a fresh full wait
    _, grip = mapper("right", features(curl=grip_curls(0.0)), 3.0)
    assert grip == 1.0


def test_trigger_stays_analog():
    mapper = ControlMapper({"trigger_from_pinch": False})
    for i, value in enumerate((0.5, 0.9, 0.5)):
        curl = (0.0, index_curl_for_trigger(value), 0.0, 0.0, 0.0)
        trigger, _ = mapper("right", features(curl=curl), i * NO_SMOOTHING_STEP_S)
        assert trigger == pytest.approx(value)


def test_grip_hold_is_per_hand():
    mapper = ControlMapper({})
    between = (DEFAULTS["latch_on"] + DEFAULTS["latch_off"]) / 2
    mapper("left", features(curl=grip_curls(1.0)), 0.0)
    _, left = mapper("left", features(curl=grip_curls(between)), NO_SMOOTHING_STEP_S)
    _, right = mapper("right", features(curl=grip_curls(between)), NO_SMOOTHING_STEP_S)
    assert left == 1.0  # held from the earlier grab
    assert right == pytest.approx(between)  # never held


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


def test_reset_forgets_a_held_grip_for_that_hand_only():
    mapper = ControlMapper({})
    between = (DEFAULTS["latch_on"] + DEFAULTS["latch_off"]) / 2
    for hand in ("left", "right"):
        mapper(hand, features(curl=grip_curls(1.0)), 0.0)
    mapper.reset("left")
    # Between the thresholds a held grip stays held, a fresh one is analog
    assert mapper("left", features(curl=grip_curls(between)), NO_SMOOTHING_STEP_S)[1] == pytest.approx(between)
    assert mapper("right", features(curl=grip_curls(between)), NO_SMOOTHING_STEP_S)[1] == 1.0
