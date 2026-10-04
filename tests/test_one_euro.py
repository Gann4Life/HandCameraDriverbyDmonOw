import math

import pytest

from utils.one_euro import OneEuroFilter, QuaternionOneEuroFilter

DT = 1.0 / 30.0


def test_first_sample_passes_through():
    assert OneEuroFilter()((1.0, 2.0, 3.0), 0.0) == (1.0, 2.0, 3.0)


def test_still_input_stays_still():
    smooth = OneEuroFilter()
    for i in range(10):
        out = smooth((0.5, -0.5), i * DT)
    assert out == pytest.approx((0.5, -0.5))


def test_a_step_is_smoothed_towards_the_new_value():
    smooth = OneEuroFilter(min_cutoff=1.0)
    smooth((0.0,), 0.0)
    (out,) = smooth((1.0,), DT)
    assert 0.0 < out < 1.0


def test_higher_beta_lags_less_on_fast_moves():
    slow, fast = OneEuroFilter(min_cutoff=1.0, beta=0.0), OneEuroFilter(min_cutoff=1.0, beta=10.0)
    for i in range(5):
        (slow_out,) = slow((i * 0.1,), i * DT)
        (fast_out,) = fast((i * 0.1,), i * DT)
    assert fast_out > slow_out


def test_a_long_gap_starts_over():
    smooth = OneEuroFilter(reset_after=0.5)
    smooth((0.0,), 0.0)
    assert smooth((1.0,), 0.6) == (1.0,)


def test_a_timestamp_that_goes_back_starts_over():
    smooth = OneEuroFilter()
    smooth((0.0,), 1.0)
    assert smooth((1.0,), 0.9) == (1.0,)


def test_reset_forgets_history():
    smooth = OneEuroFilter()
    smooth((0.0,), 0.0)
    smooth.reset()
    assert smooth((1.0,), DT) == (1.0,)


def test_quaternion_sign_flip_is_the_same_rotation():
    smooth = QuaternionOneEuroFilter()
    q = (math.cos(0.2), math.sin(0.2), 0.0, 0.0)
    smooth(q, 0.0)
    # -q is the same rotation: the output stays at q instead of averaging towards zero
    out = smooth(tuple(-c for c in q), DT)
    assert out == pytest.approx(q)


def test_quaternion_output_is_unit_length():
    smooth = QuaternionOneEuroFilter()
    smooth((1.0, 0.0, 0.0, 0.0), 0.0)
    out = smooth((math.cos(0.5), 0.0, math.sin(0.5), 0.0), DT)
    assert math.sqrt(sum(c * c for c in out)) == pytest.approx(1.0)
