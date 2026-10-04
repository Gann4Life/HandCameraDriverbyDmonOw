import math

import numpy as np
import pytest

from hand_features import FULL_CURL_DEG, OPEN_CURL_DEG, compute_features, ramp
from hands import make_hand

PALM_M = 0.095  # wrist to middle knuckle in make_hand


@pytest.mark.parametrize("value, expected", [(0.0, 0.0), (5.0, 0.5), (10.0, 1.0), (-3.0, 0.0), (42.0, 1.0)])
def test_ramp_rises_and_clamps(value, expected):
    assert ramp(value, 0.0, 10.0) == pytest.approx(expected)


def test_ramp_works_downwards():
    assert ramp(0.55, 0.55, 0.20) == 0.0
    assert ramp(0.20, 0.55, 0.20) == 1.0
    assert ramp(0.10, 0.55, 0.20) == 1.0


def test_open_hand_has_no_curl():
    features = compute_features(make_hand())
    assert features.curl == pytest.approx((0.0,) * 5)
    # arccos near 1 loses precision: straight bones measure a few millionths of a degree
    assert features.curl_deg == pytest.approx((0.0,) * 5, abs=1e-3)
    assert features.palm_size_m == pytest.approx(PALM_M)


def test_finger_curl_is_its_summed_bend_over_the_default_range():
    # Index bends 30 degrees at each of 3 joints: 90 degrees in all
    features = compute_features(make_hand({"index": 30.0}))
    assert features.curl_deg[1] == pytest.approx(90.0)
    expected = (90.0 - OPEN_CURL_DEG["index"]) / (FULL_CURL_DEG["index"] - OPEN_CURL_DEG["index"])
    assert features.curl[1] == pytest.approx(expected)
    assert features.curl[2:] == pytest.approx((0.0,) * 3)


def test_fully_bent_finger_curls_to_one():
    features = compute_features(make_hand({"middle": 75.0}))
    assert features.curl[2] == 1.0


def test_thumb_skips_its_cmc_joint():
    # Bent 30 degrees at CMC, MCP and IP: only MCP and IP count
    features = compute_features(make_hand({"thumb": 30.0}))
    assert features.curl_deg[0] == pytest.approx(60.0)


def test_calibrated_range_replaces_the_default():
    open_deg = [0.0, 0.0, 0.0, 0.0, 0.0]
    full_deg = [180.0, 180.0, 180.0, 180.0, 180.0]
    features = compute_features(make_hand({"ring": 30.0}), open_deg=open_deg, full_deg=full_deg)
    assert features.curl[3] == pytest.approx(0.5)


def test_features_do_not_depend_on_hand_size():
    hand = make_hand({"index": 30.0, "thumb": 20.0})
    small, large = compute_features(hand), compute_features((np.array(hand) * 1.4).tolist())
    assert large.curl == pytest.approx(small.curl)
    assert large.pinch_distance == pytest.approx(small.pinch_distance)
    assert large.index_tip_to_palm == pytest.approx(small.index_tip_to_palm)
    assert large.palm_size_m == pytest.approx(small.palm_size_m * 1.4)


def test_thumb_on_index_tip_is_a_full_pinch():
    hand = make_hand()
    hand[4] = list(hand[8])
    features = compute_features(hand)
    assert features.pinch_distance[0] == pytest.approx(0.0)
    assert features.pinch[0] == 1.0


def test_open_hand_has_no_pinch():
    features = compute_features(make_hand())
    assert features.pinch == pytest.approx((0.0,) * 4)


def test_splay_is_the_angle_between_flat_fingers():
    hand = make_hand()
    features = compute_features(hand)
    index_dir, middle_dir = np.array(hand[5]), np.array(hand[9])
    expected = math.degrees(math.acos(np.dot(index_dir, middle_dir)
                                      / (np.linalg.norm(index_dir) * np.linalg.norm(middle_dir))))
    assert features.splay_deg[1] == pytest.approx(expected)


def test_splay_is_unknown_for_a_finger_bent_out_of_the_palm():
    features = compute_features(make_hand({"index": 90.0}))
    # Thumb-index and index-middle involve the bent index; middle-ring doesn't
    assert math.isnan(features.splay_deg[0])
    assert math.isnan(features.splay_deg[1])
    assert not math.isnan(features.splay_deg[2])
