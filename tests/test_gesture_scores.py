from gesture_scores import DEFAULTS, GESTURES, UNKNOWN, GestureClassifier, score_gestures
from hand_features import HandFeatures


def features(curl) -> HandFeatures:
    return HandFeatures(curl=tuple(curl), curl_deg=(0.0,) * 5, splay_deg=(0.0,) * 4, pinch=(0.0,) * 4,
                        pinch_distance=(1.0,) * 4, index_tip_to_palm=1.0, palm_size_m=0.09)


def scores(**named) -> dict:
    """Every gesture at 0, except the ones named."""
    return {name: named.get(name, 0.0) for name in GESTURES}


def best(curl, pinch=0.0) -> str:
    result = score_gestures(features(curl), pinch, DEFAULTS)
    return max(result, key=result.get)


def test_each_hand_shape_scores_its_gesture():
    assert best((0.0, 0.0, 0.0, 0.0, 0.0)) == "OPEN"
    assert best((1.0, 1.0, 1.0, 1.0, 1.0)) == "FIST"
    assert best((1.0, 0.0, 1.0, 1.0, 1.0)) == "POINT"
    assert best((0.0, 1.0, 1.0, 1.0, 1.0)) == "THUMBS_UP"
    assert best((1.0, 0.0, 0.0, 1.0, 1.0)) == "PEACE"
    assert best((0.0, 0.0, 0.0, 0.0, 0.0), pinch=1.0) == "PINCH"


def test_scores_are_between_zero_and_one():
    result = score_gestures(features((0.4, 0.3, 0.6, 0.2, 0.9)), 0.5, DEFAULTS)
    assert set(result) == set(GESTURES)
    assert all(0.0 <= value <= 1.0 for value in result.values())


def test_a_new_gesture_needs_confirmation_frames():
    classify = GestureClassifier({})
    frames = DEFAULTS["gesture_confirm_frames"]
    results = [classify("left", scores(FIST=1.0)) for _ in range(frames)]
    assert results == [UNKNOWN] * (frames - 1) + ["FIST"]


def test_current_gesture_holds_until_another_clearly_wins():
    classify = GestureClassifier({"gesture_confirm_frames": 1})
    assert classify("left", scores(FIST=1.0)) == "FIST"
    # OPEN leads, but FIST is above gesture_exit and OPEN doesn't beat it by the margin
    assert classify("left", scores(FIST=0.5, OPEN=0.6)) == "FIST"
    # FIST drops below gesture_exit
    assert classify("left", scores(FIST=0.3, OPEN=0.7)) == "OPEN"


def test_nothing_scoring_enough_is_unknown():
    classify = GestureClassifier({"gesture_confirm_frames": 1})
    assert classify("left", scores(OPEN=DEFAULTS["gesture_enter"] - 0.1)) == UNKNOWN


def test_hands_are_classified_separately():
    classify = GestureClassifier({})
    frames = DEFAULTS["gesture_confirm_frames"]
    for _ in range(frames):
        left = classify("left", scores(FIST=1.0))
        right = classify("right", scores(OPEN=1.0))
    assert (left, right) == ("FIST", "OPEN")
    # A frame of the other hand doesn't reset this hand's confirmation count
    classify = GestureClassifier({})
    for _ in range(frames - 1):
        classify("left", scores(POINT=1.0))
        classify("right", scores(PEACE=1.0))
    assert classify("left", scores(POINT=1.0)) == "POINT"
