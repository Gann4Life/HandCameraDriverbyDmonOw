import math

import numpy as np

from hand_features import compute_features
from hand_fit import PALM, HandFitter, HandShape

# One frame of a recorded POV clip (960 x 540, 70 degree field of view): an
# open right hand, palm toward the camera. Last frame's fit still had the
# index curled toward the camera; MediaPipe's bends say every finger is straight.
FOCAL = 480.0 / math.tan(math.radians(70.0) / 2.0)
CENTRE = (480.0, 270.0)
OBSERVED = [[830.4, 323.8], [871.4, 293.1], [891.5, 264.1], [898.0, 234.7], [896.4, 212.2], [815.2, 250.8],
            [809.5, 222.1], [805.2, 203.1], [801.6, 189.9], [787.9, 257.1], [770.2, 224.8], [761.3, 206.0],
            [755.5, 188.6], [766.9, 264.7], [744.8, 237.1], [735.6, 217.4], [730.3, 200.5], [748.9, 276.6],
            [722.6, 261.2], [706.5, 246.4], [697.7, 233.2]]
PREVIOUS = [2.14, -0.2, -0.166, 0.245, 0.039, 0.47, -0.473, -0.241, 0.308, 0.344, 0.145, 0.309, 0.866, 0.831,
            0.327, 0.078, 0.016, 0.048, 0.271, 0.094, 0.577, 0.475, 0.325, 0.35, 0.287, 0.282]
BENDS = [0.0, 0.0, 0.229, 0.366, 0.279, 0.0, 0.068, 0.34, 0.484, 0.0, 0.269, 0.359, 0.327, 0.0, 0.188, 0.126,
         0.162, 0.0, 0.257, 0.431]


def index_curl_after_warm_fit() -> float:
    fitter = HandFitter(HandShape.default(), FOCAL, CENTRE)
    fit = fitter.fit(np.array(OBSERVED), True, np.array(PREVIOUS), palm_away=True, bends=np.array(BENDS),
                     allow_refresh=False)
    return compute_features(fit.points - fit.points[PALM].mean(axis=0)).curl[1]


def test_a_warm_start_stuck_with_a_curled_finger_lets_go_when_mediapipe_sees_it_straight():
    assert index_curl_after_warm_fit() < 0.1


def test_without_the_bends_start_the_warm_fit_stays_curled(monkeypatch):
    # The case above is the one the extra start is for
    monkeypatch.setattr(HandFitter, "BENDS_DISAGREE", math.inf)
    assert index_curl_after_warm_fit() > 0.3
