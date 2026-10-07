"""HandFlow: a lost hand's pixels are followed through the picture with optical flow."""
import cv2
import numpy as np
import pytest

from hand_flow import HandFlow

WIDTH, HEIGHT = 640, 480
HAND = 96  # pixels


def texture(seed: int, height: int, width: int) -> np.ndarray:
    """Smooth random grey texture, coarse enough to follow at a quarter of its size."""
    noise = np.random.default_rng(seed).uniform(0, 255, (height, width)).astype(np.float32)
    return cv2.GaussianBlur(noise, (0, 0), 6)


BACKGROUND = texture(1, HEIGHT, WIDTH + 200)
HAND_PIXELS = texture(2, HAND, HAND) * 2.0 - 60.0


def frame(hand_x: float, hand_y: float = 200.0, background_x: float = 0.0) -> np.ndarray:
    """A BGR picture: the background shifted left by background_x, the hand's top-left corner at (hand_x, hand_y)."""
    start = int(round(100 + background_x))
    picture = BACKGROUND[:, start:start + WIDTH].copy()
    x, y = int(round(hand_x)), int(round(hand_y))
    x0, x1 = max(x, 0), min(x + HAND, WIDTH)
    if x1 > x0:
        picture[y:y + HAND, x0:x1] = HAND_PIXELS[:, x0 - x:x1 - x]
    return cv2.cvtColor(np.clip(picture, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def landmarks(hand_x: float, hand_y: float = 200.0):
    """Normalised points spanning the hand, as MediaPipe's landmarks would."""
    return [((hand_x + dx) / WIDTH, (hand_y + dy) / HEIGHT, 0.0)
            for dx in (10, HAND / 2, HAND - 10) for dy in (10, HAND / 2, HAND - 10)]


def lost_after(flow: HandFlow, x: float, **picture) -> None:
    """The hand tracked at x on one frame."""
    flow.next_frame(frame(x, **picture))
    flow.tracked("left", landmarks(x))


def test_a_lost_hand_is_followed_as_its_pixels_move():
    flow = HandFlow()
    lost_after(flow, 200)
    for i in range(1, 4):
        flow.next_frame(frame(200 + 12 * i))   # 12 px a frame to the right
        offset = flow.follow("left")
    assert offset[0] == pytest.approx(36 / WIDTH, rel=0.25)
    assert offset[1] == pytest.approx(0.0, abs=3 / HEIGHT)


def test_a_turning_head_does_not_drag_a_hand_that_stays_put_in_the_picture():
    # POV: the head turns, the whole room slides across the picture, the hand moves with the head
    flow = HandFlow()
    lost_after(flow, 250)
    for i in range(1, 4):
        flow.next_frame(frame(250, background_x=10 * i))
        offset = flow.follow("left")
    assert np.abs(offset) == pytest.approx((0.0, 0.0), abs=4 / WIDTH)


def test_a_hand_that_left_the_picture_stays_where_it_was_last_seen():
    flow = HandFlow()
    lost_after(flow, 560)
    offsets = []
    for i in range(1, 7):
        flow.next_frame(frame(560 + 20 * i))
        offsets.append(flow.follow("left"))
    # Followed while mostly in the picture, then held
    assert offsets[0][0] > 0.0
    assert offsets[-1] == pytest.approx(offsets[-2])


def test_nothing_to_follow_without_a_tracked_frame_or_a_frame_before():
    flow = HandFlow()
    flow.next_frame(frame(200))
    assert flow.follow("left") is None           # never tracked
    flow.tracked("left", landmarks(200))
    flow.forget("left")
    assert flow.follow("left") is None           # forgotten
    first = HandFlow()
    first.tracked("left", landmarks(200))
    first.next_frame(frame(200))
    assert first.follow("left") is None         # one frame only: nothing to compare


def test_a_new_picture_size_starts_over():
    flow = HandFlow()
    lost_after(flow, 200)
    flow.next_frame(cv2.resize(frame(212), (320, 240)))
    assert flow.follow("left") is None


def test_a_hand_tracked_again_is_followed_from_its_new_place():
    flow = HandFlow()
    lost_after(flow, 200)
    flow.next_frame(frame(212))
    assert flow.follow("left")[0] > 0.0
    flow.tracked("left", landmarks(212))
    flow.next_frame(frame(212))
    assert flow.follow("left") == pytest.approx((0.0, 0.0), abs=1 / WIDTH)


def test_two_lost_hands_are_followed_each_on_its_own():
    flow = HandFlow()
    flow.next_frame(frame(100))
    flow.tracked("left", landmarks(100))
    flow.tracked("right", [(x + 400 / WIDTH, y, z) for x, y, z in landmarks(100)])  # still background there
    flow.next_frame(frame(112))
    assert flow.follow("left")[0] == pytest.approx(12 / WIDTH, rel=0.3)
    assert flow.follow("right") == pytest.approx((0.0, 0.0), abs=2 / WIDTH)
