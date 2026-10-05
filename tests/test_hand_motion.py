"""MotionPredictor: lost hands keep moving along their path, and blend back when found."""
import math

import numpy as np
import pytest

from hand_motion import (BLEND_SECONDS, MAX_EXTRA_TURN_DEG, MIN_IMAGE_SPEED, MotionPredictor, damped_distance,
                         fit_motion, quat_from_rotation_vector, rotation_vector)

FPS = 30
IDENTITY = (1.0, 0.0, 0.0, 0.0)


def track(predictor, positions, image=None, rotations=None, hand="left", start=0.0):
    """
    Feed one tracked frame per position; returns the time of the last one. By
    default the wrist crosses the picture with the hand, as at 0.6 m from the camera.
    """
    for i, position in enumerate(positions):
        wrist = image[i] if image is not None else (0.5 + position[0] / 0.6, 0.5 - position[1] / 0.6)
        rotation = rotations[i] if rotations is not None else IDENTITY
        predictor.observe(hand, start + i / FPS, position, rotation, wrist)
    return start + (len(positions) - 1) / FPS


def turned_deg(q) -> float:
    return math.degrees(np.linalg.norm(rotation_vector(q)))


def test_fit_motion_recovers_velocity_and_acceleration():
    times = np.arange(6) / FPS
    points = np.array([[1.0 + 2.0 * t + 0.5 * 3.0 * t * t] for t in times])
    velocity, acceleration = fit_motion(times, points)
    assert velocity[0] == pytest.approx(2.0 + 3.0 * times[-1])
    assert acceleration[0] == pytest.approx(3.0)


def test_a_hand_moving_steadily_keeps_going_the_same_way_when_lost():
    predictor = MotionPredictor(0.3)
    positions = [(0.5 * i / FPS, 0.0, -0.4) for i in range(5)]  # 0.5 m/s to the right
    last = track(predictor, positions)
    one_frame = predictor.predict("left", last + 1 / FPS)
    # About where it would be by now, a little short because it slows down
    assert one_frame.camera_position[0] == pytest.approx(positions[-1][0] + 0.5 / FPS, rel=0.1)
    assert one_frame.camera_position[1:] == pytest.approx(positions[-1][1:], abs=1e-6)
    later = predictor.predict("left", last + 0.2)
    assert later.camera_position[0] > one_frame.camera_position[0]
    assert later.camera_position[0] - positions[-1][0] < 0.5 * 0.2  # slower than it was moving


def test_a_curving_hand_keeps_curving():
    predictor = MotionPredictor(0.3)
    radius, speed = 0.2, 1.0  # circle in x-y, counterclockwise
    angles = [speed / radius * i / FPS for i in range(6)]
    positions = [(radius * math.cos(a), radius * math.sin(a), -0.4) for a in angles]
    last = track(predictor, positions)
    end = predictor.predict("left", last + 0.1).camera_position
    heading = np.subtract(positions[-1], positions[-2])
    moved = end - np.array(positions[-1])
    # Turned toward the centre: left of the heading for a counterclockwise path
    assert np.cross(heading[:2], moved[:2]) > 0
    straight_on = np.array(positions[-1]) + heading / np.linalg.norm(heading) * np.linalg.norm(moved)
    assert np.linalg.norm(end - straight_on) > 0.002


def test_a_hand_that_stopped_does_not_drift():
    predictor = MotionPredictor(0.3)
    last = track(predictor, [(0.1, 0.2, -0.4)] * 6)
    prediction = predictor.predict("left", last + 0.25)
    assert prediction.camera_position == pytest.approx((0.1, 0.2, -0.4), abs=1e-9)
    assert turned_deg(prediction.rotation) == pytest.approx(0.0, abs=1e-6)


def test_a_hand_lost_while_barely_moving_holds_still():
    # Its speed would be mostly noise: on recorded clips holding ended closer to where it was found
    predictor = MotionPredictor(0.3)
    slow = MIN_IMAGE_SPEED * 0.8 * 0.6  # m/s at 0.6 m, just under the threshold
    positions = [(slow * i / FPS, 0.0, -0.4 + 0.5 * i / FPS) for i in range(5)]
    last = track(predictor, positions)
    assert predictor.predict("left", last + 0.2).camera_position == pytest.approx(positions[-1], abs=1e-9)


def test_depth_is_trusted_for_less_time_than_the_picture_axes():
    predictor = MotionPredictor(0.3)
    last = track(predictor, [(0.5 * i / FPS, 0.0, -0.4 - 0.5 * i / FPS) for i in range(5)])
    moved = predictor.predict("left", last + 0.2).camera_position - np.array((0.5 * 4 / FPS, 0.0, -0.4 - 0.5 * 4 / FPS))
    assert moved[0] > 0.0 and moved[2] < 0.0  # both keep going: right, and away from the camera
    assert abs(moved[2]) < 0.5 * moved[0]


def test_past_the_horizon_there_is_no_prediction():
    predictor = MotionPredictor(0.3)
    last = track(predictor, [(0.03 * i, 0.0, -0.4) for i in range(5)])
    assert predictor.predict("left", last + 0.29) is not None
    assert predictor.predict("left", last + 0.31) is None


def test_prediction_off_predicts_nothing():
    predictor = MotionPredictor(0.0)
    last = track(predictor, [(0.03 * i, 0.0, -0.4) for i in range(5)])
    assert predictor.predict("left", last + 1 / FPS) is None


def test_the_prediction_comes_to_rest():
    # However long the gap, a hand doesn't travel further than its speed allows
    predictor = MotionPredictor(1.0)
    last = track(predictor, [(1.0 * i / FPS, 0.0, -0.4) for i in range(5)])  # 1 m/s
    near_end = predictor.predict("left", last + 0.9).camera_position[0]
    end = predictor.predict("left", last + 1.0).camera_position[0]
    assert end - near_end < 0.002
    assert end - 4 / FPS < 0.25


def test_a_hand_leaving_the_picture_eases_to_a_stop_instead_of_continuing():
    def travel(image_x):
        predictor = MotionPredictor(0.3)
        image = [(image_x + 0.02 * i, 0.5) for i in range(5)]  # moving right in the picture
        last = track(predictor, [(0.6 * i / FPS, 0.0, -0.4) for i in range(5)], image=image)
        prediction = predictor.predict("left", last + 0.2)
        return prediction.leaving, prediction.camera_position[0] - 0.6 * 4 / FPS

    leaving, near_edge = travel(0.88)   # last seen at 0.96, by the right edge
    staying, in_middle = travel(0.3)
    assert leaving and not staying
    assert 0.0 < near_edge < 0.5 * in_middle


def moving_hand(speed=0.5):
    """A predictor that tracked a hand moving right at speed m/s; and the time of its last frame."""
    predictor = MotionPredictor(0.3)
    return predictor, track(predictor, [(speed * i / FPS, 0.0, -0.4) for i in range(5)])


def test_a_hand_found_again_blends_from_where_it_was_predicted():
    predictor, last = moving_hand()
    predictor.predict("left", last + 1 / FPS)
    predictor.predict("left", last + 2 / FPS)
    found_at = last + 3 / FPS
    # Where it would have been shown on the frame it is found: the blend starts there, no stall
    twin, _ = moving_hand()
    predicted = twin.predict("left", found_at)
    measured = (predicted.camera_position[0] + 0.1, 0.0, -0.4)
    turned = quat_from_rotation_vector(np.array((0.0, 0.0, math.radians(30))))

    first, first_rotation = predictor.observe("left", found_at, measured, turned, (0.5, 0.5))
    # No jump: the first frame back is where the hand was shown
    assert first == pytest.approx(tuple(predicted.camera_position), abs=1e-9)
    assert turned_deg(first_rotation) == pytest.approx(0.0, abs=1e-6)

    halfway, _ = predictor.observe("left", found_at + BLEND_SECONDS / 2, measured, turned, (0.5, 0.5))
    assert predicted.camera_position[0] < halfway[0] < measured[0]

    after, after_rotation = predictor.observe("left", found_at + BLEND_SECONDS + 0.01, measured, turned, (0.5, 0.5))
    assert after == pytest.approx(measured)
    assert turned_deg(after_rotation) == pytest.approx(30.0)


def test_a_hand_lost_again_while_blending_in_carries_on_from_where_it_was_shown():
    # Motion blur often gives one detection between lost frames
    predictor, last = moving_hand(speed=1.0)
    for i in range(1, 4):
        predictor.predict("left", last + i / FPS)
    found_at = last + 4 / FPS
    measured = (predictor.predict("left", found_at - 1e-6).camera_position[0] + 0.1, 0.0, -0.4)
    predictor, last = moving_hand(speed=1.0)
    for i in range(1, 4):
        predictor.predict("left", last + i / FPS)
    sent, _ = predictor.observe("left", found_at, measured, IDENTITY, (0.9, 0.5))
    again = predictor.predict("left", found_at + 1 / FPS)
    # One frame of motion: the measurement 10 cm ahead says the hand is faster than
    # predicted, so a little more than 1 m/s, but no snap to the measurement
    assert np.linalg.norm(again.camera_position - np.array(sent)) < 0.05
    assert again.camera_position[0] > sent[0]  # still moving the way it was shown moving


def test_a_hand_lost_while_barely_moving_does_not_turn_either():
    predictor = MotionPredictor(0.3)
    rotations = [quat_from_rotation_vector(np.array((0.0, math.radians(5) * i, 0.0))) for i in range(5)]
    last = track(predictor, [(0.0, 0.0, -0.4)] * 5, rotations=rotations)
    assert turned_deg(predictor.predict("left", last + 0.2).rotation) == pytest.approx(20.0, abs=1e-6)


def test_a_forgotten_hand_starts_clean():
    predictor = MotionPredictor(0.3)
    last = track(predictor, [(0.5 * i / FPS, 0.0, -0.4) for i in range(5)])
    predictor.predict("left", last + 1 / FPS)
    predictor.forget("left")
    assert predictor.predict("left", last + 2 / FPS) is None
    measured = (1.0, 1.0, -1.0)
    assert predictor.observe("left", last + 3 / FPS, measured, IDENTITY, (0.5, 0.5))[0] == measured


def test_a_turning_hand_keeps_turning_but_never_more_than_the_cap():
    predictor = MotionPredictor(1.0)
    step = math.radians(20)  # 600 degrees a second about Y: 60 more before it slows to a stop
    rotations = [quat_from_rotation_vector(np.array((0.0, step * i, 0.0))) for i in range(5)]
    last = track(predictor, [(0.5 * i / FPS, 0.0, -0.4) for i in range(5)], rotations=rotations)
    soon = predictor.predict("left", last + 1 / FPS).rotation
    extra = turned_deg(soon) - math.degrees(step * 4)
    assert 0.0 < extra < 20.0
    much_later = predictor.predict("left", last + 1.0).rotation
    assert turned_deg(much_later) - math.degrees(step * 4) == pytest.approx(MAX_EXTRA_TURN_DEG, abs=0.5)


def test_hands_are_predicted_independently():
    predictor = MotionPredictor(0.3)
    last = track(predictor, [(0.5 * i / FPS, 0.0, -0.4) for i in range(5)], hand="left")
    track(predictor, [(0.0, 0.0, -0.4)] * 5, hand="right")
    assert predictor.predict("right", last + 0.1).camera_position == pytest.approx((0.0, 0.0, -0.4))
    assert predictor.predict("left", last + 0.1).camera_position[0] > 0.5 * 4 / FPS


def test_the_path_runs_from_the_tracked_frames_to_the_predicted_end():
    predictor = MotionPredictor(0.3)
    image = [(0.3 + 0.03 * i, 0.5) for i in range(5)]
    last = track(predictor, [(0.6 * i / FPS, 0.0, -0.4) for i in range(5)], image=image)
    path = predictor.predict("left", last + 1 / FPS).path
    assert path[:5] == pytest.approx(np.array(image))
    assert np.all(np.diff(path[:, 0]) > 0)  # keeps heading right, slowing down


def test_damped_distance_is_bounded():
    by_velocity, by_acceleration = damped_distance(100.0, 0.2)
    assert by_velocity == pytest.approx(0.2)
    assert by_acceleration == pytest.approx(0.04)
