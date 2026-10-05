"""MotionPredictor: lost hands keep moving along their path, and blend back when found."""
import math

import numpy as np
import pytest

from hand_motion import (BLEND_SECONDS, DEPTH_DAMPING_SECONDS, MAX_EXTRA_TURN_DEG, MAX_SPEED_M_S, MIN_IMAGE_SPEED,
                         MOMENTUM_SECONDS, POSITION_DAMPING_SECONDS, Curve, MotionPredictor, curve_offsets,
                         fit_curve, fit_motion, quat_from_rotation_vector, rotation_vector)

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


def carried_distance(tau: float, hold_seconds: float, damping_seconds: float) -> float:
    """How far a unit speed carries in tau seconds with speed_kept(), exactly: its integral from 0 to tau."""
    held = min(tau, hold_seconds)
    return held + damping_seconds * (1.0 - math.exp(-max(tau - hold_seconds, 0.0) / damping_seconds))


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
    # Where it would be by now: a frame in, it still has its full speed
    assert one_frame.camera_position[0] == pytest.approx(positions[-1][0] + 0.5 / FPS, rel=0.02)
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
    positions = [(slow * i / FPS, 0.0, -0.4) for i in range(5)]
    last = track(predictor, positions)
    assert predictor.predict("left", last + 0.2).camera_position == pytest.approx(positions[-1], abs=1e-9)


def test_a_punch_straight_at_the_camera_keeps_going_though_it_barely_crosses_the_picture():
    predictor = MotionPredictor(0.3)
    positions = [(0.0, 0.0, -0.6 + 2.0 * i / FPS) for i in range(5)]  # 2 m/s toward the camera
    last = track(predictor, positions)
    moved = predictor.predict("left", last + 0.1).camera_position - np.array(positions[-1])
    assert moved[:2] == pytest.approx((0.0, 0.0), abs=1e-9)
    assert moved[2] == pytest.approx(2.0 * carried_distance(0.1, MOMENTUM_SECONDS, DEPTH_DAMPING_SECONDS), rel=0.05)


def test_a_lost_hand_keeps_its_full_speed_at_first():
    # No snap into a slow slide: the first frames lost move as fast as the hand was moving
    predictor = MotionPredictor(0.3)
    speed = 3.0
    # Crossing the middle of the picture, nowhere near an edge
    image = [(0.2 + 0.1 * i, 0.5) for i in range(5)]
    last = track(predictor, [(speed * i / FPS, 0.0, -0.4) for i in range(5)], image=image)
    for i in range(1, 4):
        t = MOMENTUM_SECONDS * i / 3
        moved = predictor.predict("left", last + t).camera_position[0] - speed * 4 / FPS
        assert moved == pytest.approx(speed * t, rel=0.02)


def test_a_swinging_hand_keeps_curving_while_lost():
    # A swing around the elbow: a circle of 0.3 m at 3 m/s, lost for a long gap
    predictor = MotionPredictor(0.3)
    radius, speed = 0.3, 3.0
    angles = [speed / radius * i / FPS for i in range(5)]
    positions = [(radius * math.sin(a), radius * (1 - math.cos(a)), -0.4) for a in angles]
    # Kept in the middle of the picture, so it isn't leaving it
    last = track(predictor, positions, image=[(0.4 + p[0], 0.6 - p[1]) for p in positions])
    t = MOMENTUM_SECONDS
    end = predictor.predict("left", last + t).camera_position
    a = angles[-1] + speed / radius * t
    on_circle = np.array((radius * math.sin(a), radius * (1 - math.cos(a)), -0.4))
    heading = np.array((math.cos(angles[-1]), math.sin(angles[-1]), 0.0))
    straight_on = np.array(positions[-1]) + heading * speed * t
    assert np.linalg.norm(end - on_circle) < 0.3 * np.linalg.norm(straight_on - on_circle)
    # Slowing down later, it stays on the arc instead of spiralling in toward the elbow
    centre = np.array((0.0, radius))
    for t in (0.15, 0.25):
        later = predictor.predict("left", last + t).camera_position
        assert np.linalg.norm(later[:2] - centre) == pytest.approx(radius, abs=0.005)


def test_a_punch_keeps_going_away_from_the_camera():
    # A POV punch is mostly depth: steady frames moving away keep it moving
    predictor = MotionPredictor(0.3)
    last = track(predictor, [(0.6 * i / FPS, 0.0, -0.3 - 1.0 * i / FPS) for i in range(5)])
    moved = predictor.predict("left", last + 0.2).camera_position[2] - (-0.3 - 4 / FPS)
    assert moved < -0.05


def test_a_one_frame_depth_jump_does_not_send_the_hand_away():
    # A turning wrist or a hidden finger moves one camera's depth by centimetres at once
    predictor = MotionPredictor(0.3)
    positions = [(0.6 * i / FPS, 0.0, -0.4) for i in range(4)] + [(0.6 * 4 / FPS, 0.0, -0.48)]
    last = track(predictor, positions)
    assert predictor.predict("left", last + 0.2).camera_position[2] == pytest.approx(-0.48, abs=1e-9)


def test_depth_mostly_made_by_one_step_holds():
    # Steps of 6, 6, 6 and 40 mm all one way: the last one is a misread, not a punch
    predictor = MotionPredictor(0.3)
    depths = np.cumsum([-0.4, -0.006, -0.006, -0.006, -0.04])
    last = track(predictor, [(0.6 * i / FPS, 0.0, z) for i, z in enumerate(depths)])
    assert predictor.predict("left", last + 0.2).camera_position[2] == pytest.approx(depths[-1], abs=1e-9)


def test_depth_wobbling_under_the_noise_level_still_moves_with_its_trend():
    # 2 cm a frame away from the camera, with a 3 mm wobble back on one frame
    predictor = MotionPredictor(0.3)
    depths = [-0.4, -0.42, -0.44, -0.437, -0.457, -0.477]
    last = track(predictor, [(0.6 * i / FPS, 0.0, z) for i, z in enumerate(depths)])
    assert predictor.predict("left", last + 0.1).camera_position[2] < depths[-1] - 0.02


def test_two_frames_of_depth_change_are_not_enough_to_move_depth():
    # A hand turning or curling changes one camera's depth by over 10 cm a frame for two frames
    predictor = MotionPredictor(0.3)
    positions = [(0.6 * i / FPS, 0.0, -0.4) for i in range(3)]
    positions += [(0.6 * 3 / FPS, 0.0, -0.28), (0.6 * 4 / FPS, 0.0, -0.16)]
    last = track(predictor, positions)
    assert predictor.predict("left", last + 0.2).camera_position[2] == pytest.approx(-0.16, abs=1e-9)


def test_fit_curve_with_two_steps_goes_straight_at_their_speed():
    curve = fit_curve([0.0, 1 / FPS, 2 / FPS], np.array(((0.0, 0.0), (0.03, 0.0), (0.06, 0.03))))
    assert curve.turn_rate == 0.0 and curve.speed_change == 0.0
    assert np.linalg.norm(curve.velocity) == pytest.approx(0.5 * (0.9 + math.hypot(0.9, 0.9)))
    assert curve.velocity[0] == pytest.approx(curve.velocity[1])  # the last step's heading


def test_depth_going_back_and_forth_holds():
    predictor = MotionPredictor(0.3)
    positions = [(0.6 * i / FPS, 0.0, -0.4 + (0.02 if i % 2 else 0.0)) for i in range(5)]
    last = track(predictor, positions)
    prediction = predictor.predict("left", last + 0.2).camera_position
    assert prediction[2] == pytest.approx(positions[-1][2], abs=1e-9)
    assert prediction[0] > positions[-1][0]  # still moving across the picture


def test_a_wrist_that_jumps_across_the_picture_is_left_out_of_the_fit():
    # A misdetection, or the other hand taken for this one, for a frame
    predictor = MotionPredictor(0.3)
    image = [(0.3 + 0.02 * i, 0.5) for i in range(4)] + [(0.75, 0.5)]
    positions = [(0.6 * i / FPS, 0.0, -0.4) for i in range(4)] + [(0.3, 0.0, -0.4)]
    last = track(predictor, positions, image=image)
    prediction = predictor.predict("left", last + 0.2)
    # Moves on at the speed of the frames before the jump, from where the hand was last sent
    expected = 0.6 * carried_distance(0.2, MOMENTUM_SECONDS, POSITION_DAMPING_SECONDS)
    assert prediction.camera_position[0] - 0.3 == pytest.approx(expected, rel=0.1)


def test_a_lost_hand_is_moved_at_the_cap_when_it_seemed_faster():
    predictor = MotionPredictor(1.0)
    image = [(0.1 + 0.05 * i, 0.5) for i in range(5)]
    last = track(predictor, [(5.0 * i / FPS, 0.0, -0.4) for i in range(5)], image=image)
    one_frame = predictor.predict("left", last + 1 / FPS).camera_position[0] - 5.0 * 4 / FPS
    assert one_frame == pytest.approx(MAX_SPEED_M_S / FPS, rel=0.02)


def test_a_hand_speeding_up_is_not_moved_faster_than_when_it_was_lost():
    predictor = MotionPredictor(0.3)
    xs = [0.5 * (1.0 + 0.5 * i) * i / FPS for i in range(5)]   # from 0.5 m/s, faster each frame
    image = [(0.2 + x, 0.5) for x in xs]
    last = track(predictor, [(x, 0.0, -0.4) for x in xs], image=image)
    lost_at = (xs[-1] - xs[-2]) * FPS
    moved = predictor.predict("left", last + MOMENTUM_SECONDS).camera_position[0] - xs[-1]
    assert moved <= lost_at * 1.15 * MOMENTUM_SECONDS


def test_a_hand_setting_off_from_rest_is_not_taken_for_a_jump():
    predictor = MotionPredictor(0.3)
    image = [(0.4, 0.5)] * 4 + [(0.47, 0.5)]
    positions = [(0.0, 0.0, -0.4)] * 4 + [(0.042, 0.0, -0.4)]
    last = track(predictor, positions, image=image)
    assert predictor.predict("left", last + 0.1).camera_position[0] > 0.042 + 0.02


def test_a_resting_hand_that_teleports_is_left_out_of_the_fit():
    # Half the picture in one frame from rest: the other hand taken for this one, not a start
    predictor = MotionPredictor(0.3)
    image = [(0.3, 0.5)] * 4 + [(0.8, 0.5)]
    positions = [(0.0, 0.0, -0.4)] * 4 + [(0.3, 0.0, -0.4)]
    last = track(predictor, positions, image=image)
    assert predictor.predict("left", last + 0.1).camera_position == pytest.approx((0.3, 0.0, -0.4), abs=1e-9)


def test_a_hand_turning_back_carries_on_the_new_way_without_hooking():
    # A punch pulled back: right, right, right, then left, left, left
    predictor = MotionPredictor(0.3)
    xs = [0.0, 0.05, 0.1, 0.15, 0.1, 0.05, 0.0]
    last = track(predictor, [(x, 0.0, -0.4) for x in xs], image=[(0.5 + x, 0.5) for x in xs])
    moved = predictor.predict("left", last + 0.1).camera_position - np.array((0.0, 0.0, -0.4))
    assert moved[0] < -0.1
    assert abs(moved[1]) < 0.01   # straight back, no turn from the reversal


def test_a_fast_hand_about_to_cross_an_edge_is_leaving():
    # Not near the edge yet (0.9), but at 3 picture widths a second it crosses it within a frame or two
    predictor = MotionPredictor(0.3)
    image = [(0.5 + 0.1 * i, 0.5) for i in range(5)]
    last = track(predictor, [(1.8 * i / FPS, 0.0, -0.4) for i in range(5)], image=image)
    assert predictor.predict("left", last + 1 / FPS).leaving


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
    assert end - 4 / FPS == pytest.approx(1.0 * carried_distance(1.0, MOMENTUM_SECONDS, POSITION_DAMPING_SECONDS),
                                          rel=0.02)


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
    # predicted (about 1.6 m/s, 5 cm a frame), but no snap to the measurement
    assert np.linalg.norm(again.camera_position - np.array(sent)) < 0.07
    assert np.linalg.norm(again.camera_position - np.array(measured)) > 0.03
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


def test_a_curve_keeps_turning_at_the_fitted_rate():
    # 1 unit/s turning at 5 rad/s (acceleration across the path = speed * rate), no slowing down
    s = np.linspace(0.0, 0.2, 801)
    offsets = curve_offsets(s, Curve(np.array((1.0, 0.0)), 5.0, 0.0), hold_seconds=1.0, damping_seconds=0.2)
    radius = 1.0 / 5.0
    expected = np.column_stack((radius * np.sin(5.0 * s), radius * (1.0 - np.cos(5.0 * s))))
    assert offsets == pytest.approx(expected, abs=1e-5)


def test_a_curve_keeps_slowing_down_as_it_was():
    s = np.linspace(0.0, 0.1, 401)
    offsets = curve_offsets(s, Curve(np.array((1.0, 0.0)), 0.0, -4.0), hold_seconds=1.0, damping_seconds=0.2)
    assert offsets[-1] == pytest.approx((0.1 - 0.5 * 4.0 * 0.01, 0.0), abs=1e-6)


def test_fit_curve_follows_an_arc_however_far_it_turned():
    # 76 degrees of a 10 rad/s swing in five frames: a quadratic fit flattens this
    radius, speed = 0.3, 3.0
    times = np.arange(5) / FPS
    angles = speed / radius * times
    curve = fit_curve(times, np.column_stack((radius * np.sin(angles), radius * (1 - np.cos(angles)))))
    assert curve.turn_rate == pytest.approx(speed / radius, rel=0.01)
    assert np.linalg.norm(curve.velocity) == pytest.approx(speed, rel=0.02)
    assert math.atan2(curve.velocity[1], curve.velocity[0]) == pytest.approx(angles[-1], abs=0.01)
