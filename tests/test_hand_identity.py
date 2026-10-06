"""HandIdentityTracker: which side each detected hand is reported as."""
import math

import pytest

from hand_identity import (CLEAR_CURL_VOTE, ON_TOP_OVERLAP, SAME_HAND_OVERLAP, HandDetection, HandIdentityTracker,
                           combine_handedness_votes, skeleton_overlap)

LEFT_VOTE, RIGHT_VOTE = 0.9, -0.9
HAND_SIZE = 0.15


def skeleton(x, y, mirrored=False):
    """21 image points of an open hand, wrist at (x, y), fingers fanned upward, about HAND_SIZE across."""
    points = [(x, y)]
    for finger in range(5):
        angle = math.radians((60 - 30 * finger) * (-1 if mirrored else 1))
        for joint in range(1, 5):
            reach = HAND_SIZE * (0.25 + 0.17 * joint)
            points.append((x + reach * math.sin(angle) * 0.6, y - reach * math.cos(angle)))
    return points


def det(index=0, x=0.5, y=0.5, evidence=0.0, score=0.9, points=False) -> HandDetection:
    return HandDetection(index, (x, y), evidence, score, skeleton(x, y) if points else None)


def frames(tracker, hands, start=0.0, count=30, fps=30):
    """Run count frames of the same detections; the sides of the last one."""
    for i in range(count):
        sides = tracker.assign(hands, start + i / fps)
    return sides


@pytest.mark.parametrize("side, vote", [("left", LEFT_VOTE), ("right", RIGHT_VOTE)])
@pytest.mark.parametrize("x", [0.2, 0.8])
def test_a_lone_hand_moving_across_the_whole_image_keeps_its_side(side, vote, x):
    # The camera turns: the wrist sweeps from one edge to the other, faster than
    # continuity_radius per frame, with a weak vote on the way
    tracker = HandIdentityTracker()
    assert frames(tracker, [det(x=x, evidence=vote)]) == {0: side}
    for i, sweep_x in enumerate([0.95, 0.05, 0.5, 0.05, 0.95]):
        assert tracker.assign([det(x=sweep_x, evidence=0.0)], 1.0 + i / 30) == {0: side}


def test_a_lone_hand_jumping_within_a_few_frames_ignores_one_contrary_vote():
    tracker = HandIdentityTracker()
    frames(tracker, [det(x=0.1, evidence=LEFT_VOTE)], count=30)  # last frame at 29/30 s
    jump_at = 29 / 30 + HandIdentityTracker.FAST_MOVE_SECONDS - 0.01
    assert tracker.assign([det(x=0.9, evidence=RIGHT_VOTE)], jump_at) == {0: "left"}


def test_a_lone_hand_back_later_with_a_strong_vote_is_placed_by_it():
    tracker = HandIdentityTracker()
    frames(tracker, [det(x=0.1, evidence=LEFT_VOTE)], count=30)
    back_at = 29 / 30 + HandIdentityTracker.FAST_MOVE_SECONDS + 0.05
    assert tracker.assign([det(x=0.9, evidence=RIGHT_VOTE)], back_at) == {0: "right"}


def test_a_lone_hand_back_with_a_middling_vote_keeps_its_side():
    tracker = HandIdentityTracker(strong_evidence=0.5)
    frames(tracker, [det(x=0.1, evidence=LEFT_VOTE)], count=30)
    assert tracker.assign([det(x=0.9, evidence=-0.45)], 2.0) == {0: "left"}


@pytest.mark.parametrize("side, vote", [("left", LEFT_VOTE), ("right", RIGHT_VOTE)])
def test_a_lone_hand_back_after_a_short_loss_keeps_its_side(side, vote):
    tracker = HandIdentityTracker(memory_seconds=0.4)
    frames(tracker, [det(x=0.2, evidence=vote)])
    # Gone for a second (out of view), back on the other half with no clear vote
    assert tracker.assign([det(x=0.9, evidence=0.0)], 2.0) == {0: side}


@pytest.mark.parametrize("side, vote, contrary", [("left", LEFT_VOTE, RIGHT_VOTE), ("right", RIGHT_VOTE, LEFT_VOTE)])
def test_a_lone_hand_survives_a_burst_of_contrary_votes(side, vote, contrary):
    # A depth misread gives a strong wrong vote for a few frames (0.3 s)
    tracker = HandIdentityTracker()
    frames(tracker, [det(x=0.4, evidence=vote)])
    assert frames(tracker, [det(x=0.4, evidence=contrary)], start=1.0, count=9) == {0: side}


def test_a_stray_partial_hand_does_not_push_the_real_one_to_the_other_side():
    tracker = HandIdentityTracker()
    frames(tracker, [det(x=0.3, evidence=RIGHT_VOTE)])
    # Fingers of another hand show up at the edge, on the "right" side of the image
    stray = det(index=1, x=0.95, y=0.95, evidence=0.0, score=0.6)
    sides = frames(tracker, [det(x=0.3, evidence=RIGHT_VOTE), stray], start=1.0, count=5)
    assert sides[0] == "right"


def test_a_hand_is_new_when_it_appears_or_comes_back():
    tracker = HandIdentityTracker(memory_seconds=0.4)
    tracker.assign([det(x=0.3, evidence=LEFT_VOTE)], 0.0)
    assert tracker.new_sides == {"left"}
    tracker.assign([det(x=0.31, evidence=LEFT_VOTE)], 1 / 30)
    assert tracker.new_sides == set()
    tracker.assign([det(x=0.3, evidence=LEFT_VOTE)], 1.0)  # lost for longer than memory_seconds
    assert tracker.new_sides == {"left"}


def test_a_fast_move_does_not_make_a_hand_new():
    # Resetting then would drop a held grip mid-swing
    tracker = HandIdentityTracker(continuity_radius=0.15)
    tracker.assign([det(x=0.2, evidence=LEFT_VOTE)], 0.0)
    assert tracker.assign([det(x=0.6, evidence=0.0)], 1 / 30) == {0: "left"}
    assert tracker.new_sides == set()


def test_a_second_hand_appearing_does_not_take_the_tracked_hands_side():
    tracker = HandIdentityTracker()
    frames(tracker, [det(x=0.4, evidence=LEFT_VOTE)])
    # The newcomer stands where the order prior would call it left
    sides = tracker.assign([det(index=0, x=0.4, evidence=LEFT_VOTE), det(index=1, x=0.1, evidence=0.0)], 1.0)
    assert sides == {0: "left", 1: "right"}
    assert tracker.new_sides == {"right"}


def test_a_hand_that_changes_side_is_new_there():
    tracker = HandIdentityTracker(switch_frames=3)
    tracker.assign([det(x=0.3, evidence=LEFT_VOTE)], 0.0)
    for i in range(3):
        sides = tracker.assign([det(x=0.3, evidence=RIGHT_VOTE)], (i + 1) / 30)
    assert sides == {0: "right"}
    assert tracker.new_sides == {"right"}


@pytest.mark.parametrize("gone", [0.3, 1.0])
def test_the_other_hand_raised_right_after_is_placed_by_its_own_vote(gone):
    tracker = HandIdentityTracker()
    frames(tracker, [det(x=0.5, evidence=RIGHT_VOTE)])
    # Right hand lowered; the left one comes up somewhere else with a clear vote
    assert tracker.assign([det(x=0.1, evidence=LEFT_VOTE)], 1.0 + gone) == {0: "left"}


def test_after_two_hands_the_one_coming_back_alone_is_placed_by_its_vote():
    tracker = HandIdentityTracker()
    frames(tracker, [det(index=0, x=0.3, evidence=LEFT_VOTE), det(index=1, x=0.7, evidence=RIGHT_VOTE)])
    frames(tracker, [det(x=0.3, evidence=LEFT_VOTE)], start=1.0)  # the right hand leaves
    # The left hand leaves too; the right one comes back alone
    assert tracker.assign([det(x=0.7, evidence=RIGHT_VOTE)], 3.0) == {0: "right"}


def test_contrary_frames_do_not_add_up_across_two_hand_moments():
    tracker = HandIdentityTracker(switch_frames=15)
    frames(tracker, [det(x=0.3, evidence=LEFT_VOTE)])
    frames(tracker, [det(x=0.3, evidence=RIGHT_VOTE)], start=1.0, count=14)
    frames(tracker, [det(index=0, x=0.3, evidence=LEFT_VOTE), det(index=1, x=0.8, evidence=RIGHT_VOTE)],
           start=2.0, count=30)
    assert tracker.assign([det(x=0.3, evidence=RIGHT_VOTE)], 3.1) == {0: "left"}


def test_a_swapped_pair_corrects_itself_and_both_start_fresh():
    tracker = HandIdentityTracker()
    # Both hands tracked on the wrong sides of the image (as after a misread)
    frames(tracker, [det(index=0, x=0.2, evidence=0.0), det(index=1, x=0.8, evidence=0.0)])
    tracker._tracks["left"], tracker._tracks["right"] = tracker._tracks["right"], tracker._tracks["left"]
    sides = tracker.assign([det(index=0, x=0.2, evidence=0.0), det(index=1, x=0.8, evidence=0.0)], 1.0)
    assert sides == {0: "left", 1: "right"}
    assert tracker.new_sides == {"left", "right"}


def test_strong_evidence_still_switches_a_lone_hand():
    tracker = HandIdentityTracker(switch_frames=6)
    assert tracker.assign([det(x=0.3, evidence=LEFT_VOTE)], 0.0) == {0: "left"}
    sides = [tracker.assign([det(x=0.3, evidence=RIGHT_VOTE)], (i + 1) / 30)[0] for i in range(6)]
    assert sides == ["left"] * 5 + ["right"]


def test_votes_that_agree_add_up():
    assert combine_handedness_votes(0.5, 0.5, secondary_ignores_depth=True) == pytest.approx(0.5)


@pytest.mark.parametrize("curl", [CLEAR_CURL_VOTE, 0.9, -0.9])
def test_a_clear_curl_against_the_depth_free_cue_says_nothing(curl):
    # In POV a flipped MediaPipe depth reverses the curl vote; the palm-away cue cannot flip
    assert combine_handedness_votes(curl, -curl, secondary_ignores_depth=True) == 0.0


def test_a_flat_hands_curl_noise_does_not_silence_the_image_cue():
    assert combine_handedness_votes(0.05, -0.8, secondary_ignores_depth=True) == pytest.approx(0.8 * 0.05 - 0.2 * 0.8)


def test_mediapipes_label_never_vetoes_a_clear_curl():
    # Facing mode: the label assumes a palm toward the camera, so it can be the wrong one
    assert combine_handedness_votes(0.9, -0.9, secondary_ignores_depth=False) == pytest.approx(0.8 * 0.9 - 0.2 * 0.9)


def test_a_lost_hand_followed_to_its_predicted_place_is_found_there():
    # Two hands; the left one is lost on a fast move and comes back far from
    # where it was seen, close to the right hand. Without following, it would
    # look like it came from the right hand's track: new, and reset
    def run(follow: bool):
        tracker = HandIdentityTracker()
        frames(tracker, [det(0, x=0.3, evidence=LEFT_VOTE), det(1, x=0.75, evidence=RIGHT_VOTE)])
        tracker.assign([det(1, x=0.75, evidence=RIGHT_VOTE)], 1.0)
        if follow:
            tracker.follow("left", (0.55, 0.5))
        sides = tracker.assign([det(0, x=0.62, evidence=0.0), det(1, x=0.75, evidence=RIGHT_VOTE)], 1.1)
        return sides, tracker.new_sides

    assert run(follow=False)[1] == {"left"}
    sides, new_sides = run(follow=True)
    assert sides == {0: "left", 1: "right"}
    assert not new_sides


def test_following_a_side_with_no_track_does_nothing():
    tracker = HandIdentityTracker()
    tracker.follow("left", (0.5, 0.5))
    assert not tracker.has_track("left")


def test_a_lone_hand_changing_side_drops_its_old_track():
    tracker = HandIdentityTracker(switch_frames=3)
    frames(tracker, [det(x=0.5, evidence=LEFT_VOTE)], count=5)
    assert tracker.has_track("left")
    frames(tracker, [det(x=0.5, evidence=RIGHT_VOTE)], start=1.0, count=5)
    assert tracker.has_track("right") and not tracker.has_track("left")


def test_a_prediction_running_onto_the_visible_hand_leaves_the_lost_track_where_it_was():
    tracker = HandIdentityTracker()
    frames(tracker, [det(0, x=0.3, evidence=LEFT_VOTE), det(1, x=0.7, evidence=RIGHT_VOTE)])
    tracker.assign([det(1, x=0.7, evidence=RIGHT_VOTE)], 1.0)
    tracker.follow("left", (0.68, 0.5))
    # The right hand, alone in view, keeps its side
    assert tracker.assign([det(1, x=0.7, evidence=0.0)], 1.05) == {1: "right"}


def test_one_skeleton_fitted_twice_is_one_hand_even_with_its_wrists_apart():
    # A blurred hand fitted twice: the same fingers, the copy's wrist misplaced
    # further than duplicate_radius, its vote for the other side
    copy_points = skeleton(0.5, 0.6)
    copy_points[0] = (0.57, 0.6)
    copy = HandDetection(1, copy_points[0], RIGHT_VOTE, 0.8, copy_points)
    assert skeleton_overlap(skeleton(0.5, 0.6), copy_points) < SAME_HAND_OVERLAP
    tracker = HandIdentityTracker()
    for i in range(10):
        sides = tracker.assign([det(0, x=0.5, y=0.6, evidence=LEFT_VOTE, points=True), copy], i / 30)
        assert sides == {0: "left"}
    assert not tracker.has_track("right")


def test_two_hands_brought_together_both_stay():
    # Fists or palms side by side lie on top of each other, but each continues its own track
    tracker = HandIdentityTracker()
    for i in range(20):
        gap = 0.3 - i * (0.3 - 0.07) / 19
        hands = [det(0, x=0.5 - gap / 2, evidence=LEFT_VOTE, points=True),
                 det(1, x=0.5 + gap / 2, evidence=RIGHT_VOTE, points=True)]
        assert tracker.assign(hands, i / 30) == {0: "left", 1: "right"}
    assert SAME_HAND_OVERLAP < skeleton_overlap(hands[0].points, hands[1].points) < ON_TOP_OVERLAP


def test_a_hidden_hand_whose_skeleton_jumps_onto_the_other_hand_is_lost():
    # The hands cross: the left one hides behind the right, and MediaPipe fits its
    # slot onto the right hand in one frame, still voting left
    tracker = HandIdentityTracker()
    frames(tracker, [det(0, x=0.3, evidence=LEFT_VOTE, points=True), det(1, x=0.6, evidence=RIGHT_VOTE, points=True)])
    copy = det(0, x=0.53, evidence=LEFT_VOTE, score=0.95, points=True)
    right = det(1, x=0.6, evidence=RIGHT_VOTE, points=True)
    assert SAME_HAND_OVERLAP < skeleton_overlap(copy.points, right.points) < ON_TOP_OVERLAP
    assert tracker.assign([copy, right], 1.0) == {1: "right"}


def test_a_hand_appearing_next_to_the_only_tracked_one_is_kept():
    # No other track to have jumped from: a second hand raised next to the first
    tracker = HandIdentityTracker()
    frames(tracker, [det(0, x=0.4, evidence=LEFT_VOTE, points=True)])
    sides = tracker.assign([det(0, x=0.4, evidence=LEFT_VOTE, points=True),
                            det(1, x=0.47, evidence=RIGHT_VOTE, points=True)], 1.0)
    assert sides == {0: "left", 1: "right"}
