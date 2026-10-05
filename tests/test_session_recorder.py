import json
import math
import threading
import time

import cv2
import numpy as np
import pytest

import session_recorder
from hand_data import HandData, TrackedHand, TrackingFrame
from session_recorder import SessionRecorder, redact, redact_text


def _frame(width=64, height=48, hands=None) -> TrackingFrame:
    image = np.zeros((height, width, 3), np.uint8)
    return TrackingFrame(frame_rgb=image, frame_bgr=image, hands=hands or [], camera_fps=30.0, tracking_fps=30.0)


def _hand(side="right") -> TrackedHand:
    data = HandData(hand_type=side, position=(0.1, -0.2, -0.4), rotation=(1.0, 0.0, 0.0, 0.0), gesture="OPEN",
                    trigger_value=0.25, grip_value=0.0, landmarks=[(0.5, 0.5, 0.0)] * 21,
                    finger_curls=(0.1, 0.2, 0.3, 0.4, 0.5))
    return TrackedHand(data=data, camera_position=(0.1, -0.2, -0.4), camera_points=np.full((21, 3), 0.123456789),
                       gesture_scores={"OPEN": 0.9})


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _record(tmp_path, video, frames, **kwargs):
    clock = _Clock()
    recorder = SessionRecorder(tmp_path, {"camera": {"fps": 30}}, video=video, name="s", clock=clock, **kwargs)
    for i, frame in enumerate(frames):
        clock.now = 100.0 + i / 30
        recorder.add_frame(frame, clock.now)
    recorder.stop()
    return recorder


def test_data_only_writes_session_frames_and_end(tmp_path):
    recorder = _record(tmp_path, False, [_frame(hands=[_hand()]), _frame()])
    assert recorder.video_path is None
    assert sorted(p.name for p in tmp_path.iterdir()) == ["s.log"]
    records = _records(recorder.data_path)
    assert [r["type"] for r in records] == ["session", "frame", "frame", "end"]
    session, first, second, end = records
    assert session["format"] == session_recorder.FORMAT and session["video"] is None
    assert (session["width"], session["height"]) == (64, 48)
    assert first["i"] == 0 and first["t"] == 0.0 and first["video_frame"] is None
    assert math.isclose(second["t"], 1 / 30, abs_tol=1e-4)
    hand = first["hands"][0]
    assert hand["side"] == "right" and hand["predicted"] is False and hand["sendable"] is True
    assert hand["trigger"] == 0.25 and len(hand["landmarks"]) == 21
    assert hand["camera_points"][0][0] == 0.12346  # rounded to keep the file small
    assert end["frames"] == 2 and end["video_dropped"] == 0


def test_video_has_one_picture_per_frame(tmp_path):
    recorder = _record(tmp_path, True, [_frame() for _ in range(5)])
    assert recorder.video_path.name == "s.mp4" and recorder.data_path.name == "s.mp4.log"
    frames = [r for r in _records(recorder.data_path) if r["type"] == "frame"]
    assert [r["video_frame"] for r in frames] == [0, 1, 2, 3, 4]
    video = cv2.VideoCapture(str(recorder.video_path))
    assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    assert video.get(cv2.CAP_PROP_FPS) == 30  # the tracking rate, so it plays in real time
    assert (video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT)) == (64, 48)
    video.release()


def test_slow_tracking_still_plays_in_real_time(tmp_path):
    clock = _Clock()
    recorder = SessionRecorder(tmp_path, {"camera": {"fps": 30}}, video=True, name="s", clock=clock)
    for t in (0.0, 0.1, 0.2):  # 10 fps tracking into a 30 fps video
        recorder.add_frame(_frame(), 100.0 + t)
    recorder.stop()
    frames = [r for r in _records(recorder.data_path) if r["type"] == "frame"]
    assert [r["video_frame"] for r in frames] == [0, 3, 6]
    video = cv2.VideoCapture(str(recorder.video_path))
    assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == 7  # each picture shown until the next one
    video.release()


def test_frames_faster_than_the_video_keep_their_order(tmp_path):
    recorder = SessionRecorder(tmp_path, {"camera": {"fps": 30}}, video=True, name="s", clock=_Clock())
    for t in (0.0, 0.01, 0.02, 0.1):
        recorder.add_frame(_frame(), 100.0 + t)
    recorder.stop()
    frames = [r for r in _records(recorder.data_path) if r["type"] == "frame"]
    assert [r["video_frame"] for r in frames] == [0, 1, 2, 3]


class _SlowWriter:
    """A video writer stuck on its first picture until released, like a slow disk."""

    def __init__(self):
        self.writing = threading.Event()
        self.release_disk = threading.Event()
        self.written = 0

    def __call__(self, path, fps, size):
        return self

    def write(self, image):
        self.writing.set()
        self.release_disk.wait(5)
        self.written += 1

    def release(self):
        pass


def test_a_slow_disk_drops_pictures_never_data_or_tracking(tmp_path, monkeypatch):
    monkeypatch.setattr(session_recorder, "VIDEO_QUEUE_FRAMES", 2)
    writer = _SlowWriter()
    recorder = SessionRecorder(tmp_path, {}, video=True, name="s", video_writer=writer)
    recorder.add_frame(_frame(), 0.0)
    assert writer.writing.wait(5)
    started = time.perf_counter()
    for i in range(1, 5):  # two fit in the queue, two don't
        recorder.add_frame(_frame(), i / 30)
    assert time.perf_counter() - started < 0.5  # add_frame never waits for the disk
    writer.release_disk.set()
    recorder.stop()

    records = _records(recorder.data_path)
    assert [r["video_frame"] for r in records if r["type"] == "frame"] == [0, 1, 2, None, None]
    assert records[-1]["video_dropped"] == 2 and records[-1]["frames"] == 5
    assert writer.written == 3


def test_log_and_settings_are_recorded_after_the_first_frame(tmp_path):
    recorder = SessionRecorder(tmp_path, {}, video=False, name="s")
    recorder.log("before the first frame\n")
    recorder.add_frame(_frame())
    recorder.log("hel")
    recorder.log("lo")
    recorder.log("\n")
    recorder.settings_changed({"camera.device_id": "http://cam.local/video", "calibration.scale": 1.5})
    recorder.stop()
    recorder.add_frame(_frame())  # after stop: ignored
    types = [r["type"] for r in _records(recorder.data_path)]
    assert types == ["session", "frame", "log", "settings", "end"]
    log, settings = _records(recorder.data_path)[2:4]
    assert log["text"] == "hello"
    assert settings["changes"] == {"camera.device_id": session_recorder.REDACTED, "calibration.scale": 1.5}


def test_settings_lose_paths_urls_and_saved_presets(tmp_path):
    settings = {"camera": {"device_id": "rtsp://user:pass@cam/stream", "fps": 30},
                "tracking": {"view_mode": "pov"}, "presets": {"Mine": {}}, "folder": "C:\\data"}
    clean = redact({k: v for k, v in settings.items() if k != "presets"})
    assert clean == {"camera": {"device_id": session_recorder.REDACTED, "fps": 30},
                     "tracking": {"view_mode": "pov"}, "folder": session_recorder.REDACTED}
    recorder = SessionRecorder(tmp_path, settings, video=False, name="s")
    recorder.add_frame(_frame())
    recorder.stop()
    assert _records(recorder.data_path)[0]["settings"] == clean


R = session_recorder.REDACTED


@pytest.mark.parametrize("text, expected", [
    # Removed: folders, whole URLs, the user's name
    ('File "C:\\Users\\someone\\app\\Camera.py", line 3', 'File "Camera.py", line 3'),
    ("C:\\Users\\Some One\\OneDrive - Some Company\\app\\Camera.py failed", "Camera.py failed"),
    ("\\\\server\\share\\someone\\clip.mp4", "clip.mp4"),
    ("loaded /home/x/models/hand.task", "loaded hand.task"),
    ("opened http://admin:secret@10.0.0.2/video", f"opened http://{R}"),
    ("opened http://192.168.1.5:8080/video?user=admin&pwd=secret now", f"opened http://{R} now"),
    ("rtsp://admin:p@ss@cam/stream", f"rtsp://{R}"),
    ("Hello SOMEONE.", "Hello <user>."),
    # Kept: ordinary text
    ("tracking.max_hands set to 2", "tracking.max_hands set to 2"),
    ("Touch/Index/Knuckles on 2026/10/05", "Touch/Index/Knuckles on 2026/10/05"),
    ("Tracking 30 fps / camera 60 fps", "Tracking 30 fps / camera 60 fps"),
    ("someones and handsomeone stay", "someones and handsomeone stay"),
])
def test_log_text_is_safe_to_share(monkeypatch, text, expected):
    monkeypatch.setattr(session_recorder, "_user_names", session_recorder.names_pattern(["someone", "max", ""]))
    assert redact_text(text) == expected


class _BrokenWriter:
    def __call__(self, path, fps, size):
        return self

    def write(self, image):
        raise OSError("disk full")

    def release(self):
        pass


def test_a_failing_video_never_blocks_stop_or_tracking(tmp_path, monkeypatch):
    monkeypatch.setattr(session_recorder, "VIDEO_QUEUE_FRAMES", 2)
    recorder = SessionRecorder(tmp_path, {}, video=True, name="s", clock=_Clock(), video_writer=_BrokenWriter())
    for i in range(100):
        recorder.add_frame(_frame(), 100.0 + i / 30)
    stopper = threading.Thread(target=recorder.stop)
    stopper.start()
    stopper.join(5)
    assert not stopper.is_alive()
    records = _records(recorder.data_path)
    assert records[-1]["type"] == "end" and records[-1]["video_ok"] is False and records[-1]["frames"] == 100


def test_stop_while_tracking_keeps_going_ends_the_file_cleanly(tmp_path):
    recorder = SessionRecorder(tmp_path, {}, video=True, name="s")
    running = threading.Event()
    running.set()

    def track():
        while running.is_set():
            recorder.add_frame(_frame(hands=[_hand()]))
            recorder.log("a line\n")
            recorder.settings_changed({"calibration.scale": 1.0})

    tracker = threading.Thread(target=track)
    tracker.start()
    time.sleep(0.2)
    recorder.stop()
    running.clear()
    tracker.join(5)
    records = _records(recorder.data_path)
    assert records[0]["type"] == "session" and records[-1]["type"] == "end"
    assert [r for r in records if r["type"] == "end"] == [records[-1]]
    frames = [r for r in records if r["type"] == "frame"]
    assert records[-1]["frames"] == len(frames) > 0 and records[-1]["video_ok"] is True
