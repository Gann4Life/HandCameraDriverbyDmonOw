"""
Records a tracking session for later study and replay: the camera video and,
next to it, what the tracker made of every frame.

The data file is JSON Lines, one record per line, each with a "type":
- "session": the first line: format, app version, settings, frame size, video file name
- "frame": one per tracked frame: time, rates, and per hand its landmarks, pose and controls
- "settings": settings changed while recording
- "log": text the app printed
- "end": the last line: frame count, frames left out of the video, whether the video was written

The video runs at the camera's frame rate and plays in real time: each tracked frame is
placed by its time, and shows until the next one. A frame record's "video_frame" is its
picture's index in the video.

Writing happens on background threads behind bounded queues, so a slow disk never
slows tracking: when the video falls behind, frames are left out of the video (their
data is kept, with "video_frame": null) instead of delaying the next frame.
"""
import copy
import dataclasses
import getpass
import json
import math
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np

from hand_data import TrackedHand, TrackingFrame
from version import APP_VERSION

FORMAT = 1
OUTPUT_FOLDER = ".output"
VIDEO_QUEUE_FRAMES = 60  # about 2 s of 30 fps video waiting for the disk
DATA_QUEUE_RECORDS = 5000
DEFAULT_VIDEO_FPS = 30.0
MAX_LOG_LINE = 4000  # characters
DECIMALS = 5
REDACTED = "<redacted>"


def output_dir(app_dir: Path) -> Path:
    """Where recordings go: .output next to the app."""
    return app_dir / OUTPUT_FOLDER


def _rounded(value: Any) -> Any:
    """value as plain JSON types, floats rounded; NaN and infinity become null."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_rounded(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _rounded(v) for k, v in value.items()}
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), DECIMALS) if math.isfinite(value) else None
    return value


def _looks_private(text: str) -> bool:
    """A file path or a URL (which may carry a user name or password), not a setting's value."""
    return "://" in text or "\\" in text or "/" in text or (len(text) > 1 and text[1] == ":")


def redact(settings: Any) -> Any:
    """A copy of settings with every path or URL replaced, so a recording can be shared."""
    if isinstance(settings, dict):
        return {k: redact(v) for k, v in settings.items()}
    if isinstance(settings, (list, tuple)):
        return [redact(v) for v in settings]
    if isinstance(settings, str) and _looks_private(settings):
        return REDACTED
    return settings


# A whole URL: credentials hide in the user part and in the query alike
_URL = re.compile(r"\b([A-Za-z][A-Za-z0-9+.-]*)://\S+")
# The folders of a path, which hold the user's name (the file name stays). Windows folders may
# have spaces, so a Windows path's folders run to its last backslash.
_FOLDERS = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/](?:[^\\/\r\n\"'<>|]*[\\/])*"  # C:\a b\c\
                      r"|\\\\(?:[^\\\r\n\"'<>|]*\\)+"                       # \\server\share\
                      r"|(?<![^\s\"'(=])/(?:[^/\s\"'<>|]+/)+")             # /home/a/
_user_names: Optional[re.Pattern] = None


def names_pattern(names) -> re.Pattern:
    """names as whole words, any case (Windows paths ignore case); shorter than 3 letters, never."""
    names = sorted((n for n in set(names) if n and len(n) >= 3), key=len, reverse=True)
    if not names:
        return re.compile(r"(?!)")
    return re.compile(r"(?<![A-Za-z0-9_])(?:" + "|".join(map(re.escape, names)) + r")(?![A-Za-z0-9_])",
                      re.IGNORECASE)


def _user_pattern() -> re.Pattern:
    """The user's login and profile folder names, found once."""
    global _user_names
    if _user_names is None:
        names = [Path.home().name]
        try:
            names.append(getpass.getuser())
        except Exception:  # no user name in the environment
            pass
        _user_names = names_pattern(names)
    return _user_names


def redact_text(text: str) -> str:
    """text without URLs, folders or the user's name, so it can be shared (tracebacks print full paths)."""
    text = _URL.sub(lambda m: f"{m.group(1)}://{REDACTED}", text)
    text = _FOLDERS.sub("", text)
    return _user_pattern().sub("<user>", text)


def hand_record(hand: TrackedHand) -> Dict[str, Any]:
    """One hand of a frame record: image landmarks, camera-space joints, the pose sent and the controls."""
    data = hand.data
    record = {
        "side": data.hand_type,
        "predicted": bool(getattr(hand, "predicted", False)),
        "sendable": data.is_sendable_pose(),
        "position": data.position,
        "rotation": data.rotation,
        "gesture": data.gesture,
        "trigger": data.trigger_value,
        "grip": data.grip_value,
        "curls": data.finger_curls,
        "landmarks": data.landmarks,
        "camera_position": hand.camera_position,
        "camera_points": hand.camera_points,
        "features": dataclasses.asdict(hand.features) if hand.features is not None else None,
        "gesture_scores": hand.gesture_scores,
    }
    return _rounded(record)


def frame_record(frame: TrackingFrame, index: int, t: float, video_frame: Optional[int]) -> Dict[str, Any]:
    """What the tracker made of one frame; t in seconds since the recording started."""
    return {
        "type": "frame",
        "i": index,
        "t": round(t, 4),
        "video_frame": video_frame,
        "tracking_fps": _rounded(frame.tracking_fps),
        "camera_fps": _rounded(frame.camera_fps),
        "timings_ms": _rounded(frame.timings_ms),
        "driver_connected": frame.driver_connected,
        "depth": frame.depth_label,
        "hands": [hand_record(hand) for hand in frame.hands],
    }


def session_name(now: Optional[datetime] = None) -> str:
    """session-<date>-<time>, so recordings sort by when they were made."""
    return f"session-{(now or datetime.now()).strftime('%Y%m%d-%H%M%S')}"


class SessionRecorder:
    """
    One recording. add_frame is called from the tracking thread for every frame;
    log and settings_changed from any thread. stop() ends it.
    """

    def __init__(self, folder: Path, settings: Dict[str, Any], video: bool = True,
                 name: Optional[str] = None, clock: Callable[[], float] = time.perf_counter,
                 video_writer: Optional[Callable[[Path, float, Tuple[int, int]], Any]] = None):
        """
        Args:
            settings: The config in use; paths, URLs and the saved presets are left out
            video: False records the data file only
            name: The files' name without extension; session-<date>-<time> by default
            clock: Seconds, the clock add_frame's times come from
            video_writer: (path, fps, (width, height)) -> an object with write(image) and release();
                an mp4 file by default
        """
        folder.mkdir(parents=True, exist_ok=True)
        self._video_writer = video_writer or open_mp4
        name = name or session_name()
        self.video_path: Optional[Path] = folder / f"{name}.mp4" if video else None
        self.data_path = folder / (f"{name}.mp4.log" if video else f"{name}.log")
        self._clock = clock
        self._start: Optional[float] = None
        # The saved presets are left out: only the settings in use matter
        self._settings = redact(copy.deepcopy({k: v for k, v in settings.items() if k != "presets"}))
        self._frames = 0
        self._video_frames = 0
        self._video_dropped = 0
        self._data_dropped = 0
        self._stopped = False
        self._end_t = 0.0
        self._video_ok = True
        self._log_pending = ""
        self._lock = threading.Lock()
        self._data: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(DATA_QUEUE_RECORDS)
        self._video: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(VIDEO_QUEUE_FRAMES)
        self._video_size: Optional[tuple] = None
        # A fixed rate is all mp4 holds: the camera's, so no picture has to share a slot
        self._video_fps = float(self._settings.get("camera", {}).get("fps") or DEFAULT_VIDEO_FPS)
        self._data_file = open(self.data_path, "w", encoding="utf-8", newline="\n")
        self._data_thread = threading.Thread(target=self._write_data, name="recorder-data")
        self._video_thread = threading.Thread(target=self._write_video, name="recorder-video") if video else None
        self._data_thread.start()
        if self._video_thread is not None:
            self._video_thread.start()

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def elapsed_s(self) -> float:
        return 0.0 if self._start is None else self._clock() - self._start

    def add_frame(self, frame: TrackingFrame, t: Optional[float] = None) -> None:
        """Record one tracked frame; t is when the tracker got it from the camera (the clock's time)."""
        with self._lock:
            if self._stopped:
                return
            t = self._clock() if t is None else t
            if self._start is None:
                self._start = t
                self._put(self._session_record(frame))
            video_frame = None
            if self.video_path is not None:
                # Placed by its time, so the video plays in real time however fast tracking ran;
                # never before the slot after the last picture, so pictures keep their order
                slot = max(self._video_frames, round((t - self._start) * self._video_fps))
                if self._queue_video(frame.frame_bgr, slot):
                    video_frame = slot
                    self._video_frames = slot + 1
            self._put(frame_record(frame, self._frames, t - self._start, video_frame))
            self._frames += 1

    def log(self, text: str) -> None:
        """Printed text, recorded a whole line at a time (print writes a line's text and its newline apart)."""
        with self._lock:
            *lines, pending = (self._log_pending + text).split("\n")
            # Text that never ends its line (a progress bar redrawn with \r) stays bounded
            self._log_pending = pending[-MAX_LOG_LINE:]
        for line in lines:
            if line.strip():
                self._event({"type": "log", "text": redact_text(line)})

    def settings_changed(self, changes: Dict[str, Any]) -> None:
        self._event({"type": "settings", "changes": _rounded(redact(changes))})

    def stop(self, wait: bool = True) -> None:
        """End the recording. Without wait, the files finish writing in the background."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            self._end_t = round(self.elapsed_s, 4)
        # Nothing is queued after _stopped, so the ends go last. Outside the lock: the writers
        # always empty their queues, but a full one may take a moment.
        if self._video_thread is not None:
            self._video.put(None)
        self._data.put(None)
        if wait:
            self.join()

    def join(self):
        self._data_thread.join()
        if self._video_thread is not None:
            self._video_thread.join()

    # ----- tracking thread side

    def _event(self, record: Dict[str, Any]):
        with self._lock:
            if self._stopped or self._start is None:
                return  # nothing before the first frame: the session line comes first
            self._put({"t": round(self.elapsed_s, 4), **record})

    def _session_record(self, frame: TrackingFrame) -> Dict[str, Any]:
        height, width = frame.frame_bgr.shape[:2]
        return {"type": "session", "format": FORMAT, "app_version": APP_VERSION,
                "started": datetime.now().isoformat(timespec="seconds"),
                "video": self.video_path.name if self.video_path else None, "video_fps": self._video_fps,
                "width": width, "height": height, "hfov_deg": _rounded(frame.hfov_deg),
                "camera_fps": _rounded(frame.camera_fps), "settings": _rounded(self._settings)}

    def _put(self, record: Dict[str, Any]):
        try:
            self._data.put_nowait(record)
        except queue.Full:
            self._data_dropped += 1

    def _queue_video(self, image: np.ndarray, slot: int) -> bool:
        if self._video_size is None:
            self._video_size = (image.shape[1], image.shape[0])
        try:
            self._video.put_nowait((image, slot))
            return True
        except queue.Full:
            self._video_dropped += 1
            return False

    # ----- writer threads
    # A writer that fails (a full disk) says so once and keeps emptying its queue to the
    # end, so nothing waiting to queue (tracking, stop) is ever blocked by it.

    def _write_data(self):
        failed = False
        with self._data_file as f:
            while True:
                record = self._data.get()
                if record is None:
                    if self._video_thread is not None:
                        self._video_thread.join()  # the end line says whether the video was written
                    record = self._end_record()
                if not failed:
                    try:
                        f.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
                    except (OSError, TypeError, ValueError) as e:
                        failed = True
                        print(f"The recording stopped writing its data: {e}")
                if record["type"] == "end":
                    return

    def _end_record(self) -> Dict[str, Any]:
        return {"type": "end", "t": self._end_t, "frames": self._frames, "video_ok": self._video_ok,
                "video_frames": self._video_frames, "video_dropped": self._video_dropped,
                "data_dropped": self._data_dropped}

    def _write_video(self):
        writer = None
        written = 0
        previous: Optional[np.ndarray] = None
        while True:
            item = self._video.get()
            if item is None:
                break
            if not self._video_ok:
                continue
            image, slot = item
            try:
                if writer is None:
                    writer = self._open_video()
                if image.shape[1::-1] != self._video_size:
                    image = cv2.resize(image, self._video_size)  # the camera came back at another size
                # The time between tracked frames shows the last picture, as the tracker saw it
                while previous is not None and written < slot:
                    writer.write(previous)
                    written += 1
                writer.write(image)
                written += 1
                previous = image
            except Exception as e:  # cv2.error, OSError: the data file goes on without the video
                self._video_ok = False
                print(f"The recording stopped writing its video: {e}")
        if writer is not None:
            try:
                writer.release()
            except Exception as e:
                self._video_ok = False
                print(f"Could not finish the recording's video: {e}")

    def _open_video(self):
        writer = self._video_writer(self.video_path, self._video_fps, self._video_size)
        if hasattr(writer, "isOpened") and not writer.isOpened():
            raise OSError(f"could not create {self.video_path.name}")
        return writer


def open_mp4(path: Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
