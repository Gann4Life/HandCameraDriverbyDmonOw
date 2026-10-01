"""
Camera utility functions for video capture and processing.
"""
import cv2
import time
from typing import List, Optional, Tuple


BACKEND_CHOICES = {
    "auto": [("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW), ("ANY", cv2.CAP_ANY)],
    "msmf": [("MSMF", cv2.CAP_MSMF)],
    "dshow": [("DSHOW", cv2.CAP_DSHOW)],
    "any": [("ANY", cv2.CAP_ANY)],
}

FIRST_FRAME_POLL = 0.05


class CameraCapture:
    """Handles camera capture with configuration options."""

    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480,
                 fps: int = 30, flip_horizontal: bool = True,
                 backend: str = "auto", first_frame_timeout: float = 5.0,
                 rotate_180: bool = False):
        """
        Initialize camera capture.

        Args:
            device_id: Camera device ID
            width: Frame width
            height: Frame height
            fps: Target frames per second
            flip_horizontal: Whether to flip the frame horizontally
            backend: One of "auto", "msmf", "dshow", "any"
            first_frame_timeout: Seconds to wait for a real frame when probing
            rotate_180: Rotate the frame 180 degrees (camera mounted upside down)
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.flip_horizontal = flip_horizontal
        self.rotate_180 = rotate_180
        self.backend = backend
        self.first_frame_timeout = first_frame_timeout
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.consecutive_read_failures = 0
        self.last_frame_time = time.time()
        self.active_backend: Optional[str] = None

    def backend_candidates(self) -> List[Tuple[str, int]]:
        """
        Resolve the configured backend into an ordered list of candidates.

        Returns:
            List of (name, cv2 backend constant) pairs to try in order
        """
        key = str(self.backend or "auto").lower()
        if key not in BACKEND_CHOICES:
            print(f"Warning: unknown backend '{self.backend}', using auto")
            key = "auto"
        return BACKEND_CHOICES[key]

    def _apply_properties(self, cap: cv2.VideoCapture):
        """Request the configured capture properties from an opened device."""
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps > 0:
            cap.set(cv2.CAP_PROP_FPS, self.fps)

    def _read_first_frame(self, cap: cv2.VideoCapture,
                          timeout: Optional[float] = None) -> Optional[cv2.Mat]:
        """
        Poll an opened device until it yields a frame.

        Virtual cameras routinely report isOpened() while delivering nothing,
        so a successful open is never treated as success on its own.

        Args:
            cap: An opened capture device
            timeout: Seconds to keep polling; defaults to first_frame_timeout

        Returns:
            The first decoded frame, or None if none arrived in time
        """
        deadline = time.time() + (self.first_frame_timeout if timeout is None else timeout)
        while time.time() < deadline:
            ret, frame = cap.read()
            if ret and frame is not None:
                return frame
            time.sleep(FIRST_FRAME_POLL)
        return None

    def _report_frame_content(self, frame: cv2.Mat):
        """
        Warn when the first frame carries no image.

        A virtual camera with no live source still opens and still delivers
        frames, but they are a single flat colour. Detecting that here is the
        difference between a clear diagnosis and a blank window that looks
        like a tracking bug.

        Args:
            frame: The first frame received from the device
        """
        spatial = frame.reshape(-1, frame.shape[2]).std(axis=0)
        deviation = float(spatial.max())

        if deviation < 1.0:
            pixel = tuple(int(v) for v in frame[0, 0])
            print("WARNING: this device delivered a perfectly flat frame.")
            print(f"         Every pixel is BGR{pixel} - there is no camera image.")
            print("         The driver opened fine but has no live source, so the")
            print("         preview will be a solid block of colour and MediaPipe")
            print("         cannot detect hands. Check the phone app is actually")
            print("         streaming and is on the same network as this PC.")
        elif deviation < 10.0:
            print(f"Note: first frame has very little contrast (std {deviation:.2f}). "
                  f"The image may be blank or badly underexposed.")

    def start(self) -> bool:
        """
        Start camera capture, falling back across backends until frames arrive.

        Returns:
            True if a backend produced a real frame, False otherwise
        """
        try:
            problems = []

            for name, backend in self.backend_candidates():
                cap = cv2.VideoCapture(self.device_id, backend)

                if not cap.isOpened():
                    cap.release()
                    problems.append(f"{name}: could not open device index {self.device_id}")
                    continue

                self._apply_properties(cap)
                frame = self._read_first_frame(cap)

                if frame is None:
                    cap.release()
                    problems.append(
                        f"{name}: opened but delivered no frame "
                        f"within {self.first_frame_timeout:.1f}s"
                    )
                    continue

                self.cap = cap
                self.active_backend = name
                self.consecutive_read_failures = 0
                self.last_frame_time = time.time()
                self.fps_start_time = time.time()
                self.frame_count = 0

                actual_height, actual_width = frame.shape[0], frame.shape[1]
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                fps_text = f"{actual_fps:.0f}" if actual_fps > 0 else "unknown"

                print(f"Camera started: index {self.device_id} via {name}, "
                      f"{actual_width}x{actual_height} @ {fps_text}fps")
                if (actual_width, actual_height) != (self.width, self.height):
                    print(f"Note: device is delivering "
                          f"{actual_width}x{actual_height}, not the requested "
                          f"{self.width}x{self.height}")

                self._report_frame_content(frame)

                return True

            print(f"Error: could not start camera index {self.device_id}")
            for problem in problems:
                print(f"  - {problem}")
            return False

        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """
        Read a frame from the camera.

        A failed read is reported as False, but transient dropouts are counted
        rather than treated as fatal so a virtual camera restarting its stream
        does not end the session.

        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()

        if not ret or frame is None:
            self.consecutive_read_failures += 1
            return False, None

        self.consecutive_read_failures = 0
        self.last_frame_time = time.time()

        # Undo an upside-down mount before any mirroring
        if self.rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Flip frame if configured
        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)

        # Update FPS counter
        self.frame_count += 1
        elapsed_time = time.time() - self.fps_start_time
        if elapsed_time > 1.0:
            self.current_fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.fps_start_time = time.time()

        return True, frame

    def is_stalled(self, timeout: float = 2.0) -> bool:
        """
        Check whether frame delivery has been failing for too long.

        Measured in time rather than failed reads: a broken MSMF stream fails
        each read in ~12 ms, so a read count says almost nothing about how
        long the camera has actually been gone.

        Args:
            timeout: Seconds without a frame to tolerate

        Returns:
            True if the camera should be considered dead
        """
        return self.consecutive_read_failures > 0 and time.time() - self.last_frame_time > timeout

    def reopen(self) -> bool:
        """
        Release and reopen the device, falling back across backends again.

        A network camera that drops out often leaves the MSMF reader in a
        permanently failed state, so retrying reads on the old handle is not
        enough; the device has to be opened afresh.

        Returns:
            True if the camera is delivering frames again
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        return self.start()

    def get_fps(self) -> float:
        """Get current FPS."""
        return self.current_fps

    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("Camera released")

    def is_opened(self) -> bool:
        """Check if camera is opened."""
        return self.cap is not None and self.cap.isOpened()