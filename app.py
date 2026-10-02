"""
Hand Camera Driver with its desktop interface: run `python app.py`.
Camera.py stays the command-line version.
"""
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from gui.style import apply_dark_theme

FROZEN = getattr(sys, "frozen", False)


def _log_to_file():
    """
    The packaged app has no console: keep its output in a file next to it, for
    bug reports (the Log tab shows the same while it runs).
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    try:
        log = open(Path(sys.executable).parent / "HandCameraDriver.log", "w", encoding="utf-8", buffering=1)
    except OSError:
        import os
        log = open(os.devnull, "w")
    sys.stdout = sys.stdout or log
    sys.stderr = sys.stderr or log


def main() -> int:
    if FROZEN:
        _log_to_file()
    # Windows' default 15.6 ms timer tick puts a floor under every sleep and
    # wait in the tracking loop
    timer_raised = False
    if sys.platform == "win32":
        import ctypes
        try:
            timer_raised = ctypes.windll.winmm.timeBeginPeriod(1) == 0
        except (OSError, AttributeError):
            pass
    try:
        app = QApplication(sys.argv)
        app.setApplicationName("Hand Camera Driver")
        apply_dark_theme(app)
        # The packaged app keeps its config next to the exe, wherever it is started from
        default_config = str(Path(sys.executable).parent / "config.json") if FROZEN else "config.json"
        window = MainWindow(sys.argv[1] if len(sys.argv) > 1 else default_config)
        window.show()
        return app.exec()
    finally:
        if timer_raised:
            ctypes.windll.winmm.timeEndPeriod(1)


if __name__ == "__main__":
    sys.exit(main())
