"""
Hand Camera Driver with its desktop interface: run `python app.py`.
Camera.py stays the command-line version.
"""
import sys

from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from gui.style import apply_dark_theme


def main() -> int:
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
        window = MainWindow(sys.argv[1] if len(sys.argv) > 1 else "config.json")
        window.show()
        return app.exec()
    finally:
        if timer_raised:
            ctypes.windll.winmm.timeEndPeriod(1)


if __name__ == "__main__":
    sys.exit(main())
