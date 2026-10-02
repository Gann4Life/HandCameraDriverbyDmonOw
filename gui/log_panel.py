"""The tracker's console output, shown inside the app."""
import sys

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import QPlainTextEdit


class LogStream(QObject):
    """
    File-like stand-in for stdout/stderr: keeps writing to the original
    stream (if any) and forwards the text to the GUI thread.
    """

    text_written = Signal(str)

    def __init__(self, original, parent=None):
        super().__init__(parent)
        self._original = original

    def write(self, text: str):
        if self._original is not None:
            try:
                self._original.write(text)
            except (OSError, ValueError):
                pass
        if text:
            self.text_written.emit(text)

    def flush(self):
        if self._original is not None:
            try:
                self._original.flush()
            except (OSError, ValueError):
                pass


class LogPanel(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(3000)
        self.setFont(QFont("Consolas", 9))

    def append_text(self, text: str):
        self.moveCursor(QTextCursor.End)
        self.insertPlainText(text)
        self.moveCursor(QTextCursor.End)


def capture_output(panel: LogPanel):
    """Route print() and tracebacks from every thread into the panel."""
    for name in ("stdout", "stderr"):
        stream = LogStream(getattr(sys, name), panel)
        stream.text_written.connect(panel.append_text)
        setattr(sys, name, stream)
