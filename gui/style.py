"""Colours and theme shared by the GUI."""
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

HAND_COLORS = {"left": QColor(80, 200, 255), "right": QColor(255, 160, 60)}
# A lost hand moved along its predicted path, either side
PREDICTED_COLOR = QColor(200, 130, 255)
WARNING_COLOR = QColor(255, 170, 0)
OK_COLOR = QColor(110, 210, 110)
MUTED_COLOR = QColor(140, 140, 140)


def apply_dark_theme(app: QApplication):
    app.setStyle("Fusion")
    palette = QPalette()
    base, window, text = QColor(30, 30, 33), QColor(40, 40, 44), QColor(225, 225, 225)
    palette.setColor(QPalette.Window, window)
    palette.setColor(QPalette.WindowText, text)
    palette.setColor(QPalette.Base, base)
    palette.setColor(QPalette.AlternateBase, window)
    palette.setColor(QPalette.ToolTipBase, window)
    palette.setColor(QPalette.ToolTipText, text)
    palette.setColor(QPalette.Text, text)
    palette.setColor(QPalette.Button, window)
    palette.setColor(QPalette.ButtonText, text)
    palette.setColor(QPalette.Highlight, QColor(60, 130, 200))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.PlaceholderText, MUTED_COLOR)
    for role in (QPalette.Text, QPalette.ButtonText, QPalette.WindowText):
        palette.setColor(QPalette.Disabled, role, MUTED_COLOR)
    app.setPalette(palette)
