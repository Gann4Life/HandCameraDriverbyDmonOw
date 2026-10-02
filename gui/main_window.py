"""
Main window: camera and 3D previews on the left, settings / live readout / log
on the right, tracking controls in the toolbar and health in the status bar.
"""
import copy
import json
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QLabel, QMainWindow, QMessageBox, QSplitter, QTabWidget

from Camera import HandTracker
from gui.camera_view import CameraView
from gui.environments import relaunch, wilor_installed_here, wilor_launcher
from gui.hand_view_3d import HandView3D
from gui.live_panel import LivePanel
from gui.log_panel import LogPanel, capture_output
from gui.settings_panel import SettingsPanel
from gui.settings_schema import SETTINGS
from gui.style import MUTED_COLOR, OK_COLOR, WARNING_COLOR
from gui.tracker_worker import TrackerWorker
from hand_data import TrackingFrame
from utils.config_utils import get_value, set_value

APP_TITLE = "Hand Camera Driver"
REFRESH_MS = 16  # preview refresh, independent of the tracking rate


class MainWindow(QMainWindow):
    def __init__(self, config_path: str):
        super().__init__()
        self.config_path = config_path
        self.config = HandTracker.load_config(config_path)
        self.saved_config = copy.deepcopy(self.config)
        self.worker: Optional[TrackerWorker] = None
        self._shown_sequence = -1

        self.log_panel = LogPanel()
        capture_output(self.log_panel)

        self.camera_view = CameraView()
        self.hand_view = HandView3D()
        self.settings_panel = SettingsPanel(self.config, SETTINGS)
        self.settings_panel.changed.connect(self._on_settings_changed)
        self.live_panel = LivePanel()

        previews = QSplitter(Qt.Vertical)
        previews.addWidget(self.camera_view)
        previews.addWidget(self.hand_view)
        previews.setSizes([460, 340])
        tabs = QTabWidget()
        tabs.addTab(self.settings_panel, "Settings")
        tabs.addTab(self.live_panel, "Hands")
        tabs.addTab(self.log_panel, "Log")
        main = QSplitter(Qt.Horizontal)
        main.addWidget(previews)
        main.addWidget(tabs)
        main.setStretchFactor(0, 3)
        main.setStretchFactor(1, 2)
        main.setSizes([840, 560])
        self.setCentralWidget(main)

        self._build_actions()
        self._build_status_bar()
        self._update_title()
        self.resize(1400, 860)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(REFRESH_MS)
        self.start_tracking()

    # ----- toolbar and menus

    def _build_actions(self):
        toolbar = self.addToolBar("Tracking")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextOnly)

        self.start_action = QAction("Start", self)
        self.start_action.setShortcut(QKeySequence("F5"))
        self.start_action.triggered.connect(self._toggle_tracking)
        swap_action = QAction("Swap hands", self)
        swap_action.setShortcut(QKeySequence("S"))
        swap_action.setToolTip("Left and right are the wrong way round (S)")
        swap_action.triggered.connect(lambda: self.settings_panel.set_value(
            "tracking.swap_hands", not get_value(self.config, "tracking.swap_hands", False)))
        self.depth_action = QAction("WiLoR depth", self)
        self.depth_action.setCheckable(True)
        self.depth_action.setToolTip("Heavy 3D hand model for steadier depth (experimental)")
        self.depth_action.triggered.connect(lambda checked: self.settings_panel.set_value(
            "tracking.depth_source", "wilor" if checked else "mediapipe"))
        save_action = QAction("Save", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_config)
        revert_action = QAction("Revert", self)
        revert_action.setToolTip("Discard changes since the last save")
        revert_action.triggered.connect(self.revert_config)

        for action in (self.start_action, swap_action, self.depth_action):
            toolbar.addAction(action)
        toolbar.addSeparator()
        toolbar.addAction(save_action)
        toolbar.addAction(revert_action)

        view_menu = self.menuBar().addMenu("View")
        for text, widget in (("Camera view", self.camera_view), ("3D view", self.hand_view)):
            action = QAction(text, self, checkable=True, checked=True)
            action.toggled.connect(widget.setVisible)
            view_menu.addAction(action)
        landmarks = QAction("Hand skeleton on camera", self, checkable=True, checked=True)
        landmarks.toggled.connect(self._set_show_landmarks)
        view_menu.addAction(landmarks)
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(save_action)
        file_menu.addAction(revert_action)
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close)

    def _set_show_landmarks(self, show: bool):
        self.camera_view.show_landmarks = show
        self.camera_view.update()

    def _build_status_bar(self):
        self.fps_label = QLabel()
        self.latency_label = QLabel()
        self.driver_label = QLabel()
        self.depth_label = QLabel()
        for label in (self.fps_label, self.latency_label, self.driver_label, self.depth_label):
            self.statusBar().addPermanentWidget(label)
            label.setContentsMargins(8, 0, 8, 0)

    # ----- tracking lifecycle

    def start_tracking(self):
        if self.worker is not None:
            return
        self._sync_depth_action()
        self.camera_view.set_message("Opening the camera...")
        self.worker = TrackerWorker(self.config, self)
        self.worker.tracking_started.connect(self._on_tracking_started)
        self.worker.tracking_stopped.connect(self._on_tracking_stopped)
        self.worker.settings_applied.connect(self._on_settings_applied)
        self._shown_sequence = -1
        self.start_action.setText("Stop")
        self.worker.start()

    def stop_tracking(self, wait: bool = False):
        if self.worker is None:
            return
        self.worker.request_stop()
        if wait:
            self.worker.wait(10000)

    def _toggle_tracking(self):
        if self.worker is None:
            self.start_tracking()
        else:
            self.stop_tracking()

    def _on_tracking_started(self):
        self.statusBar().showMessage("Tracking", 3000)

    def _on_tracking_stopped(self, reason: str):
        self.worker = None
        self.start_action.setText("Start")
        self.camera_view.set_message(reason or "Tracking stopped. Press Start (F5) to resume.")
        self.hand_view.clear()
        self.live_panel.clear()
        self.fps_label.setText("")
        self.latency_label.setText("")
        self._set_label(self.driver_label, "Stopped", MUTED_COLOR)
        if reason:
            self.statusBar().showMessage(reason)

    def _on_settings_changed(self, changes: dict):
        if changes.get("tracking.depth_source") == "wilor" and not wilor_installed_here():
            self._switch_to_wilor_environment()
            return
        self._update_title()
        self._sync_depth_action()
        if self.worker is not None:
            self.worker.apply_settings(changes)

    def _on_settings_applied(self, effective: dict):
        """The tracker may not take a value as asked (e.g. WiLoR not installed)."""
        rejected = {k: v for k, v in effective.items() if get_value(self.config, k) != v}
        if not rejected:
            return
        for key, value in rejected.items():
            set_value(self.config, key, value)
        self.settings_panel.refresh()
        self._sync_depth_action()
        self._update_title()
        if "tracking.depth_source" in rejected:
            QMessageBox.information(self, "WiLoR depth", "WiLoR could not be loaded, so standard depth stays on. "
                                                         "The Log tab has the details.")

    def _switch_to_wilor_environment(self):
        """
        WiLoR was asked for but isn't installed in this environment: restart the
        app in the one where it is, or explain how to install it.
        """
        set_value(self.config, "tracking.depth_source", "mediapipe")
        self.settings_panel.refresh()
        self._sync_depth_action()
        self._update_title()
        launcher = wilor_launcher()
        if launcher is None:
            QMessageBox.information(self, "WiLoR depth", "WiLoR is not installed. Run install-depth.bat (release "
                                                         "package) or see INSTALL.md section 10, then try again.")
            return
        answer = QMessageBox.question(self, "WiLoR depth",
                                      "WiLoR is installed in its own environment. Restart the app there with "
                                      "WiLoR depth on?\n\nYour settings are saved first.")
        if answer != QMessageBox.Yes:
            return
        set_value(self.config, "tracking.depth_source", "wilor")
        if not self.save_config():
            return
        self.stop_tracking(wait=True)
        relaunch(*launcher, self.config_path)
        self.close()

    def _sync_depth_action(self):
        self.depth_action.setChecked(get_value(self.config, "tracking.depth_source") == "wilor")

    # ----- refresh

    def _refresh(self):
        if self.worker is None:
            return
        sequence, frame = self.worker.latest()
        if frame is None or sequence == self._shown_sequence:
            return
        self._shown_sequence = sequence
        self.camera_view.set_frame(frame)
        self.hand_view.set_frame(frame)
        self.live_panel.set_frame(frame)
        self._update_status(frame)

    def _update_status(self, frame: TrackingFrame):
        self.fps_label.setText(f"Tracking {frame.tracking_fps:.0f} fps  ·  camera {frame.camera_fps:.0f} fps")
        latency = sum(frame.timings_ms.values())
        self.latency_label.setText(f"{latency:.0f} ms per frame")
        if frame.driver_connected:
            self._set_label(self.driver_label, "SteamVR driver connected", OK_COLOR)
        else:
            self._set_label(self.driver_label, "Waiting for SteamVR driver", WARNING_COLOR)
        wilor = frame.depth_label != "MediaPipe"
        self._set_label(self.depth_label, f"Depth: {frame.depth_label}", WARNING_COLOR if wilor else MUTED_COLOR)

    @staticmethod
    def _set_label(label: QLabel, text: str, color):
        label.setText(text)
        label.setStyleSheet(f"color: {color.name()}")

    # ----- config file

    @property
    def dirty(self) -> bool:
        return self.config != self.saved_config

    def _update_title(self):
        self.setWindowTitle(f"{APP_TITLE}{' *' if self.dirty else ''}")

    def save_config(self) -> bool:
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=2)
                f.write("\n")
        except OSError as e:
            QMessageBox.critical(self, "Save", f"Could not save {self.config_path}:\n{e}")
            return False
        self.saved_config = copy.deepcopy(self.config)
        self._update_title()
        self.statusBar().showMessage(f"Saved {self.config_path}", 3000)
        return True

    def revert_config(self):
        if not self.dirty:
            return
        self.config.clear()
        self.config.update(copy.deepcopy(self.saved_config))
        self.settings_panel.refresh()
        self._update_title()
        if self.worker is not None:
            # Simplest way to apply everything at once: restart with the saved settings
            self.stop_tracking(wait=True)
            QTimer.singleShot(0, self.start_tracking)

    def closeEvent(self, event):
        if self.dirty:
            answer = QMessageBox.question(self, APP_TITLE, "Save your changes before closing?",
                                          QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            if answer == QMessageBox.Cancel or (answer == QMessageBox.Save and not self.save_config()):
                event.ignore()
                return
        self._timer.stop()
        self.stop_tracking(wait=True)
        event.accept()
