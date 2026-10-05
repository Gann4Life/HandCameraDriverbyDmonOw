"""
Main window: camera and 3D previews on the left, settings / live readout / log
on the right, tracking controls in the toolbar and health in the status bar.

Settings come in two kinds, on two tabs. Preset settings depend on where the
camera is; changes to them stay unsaved until Save preset. App settings
(camera device, tracking model, depth, driver connection...) are the same for
every preset and are saved as soon as they change.
"""
import json
from pathlib import Path
from typing import Dict, Optional

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import QLabel, QMainWindow, QMessageBox, QSplitter, QTabWidget, QVBoxLayout, QWidget

import addons
import presets
from Camera import HandTracker
from gui.addons_dialog import AddonsDialog
from gui.camera_view import CameraView
from gui.environments import relaunch, wilor_installed_here, wilor_launcher
from gui.gesture_calibration import GestureCalibrationDialog
from gui.hand_view_3d import HandView3D
from gui.live_panel import LivePanel
from gui.log_panel import LogPanel, capture_output
from gui.preset_bar import PresetBar
from gui.settings_panel import SettingsPanel
from gui.settings_schema import SETTINGS, Setting
from gui.style import MUTED_COLOR, OK_COLOR, RECORDING_COLOR, WARNING_COLOR
from gui.tracker_worker import TrackerWorker
from hand_data import TrackingFrame
from session_recorder import OUTPUT_FOLDER, SessionRecorder, output_dir
from utils.config_utils import get_value, set_value
from version import APP_VERSION

APP_TITLE = "Hand Camera Driver"
REFRESH_MS = 16  # preview refresh, independent of the tracking rate
RECENTER_DELAY_S = 3


def _shared(setting: Setting) -> bool:
    return not any(presets.is_preset_key(key) for key in setting.config_keys)


class MainWindow(QMainWindow):
    def __init__(self, config_path: str):
        super().__init__()
        self.config_path = config_path
        self.config = HandTracker.load_config(config_path)
        self.worker: Optional[TrackerWorker] = None
        self.recorder: Optional[SessionRecorder] = None
        self._shown_sequence = -1
        self.steamvr: Optional[Path] = None  # picked by hand in Add-ons when it can't be found
        self.driver_missing = False

        self.log_panel = LogPanel()
        capture_output(self.log_panel)
        self.log_panel.text_appended.connect(self._record_log)

        self.camera_view = CameraView()
        self.hand_view = HandView3D()
        self.preset_panel = SettingsPanel(
            self.config, [s for s in SETTINGS if not _shared(s)],
            header="These settings belong to the preset, so each camera position keeps its own. "
                   "Changes apply live; Save preset keeps them.")
        self.app_panel = SettingsPanel(
            self.config, [s for s in SETTINGS if _shared(s)],
            header="These settings are the same for every preset. Changes are saved automatically.")
        for panel in (self.preset_panel, self.app_panel):
            panel.changed.connect(self._on_settings_changed)
        self.preset_bar = PresetBar(self.config, self.write_config)
        self.preset_bar.switched.connect(self._on_preset_switched)
        self.preset_bar.edited.connect(self._update_title)
        preset_tab = QWidget()
        preset_layout = QVBoxLayout(preset_tab)
        preset_layout.setContentsMargins(0, 4, 0, 0)
        preset_layout.addWidget(self.preset_bar)
        preset_layout.addWidget(self.preset_panel, 1)
        self.live_panel = LivePanel()

        previews = QSplitter(Qt.Vertical)
        previews.addWidget(self.camera_view)
        previews.addWidget(self.hand_view)
        previews.setSizes([460, 340])
        tabs = QTabWidget()
        tabs.addTab(preset_tab, "Preset")
        tabs.addTab(self.app_panel, "App settings")
        tabs.addTab(self.live_panel, "Hands")
        tabs.addTab(self.log_panel, "Log")
        main = QSplitter(Qt.Horizontal)
        main.addWidget(previews)
        main.addWidget(tabs)
        main.setStretchFactor(0, 3)
        main.setStretchFactor(1, 2)
        main.setSizes([840, 560])
        self.setCentralWidget(main)

        self._recentering = False
        self._build_actions()
        self._build_status_bar()
        self._update_title()
        self.resize(1400, 860)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(REFRESH_MS)
        self.start_tracking()
        QTimer.singleShot(300, self._check_addons_at_startup)

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
        swap_action.triggered.connect(lambda: self.set_setting(
            "tracking.swap_hands", not get_value(self.config, "tracking.swap_hands", False)))
        self.depth_action = QAction("WiLoR depth", self)
        self.depth_action.setCheckable(True)
        self.depth_action.setToolTip("Heavy 3D hand model for steadier depth (experimental)")
        self.depth_action.triggered.connect(lambda checked: self.set_setting(
            "tracking.depth_source", "wilor" if checked else "mediapipe"))
        self.calibrate_action = QAction("Calibrate gestures", self)
        self.calibrate_action.setToolTip("Measure your open hand and fist from where the camera is, "
                                         "for steadier trigger and grip")
        self.calibrate_action.triggered.connect(self._calibrate_gestures)
        recenter_action = QAction("Recenter hands", self)
        recenter_action.setShortcut(QKeySequence("R"))
        # Holding R would otherwise fire it again with every key repeat
        recenter_action.setAutoRepeat(False)
        recenter_action.setToolTip(f"With Hands follow: The room. In {RECENTER_DELAY_S} seconds, the direction "
                                   "your headset faces becomes where the camera is: face it (R)")
        recenter_action.triggered.connect(self._start_recenter)
        self.record_action = QAction("Record", self)
        self.record_action.setShortcut(QKeySequence("F9"))
        self.record_action.setToolTip(f"Record the camera and what the tracker sees, to the {OUTPUT_FOLDER} "
                                      "folder next to the app, to study or replay later (F9)")
        self.record_action.triggered.connect(self._toggle_recording)
        record_video_action = QAction("Record the camera video", self, checkable=True,
                                      checked=bool(get_value(self.config, "recording.video", True)))
        record_video_action.setToolTip("Off: recordings keep only the tracking data, without the picture")
        record_video_action.toggled.connect(self._set_record_video)
        open_recordings_action = QAction("Open recordings folder", self)
        open_recordings_action.triggered.connect(self._open_recordings)
        addons_action = QAction("Add-ons", self)
        addons_action.setToolTip("Install or update the SteamVR driver and WiLoR depth")
        addons_action.triggered.connect(lambda: self.show_addons())
        save_action = QAction("Save preset", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.preset_bar.save)
        discard_action = QAction("Discard preset changes", self)
        discard_action.triggered.connect(self.preset_bar.discard)

        for action in (self.start_action, swap_action, recenter_action, self.depth_action, self.calibrate_action):
            toolbar.addAction(action)
        toolbar.addSeparator()
        toolbar.addAction(self.record_action)
        toolbar.addSeparator()
        toolbar.addAction(addons_action)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(save_action)
        file_menu.addAction(discard_action)
        file_menu.addSeparator()
        file_menu.addAction(self.record_action)
        file_menu.addAction(record_video_action)
        file_menu.addAction(open_recordings_action)
        file_menu.addSeparator()
        file_menu.addAction(addons_action)
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close)
        view_menu = self.menuBar().addMenu("View")
        for text, widget in (("Camera view", self.camera_view), ("3D view", self.hand_view)):
            action = QAction(text, self, checkable=True, checked=True)
            action.toggled.connect(widget.setVisible)
            view_menu.addAction(action)
        landmarks = QAction("Hand skeleton on camera", self, checkable=True, checked=True)
        landmarks.toggled.connect(self._set_show_landmarks)
        view_menu.addAction(landmarks)

    def _start_recenter(self):
        """One countdown at a time: pressing again while it runs does nothing."""
        if self._recentering:
            return
        if self.worker is None:
            self.statusBar().showMessage("Start tracking first.", 3000)
            return
        self._recentering = True
        self._recenter_countdown(RECENTER_DELAY_S)

    def _recenter_countdown(self, seconds: int):
        """Leave time to put the headset on and face the camera, then recenter."""
        if self.worker is None:
            self._recentering = False
            self.statusBar().showMessage("Start tracking first.", 3000)
            return
        if seconds > 0:
            self.statusBar().showMessage(f"Face the camera with your headset: recentering in {seconds}...")
            QTimer.singleShot(1000, lambda: self._recenter_countdown(seconds - 1))
            return
        self._recentering = False
        self.worker.recenter()
        self.statusBar().showMessage("Hands recentered on the camera.", 3000)

    # ----- session recording

    def _toggle_recording(self):
        if self.recorder is None:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        if self.worker is None:
            self.statusBar().showMessage("Start tracking first.", 3000)
            return
        try:
            self.recorder = SessionRecorder(self._recordings_dir(), self.config,
                                            video=bool(get_value(self.config, "recording.video", True)))
        except OSError as e:
            QMessageBox.warning(self, "Record", f"Could not start recording:\n{e}")
            return
        self.worker.recorder = self.recorder
        self.record_action.setText("Stop recording")
        name = (self.recorder.video_path or self.recorder.data_path).name
        self.statusBar().showMessage(f"Recording to {OUTPUT_FOLDER}\\{name} (F9 stops)", 5000)

    def _stop_recording(self, wait: bool = False):
        if self.recorder is None:
            return
        if self.worker is not None:
            self.worker.recorder = None
        recorder, self.recorder = self.recorder, None
        recorder.stop(wait=wait)
        self.record_action.setText("Record")
        self.rec_label.setText("")
        name = (recorder.video_path or recorder.data_path).name
        self.statusBar().showMessage(f"Recorded {recorder.frames} frames to {OUTPUT_FOLDER}\\{name}", 8000)

    def _record_log(self, text: str):
        if self.recorder is not None:
            self.recorder.log(text)

    def _set_record_video(self, video: bool):
        set_value(self.config, "recording.video", video)
        self.write_config()

    def _recordings_dir(self) -> Path:
        return output_dir(addons.layout().app_dir)

    def _open_recordings(self):
        folder = self._recordings_dir()
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.warning(self, "Recordings", f"Could not create the recordings folder:\n{e}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def set_setting(self, key: str, value):
        """Change a setting as if edited in its panel, e.g. from a toolbar button."""
        panel = self.preset_panel if self.preset_panel.has(key) else self.app_panel
        panel.set_value(key, value)

    def _refresh_panels(self):
        self.preset_panel.refresh()
        self.app_panel.refresh()

    def _set_show_landmarks(self, show: bool):
        self.camera_view.show_landmarks = show
        self.camera_view.update()

    def _build_status_bar(self):
        self.fps_label = QLabel()
        self.latency_label = QLabel()
        self.driver_label = QLabel()
        self.depth_label = QLabel()
        self.rec_label = QLabel()
        self.rec_label.setStyleSheet(f"color: {RECORDING_COLOR.name()}; font-weight: bold")
        for label in (self.rec_label, self.fps_label, self.latency_label, self.driver_label, self.depth_label):
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
        self._stop_recording()
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

    def _calibrate_gestures(self):
        if self.worker is None:
            QMessageBox.information(self, "Calibrate gestures", "Start tracking first (F5).")
            return
        current = (get_value(self.config, "gestures.curl_open_deg"), get_value(self.config, "gestures.curl_full_deg"))
        dialog = GestureCalibrationDialog(lambda: self.worker.latest() if self.worker else (-1, None), current, self)
        if dialog.exec() and dialog.changes:
            for key, value in dialog.changes.items():
                set_value(self.config, key, value)
            self._on_settings_changed(dialog.changes)
            self.statusBar().showMessage(f"Gestures calibrated for {presets.active(self.config)}. "
                                         "Save to keep it.", 5000)

    def _on_settings_changed(self, changes: dict):
        if changes.get("tracking.depth_source") == "wilor" and not wilor_installed_here():
            self._switch_to_wilor_environment()
            return
        self._settings_took_effect(changes)
        if self.worker is not None:
            self.worker.apply_settings(changes)

    def _settings_took_effect(self, changes: dict):
        if any(not presets.is_preset_key(key) for key in changes):
            self.write_config()
        self.preset_bar.refresh()
        self._update_title()
        self._sync_depth_action()

    def _on_preset_switched(self, changes: dict):
        self._refresh_panels()
        self._update_title()
        if changes and self.worker is not None:
            self.worker.apply_settings(changes)
        self.statusBar().showMessage(f"Preset: {presets.active(self.config)}", 3000)

    def _on_settings_applied(self, effective: dict):
        """The tracker may not take a value as asked (e.g. WiLoR not installed)."""
        rejected = {k: v for k, v in effective.items() if get_value(self.config, k) != v}
        if not rejected:
            return
        for key, value in rejected.items():
            set_value(self.config, key, value)
        self._refresh_panels()
        self._settings_took_effect(rejected)
        if "tracking.depth_source" in rejected:
            QMessageBox.information(self, "WiLoR depth", "WiLoR could not be loaded, so standard depth stays on. "
                                                         "The Log tab has the details.")

    def _switch_to_wilor_environment(self):
        """
        WiLoR was asked for but isn't installed in this environment: restart the
        app in the one where it is, or offer to install it.
        """
        set_value(self.config, "tracking.depth_source", "mediapipe")
        self._refresh_panels()
        self._sync_depth_action()
        self._update_title()
        launcher = wilor_launcher()
        if launcher is None:
            self.show_addons("WiLoR depth isn't installed yet. Install it below, then turn it on again.")
            return
        modified = presets.is_modified(self.config)
        answer = QMessageBox.question(
            self, "WiLoR depth", "WiLoR is installed in its own environment. Restart the app there with WiLoR "
                                 "depth on?" + (f"\n\nYour changes to the preset {presets.active(self.config)} "
                                                "are saved first." if modified else ""))
        if answer != QMessageBox.Yes:
            return
        set_value(self.config, "tracking.depth_source", "wilor")
        if modified:
            presets.store(self.config)
        if not self.write_config():
            return
        self.stop_tracking(wait=True)
        relaunch(*launcher, self.config_path)
        self.close()

    # ----- add-ons

    def show_addons(self, reason: str = ""):
        dialog = AddonsDialog(self.config, self.write_config, self.steamvr, reason, self)
        dialog.depth_installed.connect(self._offer_wilor)
        dialog.exec()
        self.steamvr = dialog.steamvr
        self.driver_missing = dialog.driver.state == "missing"

    def _offer_wilor(self):
        answer = QMessageBox.question(self, "WiLoR depth", "WiLoR depth is installed. Turn it on now? The app "
                                                           "restarts to load it.")
        if answer == QMessageBox.Yes:
            # After the add-ons window closes, so the restart doesn't happen under it
            QTimer.singleShot(0, lambda: self.set_setting("tracking.depth_source", "wilor"))

    def _check_addons_at_startup(self):
        try:
            status = addons.driver_status()
        except Exception as e:  # never keep the app from starting
            print(f"Could not check the SteamVR driver: {e}")
            return
        self.driver_missing = status.state == "missing"
        if not (get_value(self.config, "addons.check_at_startup", True) and status.needs_attention):
            return
        reasons = {
            "missing": "The SteamVR driver isn't installed yet, so your hands won't show up in SteamVR.",
            "outdated": "This app brings a newer SteamVR driver than the one installed. Update it so both "
                        "match.",
        }
        reason = reasons.get(status.state, "SteamVR turned the driver off. Turn it back on below.")
        self.show_addons(reason)

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
        if self.recorder is not None:
            seconds = int(self.recorder.elapsed_s)
            self.rec_label.setText(f"● REC {seconds // 60}:{seconds % 60:02d}")

    def _update_status(self, frame: TrackingFrame):
        self.fps_label.setText(f"Tracking {frame.tracking_fps:.0f} fps  ·  camera {frame.camera_fps:.0f} fps")
        latency = sum(frame.timings_ms.values())
        self.latency_label.setText(f"{latency:.0f} ms per frame")
        if frame.driver_connected:
            self._set_label(self.driver_label, "SteamVR driver connected", OK_COLOR)
        elif self.driver_missing:
            self._set_label(self.driver_label, "SteamVR driver not installed (see Add-ons)", WARNING_COLOR)
        else:
            self._set_label(self.driver_label, "Waiting for SteamVR driver", WARNING_COLOR)
        wilor = frame.depth_label != "MediaPipe"
        self._set_label(self.depth_label, f"Depth: {frame.depth_label}", WARNING_COLOR if wilor else MUTED_COLOR)

    @staticmethod
    def _set_label(label: QLabel, text: str, color):
        label.setText(text)
        label.setStyleSheet(f"color: {color.name()}")

    # ----- config file

    def _update_title(self):
        modified = " (unsaved changes)" if presets.is_modified(self.config) else ""
        self.setWindowTitle(f"{APP_TITLE} {APP_VERSION} - preset {presets.active(self.config)}{modified}")

    def write_config(self) -> bool:
        """
        Write the config file: the app settings in use, and the presets as saved.
        Unsaved changes to the active preset stay out until Save preset.
        """
        try:
            with open(self.config_path, "w") as f:
                json.dump(presets.without_unsaved(self.config), f, indent=2)
                f.write("\n")
        except OSError as e:
            QMessageBox.critical(self, "Save", f"Could not save {self.config_path}:\n{e}")
            return False
        return True

    def closeEvent(self, event):
        if presets.is_modified(self.config):
            answer = QMessageBox.question(self, APP_TITLE,
                                          f"Save your changes to the preset {presets.active(self.config)}?",
                                          QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            if answer == QMessageBox.Cancel or (answer == QMessageBox.Save and not self.preset_bar.save()):
                event.ignore()
                return
        self._timer.stop()
        self.stop_tracking(wait=True)
        self._stop_recording(wait=True)
        event.accept()
