"""
Add-ons window: checks the SteamVR driver and the optional WiLoR depth, and
installs, updates or removes them with one button, so nobody has to run
install scripts by hand. The checks and copies themselves are in addons.py.
"""
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
                               QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout)

import addons
from gui.style import MUTED_COLOR, OK_COLOR, WARNING_COLOR
from utils.config_utils import get_value, set_value

DEPTH_LICENSE = (
    "WiLoR is a large 3D hand model that runs on an NVIDIA GPU, next to your VR game. It downloads about "
    "3 GB (PyTorch, WiLoR and its models) from their official sources, and Python 3.10 if you don't have it.<br><br>"
    "<b>Personal, non-commercial use only.</b> These are third-party programs with their own licenses, which "
    "you accept by installing them: WiLoR (CC BY-NC-ND 4.0), the MANO hand model (non-commercial research "
    "license) and Ultralytics YOLO (AGPL-3.0). Don't use them commercially or share them."
)


class InstallProcess(QThread):
    """Runs an installer without a console window and streams its output."""

    line = Signal(str)
    done = Signal(int)

    def __init__(self, command: List[str], parent=None):
        super().__init__(parent)
        self._command = command
        self._process: Optional[subprocess.Popen] = None
        self.cancelled = False

    def run(self):
        try:
            self._process = subprocess.Popen(self._command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL, creationflags=addons.NO_WINDOW)
        except OSError as e:
            self.line.emit(f"Could not start the installer: {e}")
            self.done.emit(-1)
            return
        # PowerShell writes redirected output in the console code page
        encoding = "oem" if sys.platform == "win32" else "utf-8"
        for raw in self._process.stdout:
            self.line.emit(raw.decode(encoding, "replace").rstrip())
        self.done.emit(self._process.wait())

    def cancel(self):
        self.cancelled = True
        if self._process is not None and self._process.poll() is None:
            # pip runs in child processes: end the whole tree
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(self._process.pid)], capture_output=True,
                           creationflags=addons.NO_WINDOW)


def _status_label() -> QLabel:
    label = QLabel()
    font = label.font()
    font.setBold(True)
    label.setFont(font)
    return label


def _detail_label() -> QLabel:
    label = QLabel()
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return label


def _paint(label: QLabel, text: str, color):
    label.setText(text)
    label.setStyleSheet(f"color: {color.name()}")


class AddonsDialog(QDialog):
    depth_installed = Signal()  # WiLoR finished installing: the app may offer to turn it on

    def __init__(self, config: dict, persist, steamvr: Optional[Path] = None, reason: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add-ons")
        self.setMinimumWidth(620)
        self.config = config
        self._persist = persist
        self.layout_info = addons.layout()
        self.steamvr = steamvr
        self._installer: Optional[InstallProcess] = None

        outer = QVBoxLayout(self)
        if reason:
            banner = QLabel(reason)
            banner.setWordWrap(True)
            banner.setStyleSheet(f"color: {WARNING_COLOR.name()}")
            outer.addWidget(banner)

        # SteamVR driver
        driver_box = QGroupBox("SteamVR driver (required)")
        driver = QVBoxLayout(driver_box)
        driver.addWidget(QLabel("Shows your hands to SteamVR as controllers."))
        self.driver_status = _status_label()
        self.driver_detail = _detail_label()
        self.driver_note = _detail_label()
        self.driver_button = QPushButton()
        self.driver_button.clicked.connect(self._install_driver)
        self.enable_button = QPushButton("Turn on in SteamVR")
        self.enable_button.clicked.connect(self._enable_driver)
        self.remove_driver_button = QPushButton("Remove")
        self.remove_driver_button.clicked.connect(self._remove_driver)
        self.locate_button = QPushButton("SteamVR folder...")
        self.locate_button.clicked.connect(self._locate_steamvr)
        driver_row = QHBoxLayout()
        driver_row.addWidget(self.driver_status, 1)
        for button in (self.driver_button, self.enable_button, self.remove_driver_button, self.locate_button):
            driver_row.addWidget(button)
        driver.addLayout(driver_row)
        driver.addWidget(self.driver_detail)
        driver.addWidget(self.driver_note)
        outer.addWidget(driver_box)

        # WiLoR depth
        depth_box = QGroupBox("WiLoR depth (optional, experimental)")
        depth = QVBoxLayout(depth_box)
        depth.addWidget(QLabel("Steadier distance from the camera, at about 200 ms of extra depth lag."))
        self.depth_status = _status_label()
        self.depth_detail = _detail_label()
        self.depth_button = QPushButton()
        self.depth_button.clicked.connect(self._install_depth)
        self.remove_depth_button = QPushButton("Remove")
        self.remove_depth_button.clicked.connect(self._remove_depth)
        depth_row = QHBoxLayout()
        depth_row.addWidget(self.depth_status, 1)
        depth_row.addWidget(self.depth_button)
        depth_row.addWidget(self.remove_depth_button)
        depth.addLayout(depth_row)
        depth.addWidget(self.depth_detail)
        self.license_text = QLabel(DEPTH_LICENSE)
        self.license_text.setWordWrap(True)
        self.license_text.setTextFormat(Qt.RichText)
        self.license_check = QCheckBox("I accept these licenses, for personal, non-commercial use")
        self.license_check.toggled.connect(self._refresh_buttons)
        depth.addWidget(self.license_text)
        depth.addWidget(self.license_check)
        self.progress = QPlainTextEdit()
        self.progress.setReadOnly(True)
        self.progress.setFont(QFont("Consolas", 8))
        self.progress.setMaximumBlockCount(2000)
        self.progress.setMinimumHeight(160)
        self.progress.hide()
        depth.addWidget(self.progress)
        outer.addWidget(depth_box)

        bottom = QHBoxLayout()
        self.startup_check = QCheckBox("Check the driver when the app starts")
        self.startup_check.setChecked(bool(get_value(config, "addons.check_at_startup", True)))
        self.startup_check.toggled.connect(self._set_startup_check)
        bottom.addWidget(self.startup_check, 1)
        version = QLabel(f"App version {self.layout_info.version}")
        version.setStyleSheet(f"color: {MUTED_COLOR.name()}")
        bottom.addWidget(version)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        bottom.addWidget(self.close_button)
        outer.addLayout(bottom)

        self.refresh()

    # ----- status

    def refresh(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.driver = addons.driver_status(self.layout_info, self.steamvr)
            self.depth = addons.depth_status(self.layout_info)
        finally:
            QApplication.restoreOverrideCursor()
        self._show_driver()
        self._show_depth()
        self._refresh_buttons()

    def _show_driver(self):
        status = self.driver
        words = addons.describe(status, self.layout_info.version)
        good = status.state == "current" and not status.disabled
        _paint(self.driver_status, words["title"], OK_COLOR if good else WARNING_COLOR)
        self.driver_detail.setText(words["detail"])
        notes = []
        if status.disabled:
            notes.append("SteamVR turned it off (by hand, or after a crash). Close SteamVR and press "
                         "Turn on in SteamVR.")
        for folder in status.elsewhere:
            notes.append(f"Another copy is registered with SteamVR at {folder}; it may load instead of this one.")
        if not status.bundled:
            notes.append("This copy of the app has no driver to install. From source, build it first "
                         "(INSTALL.md section 4).")
        if status.state in ("missing", "outdated") and status.bundled:
            notes.append("Close SteamVR before installing. Start it again afterwards.")
        self.driver_note.setText("\n".join(notes))
        self.driver_note.setVisible(bool(notes))
        labels = {"missing": "Install", "outdated": "Update", "current": "Reinstall"}
        self.driver_button.setText(labels.get(status.state, "Install"))

    def _show_depth(self):
        status = self.depth
        if status.state == "missing":
            _paint(self.depth_status, "Not installed", MUTED_COLOR)
        elif status.state == "outdated":
            _paint(self.depth_status, "Update available", WARNING_COLOR)
        else:
            _paint(self.depth_status, "Installed", OK_COLOR)
        gpu = addons.gpu_summary()
        disk = addons.free_disk_gb(status.env)
        lines = [f"GPU: {gpu}" if gpu else "No NVIDIA GPU found: WiLoR needs one."]
        if disk is not None:
            lines.append(f"Free disk space: {disk:.0f} GB (needs about 6 GB)")
        if status.state != "missing":
            lines.append(f"Installed in {status.env}")
        if not status.managed and status.state == "current":
            lines.append("Set up by hand from source; manage it with pip.")
        self.depth_detail.setText("\n".join(lines))
        self.depth_button.setText("Update" if status.state == "outdated" else "Install")
        installable = status.state != "current" and status.installer
        for widget in (self.license_text, self.license_check):
            widget.setVisible(installable)

    def _refresh_buttons(self):
        busy = self._installer is not None
        driver = self.driver
        has_steamvr = driver.steamvr is not None
        self.driver_button.setEnabled(not busy and has_steamvr and driver.bundled)
        self.enable_button.setVisible(driver.disabled)
        self.remove_driver_button.setVisible(driver.state in ("current", "outdated"))
        self.remove_driver_button.setEnabled(not busy)
        depth = self.depth
        self.depth_button.setVisible(depth.state != "current" and depth.installer)
        self.depth_button.setEnabled(not busy and self.license_check.isChecked())
        self.remove_depth_button.setVisible(depth.state != "missing" and depth.managed)
        self.remove_depth_button.setEnabled(not busy)
        self.close_button.setText("Cancel install" if busy else "Close")

    # ----- driver actions

    def _locate_steamvr(self):
        start = str(self.driver.steamvr or "")
        folder = QFileDialog.getExistingDirectory(self, "SteamVR folder (Steam > SteamVR > Manage > Browse local files)",
                                                  start)
        if not folder:
            return
        path = Path(folder)
        if not ((path / "bin").is_dir() and (path / "drivers").is_dir()):
            QMessageBox.warning(self, "SteamVR folder", f"{path} doesn't look like a SteamVR folder: it should "
                                                        "have bin and drivers folders inside.")
            return
        self.steamvr = path
        self.refresh()

    def _run(self, title: str, action) -> bool:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            action()
            return True
        except addons.AddonError as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, title, str(e))
            return False
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()
            self.refresh()

    def _install_driver(self):
        if self._run("SteamVR driver", lambda: addons.install_driver(self.layout_info, self.driver.steamvr)):
            QMessageBox.information(self, "SteamVR driver",
                                    "The driver is installed. Start SteamVR (or restart it) and keep this app "
                                    "running: your hands show up as controllers once it connects.")

    def _enable_driver(self):
        def enable():
            if addons.steamvr_running():
                raise addons.AddonError("Close SteamVR first: it rewrites its settings when it exits.")
            if not addons.enable_driver_in_steamvr():
                raise addons.AddonError("Could not change SteamVR's settings. Turn it on in SteamVR instead: "
                                        "Settings > Startup / Shutdown > Manage Add-ons.")
        self._run("SteamVR driver", enable)

    def _remove_driver(self):
        answer = QMessageBox.question(self, "Remove the driver", "Remove the driver from SteamVR? Your hands "
                                      "stop showing in SteamVR until you install it again.",
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer == QMessageBox.Yes:
            self._run("Remove the driver", lambda: addons.uninstall_driver(self.driver.steamvr))

    # ----- WiLoR actions

    def _install_depth(self):
        if addons.gpu_summary() is None:
            answer = QMessageBox.question(self, "WiLoR depth", "No NVIDIA GPU was found, and WiLoR won't run "
                                          "without one. Install anyway?", QMessageBox.Yes | QMessageBox.No,
                                          QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        self.progress.clear()
        self.progress.show()
        self._installer = InstallProcess(addons.depth_install_command(self.layout_info), self)
        self._installer.line.connect(self._append_progress)
        self._installer.done.connect(self._depth_done)
        self._installer.start()
        _paint(self.depth_status, "Installing... (this takes a while)", WARNING_COLOR)
        self._refresh_buttons()

    def _append_progress(self, text: str):
        self.progress.moveCursor(QTextCursor.End)
        self.progress.appendPlainText(text)

    def _depth_done(self, code: int):
        cancelled = self._installer.cancelled
        self._installer.wait()
        self._installer = None
        self.refresh()
        if cancelled:
            self._append_progress("\nCancelled. Press Install again to continue where it stopped.")
        elif code == 0 and self.depth.state == "current":
            self.depth_installed.emit()
        else:
            QMessageBox.warning(self, "WiLoR depth", "The install did not finish. The messages above say why; "
                                                     "press Install to try again.")

    def _remove_depth(self):
        answer = QMessageBox.question(self, "Remove WiLoR depth", f"Delete {self.depth.env} (several GB)?",
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer == QMessageBox.Yes:
            self._run("Remove WiLoR depth", lambda: addons.remove_depth(self.layout_info))

    # ----- window

    def _set_startup_check(self, checked: bool):
        set_value(self.config, "addons.check_at_startup", checked)
        self._persist()

    def closeEvent(self, event):
        if self._installer is not None:
            answer = QMessageBox.question(self, "WiLoR depth", "Stop the install? You can continue it later.",
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self._installer.cancel()
            self._installer.wait(15000)
        event.accept()

    def reject(self):
        # Esc goes through closeEvent's install check too
        self.close()
