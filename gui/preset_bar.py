"""
Preset picker above the preset settings: switch, save or discard changes,
duplicate, rename, delete, and restore the built-in presets. Edits to the
preset settings stay unsaved until Save preset; everything else here is
written to the config file straight away.
"""
from typing import Callable, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QVBoxLayout, QWidget)

import presets
from gui.style import WARNING_COLOR


class PresetBar(QWidget):
    switched = Signal(dict)  # the preset's values were put in use: the changed keys and values
    edited = Signal()        # presets were saved, added, renamed, deleted or restored

    def __init__(self, config: dict, persist: Callable[[], bool], parent=None):
        super().__init__(parent)
        self.config = config
        self._persist = persist  # writes the config file

        self.combo = QComboBox()
        self.combo.setToolTip("Settings for one camera position")
        self.combo.activated.connect(lambda index: self._switch_to(self.combo.itemData(index)))
        self.modified_label = QLabel("unsaved changes")
        self.modified_label.setStyleSheet(f"color: {WARNING_COLOR.name()}")
        self.modified_label.setToolTip("Settings below changed since this preset was saved")

        self.save_button = QPushButton("Save preset")
        self.save_button.setToolTip("Keep the settings below in this preset (Ctrl+S)")
        self.save_button.clicked.connect(self.save)
        self.discard_button = QPushButton("Discard changes")
        self.discard_button.setToolTip("Go back to the preset as last saved")
        self.discard_button.clicked.connect(self.discard)
        self.duplicate_button = QPushButton("Duplicate...")
        self.duplicate_button.setToolTip("New preset from the settings in use")
        self.duplicate_button.clicked.connect(self._duplicate)
        self.rename_button = QPushButton("Rename...")
        self.rename_button.clicked.connect(self._rename)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self._delete)
        self.restore_button = QPushButton("Restore default")
        self.restore_button.setToolTip("Put this built-in preset back as shipped")
        self.restore_button.clicked.connect(self._restore_default)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 4)
        top = QHBoxLayout()
        top.addWidget(QLabel("Preset"))
        top.addWidget(self.combo, 1)
        top.addWidget(self.modified_label)
        buttons = QHBoxLayout()
        for button in (self.save_button, self.discard_button, self.duplicate_button, self.rename_button,
                       self.delete_button, self.restore_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(top)
        layout.addLayout(buttons)
        self.refresh()

    @property
    def active(self) -> str:
        return presets.active(self.config)

    def refresh(self):
        """Reload the list and the buttons from the config."""
        active = self.active
        self.combo.blockSignals(True)
        self.combo.clear()
        for name in presets.names(self.config):
            self.combo.addItem(f"{name}  (built in)" if presets.is_builtin(name) else name, name)
        self.combo.setCurrentIndex(max(self.combo.findData(active), 0))
        self.combo.blockSignals(False)
        modified = presets.is_modified(self.config)
        self.modified_label.setVisible(modified)
        self.save_button.setEnabled(modified)
        self.discard_button.setEnabled(modified)
        builtin = presets.is_builtin(active)
        self.rename_button.setEnabled(not builtin)
        self.delete_button.setEnabled(not builtin)
        self.restore_button.setEnabled(builtin and (modified or presets.has_changes_from_default(self.config, active)))
        self.delete_button.setToolTip("Built-in presets can't be deleted" if builtin else "Delete this preset")
        self.rename_button.setToolTip("Built-in presets can't be renamed" if builtin else "Rename this preset")

    def save(self) -> bool:
        """Store the settings in use in the active preset and write the file."""
        presets.store(self.config)
        saved = self._persist()
        self.refresh()
        self.edited.emit()
        return saved

    def discard(self):
        if presets.is_modified(self.config):
            self._apply(self.active)

    def _apply(self, name: str):
        changes = presets.apply(self.config, name)
        self.refresh()
        self.switched.emit(changes)

    def _switch_to(self, name: str):
        if name == self.active:
            return
        if presets.is_modified(self.config):
            answer = QMessageBox.question(
                self, "Preset", f"Save your changes to {self.active} first?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
            if answer == QMessageBox.Cancel:
                self.refresh()
                return
            if answer == QMessageBox.Save:
                presets.store(self.config)
        self._apply(name)
        self._persist()

    def _ask_name(self, title: str, text: str, current: Optional[str] = None) -> str:
        name = text
        while True:
            name, ok = QInputDialog.getText(self, title, "Name:", QLineEdit.Normal, name)
            if not ok:
                return ""
            problem = presets.validate_name(self.config, name, current)
            if problem is None:
                return name.strip()
            QMessageBox.warning(self, title, problem)

    def _changed(self):
        self._persist()
        self.refresh()
        self.edited.emit()

    def _duplicate(self):
        base = f"{self.active} copy"
        suggestion, n = base, 2
        while presets.validate_name(self.config, suggestion) is not None:
            suggestion, n = f"{base} {n}", n + 1
        name = self._ask_name("Duplicate preset", suggestion)
        if not name:
            return
        # The copy takes the settings in use, unsaved changes included; the
        # original keeps what it had when last saved
        presets.create(self.config, name)
        self._changed()

    def _rename(self):
        old = self.active
        name = self._ask_name("Rename preset", old, current=old)
        if not name or name == old:
            return
        presets.rename(self.config, old, name)
        self._changed()

    def _delete(self):
        name = self.active
        answer = QMessageBox.question(self, "Delete preset", f"Delete the preset {name}?",
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        mode = str(self.config.get("tracking", {}).get("view_mode", "pov"))
        presets.delete(self.config, name)
        self._apply(presets.MODE_PRESETS.get(mode, presets.DEFAULT_PRESET))
        self._changed()

    def _restore_default(self):
        name = self.active
        answer = QMessageBox.question(self, "Restore default",
                                      f"Put {name} back to its default settings? Your changes to it are lost.",
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        presets.restore_default(self.config, name)
        self._apply(name)
        self._changed()
