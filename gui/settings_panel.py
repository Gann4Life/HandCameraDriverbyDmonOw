"""
Settings editors generated from gui.settings_schema. Edits are written into
the GUI's config copy and announced with `changed`, so the running tracker can
apply them live; saving to disk is the main window's job. The window shows two
panels: the active preset's settings, and the ones every preset shares.
"""
from typing import Any, Callable, Dict, List, Optional, Sequence

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QScrollArea, QSlider, QSpinBox, QVBoxLayout,
                               QWidget)

from gui.settings_schema import RESOLUTIONS, SECTIONS, Apply, Setting
from utils.config_utils import get_value, set_value

SLIDER_STEPS = 1000


class FloatEditor(QWidget):
    """Slider for quick live tweaking plus a spin box for exact values."""

    value_changed = Signal(float)

    def __init__(self, setting: Setting):
        super().__init__()
        self._setting = setting
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, SLIDER_STEPS)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(setting.minimum, setting.maximum)
        self.spin.setSingleStep(setting.step)
        self.spin.setDecimals(setting.decimals)
        self.spin.setSuffix(setting.unit)
        self.spin.setKeyboardTracking(False)
        self.spin.setMinimumWidth(90)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def set_value(self, value: float):
        self.spin.blockSignals(True)
        self.slider.blockSignals(True)
        self.spin.setValue(float(value))
        self.slider.setValue(self._to_slider(float(value)))
        self.spin.blockSignals(False)
        self.slider.blockSignals(False)

    def _to_slider(self, value: float) -> int:
        span = self._setting.maximum - self._setting.minimum
        return round((value - self._setting.minimum) / span * SLIDER_STEPS) if span else 0

    def _from_slider(self, position: int):
        span = self._setting.maximum - self._setting.minimum
        value = round(self._setting.minimum + position / SLIDER_STEPS * span, self._setting.decimals)
        self.spin.blockSignals(True)
        self.spin.setValue(value)
        self.spin.blockSignals(False)
        self.value_changed.emit(value)

    def _from_spin(self, value: float):
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_slider(value))
        self.slider.blockSignals(False)
        self.value_changed.emit(round(value, self._setting.decimals))


class Vec3Editor(QWidget):
    value_changed = Signal(list)

    def __init__(self, setting: Setting):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.spins = []
        for axis in ("X", "Y", "Z") if "rotation" not in setting.key else ("Pitch", "Yaw", "Roll"):
            spin = QDoubleSpinBox()
            spin.setPrefix(f"{axis} ")
            spin.setRange(setting.minimum, setting.maximum)
            spin.setSingleStep(setting.step)
            spin.setDecimals(setting.decimals)
            spin.setSuffix(setting.unit)
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self._emit)
            layout.addWidget(spin)
            self.spins.append(spin)

    def set_value(self, value: Sequence[float]):
        for spin, v in zip(self.spins, value):
            spin.blockSignals(True)
            spin.setValue(float(v))
            spin.blockSignals(False)

    def _emit(self):
        self.value_changed.emit([spin.value() for spin in self.spins])


class _Row:
    """One setting's editor and how to load a value into it."""

    def __init__(self, setting: Setting, editor: QWidget, load: Callable[[Any], None], revert: Callable[[], None]):
        self.setting = setting
        self.editor = editor
        self.load = load
        self.revert = revert


class SettingsPanel(QWidget):
    changed = Signal(dict)  # config keys -> new values

    def __init__(self, config: dict, settings: Sequence[Setting], header: str = "", parent=None):
        super().__init__(parent)
        self.config = config
        self._rows: List[_Row] = []
        self._forms: Dict[str, QFormLayout] = {}
        self._groups: Dict[str, QGroupBox] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        if header:
            note = QLabel(header)
            note.setWordWrap(True)
            note.setStyleSheet("color: gray")
            outer.addWidget(note)
        self.advanced_toggle = QCheckBox("Show advanced settings")
        self.advanced_toggle.toggled.connect(self._update_visibility)
        outer.addWidget(self.advanced_toggle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        self._content_layout = QVBoxLayout(content)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        for section in SECTIONS:
            group = QGroupBox(section)
            form = QFormLayout(group)
            form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
            self._groups[section] = group
            self._forms[section] = form
            self._content_layout.addWidget(group)
        self._content_layout.addStretch(1)

        for setting in settings:
            self._add_row(setting)
        self.refresh()

    def setting_for(self, key: str) -> Optional[Setting]:
        return next((row.setting for row in self._rows if row.setting.key == key), None)

    def has(self, key: str) -> bool:
        return self.setting_for(key) is not None

    def set_value(self, key: str, value: Any):
        """Change a setting as if edited (with confirmation), e.g. from a toolbar button."""
        row = next((row for row in self._rows if row.setting.key == key), None)
        if row is not None:
            self._edited(row, value)
            row.load(self._current(row.setting))

    def refresh(self):
        """Reload every editor from the config (after a revert or a preset change)."""
        for row in self._rows:
            row.load(self._current(row.setting))
        self._update_visibility()

    def _current(self, setting: Setting) -> Any:
        if setting.kind == "resolution":
            return (get_value(self.config, "camera.width", 640), get_value(self.config, "camera.height", 480))
        return get_value(self.config, setting.key)

    def _add_row(self, setting: Setting):
        editor, load, signal = self._make_editor(setting)
        row = _Row(setting, editor, load, lambda: load(self._current(setting)))
        signal.connect(lambda value, r=row: self._edited(r, value))

        text = setting.label
        if setting.apply is not Apply.LIVE:
            text += f"  <span style='color:gray; font-size:small'>({setting.apply.value})</span>"
        label = QLabel(text)
        label.setTextFormat(Qt.RichText)
        help_text = setting.help
        label.setToolTip(help_text)
        editor.setToolTip(help_text)
        self._forms[setting.section].addRow(label, editor)
        self._rows.append(row)

    def _make_editor(self, setting: Setting):
        kind = setting.kind
        if kind == "float":
            editor = FloatEditor(setting)
            return editor, lambda v: editor.set_value(v if v is not None else setting.minimum), editor.value_changed
        if kind == "int":
            editor = QSpinBox()
            editor.setRange(int(setting.minimum), int(setting.maximum))
            editor.setKeyboardTracking(False)

            def load_int(v):
                editor.blockSignals(True)
                editor.setValue(int(v if v is not None else setting.minimum))
                editor.blockSignals(False)
            return editor, load_int, editor.valueChanged
        if kind == "bool":
            editor = QCheckBox()

            def load_bool(v):
                editor.blockSignals(True)
                editor.setChecked(bool(v))
                editor.blockSignals(False)
            return editor, load_bool, editor.toggled
        if kind in ("choice", "resolution"):
            editor = QComboBox()
            options = setting.choices if kind == "choice" else tuple((f"{w}x{h}", f"{w} x {h}") for w, h in RESOLUTIONS)
            for value, text in options:
                editor.addItem(text, value)

            def load_choice(v):
                editor.blockSignals(True)
                if kind == "resolution":
                    v = f"{int(v[0])}x{int(v[1])}"
                index = editor.findData(v)
                if index < 0 and v is not None:  # a value from the file that isn't a listed choice
                    editor.addItem(str(v), v)
                    index = editor.count() - 1
                editor.setCurrentIndex(max(index, 0))
                editor.blockSignals(False)
            return editor, load_choice, _ComboSignal(editor).value_changed
        if kind == "vec3":
            editor = Vec3Editor(setting)
            return editor, lambda v: editor.set_value(v or [0.0, 0.0, 0.0]), editor.value_changed
        if kind == "text":
            editor = QLineEdit()
            signal = _LineSignal(editor).value_changed

            def load_text(v):
                editor.blockSignals(True)
                editor.setText("" if v is None else str(v))
                editor.blockSignals(False)
            return editor, load_text, signal
        raise ValueError(f"Unknown setting kind '{kind}' for {setting.key}")

    def _edited(self, row: _Row, value: Any):
        setting = row.setting
        if setting.confirm is not None and value == setting.confirm[0] and self._current(setting) != value:
            answer = QMessageBox.warning(self, setting.label, setting.confirm[1],
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                row.revert()
                return
        changes = setting.changes(value)
        for key, v in changes.items():
            set_value(self.config, key, v)
        self.changed.emit(changes)

    def _update_visibility(self):
        show_advanced = self.advanced_toggle.isChecked()
        visible_sections = set()
        for row in self._rows:
            visible = show_advanced or not row.setting.advanced
            self._forms[row.setting.section].setRowVisible(row.editor, visible)
            if visible:
                visible_sections.add(row.setting.section)
        for section, group in self._groups.items():
            group.setVisible(section in visible_sections)


class _ComboSignal(QObject):
    """Re-emits a combo box's selection as its item data."""

    value_changed = Signal(object)

    def __init__(self, combo: QComboBox):
        super().__init__(combo)
        combo.currentIndexChanged.connect(lambda index: self.value_changed.emit(combo.itemData(index)))


class _LineSignal(QObject):
    """Re-emits a line edit's text when editing finishes."""

    value_changed = Signal(object)

    def __init__(self, line: QLineEdit):
        super().__init__(line)
        line.editingFinished.connect(lambda: self.value_changed.emit(line.text().strip()))
