"""PySide6 + PyQtGraph front end.

Left pane : workspace folder tree (filterable, multi-select) above the parameter form.
Right pane: M-H and chi_n(t) plots (one plot per panel) above the display list. The list
            holds the curves to draw -- fixed colour, TeX legend, panel, order -- and can be
            saved as *.view.toml. Tree selection gives a quick look; the current parameters
            are shown as a live preview.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import (
    QSize,
    QDir,
    QObject,
    QRunnable,
    QSettings,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
)
from PySide6.QtGui import QAction, QActionGroup, QColor, QFont, QGuiApplication, QIcon, QPainter, QPen, QPixmap, QValidator
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFileSystemModel,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStyledItemDelegate,
    QTabWidget,
    QToolButton,
    QTreeView,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import io, presets
from .common import Exponents, Params
from .figure import hp_mT
from .palette import (DARK, LIGHT, PAD, PT_AXIS, PT_LEGEND, PT_TICK, PT_TITLE, SERIES,
                      THEME_MODES, Theme)
from .view import (PANELS, VIEW_SUFFIX, Layer, PdfSettings, ViewState, is_view_file, load_view,
                   next_style, save_view, tex_to_html)

_QT_DASH = [Qt.PenStyle.SolidLine, Qt.PenStyle.DashLine, Qt.PenStyle.DotLine, Qt.PenStyle.DashDotLine]
_theme: Theme = LIGHT


def theme() -> Theme:
    """Colour theme of the on-screen plots (PDF output is always light)."""
    return _theme


FILTERS = {
    "データ・表示リスト (*.csv, *.view.toml)": ["*.csv", "*" + VIEW_SUFFIX],
    "パラメータ (*.toml)": ["*.toml"],
    "すべて (*.csv, *.toml)": ["*.csv", "*.toml"],
}


# ---------------------------------------------------------------------------
# widgets
# ---------------------------------------------------------------------------
class SciSpinBox(QDoubleSpinBox):
    """Spin box shown in %g notation; arrows step by a factor 10^(1/10)."""

    def __init__(self, lo=1e-12, hi=1e15, parent=None):
        super().__init__(parent)
        self.setDecimals(12)
        self.setRange(lo, hi)
        self.setKeyboardTracking(False)

    def textFromValue(self, val: float) -> str:
        return f"{val:.5g}"

    def valueFromText(self, text: str) -> float:
        return float(text)

    def validate(self, input: str, pos: int):
        try:
            float(input)
            return QValidator.State.Acceptable, input, pos
        except ValueError:
            return QValidator.State.Intermediate, input, pos

    def stepBy(self, steps: int) -> None:
        self.setValue(self.value() * 10 ** (steps / 10))


def _dspin(lo, hi, dec, step, suffix="") -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(dec)
    w.setSingleStep(step)
    w.setKeyboardTracking(False)
    if suffix:
        w.setSuffix(" " + suffix)
    return w


class FitScrollArea(QScrollArea):
    """Asks for the full height of its widget (QScrollArea caps its hint at ~24 lines)."""

    def sizeHint(self) -> QSize:
        w = self.widget()
        base = super().sizeHint()
        if w is None:
            return base
        return QSize(base.width(), w.sizeHint().height() + 2 * self.frameWidth())


class ParamPanel(QWidget):
    changed = Signal()
    save_params = Signal()
    save_run = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)

        # -- run ---------------------------------------------------------------
        g = QGroupBox("計算")
        f = QFormLayout(g)
        # presets in two levels: figure -> condition
        self.preset_fig = QComboBox()
        self.preset_fig.addItems(list(presets.GROUPS))
        self.preset_cond = QComboBox()
        fig0, cond0 = presets.locate(presets.DEFAULT_PRESET)
        self.preset_fig.setCurrentText(fig0)
        self._fill_conditions(cond0)
        self.preset_fig.activated.connect(lambda _: self._on_preset_fig())
        self.preset_cond.activated.connect(lambda _: self._apply_preset())
        f.addRow("プリセット 図", self.preset_fig)
        f.addRow("条件", self.preset_cond)
        self.kind = QComboBox()
        self.kind.addItem("M–H ループ", "mh")
        self.kind.addItem("χₙ(t) 温度掃引", "chi")
        self.engine = QComboBox()
        self.engine.addItem("解析解 (CHI999 移植)", "analytic")
        self.engine.addItem("数値解 (臨界状態ソルバー)", "numeric")
        self.name = QLineEdit("run")
        f.addRow("種類", self.kind)
        f.addRow("エンジン", self.engine)
        f.addRow("名前", self.name)
        v.addWidget(g)

        # -- material (value + exponent) --------------------------------------
        g = QGroupBox("材料 (t = 0 の値, X(t) = X·(1−t)^E)")
        f = QFormLayout(g)
        self.k = SciSpinBox(1e-3, 1e15)
        self.k.setSuffix(" N/m³")
        self.b0 = SciSpinBox(0.0, 1e5)  # %g display: p = 1000 needs B0 ~ 1.001e-3 mT
        self.b0.setSuffix(" mT")
        for sb, sample in ((self.k, "8.8888e+08 N/m³"), (self.b0, "8.8888e-08 mT")):
            # widest %.5g text + arrows: otherwise the unit is clipped ("N/n")
            sb.setMinimumWidth(sb.fontMetrics().horizontalAdvance(sample) + 36)
        self.dh = _dspin(0, 1e5, 4, 0.1, "mT")
        self.hc1 = _dspin(0, 1e5, 4, 0.5, "mT")
        self.ek, self.eb0, self.edh, self.ehc1 = (_dspin(0, 20, 3, 0.1) for _ in range(4))
        for label, w, e in (
            ("k", self.k, self.ek),
            ("B₀", self.b0, self.eb0),
            ("ΔH", self.dh, self.edh),
            ("Hc1", self.hc1, self.ehc1),
        ):
            row = QHBoxLayout()
            row.addWidget(w, 3)
            row.addWidget(QLabel("E"))
            row.addWidget(e, 1)
            f.addRow(label, row)
        v.addWidget(g)

        # -- excitation / sample ----------------------------------------------
        g = QGroupBox("励磁・試料")
        f = QFormLayout(g)
        self.hac = _dspin(1e-6, 1e5, 4, 1, "mT")
        self.hdc = _dspin(0, 1e5, 4, 1, "mT")
        self.a = _dspin(1e-5, 100, 5, 0.01, "cm")
        f.addRow("Hac", self.hac)
        f.addRow("Hdc", self.hdc)
        f.addRow("半径 a", self.a)
        v.addWidget(g)

        # -- temperature ------------------------------------------------------
        g = QGroupBox("換算温度 t = T/Tc")
        f = QFormLayout(g)
        self.t = _dspin(0, 0.9999, 4, 0.01)
        self.tmin = _dspin(0, 0.9999, 4, 0.01)
        self.tmax = _dspin(0, 0.9999, 4, 0.01)
        self.tstep = _dspin(1e-4, 1, 4, 0.001)
        f.addRow("t (M–H)", self.t)
        rng = QHBoxLayout()
        for w in (self.tmin, self.tmax, self.tstep):
            rng.addWidget(w)
        f.addRow("min / max / Δt", rng)
        v.addWidget(g)

        # -- numerics ------------------------------------------------------------
        g = QGroupBox("数値")
        f = QFormLayout(g)
        self.n_half = QSpinBox()
        self.n_half.setRange(20, 4000)
        self.n_half.setSingleStep(20)
        self.n_half.setKeyboardTracking(False)
        f.addRow("半周期の分割数", self.n_half)
        v.addWidget(g)

        self.info = QLabel()
        self.info.setWordWrap(True)
        v.addWidget(self.info)

        row = QHBoxLayout()
        b1 = QPushButton("パラメータ保存…")
        b2 = QPushButton("計算して保存")
        b2.setDefault(True)
        b1.clicked.connect(self.save_params)
        b2.clicked.connect(self.save_run)
        row.addWidget(b1)
        row.addWidget(b2)
        v.addLayout(row)
        v.addStretch(1)

        # compact rows: row pitch 31 -> 23 px, panel 725 -> 567 px (macOS style, measured)
        v.setSpacing(4)
        for box in self.findChildren(QGroupBox):
            form = box.layout()
            form.setVerticalSpacing(2)
            m = form.contentsMargins()
            form.setContentsMargins(m.left(), 6, m.right(), 6)

        for w in self._value_widgets():
            w.valueChanged.connect(self._emit)
        self.kind.currentIndexChanged.connect(self._emit)
        self.engine.currentIndexChanged.connect(self._emit)
        self.set_spec(presets.get(presets.DEFAULT_PRESET))

    def preset_key(self) -> str:
        return presets.GROUPS[self.preset_fig.currentText()][self.preset_cond.currentText()]

    def _fill_conditions(self, keep: str | None = None) -> None:
        conds = list(presets.GROUPS[self.preset_fig.currentText()])
        self.preset_cond.clear()
        self.preset_cond.addItems(conds)
        # keep the same panel letter when switching figures, e.g. (d) -> (d)
        pick = next((c for c in conds if keep and c[:3] == keep[:3]), conds[0])
        self.preset_cond.setCurrentText(pick)

    def _on_preset_fig(self) -> None:
        self._fill_conditions(self.preset_cond.currentText())
        self._apply_preset()

    def _apply_preset(self) -> None:
        self.set_spec(presets.get(self.preset_key()))

    def _value_widgets(self):
        return (self.k, self.b0, self.dh, self.hc1, self.ek, self.eb0, self.edh, self.ehc1,
                self.hac, self.hdc, self.a, self.t, self.tmin, self.tmax, self.tstep, self.n_half)

    def _emit(self, *_):
        self.changed.emit()

    def spec(self) -> io.RunSpec:
        n = self.n_half.value()
        return io.RunSpec(
            kind=self.kind.currentData(),
            engine=self.engine.currentData(),
            name=self.name.text().strip() or "run",
            params=Params(
                k=self.k.value(), b0=self.b0.value(), dh=self.dh.value(), hc1=self.hc1.value(),
                hac=self.hac.value(), hdc=self.hdc.value(), a=self.a.value(),
                n_half=n + (n % 2),
            ),
            exps=Exponents(k=self.ek.value(), b0=self.eb0.value(), dh=self.edh.value(),
                           hc1=self.ehc1.value()),
            t=self.t.value(), tmin=self.tmin.value(), tmax=self.tmax.value(),
            tstep=self.tstep.value(),
        )

    def set_spec(self, s: io.RunSpec) -> None:
        ws = [*self._value_widgets(), self.kind, self.engine]
        for w in ws:
            w.blockSignals(True)
        p, e = s.params, s.exps
        self.kind.setCurrentIndex(max(self.kind.findData(s.kind), 0))
        self.engine.setCurrentIndex(max(self.engine.findData(s.engine), 0))
        self.name.setText(s.name)
        for w, val in ((self.k, p.k), (self.b0, p.b0), (self.dh, p.dh), (self.hc1, p.hc1),
                       (self.ek, e.k), (self.eb0, e.b0), (self.edh, e.dh), (self.ehc1, e.hc1),
                       (self.hac, p.hac), (self.hdc, p.hdc), (self.a, p.a), (self.t, s.t),
                       (self.tmin, s.tmin), (self.tmax, s.tmax), (self.tstep, s.tstep)):
            w.setValue(val)
        self.n_half.setValue(p.n_half)
        for w in ws:
            w.blockSignals(False)
        self.changed.emit()


# ---------------------------------------------------------------------------
# background computation for the preview
# ---------------------------------------------------------------------------
class _Signals(QObject):
    done = Signal(int, object, str)


class _Job(QRunnable):
    def __init__(self, gen: int, spec: io.RunSpec, sig: _Signals):
        super().__init__()
        self.gen, self.spec, self.sig = gen, spec, sig

    def run(self):
        try:
            self.sig.done.emit(self.gen, io.compute(self.spec), "")
        except Exception as exc:  # shown in the status bar
            self.sig.done.emit(self.gen, None, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# PDF export dialog
# ---------------------------------------------------------------------------
class PdfDialog(QDialog):
    WIDTHS = {"片段 (86 mm)": 86.0, "全幅 (178 mm)": 178.0, "スライド (120 mm)": 120.0}

    def __init__(self, settings: PdfSettings | None = None, n_panels: int = 1, parent=None):
        from .figure import ARRANGEMENTS, FONTS

        super().__init__(parent)
        self.setWindowTitle("論文用 PDF")
        st = settings if settings is not None else PdfSettings()
        f = QFormLayout(self)
        self.preset = QComboBox()
        self.preset.addItems([*self.WIDTHS, "任意"])
        self.width_mm = _dspin(20, 400, 1, 1, "mm")
        self.width_mm.setValue(st.width_mm)
        match = [k for k, v in self.WIDTHS.items() if v == st.width_mm]
        self.preset.setCurrentText(match[0] if match else "任意")
        self.preset.currentTextChanged.connect(
            lambda s: self.width_mm.setValue(self.WIDTHS[s]) if s in self.WIDTHS else None)
        self.aspect = _dspin(0.2, 3, 3, 0.05)
        self.aspect.setValue(st.aspect)
        self.font_box = QComboBox()
        self.font_box.addItems(list(FONTS))
        self.font_box.setCurrentText(st.font)
        self.pt = _dspin(5, 14, 1, 0.5, "pt")
        self.pt.setValue(st.font_pt)
        self.legend = QCheckBox("凡例を付ける")
        self.legend.setChecked(st.legend)
        self.arrangement = QComboBox()
        for k, v in ARRANGEMENTS.items():
            self.arrangement.addItem(v, k)
        self.arrangement.setCurrentIndex(max(self.arrangement.findData(st.arrangement), 0))
        self.panel_labels = QCheckBox("(a), (b), … を付ける")
        self.panel_labels.setChecked(st.panel_labels)
        f.addRow("幅 (図全体)", self.preset)
        f.addRow("", self.width_mm)
        f.addRow("縦横比 (パネル 1 枚)", self.aspect)
        f.addRow("フォント", self.font_box)
        f.addRow("文字サイズ", self.pt)
        f.addRow("", self.legend)
        f.addRow(f"パネル配置 ({n_panels} 枚)", self.arrangement)
        f.addRow("", self.panel_labels)
        for w in (self.arrangement, self.panel_labels):
            w.setEnabled(n_panels > 1)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    def figure_style(self) -> PdfSettings:
        return PdfSettings(self.width_mm.value(), self.aspect.value(), self.font_box.currentText(),
                           self.pt.value(), self.legend.isChecked(),
                           self.arrangement.currentData(), self.panel_labels.isChecked())


# ---------------------------------------------------------------------------
# display list
# ---------------------------------------------------------------------------
_ROLE = Qt.ItemDataRole.UserRole
C_NAME, C_COLOR, C_LABEL, C_PANEL = range(4)


class _PanelDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        cb = QComboBox(parent)
        cb.addItems(list(PANELS))
        return cb

    def setEditorData(self, editor, index):
        editor.setCurrentText(index.data() or "a")

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText())


class LayerList(QTreeWidget):
    """One row per curve: visibility, fixed colour, TeX legend, panel. Drag to reorder."""

    changed = Signal()
    files_dropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False
        self.setColumnCount(4)
        self.setHeaderLabels(["データ", "色", "凡例 (TeX 可, ダブルクリックで編集)", "パネル"])
        self.setRootIsDecorated(False)
        self.setUniformRowHeights(True)
        self.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self.setDragDropMode(QTreeWidget.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setDropIndicatorShown(True)
        self.setEditTriggers(QTreeWidget.EditTrigger.NoEditTriggers)
        self.invisibleRootItem().setFlags(Qt.ItemFlag.ItemIsDropEnabled
                                          | Qt.ItemFlag.ItemIsEnabled)
        self.setItemDelegateForColumn(C_PANEL, _PanelDelegate(self))
        hdr = self.header()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(C_LABEL, QHeaderView.ResizeMode.Stretch)
        for c in (C_NAME, C_COLOR, C_PANEL):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.itemChanged.connect(self._on_item_changed)
        self.itemClicked.connect(self._on_clicked)
        self.itemDoubleClicked.connect(self._on_double)

    MAX_ROWS = 8  # beyond this the list scrolls

    def sizeHint(self) -> QSize:
        """Height of header + rows (1 .. MAX_ROWS); the plot gets everything else."""
        n = self.topLevelItemCount()
        rh = self.sizeHintForRow(0) if n else self.fontMetrics().height() + 6
        rows = min(max(n, 1), self.MAX_ROWS)
        h = self.header().sizeHint().height() + rows * rh + 2 * self.frameWidth()
        return QSize(super().sizeHint().width(), h)

    def rowsChanged(self) -> None:
        self.updateGeometry()

    # ---- items <-> layers ---------------------------------------------------
    @staticmethod
    def _swatch(color: str, dash: int) -> QIcon:
        pm = QPixmap(28, 12)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(theme().display(color)), 3, _QT_DASH[dash % 4]))
        p.drawLine(2, 6, 26, 6)
        p.end()
        return QIcon(pm)

    def add_layer(self, layer: Layer, kind: str) -> None:
        self._updating = True
        it = QTreeWidgetItem()
        it.setData(0, _ROLE, layer)
        it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsDragEnabled
                    | Qt.ItemFlag.ItemIsEditable)
        it.setCheckState(C_NAME, Qt.CheckState.Checked if layer.visible
                         else Qt.CheckState.Unchecked)
        it.setText(C_NAME, f"{'χ' if kind == 'chi' else 'M–H'}  {layer.path.stem}")
        it.setToolTip(C_NAME, str(layer.path))
        it.setIcon(C_COLOR, self._swatch(layer.color, layer.dash))
        it.setToolTip(C_COLOR, "クリックで色を変更")
        it.setText(C_LABEL, layer.label)
        it.setText(C_PANEL, layer.panel)
        it.setTextAlignment(C_PANEL, Qt.AlignmentFlag.AlignCenter)
        self.addTopLevelItem(it)
        self._updating = False
        self.rowsChanged()

    def layers(self) -> list[Layer]:
        out = []
        for i in range(self.topLevelItemCount()):
            it = self.topLevelItem(i)
            lay: Layer = it.data(0, _ROLE)
            lay.visible = it.checkState(C_NAME) == Qt.CheckState.Checked
            lay.label = it.text(C_LABEL)
            lay.panel = (it.text(C_PANEL).strip().lower() or "a")[:1]
            out.append(lay)
        return out

    def paths(self) -> set[Path]:
        return {l.path for l in self.layers()}

    def set_layers(self, items: list[tuple[Layer, str]]) -> None:
        self.clear()
        for lay, kind in items:
            self.add_layer(lay, kind)
        self.rowsChanged()
        self.changed.emit()

    def recolor_by_panel(self) -> None:
        """Explicit action: the i-th curve of every panel gets palette slot i (solid)."""
        count: dict[str, int] = {}
        for i in range(self.topLevelItemCount()):
            it = self.topLevelItem(i)
            lay: Layer = it.data(0, _ROLE)
            k = (it.text(C_PANEL).strip().lower() or "a")[:1]
            j = count.get(k, 0)
            count[k] = j + 1
            lay.color, lay.dash = SERIES[j % len(SERIES)], j // len(SERIES)
            it.setIcon(C_COLOR, self._swatch(lay.color, lay.dash))
        self.changed.emit()

    def refresh_swatches(self) -> None:
        for i in range(self.topLevelItemCount()):
            it = self.topLevelItem(i)
            lay: Layer = it.data(0, _ROLE)
            it.setIcon(C_COLOR, self._swatch(lay.color, lay.dash))

    def remove_selected(self) -> None:
        for it in self.selectedItems():
            self.takeTopLevelItem(self.indexOfTopLevelItem(it))
        self.rowsChanged()
        self.changed.emit()

    # ---- interaction --------------------------------------------------------
    def _on_item_changed(self, it, col):
        if not self._updating:
            self.changed.emit()

    def _on_clicked(self, it, col):
        if col != C_COLOR:
            return
        lay: Layer = it.data(0, _ROLE)
        c = QColorDialog.getColor(QColor(lay.color), self, "線の色")
        if c.isValid():
            lay.color = c.name()
            it.setIcon(C_COLOR, self._swatch(lay.color, lay.dash))
            self.changed.emit()

    def _on_double(self, it, col):
        if col in (C_LABEL, C_PANEL):
            self.editItem(it, col)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self.state() != \
                QTreeWidget.State.EditingState:
            self.remove_selected()
            return
        super().keyPressEvent(e)

    def dragEnterEvent(self, e):
        if e.source() is not self and e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e):
        if e.source() is not self and e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e):
        if e.source() is self:
            super().dropEvent(e)
            QTimer.singleShot(0, self.changed.emit)
        elif e.mimeData().hasUrls():
            self.files_dropped.emit([Path(u.toLocalFile()) for u in e.mimeData().urls()])
            e.acceptProposedAction()


# ---------------------------------------------------------------------------
# plots: one PlotItem per panel key
# ---------------------------------------------------------------------------
class PanelView(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.items: dict[str, pg.PlotItem] = {}
        self._keys: list[str] = []
        self.restyle()
        self.set_panels(["a"])

    def restyle(self) -> None:
        """Take the current theme; the plots are rebuilt on the next set_panels()."""
        self.setBackground(theme().surface)
        self._keys = []

    def set_panels(self, keys: list[str]) -> None:
        if keys == self._keys:
            for p in self.items.values():
                for it in list(p.listDataItems()):
                    p.removeItem(it)
                p.legend.clear()
            return
        self.clear()
        self.items = {}
        th = theme()
        cols = 1 if len(keys) == 1 else 2
        for i, k in enumerate(keys):
            p = self.addPlot(row=i // cols, col=i % cols)
            p.showGrid(x=True, y=True, alpha=0.25)
            # the range is set by fit(); a mouse pan/zoom keeps the view until "A" is pressed
            p._fit = True
            p.getViewBox().sigRangeChangedManually.connect(
                lambda *_, p=p: setattr(p, "_fit", False))
            p.autoBtn.clicked.connect(lambda *_, p=p: (setattr(p, "_fit", True), self._fit(p)))
            for side in ("bottom", "left"):
                ax = p.getAxis(side)
                ax.setPen(pg.mkPen(th.muted))
                ax.setTextPen(pg.mkPen(th.ink2))
                tf = QFont(ax.font())
                tf.setPointSize(PT_TICK)
                ax.setStyle(tickFont=tf)
                ax.enableAutoSIPrefix(False)  # mT and dimensionless chi: no "kmT", "x0.001"
            p.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen(th.grid)), ignoreBounds=True)
            p.addItem(pg.InfiniteLine(0, angle=90, pen=pg.mkPen(th.grid)), ignoreBounds=True)
            p.addLegend(offset=(10, 10), labelTextColor=th.ink2, labelTextSize=f"{PT_LEGEND}pt")
            if len(keys) > 1:
                p.setTitle(f"({k})", color=th.ink2, size=f"{PT_TITLE}pt")
            self.items[k] = p
        self._keys = list(keys)

    def set_labels(self, x: str, xu: str | None, y: str, yu: str | None) -> None:
        for p in self.items.values():
            css = {"font-size": f"{PT_AXIS}pt", "color": theme().ink2}
            p.setLabel("bottom", x, units=xu, **css)
            p.setLabel("left", y, units=yu, **css)

    def first(self) -> pg.PlotItem:
        return self.items[self._keys[0]]

    @staticmethod
    def _fit(p: pg.PlotItem) -> None:
        """Data span = DATA_FILL of the frame. pyqtgraph's own auto-range adds a pen-width
        pixel margin on top of its padding (measured 84 % / 82.5 %), so set the range here."""
        xs, ys = [], []
        for it in p.listDataItems():
            x, y = it.getData()
            if x is None or len(x) == 0:
                continue
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.any():
                xs += [x[ok].min(), x[ok].max()]
                ys += [y[ok].min(), y[ok].max()]
        if xs:
            p.getViewBox().setRange(xRange=(min(xs), max(xs)), yRange=(min(ys), max(ys)),
                                    padding=PAD)

    def fit(self) -> None:
        for p in self.items.values():
            if p._fit:
                self._fit(p)


# ---------------------------------------------------------------------------
# main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, root: Path):
        super().__init__()
        self.setWindowTitle("mh-curve — modified Kim–Anderson model")
        self.settings = QSettings("mh-curve", "mh-curve")
        self._cache: dict[Path, tuple[float, io.Dataset]] = {}
        self._quick: list[io.Dataset] = []  # selected in the tree, not in the list
        self._samples = self._make_samples()  # start-up screen while the list is empty
        self._preview: io.Dataset | None = None
        self._pdf = PdfSettings()
        self._view_path: Path | None = None
        self._gen = 0
        self._pool = QThreadPool.globalInstance()
        self._sig = _Signals()
        self._sig.done.connect(self._on_preview)

        # ---- left: tree -----------------------------------------------------
        self.model = QFileSystemModel(self)
        self.model.setNameFilterDisables(False)  # hide, not grey out
        self.model.setFilter(QDir.Filter.AllDirs | QDir.Filter.Files | QDir.Filter.NoDotAndDotDot)
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setSelectionMode(QTreeView.SelectionMode.ExtendedSelection)
        self.tree.setSortingEnabled(True)
        self.tree.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self.tree.setDragEnabled(True)
        self.tree.setDragDropMode(QTreeView.DragDropMode.DragOnly)
        for c in (1, 2):
            self.tree.hideColumn(c)
        self.tree.selectionModel().selectionChanged.connect(self._on_selection)
        self.tree.doubleClicked.connect(self._on_tree_double)
        self.model.dataChanged.connect(lambda *_: self._redraw())

        self.root_edit = QLineEdit()
        self.root_edit.setReadOnly(True)
        pick = QToolButton()
        pick.setText("…")
        pick.clicked.connect(self._choose_root)
        self.filter = QComboBox()
        self.filter.addItems(list(FILTERS))
        self.filter.currentTextChanged.connect(self._apply_filter)
        add_btn = QPushButton("表示リストに追加 →")
        add_btn.setToolTip("ツリーで選んだデータを表示リストへ（ダブルクリック・ドラッグでも可）")
        add_btn.clicked.connect(lambda: self._add_files(self._selected_paths()))

        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(4, 4, 4, 4)
        r = QHBoxLayout()
        r.addWidget(self.root_edit, 1)
        r.addWidget(pick)
        tl.addLayout(r)
        tl.addWidget(self.filter)
        tl.addWidget(self.tree, 1)
        tl.addWidget(add_btn)

        self.panel = ParamPanel()
        self.panel.changed.connect(self._schedule_preview)
        self.panel.save_params.connect(self._save_params)
        self.panel.save_run.connect(self._save_run)
        scroll = FitScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.panel)

        # the parameter form asks for exactly its own height; all spare height goes to the tree
        left = QWidget()
        # never narrower than the form: nothing is clipped at the right edge
        left_w = (self.panel.minimumSizeHint().width() + scroll.verticalScrollBar().sizeHint().width()
                  + 2 * scroll.frameWidth())
        left.setMinimumWidth(left_w)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(0)
        lv.addWidget(top, 1)
        lv.addWidget(scroll, 0)

        # ---- right: plots -----------------------------------------------------
        pg.setConfigOptions(antialias=True, background=LIGHT.surface, foreground=LIGHT.ink2)
        self.v_mh = PanelView()
        self.v_chi = PanelView()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.v_mh, "M–H")
        self.tabs.addTab(self.v_chi, "χₙ(t)")

        self.show_preview = QCheckBox("プレビュー")
        self.show_preview.setChecked(True)
        self.normalize = QCheckBox("Hp で規格化")
        self.normalize.setChecked(True)
        self.normalize.setToolTip("M–H の軸を H/Hp, M/Hp にする（Hp は各データに記録済み）")
        self.flip = QCheckBox("−M を上向き")
        self.harm = QSpinBox()
        self.harm.setRange(1, io.NMAX)
        self.harm.setPrefix("n = ")
        self.show_re = QCheckBox("χₙ′")
        self.show_im = QCheckBox("χₙ″")
        self.show_re.setChecked(True)
        self.show_im.setChecked(True)
        pdf_btn = QPushButton("論文用 PDF…")
        pdf_btn.clicked.connect(self._export_pdf)
        for w in (self.normalize, self.flip, self.show_preview, self.show_re, self.show_im):
            w.toggled.connect(self._redraw)
        self.harm.valueChanged.connect(self._redraw)

        bar = QHBoxLayout()
        for w in (self.show_preview, self.normalize, self.flip, QLabel("  χ:"), self.harm,
                  self.show_re, self.show_im):
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(pdf_btn)

        # display list
        self.layers = LayerList()
        self.layers.changed.connect(self._redraw)
        self.layers.files_dropped.connect(self._add_files)
        lbar = QHBoxLayout()
        self.view_label = QLabel("表示リスト")
        lbar.addWidget(self.view_label)
        lbar.addStretch(1)
        for text, slot in (("色をパネルごとに振り直す", self.layers.recolor_by_panel),
                           ("削除", self.layers.remove_selected),
                           ("全消去", self._clear_layers),
                           ("表示を開く…", self._open_view_dialog),
                           ("表示を保存…", self._save_view)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            lbar.addWidget(b)
        lw = QWidget()
        ll = QVBoxLayout(lw)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addLayout(lbar)
        ll.addWidget(self.layers, 1)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)
        rl.addLayout(bar)
        rl.addWidget(self.tabs, 1)  # all spare height goes to the plots
        rl.addWidget(lw, 0)  # the list asks for its rows only (LayerList.sizeHint)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([max(400, left_w), 1400 - max(400, left_w)])
        self.setCentralWidget(split)
        self.resize(1400, 950)

        vmenu = self.menuBar().addMenu("表示")
        self.sample_act = QAction("初期画面の見本 (p = 1000, d)", self, checkable=True)
        self.sample_act.setChecked(True)
        self.sample_act.setToolTip("表示リストが空のとき Hac = Hp, 1.5Hp, 4Hp の 3 本を描く")
        self.sample_act.toggled.connect(self._redraw)
        vmenu.addAction(self.sample_act)
        cmenu = vmenu.addMenu("色のモード")
        group = QActionGroup(self)
        self._theme_mode = str(self.settings.value("theme", "light"))
        if self._theme_mode not in THEME_MODES:
            self._theme_mode = "light"
        for key, text in THEME_MODES.items():
            act = QAction(text, self, checkable=True)
            act.setChecked(key == self._theme_mode)
            act.triggered.connect(lambda _=False, k=key: self._set_theme_mode(k))
            group.addAction(act)
            cmenu.addAction(act)
        QGuiApplication.styleHints().colorSchemeChanged.connect(self._on_system_scheme)

        menu = self.menuBar().addMenu("ファイル")
        for text, slot in (("フォルダを開く…", self._choose_root),
                           ("表示を開く…", self._open_view_dialog),
                           ("表示を保存…", self._save_view)):
            act = QAction(text, self)
            act.triggered.connect(slot)
            menu.addAction(act)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._run_preview)

        self._apply_theme()
        self._set_root(root)
        self._apply_filter(self.filter.currentText())
        self._schedule_preview()

    # ---- start-up sample --------------------------------------------------------------
    @staticmethod
    def _make_samples() -> list[tuple[io.Dataset, str]]:
        """p = 1000, dH = 0.1Hp, Hc1 = 0.15Hp; Hac = Hp, 1.5Hp, 4Hp."""
        base = presets.get(presets.DEFAULT_PRESET)
        out = []
        for h, tex in ((1.0, ""), (1.5, "1.5"), (4.0, "4")):
            spec = replace(base, name=f"Hac={h:g}Hp", params=replace(base.params, hac=h))
            out.append((io.compute(spec), rf"$H_\mathrm{{ac}} = {tex}H_\mathrm{{p}}$"))
        return out

    def _shown_samples(self) -> list[tuple[io.Dataset, str]]:
        if self.sample_act.isChecked() and self.layers.topLevelItemCount() == 0:
            return self._samples
        return []

    # ---- colour theme ----------------------------------------------------------------
    def _set_theme_mode(self, mode: str) -> None:
        self._theme_mode = mode
        self.settings.setValue("theme", mode)
        self._apply_theme()

    def _on_system_scheme(self, *_):
        if self._theme_mode == "system":
            self._apply_theme()

    def _apply_theme(self) -> None:
        global _theme
        if getattr(self, "_in_theme", False):
            return  # setColorScheme() below re-enters through colorSchemeChanged
        self._in_theme = True
        try:
            hints = QGuiApplication.styleHints()
            scheme = {"light": Qt.ColorScheme.Light, "dark": Qt.ColorScheme.Dark,
                      "system": Qt.ColorScheme.Unknown}[self._theme_mode]
            hints.setColorScheme(scheme)  # widgets follow (Qt >= 6.8); Unknown = follow the OS
            if self._theme_mode == "system":
                dark = hints.colorScheme() == Qt.ColorScheme.Dark
            else:
                dark = self._theme_mode == "dark"
        finally:
            self._in_theme = False
        new = DARK if dark else LIGHT
        _theme = new
        pg.setConfigOptions(background=new.surface, foreground=new.ink2)
        for v in (self.v_mh, self.v_chi):
            v.restyle()
        self.layers.refresh_swatches()
        for lab in (self.panel.info, self.view_label):
            lab.setStyleSheet(f"color: {new.ink2};")
        self._redraw()

    # ---- data access ---------------------------------------------------------------
    def _load(self, path: Path) -> io.Dataset | None:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        hit = self._cache.get(path)
        if hit and hit[0] == mtime:
            return hit[1]
        try:
            ds = io.load_dataset(path)
        except Exception as exc:
            self.statusBar().showMessage(f"{path.name}: 読み込めません ({exc})", 8000)
            return None
        self._cache[path] = (mtime, ds)
        return ds

    def _visible(self, kind: str) -> list[tuple[Layer, io.Dataset]]:
        out = []
        for lay in self.layers.layers():
            if lay.visible and (ds := self._load(lay.path)) is not None and ds.kind == kind:
                out.append((lay, ds))
        return out

    @staticmethod
    def _panels(items: list[tuple[Layer, io.Dataset]]) -> list[str]:
        return sorted({lay.panel for lay, _ in items}) or ["a"]

    # ---- drawing ---------------------------------------------------------------------
    def _redraw(self, *_):
        th = theme()
        pv = self._preview if self.show_preview.isChecked() else None
        norm = self.normalize.isChecked()
        sgn = -1.0 if self.flip.isChecked() else 1.0

        # M-H
        items = self._visible("mh")
        self.v_mh.set_panels(self._panels(items))
        if norm:
            self.v_mh.set_labels("H / Hp", None, ("−" if sgn < 0 else "") + "M / Hp", None)
        else:
            self.v_mh.set_labels("μ₀H", "mT", ("−" if sgn < 0 else "") + "μ₀M", "mT")
        first = self.v_mh.first()
        # background first: preview band, then quick-look, then the list in its order
        if pv is not None and pv.kind == "mh":
            self._draw_mh(first, pv, th.preview, 0, "プレビュー", sgn, norm, width=4)
        for ds in self._quick:
            if ds.kind == "mh":
                self._draw_mh(first, ds, th.ink2, 1, f"(選択中) {ds.label}", sgn, norm, width=1)
        for i, (ds, label) in enumerate(self._shown_samples()):
            self._draw_mh(first, ds, th.series[i], 0, tex_to_html(label), sgn, norm)
        for lay, ds in items:
            self._draw_mh(self.v_mh.items[lay.panel], ds, th.display(lay.color), lay.dash,
                          tex_to_html(lay.label), sgn, norm)

        # chi
        n = self.harm.value()
        items = self._visible("chi")
        self.v_chi.set_panels(self._panels(items))
        self.v_chi.set_labels("t = T/Tc", None, f"χ{n}′ (実線), χ{n}″ (破線)", None)
        first = self.v_chi.first()
        if pv is not None and pv.kind == "chi":
            self._draw_chi(first, pv, th.preview, 0, "プレビュー", n, width=4)
        for ds in self._quick:
            if ds.kind == "chi":
                self._draw_chi(first, ds, th.ink2, 0, f"(選択中) {ds.label}", n, width=1)
        for lay, ds in items:
            self._draw_chi(self.v_chi.items[lay.panel], ds, th.display(lay.color), lay.dash,
                           tex_to_html(lay.label), n)
        self.v_mh.fit()
        self.v_chi.fit()

    @staticmethod
    def _draw_mh(p, ds, color, dash, label, sgn, norm, width=1.6):
        s = hp_mT(ds) if norm else 1.0
        br = ds.columns["branch"]
        H, M = ds.columns["H_mT"] / s, ds.columns["M_mT"] * sgn / s
        lp, ini = br == "loop", br == "init"
        p.plot(H[lp], M[lp], pen=pg.mkPen(color, width=width, style=_QT_DASH[dash % 4]),
               name=label)
        p.plot(H[ini], M[ini], pen=pg.mkPen(color, width=1.0, style=Qt.PenStyle.DotLine))

    def _draw_chi(self, p, ds, color, dash, label, n, width=1.6):
        t = ds.columns["t"]
        named = False
        if self.show_re.isChecked():
            p.plot(t, ds.columns[f"chi{n}_re"],
                   pen=pg.mkPen(color, width=width, style=_QT_DASH[dash % 4]),
                   name=label, connect="finite")
            named = True
        if self.show_im.isChecked():
            p.plot(t, ds.columns[f"chi{n}_im"],
                   pen=pg.mkPen(color, width=width, style=Qt.PenStyle.DashLine),
                   name=None if named else label, connect="finite")

    # ---- display list ------------------------------------------------------------------
    def _add_files(self, paths: list[Path]) -> None:
        have = self.layers.paths()
        added = 0
        for p in paths:
            p = Path(p)
            if p.suffix != ".csv" or p in have or (ds := self._load(p)) is None:
                continue
            color, dash = next_style(self.layers.layers())
            self.layers.add_layer(Layer(p, p.stem, color, "a", True, dash), ds.kind)
            have.add(p)
            added += 1
            self.tabs.setCurrentWidget(self.v_chi if ds.kind == "chi" else self.v_mh)
        if added:
            self.layers.changed.emit()
            self._on_selection()  # drop them from the quick-look

    def _clear_layers(self):
        if self.layers.topLevelItemCount() and QMessageBox.question(
                self, "表示リスト", "表示リストを空にしますか？") != QMessageBox.StandardButton.Yes:
            return
        self.layers.set_layers([])
        self._set_view_path(None)

    def _state(self) -> ViewState:
        return ViewState(self.layers.layers(), self.normalize.isChecked(), self.flip.isChecked(),
                         self.harm.value(), self.show_re.isChecked(), self.show_im.isChecked(),
                         self._pdf)

    def _apply_state(self, st: ViewState) -> None:
        for w, v in ((self.normalize, st.normalize), (self.flip, st.flip),
                     (self.show_re, st.show_re), (self.show_im, st.show_im)):
            w.blockSignals(True)
            w.setChecked(v)
            w.blockSignals(False)
        self.harm.blockSignals(True)
        self.harm.setValue(st.harmonic)
        self.harm.blockSignals(False)
        self._pdf = st.pdf
        items, missing = [], []
        for lay in st.layers:
            ds = self._load(lay.path)
            if ds is None:
                missing.append(lay.path.name)
                continue
            items.append((lay, ds.kind))
        self.layers.set_layers(items)
        if items:
            self.tabs.setCurrentWidget(self.v_chi if items[0][1] == "chi" else self.v_mh)
        if missing:
            QMessageBox.warning(self, "表示リスト", "見つからないデータ:\n" + "\n".join(missing))

    def _set_view_path(self, p: Path | None) -> None:
        self._view_path = p
        self.view_label.setText(f"表示リスト — {p.name}" if p else "表示リスト")

    def _open_view(self, p: Path) -> None:
        try:
            st = load_view(p)
        except Exception as exc:
            QMessageBox.warning(self, "表示リスト", f"{p.name}\n{exc}")
            return
        self._apply_state(st)
        self._set_view_path(p)
        self.statusBar().showMessage(f"{p.name} を開きました", 5000)

    def _open_view_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "表示を開く", str(self.root),
                                              f"表示リスト (*{VIEW_SUFFIX})")
        if path:
            self._open_view(Path(path))

    def _save_view(self):
        default = self._view_path or self.root / f"view{VIEW_SUFFIX}"
        path, _ = QFileDialog.getSaveFileName(self, "表示を保存", str(default),
                                              f"表示リスト (*{VIEW_SUFFIX})")
        if not path:
            return
        p = Path(path)
        if not p.name.endswith(VIEW_SUFFIX):
            p = p.with_name(p.name.removesuffix(".toml") + VIEW_SUFFIX)
        save_view(self._state(), p)
        self._set_view_path(p)
        self.statusBar().showMessage(f"{p.name} を保存しました", 5000)

    # ---- workspace / tree --------------------------------------------------------------
    def _set_root(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.root_edit.setText(str(root))
        self.tree.setRootIndex(self.model.setRootPath(str(root)))
        self.settings.setValue("root", str(root))

    def _choose_root(self):
        d = QFileDialog.getExistingDirectory(self, "データフォルダ", str(self.root))
        if d:
            self._set_root(Path(d))

    def _apply_filter(self, text: str):
        self.model.setNameFilters(FILTERS[text])

    def _selected_paths(self) -> list[Path]:
        rows = self.tree.selectionModel().selectedRows(0)
        return [Path(self.model.filePath(ix)) for ix in rows if not self.model.isDir(ix)]

    def _on_tree_double(self, ix):
        p = Path(self.model.filePath(ix))
        if p.suffix == ".csv":
            self._add_files([p])

    def _on_selection(self, *_):
        paths = self._selected_paths()
        if len(paths) == 1 and is_view_file(paths[0]):
            if paths[0] != self._view_path:
                self._open_view(paths[0])
        elif len(paths) == 1 and paths[0].suffix == ".toml":
            try:
                self.panel.set_spec(io.load_spec(paths[0]))
                self.statusBar().showMessage(f"{paths[0].name} を読み込みました", 5000)
            except Exception as exc:
                QMessageBox.warning(self, "パラメータ", f"{paths[0].name}\n{exc}")
        listed = self.layers.paths()
        self._quick = [d for p in paths if p.suffix == ".csv" and p not in listed
                       if (d := self._load(p))]
        kinds = {d.kind for d in self._quick}
        if kinds == {"chi"}:
            self.tabs.setCurrentWidget(self.v_chi)
        elif kinds == {"mh"}:
            self.tabs.setCurrentWidget(self.v_mh)
        self._redraw()

    # ---- preview -------------------------------------------------------------------------
    def _schedule_preview(self):
        spec = self.panel.spec()
        try:
            self.panel.info.setText(io.describe(spec))
        except Exception as exc:
            self.panel.info.setText(str(exc))
        self._timer.start()

    def _run_preview(self):
        self._gen += 1
        self._pool.start(_Job(self._gen, self.panel.spec(), self._sig))

    def _on_preview(self, gen: int, ds, err: str):
        if gen != self._gen:
            return  # superseded by a newer request
        if err:
            self.statusBar().showMessage(f"計算エラー: {err}", 10000)
            return
        self._preview = ds
        if self.layers.topLevelItemCount() == 0 and not self._quick:
            self.tabs.setCurrentWidget(self.v_chi if ds.kind == "chi" else self.v_mh)
        self._redraw()

    # ---- saving ----------------------------------------------------------------------------
    def _target_dir(self) -> Path:
        ix = self.tree.currentIndex()
        if ix.isValid():
            p = Path(self.model.filePath(ix))
            return p if p.is_dir() else p.parent
        return self.root

    def _confirm_overwrite(self, *paths: Path) -> bool:
        exist = [p.name for p in paths if p.exists()]
        if not exist:
            return True
        return QMessageBox.question(self, "上書き確認", "上書きしますか？\n" + "\n".join(exist)) \
            == QMessageBox.StandardButton.Yes

    def _save_params(self):
        spec = self.panel.spec()
        path, _ = QFileDialog.getSaveFileName(
            self, "パラメータ保存", str(self._target_dir() / f"{spec.name}.toml"), "TOML (*.toml)")
        if path:
            io.save_spec(spec, path)
            self.statusBar().showMessage(f"{Path(path).name} を保存しました", 5000)

    def _save_run(self):
        spec = self.panel.spec()
        d = self._target_dir()
        csv_p, toml_p = d / f"{spec.name}.csv", d / f"{spec.name}.toml"
        if not self._confirm_overwrite(csv_p, toml_p):
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            ds = io.compute(spec)
            io.save_dataset(ds, csv_p)
            io.save_spec(spec, toml_p)
        except Exception as exc:
            QMessageBox.critical(self, "計算エラー", f"{exc}\n\n{traceback.format_exc()}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self._cache.pop(csv_p, None)
        self.statusBar().showMessage(f"{csv_p.name}, {toml_p.name} を保存しました", 8000)
        if csv_p in self.layers.paths():
            self._redraw()  # overwritten: same layer, new data
        else:
            self._add_files([csv_p])

    # ---- PDF -------------------------------------------------------------------------------
    def _export_pdf(self):
        from . import figure

        on_chi = self.tabs.currentWidget() is self.v_chi
        kind = "chi" if on_chi else "mh"
        items = self._visible(kind)
        if items:
            keys = self._panels(items)
            panels = [(k, [figure.Curve(ds, lay.color, lay.dash, lay.label)
                           for lay, ds in items if lay.panel == k]) for k in keys]
        elif kind == "mh" and self._shown_samples():
            panels = [("a", [figure.Curve(ds, SERIES[i], 0, label)
                             for i, (ds, label) in enumerate(self._shown_samples())])]
        elif self._preview is not None and self._preview.kind == kind:
            panels = [("a", [figure.Curve(self._preview, SERIES[0], 0, self._preview.label)])]
        else:
            QMessageBox.information(self, "論文用 PDF", "描画するデータがありません。")
            return
        dlg = PdfDialog(self._pdf, len(panels), self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._pdf = dlg.figure_style()
        if self._view_path:
            stem = self._view_path.name.removesuffix(VIEW_SUFFIX)
        else:
            curves = [c for _, cs in panels for c in cs]
            stem = curves[0].ds.label if len(curves) == 1 else f"{kind}_{len(curves)}"
        if on_chi:
            stem += f"_chi{self.harm.value()}"
        default = self.root / "figures" / f"{stem}.pdf"
        default.parent.mkdir(exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(self, "PDF 出力", str(default), "PDF (*.pdf)")
        if not path:
            return
        path = Path(path)
        try:
            figure.render(kind, panels, path, self._pdf, flip=self.flip.isChecked(),
                          normalize=self.normalize.isChecked(), n=self.harm.value(),
                          show_re=self.show_re.isChecked(), show_im=self.show_im.isChecked())
        except Exception as exc:
            QMessageBox.critical(self, "PDF 出力", f"{exc}\n\n{traceback.format_exc()}")
            return
        QGuiApplication.clipboard().setText(figure.latex_snippet(path, self.root))
        self.statusBar().showMessage(
            f"{path.name} を出力しました（\\includegraphics の雛形をクリップボードにコピー）", 10000)


def main() -> None:
    app = QApplication(sys.argv)
    settings = QSettings("mh-curve", "mh-curve")
    arg = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    root = arg or Path(str(settings.value("root", str(Path.cwd() / "data"))))
    w = MainWindow(root)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
