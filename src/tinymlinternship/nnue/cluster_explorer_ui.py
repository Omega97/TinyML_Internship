"""PyQt6 clustering explorer: scatter, chess inspector, and dynamic link overlay."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QByteArray, QPointF, QRect, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from tinymlinternship.nnue.cluster_explore import (
    ExplorerData,
    FenResolver,
    ModelPredictor,
    PositionInfo,
    SelectionModel,
    cluster_color,
    cluster_palette,
    recluster,
    resolve_position,
)

UNSELECTED_ALPHA = 0.65
UNSELECTED_SIZE = 8.0
SELECTED_SIZE_SCALE = 2.0
CLICK_RADIUS_PX = 12.0
INSPECTOR_WIDTH = 340
BOARD_SVG_SIZE = 220

_DARK_BG = "#121418"
_PANEL_BG = "#1b1f27"
_CARD_BG = "#242a35"
_TEXT = "#e8edf4"
_MUTED = "#9aa6b8"


def _qcolor(hex_color: str, alpha: float = 1.0) -> QColor:
    c = QColor(hex_color)
    c.setAlphaF(max(0.0, min(1.0, float(alpha))))
    return c


def _fmt_eval(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:+.3f}"


def _short_fen(fen: str | None, width: int = 42) -> str:
    if not fen:
        return "FEN unavailable"
    fen = str(fen)
    return fen if len(fen) <= width else fen[: width - 1] + "…"


def _board_svg(fen: str, size: int = BOARD_SVG_SIZE) -> bytes:
    import chess
    import chess.svg

    board = chess.Board(fen)
    return chess.svg.board(board, size=int(size), coordinates=True).encode("utf-8")


class LinkOverlay(QWidget):
    """Transparent layer that redraws point→card segments after pan/zoom/layout."""

    def __init__(self, host: QWidget, window: "ClusterExplorerWindow") -> None:
        super().__init__(host)
        self._window = window
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")

    def paintEvent(self, event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        plot_rect = self._window.plot_rect_in(self)
        if plot_rect is not None:
            painter.setClipRect(self.rect())
        for src, dst, color in self._window.iter_link_geometry():
            if src is None or dst is None:
                continue
            if plot_rect is not None and not plot_rect.contains(src.toPoint()):
                continue
            glow = _qcolor(color, 0.22)
            painter.setPen(QPen(glow, 8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(src, dst)
            line = _qcolor(color, 0.85)
            painter.setPen(QPen(line, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(src, dst)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(line)
            painter.drawEllipse(src, 3.4, 3.4)
            painter.setBrush(_qcolor("#ffffff", 0.9))
            painter.drawEllipse(dst, 2.6, 2.6)


class BoardCard(QFrame):
    closed = pyqtSignal(int)

    def __init__(self, info: PositionInfo, color: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.index = int(info.index)
        self.color = color
        self.setObjectName("boardCard")
        self._apply_chrome(color)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)

        header = QHBoxLayout()
        self.header_label = QLabel(f"Cluster {info.cluster_id}  ·  #{info.sample_id}")
        self.header_label.setFont(QFont("Sans Serif", 10, QFont.Weight.DemiBold))
        close_btn = QPushButton("×")
        close_btn.setFixedSize(22, 22)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet(
            "QPushButton { background: #3a4454; color: white; border: none; border-radius: 11px; }"
            "QPushButton:hover { background: #e74c3c; }"
        )
        close_btn.clicked.connect(lambda: self.closed.emit(self.index))
        header.addWidget(self.header_label, 1)
        header.addWidget(close_btn, 0)
        layout.addLayout(header)
        self.anchor = self.header_label

        self.svg = QSvgWidget()
        self.svg.setFixedSize(QSize(BOARD_SVG_SIZE, BOARD_SVG_SIZE))
        layout.addWidget(self.svg, 0, Qt.AlignmentFlag.AlignHCenter)
        self.meta = QLabel()
        self.meta.setWordWrap(True)
        self.meta.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.meta.setStyleSheet(f"color: {_MUTED}; font-size: 11px;")
        layout.addWidget(self.meta)
        self.set_info(info)

    def _apply_chrome(self, color: str) -> None:
        self.color = color
        self.setStyleSheet(
            f"""
            QFrame#boardCard {{
                background: {_CARD_BG};
                border: 1px solid #3a4454;
                border-left: 5px solid {color};
                border-radius: 8px;
            }}
            QLabel {{ color: {_TEXT}; }}
            """
        )

    def set_info(self, info: PositionInfo, color: str | None = None) -> None:
        if color is not None:
            self._apply_chrome(color)
        self.header_label.setText(f"Cluster {info.cluster_id}  ·  #{info.sample_id}")
        if info.fen:
            try:
                self.svg.load(QByteArray(_board_svg(info.fen)))
                self.svg.show()
            except Exception as exc:
                self.svg.hide()
                self.meta.setText(f"Board render failed: {exc}\n{_short_fen(info.fen, 80)}")
                return
        else:
            self.svg.hide()
        lines = [
            f"FEN  {info.fen or '—'}",
            f"y (teacher)  {_fmt_eval(info.eval_target)}    ŷ (model)  {_fmt_eval(info.eval_pred)}",
            f"cluster  {info.cluster_id}    sample  {info.sample_id}    ||Δ||  {_fmt_eval(info.grad_norm).lstrip('+')}",
            f"proj  ({info.coord_x:+.3f}, {info.coord_y:+.3f})",
        ]
        if info.slice_name:
            lines.append(f"slice  {info.slice_name}  row {info.local_row}")
        self.meta.setText("\n".join(lines))


class ClusterExplorerWindow(QMainWindow):
    def __init__(
        self,
        data: ExplorerData,
        *,
        max_select: int = 5,
        checkpoint: Path | None = None,
    ) -> None:
        super().__init__()
        self.data = data
        self.selection = SelectionModel(max_n=int(max_select))
        self.resolver = FenResolver(data.folders)
        self.predictor = ModelPredictor(checkpoint)
        self._shown = np.arange(len(data), dtype=np.int64)
        self.cards: dict[int, BoardCard] = {}
        self._hover_idx: int | None = None

        self._set_title()
        self.resize(1280, 820)
        self.setMinimumSize(960, 620)
        self.setStyleSheet(
            f"""
            QMainWindow, QWidget {{ background: {_DARK_BG}; color: {_TEXT}; }}
            QScrollArea {{ border: none; background: {_PANEL_BG}; }}
            QLineEdit, QSpinBox, QComboBox {{
                background: #2a3140; color: {_TEXT}; border: 1px solid #3a4454;
                border-radius: 4px; padding: 3px 6px;
            }}
            QPushButton {{
                background: #2f3948; color: {_TEXT}; border: 1px solid #4a5568;
                border-radius: 4px; padding: 4px 10px;
            }}
            QPushButton:hover {{ background: #3d4a5c; }}
            QLabel {{ color: {_TEXT}; }}
            """
        )

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.body = QWidget()
        body_layout = QHBoxLayout(self.body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(3)

        pg.setConfigOptions(antialias=True, background=_DARK_BG, foreground=_TEXT)
        self.plot = pg.PlotWidget()
        self.plot.setBackground(_DARK_BG)
        self.plot.showGrid(x=True, y=True, alpha=0.18)
        self.plot.setLabel("bottom", f"{data.method.upper()}-1")
        self.plot.setLabel("left", f"{data.method.upper()}-2")
        self.plot.getPlotItem().setMenuEnabled(False)
        legend = self.plot.addLegend(offset=(8, 8))
        legend.setLabelTextColor(_TEXT)
        self.scatter = pg.ScatterPlotItem(
            pxMode=True,
            hoverable=False,
            size=UNSELECTED_SIZE,
            pen=None,
        )
        self.selected_scatter = pg.ScatterPlotItem(
            pxMode=True,
            hoverable=False,
            size=UNSELECTED_SIZE * SELECTED_SIZE_SCALE,
        )
        self.plot.addItem(self.scatter)
        self.plot.addItem(self.selected_scatter)
        splitter.addWidget(self.plot)

        inspector_host = QWidget()
        inspector_host.setStyleSheet(f"background: {_PANEL_BG};")
        inspector_layout = QVBoxLayout(inspector_host)
        inspector_layout.setContentsMargins(10, 10, 10, 10)
        inspector_layout.setSpacing(8)
        title = QLabel("Side inspector")
        title.setFont(QFont("Sans Serif", 11, QFont.Weight.DemiBold))
        inspector_layout.addWidget(title)
        hint = QLabel("Click a point to pin a board. Click it again to unpin.")
        hint.setStyleSheet(f"color: {_MUTED}; font-size: 11px;")
        hint.setWordWrap(True)
        inspector_layout.addWidget(hint)
        self.empty_label = QLabel("No positions selected.")
        self.empty_label.setStyleSheet(f"color: {_MUTED};")
        inspector_layout.addWidget(self.empty_label)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.card_host = QWidget()
        self.card_layout = QVBoxLayout(self.card_host)
        self.card_layout.setContentsMargins(0, 0, 4, 0)
        self.card_layout.setSpacing(10)
        self.card_layout.addStretch(1)
        self.scroll.setWidget(self.card_host)
        inspector_layout.addWidget(self.scroll, 1)
        inspector_host.setMinimumWidth(280)
        inspector_host.setMaximumWidth(420)
        splitter.addWidget(inspector_host)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([900, INSPECTOR_WIDTH])

        body_layout.addWidget(splitter)
        root.addWidget(self.body, 1)
        root.addWidget(self._build_controls())

        self.overlay = LinkOverlay(self.body, self)
        self.overlay.setGeometry(self.body.rect())
        self.overlay.raise_()

        self.body.installEventFilter(self)
        self.plot.scene().sigMouseClicked.connect(self._on_click)
        self.plot.scene().sigMouseMoved.connect(self._on_move)
        vb = self.plot.getViewBox()
        vb.sigRangeChanged.connect(self._on_view_changed)
        vb.sigTransformChanged.connect(self._on_view_changed)
        self.scroll.verticalScrollBar().valueChanged.connect(self._on_view_changed)
        splitter.splitterMoved.connect(self._on_view_changed)

        self._rebuild_scatter()
        self._refresh_status()
        QTimer.singleShot(0, self._sync_overlay)

    def _build_controls(self) -> QWidget:
        bar = QFrame()
        bar.setFixedHeight(64)
        bar.setStyleSheet(f"QFrame {{ background: {_PANEL_BG}; border-top: 1px solid #2e3644; }}")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Clusters"))
        self.k_spin = QSpinBox()
        self.k_spin.setRange(2, 32)
        self.k_spin.setValue(max(2, int(self.data.n_clusters) or 4))
        self.k_spin.setToolTip("Number of k-means clusters (B). Changing this re-fits on the gradients.")
        self.k_spin.valueChanged.connect(self._on_n_clusters_changed)
        layout.addWidget(self.k_spin)

        self.cluster_box_host = QWidget()
        self.cluster_box_layout = QHBoxLayout(self.cluster_box_host)
        self.cluster_box_layout.setContentsMargins(0, 0, 0, 0)
        self.cluster_box_layout.setSpacing(6)
        self.cluster_boxes: list[QCheckBox] = []
        layout.addWidget(self.cluster_box_host)
        self._rebuild_cluster_boxes()

        layout.addSpacing(8)
        layout.addWidget(QLabel("Max pins"))
        self.max_spin = QSpinBox()
        self.max_spin.setRange(1, 12)
        self.max_spin.setValue(self.selection.max_n)
        self.max_spin.valueChanged.connect(self._on_max_changed)
        layout.addWidget(self.max_spin)

        layout.addWidget(QLabel("Slice"))
        self.slice_edit = QLineEdit()
        self.slice_edit.setPlaceholderText("filter slice name…")
        self.slice_edit.setFixedWidth(180)
        self.slice_edit.textChanged.connect(self._rebuild_scatter)
        layout.addWidget(self.slice_edit)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_selection)
        layout.addWidget(self.clear_btn)

        layout.addStretch(1)
        self.status = QLabel()
        self.status.setStyleSheet(f"color: {_MUTED};")
        layout.addWidget(self.status)
        return bar

    def eventFilter(self, obj, event):  # noqa: ANN001
        if obj is self.body and event.type() == event.Type.Resize:
            self._sync_overlay()
        return super().eventFilter(obj, event)

    def _sync_overlay(self) -> None:
        self.overlay.setGeometry(self.body.rect())
        self.overlay.raise_()
        self.overlay.update()

    def _on_view_changed(self, *args) -> None:  # noqa: ANN002
        self.overlay.update()

    def _visible_clusters(self) -> list[int]:
        return [i for i, box in enumerate(self.cluster_boxes) if box.isChecked()]

    def _rebuild_cluster_boxes(self) -> None:
        while self.cluster_box_layout.count():
            item = self.cluster_box_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.cluster_boxes = []
        palette = cluster_palette(self.data.n_clusters)
        for cid, color in enumerate(palette):
            box = QCheckBox(str(cid))
            box.setChecked(True)
            box.setStyleSheet(
                f"QCheckBox {{ color: {color}; font-weight: 600; }}"
                f"QCheckBox::indicator {{ width: 13px; height: 13px; }}"
            )
            self.cluster_boxes.append(box)
            self.cluster_box_layout.addWidget(box)
            box.stateChanged.connect(self._rebuild_scatter)

    def _refresh_pinned_cards(self) -> None:
        for index, card in list(self.cards.items()):
            info = resolve_position(
                self.data, index, resolver=self.resolver, predictor=self.predictor
            )
            color = cluster_color(info.cluster_id, self.data.n_clusters)
            card.set_info(info, color=color)

    def _set_title(self) -> None:
        self.setWindowTitle(
            f"SARDINE cluster explorer  ·  {len(self.data):,} pts  "
            f"·  {self.data.n_clusters} clusters  ·  {self.data.method}"
        )

    def _on_n_clusters_changed(self, value: int) -> None:
        k = int(value)
        if k == int(self.data.n_clusters) and self.data.diagnostics.get("n_clusters") == k:
            return
        if k < 2 or k > len(self.data):
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            recluster(self.data, k, seed=self.data.cluster_seed)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            self.status.setText(f"k-means failed: {exc}")
            self.k_spin.blockSignals(True)
            self.k_spin.setValue(max(2, int(self.data.n_clusters) or 2))
            self.k_spin.blockSignals(False)
            return
        QApplication.restoreOverrideCursor()
        self._rebuild_cluster_boxes()
        self._refresh_pinned_cards()
        self._rebuild_scatter()
        self._set_title()
        QTimer.singleShot(0, self.overlay.update)

    def _rebuild_scatter(self) -> None:
        mask = self.data.visible_mask(
            clusters=self._visible_clusters(),
            slice_substr=self.slice_edit.text() if hasattr(self, "slice_edit") else "",
        )
        self._shown = np.flatnonzero(mask).astype(np.int64, copy=False)
        xs = self.data.coord_x[self._shown]
        ys = self.data.coord_y[self._shown]
        labels = self.data.cluster_id[self._shown]
        brushes = [_qcolor(cluster_color(int(c), self.data.n_clusters), UNSELECTED_ALPHA) for c in labels]
        self.scatter.setData(x=xs, y=ys, brush=brushes, pen=None, size=UNSELECTED_SIZE)
        self._refresh_selected_markers()
        if hasattr(self, "status"):
            self._refresh_status()
        if hasattr(self, "overlay"):
            self.overlay.update()

    def _refresh_selected_markers(self) -> None:
        if not self.selection.order:
            self.selected_scatter.setData(x=[], y=[])
            return
        idx = np.asarray(self.selection.order, dtype=np.int64)
        xs = self.data.coord_x[idx]
        ys = self.data.coord_y[idx]
        pens = []
        brushes = []
        for i in idx:
            color = cluster_color(int(self.data.cluster_id[i]), self.data.n_clusters)
            pens.append(pg.mkPen("#ffffff", width=2.4))
            brushes.append(_qcolor(color, 1.0))
        self.selected_scatter.setData(
            x=xs,
            y=ys,
            brush=brushes,
            pen=pens,
            size=UNSELECTED_SIZE * SELECTED_SIZE_SCALE,
        )
        self.selected_scatter.setZValue(10)
        self.scatter.setZValue(0)

    def _nearest(self, x: float, y: float) -> int | None:
        if self._shown.size == 0:
            return None
        vb = self.plot.getViewBox()
        (xmin, xmax), (ymin, ymax) = vb.viewRange()
        w = max(float(vb.width()), 1.0)
        h = max(float(vb.height()), 1.0)
        sx = (xmax - xmin) / w
        sy = (ymax - ymin) / h
        dx = (self.data.coord_x[self._shown] - float(x)) / sx
        dy = (self.data.coord_y[self._shown] - float(y)) / sy
        d2 = dx * dx + dy * dy
        j = int(np.argmin(d2))
        if float(d2[j]) > CLICK_RADIUS_PX * CLICK_RADIUS_PX:
            return None
        return int(self._shown[j])

    def _on_click(self, event) -> None:  # noqa: ANN001
        if event.button() != Qt.MouseButton.LeftButton or event.double():
            return
        view_pt = self.plot.getViewBox().mapSceneToView(event.scenePos())
        idx = self._nearest(view_pt.x(), view_pt.y())
        if idx is None:
            return
        event.accept()
        self._toggle(idx)

    def _on_move(self, pos) -> None:  # noqa: ANN001
        view_pt = self.plot.getViewBox().mapSceneToView(pos)
        idx = self._nearest(view_pt.x(), view_pt.y())
        self._hover_idx = idx
        if idx is None:
            QToolTip.hideText()
            self._refresh_status()
            return
        info = self.data.row(idx)
        fen = _short_fen(info.fen, 36)
        text = (
            f"#{info.sample_id}  cluster {info.cluster_id}\n"
            f"({info.coord_x:+.3f}, {info.coord_y:+.3f})\n"
            f"{fen}"
        )
        scene_pt = self.plot.getViewBox().mapViewToScene(view_pt)
        widget_pt = self.plot.mapFromScene(scene_pt)
        QToolTip.showText(self.plot.mapToGlobal(widget_pt), text, self.plot)
        self._refresh_status()

    def _toggle(self, index: int) -> None:
        selected, dropped = self.selection.toggle(index)
        if dropped is not None:
            self._remove_card(dropped)
        if index in selected:
            self._add_card(index)
        else:
            self._remove_card(index)
        self._refresh_selected_markers()
        self._refresh_status()
        QTimer.singleShot(0, self.overlay.update)

    def _add_card(self, index: int) -> None:
        if index in self.cards:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            info = resolve_position(
                self.data, index, resolver=self.resolver, predictor=self.predictor
            )
        finally:
            QApplication.restoreOverrideCursor()
        color = cluster_color(info.cluster_id, self.data.n_clusters)
        card = BoardCard(info, color)
        card.closed.connect(self._toggle)
        self.cards[index] = card
        # Keep card order matching selection FIFO.
        stretch = self.card_layout.takeAt(self.card_layout.count() - 1)
        self.card_layout.addWidget(card)
        self.card_layout.addItem(stretch)
        self.empty_label.hide()
        QTimer.singleShot(0, self.overlay.update)

    def _remove_card(self, index: int) -> None:
        card = self.cards.pop(index, None)
        if card is None:
            return
        self.card_layout.removeWidget(card)
        card.setParent(None)
        card.deleteLater()
        if not self.cards:
            self.empty_label.show()
        QTimer.singleShot(0, self.overlay.update)

    def _clear_selection(self) -> None:
        for idx in list(self.selection.order):
            self._remove_card(idx)
        self.selection.clear()
        self._refresh_selected_markers()
        self._refresh_status()
        self.overlay.update()

    def _on_max_changed(self, value: int) -> None:
        previous = list(self.selection.order)
        self.selection.max_n = int(value)
        dropped = [i for i in previous if i not in self.selection.order]
        for idx in dropped:
            self._remove_card(idx)
        self._refresh_selected_markers()
        self._refresh_status()
        self.overlay.update()

    def _refresh_status(self) -> None:
        hover = ""
        if self._hover_idx is not None:
            info = self.data.row(self._hover_idx)
            hover = f"  ·  hover #{info.sample_id} c{info.cluster_id}"
        sizes = self.data.diagnostics.get("sizes")
        size_txt = f"  ·  sizes {sizes}" if sizes else ""
        space = self.data.diagnostics.get("clustered_on", "")
        space_txt = f"  ·  k-means on {space}" if space else ""
        self.status.setText(
            f"{len(self._shown):,} / {len(self.data):,} shown"
            f"  ·  {self.data.n_source_rows:,} in run"
            f"  ·  B={self.data.n_clusters}"
            f"  ·  {len(self.selection)}/{self.selection.max_n} pinned"
            f"  ·  {self.data.method}{space_txt}{size_txt}{hover}"
        )

    def _to_overlay(self, widget: QWidget, local) -> QPointF:
        # Overlay is a child of body covering body.rect(); map via body (an ancestor).
        return QPointF(widget.mapTo(self.body, local))

    def plot_rect_in(self, widget: QWidget) -> QRect | None:
        top_left = self.plot.mapTo(self.body, self.plot.rect().topLeft())
        rect = QRect(top_left, self.plot.size())
        if widget is self.body or widget is self.overlay:
            return rect
        return QRect(widget.mapFrom(self.body, rect.topLeft()), rect.size())

    def _point_in_overlay(self, index: int) -> QPointF | None:
        x = float(self.data.coord_x[index])
        y = float(self.data.coord_y[index])
        scene_pt = self.plot.getViewBox().mapViewToScene(pg.Point(x, y))
        plot_pt = self.plot.mapFromScene(scene_pt)
        if not self.plot.rect().contains(plot_pt):
            return None
        return self._to_overlay(self.plot, plot_pt)

    def _card_anchor_in_overlay(self, card: BoardCard) -> QPointF:
        target = card.anchor
        local = target.rect().topLeft()
        local.setY(int(target.height() / 2))
        return self._to_overlay(target, local)

    def iter_link_geometry(self):
        for index in self.selection.order:
            card = self.cards.get(index)
            if card is None:
                continue
            src = self._point_in_overlay(index)
            dst = self._card_anchor_in_overlay(card)
            color = cluster_color(int(self.data.cluster_id[index]), self.data.n_clusters)
            yield src, dst, color

    def keyPressEvent(self, event) -> None:  # noqa: ANN001
        if event.key() == Qt.Key.Key_Escape:
            self._clear_selection()
            return
        super().keyPressEvent(event)


def run_explorer(
    data: ExplorerData,
    *,
    max_select: int = 5,
    checkpoint: Path | None = None,
    argv: list[str] | None = None,
) -> int:
    app = QApplication.instance()
    owns = app is None
    if owns:
        app = QApplication([] if argv is None else argv)
        app.setApplicationName("SARDINE cluster explorer")
    win = ClusterExplorerWindow(data, max_select=max_select, checkpoint=checkpoint)
    win.show()
    win.raise_()
    win.activateWindow()
    if owns:
        return int(app.exec())
    return 0
