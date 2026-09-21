"""PyQt6 clustering explorer: scatter, chess inspector, and dynamic link overlay."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QByteArray, QEvent, QPointF, QRect, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPalette, QPen, QPixmap
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from tinymlinternship.nnue.cluster_explore import (
    CLUSTER_ALGO_LABELS,
    CLUSTER_ALGORITHMS,
    POOL_SIZES,
    PROJECTION_LABELS,
    PROJECTION_METHODS,
    ExplorerData,
    FenResolver,
    ModelPredictor,
    PositionInfo,
    SelectionModel,
    cluster_color,
    normalize_cluster_algorithm,
    normalize_projection_method,
    projection_label,
    recluster,
    reload_pool,
    orbit_project,
    reproject,
    resolve_position,
    umap_available,
)
from tinymlinternship.nnue.grad_graph import (
    NnueWeightGrads,
    edge_rgba,
    layer_edges,
    neuron_xy,
    standardize_weight_grads,
    toy_weight_grads,
)

UNSELECTED_ALPHA = 0.65
UNSELECTED_SIZE = 8.0
SELECTED_SIZE_SCALE = 2.0
CLICK_RADIUS_PX = 12.0
INSPECTOR_WIDTH = 400
BOARD_SVG_SIZE = 165
GRAD_GRAPH_PAD = 7
ORBIT_DRAG_PX = 5.0
ORBIT_DEG_PER_PX = 0.4
ORBIT_ELEV_MAX = 85.0


@dataclass(frozen=True)
class ExplorerTheme:
    """UI chrome. Cluster colors and chess-piece colors are independent of this."""

    window_bg: str
    panel_bg: str
    card_bg: str
    text: str
    muted: str
    border: str
    bar_edge: str
    input_bg: str
    button_bg: str
    button_hover: str
    button_checked_bg: str
    button_checked_border: str
    disabled: str
    slider_groove: str
    slider_handle: str
    slider_fill: str
    close_bg: str
    close_fg: str
    hover_bg: str
    hover_border: str
    selected_pen: str
    link_dst: str
    grid_alpha: float


DARK_THEME = ExplorerTheme(
    window_bg="#121418",
    panel_bg="#1b1f27",
    card_bg="#242a35",
    text="#e8edf4",
    muted="#9aa6b8",
    border="#3a4454",
    bar_edge="#2e3644",
    input_bg="#1b1f27",
    button_bg="#2f3948",
    button_hover="#3d4a5c",
    button_checked_bg="#3d5a80",
    button_checked_border="#7aa2d6",
    disabled="#6a7384",
    slider_groove="#2a3140",
    slider_handle="#7aa2d6",
    slider_fill="#3d5a80",
    close_bg="#3a4454",
    close_fg="#ffffff",
    hover_bg="#1b1f27",
    hover_border="#4a5568",
    selected_pen="#ffffff",
    link_dst="#ffffff",
    grid_alpha=0.18,
)

LIGHT_THEME = ExplorerTheme(
    window_bg="#ffffff",
    panel_bg="#f4f6f8",
    card_bg="#ffffff",
    text="#111111",
    muted="#5c6570",
    border="#c5ccd6",
    bar_edge="#e2e6eb",
    input_bg="#f4f6f8",
    button_bg="#eef1f4",
    button_hover="#e2e7ee",
    button_checked_bg="#d6e4f5",
    button_checked_border="#3d6ea8",
    disabled="#9aa3ad",
    slider_groove="#e2e6eb",
    slider_handle="#3d6ea8",
    slider_fill="#a8c4e8",
    close_bg="#e2e6ec",
    close_fg="#111111",
    hover_bg="#ffffff",
    hover_border="#c5ccd6",
    selected_pen="#111111",
    link_dst="#111111",
    grid_alpha=0.22,
)


def _qcolor(hex_color: str, alpha: float = 1.0) -> QColor:
    c = QColor(hex_color)
    c.setAlphaF(max(0.0, min(1.0, float(alpha))))
    return c


def _window_stylesheet(theme: ExplorerTheme) -> str:
    t = theme
    return f"""
        QMainWindow, QWidget {{ background: {t.window_bg}; color: {t.text}; }}
        QScrollArea {{ border: none; background: {t.panel_bg}; }}
        QLabel, QCheckBox {{ background: transparent; color: {t.text}; }}
        QWidget#chromeHost {{ background: transparent; }}
        QLineEdit, QSpinBox, QComboBox, QAbstractSpinBox {{
            background: {t.panel_bg}; color: {t.text}; border: 1px solid {t.border};
            border-radius: 4px; padding: 3px 6px;
        }}
        QAbstractSpinBox QLineEdit {{
            background: {t.panel_bg}; color: {t.text}; border: none;
        }}
        QSpinBox::up-button, QSpinBox::down-button {{
            background: {t.panel_bg}; border: none;
        }}
        QPushButton {{
            background: {t.button_bg}; color: {t.text}; border: 1px solid {t.border};
            border-radius: 4px; padding: 4px 10px;
        }}
        QPushButton:hover {{ background: {t.button_hover}; }}
        QPushButton#projBtn:checked, QPushButton#algoBtn:checked, QPushButton#poolBtn:checked {{
            background: {t.button_checked_bg}; border: 1px solid {t.button_checked_border}; font-weight: 600;
        }}
        QPushButton#projBtn:disabled, QPushButton#algoBtn:disabled, QPushButton#poolBtn:disabled {{
            color: {t.disabled};
        }}
        QSlider::groove:horizontal {{
            height: 6px; background: {t.slider_groove}; border-radius: 3px;
        }}
        QSlider::handle:horizontal {{
            width: 14px; height: 14px; margin: -5px 0;
            background: {t.slider_handle}; border-radius: 7px;
        }}
        QSlider::sub-page:horizontal {{ background: {t.slider_fill}; border-radius: 3px; }}
        QLabel#mutedHint, QLabel#emptyLabel, QLabel#statusLabel {{
            color: {t.muted};
        }}
        QLabel#mutedHint {{ font-size: 11px; }}
        QToolTip {{
            background: {t.hover_bg}; color: {t.text}; border: 1px solid {t.hover_border};
            padding: 4px 6px;
        }}
        QFrame#topBar {{
            background: {t.panel_bg}; border-bottom: 1px solid {t.bar_edge};
        }}
        QFrame#bottomBar {{
            background: {t.panel_bg}; border-top: 1px solid {t.bar_edge};
        }}
        QWidget#inspector {{ background: {t.panel_bg}; }}
    """


def _fmt_eval(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:+.3f}"


def _short_fen(fen: str | None, width: int = 42) -> str:
    if not fen:
        return "FEN unavailable"
    fen = str(fen)
    return fen if len(fen) <= width else fen[: width - 1] + "…"


# Dark frame (black to move): default python-chess margin/coords.
# Light frame (white to move): white margin, black file/rank labels.
_BOARD_COLORS_WHITE = {
    "margin": "#f2f2f2",
    "coord": "#111111",
    "outer border": "#d8d8d8",
    "inner border": "#d8d8d8",
}
_BOARD_COLORS_BLACK = {
    "margin": "#212121",
    "coord": "#e5e5e5",
    "outer border": "#111111",
    "inner border": "#111111",
}


def side_to_move_name(fen: str | None) -> str | None:
    """``White`` / ``Black`` from the FEN STM field, else ``None``."""
    if not fen:
        return None
    parts = str(fen).split()
    if len(parts) < 2:
        return None
    if parts[1] == "w":
        return "White"
    if parts[1] == "b":
        return "Black"
    return None


def position_detail_text(info: PositionInfo) -> str:
    stm = side_to_move_name(info.fen)
    lines = [f"to play  {stm or '—'}"]
    lines.extend(
        [
            f"FEN  {info.fen or '—'}",
            f"y (teacher)  {_fmt_eval(info.eval_target)}    ŷ (model)  {_fmt_eval(info.eval_pred)}",
            f"cluster  {info.cluster_id}    sample  {info.sample_id}    ||Δ||  {_fmt_eval(info.grad_norm).lstrip('+')}",
            (
                f"proj  ({info.coord_x:+.3f}, {info.coord_y:+.3f}, {info.coord_z:+.3f})"
                if info.coord_z is not None
                else f"proj  ({info.coord_x:+.3f}, {info.coord_y:+.3f})"
            ),
        ]
    )
    if info.slice_name:
        lines.append(f"slice  {info.slice_name}  row {info.local_row}")
    return "\n".join(lines)


def _board_svg(fen: str, size: int = BOARD_SVG_SIZE) -> bytes:
    import chess
    import chess.svg

    board = chess.Board(fen)
    colors = _BOARD_COLORS_WHITE if board.turn == chess.WHITE else _BOARD_COLORS_BLACK
    return chess.svg.board(
        board, size=int(size), coordinates=True, colors=colors
    ).encode("utf-8")


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
            painter.setBrush(_qcolor(self._window.theme.link_dst, 0.9))
            painter.drawEllipse(dst, 2.6, 2.6)


def _render_grad_pixmap(
    grads: NnueWeightGrads,
    size: int,
    *,
    rail_color: str = "#9aa6b8",
) -> QPixmap:
    """Paint standardized weight edges: blue +, red −, fade near 0."""
    size = max(int(size), 32)
    img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(QColor(0, 0, 0, 0))
    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    pad = float(GRAD_GRAPH_PAD)
    span = max(float(size) - 2.0 * pad, 1.0)
    sizes = grads.sizes
    rail = _qcolor(rail_color, 0.35)

    def to_px(x: float, y: float) -> tuple[float, float]:
        return pad + float(x) * span, pad + float(y) * span

    for layer, n in enumerate(sizes):
        x0, y0 = to_px(*neuron_xy(layer, 0, n))
        x1, y1 = to_px(*neuron_xy(layer, max(n - 1, 0), n))
        painter.setPen(QPen(rail, 1))
        painter.drawLine(QPointF(x0, y0), QPointF(x1, y1))
        if n <= 24:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(rail)
            for i in range(n):
                px, py = to_px(*neuron_xy(layer, i, n))
                painter.drawEllipse(QPointF(px, py), 2.2, 2.2)

    for src_layer, weight in enumerate(grads.matrices):
        src_n = sizes[src_layer]
        dst_n = sizes[src_layer + 1]
        i_s, js, zs = layer_edges(weight)
        for i, j, z in zip(i_s.tolist(), js.tolist(), zs.tolist()):
            r, g, b, a = edge_rgba(float(z))
            if a < 8:
                continue
            painter.setPen(QPen(_qcolor(f"#{r:02x}{g:02x}{b:02x}", a / 255.0), 1))
            x0, y0 = to_px(*neuron_xy(src_layer, int(i), src_n))
            x1, y1 = to_px(*neuron_xy(src_layer + 1, int(j), dst_n))
            painter.drawLine(QPointF(x0, y0), QPointF(x1, y1))
    painter.end()
    return QPixmap.fromImage(img)


class GradNetWidget(QWidget):
    """Square NNUE-shaped sample-gradient graph, same size as the board."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._grads: NnueWeightGrads | None = None
        self._pixmap: QPixmap | None = None
        self._rail = "#9aa6b8"
        self.setFixedSize(QSize(BOARD_SVG_SIZE, BOARD_SVG_SIZE))
        self.setToolTip(
            "Per-sample NNUE gradient (σ=1).\n"
            "Blue = positive, red = negative; more transparent near 0."
        )

    def set_rail_color(self, color: str) -> None:
        if color == self._rail:
            return
        self._rail = color
        self._pixmap = None
        self.update()

    def set_grads(self, grads: NnueWeightGrads | None) -> None:
        self._grads = grads
        self._pixmap = None
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        if self._grads is None:
            return
        if self._pixmap is None or self._pixmap.size() != self.size():
            self._pixmap = _render_grad_pixmap(
                self._grads, self.width(), rail_color=self._rail
            )
        painter.drawPixmap(0, 0, self._pixmap)


class BoardCard(QFrame):
    closed = pyqtSignal(int)

    def __init__(
        self,
        info: PositionInfo,
        color: str,
        parent: QWidget | None = None,
        *,
        theme: ExplorerTheme = DARK_THEME,
        grads: NnueWeightGrads | None = None,
    ) -> None:
        super().__init__(parent)
        self.index = int(info.index)
        self.color = color
        self.theme = theme
        self.setObjectName("boardCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)

        header = QHBoxLayout()
        self.header_label = QLabel(f"Cluster {info.cluster_id}  ·  #{info.sample_id}")
        self.header_label.setFont(QFont("Sans Serif", 10, QFont.Weight.DemiBold))
        self.close_btn = QPushButton("×")
        self.close_btn.setFixedSize(22, 22)
        self.close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_btn.clicked.connect(lambda: self.closed.emit(self.index))
        header.addWidget(self.header_label, 1)
        header.addWidget(self.close_btn, 0)
        layout.addLayout(header)
        self.anchor = self.header_label

        boards = QHBoxLayout()
        boards.setContentsMargins(0, 0, 0, 0)
        boards.setSpacing(8)
        self.svg = QSvgWidget()
        self.svg.setFixedSize(QSize(BOARD_SVG_SIZE, BOARD_SVG_SIZE))
        self.svg.setMouseTracking(True)
        boards.addWidget(self.svg, 0, Qt.AlignmentFlag.AlignLeft)
        self.grad_view = GradNetWidget()
        boards.addWidget(self.grad_view, 0, Qt.AlignmentFlag.AlignLeft)
        boards.addStretch(1)
        layout.addLayout(boards)
        if grads is not None:
            self.grad_view.set_grads(grads)

        self.detail_text = ""
        self.hover_info = QLabel()
        self.hover_info.setWindowFlags(
            Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        )
        self.hover_info.setWordWrap(True)
        self.hover_info.setFixedWidth(340)
        self.hover_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.hover_info.hide()
        self.svg.installEventFilter(self)
        self.hover_info.installEventFilter(self)
        self.apply_theme(theme)
        self.set_info(info)

    def apply_theme(self, theme: ExplorerTheme) -> None:
        self.theme = theme
        self._apply_chrome(self.color)
        t = theme
        self.grad_view.set_rail_color(t.muted)
        self.close_btn.setStyleSheet(
            f"QPushButton {{ background: {t.close_bg}; color: {t.close_fg}; border: none; border-radius: 11px; }}"
            "QPushButton:hover { background: #e74c3c; color: #ffffff; }"
        )
        self.hover_info.setStyleSheet(
            f"""
            QLabel {{
                background: {t.hover_bg};
                color: {t.text};
                border: 1px solid {t.hover_border};
                border-radius: 6px;
                padding: 8px 10px;
                font-size: 11px;
            }}
            """
        )

    def _apply_chrome(self, color: str) -> None:
        self.color = color
        t = self.theme
        self.setStyleSheet(
            f"""
            QFrame#boardCard {{
                background: {t.card_bg};
                border: 1px solid {t.border};
                border-left: 5px solid {color};
                border-radius: 8px;
            }}
            QLabel {{ color: {t.text}; }}
            """
        )

    def set_info(self, info: PositionInfo, color: str | None = None) -> None:
        if color is not None:
            self._apply_chrome(color)
        self.header_label.setText(f"Cluster {info.cluster_id}  ·  #{info.sample_id}")
        self.detail_text = position_detail_text(info)
        if info.fen:
            try:
                self.svg.load(QByteArray(_board_svg(info.fen)))
                self.svg.show()
            except Exception as exc:
                self.svg.hide()
                self.detail_text = f"Board render failed: {exc}\n{self.detail_text}"
        else:
            self.svg.hide()
        self.hover_info.setText(self.detail_text)
        if self.hover_info.isVisible():
            self._place_hover()

    def eventFilter(self, obj, event):  # noqa: ANN001
        svg = getattr(self, "svg", None)
        hover = getattr(self, "hover_info", None)
        if svg is None or hover is None:
            return super().eventFilter(obj, event)
        if obj in (svg, hover):
            kind = event.type()
            if kind == QEvent.Type.Enter:
                self._show_hover()
            elif kind == QEvent.Type.Leave:
                QTimer.singleShot(80, self._maybe_hide_hover)
        return super().eventFilter(obj, event)

    def _show_hover(self) -> None:
        if not self.detail_text:
            return
        self.hover_info.setText(self.detail_text)
        self.hover_info.adjustSize()
        self._place_hover()
        self.hover_info.show()
        self.hover_info.raise_()

    def _place_hover(self) -> None:
        board = self.svg if self.svg.isVisible() else self
        top_left = board.mapToGlobal(board.rect().topLeft())
        tip_w = max(self.hover_info.sizeHint().width(), 240)
        self.hover_info.resize(tip_w, self.hover_info.sizeHint().height())
        x = top_left.x() - self.hover_info.width() - 10
        y = top_left.y()
        if x < 8:
            below = board.mapToGlobal(board.rect().bottomLeft())
            x = below.x()
            y = below.y() + 8
        self.hover_info.move(x, y)

    def _maybe_hide_hover(self) -> None:
        if self.svg.underMouse() or self.hover_info.underMouse():
            return
        self.hover_info.hide()

    def hideEvent(self, event) -> None:  # noqa: ANN001
        self.hover_info.hide()
        super().hideEvent(event)


class ClusterExplorerWindow(QMainWindow):
    def __init__(
        self,
        data: ExplorerData,
        *,
        checkpoint: Path | None = None,
    ) -> None:
        super().__init__()
        self.data = data
        self.theme = DARK_THEME
        self.selection = SelectionModel()
        self.resolver = FenResolver(data.folders)
        self.predictor = ModelPredictor(checkpoint)
        self._shown = np.arange(len(data), dtype=np.int64)
        self.cards: dict[int, BoardCard] = {}
        self._hover_idx: int | None = None
        self._pool_n = min(POOL_SIZES, key=lambda size: abs(size - len(data)))
        self._view3d = False
        self._azimuth = 45.0
        self._elevation = 25.0
        self._orbit_dragging = False
        self._orbit_moved = False
        self._orbit_last: QPointF | None = None
        self._orbit_press: QPointF | None = None
        self._scatter_brushes: list[QColor] = []
        self._disp_x = data.coord_x
        self._disp_y = data.coord_y
        self._shuffle_display_perm()

        self._set_title()
        self.resize(1280, 820)
        self.setMinimumSize(960, 620)

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

        pg.setConfigOptions(antialias=True, background=self.theme.window_bg, foreground=self.theme.text)
        self.plot = pg.PlotWidget()
        self._set_axis_labels()
        self.plot.getPlotItem().setMenuEnabled(False)
        self.legend = self.plot.addLegend(offset=(8, 8))
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
        inspector_host.setObjectName("inspector")
        self.inspector_host = inspector_host
        inspector_layout = QVBoxLayout(inspector_host)
        inspector_layout.setContentsMargins(10, 10, 10, 10)
        inspector_layout.setSpacing(8)
        title = QLabel("Side inspector")
        title.setFont(QFont("Sans Serif", 11, QFont.Weight.DemiBold))
        inspector_layout.addWidget(title)
        hint = QLabel(
            "Click a point to pin a board. Hover the board for FEN and eval. "
            "The graph is that sample's NNUE gradient (blue +, red −)."
        )
        hint.setObjectName("mutedHint")
        hint.setWordWrap(True)
        inspector_layout.addWidget(hint)
        self.empty_label = QLabel("No positions selected.")
        self.empty_label.setObjectName("emptyLabel")
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
        inspector_host.setMinimumWidth(360)
        inspector_host.setMaximumWidth(520)
        splitter.addWidget(inspector_host)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([900, INSPECTOR_WIDTH])

        body_layout.addWidget(splitter)
        root.addWidget(self._build_top_bar())
        root.addWidget(self.body, 1)
        root.addWidget(self._build_controls())

        self.overlay = LinkOverlay(self.body, self)
        self.overlay.setGeometry(self.body.rect())
        self.overlay.raise_()

        self.body.installEventFilter(self)
        self.plot.viewport().installEventFilter(self)
        self.plot.scene().sigMouseClicked.connect(self._on_click)
        self.plot.scene().sigMouseMoved.connect(self._on_move)
        vb = self.plot.getViewBox()
        vb.sigRangeChanged.connect(self._on_view_changed)
        vb.sigTransformChanged.connect(self._on_view_changed)
        self.scroll.verticalScrollBar().valueChanged.connect(self._on_view_changed)
        splitter.splitterMoved.connect(self._on_view_changed)

        self._apply_theme()
        self._rebuild_scatter()
        self._refresh_status()
        QTimer.singleShot(0, self._sync_overlay)

    def _shuffle_display_perm(self) -> None:
        rng = np.random.RandomState(int(self.data.cluster_seed) + 17)
        self._display_perm = rng.permutation(len(self.data)).astype(np.int64, copy=False)

    def _build_top_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("topBar")
        bar.setFixedHeight(52)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Algorithm"))
        self.algo_group = QButtonGroup(self)
        self.algo_group.setExclusive(True)
        self.algo_buttons: dict[str, QPushButton] = {}
        current_algo = "kmeans"
        try:
            current_algo = normalize_cluster_algorithm(self.data.algorithm)
        except ValueError:
            current_algo = "kmeans"
        for key in CLUSTER_ALGORITHMS:
            btn = QPushButton(CLUSTER_ALGO_LABELS[key])
            btn.setObjectName("algoBtn")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setProperty("algo", key)
            btn.setToolTip(f"Re-cluster with {CLUSTER_ALGO_LABELS[key]}")
            self.algo_group.addButton(btn)
            self.algo_buttons[key] = btn
            layout.addWidget(btn)
        self.algo_buttons.get(current_algo, self.algo_buttons["kmeans"]).setChecked(True)
        self.algo_group.buttonToggled.connect(self._on_algorithm_toggled)

        layout.addSpacing(14)
        layout.addWidget(QLabel("Points"))
        self.pool_group = QButtonGroup(self)
        self.pool_group.setExclusive(True)
        self.pool_buttons: dict[int, QPushButton] = {}
        for size in POOL_SIZES:
            label = f"{size // 1000}k" if size >= 1000 else str(size)
            btn = QPushButton(label)
            btn.setObjectName("poolBtn")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setProperty("pool_n", int(size))
            btn.setToolTip(f"Use {size:,} points for projection and clustering")
            self.pool_group.addButton(btn)
            self.pool_buttons[size] = btn
            layout.addWidget(btn)
        if self._pool_n in self.pool_buttons:
            self.pool_buttons[self._pool_n].setChecked(True)
        else:
            self.pool_buttons[POOL_SIZES[0]].setChecked(True)
        self.pool_group.buttonToggled.connect(self._on_pool_toggled)

        layout.addSpacing(14)
        layout.addWidget(QLabel("Display"))
        self.display_slider = QSlider(Qt.Orientation.Horizontal)
        self.display_slider.setRange(5, 100)
        self.display_slider.setValue(100)
        self.display_slider.setFixedWidth(180)
        self.display_slider.setToolTip("Fraction of the working set drawn. Does not re-cluster.")
        self.display_label = QLabel("100%")
        self.display_label.setFixedWidth(44)
        self.display_slider.valueChanged.connect(self._on_display_pct)
        layout.addWidget(self.display_slider)
        layout.addWidget(self.display_label)
        layout.addStretch(1)
        self.light_mode = QCheckBox("Light mode")
        self.light_mode.setChecked(False)
        self.light_mode.setCursor(Qt.CursorShape.PointingHandCursor)
        self.light_mode.setToolTip(
            "Pure white plot background for thesis screenshots. "
            "Cluster colors and chess pieces stay the same."
        )
        self.light_mode.toggled.connect(self._on_light_mode)
        layout.addWidget(self.light_mode)
        self.view3d = QCheckBox("3D view")
        self.view3d.setChecked(False)
        self.view3d.setCursor(Qt.CursorShape.PointingHandCursor)
        self.view3d.setToolTip(
            "Recompute the embedding in 3D. Drag to orbit around the origin; "
            "scroll still zooms. Cluster labels stay."
        )
        self.view3d.toggled.connect(self._on_view3d)
        layout.addWidget(self.view3d)
        return bar

    def _build_controls(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("bottomBar")
        bar.setFixedHeight(64)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Projection"))
        self.proj_group = QButtonGroup(self)
        self.proj_group.setExclusive(True)
        self.proj_buttons: dict[str, QPushButton] = {}
        current = "pca"
        try:
            current = normalize_projection_method(self.data.method)
        except ValueError:
            current = "pca"
        umap_ok = umap_available()
        for key in PROJECTION_METHODS:
            btn = QPushButton(PROJECTION_LABELS[key])
            btn.setObjectName("projBtn")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setProperty("proj_method", key)
            if key == "umap" and not umap_ok:
                btn.setEnabled(False)
                btn.setToolTip("pip install umap-learn")
            else:
                btn.setToolTip(f"Recompute the layout with {PROJECTION_LABELS[key]}")
            self.proj_group.addButton(btn)
            self.proj_buttons[key] = btn
            layout.addWidget(btn)
        if current in self.proj_buttons and self.proj_buttons[current].isEnabled():
            self.proj_buttons[current].setChecked(True)
        else:
            self.proj_buttons["pca"].setChecked(True)
        self.proj_group.buttonToggled.connect(self._on_projection_toggled)

        layout.addSpacing(8)
        layout.addWidget(QLabel("Clusters"))
        self.k_spin = QSpinBox()
        self.k_spin.setRange(2, 32)
        self.k_spin.setValue(max(2, int(self.data.n_clusters) or 4))
        self.k_spin.setAutoFillBackground(True)
        self.k_spin.setToolTip("Number of clusters (B). Used by k-Means and k-Medoids.")
        self.k_spin.valueChanged.connect(self._on_n_clusters_changed)
        layout.addWidget(self.k_spin)

        self.cluster_box_host = QWidget()
        self.cluster_box_host.setObjectName("chromeHost")
        self.cluster_box_host.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.cluster_box_layout = QHBoxLayout(self.cluster_box_host)
        self.cluster_box_layout.setContentsMargins(0, 0, 0, 0)
        self.cluster_box_layout.setSpacing(6)
        self.cluster_boxes: list[QCheckBox] = []
        layout.addWidget(self.cluster_box_host)
        self._rebuild_cluster_boxes()
        self._sync_k_spin()

        layout.addSpacing(8)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_selection)
        layout.addWidget(self.clear_btn)

        layout.addStretch(1)
        self.status = QLabel()
        self.status.setObjectName("statusLabel")
        layout.addWidget(self.status)
        return bar

    def _on_light_mode(self, checked: bool) -> None:
        self.theme = LIGHT_THEME if checked else DARK_THEME
        self._apply_theme()

    def _apply_theme(self) -> None:
        t = self.theme
        self.setStyleSheet(_window_stylesheet(t))
        if hasattr(self, "k_spin"):
            pal = self.k_spin.palette()
            fill = _qcolor(t.panel_bg)
            pal.setColor(QPalette.ColorRole.Base, fill)
            pal.setColor(QPalette.ColorRole.Button, fill)
            pal.setColor(QPalette.ColorRole.Window, fill)
            pal.setColor(QPalette.ColorRole.Text, _qcolor(t.text))
            self.k_spin.setPalette(pal)
        pg.setConfigOption("background", t.window_bg)
        pg.setConfigOption("foreground", t.text)
        self.plot.setBackground(t.window_bg)
        self.plot.getViewBox().setBackgroundColor(_qcolor(t.window_bg))
        self._set_axis_labels()
        for name in ("bottom", "left", "top", "right"):
            axis = self.plot.getPlotItem().getAxis(name)
            axis.setPen(t.text)
            axis.setTextPen(t.text)
        self.plot.showGrid(x=True, y=True, alpha=t.grid_alpha)
        if self.legend is not None:
            self.legend.setLabelTextColor(t.text)
        for card in self.cards.values():
            card.apply_theme(t)
        if hasattr(self, "overlay"):
            self.overlay.update()
        if hasattr(self, "selected_scatter"):
            self._refresh_selected_markers()

    def _n_components(self) -> int:
        return 3 if self._view3d else 2

    def _on_view3d(self, checked: bool) -> None:
        want = bool(checked)
        if want == self._view3d:
            return
        method = self.data.method
        try:
            method = normalize_projection_method(method)
        except ValueError:
            method = "pca"
        label = projection_label(method)
        dim = "3D" if want else "2D"
        self.status.setText(f"computing {label} {dim}…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            reproject(
                self.data,
                method,
                seed=self.data.cluster_seed,
                n_components=3 if want else 2,
            )
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            self.status.setText(f"{label} {dim} failed: {exc}")
            self.view3d.blockSignals(True)
            self.view3d.setChecked(self._view3d)
            self.view3d.blockSignals(False)
            return
        QApplication.restoreOverrideCursor()
        self._view3d = want
        if want:
            self._azimuth = 45.0
            self._elevation = 25.0
        self._sync_orbit_mouse()
        self._update_display_coords()
        self._set_axis_labels()
        self._rebuild_scatter()
        if want:
            self._fit_orbit_view()
        else:
            vb = self.plot.getViewBox()
            vb.setAspectLocked(False)
            vb.autoRange()
        self._refresh_pinned_cards()
        self._set_title()
        QTimer.singleShot(0, self.overlay.update)

    def _sync_orbit_mouse(self) -> None:
        if self._view3d:
            self.plot.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.plot.unsetCursor()
            self._orbit_dragging = False
            self._orbit_moved = False
            self._orbit_last = None
            self._orbit_press = None

    def _update_display_coords(self) -> None:
        if self._view3d:
            z = self.data.coord_z
            if z is None:
                z = np.zeros_like(self.data.coord_x)
            self._disp_x, self._disp_y = orbit_project(
                self.data.coord_x,
                self.data.coord_y,
                z,
                self._azimuth,
                self._elevation,
            )
        else:
            self._disp_x = self.data.coord_x
            self._disp_y = self.data.coord_y

    def _fit_orbit_view(self) -> None:
        x = np.asarray(self.data.coord_x, dtype=np.float64)
        y = np.asarray(self.data.coord_y, dtype=np.float64)
        z = (
            np.asarray(self.data.coord_z, dtype=np.float64)
            if self.data.coord_z is not None
            else np.zeros_like(x)
        )
        r = float(np.sqrt(x * x + y * y + z * z).max()) if x.size else 1.0
        half = max(r * 1.2, 1e-3)
        vb = self.plot.getViewBox()
        vb.setAspectLocked(True, ratio=1.0)
        vb.setRange(xRange=(-half, half), yRange=(-half, half), padding=0.0)

    def _handle_orbit_event(self, event) -> bool:  # noqa: ANN001
        kind = event.type()
        if kind == QEvent.Type.MouseButtonPress:
            if event.button() != Qt.MouseButton.LeftButton:
                return False
            self._orbit_dragging = True
            self._orbit_moved = False
            self._orbit_last = QPointF(event.position())
            self._orbit_press = QPointF(event.position())
            self.plot.setCursor(Qt.CursorShape.ClosedHandCursor)
            return True
        if kind == QEvent.Type.MouseMove:
            if not self._orbit_dragging or self._orbit_last is None:
                return False
            pos = QPointF(event.position())
            dx = float(pos.x() - self._orbit_last.x())
            dy = float(pos.y() - self._orbit_last.y())
            self._orbit_last = pos
            if self._orbit_press is not None:
                total = hypot(
                    float(pos.x() - self._orbit_press.x()),
                    float(pos.y() - self._orbit_press.y()),
                )
                if total >= ORBIT_DRAG_PX:
                    self._orbit_moved = True
            if self._orbit_moved:
                self._azimuth = (self._azimuth + dx * ORBIT_DEG_PER_PX) % 360.0
                self._elevation = float(
                    np.clip(
                        self._elevation - dy * ORBIT_DEG_PER_PX,
                        -ORBIT_ELEV_MAX,
                        ORBIT_ELEV_MAX,
                    )
                )
                self._update_display_coords()
                self._apply_display_coords()
                self._refresh_selected_markers()
                if hasattr(self, "overlay"):
                    self.overlay.update()
            return True
        if kind == QEvent.Type.MouseButtonRelease:
            if event.button() != Qt.MouseButton.LeftButton or not self._orbit_dragging:
                return False
            moved = self._orbit_moved
            self._orbit_dragging = False
            self._orbit_moved = False
            self._orbit_last = None
            self._orbit_press = None
            self.plot.setCursor(Qt.CursorShape.OpenHandCursor)
            if not moved:
                scene_pt = self.plot.mapToScene(event.position().toPoint())
                view_pt = self.plot.getViewBox().mapSceneToView(scene_pt)
                idx = self._nearest(view_pt.x(), view_pt.y())
                if idx is not None:
                    self._toggle(idx)
            return True
        return False

    def _apply_display_coords(self) -> None:
        if self._shown.size == 0:
            self.scatter.setData(x=[], y=[])
            return
        xs = self._disp_x[self._shown]
        ys = self._disp_y[self._shown]
        self.scatter.setData(
            x=xs,
            y=ys,
            brush=self._scatter_brushes,
            pen=None,
            size=UNSELECTED_SIZE,
        )

    def eventFilter(self, obj, event):  # noqa: ANN001
        if obj is self.body and event.type() == QEvent.Type.Resize:
            self._sync_overlay()
        elif (
            self._view3d
            and hasattr(self, "plot")
            and obj is self.plot.viewport()
            and self._handle_orbit_event(event)
        ):
            return True
        return super().eventFilter(obj, event)

    def _sync_overlay(self) -> None:
        self.overlay.setGeometry(self.body.rect())
        self.overlay.raise_()
        self.overlay.update()

    def _on_view_changed(self, *args) -> None:  # noqa: ANN002
        self.overlay.update()

    def _visible_clusters(self) -> list[int]:
        ids: list[int] = []
        for box in self.cluster_boxes:
            if not box.isChecked():
                continue
            raw = box.property("cluster_id")
            ids.append(int(raw) if raw is not None else 0)
        return ids

    def _rebuild_cluster_boxes(self) -> None:
        while self.cluster_box_layout.count():
            item = self.cluster_box_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.cluster_boxes = []
        ids = list(range(max(int(self.data.n_clusters), 0)))
        if self.data.has_noise:
            ids = [-1, *ids]
        for cid in ids:
            color = cluster_color(cid, self.data.n_clusters)
            box = QCheckBox("noise" if cid < 0 else str(cid))
            box.setChecked(True)
            box.setProperty("cluster_id", int(cid))
            box.setStyleSheet(
                f"QCheckBox {{ color: {color}; font-weight: 600; }}"
                f"QCheckBox::indicator {{ width: 13px; height: 13px; }}"
            )
            self.cluster_boxes.append(box)
            self.cluster_box_layout.addWidget(box)
            box.stateChanged.connect(self._rebuild_scatter)

    def _sync_k_spin(self) -> None:
        if not hasattr(self, "k_spin"):
            return
        algo = "kmeans"
        try:
            algo = normalize_cluster_algorithm(self.data.algorithm)
        except ValueError:
            pass
        self.k_spin.setEnabled(algo != "dbscan")
        if algo != "dbscan" and int(self.data.n_clusters) >= 2:
            self.k_spin.blockSignals(True)
            self.k_spin.setValue(int(self.data.n_clusters))
            self.k_spin.blockSignals(False)

    def _on_display_pct(self, value: int) -> None:
        self.display_label.setText(f"{int(value)}%")
        self._rebuild_scatter()

    def _on_algorithm_toggled(self, button: QPushButton, checked: bool) -> None:
        if not checked:
            return
        try:
            algo = normalize_cluster_algorithm(str(button.property("algo") or ""))
        except ValueError:
            return
        current = "kmeans"
        try:
            current = normalize_cluster_algorithm(self.data.algorithm)
        except ValueError:
            pass
        if algo == current:
            return
        k = max(2, int(self.k_spin.value()) if hasattr(self, "k_spin") else int(self.data.n_clusters or 2))
        self.status.setText(f"clustering with {CLUSTER_ALGO_LABELS[algo]}…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            recluster(self.data, k, seed=self.data.cluster_seed, algorithm=algo)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            self.status.setText(f"{CLUSTER_ALGO_LABELS[algo]} failed: {exc}")
            self.algo_group.blockSignals(True)
            self.algo_buttons[current].setChecked(True)
            self.algo_group.blockSignals(False)
            return
        QApplication.restoreOverrideCursor()
        self._sync_k_spin()
        self._rebuild_cluster_boxes()
        self._refresh_pinned_cards()
        self._rebuild_scatter()
        self._set_title()
        QTimer.singleShot(0, self.overlay.update)

    def _on_pool_toggled(self, button: QPushButton, checked: bool) -> None:
        if not checked:
            return
        n_points = int(button.property("pool_n") or 0)
        if n_points < 2:
            return
        source_n = int(self.data.n_source_rows or len(self.data))
        if str(self.data.source) == "demo":
            already = n_points == len(self.data)
        else:
            already = len(self.data) == min(n_points, source_n)
        if already:
            self._pool_n = n_points
            return
        self.status.setText(f"loading {n_points:,} points…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            new = reload_pool(
                self.data,
                n_points,
                method=self.data.method,
                n_clusters=max(2, int(self.k_spin.value()) if hasattr(self, "k_spin") else 4),
                algorithm=self.data.algorithm,
                n_components=self._n_components(),
            )
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            self.status.setText(f"reload failed: {exc}")
            fallback = int(self._pool_n)
            if fallback in self.pool_buttons:
                self.pool_group.blockSignals(True)
                self.pool_buttons[fallback].setChecked(True)
                self.pool_group.blockSignals(False)
            return
        QApplication.restoreOverrideCursor()
        self._apply_new_data(new, pool_n=n_points)

    def _apply_new_data(self, data: ExplorerData, *, pool_n: int) -> None:
        self._clear_selection()
        self.data = data
        self._pool_n = int(pool_n)
        self.resolver = FenResolver(data.folders)
        self._shuffle_display_perm()
        self._update_display_coords()
        self._set_axis_labels()
        self._sync_k_spin()
        self._rebuild_cluster_boxes()
        self._rebuild_scatter()
        if self._view3d:
            self._fit_orbit_view()
        else:
            self.plot.getViewBox().autoRange()
        self._set_title()
        QTimer.singleShot(0, self.overlay.update)

    def _refresh_pinned_cards(self) -> None:
        for index, card in list(self.cards.items()):
            info = resolve_position(
                self.data, index, resolver=self.resolver, predictor=self.predictor
            )
            color = cluster_color(info.cluster_id, self.data.n_clusters)
            card.set_info(info, color=color)

    def _set_axis_labels(self) -> None:
        if self._view3d:
            self.plot.setLabel("bottom", "view X")
            self.plot.setLabel("left", "view Y")
            return
        label = projection_label(self.data.method) if self.data.method else "proj"
        self.plot.setLabel("bottom", f"{label}-1")
        self.plot.setLabel("left", f"{label}-2")

    def _set_title(self) -> None:
        try:
            method = projection_label(self.data.method)
        except ValueError:
            method = self.data.method
        dim = "3D" if self._view3d else "2D"
        self.setWindowTitle(
            f"SARDINE cluster explorer  ·  {len(self.data):,} pts  "
            f"·  {self.data.n_clusters} clusters  ·  {method} {dim}"
        )

    def _on_projection_toggled(self, button: QPushButton, checked: bool) -> None:
        if not checked:
            return
        method = str(button.property("proj_method") or "")
        try:
            method = normalize_projection_method(method)
        except ValueError:
            return
        current = self.data.method
        try:
            current = normalize_projection_method(current)
        except ValueError:
            current = ""
        if method == current:
            return
        label = projection_label(method)
        self.status.setText(f"computing {label}…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            reproject(
                self.data,
                method,
                seed=self.data.cluster_seed,
                n_components=self._n_components(),
            )
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            self.status.setText(f"{label} failed: {exc}")
            fallback = current if current in self.proj_buttons else "pca"
            self.proj_group.blockSignals(True)
            self.proj_buttons[fallback].setChecked(True)
            self.proj_group.blockSignals(False)
            return
        QApplication.restoreOverrideCursor()
        self._update_display_coords()
        self._set_axis_labels()
        self._rebuild_scatter()
        if self._view3d:
            self._fit_orbit_view()
        else:
            self.plot.getViewBox().autoRange()
        self._refresh_pinned_cards()
        self._set_title()
        QTimer.singleShot(0, self.overlay.update)

    def _on_n_clusters_changed(self, value: int) -> None:
        algo = "kmeans"
        try:
            algo = normalize_cluster_algorithm(self.data.algorithm)
        except ValueError:
            pass
        if algo == "dbscan":
            return
        k = int(value)
        if k == int(self.data.n_clusters) and self.data.diagnostics.get("n_clusters") == k:
            return
        if k < 2 or k > len(self.data):
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            recluster(self.data, k, seed=self.data.cluster_seed, algorithm=algo)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            self.status.setText(f"clustering failed: {exc}")
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
            clusters=self._visible_clusters() if self.cluster_boxes else None,
        )
        if hasattr(self, "display_slider") and hasattr(self, "_display_perm"):
            pct = int(self.display_slider.value())
            n = len(self.data)
            take = max(1, min(n, int(round(n * pct / 100.0))))
            take = min(take, int(self._display_perm.size))
            drawn = np.zeros(n, dtype=np.bool_)
            drawn[self._display_perm[:take]] = True
            if self.selection.order:
                drawn[np.asarray(self.selection.order, dtype=np.int64)] = True
            mask = mask & drawn
        self._shown = np.flatnonzero(mask).astype(np.int64, copy=False)
        self._update_display_coords()
        labels = self.data.cluster_id[self._shown]
        self._scatter_brushes = [
            _qcolor(cluster_color(int(c), self.data.n_clusters), UNSELECTED_ALPHA) for c in labels
        ]
        self._apply_display_coords()
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
        xs = self._disp_x[idx]
        ys = self._disp_y[idx]
        pens = []
        brushes = []
        for i in idx:
            color = cluster_color(int(self.data.cluster_id[i]), self.data.n_clusters)
            pens.append(pg.mkPen(self.theme.selected_pen, width=2.4))
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
        dx = (self._disp_x[self._shown] - float(x)) / sx
        dy = (self._disp_y[self._shown] - float(y)) / sy
        d2 = dx * dx + dy * dy
        j = int(np.argmin(d2))
        if float(d2[j]) > CLICK_RADIUS_PX * CLICK_RADIUS_PX:
            return None
        return int(self._shown[j])

    def _on_click(self, event) -> None:  # noqa: ANN001
        if self._view3d:
            return
        if event.button() != Qt.MouseButton.LeftButton or event.double():
            return
        view_pt = self.plot.getViewBox().mapSceneToView(event.scenePos())
        idx = self._nearest(view_pt.x(), view_pt.y())
        if idx is None:
            return
        event.accept()
        self._toggle(idx)

    def _on_move(self, pos) -> None:  # noqa: ANN001
        if self._orbit_dragging:
            return
        view_pt = self.plot.getViewBox().mapSceneToView(pos)
        idx = self._nearest(view_pt.x(), view_pt.y())
        self._hover_idx = idx
        if idx is None:
            QToolTip.hideText()
            self._refresh_status()
            return
        info = self.data.row(idx)
        fen = _short_fen(info.fen, 36)
        if info.coord_z is not None:
            xyz = f"({info.coord_x:+.3f}, {info.coord_y:+.3f}, {info.coord_z:+.3f})"
        else:
            xyz = f"({info.coord_x:+.3f}, {info.coord_y:+.3f})"
        text = (
            f"#{info.sample_id}  cluster {info.cluster_id}\n"
            f"{xyz}\n"
            f"{fen}"
        )
        scene_pt = self.plot.getViewBox().mapViewToScene(view_pt)
        widget_pt = self.plot.mapFromScene(scene_pt)
        QToolTip.showText(self.plot.mapToGlobal(widget_pt), text, self.plot)
        self._refresh_status()

    def _toggle(self, index: int) -> None:
        selected = self.selection.toggle(index)
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
        grads = None
        if self.predictor is not None:
            grads = self.predictor.weight_grads(
                self.data.folders,
                info.slice_id,
                info.local_row,
                fen=info.fen,
                eval_target=info.eval_target,
            )
        if grads is None:
            grads = toy_weight_grads(int(info.sample_id))
        grads = standardize_weight_grads(grads)
        card = BoardCard(info, color, theme=self.theme, grads=grads)
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

    def _refresh_status(self) -> None:
        hover = ""
        if self._hover_idx is not None:
            info = self.data.row(self._hover_idx)
            hover = f"  ·  hover #{info.sample_id} c{info.cluster_id}"
        sizes = self.data.diagnostics.get("sizes")
        size_txt = f"  ·  sizes {sizes}" if sizes else ""
        noise = self.data.diagnostics.get("noise")
        noise_txt = f"  ·  noise {noise}" if noise else ""
        algo = CLUSTER_ALGO_LABELS.get(str(self.data.algorithm), str(self.data.algorithm))
        pct = int(self.display_slider.value()) if hasattr(self, "display_slider") else 100
        self.status.setText(
            f"{len(self._shown):,} / {len(self.data):,} shown ({pct}%)"
            f"  ·  {self.data.n_source_rows:,} in run"
            f"  ·  {algo}"
            f"  ·  B={self.data.n_clusters}"
            f"  ·  {len(self.selection)} pinned"
            f"  ·  {self.data.method}"
            f"{' 3D' if self._view3d else ''}"
            f"{size_txt}{noise_txt}{hover}"
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
        x = float(self._disp_x[index])
        y = float(self._disp_y[index])
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
    checkpoint: Path | None = None,
    argv: list[str] | None = None,
) -> int:
    app = QApplication.instance()
    owns = app is None
    if owns:
        app = QApplication([] if argv is None else argv)
        app.setApplicationName("SARDINE cluster explorer")
    win = ClusterExplorerWindow(data, checkpoint=checkpoint)
    win.show()
    win.raise_()
    win.activateWindow()
    if owns:
        return int(app.exec())
    return 0
