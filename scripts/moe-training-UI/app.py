"""MoE training UI.

A small PyQt6 window to configure a hard-MoE training run (base model,
clustering, dispatcher, expert fine-tuning) and watch live per-stage progress
(percent + dispatcher accuracy + expert / MoE cross-entropy).

Run it from the repo root::

    python scripts/moe-training-UI/run.py
"""

from __future__ import annotations

import io
import math
import re
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from PyQt6.QtCore import QObject, QSettings, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QColor, QDesktopServices, QImage, QPalette, QPixmap, QWheelEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from runner import (
    CancelToken,
    CancelledError,
    TrainingConfig,
    iter_checkpoints_info,
    iter_dispatchers,
    run_moe_training,
)

STAGE_NAMES = ["loading", "base", "world_model", "cluster", "dispatcher", "experts", "switch", "eval"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class _Relay(QObject):
    event = pyqtSignal(object)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)


def _fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"ETA {max(1, int(seconds))}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"ETA {m}m {s:02d}s"
    h, rem = divmod(int(seconds), 3600)
    return f"ETA {h}h {rem // 60:02d}m"


class _LogStepDoubleSpinBox(QDoubleSpinBox):
    """Double spin box whose mouse-wheel scroll multiplies/divides by 10."""

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        factor = 10.0 if delta > 0 else 0.1
        self.setValue(self.value() * factor)
        event.accept()


class _StageRow(QWidget):
    def __init__(self, name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.name = name
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(2)

        top = QHBoxLayout()
        title = QLabel(name)
        title.setMinimumWidth(90)
        self.metric = QLabel("—")
        self.metric.setStyleSheet("color: #9aa6b8;")
        self.eta = QLabel("")
        self.eta.setStyleSheet("color: #6a7384;")
        self.eta.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.eta.setMinimumWidth(92)
        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(self.metric)
        top.addWidget(self.eta)
        layout.addLayout(top)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(True)
        layout.addWidget(self.bar)

        self._start_time: float | None = None

    def set_progress(self, fraction: float | None) -> None:
        if fraction is None:
            self.bar.setRange(0, 0)
            self._clear_eta()
        else:
            self.bar.setRange(0, 100)
            self.bar.setValue(int(round(100.0 * max(0.0, min(1.0, fraction)))))
            self._update_eta(float(fraction))

    def _update_eta(self, fraction: float) -> None:
        if fraction <= 0.0:
            self._start_time = time.perf_counter()
            self.eta.setText("")
            return
        if fraction >= 1.0:
            self.eta.setText("")
            return
        if self._start_time is None:
            self._start_time = time.perf_counter()
        elapsed = time.perf_counter() - self._start_time
        if elapsed < 1.0:
            self.eta.setText("")
            return
        remaining = elapsed * (1.0 - fraction) / fraction
        self.eta.setText(_fmt_eta(remaining))

    def _clear_eta(self) -> None:
        self._start_time = None
        self.eta.setText("")

    def set_metric(self, text: str) -> None:
        self.metric.setText(text)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MoE Training")
        self.resize(1080, 720)

        self._relay = _Relay(self)
        self._relay.event.connect(self._on_event)
        self._relay.finished.connect(self._on_finished)
        self._relay.failed.connect(self._on_failed)

        self._cancel = CancelToken()
        self._thread: threading.Thread | None = None
        self._last_run_name: str | None = None
        self._live_key: str | None = None
        self._live_title = ""
        self._live_ylabel = ""
        self._live_series: dict[str, list[tuple[float, float]]] = {}
        self._plots: dict[str, QPixmap] = {}

        central = QWidget()
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._build_config())
        split.addWidget(self._build_progress())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([420, 640])
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.addWidget(split)
        self.setCentralWidget(central)

        self.status = self.statusBar()
        self.status.showMessage("Ready")

        self.settings = QSettings("tinymlinternship", "moe_training_ui")
        self._apply_theme()
        self._load_settings()
        self._sync_enabled_state()
        self._refresh_run_name()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_settings()
        super().closeEvent(event)

    # ---------------------------------------------------------------- config
    def _build_config(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        layout = QVBoxLayout(host)

        # Technique
        tech_group = QGroupBox("Technique")
        tf = QFormLayout(tech_group)
        self.technique = QComboBox()
        self.technique.addItem("Hard MoE (cluster + router)", "hard_moe")
        self.technique.addItem("Switch (end-to-end top-1)", "switch")
        self.technique.currentIndexChanged.connect(self._sync_enabled_state)
        self.k_combo = QComboBox()
        self.k_combo.addItem("1 (no MoE)", 1)
        for k in (2, 4, 8, 16):
            self.k_combo.addItem(str(k), k)
        self.k_combo.setCurrentIndex(3)  # 8
        tf.addRow("Method", self.technique)
        tf.addRow("Heads K", self.k_combo)
        layout.addWidget(tech_group)
        self.tech_group = tech_group

        # Base model
        base_group = QGroupBox("Base model")
        bf = QFormLayout(base_group)
        self.base_source = QComboBox()
        self.base_source.addItems(["Load checkpoint", "Train new"])
        self.base_source.currentIndexChanged.connect(self._sync_enabled_state)
        self.checkpoint = QComboBox()
        self._refresh_checkpoints()
        self.h_spin = QSpinBox()
        self.h_spin.setRange(16, 512)
        self.h_spin.setValue(128)
        self.H_spin = QSpinBox()
        self.H_spin.setRange(16, 1024)
        self.H_spin.setValue(256)
        self.base_epochs = QSpinBox()
        self.base_epochs.setRange(1, 200)
        self.base_epochs.setValue(20)
        bf.addRow("Source", self.base_source)
        bf.addRow("Checkpoint", self.checkpoint)
        bf.addRow("h (accumulator)", self.h_spin)
        bf.addRow("H (hidden)", self.H_spin)
        bf.addRow("Base epochs", self.base_epochs)
        layout.addWidget(base_group)

        # Encoder (L1): standard accumulator vs self-supervised world model
        enc_group = QGroupBox("Encoder (L1)")
        enf = QFormLayout(enc_group)
        self.enc_layout = enf
        self.encoder = QComboBox()
        self.encoder.addItem("Standard L1 accumulator", "standard")
        self.encoder.addItem("World model (self-supervised)", "world_model")
        self.encoder.currentIndexChanged.connect(self._sync_enabled_state)
        self.world_model_source = QComboBox()
        self.world_model_source.addItem("Train", "train")
        self.world_model_source.addItem("Load", "load")
        self.world_model_source.currentIndexChanged.connect(self._sync_enabled_state)
        self.wm_checkpoint = QComboBox()
        self._refresh_world_models()
        self.wm_loss = QComboBox()
        self.wm_loss.addItem("InfoNCE", "infonce")
        self.wm_loss.addItem("VICReg", "vicreg")
        self.wm_tau = _LogStepDoubleSpinBox()
        self.wm_tau.setDecimals(3)
        self.wm_tau.setRange(0.01, 1.0)
        self.wm_tau.setValue(0.1)
        self.wm_epochs = QSpinBox()
        self.wm_epochs.setRange(1, 200)
        self.wm_epochs.setValue(5)
        self.wm_lr = _LogStepDoubleSpinBox()
        self.wm_lr.setDecimals(4)
        self.wm_lr.setRange(1e-5, 1.0)
        self.wm_lr.setValue(1e-3)
        self.wm_normalize = QCheckBox("Unit-hypersphere normalize")
        self.wm_normalize.setChecked(True)
        self.wm_vicreg_gamma = QDoubleSpinBox()
        self.wm_vicreg_gamma.setDecimals(2)
        self.wm_vicreg_gamma.setRange(0.1, 5.0)
        self.wm_vicreg_gamma.setValue(1.0)
        enf.addRow("Type", self.encoder)
        enf.addRow("Source", self.world_model_source)
        enf.addRow("World model", self.wm_checkpoint)
        enf.addRow("Loss", self.wm_loss)
        enf.addRow("Temperature τ", self.wm_tau)
        enf.addRow("Epochs", self.wm_epochs)
        enf.addRow("LR", self.wm_lr)
        enf.addRow("", self.wm_normalize)
        enf.addRow("VICReg γ", self.wm_vicreg_gamma)
        self.refresh_models_btn = QPushButton("Refresh model list")
        self.refresh_models_btn.clicked.connect(self._refresh_models)
        enf.addRow("", self.refresh_models_btn)
        layout.addWidget(enc_group)
        self.enc_group = enc_group

        # Data
        data_group = QGroupBox("Data")
        df = QFormLayout(data_group)
        self.max_rows = QSpinBox()
        self.max_rows.setRange(1, 100)
        self.max_rows.setSuffix(" M")
        self.max_rows.setValue(2)
        self.max_test = QSpinBox()
        self.max_test.setRange(1, 500)
        self.max_test.setSuffix(" k")
        self.max_test.setValue(50)
        df.addRow("Train rows", self.max_rows)
        df.addRow("Test rows", self.max_test)
        layout.addWidget(data_group)

        # Clustering
        cluster_group = QGroupBox("Clustering")
        cf = QFormLayout(cluster_group)
        self.clustering = QComboBox()
        self.clustering.addItem("Sample gradients (48-d)", "gradient")
        self.clustering.addItem("L1 activations (256-d)", "l1")
        self.clustering.addItem("Piece count (rule)", "piece_count")
        self.clustering.currentIndexChanged.connect(self._sync_enabled_state)
        cf.addRow("Representation", self.clustering)
        layout.addWidget(cluster_group)
        self.cluster_group = cluster_group

        # Dispatcher
        disp_group = QGroupBox("Dispatcher")
        pf = QFormLayout(disp_group)
        self.disp_layout = pf
        self.dispatcher_source = QComboBox()
        self.dispatcher_source.addItem("Train", "train")
        self.dispatcher_source.addItem("Load", "load")
        self.dispatcher_source.currentIndexChanged.connect(self._sync_enabled_state)
        self.dispatcher_checkpoint = QComboBox()
        self._refresh_dispatchers()
        self.dispatcher_type = QComboBox()
        self.dispatcher_type.addItem("MLP", "mlp")
        self.dispatcher_type.addItem("Linear", "linear")
        self.dispatcher_type.currentIndexChanged.connect(self._sync_enabled_state)
        self.dispatcher_hidden = QComboBox()
        for h in (32, 64, 128):
            self.dispatcher_hidden.addItem(str(h), h)
        self.dispatcher_hidden.setCurrentIndex(1)  # 64
        self.dispatcher_epochs = QSpinBox()
        self.dispatcher_epochs.setRange(1, 1000)
        self.dispatcher_epochs.setValue(8)
        self.dispatcher_lr = _LogStepDoubleSpinBox()
        self.dispatcher_lr.setDecimals(4)
        self.dispatcher_lr.setRange(1e-5, 1.0)
        self.dispatcher_lr.setValue(1e-2)
        pf.addRow("Source", self.dispatcher_source)
        pf.addRow("Dispatcher", self.dispatcher_checkpoint)
        pf.addRow("Type", self.dispatcher_type)
        pf.addRow("Hidden width", self.dispatcher_hidden)
        pf.addRow("Epochs", self.dispatcher_epochs)
        pf.addRow("LR", self.dispatcher_lr)
        layout.addWidget(disp_group)
        self.disp_group = disp_group

        # Experts
        exp_group = QGroupBox("Experts")
        ef = QFormLayout(exp_group)
        self.expert_epochs = QSpinBox()
        self.expert_epochs.setRange(1, 1000)
        self.expert_epochs.setValue(2)
        self.expert_lr = _LogStepDoubleSpinBox()
        self.expert_lr.setDecimals(4)
        self.expert_lr.setRange(1e-5, 1.0)
        self.expert_lr.setValue(1e-3)
        self.expert_lr_end = _LogStepDoubleSpinBox()
        self.expert_lr_end.setDecimals(5)
        self.expert_lr_end.setRange(0.0, 1.0)
        self.expert_lr_end.setValue(1e-4)
        self.expert_optimizer = QComboBox()
        self.expert_optimizer.addItem("Adam", "adam")
        self.expert_optimizer.addItem("SGD", "sgd")
        self.expert_optimizer.addItem("Rprop", "rprop")
        self.expert_optimizer.currentIndexChanged.connect(self._sync_enabled_state)
        self.l1_frozen = QCheckBox("Freeze L1")
        self.l1_frozen.setChecked(True)
        ef.addRow("Epochs", self.expert_epochs)
        ef.addRow("LR", self.expert_lr)
        ef.addRow("LR end", self.expert_lr_end)
        ef.addRow("Optimizer", self.expert_optimizer)
        ef.addRow("", self.l1_frozen)
        layout.addWidget(exp_group)
        self.exp_group = exp_group

        # Switch (end-to-end top-1)
        switch_group = QGroupBox("Switch (top-1)")
        sf = QFormLayout(switch_group)
        self.switch_alpha = QDoubleSpinBox()
        self.switch_alpha.setDecimals(4)
        self.switch_alpha.setRange(0.0, 1.0)
        self.switch_alpha.setValue(0.01)
        self.switch_epochs = QSpinBox()
        self.switch_epochs.setRange(1, 100)
        self.switch_epochs.setValue(5)
        self.switch_lr = _LogStepDoubleSpinBox()
        self.switch_lr.setDecimals(4)
        self.switch_lr.setRange(1e-5, 1.0)
        self.switch_lr.setValue(1e-3)
        sf.addRow("Load-balancing α", self.switch_alpha)
        sf.addRow("Epochs", self.switch_epochs)
        sf.addRow("LR", self.switch_lr)
        layout.addWidget(switch_group)
        self.switch_group = switch_group

        # Misc
        misc_group = QGroupBox("Run")
        mf = QFormLayout(misc_group)
        self.device = QComboBox()
        self.device.addItem("auto", "auto")
        self.device.addItem("cuda", "cuda")
        self.device.addItem("cpu", "cpu")
        self.run_name_label = QLabel("")
        self.run_name_label.setWordWrap(True)
        self.run_name_label.setStyleSheet("color: #9aa6b8;")
        mf.addRow("Device", self.device)
        mf.addRow("Run name", self.run_name_label)
        layout.addWidget(misc_group)

        for w in (
            self.technique, self.base_source, self.checkpoint, self.clustering,
            self.k_combo, self.dispatcher_source, self.dispatcher_checkpoint,
            self.dispatcher_type, self.dispatcher_hidden,
            self.expert_optimizer,
            self.encoder, self.world_model_source, self.wm_checkpoint, self.wm_loss,
        ):
            w.currentIndexChanged.connect(self._refresh_run_name)
        for w in (
            self.h_spin, self.H_spin, self.base_epochs, self.max_rows,
            self.max_test, self.expert_epochs, self.switch_epochs,
            self.wm_epochs, self.wm_vicreg_gamma,
        ):
            w.valueChanged.connect(self._refresh_run_name)
        for w in (self.switch_alpha, self.switch_lr, self.wm_tau, self.wm_lr):
            w.valueChanged.connect(self._refresh_run_name)
        self._refresh_run_name()

        layout.addStretch(1)
        scroll.setWidget(host)
        return scroll

    def _build_progress(self) -> QWidget:
        host = QWidget()
        layout = QVBoxLayout(host)

        top_split = QSplitter(Qt.Orientation.Horizontal)
        top_split.addWidget(self._build_stats_panel())
        top_split.addWidget(self._build_console_panel())
        top_split.setStretchFactor(0, 1)
        top_split.setStretchFactor(1, 1)
        top_split.setSizes([440, 308])
        top_split.setChildrenCollapsible(False)

        plot_group = QGroupBox("Results — CE comparison")
        pl = QVBoxLayout(plot_group)
        self.plot_combo = QComboBox()
        self.plot_combo.setEnabled(False)
        self.plot_combo.currentIndexChanged.connect(self._on_plot_selected)
        pl.addWidget(self.plot_combo)
        self.plot_label = QLabel("Run a training to see the CE comparison")
        self.plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plot_label.setScaledContents(True)
        self.plot_label.setMinimumHeight(330)
        pl.addWidget(self.plot_label)

        vertical = QSplitter(Qt.Orientation.Vertical)
        vertical.addWidget(top_split)
        vertical.addWidget(plot_group)
        vertical.setStretchFactor(0, 3)
        vertical.setStretchFactor(1, 2)
        vertical.setSizes([360, 330])
        vertical.setChildrenCollapsible(False)
        layout.addWidget(vertical)

        return host

    def _build_stats_panel(self) -> QWidget:
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)

        self.stage_rows: dict[str, _StageRow] = {}
        for name in STAGE_NAMES:
            row = _StageRow(name)
            self.stage_rows[name] = row
            layout.addWidget(row)

        self.result_label = QLabel("")
        self.result_label.setStyleSheet(
            "padding: 8px; background: #1b1f27; border-radius: 4px; font-family: monospace;"
        )
        self.result_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.result_label)

        buttons = QHBoxLayout()
        self.run_btn = QPushButton("RUN")
        self.run_btn.setObjectName("runButton")
        self.run_btn.clicked.connect(self._start)
        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        self.open_results_btn = QPushButton("Open results folder")
        self.open_results_btn.setVisible(False)
        self.open_results_btn.clicked.connect(self._open_results)
        buttons.addWidget(self.run_btn)
        buttons.addWidget(self.stop_btn)
        buttons.addWidget(self.open_results_btn)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        layout.addStretch(1)
        return host

    def _build_console_panel(self) -> QWidget:
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        layout.addWidget(self.log, 1)
        return host

    # ---------------------------------------------------------------- helpers
    def _refresh_checkpoints(self) -> None:
        self.checkpoint.clear()
        for info in iter_checkpoints_info():
            badge = " [world model]" if info["is_world_model"] else ""
            self.checkpoint.addItem(info["name"] + badge, info["path"])

    def _refresh_world_models(self) -> None:
        self.wm_checkpoint.clear()
        for info in iter_checkpoints_info():
            if info["is_world_model"]:
                dims = f"W{info['hidden_dim']}_H{info['hidden2_dim']}" if info["hidden_dim"] > 0 else "?"
                self.wm_checkpoint.addItem(f"{info['name']} ({dims})", info["path"])

    def _refresh_dispatchers(self) -> None:
        self.dispatcher_checkpoint.clear()
        for info in iter_dispatchers():
            label = (
                f"{info['name']} ({info['dispatcher_type']}, "
                f"in{info['in_dim']}, K{info['n_clusters']})"
            )
            self.dispatcher_checkpoint.addItem(label, info["path"])

    def _refresh_models(self) -> None:
        self._refresh_checkpoints()
        self._refresh_world_models()
        self._refresh_dispatchers()

    # ---------------------------------------------------------------- settings
    def _save_settings(self) -> None:
        s = self.settings
        s.setValue("technique", self.technique.currentData())
        s.setValue("base_source", self.base_source.currentIndex())
        s.setValue("checkpoint", self.checkpoint.currentData() or "")
        s.setValue("h", self.h_spin.value())
        s.setValue("H", self.H_spin.value())
        s.setValue("base_epochs", self.base_epochs.value())
        s.setValue("encoder", self.encoder.currentData())
        s.setValue("world_model_source", self.world_model_source.currentData())
        s.setValue("wm_checkpoint", self.wm_checkpoint.currentData() or "")
        s.setValue("wm_loss", self.wm_loss.currentData())
        s.setValue("wm_tau", self.wm_tau.value())
        s.setValue("wm_epochs", self.wm_epochs.value())
        s.setValue("wm_lr", self.wm_lr.value())
        s.setValue("wm_normalize", self.wm_normalize.isChecked())
        s.setValue("wm_vicreg_gamma", self.wm_vicreg_gamma.value())
        s.setValue("max_rows", self.max_rows.value())
        s.setValue("max_test", self.max_test.value())
        s.setValue("clustering", self.clustering.currentData())
        s.setValue("k", self.k_combo.currentData())
        s.setValue("dispatcher_source", self.dispatcher_source.currentData())
        s.setValue("dispatcher_checkpoint", self.dispatcher_checkpoint.currentData() or "")
        s.setValue("dispatcher_type", self.dispatcher_type.currentData())
        s.setValue("dispatcher_hidden", self.dispatcher_hidden.currentData())
        s.setValue("dispatcher_epochs", self.dispatcher_epochs.value())
        s.setValue("dispatcher_lr", self.dispatcher_lr.value())
        s.setValue("expert_epochs", self.expert_epochs.value())
        s.setValue("expert_lr", self.expert_lr.value())
        s.setValue("expert_lr_end", self.expert_lr_end.value())
        s.setValue("expert_optimizer", self.expert_optimizer.currentData())
        s.setValue("l1_frozen", self.l1_frozen.isChecked())
        s.setValue("switch_alpha", self.switch_alpha.value())
        s.setValue("switch_epochs", self.switch_epochs.value())
        s.setValue("switch_lr", self.switch_lr.value())
        s.setValue("device", self.device.currentData())

    def _restore_combo_data(self, combo: QComboBox, value) -> None:
        if value is None:
            return
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _restore_spin(self, spin: QSpinBox, value) -> None:
        if value is None:
            return
        try:
            spin.setValue(int(value))
        except (TypeError, ValueError):
            pass

    def _restore_double(self, spin: QDoubleSpinBox, value) -> None:
        if value is None:
            return
        try:
            spin.setValue(float(value))
        except (TypeError, ValueError):
            pass

    def _load_settings(self) -> None:
        s = self.settings
        self._restore_combo_data(self.technique, s.value("technique"))
        base_source = s.value("base_source")
        if isinstance(base_source, int) and 0 <= base_source < self.base_source.count():
            self.base_source.setCurrentIndex(base_source)
        self._restore_combo_data(self.checkpoint, s.value("checkpoint"))
        self._restore_spin(self.h_spin, s.value("h"))
        self._restore_spin(self.H_spin, s.value("H"))
        self._restore_spin(self.base_epochs, s.value("base_epochs"))
        self._restore_combo_data(self.encoder, s.value("encoder"))
        self._restore_combo_data(self.world_model_source, s.value("world_model_source"))
        self._restore_combo_data(self.wm_checkpoint, s.value("wm_checkpoint"))
        self._restore_combo_data(self.wm_loss, s.value("wm_loss"))
        self._restore_double(self.wm_tau, s.value("wm_tau"))
        self._restore_spin(self.wm_epochs, s.value("wm_epochs"))
        self._restore_double(self.wm_lr, s.value("wm_lr"))
        normalize = s.value("wm_normalize")
        if normalize is not None:
            self.wm_normalize.setChecked(normalize in (True, "true", "1", 1))
        self._restore_double(self.wm_vicreg_gamma, s.value("wm_vicreg_gamma"))
        self._restore_spin(self.max_rows, s.value("max_rows"))
        self._restore_spin(self.max_test, s.value("max_test"))
        self._restore_combo_data(self.clustering, s.value("clustering"))
        self._restore_combo_data(self.k_combo, s.value("k"))
        self._restore_combo_data(self.dispatcher_source, s.value("dispatcher_source"))
        self._restore_combo_data(self.dispatcher_checkpoint, s.value("dispatcher_checkpoint"))
        self._restore_combo_data(self.dispatcher_type, s.value("dispatcher_type"))
        self._restore_combo_data(self.dispatcher_hidden, s.value("dispatcher_hidden"))
        self._restore_spin(self.dispatcher_epochs, s.value("dispatcher_epochs"))
        self._restore_double(self.dispatcher_lr, s.value("dispatcher_lr"))
        self._restore_spin(self.expert_epochs, s.value("expert_epochs"))
        self._restore_double(self.expert_lr, s.value("expert_lr"))
        self._restore_double(self.expert_lr_end, s.value("expert_lr_end"))
        self._restore_combo_data(self.expert_optimizer, s.value("expert_optimizer"))
        frozen = s.value("l1_frozen")
        if frozen is not None:
            self.l1_frozen.setChecked(frozen in (True, "true", "1", 1))
        self._restore_double(self.switch_alpha, s.value("switch_alpha"))
        self._restore_spin(self.switch_epochs, s.value("switch_epochs"))
        self._restore_double(self.switch_lr, s.value("switch_lr"))
        self._restore_combo_data(self.device, s.value("device"))

    def _sync_enabled_state(self) -> None:
        is_switch = self.technique.currentData() == "switch"
        is_world_model = self.encoder.currentData() == "world_model"
        wm_load = is_world_model and self.world_model_source.currentData() == "load"
        eff_k = int(self.k_combo.currentData())
        no_moe = eff_k <= 1

        show_moe = not no_moe
        self.cluster_group.setVisible(show_moe and not is_switch)
        self.disp_group.setVisible(show_moe and not is_switch)
        self.exp_group.setVisible(show_moe and not is_switch)
        self.switch_group.setVisible(show_moe and is_switch)

        for name in ("cluster", "dispatcher", "experts"):
            self.stage_rows[name].setVisible(show_moe and not is_switch)
        self.stage_rows["switch"].setVisible(show_moe and is_switch)
        self.stage_rows["world_model"].setVisible(is_world_model and not wm_load)

        # Encoder group: world-model controls
        self._set_row_visible(self.wm_checkpoint, wm_load)
        wm_train = is_world_model and not wm_load
        for w in (self.wm_loss, self.wm_tau, self.wm_epochs, self.wm_lr, self.wm_normalize, self.wm_vicreg_gamma):
            self._set_row_visible(w, wm_train)

        # Base model group: when loading a world model, the checkpoint decides
        # the architecture, so the base source / dims are irrelevant.
        if wm_load:
            self.base_source.setEnabled(False)
            self.checkpoint.setEnabled(False)
            self.h_spin.setEnabled(False)
            self.H_spin.setEnabled(False)
            self.base_epochs.setEnabled(False)
        else:
            training_new = self.base_source.currentIndex() == 1
            self.checkpoint.setEnabled(not training_new)
            self.h_spin.setEnabled(training_new)
            self.H_spin.setEnabled(training_new)
            self.base_epochs.setEnabled(training_new)

        piece = self.clustering.currentData() == "piece_count"
        if piece and not no_moe:
            self.k_combo.setCurrentIndex(3)  # lock to K=8
        # Heads K stays editable in every mode so the user can always switch
        # between "1 (no MoE)" and a real MoE; only piece-count pins it to 8.
        self.k_combo.setEnabled(not piece)

        # Dispatcher: piece-count is rule-based (no dispatcher), so hide the
        # whole group; otherwise show the load-vs-train source toggle.
        self.disp_group.setVisible(show_moe and not is_switch and not piece)
        disp_active = show_moe and not is_switch and not piece
        disp_load = self.dispatcher_source.currentData() == "load"
        self._set_row_visible(self.dispatcher_checkpoint, disp_load, self.disp_layout)
        for w in (self.dispatcher_type, self.dispatcher_hidden, self.dispatcher_epochs, self.dispatcher_lr):
            self._set_row_visible(w, not disp_load, self.disp_layout)
        self.dispatcher_source.setEnabled(disp_active)
        self.dispatcher_checkpoint.setEnabled(disp_active and disp_load)
        self.dispatcher_type.setEnabled(disp_active and not disp_load)
        self.dispatcher_hidden.setEnabled(
            disp_active and not disp_load and self.dispatcher_type.currentData() == "mlp"
        )
        self.dispatcher_epochs.setEnabled(disp_active and not disp_load)
        self.dispatcher_lr.setEnabled(disp_active and not disp_load)

        # Rprop has no learning-rate schedule, so "LR end" is irrelevant.
        self.expert_lr_end.setEnabled(self.expert_optimizer.currentData() != "rprop")

    def _set_row_visible(self, widget: QWidget, visible: bool, layout: QFormLayout | None = None) -> None:
        layout = layout if layout is not None else self.enc_layout
        layout.setRowVisible(widget, visible)
        widget.setVisible(visible)

    def _checkpoint_name(self) -> str:
        data = self.checkpoint.currentData()
        return Path(data).parent.name if data else ""

    def _wm_name(self) -> str:
        data = self.wm_checkpoint.currentData()
        return Path(data).parent.name if data else ""

    def _dispatcher_name(self) -> str:
        data = self.dispatcher_checkpoint.currentData()
        return Path(data).parent.name if data else ""

    def _auto_run_name(self) -> str:
        parts: list[str] = []
        if self.encoder.currentData() == "world_model":
            parts.append("wm")
            if self.world_model_source.currentData() == "load":
                wm = self._wm_name()
                parts.append(f"load_{wm}" if wm else "load")
            else:
                parts.append(self.wm_loss.currentData())
                parts.append(f"tau{self.wm_tau.value():g}")
                parts.append(f"ep{self.wm_epochs.value()}")
                if self.base_source.currentIndex() == 1:
                    parts.append(f"h{self.h_spin.value()}_H{self.H_spin.value()}_be{self.base_epochs.value()}")
                else:
                    ckpt = self._checkpoint_name()
                    parts.append(f"from_{ckpt}" if ckpt else "from_base")
        else:
            if self.base_source.currentIndex() == 1:
                parts.append(
                    f"new_h{self.h_spin.value()}_H{self.H_spin.value()}_be{self.base_epochs.value()}"
                )
            else:
                ckpt = self._checkpoint_name()
                parts.append(f"load_{ckpt}" if ckpt else "load")
        if self.technique.currentData() == "switch":
            if int(self.k_combo.currentData()) <= 1:
                parts.append("base")
            else:
                parts.append("switch")
                parts.append(f"k{int(self.k_combo.currentData())}")
                parts.append(f"a{self.switch_alpha.value():g}")
                parts.append(f"ep{self.switch_epochs.value()}")
                parts.append(f"lr{self.switch_lr.value():g}")
        else:
            if int(self.k_combo.currentData()) <= 1:
                parts.append("base")
            else:
                parts.append(str(self.clustering.currentData()))
                parts.append(f"k{int(self.k_combo.currentData())}")
                if self.clustering.currentData() != "piece_count":
                    if self.dispatcher_source.currentData() == "load":
                        disp = self._dispatcher_name()
                        parts.append(f"disp_{disp}" if disp else "disp_load")
                    else:
                        if self.dispatcher_type.currentData() == "mlp":
                            parts.append(f"mlp{int(self.dispatcher_hidden.currentData())}")
                        else:
                            parts.append("linear")
                parts.append(f"ep{self.expert_epochs.value()}")
                parts.append(str(self.expert_optimizer.currentData()))
        parts.append(f"{self.max_rows.value()}M_{self.max_test.value()}k")
        name = "_".join(parts)
        return re.sub(r"[^A-Za-z0-9._-]", "_", name)

    def _refresh_run_name(self, *_args) -> None:
        self.run_name_label.setText(self._auto_run_name())

    def _collect_config(self) -> TrainingConfig:
        technique = self.technique.currentData()
        k = int(self.k_combo.currentData())
        return TrainingConfig(
            technique=technique,
            base_source="new" if self.base_source.currentIndex() == 1 else "load",
            base_checkpoint=self.checkpoint.currentData() or "",
            h=self.h_spin.value(),
            H=self.H_spin.value(),
            base_epochs=self.base_epochs.value(),
            encoder=self.encoder.currentData(),
            world_model_source=self.world_model_source.currentData(),
            wm_checkpoint=self.wm_checkpoint.currentData() or "",
            wm_loss=self.wm_loss.currentData(),
            wm_tau=float(self.wm_tau.value()),
            wm_epochs=self.wm_epochs.value(),
            wm_lr=float(self.wm_lr.value()),
            wm_normalize=self.wm_normalize.isChecked(),
            wm_vicreg_gamma=float(self.wm_vicreg_gamma.value()),
            max_rows=self.max_rows.value() * 1_000_000,
            max_test=self.max_test.value() * 1_000,
            clustering=self.clustering.currentData(),
            k=k,
            dispatcher_source=self.dispatcher_source.currentData(),
            dispatcher_checkpoint=self.dispatcher_checkpoint.currentData() or "",
            dispatcher_type=self.dispatcher_type.currentData(),
            dispatcher_hidden=int(self.dispatcher_hidden.currentData()),
            dispatcher_epochs=self.dispatcher_epochs.value(),
            dispatcher_lr=float(self.dispatcher_lr.value()),
            expert_epochs=self.expert_epochs.value(),
            expert_lr=float(self.expert_lr.value()),
            expert_lr_end=float(self.expert_lr_end.value()),
            expert_optimizer=self.expert_optimizer.currentData(),
            l1_frozen=self.l1_frozen.isChecked(),
            switch_alpha=float(self.switch_alpha.value()),
            switch_epochs=self.switch_epochs.value(),
            switch_lr=float(self.switch_lr.value()),
            device=self.device.currentData(),
            run_name=self._auto_run_name(),
        )

    # ---------------------------------------------------------------- run
    def _start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        cfg = self._collect_config()
        self._last_run_name = cfg.run_name
        self._save_settings()
        self._reset_stages()
        self._cancel = CancelToken()
        self._set_running(True)
        self._log(f"starting run '{cfg.run_name}'")
        self._thread = threading.Thread(target=self._worker, args=(cfg,), daemon=True)
        self._thread.start()

    def _worker(self, cfg: TrainingConfig) -> None:
        try:
            metrics = run_moe_training(cfg, self._relay.event.emit, self._cancel)
            self._relay.finished.emit(metrics)
        except CancelledError:
            self._relay.failed.emit("__cancelled__")
        except Exception:  # noqa: BLE001
            self._relay.failed.emit(traceback.format_exc())

    def _stop(self) -> None:
        self._cancel.cancel()
        self.status.showMessage("Stopping…")
        self._log("stop requested")

    def _open_results(self) -> None:
        if not self._last_run_name:
            return
        path = PROJECT_ROOT / "RESULTS" / "moe" / self._last_run_name
        if not path.is_dir():
            path = path.parent
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _set_running(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        for widget in (
            self.technique, self.base_source, self.checkpoint, self.h_spin,
            self.H_spin, self.base_epochs, self.max_rows, self.max_test,
            self.encoder, self.world_model_source, self.wm_checkpoint,
            self.wm_loss, self.wm_tau, self.wm_epochs, self.wm_lr,
            self.wm_normalize, self.wm_vicreg_gamma,
            self.clustering, self.k_combo, self.dispatcher_source,
            self.dispatcher_checkpoint, self.dispatcher_type,
            self.dispatcher_hidden, self.dispatcher_epochs, self.dispatcher_lr,
            self.expert_epochs, self.expert_lr, self.expert_lr_end,
            self.expert_optimizer,
            self.l1_frozen, self.switch_alpha,
            self.switch_epochs, self.switch_lr, self.device,
            self.refresh_models_btn,
        ):
            widget.setEnabled(False if running else True)
        if not running:
            self._sync_enabled_state()

    def _reset_stages(self) -> None:
        for row in self.stage_rows.values():
            row.set_progress(0.0)
            row.set_metric("—")
        self.result_label.setText("")
        self.plot_label.setPixmap(QPixmap())
        self.plot_label.setText("Run a training to see the CE comparison")
        self.open_results_btn.setVisible(False)
        self._live_key = None
        self._live_title = ""
        self._live_ylabel = ""
        self._live_series = {}
        self._plots = {}
        self.plot_combo.clear()
        self.plot_combo.setEnabled(False)

    def _log(self, text: str) -> None:
        self.log.appendPlainText(text)

    # ---------------------------------------------------------------- events
    def _on_event(self, ev: dict) -> None:
        # Any unhandled exception inside a Qt slot aborts the whole process
        # (SIGABRT), which surfaces as a "frozen" window. Catch it and keep the
        # UI alive instead.
        try:
            self._dispatch_event(ev)
        except Exception:  # noqa: BLE001
            self.status.showMessage("Error handling event")
            self._log("ERROR in UI event handler:\n" + traceback.format_exc())

    def _dispatch_event(self, ev: dict) -> None:
        stage = str(ev.get("stage", ""))
        event = str(ev.get("event", ""))

        if stage == "init":
            self.status.showMessage(f"Running on {ev.get('device')}")
            self._log(f"device={ev.get('device')}")
            return

        if stage == "base":
            row = self.stage_rows["base"]
            if event == "start":
                row.set_progress(0.0)
                row.set_metric("…")
            elif event == "start_train":
                row.set_metric(f"W{ev.get('h')}_H{ev.get('H')} · {ev.get('params'):,} params")
            elif event == "epoch":
                row.set_progress(ev.get("progress"))
                row.set_metric(f"epoch {ev.get('epoch')}/{ev.get('epochs')} · CE {ev.get('ce'):.4f}")
                self._add_live_point(
                    "base", "Base model", "CE", "CE",
                    float(ev.get("epoch")), float(ev.get("ce")),
                )
            elif event == "done":
                row.set_progress(1.0)
                row.set_metric(ev.get("msg", f"W{ev.get('h')}_H{ev.get('H')}"))
            return

        if stage == "world_model":
            row = self.stage_rows["world_model"]
            if event == "start":
                row.set_progress(0.0)
                row.set_metric(f"{ev.get('loss')} · normalize={ev.get('normalize')}")
            elif event == "epoch":
                row.set_progress(ev.get("progress"))
                row.set_metric(
                    f"epoch {ev.get('epoch')}/{ev.get('epochs')} · loss {ev.get('loss'):.4f}"
                )
                self._add_live_point(
                    "world_model", "World model (L1)", "loss", "loss",
                    float(ev.get("epoch")), float(ev.get("loss")),
                )
            elif event == "done":
                row.set_progress(1.0)
                diag = ev.get("diagnostics") or {}
                row.set_metric(f"eff. rank {diag.get('effective_rank', 'n/a')}")
            return

        if stage == "loading":
            row = self.stage_rows["loading"]
            if event == "start":
                row.set_progress(0.0)
                row.set_metric(ev.get("msg", ""))
            elif event == "progress":
                row.set_progress(ev.get("progress"))
                if ev.get("msg"):
                    row.set_metric(str(ev.get("msg")))
            elif event == "pack_done":
                row.set_progress(1.0)
                row.set_metric(f"{ev.get('rows'):,} rows")
            elif event == "skip":
                row.set_progress(1.0)
                row.set_metric(ev.get("msg", "skipped"))
            return

        if stage == "cluster":
            row = self.stage_rows["cluster"]
            if event == "start":
                row.set_progress(0.0)
                row.set_metric(f"{ev.get('clustering')} · K={ev.get('k')}")
            elif event == "progress":
                row.set_progress(ev.get("progress"))
                if ev.get("msg"):
                    row.set_metric(str(ev.get("msg")))
            elif event == "done":
                row.set_progress(1.0)
                sizes = ev.get("sizes") or []
                row.set_metric(f"sizes {sizes}")
            return

        if stage == "dispatcher":
            row = self.stage_rows["dispatcher"]
            if event == "start":
                row.set_progress(0.0)
                if ev.get("type") == "load":
                    row.set_metric(f"load {ev.get('checkpoint', '')}")
                else:
                    row.set_metric(f"type {ev.get('type')}")
            elif event == "epoch":
                row.set_progress(ev.get("progress"))
                row.set_metric(
                    f"epoch {ev.get('epoch')}/{ev.get('epochs')} · "
                    f"train {ev.get('train_acc'):.3f} · val {ev.get('val_acc'):.3f}"
                )
                self._add_live_point(
                    "dispatcher", "Dispatcher", "val acc", "val acc",
                    float(ev.get("epoch")), float(ev.get("val_acc")),
                )
            elif event == "done":
                row.set_progress(1.0)
                if ev.get("best_val_acc") is not None:
                    row.set_metric(f"val_acc {ev.get('best_val_acc'):.4f}")
                else:
                    row.set_metric("loaded")
            elif event == "skip":
                row.set_progress(1.0)
                row.set_metric(ev.get("msg", "rule-based"))
            return

        if stage == "expert":
            row = self.stage_rows["experts"]
            k = int(ev.get("k", 1))
            eid = int(ev.get("expert_id", 0))
            frac = float(ev.get("progress", 0.0))
            overall = (eid + frac) / max(k, 1)
            row.set_progress(overall)
            if event == "epoch":
                epoch = int(ev.get("epoch"))
                hold_ce = float(ev.get("hold_ce"))
                base_ce = float(ev.get("base_hold_ce"))
                pct = (hold_ce / base_ce * 100.0) if base_ce > 0.0 else 0.0
                row.set_metric(
                    f"expert {eid}/{k} · epoch {epoch}/{ev.get('epochs')} · "
                    f"{pct:.1f}% of base"
                )
                if epoch == 1:
                    self._add_live_point(
                        "experts", "Experts fine-tune", "hold CE (% of base)",
                        f"expert {eid}", 0.0, 100.0,
                    )
                self._add_live_point(
                    "experts", "Experts fine-tune", "hold CE (% of base)",
                    f"expert {eid}", float(epoch), pct,
                )
            elif event == "done":
                row.set_metric(
                    f"expert {eid}/{k} done · hold {ev.get('hold_ce'):.4f} "
                    f"(base {ev.get('base_hold_ce'):.4f})"
                )
            elif event == "skip":
                row.set_metric(str(ev.get("msg", f"expert {eid} skipped")))
            return

        if stage == "experts":
            row = self.stage_rows["experts"]
            if event == "start":
                row.set_progress(0.0)
                row.set_metric(f"0/{ev.get('k')}")
            elif event == "done":
                row.set_progress(1.0)
            return

        if stage == "switch":
            row = self.stage_rows["switch"]
            if event == "start":
                row.set_progress(0.0)
                row.set_metric(f"K={ev.get('k')} · α={ev.get('alpha')}")
            elif event == "epoch":
                row.set_progress(ev.get("progress"))
                row.set_metric(
                    f"epoch {ev.get('epoch')}/{ev.get('epochs')} · "
                    f"train {ev.get('train_ce'):.4f} · test {ev.get('test_ce'):.4f}"
                )
                self._add_live_point(
                    "switch", "Switch top-1", "CE", "train",
                    float(ev.get("epoch")), float(ev.get("train_ce")),
                )
                self._add_live_point(
                    "switch", "Switch top-1", "CE", "test",
                    float(ev.get("epoch")), float(ev.get("test_ce")),
                )
            elif event == "done":
                row.set_progress(1.0)
            return

        if stage == "eval":
            row = self.stage_rows["eval"]
            if event == "start":
                row.set_progress(0.0)
                row.set_metric("…")
            elif event == "progress":
                row.set_progress(ev.get("progress"))
                if ev.get("msg"):
                    row.set_metric(str(ev.get("msg")))
            elif event == "done":
                row.set_progress(1.0)
                parts = []
                for key, label in (
                    ("base_ce", "base"),
                    ("moe_ce", "moe"),
                    ("perfect_ce", "perfect"),
                    ("best_expert_ce", "best"),
                ):
                    if ev.get(key) is not None:
                        parts.append(f"{label} {ev.get(key):.5f}")
                row.set_metric(" · ".join(parts))
                self._show_plot(ev)
            return

        if stage == "done":
            summary = ev.get("summary", {})
            metrics = summary.get("metrics", {})
            self.result_label.setText(self._format_result(metrics))
            return

    def _moe_score(self, m: dict) -> float:
        base = m.get("base_ce")
        moe = m.get("moe_ce")
        best = m.get("best_expert_ce")
        if (
            isinstance(base, (int, float))
            and isinstance(moe, (int, float))
            and isinstance(best, (int, float))
        ):
            denom = base - best
            if denom != 0.0:
                return (base - moe) / denom
        return float("nan")

    def _format_result(self, m: dict) -> str:
        if not m:
            return ""
        base = m.get("base_ce", float("nan"))
        moe = m.get("moe_ce", float("nan"))
        diff = moe - base if isinstance(moe, float) and isinstance(base, float) else float("nan")
        arrow = "↓" if diff < 0 else "↑"
        line1 = f"base {base:.5f}   ·   MoE {moe:.5f}"
        if m.get("perfect_ce") is not None:
            line1 += f"   ·   perfect {m['perfect_ce']:.5f}"
        if m.get("best_expert_ce") is not None:
            line1 += f"   ·   best expert {m['best_expert_ce']:.5f}"
        score = self._moe_score(m)
        score_txt = f"{score * 100.0:.1f}%" if math.isfinite(score) else "n/a"
        return (
            f"{line1}\n"
            f"base MAE {m.get('base_mae', float('nan')):.5f}   ·   "
            f"MoE MAE {m.get('moe_mae', float('nan')):.5f}\n"
            f"ΔCE {diff:+.5f} {arrow} vs base   ·   MoE score {score_txt}"
        )

    # ---------------------------------------------------------------- plot
    def _add_live_point(
        self, key: str, title: str, ylabel: str, series: str, epoch: float, value: float
    ) -> None:
        if key != self._live_key:
            self._finalize_live_plot()
            self._live_key = key
            self._live_title = title
            self._live_ylabel = ylabel
            self._live_series = {}
        self._live_series.setdefault(series, []).append((epoch, value))
        self._render_live_plot()

    def _finalize_live_plot(self) -> None:
        if not self._live_title:
            return
        self._register_plot(self._live_title, self.plot_label.pixmap())

    def _register_plot(self, name: str, pixmap: QPixmap) -> None:
        if pixmap is None or pixmap.isNull():
            return
        if name in self._plots:
            self._plots[name] = pixmap
            idx = self.plot_combo.findData(name)
            if idx >= 0:
                self.plot_combo.setCurrentIndex(idx)
            return
        self._plots[name] = pixmap
        self.plot_combo.addItem(name, name)
        self.plot_combo.setEnabled(True)
        self.plot_combo.setCurrentIndex(self.plot_combo.count() - 1)

    def _on_plot_selected(self, index: int) -> None:
        name = self.plot_combo.itemData(index)
        if isinstance(name, str) and name in self._plots:
            self.plot_label.setPixmap(self._plots[name])

    def _render_live_plot(self) -> None:
        if not self._live_series:
            return
        fig, ax = plt.subplots(figsize=(7, 3.2), dpi=100)
        fig.patch.set_facecolor("#121418")
        ax.set_facecolor("#121418")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#3a4454")
        ax.spines["bottom"].set_color("#3a4454")
        ax.tick_params(colors="#9aa6b8")
        ax.set_xlabel("epoch", color="#e8edf4")
        ax.set_ylabel(self._live_ylabel, color="#e8edf4")
        ax.set_title(self._live_title, color="#e8edf4")
        colors = [
            "#ff9f43", "#5b8dd9", "#6cc07a", "#c678dd",
            "#e06c75", "#56b6c2", "#e0a24b", "#98c379",
        ]
        for i, (name, points) in enumerate(self._live_series.items()):
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(
                xs, ys,
                color=colors[i % len(colors)], linewidth=1.5, linestyle="-", label=name,
            )
        if len(self._live_series) > 1:
            ax.legend(
                facecolor="#1b1f27", edgecolor="#3a4454",
                labelcolor="#e8edf4", fontsize=9, loc="best",
            )
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        self.plot_label.setPixmap(QPixmap.fromImage(QImage.fromData(buf.getvalue(), "PNG")))

    def _render_bar_plot(self, metrics: dict) -> QPixmap | None:
        entries = [
            ("base_ce", "Base model", "#5b8dd9"),
            ("moe_ce", "MoE", "#e0a24b"),
            ("perfect_ce", "Perfect dispatcher", "#6cc07a"),
            ("best_expert_ce", "Best expert", "#c678dd"),
        ]
        labels, values, colors = [], [], []
        for key, label, color in entries:
            if metrics.get(key) is not None:
                labels.append(label)
                values.append(metrics[key])
                colors.append(color)
        if not values:
            return None
        fig, ax = plt.subplots(figsize=(7, 3.2), dpi=100)
        fig.patch.set_facecolor("#121418")
        ax.set_facecolor("#121418")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#3a4454")
        ax.spines["bottom"].set_color("#3a4454")
        ax.tick_params(colors="#9aa6b8")
        ax.set_ylabel("Cross-entropy", color="#e8edf4")
        ax.set_title("CE comparison", color="#e8edf4")
        bars = ax.bar(labels, values, color=colors)
        for bar, value in zip(bars, values):
            if value is not None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.5f}",
                    ha="center",
                    va="bottom",
                    color="#e8edf4",
                    fontsize=9,
                )
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return QPixmap.fromImage(QImage.fromData(buf.getvalue(), "PNG"))

    def _render_score_plot(self, metrics: dict) -> QPixmap | None:
        score = self._moe_score(metrics)
        if not math.isfinite(score):
            return None
        pct = score * 100.0
        fig, ax = plt.subplots(figsize=(7, 3.2), dpi=100)
        fig.patch.set_facecolor("#121418")
        ax.set_facecolor("#121418")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#3a4454")
        ax.tick_params(colors="#9aa6b8")
        ax.set_title("MoE score", color="#e8edf4")
        ax.set_xlabel("MoE score (%)", color="#e8edf4")
        color = "#6cc07a" if pct >= 0.0 else "#e06c75"
        ax.barh(0, pct, color=color, height=0.5, left=0.0)
        ax.axvline(0.0, color="#e8edf4", linewidth=1.0)
        ax.axvline(100.0, color="#e8edf4", linewidth=0.8, linestyle="--", alpha=0.4)
        lim = max(abs(pct), 100.0) * 1.15
        ax.set_xlim(-lim, lim)
        ax.set_yticks([])
        xpad = 0.03 * lim
        ax.text(
            pct + (xpad if pct >= 0 else -xpad),
            0.0,
            f"{pct:+.1f}%",
            va="center",
            ha="left" if pct >= 0 else "right",
            color="#e8edf4",
            fontsize=9,
        )
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return QPixmap.fromImage(QImage.fromData(buf.getvalue(), "PNG"))

    def _show_plot(self, metrics: dict) -> None:
        self._finalize_live_plot()
        pixmap = self._render_bar_plot(metrics)
        if pixmap is not None:
            self.plot_label.setPixmap(pixmap)
            self._register_plot("CE comparison", pixmap)
        score_pixmap = self._render_score_plot(metrics)
        if score_pixmap is not None:
            self._register_plot("MoE score", score_pixmap)

    def _on_finished(self, metrics: dict) -> None:
        self._set_running(False)
        self.status.showMessage("Done")
        self._log("run finished")
        self.open_results_btn.setVisible(True)

    def _on_failed(self, err: str) -> None:
        self._set_running(False)
        if err == "__cancelled__":
            self.status.showMessage("Cancelled")
            self._log("run cancelled")
        else:
            self.status.showMessage("Failed")
            self._log("ERROR:\n" + err)

    # ---------------------------------------------------------------- theme
    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #121418; color: #e8edf4; }
            QGroupBox {
                border: 1px solid #3a4454; border-radius: 6px;
                margin-top: 10px; padding-top: 6px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background: #1b1f27; color: #e8edf4;
                border: 1px solid #3a4454; border-radius: 4px; padding: 3px 6px;
            }
            QComboBox QAbstractItemView { background: #1b1f27; color: #e8edf4; }
            QComboBox:disabled {
                background: rgba(128, 136, 148, 0.30);
                color: #6a7384;
                border: 1px solid #2a3140;
            }
            QSpinBox:disabled, QDoubleSpinBox:disabled {
                background: rgba(128, 136, 148, 0.30);
                color: #6a7384;
                border: 1px solid #2a3140;
            }
            QPushButton {
                background: #2f3948; color: #e8edf4; border: 1px solid #3a4454;
                border-radius: 4px; padding: 6px 16px;
            }
            QPushButton:hover { background: #3d4a5c; }
            QPushButton:disabled { color: #6a7384; }
            QPushButton#runButton {
                background: #2f9e44; color: #ffffff; border: 1px solid #2f9e44;
            }
            QPushButton#runButton:hover { background: #37b24d; border: 1px solid #37b24d; }
            QPushButton#runButton:disabled {
                background: #2a3140; color: #6a7384; border: 1px solid #3a4454;
            }
            QProgressBar {
                background: #1b1f27; border: 1px solid #3a4454; border-radius: 3px;
                text-align: center; color: #e8edf4; height: 14px;
            }
            QProgressBar::chunk { background: #3d5a80; border-radius: 2px; }
            QPlainTextEdit {
                background: #0e1013; color: #c6cedb; border: 1px solid #3a4454;
                border-radius: 4px; font-family: monospace;
            }
            """
        )


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#121418"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#e8edf4"))
    app.setPalette(palette)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
