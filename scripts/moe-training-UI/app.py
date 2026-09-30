"""MoE training UI.

A small PyQt6 window to configure a hard-MoE training run (base model,
clustering, dispatcher, expert fine-tuning) and watch live per-stage progress
(percent + dispatcher accuracy + expert / MoE cross-entropy).

Run it from the repo root::

    python scripts/moe-training-UI/run.py
"""

from __future__ import annotations

import io
import sys
import threading
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPalette, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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

from runner import CancelToken, CancelledError, TrainingConfig, iter_checkpoints, run_moe_training

STAGE_NAMES = ["base", "loading", "cluster", "dispatcher", "experts", "eval"]


class _Relay(QObject):
    event = pyqtSignal(object)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)


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
        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(self.metric)
        layout.addLayout(top)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(True)
        layout.addWidget(self.bar)

    def set_progress(self, fraction: float | None) -> None:
        if fraction is None:
            self.bar.setRange(0, 0)
        else:
            self.bar.setRange(0, 100)
            self.bar.setValue(int(round(100.0 * max(0.0, min(1.0, fraction)))))

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

        self._apply_theme()
        self._sync_enabled_state()

    # ---------------------------------------------------------------- config
    def _build_config(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        layout = QVBoxLayout(host)

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

        # Data
        data_group = QGroupBox("Data")
        df = QFormLayout(data_group)
        self.max_rows = QSpinBox()
        self.max_rows.setRange(10_000, 100_000_000)
        self.max_rows.setSingleStep(100_000)
        self.max_rows.setValue(2_000_000)
        self.max_test = QSpinBox()
        self.max_test.setRange(1_000, 500_000)
        self.max_test.setValue(50_000)
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
        self.k_combo = QComboBox()
        for k in (2, 4, 8, 16):
            self.k_combo.addItem(str(k), k)
        self.k_combo.setCurrentIndex(2)  # 8
        cf.addRow("Representation", self.clustering)
        cf.addRow("Buckets K", self.k_combo)
        layout.addWidget(cluster_group)

        # Dispatcher
        disp_group = QGroupBox("Dispatcher")
        pf = QFormLayout(disp_group)
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
        self.dispatcher_lr = QDoubleSpinBox()
        self.dispatcher_lr.setDecimals(4)
        self.dispatcher_lr.setRange(1e-5, 1.0)
        self.dispatcher_lr.setValue(1e-2)
        pf.addRow("Type", self.dispatcher_type)
        pf.addRow("Hidden width", self.dispatcher_hidden)
        pf.addRow("Epochs", self.dispatcher_epochs)
        pf.addRow("LR", self.dispatcher_lr)
        layout.addWidget(disp_group)

        # Experts
        exp_group = QGroupBox("Experts")
        ef = QFormLayout(exp_group)
        self.expert_epochs = QSpinBox()
        self.expert_epochs.setRange(1, 1000)
        self.expert_epochs.setValue(2)
        self.expert_lr = QDoubleSpinBox()
        self.expert_lr.setDecimals(4)
        self.expert_lr.setRange(1e-5, 1.0)
        self.expert_lr.setValue(1e-3)
        self.expert_lr_end = QDoubleSpinBox()
        self.expert_lr_end.setDecimals(5)
        self.expert_lr_end.setRange(0.0, 1.0)
        self.expert_lr_end.setValue(1e-4)
        self.l1_frozen = QCheckBox("Freeze L1")
        self.l1_frozen.setChecked(True)
        ef.addRow("Epochs", self.expert_epochs)
        ef.addRow("LR", self.expert_lr)
        ef.addRow("LR end", self.expert_lr_end)
        ef.addRow("", self.l1_frozen)
        layout.addWidget(exp_group)

        # Misc
        misc_group = QGroupBox("Run")
        mf = QFormLayout(misc_group)
        self.device = QComboBox()
        self.device.addItem("auto", "auto")
        self.device.addItem("cuda", "cuda")
        self.device.addItem("cpu", "cpu")
        self.run_name = QLineEdit("ui_run")
        mf.addRow("Device", self.device)
        mf.addRow("Run name", self.run_name)
        layout.addWidget(misc_group)

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
        top_split.setSizes([440, 440])
        layout.addWidget(top_split, 3)

        plot_group = QGroupBox("Results — CE comparison")
        pl = QVBoxLayout(plot_group)
        self.plot_label = QLabel("Run a training to see the CE comparison")
        self.plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plot_label.setScaledContents(True)
        self.plot_label.setMinimumHeight(220)
        pl.addWidget(self.plot_label)
        layout.addWidget(plot_group, 2)

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
        buttons.addWidget(self.run_btn)
        buttons.addWidget(self.stop_btn)
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
        for best in iter_checkpoints():
            self.checkpoint.addItem(str(best.parent.name), str(best))

    def _sync_enabled_state(self) -> None:
        training_new = self.base_source.currentIndex() == 1
        self.checkpoint.setEnabled(not training_new)
        self.h_spin.setEnabled(training_new)
        self.H_spin.setEnabled(training_new)
        self.base_epochs.setEnabled(training_new)

        piece = self.clustering.currentData() == "piece_count"
        self.dispatcher_type.setEnabled(not piece)
        self.dispatcher_hidden.setEnabled(not piece and self.dispatcher_type.currentData() == "mlp")
        self.dispatcher_epochs.setEnabled(not piece)
        self.dispatcher_lr.setEnabled(not piece)

    def _collect_config(self) -> TrainingConfig:
        return TrainingConfig(
            base_source="new" if self.base_source.currentIndex() == 1 else "load",
            base_checkpoint=self.checkpoint.currentData() or "",
            h=self.h_spin.value(),
            H=self.H_spin.value(),
            base_epochs=self.base_epochs.value(),
            max_rows=self.max_rows.value(),
            max_test=self.max_test.value(),
            clustering=self.clustering.currentData(),
            k=int(self.k_combo.currentData()),
            dispatcher_type=self.dispatcher_type.currentData(),
            dispatcher_hidden=int(self.dispatcher_hidden.currentData()),
            dispatcher_epochs=self.dispatcher_epochs.value(),
            dispatcher_lr=float(self.dispatcher_lr.value()),
            expert_epochs=self.expert_epochs.value(),
            expert_lr=float(self.expert_lr.value()),
            expert_lr_end=float(self.expert_lr_end.value()),
            l1_frozen=self.l1_frozen.isChecked(),
            device=self.device.currentData(),
            run_name=self.run_name.text().strip() or "ui_run",
        )

    # ---------------------------------------------------------------- run
    def _start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        cfg = self._collect_config()
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

    def _set_running(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        for widget in (
            self.base_source, self.checkpoint, self.h_spin, self.H_spin,
            self.base_epochs, self.max_rows, self.max_test, self.clustering,
            self.k_combo, self.dispatcher_type, self.dispatcher_hidden,
            self.dispatcher_epochs, self.dispatcher_lr, self.expert_epochs,
            self.expert_lr, self.expert_lr_end, self.l1_frozen, self.device,
            self.run_name,
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

    def _log(self, text: str) -> None:
        self.log.appendPlainText(text)

    # ---------------------------------------------------------------- events
    def _on_event(self, ev: dict) -> None:
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
            elif event == "done":
                row.set_progress(1.0)
                row.set_metric(ev.get("msg", f"W{ev.get('h')}_H{ev.get('H')}"))
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
                row.set_metric(f"type {ev.get('type')}")
            elif event == "epoch":
                row.set_progress(ev.get("progress"))
                row.set_metric(
                    f"epoch {ev.get('epoch')}/{ev.get('epochs')} · "
                    f"train {ev.get('train_acc'):.3f} · val {ev.get('val_acc'):.3f}"
                )
            elif event == "done":
                row.set_progress(1.0)
                row.set_metric(f"val_acc {ev.get('best_val_acc'):.4f}")
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
                row.set_metric(
                    f"expert {eid}/{k} · epoch {ev.get('epoch')}/{ev.get('epochs')} · "
                    f"hold {ev.get('hold_ce'):.4f} (base {ev.get('base_hold_ce'):.4f})"
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
                row.set_metric(f"base {ev.get('base_ce'):.5f} · moe {ev.get('moe_ce'):.5f}")
                self._show_plot(ev)
            return

        if stage == "done":
            summary = ev.get("summary", {})
            metrics = summary.get("metrics", {})
            self.result_label.setText(self._format_result(metrics))
            return

    def _format_result(self, m: dict) -> str:
        if not m:
            return ""
        base = m.get("base_ce", float("nan"))
        moe = m.get("moe_ce", float("nan"))
        oracle = m.get("oracle_ce", float("nan"))
        diff = moe - base if isinstance(moe, float) and isinstance(base, float) else float("nan")
        arrow = "↓" if diff < 0 else "↑"
        return (
            f"base CE {base:.5f}   ·   MoE CE {moe:.5f}   ·   oracle CE {oracle:.5f}\n"
            f"base MAE {m.get('base_mae', float('nan')):.5f}   ·   "
            f"MoE MAE {m.get('moe_mae', float('nan')):.5f}\n"
            f"ΔCE {diff:+.5f} {arrow} vs base"
        )

    # ---------------------------------------------------------------- plot
    def _render_bar_plot(self, metrics: dict) -> QPixmap | None:
        base = metrics.get("base_ce")
        moe = metrics.get("moe_ce")
        oracle = metrics.get("oracle_ce")
        if base is None and moe is None and oracle is None:
            return None
        labels = ["Base CE", "MoE CE (dispatcher)", "Oracle CE (100%)"]
        values = [base, moe, oracle]
        colors = ["#5b8dd9", "#e0a24b", "#6cc07a"]
        fig, ax = plt.subplots(figsize=(7, 3.2), dpi=100)
        fig.patch.set_facecolor("#121418")
        ax.set_facecolor("#121418")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#3a4454")
        ax.spines["bottom"].set_color("#3a4454")
        ax.tick_params(colors="#9aa6b8")
        ax.set_ylabel("Cross-entropy", color="#e8edf4")
        ax.set_title("MoE vs base vs oracle", color="#e8edf4")
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

    def _show_plot(self, metrics: dict) -> None:
        pixmap = self._render_bar_plot(metrics)
        if pixmap is not None:
            self.plot_label.setPixmap(pixmap)

    def _on_finished(self, metrics: dict) -> None:
        self._set_running(False)
        self.status.showMessage("Done")
        self._log("run finished")

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
