"""Plot the thesis two-stage dispatcher architecture (Model + Dispatcher).

Renders the mermaid diagram from NOTES/Thesis.md as a publication-style PNG,
using the same visual language as scripts/plot_nnue_architecture.py.

  Model: state → L1 → L2 → value
  Dispatcher: L1 embeddings → cluster assignment logits --softmax→ proba

Usage:
  py -3.12 scripts/plot_thesis_dispatcher.py
  → plots/thesis_dispatcher_architecture.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from tinymlinternship.config.settings import PROJECT_ROOT


PALETTE = {
    "state": ("#e8f4f8", "#2196F3"),
    "l1": ("#e8f5e9", "#4CAF50"),
    "l2": ("#ede7f6", "#673AB7"),
    "value": ("#ede7f6", "#673AB7"),  # was ("#e0f2f1", "#009688"),
    "logits": ("#fff9c4", "#FFC107"),
    "proba": ("#FFF5E8", "#FFD79E"),  # was ("#fff3e0", "#FF9800"),
    "model_group": ("#f5f5f5", "#757575"),
    "dispatcher_group": ("#f5f5f5", "#757575"),
    "expert_group": ("#DFD1F0", "#9c7bb8"),  # matches NNUE L2 Expert Head
    "arrow": "#5f6368",
}


def _box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    face: str,
    edge: str,
    fontsize: int = 11,
    weight: str = "normal",
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.6,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        wrap=True,
    )
    return patch


def _arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    label: str = "",
    label_offset: tuple[float, float] = (0.0, 0.12),
    connectionstyle: str = "arc3,rad=0",
    alpha: float = 1.0,
    zorder: float = 1,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.4,
            color=PALETTE["arrow"],
            alpha=alpha,
            shrinkA=4,
            shrinkB=4,
            connectionstyle=connectionstyle,
            zorder=zorder,
        )
    )
    if label:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(
            mx,
            my,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#444444",
            style="italic",
            zorder=3,
        )


def _group(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    *,
    subtitle: str = "",
    face: str,
    edge: str,
    alpha: float = 1.0,
) -> FancyBboxPatch:
    """Dashed container with title on the top edge."""
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=2.0,
        edgecolor=edge,
        facecolor=face,
        linestyle="--",
        alpha=alpha,
        zorder=0,
    )
    ax.add_patch(patch)

    title_y = xy[1] + height - 0.22
    sub_y = xy[1] + height - 0.48
    ax.text(
        xy[0] + width / 2,
        title_y,
        title,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=edge,
        zorder=1,
    )
    if subtitle:
        ax.text(
            xy[0] + width / 2,
            sub_y,
            subtitle,
            ha="center",
            va="center",
            fontsize=9,
            color="#444444",
            zorder=1,
        )
    return patch


def plot_architecture(
    *,
    output_path: Path,
    dpi: int = 160,
) -> None:
    # --- CANVAS -----------------------------------------------------------
    # Title band [6.9, 8.0]; content below; footer near y≈0.3
    fig_w, x_max = 11.0, 11.0
    fig, ax = plt.subplots(figsize=(fig_w, 8.2))
    ax.set_xlim(0, x_max)
    ax.set_ylim(0.0, 8.0)
    ax.axis("off")

    # --- TITLE (above dashed groups) --------------------------------------
    ax.text(
        x_max / 2,
        7.65,
        "Two-Stage Proxy Clustering for Expert Dispatching",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        x_max / 2,
        7.22,
        "Shared L1 embeddings feed a lightweight dispatcher "
        "(logits → softmax → cluster proba)",
        ha="center",
        va="center",
        fontsize=10,
        color="#444444",
    )

    # --- BOX GEOMETRY -----------------------------------------------------
    box_w, box_h = 2.4, 0.72
    box_gap = 0.30  # vertical gap between stacked boxes

    # Model column (left)
    model_cx = 3.0
    model_x = model_cx - box_w / 2
    # Dispatcher column (right) — slightly wider labels
    disp_w = 3.1
    disp_cx = 8.2
    disp_x = disp_cx - disp_w / 2

    # Nested Expert head group around L2 + value (title band above L2)
    expert_pad_x = 0.18
    expert_top_space = 0.42  # room for "Expert head" title (no subtitle)
    expert_bot_space = 0.16
    expert_inner_gap = box_gap  # L2 ↔ value

    # Vertical stack: state → L1 → [Expert head: L2 → value]
    # Extra gap L1 → L2 leaves room for the Expert head title band.
    state_y = 5.55
    l1_y = state_y - box_h - box_gap
    l2_y = l1_y - box_h - box_gap - expert_top_space
    value_y = l2_y - box_h - expert_inner_gap

    expert_group_x = model_x - expert_pad_x
    expert_group_w = box_w + 2 * expert_pad_x
    expert_group_y = value_y - expert_bot_space
    expert_group_h = (l2_y + box_h - value_y) + expert_top_space + expert_bot_space

    # Dispatcher: logits aligned with L1; proba below with room for "softmax"
    logits_y = l1_y
    proba_y = logits_y - box_h - box_gap - 0.20

    # --- MODEL GROUP (wraps state, L1, nested Expert head) ----------------
    model_pad_x = 0.22
    model_top_space = 0.62
    model_bot_space = 0.18
    model_group_x = expert_group_x - model_pad_x
    model_group_w = expert_group_w + 2 * model_pad_x
    model_group_y = expert_group_y - model_bot_space
    model_group_h = (state_y + box_h - expert_group_y) + model_top_space + model_bot_space

    _group(
        ax,
        (model_group_x, model_group_y),
        model_group_w,
        model_group_h,
        "Model",
        subtitle=r"base value network $f_\theta$",
        face=PALETTE["model_group"][0],
        edge=PALETTE["model_group"][1],
        alpha=0.55,
    )

    _box(
        ax,
        (model_x, state_y),
        box_w,
        box_h,
        "state",
        face=PALETTE["state"][0],
        edge=PALETTE["state"][1],
        weight="bold",
    )
    _box(
        ax,
        (model_x, l1_y),
        box_w,
        box_h,
        "L1 accumulator",
        face=PALETTE["l1"][0],
        edge=PALETTE["l1"][1],
        weight="bold",
    )

    # Nested Expert head (L2 + value) inside Model
    _group(
        ax,
        (expert_group_x, expert_group_y),
        expert_group_w,
        expert_group_h,
        "Expert head",
        face=PALETTE["expert_group"][0],
        edge=PALETTE["expert_group"][1],
        alpha=0.55,
    )
    _box(
        ax,
        (model_x, l2_y),
        box_w,
        box_h,
        "L2",
        face=PALETTE["l2"][0],
        edge=PALETTE["l2"][1],
        weight="bold",
    )
    _box(
        ax,
        (model_x, value_y),
        box_w,
        box_h,
        "value",
        face=PALETTE["value"][0],
        edge=PALETTE["value"][1],
        weight="bold",
    )

    # Caption under the Model dashed frame
    ax.text(
        model_cx,
        model_group_y - 0.22,
        "Trained on a dataset\n"
        "of state-value pairs",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#555555",
        zorder=1,
    )

    # --- DISPATCHER GROUP (snug around logits + proba) --------------------
    disp_pad_x = 0.35
    disp_top_space = 0.62
    disp_bot_space = 0.22
    disp_group_x = disp_x - disp_pad_x
    disp_group_w = disp_w + 2 * disp_pad_x
    disp_group_y = proba_y - disp_bot_space
    disp_group_h = (logits_y + box_h - proba_y) + disp_top_space + disp_bot_space

    _group(
        ax,
        (disp_group_x, disp_group_y),
        disp_group_w,
        disp_group_h,
        "Dispatcher",
        subtitle=r"$g_\phi$ · train with softmax, infer argmax",
        face=PALETTE["dispatcher_group"][0],
        edge=PALETTE["dispatcher_group"][1],
        alpha=0.55,
    )

    _box(
        ax,
        (disp_x, logits_y),
        disp_w,
        box_h,
        "cluster assignment logits",
        face=PALETTE["logits"][0],
        edge=PALETTE["logits"][1],
        weight="bold",
        fontsize=10,
    )
    _box(
        ax,
        (disp_x, proba_y),
        disp_w,
        box_h,
        "cluster assignment proba",
        face=PALETTE["proba"][0],
        edge=PALETTE["proba"][1],
        weight="bold",
        fontsize=10,
    )

    # Caption under the Dispatcher dashed frame
    ax.text(
        disp_cx,
        disp_group_y - 0.22,
        "Trained on cluster labels\n"
        "of the task vectors\n"
        r"$\delta_i = \nabla_w \; \mathcal{L}_{acc}(f_\theta(s_i), v_i)$",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#555555",
        zorder=1,
    )

    # --- ARROWS (Model stack) ---------------------------------------------
    _arrow(ax, (model_cx, state_y), (model_cx, l1_y + box_h))
    # L1 → Expert head (land on the group frame top / L2)
    _arrow(ax, (model_cx, l1_y), (model_cx, expert_group_y + expert_group_h))
    _arrow(ax, (model_cx, l2_y), (model_cx, value_y + box_h))

    # --- ARROW (Dispatcher softmax) ---------------------------------------
    _arrow(
        ax,
        (disp_cx, logits_y),
        (disp_cx, proba_y + box_h),
        label="softmax",
        label_offset=(0.55, 0.0),
    )

    # --- HORIZONTAL LINK: L1 → Dispatcher (embeddings) --------------------
    l1_right = (model_x + box_w, l1_y + box_h / 2)
    logits_left = (disp_x, logits_y + box_h / 2)
    _arrow(
        ax,
        l1_right,
        logits_left,
        label="embeddings",
        label_offset=(0.0, 0.18),
        zorder=1,
    )

    # --- LINK: logits → Expert head (expert selection) --------------------
    # From left side of logits down-left into the right edge of Expert head.
    logits_left_bot = (disp_x, logits_y + box_h * 0.25)
    expert_right = (
        expert_group_x + expert_group_w,
        expert_group_y + expert_group_h * 0.55,
    )
    _arrow(
        ax,
        logits_left_bot,
        expert_right,
        label="expert\nselection",
        label_offset=(0.0, -0.28),
        connectionstyle="arc3,rad=0.12",
        zorder=1,
    )

    # --- FOOTER -----------------------------------------------------------
    ax.text(
        0.4,
        0.35,
        r"Pseudo-labels from clustering $\delta_i$ in $\mathcal{W}$; "
        r"dispatcher trains on L1 embeddings of $s \in \mathcal{S}$",
        fontsize=8.5,
        color="#666666",
    )
    ax.text(
        x_max - 0.4,
        0.35,
        "NOTES/Thesis.md",
        ha="right",
        fontsize=8.5,
        color="#666666",
        style="italic",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot thesis Model+Dispatcher architecture (from NOTES/Thesis.md mermaid)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "plots" / "thesis_dispatcher_architecture.png",
    )
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args(argv)

    plot_architecture(output_path=args.output, dpi=args.dpi)
    print(f"Saved plot: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
