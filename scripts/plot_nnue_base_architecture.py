from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from tinymlinternship.config.settings import PROJECT_ROOT
from tinymlinternship.features import FEATURE_DIM

PALETTE = {
    "input": ("#e8f4f8", "#2196F3"),
    "input_group": ("#eef6fc", "#42A5F5"),  # light blue dashed container
    "l1": ("#e8f5e9", "#4CAF50"),
    "l1_group": ("#f1f8f2", "#4CAF50"),
    "reorder": ("#f3e5f5", "#7B1FA2"),
    "concat": ("#fff3e0", "#FF9800"),
    "router": ("#fff9c4", "#FFC107"),
    "expert": ("#ede7f6", "#673AB7"),
    "l2_group": ("#DFD1F0", "#9c7bb8"),  # light purple dashed container
    "out": ("#e0f2f1", "#009688"),
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
    fontsize: int = 10,  # default label size if caller omits fontsize=
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
    alpha: float = 1.0,
    *,
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
            zorder=zorder,
        )
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
    """Draws a dashed container box and places its title/subtitle nicely on top of it."""
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
    
    # Title placed right above the top boundary of the group frame to prevent overlap
    title_y = xy[1] + height - 0.22
    sub_y = xy[1] + height - 0.48

    ax.text(
        xy[0] + width / 2,
        title_y,
        title,
        ha="center",
        va="center",
        fontsize=11,
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


def count_parameters(feature_dim: int, hidden_dim: int) -> dict[str, int]:
    l1 = feature_dim * hidden_dim + hidden_dim
    head = (hidden_dim * 2) * 3 + 3
    return {
        "l1": l1,
        "head": head,
        "total": l1 + head,
    }


def plot_architecture(
    *,
    hidden_dim: int = 128,
    output_path: Path,
    dpi: int = 160,
) -> None:
    feature_dim = FEATURE_DIM
    concat_dim = hidden_dim * 2
    params = count_parameters(feature_dim, hidden_dim)

    # --- GLOBAL SIZE KNOB -------------------------------------------------
    S = 0.80

    def sc(v: float) -> float:
        return v * S

    # --- CANVAS -----------------------------------------------------------
    fig_w, x_max, cx = 10.0, 10.0, 5.0
    fig, ax = plt.subplots(figsize=(fig_w, 11.5))
    ax.set_xlim(0, x_max)
    ax.set_ylim(-1.15, 10.45)
    ax.axis("off")

    # --- TITLE (page header) ----------------------------------------------
    ax.text(
        cx,
        10.10,
        "Base NNUE",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
    )
    ax.text(
        cx,
        9.72,
        f"844 sparse · shared L1 W={hidden_dim} · 1 head · "
        f"{params['total']:,} params",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#444444",
    )

    # --- INPUT (White / Black POV) ----------------------------------------
    in_box_h = sc(0.85)
    in_box_w = sc(3.6)
    in_gap = 0.55
    in_total = 2 * in_box_w + in_gap
    in_x0 = (x_max - in_total) / 2
    in_box_y = 7.90
    
    # Corrected container dimensions & margins
    in_group_pad_x = 0.28
    in_group_top_space = 0.60
    in_group_bot_space = 0.15
    in_group_x = in_x0 - in_group_pad_x
    in_group_w = in_total + 2 * in_group_pad_x
    in_group_y = in_box_y - in_group_bot_space
    in_group_h = in_box_h + in_group_top_space + in_group_bot_space
    
    _group(
        ax,
        (in_group_x, in_group_y),
        in_group_w,
        in_group_h,
        "Input",
        subtitle="Dual board POV · 844 sparse binary features each",
        face=PALETTE["input_group"][0],
        edge=PALETTE["input_group"][1],
    )
    _box(
        ax,
        (in_x0, in_box_y),
        in_box_w,
        in_box_h,
        "White POV input\n844 sparse (binary)",
        face=PALETTE["input"][0],
        edge=PALETTE["input"][1],
        fontsize=sc(10),
    )
    _box(
        ax,
        (in_x0 + in_box_w + in_gap, in_box_y),
        in_box_w,
        in_box_h,
        "Black POV input\n844 sparse (binary)",
        face=PALETTE["input"][0],
        edge=PALETTE["input"][1],
        fontsize=sc(10),
    )

    # --- SHARED L1: accumulators + STM reorder + concat -------------------
    acc_w, acc_h = sc(3.9), sc(1.35)
    acc_gap = 0.5
    acc_total = 2 * acc_w + acc_gap
    acc_x0 = (x_max - acc_total) / 2
    acc_y = 5.30

    mid_w = sc(5.0)
    mid_x = (x_max - mid_w) / 2
    reorder_h = sc(0.85)
    reorder_y = 4.10
    concat_w = sc(4.7)
    concat_h = sc(0.8)
    concat_x = (x_max - concat_w) / 2
    concat_y = 3.15

    # Shared L1 group position: span accumulators → concatenate (router stays outside)
    # l1_x / l1_y = bottom-left; l1_w / l1_h = size (includes title/subtitle top padding)
    l1_pad_x = 0.28
    l1_top_space = 0.60
    l1_bot_space = 0.18
    l1_x = min(acc_x0, mid_x, concat_x) - l1_pad_x
    l1_right = max(acc_x0 + acc_total, mid_x + mid_w, concat_x + concat_w) + l1_pad_x
    l1_w = l1_right - l1_x
    l1_y = concat_y - l1_bot_space
    l1_h = (acc_y + acc_h - concat_y) + l1_top_space + l1_bot_space

    _group(
        ax,
        (l1_x, l1_y),
        l1_w,
        l1_h,
        "Shared L1 (same weights, called twice)",
        subtitle=f"Linear {feature_dim} → {hidden_dim} · STM reorder · concat {concat_dim} · prune 70–80%",
        face=PALETTE["l1_group"][0],
        edge=PALETTE["l1_group"][1],
        alpha=0.5,
    )
    _box(
        ax,
        (acc_x0, acc_y),
        acc_w,
        acc_h,
        f"White accumulator\nLinear {feature_dim} → {hidden_dim}\n"
        f"CReLU [0,127] · incremental add/sub",
        face=PALETTE["l1"][0],
        edge=PALETTE["l1"][1],
        fontsize=sc(9),
    )
    _box(
        ax,
        (acc_x0 + acc_w + acc_gap, acc_y),
        acc_w,
        acc_h,
        f"Black accumulator\nLinear {feature_dim} → {hidden_dim}\n"
        f"CReLU [0,127] · incremental add/sub",
        face=PALETTE["l1"][0],
        edge=PALETTE["l1"][1],
        fontsize=sc(9),
    )
    # Yellow (router palette) for mid steps inside Shared L1
    _box(
        ax,
        (mid_x, reorder_y),
        mid_w,
        reorder_h,
        "STM reorder\nstm ← POV of side-to-move · opp ← other POV",
        face=PALETTE["router"][0],
        edge=PALETTE["router"][1],
        fontsize=sc(9),
        weight="bold",
    )
    _box(
        ax,
        (concat_x, concat_y),
        concat_w,
        concat_h,
        f"Concatenate (STM ‖ Opp)\n{concat_dim}-dim vector",
        face=PALETTE["router"][0],
        edge=PALETTE["router"][1],
        weight="bold",
        fontsize=sc(10),
    )

    # --- Single L2 head + WDL softmax (no bucket router) -------------------
    head_w, head_h = sc(3.4), sc(0.88)
    out_w, out_h = sc(3.4), sc(0.72)
    head_x = (x_max - head_w) / 2
    out_x = (x_max - out_w) / 2
    head_y = 0.90
    out_y = -0.10

    l2_pad_x = 0.28
    l2_top_space = 0.72
    l2_bot_space = 0.22
    l2_x = head_x - l2_pad_x
    l2_w = head_w + 2 * l2_pad_x
    l2_y = out_y - l2_bot_space
    l2_h = (head_y + head_h - out_y) + l2_top_space + l2_bot_space

    _group(
        ax,
        (l2_x, l2_y),
        l2_w,
        l2_h,
        "L2 Head",
        subtitle=f"Linear {concat_dim}→3 · WDL softmax",
        face=PALETTE["l2_group"][0],
        edge=PALETTE["l2_group"][1],
        alpha=0.5,
    )

    face, edge = PALETTE["expert"]
    _box(
        ax,
        (head_x, head_y),
        head_w,
        head_h,
        f"L2 head\nLinear {concat_dim}→3",
        face=face,
        edge=edge,
        fontsize=sc(10),
        weight="bold",
    )
    out_face, out_edge = PALETTE["out"]
    _box(
        ax,
        (out_x, out_y),
        out_w,
        out_h,
        "WDL softmax\n[0, 1]^3",
        face=out_face,
        edge=out_edge,
        fontsize=sc(10),
        weight="bold",
    )
    _arrow(ax, (cx, head_y), (cx, out_y + out_h))

    # --- ARROWS -----------------------------------------------------------
    # arrow input layer — from White/Black POV boxes (not the Input group frame)
    # zorder=1: above Input group (0), below POV boxes (2) so shafts leave each POV box
    white_pov_bottom = (in_x0 + in_box_w, in_box_y)
    black_pov_bottom = (in_x0 + in_box_w + in_gap, in_box_y)
    white_acc_top = (acc_x0 + acc_w / 2, acc_y + acc_h)
    black_acc_top = (acc_x0 + acc_w + acc_gap + acc_w / 2, acc_y + acc_h)
    _arrow(ax, white_pov_bottom, white_acc_top, zorder=-1)
    _arrow(ax, black_pov_bottom, black_acc_top, zorder=-1)
    _arrow(ax, (acc_x0 + acc_w / 2, acc_y), (mid_x + mid_w * 0.35, reorder_y + reorder_h))
    _arrow(
        ax,
        (acc_x0 + acc_w + acc_gap + acc_w / 2, acc_y),
        (mid_x + mid_w * 0.65, reorder_y + reorder_h),
    )
    _arrow(ax, (cx, reorder_y), (cx, concat_y + concat_h))
    _arrow(ax, (cx, concat_y), (cx, l2_y + l2_h), zorder=-1)

    # --- FOOTER -----------------------------------------------------------
    ax.text(
        0.3,
        -0.95,
        "Student: DualHidden WDL · L1 W=64 · L2=128 · CE vs Lc0 STM (W,D,L)",
        fontsize=8.5,
        color="#666666",
    )
    ax.text(
        x_max - 0.3,
        -0.95,
        "encode_dual()",
        ha="right",
        fontsize=8.5,
        color="#666666",
        style="italic",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot base NNUE architecture")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "THESIS" / "thesis-plots" / "nnue_base_architecture.png",
    )
    parser.add_argument("--hidden-dim", type=int, default=128, choices=[128, 256])
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args(argv)

    plot_architecture(hidden_dim=args.hidden_dim, output_path=args.output, dpi=args.dpi)
    print(f"Saved plot: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
