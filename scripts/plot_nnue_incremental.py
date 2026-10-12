"""Two consecutive positions through one NNUE, with an incremental L1 update.

Left is White to move (chess_white.png, the starting position). Right is
Black to move (chess_black.png, after 1.e4). The accumulator pair is updated
from White to Black; the STM reorder swaps which perspective is read first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT / "THESIS" / "thesis-plots"

sys.path.insert(0, str(ROOT / "src"))

BLACK_FACE = "#eceff1"
BLACK_EDGE = "#546e7a"
WHITE_FACE = "#ffffff"
WHITE_EDGE = "#90a4ae"
REORDER_FACE = "#fff9c4"
REORDER_EDGE = "#F9A825"
L2_FACE = "#ede7f6"
L2_EDGE = "#673AB7"
OUT_FACE = "#e0f2f1"
OUT_EDGE = "#009688"
INK = "#37474f"
BLUE = "#1565C0"


def _box(ax, xy, w, h, text, *, face, edge, fontsize=9, weight="normal", lw=1.5):
    ax.add_patch(
        FancyBboxPatch(
            xy,
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=lw,
            edgecolor=edge,
            facecolor=face,
            zorder=2,
        )
    )
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#212121",
        zorder=3,
    )


def _arrow(ax, start, end, *, color=INK, lw=1.3, ms=11, zorder=1, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            shrinkA=3,
            shrinkB=3,
            zorder=zorder,
        )
    )


def _column(ax, cx, board, *, tag: str, stm_black: bool, caption: str) -> dict[str, float]:
    """Draw one evaluation. Returns the y-centers of the two accumulator bars."""
    board_w = 3.35
    board_y0 = 0.52
    x0 = cx - board_w / 2
    ax.imshow(
        board,
        extent=(x0, x0 + board_w, board_y0, board_y0 + board_w),
        origin="upper",
        zorder=2,
        interpolation="lanczos",
    )
    ax.text(
        cx,
        0.24,
        caption,
        ha="center",
        va="center",
        fontsize=11,
        color="#212121",
    )

    acc_w, acc_h = 3.55, 0.66
    acc_x = cx - acc_w / 2
    white_y = 4.28
    black_y = 5.08
    pad_x, pad_bot, pad_top = 0.14, 0.16, 0.14
    group_x = acc_x - pad_x
    group_y = white_y - pad_bot
    group_w = acc_w + 2 * pad_x
    group_h = (black_y + acc_h + pad_top) - group_y
    ax.add_patch(
        FancyBboxPatch(
            (group_x, group_y),
            group_w,
            group_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.3,
            edgecolor="#90a4ae",
            facecolor="#f8fafb",
            linestyle="--",
            zorder=0,
        )
    )
    _arrow(ax, (cx, board_y0 + board_w), (cx, group_y))

    superscript = tag
    _box(
        ax,
        (acc_x, white_y),
        acc_w,
        acc_h,
        f"White accumulator\n$v^{{({superscript},\\mathrm{{white}})}}$",
        face=WHITE_FACE,
        edge=WHITE_EDGE,
        fontsize=9,
    )
    _box(
        ax,
        (acc_x, black_y),
        acc_w,
        acc_h,
        f"Black accumulator\n$v^{{({superscript},\\mathrm{{black}})}}$",
        face=BLACK_FACE,
        edge=BLACK_EDGE,
        fontsize=9,
        weight="bold",
    )
    reorder_w, reorder_h = 4.05, 0.62
    reorder_y = 6.18
    _arrow(ax, (cx, group_y + group_h), (cx, reorder_y))
    if stm_black:
        reorder_text = "STM reorder\nblack to move  →  Black ‖ White"
        halves = (
            ("Black\nSTM", BLACK_FACE, BLACK_EDGE),
            ("White\nOpp", WHITE_FACE, WHITE_EDGE),
        )
    else:
        reorder_text = "STM reorder\nwhite to move  →  White ‖ Black"
        halves = (
            ("White\nSTM", WHITE_FACE, WHITE_EDGE),
            ("Black\nOpp", BLACK_FACE, BLACK_EDGE),
        )
    _box(
        ax,
        (cx - reorder_w / 2, reorder_y),
        reorder_w,
        reorder_h,
        reorder_text,
        face=REORDER_FACE,
        edge=REORDER_EDGE,
        fontsize=8.5,
        weight="bold",
    )
    concat_w, concat_h, gap = 4.15, 0.78, 0.10
    concat_y = 7.04
    half_w = (concat_w - gap) / 2
    left_x = cx - concat_w / 2
    _arrow(ax, (cx, reorder_y + reorder_h), (cx, concat_y))
    for i, (label, face, edge) in enumerate(halves):
        _box(
            ax,
            (left_x + i * (half_w + gap), concat_y),
            half_w,
            concat_h,
            label,
            face=face,
            edge=edge,
            fontsize=9,
            weight="bold",
        )

    l2_w, l2_h = 2.7, 0.58
    l2_y = 8.06
    _arrow(ax, (cx, concat_y + concat_h), (cx, l2_y))
    _box(
        ax,
        (cx - l2_w / 2, l2_y),
        l2_w,
        l2_h,
        "L2 head",
        face=L2_FACE,
        edge=L2_EDGE,
        fontsize=10,
        weight="bold",
    )

    out_w, out_h = 2.35, 0.64
    out_y = 8.88
    _arrow(ax, (cx, l2_y + l2_h), (cx, out_y))
    _box(
        ax,
        (cx - out_w / 2, out_y),
        out_w,
        out_h,
        f"$y^{{({superscript})}}$\nWDL softmax",
        face=OUT_FACE,
        edge=OUT_EDGE,
        fontsize=9.5,
        weight="bold",
    )
    return {
        "acc_x0": acc_x,
        "acc_x1": acc_x + acc_w,
        "white_mid": white_y + acc_h / 2,
        "black_mid": black_y + acc_h / 2,
    }


def plot(output_path: Path, dpi: int = 160) -> None:
    black = imread(PLOTS / "chess_black.png")
    white = imread(PLOTS / "chess_white.png")
    if black.dtype != np.float32 and black.dtype != np.float64:
        black = black.astype(np.float32) / 255.0
        white = white.astype(np.float32) / 255.0

    fig_w, fig_h = 15.6, 10.75
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        fig_w / 2,
        10.38,
        "Same NNUE, one ply apart",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#212121",
    )

    left = _column(
        ax,
        3.55,
        white,
        tag="p",
        stm_black=False,
        caption="Position $p$ (white's turn)",
    )
    right = _column(
        ax,
        12.05,
        black,
        tag="q",
        stm_black=True,
        caption="Position $q$ (black's turn)",
    )

    # Accumulator state carries forward one ply; upper layers do not.
    for y_key in ("black_mid", "white_mid"):
        _arrow(
            ax,
            (left["acc_x1"], left[y_key]),
            (right["acc_x0"], right[y_key]),
            color=BLUE,
            lw=1.8,
            ms=13,
            zorder=1,
            style="-|>",
        )
    mid_x = (left["acc_x1"] + right["acc_x0"]) / 2
    ax.text(
        mid_x,
        left["black_mid"] + 0.46,
        "Incremental update with $W_{L1}$",
        ha="center",
        va="center",
        fontsize=10,
        color=BLUE,
        fontweight="bold",
        zorder=4,
    )
    ax.text(
        mid_x,
        left["white_mid"] - 0.42,
        "white pawn e2  →  e4",
        ha="center",
        va="center",
        fontsize=9,
        color="#546e7a",
        zorder=4,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot incremental NNUE update on two positions")
    parser.add_argument(
        "--output",
        type=Path,
        default=PLOTS / "nnue_incremental_update.png",
    )
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args(argv)
    plot(args.output, dpi=args.dpi)
    print(f"Saved plot: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
