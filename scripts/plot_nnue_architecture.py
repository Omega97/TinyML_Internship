#!/usr/bin/env python3
"""Draw the SARDINE bucketed NNUE architecture diagram."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from tinymlinternship.config.settings import PROJECT_ROOT
from tinymlinternship.features import FEATURE_DIM, NUM_BUCKETS


BUCKET_LABELS = [
    "0: p≤12\nendgame",
    "1: p∈[13,21]\nno Q",
    "2: p∈[13,21]\n+ Q",
    "3: p∈[22,27]\nno Q",
    "4: p∈[22,27]\n+ Q",
    "5: p∈[28,31]\nno Q",
    "6: p∈[28,31]\n+ Q",
    "7: p=32\nopening",
]

PALETTE = {
    "input": ("#e8f4f8", "#2196F3"),
    "l1": ("#e8f5e9", "#4CAF50"),
    "l1_group": ("#f1f8f2", "#4CAF50"),
    "reorder": ("#f3e5f5", "#7B1FA2"),
    "concat": ("#fff3e0", "#FF9800"),
    "router": ("#fff9c4", "#FFC107"),
    "expert": ("#ede7f6", "#673AB7"),
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
    fontsize: int = 10,
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


def _arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.4,
            color=PALETTE["arrow"],
            shrinkA=4,
            shrinkB=4,
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
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=2.0,
        edgecolor=edge,
        facecolor=face,
        linestyle="--",
        zorder=0,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height - 0.22,
        title,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=edge,
        zorder=1,
    )
    if subtitle:
        ax.text(
            xy[0] + width / 2,
            xy[1] + height - 0.55,
            subtitle,
            ha="center",
            va="top",
            fontsize=9,
            color="#444444",
            zorder=1,
        )
    return patch


def count_parameters(feature_dim: int, hidden_dim: int, num_buckets: int) -> dict[str, int]:
    l1 = feature_dim * hidden_dim + hidden_dim
    expert = (hidden_dim * 2) * 1 + 1
    experts = expert * num_buckets
    return {
        "l1": l1,
        "expert_each": expert,
        "experts": experts,
        "total": l1 + experts,
    }


def plot_architecture(
    *,
    hidden_dim: int = 128,
    output_path: Path,
    dpi: int = 160,
) -> None:
    feature_dim = FEATURE_DIM
    concat_dim = hidden_dim * 2
    params = count_parameters(feature_dim, hidden_dim, NUM_BUCKETS)

    fig, ax = plt.subplots(figsize=(14, 10.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.6, 10)
    ax.axis("off")

    ax.text(
        7,
        9.55,
        "SARDINE Bucketed NNUE",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )
    ax.text(
        7,
        9.05,
        f"844 sparse features · shared L1 W={hidden_dim} · {NUM_BUCKETS} expert heads · "
        f"{params['total']:,} trainable params",
        ha="center",
        va="center",
        fontsize=11,
        color="#444444",
    )

    # Inputs
    _box(
        ax,
        (1.0, 7.85),
        4.0,
        1.0,
        "White POV input\n844 sparse (binary)",
        face=PALETTE["input"][0],
        edge=PALETTE["input"][1],
    )
    _box(
        ax,
        (9.0, 7.85),
        4.0,
        1.0,
        "Black POV input\n844 sparse (binary)",
        face=PALETTE["input"][0],
        edge=PALETTE["input"][1],
    )

    # Shared L1 region: same weight matrix applied once per POV → two accumulators
    _group(
        ax,
        (0.55, 5.15),
        12.9,
        2.45,
        "Shared L1 (same weights, called twice)",
        subtitle=f"Linear {feature_dim} → {hidden_dim} · dense train → gradual prune 70–80%",
        face=PALETTE["l1_group"][0],
        edge=PALETTE["l1_group"][1],
    )
    _box(
        ax,
        (1.1, 5.55),
        4.7,
        1.55,
        f"White accumulator\nLinear {feature_dim} → {hidden_dim}\n"
        f"CReLU [0,127] · incremental add/sub",
        face=PALETTE["l1"][0],
        edge=PALETTE["l1"][1],
        fontsize=9.5,
    )
    _box(
        ax,
        (8.2, 5.55),
        4.7,
        1.55,
        f"Black accumulator\nLinear {feature_dim} → {hidden_dim}\n"
        f"CReLU [0,127] · incremental add/sub",
        face=PALETTE["l1"][0],
        edge=PALETTE["l1"][1],
        fontsize=9.5,
    )

    # STM ordering (no extra matmul — selects which POV is stm vs opponent)
    _box(
        ax,
        (4.35, 3.85),
        5.3,
        0.9,
        "STM reorder\nstm ← POV of side-to-move · opp ← other POV",
        face=PALETTE["reorder"][0],
        edge=PALETTE["reorder"][1],
        fontsize=9.5,
        weight="bold",
    )

    # Concat
    _box(
        ax,
        (4.6, 2.75),
        4.8,
        0.9,
        f"Concatenate (STM ‖ Opp)\n{concat_dim}-dim vector",
        face=PALETTE["concat"][0],
        edge=PALETTE["concat"][1],
        weight="bold",
    )

    # Router
    _box(
        ax,
        (4.9, 1.7),
        4.2,
        0.8,
        "Bucket router\npiece count + queen → 1 of 8",
        face=PALETTE["router"][0],
        edge=PALETTE["router"][1],
        weight="bold",
    )

    # Experts
    expert_w = 1.45
    expert_h = 0.95
    expert_y = 0.55
    gap = 0.12
    total_w = NUM_BUCKETS * expert_w + (NUM_BUCKETS - 1) * gap
    start_x = (14 - total_w) / 2
    for i, label in enumerate(BUCKET_LABELS):
        x = start_x + i * (expert_w + gap)
        active = i == 4
        face, edge = PALETTE["expert"]
        lw = 2.4 if active else 1.4
        patch = FancyBboxPatch(
            (x, expert_y),
            expert_w,
            expert_h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=lw,
            edgecolor=edge,
            facecolor=face,
            linestyle="-" if active else "--",
            alpha=1.0 if active else 0.72,
        )
        ax.add_patch(patch)
        ax.text(
            x + expert_w / 2,
            expert_y + expert_h / 2,
            f"Expert {label}\nLinear {concat_dim}→1",
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold" if active else "normal",
        )
        _arrow(ax, (7.0, 1.7), (x + expert_w / 2, expert_y + expert_h))

    # Output
    _box(
        ax,
        (5.1, -0.35),
        3.8,
        0.75,
        "tanh → expected reward\n[-1, +1] STM POV",
        face=PALETTE["out"][0],
        edge=PALETTE["out"][1],
        fontsize=10,
        weight="bold",
    )

    # Arrows between main blocks
    _arrow(ax, (3.0, 7.85), (3.45, 7.1))   # white input → white accumulator
    _arrow(ax, (11.0, 7.85), (10.55, 7.1))  # black input → black accumulator
    _arrow(ax, (3.45, 5.55), (5.4, 4.75))   # white accumulator → STM reorder
    _arrow(ax, (10.55, 5.55), (8.6, 4.75))  # black accumulator → STM reorder
    _arrow(ax, (7.0, 3.85), (7.0, 3.65))    # STM reorder → concat
    _arrow(ax, (7.0, 2.75), (7.0, 2.5))     # concat → router
    _arrow(ax, (7.0, 0.55), (7.0, 0.4))     # experts → output

    ax.text(
        0.35,
        -0.6,
        "Pilot checkpoint: W=128 · val_mse≈0.058 · inference via evaluate_nnue()",
        fontsize=9,
        color="#666666",
    )
    ax.text(
        13.65,
        -0.6,
        "encode_dual()",
        ha="right",
        fontsize=9,
        color="#666666",
        style="italic",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot SARDINE NNUE architecture")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "plots" / "sardine_nnue_architecture.png",
    )
    parser.add_argument("--hidden-dim", type=int, default=128, choices=[128, 256])
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args(argv)

    plot_architecture(hidden_dim=args.hidden_dim, output_path=args.output, dpi=args.dpi)
    print(f"Saved plot: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())