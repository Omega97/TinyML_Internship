#!/usr/bin/env python3
"""Download Hugging Face teacher checkpoints (chess_lite, Artoria Zero small)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from huggingface_hub import hf_hub_download

from tinymlinternship.config.settings import (
    ARTORIA_SMALL_CKPT,
    ARTORIA_SMALL_CONFIG,
    CHESS_LITE_WEIGHTS,
    TEACHER_DIR,
)


def download_chess_lite() -> Path:
    dest = CHESS_LITE_WEIGHTS.parent
    dest.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download("satana123/chess_lite", "chess_lite.pth", local_dir=str(dest))
    )


def download_artoria_small() -> tuple[Path, Path]:
    dest = ARTORIA_SMALL_CKPT.parent
    dest.mkdir(parents=True, exist_ok=True)
    ckpt = Path(
        hf_hub_download(
            "Shinapri/artoria-zero",
            "small/checkpoint.pt",
            local_dir=str(dest.parent),
        )
    )
    cfg = Path(
        hf_hub_download(
            "Shinapri/artoria-zero",
            "small/config.json",
            local_dir=str(dest.parent),
        )
    )
    return ckpt, cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download HF teacher models")
    parser.add_argument(
        "--model",
        choices=("all", "chess_lite", "artoria"),
        default="all",
        help="Which checkpoint to fetch (default: all)",
    )
    args = parser.parse_args(argv)

    TEACHER_DIR.mkdir(parents=True, exist_ok=True)

    if args.model in ("all", "chess_lite"):
        path = download_chess_lite()
        print(f"chess_lite: {path} ({path.stat().st_size // (1024 * 1024)} MiB)")

    if args.model in ("all", "artoria"):
        ckpt, cfg = download_artoria_small()
        print(f"artoria-small: {ckpt} ({ckpt.stat().st_size // (1024 * 1024)} MiB)")
        print(f"artoria config: {cfg}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())