#!/usr/bin/env python3
"""Rank test-slice positions by WDL cross-entropy for a DualHidden or linear checkpoint.

Prints the lowest-CE (best) and highest-CE (worst) FENs with teacher STM
``wdl`` and model ``hat wdl``. Optional ``--sample`` ranks a random subset.

The slice JSON is aligned with ``features.npz`` (same unique-EPD order as encode).

Examples::

    py -3.12 -u scripts/inspect_nnue_positions.py dual_W64_H128
    py -3.12 -u scripts/inspect_nnue_positions.py dual_W64_H128 --ckpt best --n 10
    py -3.12 -u scripts/inspect_nnue_positions.py linear_wdl_smoke --ckpt best --n 20 --sample 1000 --slice fen_value_visits_lichess_db_standard_rated_2026-07_0-5000_d90
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from tinymlinternship.config.settings import NNUE_CHECKPOINTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from tinymlinternship.data.wdl import wdl_from_row
from tinymlinternship.data.board_store import BOARD_EVAL_DIR_NAME, FEN_VALUE_VISITS_DIR_NAME
from tinymlinternship.nnue.dataset import (
    FenValueVisitsDataset,
    collate_sparse,
    epd_key,
    load_fen_value_visits,
    load_slice_feature_db,
    slice_source_json,
)
from tinymlinternship.nnue.model import DualHiddenNNUE, LinearWDLNNUE, MediumWDLNNUE

DEFAULT_SLICES = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_DIR_NAME
DEFAULT_TEST_SLICE = "fen_value_visits_lichess_db_standard_rated_2026-07_100000-105000_d80_draw5"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_checkpoint(name: str, *, ckpt: str) -> Path:
    """``dual_W64_H128`` → ``models/checkpoints/nnue/dual_W64_H128/{last|best}.pt``."""
    raw = Path(name)
    if raw.suffix.lower() == ".pt":
        path = _resolve(raw)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    run_dir = raw if raw.is_absolute() else (NNUE_CHECKPOINTS_DIR / name)
    if ckpt not in {"last", "best"}:
        raise ValueError("--ckpt must be last or best")
    path = run_dir / f"{ckpt}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def aligned_fens(folder: Path) -> list[str]:
    """FENs in the same unique-EPD order as ``features.npz``."""
    source = slice_source_json(folder)
    if source is None:
        raise FileNotFoundError(f"no slice JSON in {folder}")
    df = load_fen_value_visits(source)
    df = df.copy()
    df["epd"] = df["fen"].map(epd_key)
    df = df.drop_duplicates(subset=["epd"], keep="first").reset_index(drop=True)
    stored = load_slice_feature_db(folder)["wdl"]
    teacher = np.stack(
        [np.asarray(wdl_from_row(row), dtype=np.float32) for row in df.to_dict(orient="records")]
    )
    if len(teacher) != len(stored):
        raise ValueError(
            f"{folder.name}: JSON unique EPDs {len(teacher):,} != features.npz {len(stored):,}"
        )
    if not np.allclose(teacher, stored, atol=1e-3, rtol=0.0):
        raise ValueError(f"{folder.name}: JSON WDL does not match features.npz (re-encode the slice)")
    return df["fen"].astype(str).tolist()


def load_model(path: Path) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    architecture = str(payload.get("architecture", "dual_hidden_wdl"))
    if architecture == LinearWDLNNUE.architecture:
        model: torch.nn.Module = LinearWDLNNUE()
    elif architecture == MediumWDLNNUE.architecture:
        model = MediumWDLNNUE(hidden_dim=int(payload.get("hidden_dim", 20)))
    else:
        model = DualHiddenNNUE(
            hidden_dim=int(payload.get("hidden_dim", 64)),
            hidden2_dim=int(payload.get("hidden2_dim", 128)),
        )
    state = payload["model_state_dict"]
    state = {key.replace("_orig_mod.", ""): tensor for key, tensor in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def predict(model: torch.nn.Module, folder: Path) -> np.ndarray:
    dataset = FenValueVisitsDataset(folder, progress=False)
    loader = DataLoader(dataset, batch_size=2048, shuffle=False, collate_fn=collate_sparse)
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            chunks.append(
                model.forward_sparse(
                    batch["white_idx"],
                    batch["white_mask"],
                    batch["black_idx"],
                    batch["black_mask"],
                    batch["stm_white"],
                ).cpu()
            )
    logits = torch.cat(chunks)
    probs = torch.softmax(logits, dim=-1)
    return probs.numpy().astype(np.float64)


def rank_rows(
    fens: list[str],
    teacher: np.ndarray,
    pred: np.ndarray,
    *,
    n: int,
) -> tuple[list[dict], list[dict]]:
    t = np.clip(teacher, 1e-8, 1.0)
    p = np.clip(pred, 1e-8, 1.0)
    ce = -np.sum(t * np.log(p), axis=-1)
    n = min(int(n), len(fens))
    best_idx = np.argsort(ce)[:n]
    worst_idx = np.argsort(ce)[::-1][:n]

    def pack(indices: np.ndarray) -> list[dict]:
        rows = []
        for i in indices:
            tw, td, tl = (float(x) for x in teacher[int(i)])
            pw, pd, pl = (float(x) for x in pred[int(i)])
            rows.append(
                {
                    "fen": fens[int(i)],
                    "wdl": [round(tw, 4), round(td, 4), round(tl, 4)],
                    "hat_wdl": [round(pw, 4), round(pd, 4), round(pl, 4)],
                    "v": round(tw - tl, 4),
                    "hat_v": round(pw - pl, 4),
                    "ce": round(float(ce[int(i)]), 4),
                }
            )
        return rows

    return pack(best_idx), pack(worst_idx)


def sample_subset(
    fens: list[str],
    teacher: np.ndarray,
    pred: np.ndarray,
    *,
    n: int,
    seed: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Draw ``n`` positions without replacement (full slice if ``n >= N``)."""
    n_all = len(fens)
    take = min(int(n), n_all)
    if take <= 0:
        raise ValueError("--sample must be > 0")
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n_all, size=take, replace=False)
    picked = [fens[int(i)] for i in idx]
    return picked, teacher[idx], pred[idx]


def _print_block(title: str, rows: list[dict]) -> None:
    print(title)
    print(
        f"{'#':>3}  {'W':>6} {'D':>6} {'L':>6}  "
        f"{'pW':>6} {'pD':>6} {'pL':>6}  {'CE':>7}  fen"
    )
    for i, row in enumerate(rows, start=1):
        tw, td, tl = row["wdl"]
        pw, pd, pl = row["hat_wdl"]
        print(
            f"{i:3d}  {tw:6.3f} {td:6.3f} {tl:6.3f}  "
            f"{pw:6.3f} {pd:6.3f} {pl:6.3f}  {row['ce']:7.3f}  {row['fen']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Top-N best and worst WDL predictions on a slice"
    )
    parser.add_argument(
        "model",
        help="Run name under models/checkpoints/nnue/ (e.g. dual_W64_H128) or a .pt path",
    )
    parser.add_argument("--ckpt", choices=("last", "best"), default="last")
    parser.add_argument("--n", type=int, default=10, help="How many FENs per list")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Rank a random subset of this many positions (0 = full slice)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for --sample")
    parser.add_argument(
        "--slice",
        default=DEFAULT_TEST_SLICE,
        help="Slice folder name under fen_value_visits/ (default: test holdout)",
    )
    parser.add_argument(
        "--slices-dir",
        type=Path,
        default=DEFAULT_SLICES,
    )
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON dump of both lists")
    args = parser.parse_args(argv)

    try:
        ckpt_path = resolve_checkpoint(args.model, ckpt=args.ckpt)
    except FileNotFoundError as exc:
        print(f"checkpoint not found: {exc}", file=sys.stderr)
        return 1

    folder = _resolve(args.slices_dir) / args.slice
    if not folder.is_dir():
        print(f"slice folder not found: {folder}", file=sys.stderr)
        return 1

    print(f"model  {ckpt_path}")
    print(f"slice  {folder}")
    fens = aligned_fens(folder)
    teacher = load_slice_feature_db(folder)["wdl"].astype(np.float64)
    model = load_model(ckpt_path)
    pred = predict(model, folder)
    if len(pred) != len(fens):
        print(f"length mismatch: {len(fens)} FENs vs {len(pred)} preds", file=sys.stderr)
        return 1

    n_all = len(fens)
    if args.sample > 0:
        fens, teacher, pred = sample_subset(
            fens, teacher, pred, n=args.sample, seed=args.seed
        )
        print(f"sample {len(fens):,} of {n_all:,} (seed={args.seed})")
    else:
        print(f"rows   {n_all:,} (full slice)")

    best, worst = rank_rows(fens, teacher, pred, n=args.n)
    print()
    _print_block(f"Top {len(best)} best  (lowest CE)", best)
    print()
    _print_block(f"Top {len(worst)} worst (highest CE)", worst)

    if args.json is not None:
        out = _resolve(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "checkpoint": str(ckpt_path),
                    "slice": args.slice,
                    "sample": int(args.sample),
                    "seed": int(args.seed),
                    "n_ranked": len(fens),
                    "best": best,
                    "worst": worst,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nJSON → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
