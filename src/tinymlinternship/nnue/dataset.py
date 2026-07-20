"""Labeled / ChessBench parquet → PyTorch batches for NNUE training.

Accepts either precomputed sparse feature columns (``white_features`` /
``black_features``) or production rows with ``fen`` only — features are then
encoded on the fly via ``encode_dual`` (ASSETS schema).
"""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from tinymlinternship.features import FEATURE_DIM, encode_dual


def indices_to_binary(indices: np.ndarray | list[int], dim: int = FEATURE_DIM) -> torch.Tensor:
    x = torch.zeros(dim, dtype=torch.float32)
    if len(indices) > 0:
        idx = torch.as_tensor(np.array(indices, dtype=np.int64, copy=True), dtype=torch.long)
        x[idx] = 1.0
    return x


class ChessbenchDataset(Dataset):
    def __init__(self, parquet_path: Path | str) -> None:
        self.df = pd.read_parquet(parquet_path)
        has_feat = (
            "white_features" in self.df.columns and "black_features" in self.df.columns
        )
        has_fen = "fen" in self.df.columns
        if not has_feat and not has_fen:
            raise ValueError(
                f"{parquet_path}: need white_features/black_features or fen "
                "(production labeled schema)"
            )
        if "expected_reward" not in self.df.columns:
            raise ValueError(f"{parquet_path}: missing expected_reward")
        self._encode_from_fen = not has_feat

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | bool]:
        row = self.df.iloc[index]
        if self._encode_from_fen:
            white_idx, black_idx = encode_dual(chess.Board(str(row["fen"])))
            white = indices_to_binary(white_idx)
            black = indices_to_binary(black_idx)
        else:
            white = indices_to_binary(row["white_features"])
            black = indices_to_binary(row["black_features"])
        return {
            "white_features": white,
            "black_features": black,
            "bucket_id": int(row["bucket_id"]),
            "stm_white": bool(row["stm_white"]),
            "target": float(row["expected_reward"]),
        }