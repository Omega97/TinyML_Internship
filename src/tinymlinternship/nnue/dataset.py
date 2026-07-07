"""ChessBench parquet → PyTorch batches for NNUE training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from tinymlinternship.features import FEATURE_DIM


def indices_to_binary(indices: np.ndarray | list[int], dim: int = FEATURE_DIM) -> torch.Tensor:
    x = torch.zeros(dim, dtype=torch.float32)
    if len(indices) > 0:
        idx = torch.as_tensor(np.array(indices, dtype=np.int64, copy=True), dtype=torch.long)
        x[idx] = 1.0
    return x


class ChessbenchDataset(Dataset):
    def __init__(self, parquet_path: Path | str) -> None:
        self.df = pd.read_parquet(parquet_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | bool]:
        row = self.df.iloc[index]
        return {
            "white_features": indices_to_binary(row["white_features"]),
            "black_features": indices_to_binary(row["black_features"]),
            "bucket_id": int(row["bucket_id"]),
            "stm_white": bool(row["stm_white"]),
            "target": float(row["expected_reward"]),
        }