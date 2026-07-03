"""Curated Lc0 training-data shard catalog (no network I/O)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

LC0_BASE_URL = "https://storage.lczero.org/files/training_data/"


@dataclass(frozen=True)
class Lc0Shard:
    """One training-data tar shard hosted on storage.lczero.org."""

    name: str
    size_bytes: int
    note: str = ""

    @property
    def url(self) -> str:
        return f"{LC0_BASE_URL}{self.name}"

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024**3)


# Curated shards that fit the blueprint ~1–2 GB budget (sizes from the public index).
DEFAULT_SHARDS: tuple[Lc0Shard, ...] = (
    Lc0Shard(
        "training-run1--20240819-1917.tar",
        798_709_760,
        "run1, Aug 2024 — ~762 MiB",
    ),
    Lc0Shard(
        "training-run1--20250209-1017.tar",
        439_726_080,
        "run1, Feb 2025 — ~419 MiB",
    ),
)

EXTRA_SHARDS: tuple[Lc0Shard, ...] = (
    Lc0Shard("training-run1--20230505-0917.tar", 407_756_800, "run1, May 2023 — ~389 MiB"),
    Lc0Shard("training-run2--20210605-0517.tar", 770_478_080, "run2, Jun 2021 — ~735 MiB"),
    Lc0Shard("training-run1--20210605-0516.tar", 58_460_160, "run1, Jun 2021 — ~56 MiB"),
)

ALL_SHARDS: tuple[Lc0Shard, ...] = DEFAULT_SHARDS + EXTRA_SHARDS


def select_shards(
    shards: Iterable[Lc0Shard],
    max_gb: float,
    names: list[str] | None = None,
) -> list[Lc0Shard]:
    """Pick shards until the byte budget is reached."""
    if names:
        by_name = {s.name: s for s in ALL_SHARDS}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise ValueError(f"Unknown shard(s): {', '.join(missing)}")
        return [by_name[n] for n in names]

    chosen: list[Lc0Shard] = []
    budget = int(max_gb * (1024**3))
    total = 0
    for shard in shards:
        if total + shard.size_bytes > budget and chosen:
            break
        chosen.append(shard)
        total += shard.size_bytes
    return chosen