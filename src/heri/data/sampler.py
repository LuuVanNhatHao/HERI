"""Optional ID-aware sampler foundation."""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Mapping, Sequence

from torch.utils.data import Sampler


class IDAwareSampler(Sampler[int]):
    """Deterministic sampler that preserves all variants while shuffling by ID.

    Disabled by default; preserves all variants when enabled.
    """

    def __init__(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        seed: int,
        shuffle: bool = True,
    ) -> None:
        self.rows = rows
        self.seed = seed
        self.shuffle = shuffle
        groups: dict[int, list[int]] = {}
        for index, row in enumerate(rows):
            groups.setdefault(int(row["id"]), []).append(index)
        self.groups = groups

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed)
        ids = list(self.groups)
        if self.shuffle:
            rng.shuffle(ids)
        for sample_id in ids:
            indices = list(self.groups[sample_id])
            if self.shuffle:
                rng.shuffle(indices)
            yield from indices

    def __len__(self) -> int:
        return len(self.rows)
