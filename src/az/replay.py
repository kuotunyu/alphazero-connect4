"""Replay buffer of (bitboard state, pi, z) with sample-time augmentation.

Positions are stored compactly (two uint64 + float16 pi + int8 z, ~25 bytes
each) and decoded to planes only when sampled. Horizontal-flip augmentation
happens at sample time with p=0.5 so every draw of a position may be either
mirror image; the buffer itself stores originals only.
"""

from __future__ import annotations

import numpy as np

from az import game


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.positions = np.zeros(capacity, dtype=np.uint64)
        self.masks = np.zeros(capacity, dtype=np.uint64)
        self.pis = np.zeros((capacity, game.COLS), dtype=np.float16)
        self.zs = np.zeros(capacity, dtype=np.int8)
        self.size = 0
        self.next = 0

    def __len__(self) -> int:
        return self.size

    def add(self, samples) -> None:
        """samples: iterable of ((position, mask), pi, z)."""
        for (position, mask), pi, z in samples:
            i = self.next
            self.positions[i] = position
            self.masks[i] = mask
            self.pis[i] = pi
            self.zs[i] = int(z)
            self.next = (i + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng, augment: bool = True):
        """Returns (planes [B,3,6,7] f32, pis [B,7] f32, zs [B] f32)."""
        idx = rng.integers(self.size, size=batch_size)
        planes = np.empty((batch_size, 3, game.ROWS, game.COLS), dtype=np.float32)
        pis = self.pis[idx].astype(np.float32)
        zs = self.zs[idx].astype(np.float32)
        flip = rng.random(batch_size) < 0.5 if augment else np.zeros(batch_size, bool)
        for j, i in enumerate(idx):
            p = game.encode((int(self.positions[i]), int(self.masks[i])))
            planes[j] = p[:, :, ::-1] if flip[j] else p
        if augment and flip.any():
            pis[flip] = pis[flip, ::-1]
        return planes, pis, zs

    # --- checkpoint (de)serialization -----------------------------------

    def state_dict(self) -> dict:
        n = self.size
        return {
            "positions": self.positions[:n].copy(),
            "masks": self.masks[:n].copy(),
            "pis": self.pis[:n].copy(),
            "zs": self.zs[:n].copy(),
            "capacity": self.capacity,
            "next": self.next,
        }

    @classmethod
    def from_state_dict(cls, sd: dict) -> "ReplayBuffer":
        buf = cls(sd["capacity"])
        n = len(sd["positions"])
        buf.positions[:n] = sd["positions"]
        buf.masks[:n] = sd["masks"]
        buf.pis[:n] = sd["pis"]
        buf.zs[:n] = sd["zs"]
        buf.size = n
        buf.next = sd["next"] if n == buf.capacity else n % buf.capacity
        return buf
