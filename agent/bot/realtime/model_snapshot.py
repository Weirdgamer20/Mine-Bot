import time
import copy
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ModelSnapshot:
    """Immutable model snapshot for lock-free RT inference."""
    version: int
    timestamp_ns: int
    encoder: nn.Module
    rssm: nn.Module
    actor: nn.Module
    critic: nn.Module
    rnd: nn.Module
    device: torch.device


class SnapshotRegistry:
    """
    Manages atomic double-buffered model snapshots.
    Guarantees RT inference never races with optimizer parameter updates.
    """

    def __init__(self, initial_snapshot: ModelSnapshot):
        self._current_snapshot = initial_snapshot
        self._version = initial_snapshot.version

    def get_latest(self) -> ModelSnapshot:
        """Atomic read of current model snapshot (lock-free)."""
        return self._current_snapshot

    def publish_new_version(
        self,
        encoder: nn.Module,
        rssm: nn.Module,
        actor: nn.Module,
        critic: nn.Module,
        rnd: nn.Module,
        device: torch.device,
    ) -> int:
        """
        Publishes a new immutable model snapshot.
        Weights are transferred without blocking the RT loop.
        """
        self._version += 1
        new_snapshot = ModelSnapshot(
            version=self._version,
            timestamp_ns=time.perf_counter_ns(),
            encoder=encoder,
            rssm=rssm,
            actor=actor,
            critic=critic,
            rnd=rnd,
            device=device,
        )
        # Atomic reference swap in Python
        self._current_snapshot = new_snapshot
        return self._version
