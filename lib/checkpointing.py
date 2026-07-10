from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch


@dataclass(frozen=True)
class ResumeState:
    epoch: int
    best_score: float
    best_metrics: Dict[str, float]
    iterations: int


def build_checkpoint(
    *,
    epoch: int,
    model_state: object,
    optimizer: torch.optim.Optimizer,
    best_score: float,
    best_metrics: Mapping[str, float],
    iterations: int,
    config: Mapping[str, object],
    scheduler_state: Optional[Mapping[str, object]] = None,
    input_manifest: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    return {
        "format_version": 2,
        "epoch": int(epoch),
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": dict(scheduler_state or {}),
        "best_score": float(best_score),
        "best_metrics": {key: float(value) for key, value in best_metrics.items()},
        "iterations": int(iterations),
        "config": dict(config),
        "input_manifest": dict(input_manifest or {}),
    }


def restore_checkpoint(
    checkpoint: Mapping[str, object],
    optimizer: torch.optim.Optimizer,
    *,
    weights_only: bool = False,
) -> ResumeState:
    if int(checkpoint.get("format_version", 0)) < 2:
        if not weights_only:
            raise ValueError(
                "legacy checkpoint lacks optimizer state; use weights-only loading instead of resume"
            )
        return ResumeState(epoch=0, best_score=0.0, best_metrics={}, iterations=0)
    optimizer_state = checkpoint.get("optimizer")
    if not isinstance(optimizer_state, dict):
        raise ValueError("checkpoint is missing optimizer state")
    optimizer.load_state_dict(optimizer_state)
    raw_metrics = checkpoint.get("best_metrics") or {}
    if not isinstance(raw_metrics, Mapping):
        raise ValueError("checkpoint best_metrics must be a mapping")
    return ResumeState(
        epoch=int(checkpoint.get("epoch", 0)),
        best_score=float(checkpoint.get("best_score", 0.0)),
        best_metrics={key: float(value) for key, value in raw_metrics.items()},
        iterations=int(checkpoint.get("iterations", 0)),
    )
