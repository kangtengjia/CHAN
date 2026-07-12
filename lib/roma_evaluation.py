from __future__ import annotations

import statistics
from typing import Dict, Sequence

import numpy as np


DEFAULT_KS = (1, 5, 10, 30)


def unique_scene_rows(
    encoded_rows: np.ndarray,
    row_lengths: Sequence[int],
    scene_indices: Sequence[int],
    *,
    scene_count: int,
) -> tuple[np.ndarray, list[int]]:
    if len(encoded_rows) != len(scene_indices) or len(row_lengths) != len(scene_indices):
        raise ValueError("encoded row, length, and scene-index counts must match")
    selected = [None] * int(scene_count)
    selected_lengths = [None] * int(scene_count)
    for row_index, scene_index in enumerate(scene_indices):
        target = int(scene_index)
        if not 0 <= target < scene_count:
            raise ValueError(f"scene index {target} is outside [0, {scene_count})")
        if selected[target] is None:
            selected[target] = np.asarray(encoded_rows[row_index])
            selected_lengths[target] = int(row_lengths[row_index])
    missing = [index for index, row in enumerate(selected) if row is None]
    if missing:
        raise ValueError(f"encoded captions do not cover feature scenes: {missing[:5]}")
    return np.stack(selected), [int(value) for value in selected_lengths]


def text_to_scene_metrics(
    similarities: np.ndarray,
    caption_scene_indices: Sequence[int],
    *,
    ks: Sequence[int] = DEFAULT_KS,
) -> Dict[str, float]:
    scores = np.asarray(similarities)
    if scores.ndim != 2:
        raise ValueError(f"similarities must be rank 2, found {scores.shape}")
    if scores.shape[0] != len(caption_scene_indices):
        raise ValueError(
            f"caption count mismatch: {scores.shape[0]} similarity rows vs "
            f"{len(caption_scene_indices)} labels"
        )
    scene_count = scores.shape[1]
    ranks = []
    for row_index, scene_index in enumerate(caption_scene_indices):
        target = int(scene_index)
        if not 0 <= target < scene_count:
            raise ValueError(f"scene index {target} is outside [0, {scene_count})")
        ordered = np.argsort(scores[row_index])[::-1]
        ranks.append(int(np.where(ordered == target)[0][0]) + 1)
    denominator = max(len(ranks), 1)
    metrics: Dict[str, float] = {"queries": len(ranks)}
    recall_sum = 0.0
    for k in ks:
        cutoff = int(k)
        recall = 100.0 * sum(rank <= cutoff for rank in ranks) / denominator
        metrics[f"R@{cutoff}"] = recall
        recall_sum += recall
    metrics["Rsum"] = recall_sum
    metrics["MedR"] = float(statistics.median(ranks)) if ranks else 0.0
    metrics["MeanR"] = float(np.mean(ranks)) if ranks else 0.0
    metrics["MRR"] = 100.0 * float(np.mean([1.0 / rank for rank in ranks])) if ranks else 0.0
    return metrics
