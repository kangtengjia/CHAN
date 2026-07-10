from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RunPreset:
    epochs: int
    learning_rate: float
    lr_update: int
    batch_size: int
    precomp_enc_type: str


RUN_PRESETS = {
    ("scenedepict", "bigru"): RunPreset(1000, 3e-4, 20, 8, "selfattention"),
    ("scenedepict", "bert"): RunPreset(45, 3e-4, 20, 8, "basic"),
    ("scanrefer", "bigru"): RunPreset(300, 3e-4, 14, 8, "selfattention"),
    ("scanrefer", "bert"): RunPreset(30, 3e-4, 14, 8, "basic"),
    ("nr3d", "bigru"): RunPreset(800, 3e-4, 14, 128, "selfattention"),
    ("nr3d", "bert"): RunPreset(30, 3e-4, 14, 8, "basic"),
    ("3dllm", "bigru"): RunPreset(1000, 3e-5, 14, 16, "selfattention"),
    ("3dllm", "bert"): RunPreset(30, 3e-4, 14, 8, "basic"),
}


def format_markdown_metrics(dataset: str, text_encoder: str, metrics: Mapping[str, float]) -> str:
    headers = ["Dataset", "Text", "R@1", "R@5", "R@10", "R@30", "Rsum", "MedR", "MeanR"]
    values = [
        dataset,
        text_encoder,
        *(f"{float(metrics[key]):.2f}" for key in ["R@1", "R@5", "R@10", "R@30", "Rsum", "MedR", "MeanR"]),
    ]
    return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n| " + " | ".join(values) + " |\n"
