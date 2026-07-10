from __future__ import annotations

from pathlib import Path

from lib.datasets.roma import canonical_dataset_name


def vocab_filename(data_name: str) -> str:
    dataset = canonical_dataset_name(data_name)
    if dataset == "nr3d":
        return "nr3d_vocab.json"
    if dataset in {"scenedepict", "scanrefer", "3dllm"}:
        return "my_data_vocab.json"
    if "coco" in dataset:
        return "coco_precomp_vocab.json"
    return "f30k_precomp_vocab.json"


def require_bert_path(value: str | Path) -> Path:
    if not str(value):
        raise ValueError("BERT_PATH/--bert_path is required for BERT training")
    path = Path(value)
    required = [path / "config.json", path / "vocab.txt"]
    missing = [item.name for item in required if not item.is_file()]
    has_weights = (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file()
    if missing or not has_weights:
        details = [*missing]
        if not has_weights:
            details.append("model.safetensors or pytorch_model.bin")
        raise ValueError(f"invalid BERT_PATH {path}: missing {', '.join(details)}")
    return path
