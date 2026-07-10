from __future__ import annotations

import json
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterator, List, Sequence

import numpy as np
from torch.utils.data import Sampler


DATASET_ALIASES = {
    "our_data": "scenedepict",
    "3d_text_retrv": "scenedepict",
    "scenedepict-3d2t": "scenedepict",
    "3dllm": "3dllm",
    "3d_llm": "3dllm",
    "llm-3d-scene": "3dllm",
}


@dataclass(frozen=True)
class RoMaBundle:
    captions: List[str]
    scene_ids: List[str]
    scene_indices: List[int]
    feature_scene_ids: List[str]
    features: np.ndarray


def source_paths(data_root: str | Path, data_name: str, data_split: str) -> List[Path]:
    root = Path(data_root)
    dataset = canonical_dataset_name(data_name)
    split = _split_tag(data_split)
    if dataset == "scanrefer":
        return [root / f"scanrefer_{split}.jsonl", root / f"pt2vec_200_random_{split}.npy"]
    if dataset == "nr3d":
        return [
            root / f"nr3d_{split}.jsonl",
            root / "scanrefer_train.jsonl",
            root / "scanrefer_val.jsonl",
            root / "pt2vec_200_random_train.npy",
            root / "pt2vec_200_random_val.npy",
            root / "3D_Text_Retrv_train_final_sorted.json",
            root / "3D_Text_Retrv_val_final_sorted.json",
            root / "3D_Text_Retrv_grid_train.npy",
            root / "3D_Text_Retrv_grid_val.npy",
        ]
    if dataset == "scenedepict":
        return [
            root / f"3D_Text_Retrv_{split}_final_sorted.json",
            root / f"3D_Text_Retrv_grid_{split}.npy",
        ]
    if dataset == "3dllm":
        return [
            root / f"3d_llm_scene_description_{split}_sorted.json",
            root / f"3d_llm_grid_{split}.npy",
        ]
    raise ValueError(f"unsupported RoMa dataset: {dataset}")


def canonical_dataset_name(data_name: str) -> str:
    lowered = str(data_name).strip().lower()
    return DATASET_ALIASES.get(lowered, lowered)


def _split_tag(data_split: str) -> str:
    return "train" if data_split == "train" else "val"


def _read_json(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON list: {path}")
    return payload


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _ordered_unique(values: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(values))


def _feature_filename(dataset: str, split: str, pc_variant: str, uni3d_inst_k: int) -> str:
    if dataset in {"scanrefer", "nr3d"}:
        if pc_variant == "base_plus_uni3dinst":
            return f"pt2vec_200_plus_uni3dinst_k{int(uni3d_inst_k)}_{split}.npy"
        return f"pt2vec_200_random_{split}.npy"
    if dataset == "scenedepict":
        return f"3D_Text_Retrv_grid_{split}.npy"
    if dataset == "3dllm":
        return f"3d_llm_grid_{split}.npy"
    raise ValueError(f"unsupported RoMa dataset: {dataset}")


def _load_rows(root: Path, dataset: str, split: str) -> tuple[List[str], List[str]]:
    if dataset == "scanrefer":
        rows = _read_jsonl(root / f"scanrefer_{split}.jsonl")
        return [str(row["description"]) for row in rows], [str(row["scene_id"]) for row in rows]
    if dataset == "nr3d":
        rows = _read_jsonl(root / f"nr3d_{split}.jsonl")
        return [str(row["description"]) for row in rows], [str(row.get("scan_id") or row["scene_id"]) for row in rows]
    if dataset == "scenedepict":
        rows = _read_json(root / f"3D_Text_Retrv_{split}_final_sorted.json")
        return [str(row["description"]) for row in rows], [str(row["scene_id"]) for row in rows]
    if dataset == "3dllm":
        rows = _read_json(root / f"3d_llm_scene_description_{split}_sorted.json")
        captions = []
        for row in rows:
            answers = row.get("answers")
            if not isinstance(answers, list) or not answers:
                raise ValueError("3dllm row must contain a non-empty answers list")
            captions.append(str(answers[0]))
        return captions, [str(row["scene_id"]) for row in rows]
    raise ValueError(f"unsupported RoMa dataset: {dataset}")


def _scanrefer_feature_pool(
    root: Path,
    *,
    pc_variant: str,
    uni3d_inst_k: int,
) -> tuple[List[str], np.ndarray]:
    orders = []
    arrays = []
    for split in ("train", "val"):
        rows = _read_jsonl(root / f"scanrefer_{split}.jsonl")
        order = _ordered_unique([str(row["scene_id"]) for row in rows])
        path = root / _feature_filename("scanrefer", split, pc_variant, uni3d_inst_k)
        array = np.load(path)
        if len(order) != len(array):
            raise ValueError(
                f"ScanRefer feature scene count mismatch for {split}: {len(order)} ids vs {len(array)} rows"
            )
        orders.extend(order)
        arrays.append(array)
    if len(orders) != len(set(orders)):
        raise ValueError("ScanRefer train/val feature scene orders overlap")
    return orders, np.concatenate(arrays, axis=0)


def _scenedepict_feature_pool(root: Path) -> tuple[List[str], np.ndarray]:
    orders = []
    arrays = []
    for split in ("train", "val"):
        rows = _read_json(root / f"3D_Text_Retrv_{split}_final_sorted.json")
        order = _ordered_unique([str(row["scene_id"]) for row in rows])
        array = np.load(root / f"3D_Text_Retrv_grid_{split}.npy")
        if len(order) != len(array):
            raise ValueError(
                f"SceneDepict feature scene count mismatch for {split}: {len(order)} ids vs {len(array)} rows"
            )
        orders.extend(order)
        arrays.append(array)
    return orders, np.concatenate(arrays, axis=0)


def load_roma_bundle(
    data_root: str | Path,
    data_name: str,
    data_split: str,
    *,
    pc_variant: str = "base",
    uni3d_inst_k: int = 20,
    feature_path: str | Path | None = None,
) -> RoMaBundle:
    root = Path(data_root)
    dataset = canonical_dataset_name(data_name)
    split = _split_tag(data_split)
    captions, scene_ids = _load_rows(root, dataset, split)
    if dataset == "nr3d" and feature_path is None:
        pool_scene_ids, pool_features = _scanrefer_feature_pool(
            root, pc_variant=pc_variant, uni3d_inst_k=uni3d_inst_k
        )
        depict_scene_ids, depict_features = _scenedepict_feature_pool(root)
        feature_by_scene = {
            scene_id: pool_features[index]
            for index, scene_id in enumerate(pool_scene_ids)
        }
        for index, scene_id in enumerate(depict_scene_ids):
            feature_by_scene.setdefault(scene_id, depict_features[index])
        feature_scene_ids = _ordered_unique(scene_ids)
        missing = sorted(set(feature_scene_ids) - set(feature_by_scene))
        if missing:
            raise ValueError(f"caption scenes missing from RoMa DGCNN feature pools: {missing[:5]}")
        features = np.stack([feature_by_scene[scene_id] for scene_id in feature_scene_ids])
        resolved_feature_path = root
    else:
        feature_scene_ids = _ordered_unique(scene_ids)
        resolved_feature_path = Path(feature_path) if feature_path else root / _feature_filename(
            dataset, split, pc_variant, uni3d_inst_k
        )
        features = np.load(resolved_feature_path)
    if features.ndim != 3:
        raise ValueError(f"expected rank-3 point features, found {features.shape}: {resolved_feature_path}")
    if len(feature_scene_ids) != len(features):
        raise ValueError(
            f"feature scene count mismatch for {dataset}/{split}: "
            f"{len(feature_scene_ids)} scene ids vs {len(features)} feature rows"
        )
    feature_index = {scene_id: index for index, scene_id in enumerate(feature_scene_ids)}
    missing = sorted(set(scene_ids) - set(feature_index))
    if missing:
        raise ValueError(f"caption scenes missing from feature order: {missing[:5]}")
    scene_indices = [feature_index[scene_id] for scene_id in scene_ids]
    return RoMaBundle(captions, scene_ids, scene_indices, feature_scene_ids, features)


class SceneUniqueBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        scene_indices: Sequence[int],
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 2022,
    ) -> None:
        self.scene_indices = list(scene_indices)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        unique_scenes = len(set(self.scene_indices))
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.batch_size > unique_scenes:
            raise ValueError(
                f"batch_size {self.batch_size} exceeds the {unique_scenes} unique scenes"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        grouped: Dict[int, Deque[int]] = defaultdict(deque)
        for index, scene_index in enumerate(self.scene_indices):
            grouped[scene_index].append(index)
        if self.shuffle:
            for indices in grouped.values():
                shuffled = list(indices)
                rng.shuffle(shuffled)
                indices.clear()
                indices.extend(shuffled)

        active = list(grouped)
        while active:
            if self.shuffle:
                rng.shuffle(active)
            batch_scenes = active[: self.batch_size]
            batch = [grouped[scene_index].popleft() for scene_index in batch_scenes]
            active = [scene_index for scene_index in active if grouped[scene_index]]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.scene_indices) // self.batch_size
        return math.ceil(len(self.scene_indices) / self.batch_size)
