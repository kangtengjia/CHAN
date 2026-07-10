import json

import numpy as np
import pytest

from lib.datasets.roma import SceneUniqueBatchSampler, load_roma_bundle


def _write_json(path, rows):
    path.write_text(json.dumps(rows), encoding="utf-8")


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_loads_scenedepict_with_explicit_scene_indices(tmp_path):
    _write_json(
        tmp_path / "3D_Text_Retrv_train_final_sorted.json",
        [
            {"scene_id": "scene_a", "description": "a one"},
            {"scene_id": "scene_a", "description": "a two"},
            {"scene_id": "scene_b", "description": "b one"},
        ],
    )
    np.save(tmp_path / "3D_Text_Retrv_grid_train.npy", np.zeros((2, 3, 4), dtype=np.float32))

    bundle = load_roma_bundle(tmp_path, "scenedepict", "train")

    assert bundle.captions == ["a one", "a two", "b one"]
    assert bundle.scene_ids == ["scene_a", "scene_a", "scene_b"]
    assert bundle.scene_indices == [0, 0, 1]
    assert bundle.feature_scene_ids == ["scene_a", "scene_b"]
    assert bundle.features.shape == (2, 3, 4)


def test_nr3d_maps_noncontiguous_rows_through_scanrefer_feature_order(tmp_path):
    _write_jsonl(
        tmp_path / "scanrefer_train.jsonl",
        [
            {"scene_id": "scene_a", "description": "a"},
            {"scene_id": "scene_a", "description": "a2"},
            {"scene_id": "scene_b", "description": "b"},
            {"scene_id": "scene_c", "description": "c"},
        ],
    )
    _write_jsonl(
        tmp_path / "scanrefer_val.jsonl",
        [{"scene_id": "scene_d", "description": "d"}],
    )
    _write_json(
        tmp_path / "3D_Text_Retrv_train_final_sorted.json",
        [{"scene_id": "scene_e", "description": "e"}],
    )
    _write_json(tmp_path / "3D_Text_Retrv_val_final_sorted.json", [])
    _write_jsonl(
        tmp_path / "nr3d_train.jsonl",
        [
            {"scan_id": "scene_c", "description": "c first"},
            {"scan_id": "scene_a", "description": "a first"},
            {"scan_id": "scene_d", "description": "d first"},
            {"scan_id": "scene_e", "description": "e first"},
        ],
    )
    np.save(tmp_path / "pt2vec_200_random_train.npy", np.zeros((3, 2, 4), dtype=np.float32))
    np.save(tmp_path / "pt2vec_200_random_val.npy", np.zeros((1, 2, 4), dtype=np.float32))
    np.save(tmp_path / "3D_Text_Retrv_grid_train.npy", np.ones((1, 2, 4), dtype=np.float32))
    np.save(tmp_path / "3D_Text_Retrv_grid_val.npy", np.zeros((0, 2, 4), dtype=np.float32))

    bundle = load_roma_bundle(tmp_path, "nr3d", "train")

    assert bundle.feature_scene_ids == ["scene_c", "scene_a", "scene_d", "scene_e"]
    assert bundle.scene_indices == [0, 1, 2, 3]
    assert bundle.features[3].mean() == 1.0
    assert len(bundle.captions) == 4


def test_rejects_caption_scene_missing_from_feature_order(tmp_path):
    _write_jsonl(
        tmp_path / "scanrefer_val.jsonl",
        [{"scene_id": "scene_a", "description": "a"}],
    )
    _write_jsonl(tmp_path / "scanrefer_train.jsonl", [])
    _write_json(tmp_path / "3D_Text_Retrv_train_final_sorted.json", [])
    _write_json(tmp_path / "3D_Text_Retrv_val_final_sorted.json", [])
    _write_jsonl(
        tmp_path / "nr3d_val.jsonl",
        [{"scan_id": "scene_missing", "description": "missing"}],
    )
    np.save(tmp_path / "pt2vec_200_random_val.npy", np.zeros((1, 2, 4), dtype=np.float32))
    np.save(tmp_path / "pt2vec_200_random_train.npy", np.zeros((0, 2, 4), dtype=np.float32))
    np.save(tmp_path / "3D_Text_Retrv_grid_train.npy", np.zeros((0, 2, 4), dtype=np.float32))
    np.save(tmp_path / "3D_Text_Retrv_grid_val.npy", np.zeros((0, 2, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="scene_missing"):
        load_roma_bundle(tmp_path, "nr3d", "val")


def test_scene_unique_sampler_emits_every_caption_once_without_batch_collisions():
    sampler = SceneUniqueBatchSampler(
        scene_indices=[0, 0, 0, 1, 1, 2, 3, 3],
        batch_size=3,
        shuffle=False,
    )

    batches = list(sampler)
    flattened = [index for batch in batches for index in batch]

    assert sorted(flattened) == list(range(8))
    for batch in batches:
        scenes = [sampler.scene_indices[index] for index in batch]
        assert len(scenes) == len(set(scenes))


def test_scene_unique_sampler_rejects_batch_size_larger_than_scene_universe():
    with pytest.raises(ValueError, match="unique scenes"):
        SceneUniqueBatchSampler([0, 0, 1], batch_size=3)


def test_scene_unique_sampler_keeps_imbalanced_tail_as_partial_batches():
    sampler = SceneUniqueBatchSampler(
        scene_indices=[0, 0, 0, 0, 1, 2],
        batch_size=3,
        shuffle=False,
        drop_last=False,
    )

    batches = list(sampler)

    assert sorted(index for batch in batches for index in batch) == list(range(6))
    assert [len(batch) for batch in batches] == [3, 1, 1, 1]
