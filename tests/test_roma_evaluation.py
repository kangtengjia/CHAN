import numpy as np

from lib.roma_evaluation import unique_scene_rows, text_to_scene_metrics


def test_text_to_scene_metrics_support_variable_captions_per_scene():
    similarities = np.asarray(
        [
            [0.9, 0.1, 0.0],
            [0.7, 0.8, 0.1],
            [0.1, 0.9, 0.2],
            [0.3, 0.4, 0.8],
        ],
        dtype=np.float32,
    )
    caption_scene_indices = [0, 0, 1, 2]

    metrics = text_to_scene_metrics(similarities, caption_scene_indices, ks=(1, 2, 3))

    assert metrics["queries"] == 4
    assert metrics["R@1"] == 75.0
    assert metrics["R@2"] == 100.0
    assert metrics["R@3"] == 100.0
    assert metrics["Rsum"] == 275.0
    assert metrics["MedR"] == 1.0
    assert metrics["MRR"] == 87.5


def test_text_to_scene_metrics_rejects_shape_mismatch():
    similarities = np.zeros((2, 3), dtype=np.float32)

    try:
        text_to_scene_metrics(similarities, [0])
    except ValueError as error:
        assert "caption count" in str(error)
    else:
        raise AssertionError("expected a caption-count validation error")


def test_unique_scene_rows_selects_one_encoded_copy_per_feature_scene():
    encoded = np.asarray([[10], [11], [20], [12]], dtype=np.float32)
    image_lengths = [5, 5, 6, 5]
    scene_indices = [0, 0, 1, 0]

    scene_encoded, scene_lengths = unique_scene_rows(encoded, image_lengths, scene_indices, scene_count=2)

    assert scene_encoded.tolist() == [[10.0], [20.0]]
    assert scene_lengths == [5, 6]
