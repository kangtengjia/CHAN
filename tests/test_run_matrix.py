from lib.roma_runs import RUN_PRESETS, format_markdown_metrics


def test_run_matrix_contains_four_datasets_and_two_text_encoders():
    assert set(RUN_PRESETS) == {
        ("scenedepict", "bigru"),
        ("scenedepict", "bert"),
        ("scanrefer", "bigru"),
        ("scanrefer", "bert"),
        ("nr3d", "bigru"),
        ("nr3d", "bert"),
        ("3dllm", "bigru"),
        ("3dllm", "bert"),
    }
    assert RUN_PRESETS[("nr3d", "bigru")].batch_size == 128
    assert RUN_PRESETS[("3dllm", "bigru")].learning_rate == 3e-5
    assert RUN_PRESETS[("3dllm", "bert")].epochs == 30


def test_markdown_metrics_reports_unified_four_recalls():
    table = format_markdown_metrics(
        "scanrefer",
        "bigru",
        {"R@1": 1.0, "R@5": 2.0, "R@10": 3.0, "R@30": 4.0, "Rsum": 10.0, "MedR": 20.0, "MeanR": 30.0, "MRR": 40.0},
    )

    assert "R@1" in table
    assert "R@30" in table
    assert "10.00" in table
    assert "MRR" in table
    assert "40.00" in table
