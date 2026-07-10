from pathlib import Path


def test_parameterized_training_script_uses_faithful_chan_settings():
    script = Path("scripts/train_roma.sh").read_text(encoding="utf-8")

    assert "VHACoding" in script
    assert "LSEPooling" in script
    assert "ContrastiveLoss" in script
    assert "--margin 0.05" in script
    assert "RUN_PRESETS" in script


def test_matrix_launcher_covers_both_text_encoders_and_four_datasets():
    script = Path("scripts/run_roma_matrix.sh").read_text(encoding="utf-8")

    assert "scenedepict,scanrefer,nr3d,3dllm" in script
    assert "bigru,bert" in script


def test_preflight_checks_bert_and_every_dataset():
    script = Path("scripts/preflight_roma.sh").read_text(encoding="utf-8")

    assert "require_bert_path" in script
    for dataset in ["scenedepict", "scanrefer", "nr3d", "3dllm"]:
        assert dataset in script
