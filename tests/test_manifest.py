from pathlib import Path

from lib.manifest import build_run_manifest


def test_run_manifest_hashes_inputs_and_records_commit(tmp_path):
    data = tmp_path / "data.jsonl"
    features = tmp_path / "features.npy"
    data.write_text("row\n", encoding="utf-8")
    features.write_bytes(b"features")

    manifest = build_run_manifest([data, features], repo_root=Path.cwd())

    assert manifest["git_commit"]
    assert manifest["inputs"][str(data.resolve())]["sha256"]
    assert manifest["inputs"][str(features.resolve())]["bytes"] == 8
