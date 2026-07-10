from pathlib import Path

import pytest

from lib.roma_config import require_bert_path, vocab_filename


@pytest.mark.parametrize(
    ("dataset", "expected"),
    [
        ("scenedepict", "my_data_vocab.json"),
        ("scanrefer", "my_data_vocab.json"),
        ("nr3d", "nr3d_vocab.json"),
        ("3dllm", "my_data_vocab.json"),
    ],
)
def test_vocab_filename_matches_roma_protocol(dataset, expected):
    assert vocab_filename(dataset) == expected


def test_require_bert_path_checks_required_files(tmp_path):
    with pytest.raises(ValueError, match="BERT_PATH"):
        require_bert_path("")

    path = tmp_path / "bert"
    path.mkdir()
    (path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="vocab.txt"):
        require_bert_path(path)

    (path / "vocab.txt").write_text("[PAD]\n", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")
    assert require_bert_path(path) == Path(path)
