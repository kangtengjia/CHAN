from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Iterable


def _sha256(path: Path, *, sample_large_files: bool = True) -> tuple[str, str]:
    digest = hashlib.sha256()
    size = path.stat().st_size
    with path.open("rb") as handle:
        if sample_large_files and size > 64 * 1024 * 1024:
            digest.update(handle.read(1024 * 1024))
            handle.seek(max(0, size // 2 - 512 * 1024))
            digest.update(handle.read(1024 * 1024))
            handle.seek(max(0, size - 1024 * 1024))
            digest.update(handle.read(1024 * 1024))
            return digest.hexdigest(), "sampled"
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), "full"


def build_run_manifest(paths: Iterable[str | Path], *, repo_root: str | Path) -> dict:
    root = Path(repo_root)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    inputs = {}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        digest, mode = _sha256(path)
        inputs[str(path)] = {"sha256": digest, "hash_mode": mode, "bytes": path.stat().st_size}
    return {"git_commit": commit, "git_dirty": dirty, "inputs": inputs}
