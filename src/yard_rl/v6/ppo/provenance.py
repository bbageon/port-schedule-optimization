"""Pin a continuous run to a clean source tree, with portable text hashes."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import platform
import subprocess

import numpy as np
import torch


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_stamp():
    repo = Path(__file__).resolve().parents[4]

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    top = Path(git("rev-parse", "--show-toplevel")).resolve()
    if top != repo:
        raise RuntimeError("Git provenance points to a different worktree")
    dirty = git("status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise RuntimeError("Continuous training requires a clean, isolated worktree")
    digest = hashlib.sha256()
    source_root = Path(__file__).resolve().parents[1]
    for path in sorted(source_root.rglob("*.py")):
        digest.update(path.relative_to(source_root).as_posix().encode())
        digest.update(path.read_text(encoding="utf-8").encode())
    return {"git_head": git("rev-parse", "HEAD"), "git_dirty": False,
            "source_sha256": digest.hexdigest(), "worktree": str(repo),
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "pid": os.getpid(), "torch_threads": 1}
