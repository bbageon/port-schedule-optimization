"""v7 is an independent v6 copy (YR-332-a): the physical world is byte-identical to v6.

v7 changes the stage (arm, lead time, crane action range) and the cost table, never
`world/`. The manifest pins the v6 world at the declared base commit, so a later edit
to either the moving v6 tree or the v7 copy cannot silently pass.
"""
import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CLONE = ROOT / "src/yard_rl/v7"
MANIFEST = json.loads(Path(__file__).with_name("v6-copy-manifest.json").read_text(encoding="utf-8"))


def test_all_v6_source_files_were_copied():
    missing = [rel for rel in MANIFEST["source_files"] if not (CLONE / rel).exists()]
    assert not missing


@pytest.mark.parametrize("rel", sorted(MANIFEST["world_sha256"]))
def test_physical_world_is_unchanged_from_v6(rel):
    text = (CLONE / rel).read_text(encoding="utf-8").replace("\r\n", "\n").rstrip() + "\n"
    assert hashlib.sha256(text.encode()).hexdigest() == MANIFEST["world_sha256"][rel]


def test_copy_manifest_pins_the_declared_base_commit():
    from yard_rl.v7 import V6_BASE_COMMIT
    assert MANIFEST["base_commit"] == V6_BASE_COMMIT
    assert len(MANIFEST["source_files"]) == len(set(MANIFEST["source_files"]))
    actual = {p.relative_to(CLONE).as_posix() for p in (CLONE / "world").rglob("*.py")}
    assert actual == set(MANIFEST["world_sha256"])


def test_v7_has_no_absolute_import_from_another_generation():
    bad = []
    for path in CLONE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = ([n.name for n in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level
                     else [])
            bad += [(str(path), n) for n in names
                    if n.startswith("yard_rl.") and not n.startswith("yard_rl.v7.")
                    and n != "yard_rl.v7"]
    assert not bad
