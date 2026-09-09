"""v5 physical world is an independent v4 copy, not the stale v3 test target."""
import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CLONE = ROOT / "src/yard_rl/v5"
MANIFEST = json.loads(Path(__file__).with_name("v4-copy-manifest.json").read_text(encoding="utf-8"))


def test_all_v4_source_files_were_copied():
    missing = [rel for rel in MANIFEST["source_files"] if not (CLONE / rel).exists()]
    assert not missing


@pytest.mark.parametrize("rel", sorted(MANIFEST["world_sha256"]))
def test_physical_world_is_unchanged_from_v4(rel):
    # Compare with the ORIGINAL copy commit, never the moving v4 working tree.
    text = (CLONE / rel).read_text(encoding="utf-8").rstrip() + "\n"
    assert hashlib.sha256(text.encode()).hexdigest() == MANIFEST["world_sha256"][rel]


def test_copy_manifest_pins_the_declared_base_commit():
    from yard_rl.v5 import V4_BASE_COMMIT
    assert MANIFEST["base_commit"] == V4_BASE_COMMIT
    assert len(MANIFEST["source_files"]) == len(set(MANIFEST["source_files"]))
    actual = {p.relative_to(CLONE).as_posix() for p in (CLONE / "world").rglob("*.py")}
    assert actual == set(MANIFEST["world_sha256"])


def test_v5_has_no_absolute_import_from_another_generation():
    bad = []
    for path in CLONE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = ([n.name for n in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level
                     else [])
            bad += [(str(path), n) for n in names
                    if n.startswith("yard_rl.") and not n.startswith("yard_rl.v5.")
                    and n != "yard_rl.v5"]
    assert not bad
