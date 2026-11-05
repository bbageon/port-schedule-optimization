"""v7 retains its pinned v6 base except explicitly registered physical corrections.

The original v6 hashes remain pinned; overrides record the authorized discharge
correction (YR-332-h) and computation-preserving dispatch optimization (YR-332-j).
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
    overrides = MANIFEST.get("v7_world_overrides", {})
    expected = overrides.get(rel, {}).get("sha256", MANIFEST["world_sha256"][rel])
    assert hashlib.sha256(text.encode()).hexdigest() == expected


def test_physical_corrections_are_explicit_and_scoped():
    corrections = MANIFEST["v7_world_overrides"]
    assert set(corrections) == {"world/integrated/engine.py", "world/domain/models.py",
                                "world/integrated/candidates.py", "world/integrated/resolver.py"}
    assert all(c['task'] == 'YR-332-j' for c in corrections.values())
    assert corrections['world/integrated/engine.py']['prior_task'] == 'YR-332-h'


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
