"""v5 physical world is an independent v4 copy, not the stale v3 test target."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/yard_rl/v4"
CLONE = ROOT / "src/yard_rl/v5"


def test_all_v4_source_files_were_copied():
    missing = [p.relative_to(SOURCE).as_posix() for p in SOURCE.rglob("*.py")
               if not (CLONE / p.relative_to(SOURCE)).exists()]
    assert not missing


@pytest.mark.parametrize("rel", [p.relative_to(SOURCE).as_posix()
                               for p in (SOURCE / "world").rglob("*.py")])
def test_physical_world_is_unchanged_from_v4(rel):
    expected = ((SOURCE / rel).read_text(encoding="utf-8")
                .replace("yard_rl.v4", "yard_rl.v5").replace("tests/v4", "tests/v5"))
    assert (CLONE / rel).read_text(encoding="utf-8") == expected


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
