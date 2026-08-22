"""v3 세계 사본이 원본과 어긋나지 않았는가.

사용자 지시 2026-08-22 로 v3 는 엔진까지 자기 사본을 갖는다(`v3/world/`).
사본의 대가는 **조용한 표류**다 — 원본이든 사본이든 한쪽만 고쳐지면 아무도 모르고,
그때부터 "v2 와 v3 가 같은 무대에서 돌았다" 는 전제가 거짓이 된다.
세대 간 짝비교가 통째로 무효가 되므로 여기서 기계로 막는다.

v3 전용 확장 세 파일만 다를 수 있고, 그 파일도 **원본을 접두로 포함**해야 한다
(덧붙이기만 했다는 뜻). 나머지 51 파일은 바이트 동일이다.
"""
from __future__ import annotations

import hashlib
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "yard_rl"
WORLD = SRC / "v3" / "world"

#: v3 가 무대에 덧붙인 파일 — `v3/world/__init__.py` 의 표와 같아야 한다.
EXTENDED = {
    "integrated/yard_layout.py",        # 안벽 축 (02-무대 §1)
    "integrated/vessel.py",             # 선급 3종 (02b-본선)
    "integrated/terminal_stream.py",    # 부하 3수준 · 리드 분포 (02-무대 §2-1·§4)
}


def _rel(p: pathlib.Path) -> str:
    return p.relative_to(WORLD).as_posix()


def _pairs():
    out = []
    for c in sorted(WORLD.rglob("*.py")):
        rel = _rel(c)
        if rel == "__init__.py":
            continue                     # 사본임을 설명하는 문서 — 원본이 없다
        out.append((rel, SRC / rel, c))
    return out


def test_every_clone_has_an_origin():
    """사본에 원본 없는 파일이 있으면 안 된다 — 몰래 새로 만든 것이다."""
    missing = [rel for rel, orig, _ in _pairs() if not orig.exists()]
    assert not missing, f"원본 없는 사본: {missing}"


def test_unextended_files_are_byte_identical():
    """확장 셋을 뺀 전부는 **바이트 동일**하다."""
    drift = []
    for rel, orig, clone in _pairs():
        if rel in EXTENDED:
            continue
        a = hashlib.sha256(orig.read_bytes()).hexdigest()
        b = hashlib.sha256(clone.read_bytes()).hexdigest()
        if a != b:
            drift.append(rel)
    assert not drift, (
        f"사본이 원본과 어긋났다 — {drift}. 어느 쪽을 고칠지 판정하고, v3 가 일부러 "
        f"바꾼 것이면 EXTENDED 와 v3/world/__init__.py 표에 등록하라.")


@pytest.mark.parametrize("rel", sorted(EXTENDED))
def test_extended_files_only_append(rel):
    """확장 파일도 **덧붙이기만** 했는가 — 원본 줄을 고치면 v2 동작이 갈린다."""
    orig = (SRC / rel).read_text(encoding="utf-8")
    clone = (WORLD / rel).read_text(encoding="utf-8")
    assert len(clone) > len(orig), f"{rel}: 확장 파일인데 원본보다 짧다"
    # 원본의 모든 줄이 사본에 **순서대로** 남아 있어야 한다.
    it = iter(clone.splitlines())
    for ln in orig.splitlines():
        for got in it:
            if got == ln:
                break
        else:
            pytest.fail(f"{rel}: 원본 줄이 사라졌다 — {ln[:60]!r}")


def test_v3_imports_nothing_outside_v3():
    """v3(사본 제외)가 v3 밖 모듈을 하나도 안 끌어오는가 — 지시 2026-08-22."""
    import sys
    before = set(sys.modules)
    import yard_rl.v3.eval  # noqa: F401
    import yard_rl.v3.stage  # noqa: F401
    import yard_rl.v3.train  # noqa: F401
    ext = sorted(m for m in set(sys.modules) - before
                 if m.startswith("yard_rl.") and not m.startswith("yard_rl.v3"))
    assert not ext, f"v3 가 밖을 임포트한다: {ext}"
