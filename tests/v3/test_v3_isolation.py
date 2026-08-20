"""세대 격리를 기계로 강제한다 (사용자 지시 2026-08-20).

  "v1 은 v1 만, v2 는 v2 만, v3 는 v3 만 있어야 한다"

규칙을 사람이 지키기로 하면 반드시 샌다. 여기서 import 그래프로 막는다.

■ 무엇을 막나
  세대끼리의 **모든** 참조 — v3→v2, v2→v3, v1→v2 … 전부. 한 세대가 다른
  세대의 코드를 쓰기 시작하면 "v3 를 고쳤는데 v2 판정이 달라지는" 일이 생기고,
  그 순간 세대 비교가 끊긴다. 식이 같아도 **사본을 따로 둔다**(의도된 중복).

  `shared/` 같은 중간 계층도 막는다 — 만들면 결국 거기로 다 모인다.

■ 무엇을 허용하나
  세대 → `integrated` (무대: 엔진·비용·레이아웃·배정기)

  무대는 복제하지 않는다. 판정이 **짝비교**라 세 세대가 같은 무대를 받아야
  비교가 성립하고, 복제하면 v2 와 v3 를 견줄 근거가 사라진다.
"""
from __future__ import annotations

import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "yard_rl"
V3 = SRC / "v3"

#: 세대 패키지 — 서로를 참조하면 안 된다
GENERATIONS = ("v1", "v2", "v3")

#: 세대 사이에 끼워서는 안 되는 중간 계층 (만들면 결국 거기로 다 모인다)
BANNED_LAYERS = ("shared", "common", "core")


def _imports(path: pathlib.Path) -> list[str]:
    """그 파일이 참조하는 `yard_rl` 바로 아래 패키지 이름들.

    상대 import 는 **파일 위치를 기준으로 풀어야** 한다. 점 개수만 보고 넘겨짚으면
    `shared/features.py` 의 `from ..integrated import` 를 엉뚱한 패키지로 읽는다.
    """
    text = path.read_text(encoding="utf-8")
    pkg = list(path.relative_to(SRC).parts[:-1])       # 예: ["v3", "reward"]
    hits = []

    def top_of(parts: list[str]) -> str | None:
        return parts[0] if parts else None

    for m in re.finditer(r"^\s*from\s+(\.+|yard_rl\.)([\w.]*)\s+import", text, re.M):
        lead, mod = m.group(1), m.group(2)
        tail = [p for p in mod.split(".") if p]
        if lead.startswith("yard_rl"):
            hit = top_of(tail)
        else:
            up = len(lead) - 1                          # 점 1개 = 같은 패키지
            base = pkg[:len(pkg) - up] if up <= len(pkg) else []
            hit = top_of(base + tail)
        if hit:
            hits.append(hit)
    for m in re.finditer(r"^\s*import\s+yard_rl\.([\w.]+)", text, re.M):
        hits.append(m.group(1).split(".")[0])
    return hits


def test_generations_never_reference_each_other():
    """세대끼리 참조하지 않는다 — 어느 방향이든."""
    offenders = []
    for gen in GENERATIONS:
        root = SRC / gen
        if not root.exists():
            continue
        others = [g for g in GENERATIONS if g != gen]
        for f in sorted(root.rglob("*.py")):
            for pkg in _imports(f):
                if pkg in others:
                    offenders.append(f"{f.relative_to(SRC).as_posix()} → yard_rl.{pkg}")
    assert not offenders, (
        "세대가 다른 세대를 참조한다. 식이 같아도 **사본을 자기 폴더에 둔다**:\n  "
        + "\n  ".join(offenders))


def test_no_shared_layer_between_generations():
    """`shared/` 같은 중간 계층을 만들지 않는다 — 만들면 결국 거기로 다 모인다."""
    found = [n for n in BANNED_LAYERS if (SRC / n).exists()]
    assert not found, (
        f"세대 사이 중간 계층이 생겼다: {found}. 각 세대가 자기 사본을 갖는다."
        " 무대(엔진·비용·레이아웃·배정기)는 `integrated/` 에 둔다.")


def test_generations_use_the_shared_stage():
    """무대는 공유가 정상이다 — 짝비교가 성립하려면 같은 무대여야 한다.

    금지가 아니라 **기대**를 적어둔다. 세대가 무대를 아예 안 쓰면 자기 시뮬레이터를
    따로 들고 있다는 뜻이고, 그러면 세대 비교의 전제가 깨진다.
    """
    for gen in ("v1", "v2"):                      # v3 는 아직 비어 있다
        pkgs = {p for f in (SRC / gen).rglob("*.py") for p in _imports(f)}
        assert "integrated" in pkgs, f"{gen} 이 무대(integrated)를 안 쓴다"


def test_v3_axis_packages_exist():
    """네 축 + 진입점 폴더가 다 있고 설명이 붙어 있다."""
    for name in ("schema", "reward", "actors", "features", "train", "eval"):
        init = V3 / name / "__init__.py"
        assert init.exists(), f"v3/{name}/__init__.py 가 없다"
        doc = init.read_text(encoding="utf-8")
        assert doc.lstrip().startswith('"""'), f"v3/{name} 에 설명이 없다"


def test_v3_horizon_matches_architecture_contract():
    """반사실 지평은 설계 문서의 계약값(3600초)과 같아야 한다."""
    from yard_rl.v3 import CF_HORIZON_S

    doc = (pathlib.Path(__file__).resolve().parents[2] / ".claude" / "docs"
           / "architecture" / "04b-학습-잣대.md").read_text(encoding="utf-8")
    m = re.search(r"counterfactual_h_s_target\s*=\s*(\d+)", doc)
    assert m, "04b 문서에 counterfactual_h_s_target 계약이 없다"
    assert CF_HORIZON_S == float(m.group(1)), (
        f"v3.CF_HORIZON_S={CF_HORIZON_S} 인데 문서 계약은 {m.group(1)} 이다")
