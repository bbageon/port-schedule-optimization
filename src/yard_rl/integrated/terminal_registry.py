"""부산항 10개 터미널 공개자료 레지스트리 + 구조군 선택기 (YR-082-A).

두 층으로 환경을 모듈화한다.
  ① nameplate 층 : `configs/terminals/busan/manifest.yaml` — 필드별 증거등급
     (confirmed/derived/assumed/unresolved). 미확인은 null 박제, 0·평균 대입 금지.
  ② archetype 층 : spec 4개 구조군. `build_stress_profile()` 는 **현 2크레인 공동경합
     엔진에서 실행 가능한 구조군(수평 PROVISIONAL)만** IntegratedProfile 로 빌드하고,
     수직형(ARMG·AGV / S/C)·북항 혼합형은 `StructureBlockedError` 로 거부한다
     (이름만 바꿔 넣는 시험 금지 — spec. 실제 소비는 YR-083 런타임화 뒤).

주의(주장 게이트): 여기서 나오는 프로파일은 전부 **Level 1 stress** 다. 문헌 보정 assumed
physics 위에 공개 확인값을 극소 오버레이할 뿐 — **터미널별 실운영 성능 주장 금지**.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from ..io.profile_loader import load_profile
from .profile import IntegratedProfile
from .profiles import build_dgt_approx_profile

MANIFEST = Path("configs/terminals/busan/manifest.yaml")


class StructureBlockedError(RuntimeError):
    """현 엔진에서 실행 금지된 구조군을 빌드하려 할 때 (수직·혼합형)."""


@dataclass(frozen=True)
class StressEnv:
    """구조군 선택 결과 — `.profile` 은 build_calibrated_profile() 자리에 그대로 꽂힌다."""

    terminal_id: str
    archetype: str
    profile: IntegratedProfile
    data_grade: str                 # 예: "Level1-STRESS"
    orientation_status: str         # confirmed | unresolved
    physics_overlays: dict          # 실제 물리에 반영한 확인/유도 오버레이
    unresolved_fields: tuple[str, ...]
    claim_gate: str


def load_manifest(path: str | Path = MANIFEST) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def list_terminals(path: str | Path = MANIFEST) -> list[dict]:
    """선택 표면 — 터미널별 (id·이름·구조군·Level·현엔진 실행가능) 한 줄 요약."""
    m = load_manifest(path)
    arch = m["archetypes"]
    lv = m["data_sufficiency"]["levels"]
    out = []
    for tid, e in m["terminals"].items():
        a = e["archetype"]
        out.append({"id": tid, "name_kr": e["name_kr"], "archetype": a,
                    "level": lv.get(tid),
                    "runnable_in_current_engine": arch[a]["runnable_in_current_engine"],
                    "role_separated": arch[a]["role_separated"]})
    return out


def _walk_unresolved(node, prefix: str = "") -> list[str]:
    """entry 를 훑어 status=='unresolved' 인 필드 경로를 모은다."""
    found: list[str] = []
    if isinstance(node, dict):
        if node.get("s") == "unresolved":
            found.append(prefix.rstrip("."))
            return found
        for k, v in node.items():
            found += _walk_unresolved(v, f"{prefix}{k}.")
    return found


def build_stress_profile(terminal_id: str, path: str | Path = MANIFEST) -> StressEnv:
    """터미널 선택 → 구조군 게이트 → (수평형만) Level1 stress IntegratedProfile.

    수직형·혼합형은 StructureBlockedError. 물리 오버레이는 공개 확인값이 있는 경우만
    (현실적으로 레일간격→열폭 정도) 반영하고 나머지는 archetype physics_base 의 assumed 유지.
    """
    m = load_manifest(path)
    if terminal_id not in m["terminals"]:
        raise KeyError(f"미등록 터미널: {terminal_id!r} (등록: {sorted(m['terminals'])})")
    entry = m["terminals"][terminal_id]
    arch_name = entry["archetype"]
    arch = m["archetypes"][arch_name]

    if not arch["runnable_in_current_engine"]:
        raise StructureBlockedError(
            f"{terminal_id}({arch_name})는 현 2크레인 공동경합 엔진에서 실행 금지 — "
            f"{arch['desc']}. 역할분리·AGV/SC 흐름은 미구현이라 이름만 바꿔 넣는 시험은 "
            f"spec 이 금지한다. 실제 소비는 YR-083(도로·인계점·크레인 역할 런타임화) 뒤.")

    # 수평 PROVISIONAL: physics_base(문헌 보정 assumed) 위 확인값 극소 오버레이.
    base = build_dgt_approx_profile(arch["physics_base"])
    overlays: dict = {}
    layout = entry.get("layout", {})
    rail = layout.get("rail_gap_m", {})
    rows_f = layout.get("rows", {})
    if rail.get("s") == "confirmed" and rail.get("v"):
        rows_v = rows_f["v"] if rows_f.get("s") == "confirmed" else base.block.row_count
        row_width = rail["v"] / rows_v
        status = "confirmed" if rows_f.get("s") == "confirmed" else "derived"
        base = replace(base, block=replace(base.block, row_width_m=round(row_width, 3)))
        overlays["row_width_m"] = {"value": round(row_width, 3), "status": status,
                                   "from": f"rail_gap {rail['v']}m / rows {rows_v}"}

    level = m["data_sufficiency"]["levels"].get(terminal_id)
    profile = replace(base, terminal_id=f"{terminal_id}-STRESS")
    return StressEnv(
        terminal_id=terminal_id, archetype=arch_name, profile=profile,
        data_grade=f"Level{level}-STRESS",
        orientation_status=layout.get("orientation", {}).get("s", "unresolved"),
        physics_overlays=overlays,
        unresolved_fields=tuple(_walk_unresolved(entry)),
        claim_gate=m["claim_gate"].strip())


def runnable_terminals(path: str | Path = MANIFEST) -> list[str]:
    """현 엔진에서 stress 빌드 가능한 터미널 (수평 PROVISIONAL 군)."""
    return [t["id"] for t in list_terminals(path) if t["runnable_in_current_engine"]]
