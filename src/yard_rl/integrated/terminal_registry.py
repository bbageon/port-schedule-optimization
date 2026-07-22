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
    """구조군 선택 결과 — `.profile` 은 build_calibrated_profile() 자리에 그대로 꽂힌다.

    faithful=True (수평 PROVISIONAL): 현 엔진이 구조를 충실히 실행 (SHARED 2크레인·YT).
    faithful=False (수직·혼합): 공용 assumed substrate 위의 **이름표 stress** — 역할분리·
    S/C·AGV 역학 미모형. 실행은 되지만 성능·구조 주장 금지(warnings 참조·YR-083 후 충실화).
    """

    terminal_id: str
    archetype: str
    profile: IntegratedProfile
    faithful: bool                  # 현 엔진이 이 구조를 충실히 실행하는가
    data_grade: str                 # 예: "Level1-STRESS" / "Level0-NAMEPLATE(구조미충실)"
    orientation_status: str         # confirmed | unresolved
    physics_overlays: dict          # 실제 물리에 반영한 확인/유도 오버레이
    unresolved_fields: tuple[str, ...]
    warnings: tuple[str, ...]       # faithful=False 면 미충실 사유·주장 금지 경고
    claim_gate: str


def load_manifest(path: str | Path = MANIFEST) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def list_terminals(path: str | Path = MANIFEST) -> list[dict]:
    """선택 표면 — 터미널별 (id·이름·구조군·Level·선택가능·충실도) 한 줄 요약.

    selectable 은 항상 True (10개 전부 stress 로 선택·실행 가능). faithful 은 현 엔진이
    구조를 충실히 실행하는지 — False 면 이름표 stress(성능·구조 주장 금지).
    """
    m = load_manifest(path)
    arch = m["archetypes"]
    lv = m["data_sufficiency"]["levels"]
    out = []
    for tid, e in m["terminals"].items():
        a = e["archetype"]
        faithful = bool(arch[a]["runnable_in_current_engine"])
        out.append({"id": tid, "name_kr": e["name_kr"], "archetype": a,
                    "level": lv.get(tid), "selectable": True, "faithful": faithful,
                    "runnable_in_current_engine": faithful,   # 하위호환
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


_KIND = {"YT": "YT", "AGV": "AGV", "SC": "SC"}


def build_stress_profile(terminal_id: str, path: str | Path = MANIFEST, *,
                         require_faithful: bool = False) -> StressEnv:
    """터미널 선택 → 실행 가능한 stress IntegratedProfile (**10개 전부 선택 가능**).

    - 수평 PROVISIONAL: faithful=True — 현 엔진이 SHARED 2크레인·YT 구조를 충실히 실행.
    - 수직·혼합: faithful=False — 공용 assumed substrate(fallback_physics_base) 위의
      **이름표 stress**. 실행은 되지만 역할분리·S/C·AGV 역학 미모형이라 warnings 동반,
      성능·구조 주장 금지. 충실 실행은 YR-083 후.

    require_faithful=True 면 faithful=False 터미널에 StructureBlockedError (claim-bearing
    실험 코드 보호용). 물리 오버레이는 공개 확인값(레일간격→열폭)만 반영.
    """
    m = load_manifest(path)
    if terminal_id not in m["terminals"]:
        raise KeyError(f"미등록 터미널: {terminal_id!r} (등록: {sorted(m['terminals'])})")
    entry = m["terminals"][terminal_id]
    arch_name = entry["archetype"]
    arch = m["archetypes"][arch_name]
    faithful = bool(arch["runnable_in_current_engine"])

    if require_faithful and not faithful:
        raise StructureBlockedError(
            f"{terminal_id}({arch_name})는 현 엔진에서 구조 미충실 — {arch['desc']}. "
            f"require_faithful=True 라 차단. 이름표 stress 로 쓰려면 require_faithful=False. "
            f"충실 실행은 YR-083(도로·인계점·크레인 역할 런타임화) 뒤.")

    # physics_base: 구조군 것 우선, 없으면(수직 S/C·북항) 공용 fallback substrate.
    physics_base = arch.get("physics_base") or m["fallback_physics_base"]
    base = build_dgt_approx_profile(physics_base)

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

    # 이송차종 라벨(nameplate) — 확인값이 있으면 반영. 역학(n_units·move_time)은 base assumed.
    tk = entry.get("transfer", {}).get("kind", {})
    kind = _KIND.get(tk.get("v")) if tk.get("s") == "confirmed" else None
    if kind:
        base = replace(base, transfer=replace(base.transfer, kind=kind))

    level = m["data_sufficiency"]["levels"].get(terminal_id)
    profile = replace(base, terminal_id=f"{terminal_id}-STRESS")
    warnings: list[str] = []
    if not faithful:
        warnings.append(f"구조 미충실 — {arch['desc']}: 역할분리/{kind or '이송'} 역학이 현 엔진에 "
                        "없어 공용 assumed 2크레인 substrate 로 근사(이름표 stress).")
        warnings.append("성능·구조 주장 금지 — 이 실행을 '해당 터미널 성능'으로 쓰지 말 것. "
                        "충실 실행은 YR-083 후.")
    grade = f"Level{level}-STRESS" if faithful else f"Level{level}-NAMEPLATE(구조미충실)"
    return StressEnv(
        terminal_id=terminal_id, archetype=arch_name, profile=profile, faithful=faithful,
        data_grade=grade,
        orientation_status=layout.get("orientation", {}).get("s", "unresolved"),
        physics_overlays=overlays,
        unresolved_fields=tuple(_walk_unresolved(entry)),
        warnings=tuple(warnings), claim_gate=m["claim_gate"].strip())


def faithful_terminals(path: str | Path = MANIFEST) -> list[str]:
    """현 엔진이 구조를 **충실히** 실행하는 터미널 (수평 PROVISIONAL 군)."""
    return [t["id"] for t in list_terminals(path) if t["runnable_in_current_engine"]]


# 하위호환 별칭 — 이전엔 "실행 가능"이 곧 "충실"이었다. 지금은 10개 다 실행 가능하고
# 이 함수는 그중 '충실' 집합을 뜻한다.
runnable_terminals = faithful_terminals
