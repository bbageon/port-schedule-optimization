"""재현 스탬프 — 결과 JSON 하나만 보면 런을 되살릴 수 있게 (YR-106-b 게이트 B).

**왜**: 판정 결과 JSON 이 요약만 남기고 재현 계약(절대 시드·유효 파라미터·코드 버전·
프로파일)을 남기지 않아 왔다. 실제 사고 2건 —
  · YR-099-mid 확증런의 대역 842k 는 report.md 산문에만 있고 JSON·코드 어디에도 없어 재현 불가.
  · YR-109 원자료 `legacy_vs_achievable.json` 은 **생성 스크립트가 저장소에 없다**
    (evidence 로 박제된 표를 재생산할 수 없었다).

**설계 원칙**: 시나리오 `meta` 에는 손대지 않는다 — 저장소에 "비기본 파라미터만 meta 에
스탬프" 라는 계약이 있고 테스트 4건이 키 부재를 검사한다. 재현 정보는 **결과 JSON** 에 둔다.
"""
from __future__ import annotations

import dataclasses
import subprocess
import sys
from typing import Any


def git_head(short: bool = False) -> str | None:
    """현재 커밋 해시 — 저장소 밖이거나 git 이 없으면 None (예외 없음)."""
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD"]
        out = subprocess.run(args, capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except Exception:
        return None


def git_dirty() -> bool | None:
    """추적 파일에 커밋 안 된 변경이 있는지 — 판정런이 '커밋된 코드'인지 구분한다."""
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                             capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            return None
        return bool(out.stdout.strip())
    except Exception:
        return None


def params_dict(params: Any) -> dict:
    """dataclass 파라미터 → 전 필드 dict (기본값 포함 — 재현에는 전문이 필요하다)."""
    if dataclasses.is_dataclass(params) and not isinstance(params, type):
        return {k: (v if isinstance(v, (int, float, str, bool, type(None))) else repr(v))
                for k, v in dataclasses.asdict(params).items()}
    return {"repr": repr(params)}


def repro_stamp(*, experiment: str, seeds: dict[str, list[int]],
                params: dict[str, Any] | None = None,
                profile_id: str | None = None,
                prereg: str | None = None,
                extra: dict | None = None) -> dict:
    """결과 JSON 최상단에 넣을 재현 블록.

    seeds: arm/블록별 **절대 시드 목록** (base+i 가 아니라 실제 값 — 대역 상수가 코드에서
      바뀌어도 과거 런을 되살릴 수 있어야 한다).
    params: 이름 → 파라미터 dataclass. 전 필드가 펼쳐진다.
    """
    return {
        "experiment": experiment,
        "code": {"git_head": git_head(), "git_dirty": git_dirty(),
                 "python": sys.version.split()[0]},
        "profile_id": profile_id,
        "seeds": {k: list(v) for k, v in seeds.items()},
        "params": {k: params_dict(v) for k, v in (params or {}).items()},
        "prereg": prereg,
        **(extra or {}),
    }


def vessel_physics_rows(scenario) -> list[dict]:
    """본선별 마감 물리 상태 — 유효 계획완료·물리 하한·구조적 최소초과를 시드별로 박제.

    (요청 배율 `vessel_deadline_mult` 만 남기면 클램프가 걸렸는지 알 수 없다.)
    """
    out = []
    for v in scenario.vessels:
        pl = v.plan
        out.append({
            "vessel_id": v.vessel_id, "work": v.work_type.value,
            "moves": pl.total_moves, "cadence_s": round(pl.sts_move_interval_s, 3),
            "planned_start_s": round(pl.planned_start_s, 3),
            "planned_completion_s": (None if pl.planned_completion_s is None
                                     else round(pl.planned_completion_s, 3)),
            "phys_min_completion_s": (None if pl.phys_min_completion_s is None
                                      else round(pl.phys_min_completion_s, 3)),
            "structural_min_overrun_s": round(v.structural_min_overrun_s(), 3),
            # 유효 배율 = 계획완료가 STS 물리 최소완료의 몇 배인지 (요청값과 다를 수 있다)
            "effective_deadline_mult": (
                None if pl.planned_completion_s is None or pl.total_moves == 0 else
                round((pl.planned_completion_s - pl.planned_start_s)
                      / (pl.total_moves * pl.sts_move_interval_s), 4)),
        })
    return out
