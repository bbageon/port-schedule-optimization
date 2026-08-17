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
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any



# ------------------------------------------------------------------ 결과 저장
SIDECAR_SUFFIX = ".sha256"


def write_result(path: str | Path, payload: Any, *, indent: int = 1) -> str:
    """결과 JSON 을 쓰고 **sidecar 에 해시를 남긴다** (YR-155).

    ■ 왜 파일 안에 자기 해시를 적으면 안 되나
      적는 순간 내용이 바뀌므로 기록값은 **덧쓰기 전 파일의 해시**가 된다.
      검증하는 쪽이 쓸 수 없는 값이다 (2026-08-06 YR-151 0A 실측:
      기록 `4862ae71…` vs 실제 `c287a9d5…`). 원리상 자기검증은 불가능하다.

    ■ 규약
      `<result>.json` 을 먼저 확정해 쓰고, 그 파일의 해시를
      `<result>.json.sha256` 에 별도로 남긴다. 검증은
      `sha256(<result>.json) == <result>.json.sha256` 로 성립한다.

    반환값은 기록한 sha256 이다.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=indent,
                            default=str), encoding="utf-8")
    digest = sha256_file(p)
    p.with_name(p.name + SIDECAR_SUFFIX).write_text(digest + "\n",
                                                    encoding="utf-8")
    return digest


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_result(path: str | Path) -> bool | None:
    """sidecar 로 결과 파일을 검증한다. sidecar 가 없으면 None (구 산출물)."""
    p = Path(path)
    side = p.with_name(p.name + SIDECAR_SUFFIX)
    if not p.is_file() or not side.is_file():
        return None
    return side.read_text(encoding="utf-8").strip() == sha256_file(p)


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


def code_dirty(paths: tuple[str, ...] = ("src", "tests")) -> bool | None:
    """**실행 코드**가 커밋에 다 들어있는지 — `git_dirty()` 로는 잡히지 않는 구멍을 막는다.

    `git_dirty()` 는 `--untracked-files=no` 라서 **새로 만든 파일을 깨끗하다고 본다**.
    새 실험은 대부분 새 파일이므로, 구현을 커밋하지 않고 판정을 돌려도
    `git_dirty=false` 가 찍혀 재현 사슬이 조용히 끊긴다(2026-08-06 YR-150 1단계 실측).
    여기서는 `src`·`tests` 아래의 **수정분과 미추적 신규 파일을 모두** 본다
    (판정과 무관한 `outputs/` 산출물 잡음은 범위에서 제외).
    """
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--", *paths],
                             capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            return None
        return bool(out.stdout.strip())
    except Exception:
        return None


def git_dirty() -> bool | None:
    """추적 파일에 커밋 안 된 변경이 있는지 — 판정런이 '커밋된 코드'인지 구분한다.

    **주의**: 미추적 신규 파일은 잡지 못한다. 실행 코드 완전성은 `code_dirty()` 를 쓴다.
    """
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
