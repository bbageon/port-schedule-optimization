"""오더 하나가 지나가는 길 — 다섯 단계와 그 불변식.

설계 정본: `.claude/docs/architecture/00-라이프사이클.md`

    코피노 수신 ──► 게이트 진입 ──► 블록 진입 ──► 작업 완료 ──► 게이트 아웃
    copinoNotice    gateIn         blockIn       jobDone       gateOut

■ 불변식
  · `copinoNotice ≤ gateIn ≤ blockIn ≤ jobDone ≤ gateOut` (단조 증가)
  · 건너뛰기 없음 — 뒤 단계가 찍혔는데 앞 단계가 비어 있을 수 없다
  · 되돌아가기 없음 — 각 단계는 오더당 한 번

■ 시계
  시뮬레이션은 시계 `t` 를 하나만 갖는다(초 · t=0 이 하루 시작). 다섯 시각은 전부
  **그 이벤트가 일어난 순간의 `t`** 다. 나중에 되계산하지 않고, 아직 안 온 단계는
  **비어 있다**(None) — 0 이나 추정값으로 채우지 않는다.

  "미발생은 비어 있다" 가 정보 경계의 실체다. 정책이 미래를 못 보는 이유는 우리가
  가려서가 아니라 **그 칸이 아직 비어 있기 때문**이다.

■ serviceStart 는 단계가 아니다
  터미널이 전송하지 않는다(DGT 실데이터에도 없다). blockIn 과 jobDone 사이에 끼는
  **시뮬레이터 내부 관측**이며, 보상의 크레인 점유 계산에만 쓴다.
"""
from __future__ import annotations

from enum import IntEnum

#: 라이프사이클 단계 수 — 계약(`lifecycle_stages_target = 5`)
LIFECYCLE_STAGES = 5


class Stage(IntEnum):
    """오더가 지나는 다섯 단계. 값의 순서가 곧 시간 순서다."""

    COPINO_NOTICE = 0      # ① 코피노 수신 — 오더가 생긴다
    GATE_IN = 1            # ② 게이트 진입
    BLOCK_IN = 2           # ③ 블록 진입
    JOB_DONE = 3           # ④ 작업 완료
    GATE_OUT = 4           # ⑤ 게이트 아웃


#: 단계 → 레코드 필드 이름 (schema.record.ExecutionRecord)
STAGE_FIELD: dict[Stage, str] = {
    Stage.COPINO_NOTICE: "copino_notice_s",
    Stage.GATE_IN: "gate_in_s",
    Stage.BLOCK_IN: "block_in_s",
    Stage.JOB_DONE: "job_done_s",
    Stage.GATE_OUT: "gate_out_s",
}

#: 터미널이 이벤트로 **전송해 주는** 단계 (코피노는 오더 수신이라 별개)
TRANSMITTED = (Stage.GATE_IN, Stage.BLOCK_IN, Stage.JOB_DONE, Stage.GATE_OUT)


class LifecycleError(RuntimeError):
    """불변식 위반 — 실패를 삼키지 않는다(fail-closed)."""


def stage_times(rec) -> list[float | None]:
    """레코드에서 다섯 시각을 단계 순서로 뽑는다."""
    return [getattr(rec, STAGE_FIELD[s]) for s in Stage]


def reached(rec) -> Stage | None:
    """지금까지 도달한 마지막 단계. 아무것도 없으면 None."""
    last: Stage | None = None
    for s in Stage:
        if getattr(rec, STAGE_FIELD[s]) is None:
            break
        last = s
    return last


def validate(rec, *, eps: float = 1e-6) -> None:
    """세 불변식을 검사한다. 위반이면 `LifecycleError`.

    ① 건너뛰기 없음 — 앞이 비었는데 뒤가 찍힌 경우
    ② 단조 증가 — 시각이 뒤로 가는 경우
    (③ 되돌아가기 없음은 기록 API 가 재기록을 막아 구조로 보장한다 → record.py)
    """
    ts = stage_times(rec)
    seen_gap = False
    for s in Stage:
        v = ts[int(s)]
        if v is None:
            seen_gap = True
            continue
        if seen_gap:
            missing = [STAGE_FIELD[x] for x in Stage
                       if ts[int(x)] is None and int(x) < int(s)]
            raise LifecycleError(
                f"{getattr(rec, 'doc_key', '?')}: 단계 건너뜀 — "
                f"{STAGE_FIELD[s]} 가 찍혔는데 {missing} 이 비어 있다")
    prev_name, prev = None, None
    for s in Stage:
        v = ts[int(s)]
        if v is None:
            break
        if prev is not None and v < prev - eps:
            raise LifecycleError(
                f"{getattr(rec, 'doc_key', '?')}: 시각 역전 — "
                f"{prev_name}={prev:.1f} > {STAGE_FIELD[s]}={v:.1f}")
        prev_name, prev = STAGE_FIELD[s], v


# ------------------------------------------------------------------ 파생 (저장하지 않음)
def turn_time_s(rec) -> float | None:
    """★턴타임 = 게이트아웃 − 게이트인. 비용 항1 의 원료([04](04-비용과-보상.md))."""
    if rec.gate_out_s is None or rec.gate_in_s is None:
        return None
    return rec.gate_out_s - rec.gate_in_s


def lead_time_s(rec) -> float | None:
    """리드타임 = 게이트인 − 코피노 수신. "얼마나 미리 알았나"."""
    if rec.gate_in_s is None or rec.copino_notice_s is None:
        return None
    return rec.gate_in_s - rec.copino_notice_s


def adherence_error_s(rec, reserve_s: float | None) -> float | None:
    """예약 준수 오차 = 게이트인 − 반출입 예정. "약속을 지켰나"."""
    if rec.gate_in_s is None or reserve_s is None:
        return None
    return rec.gate_in_s - reserve_s


def censored_turn_time_s(rec, end_s: float) -> float | None:
    """미완료 검열 턴타임 — 끝에 남은 트럭이 비용을 피하지 못하게 한다.

    완료차는 `O − A`, 미완료차는 `T − A`. 게이트인조차 없으면 아직 무대 밖이라 None.
    """
    if rec.gate_in_s is None:
        return None
    end = rec.gate_out_s if rec.gate_out_s is not None else end_s
    return max(0.0, end - rec.gate_in_s)
