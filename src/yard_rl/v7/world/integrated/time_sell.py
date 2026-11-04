"""YR-161 — 시간 판매(재예약 요청) 후보·적용 계층 (사용자 제안 2026-08-08).

■ 판매 축 확정 (사용자 결정 2026-08-08)
  | 작업 | 판매 | 축 |
  |---|---|---|
  | 반입(GATE_IN)   | ✅ | **공간** — 게이트 진입 전 목적지 블록 변경 (YR-151·pre_gate) |
  | 반출(GATE_OUT)  | ✅ | **시간** — "다른 시각에 와 주세요" 재예약 (본 모듈) |
  | 양하·적하(본선) | ❌ | **제외 확정** — 적하: 본선 지연 직결·기사 없음·기존 크레인
  |                 |    | 행동과 중복. 양하 시간축: 같은 이유. 양하 공간축은 원리상
  |                 |    | 유효하나 반입 실증·엔진 배선 전에는 열지 않는다(확장 후보로만).

  (구현 이력: 적하 embargo 기술 계층을 만들었다가 위 결정으로 **제거**했다 —
   정책에 편입되지 않는 계층을 코드에 남기면 대시보드↔코드 정합이 깨진다.)

■ 반출 시간 판매의 뜻
블록이 혼잡하면 기사에게 "다른 시간을 예약해서 들어와 주세요"라고 요청한다. 본 연구는
**전원 수락 가정**(D5 assumed — 실제 수락률·보상 비용은 Level 3 자료 확보 전 미지)이며,
주장 범위는 "수락률 100% 일 때의 상한 효과"로 한정한다.

■ 퇴화 방지 계약 (필수)
전원 수락 + 이연 비용 0 이면 정책이 트럭을 무한정 미뤄 혼잡을 장부에서 지운다
(YR-052/119 무기한 WAIT 퇴화의 재판). 그래서
  · **비용 원점 = `appointment_gate_time`(최초 통지 시각)** — 이연으로 생기는 기사
    외부 대기(실제 gate-in − 최초 통지)가 장부에 남는다(`deferral_ledger`).
  · 작업당 이연 상한 — 엔진(`defer_admitted_entry`)이 fail-closed 로 강제.

■ 선결: 사전 통지 lead>0 (walk-in lead 0 이면 진입 전 창이 없어 후보 0).
  lead 설계는 공간 판매 0B 와 공유한다.
■ 테스트 유예 (2026-08-08 빌드 우선 지시) — 디버깅 국면에서 계약 테스트 몰아서 작성.
"""
from __future__ import annotations

from ..domain.enums import JobStatus

DEFER_WINDOW_S = 1_800.0      # 재예약 검토 창 — 공개 예측 도착 30분 전 (공간 판매와 동일)
DEFER_DELTA_S = 900.0         # 1회 이연량 기본값 15분 (사전등록 동결 대상 — 튜닝 금지)
MAX_ENTRY_DEFERRALS = 1       # 작업당 이연 상한 (이송 1회 상한과 대칭·엔진 fail-closed)


def notified_gate_in(job) -> float | None:
    """정책·resolver 가 읽어도 되는 **공개 통지 진입시각** (정보경계 정정 — 감사 치명 6).

    실현 `actual_gate_in` 을 직접 읽으면, 예약 준수오차가 도입되는 순간 미래정보 누출이
    된다(현 walk-in 에서는 둘이 우연히 같아 문제가 숨었다). 공개값의 유일한 원천은
    ①통지/재예약 시각(`notified_gate_in_s` — 투입·이연 때 갱신) ②없으면 최초 예약
    (`appointment_gate_time`)이다. 정책 경로는 반드시 이 접근자만 쓴다.
    """
    v = getattr(job, "notified_gate_in_s", None)
    if v is not None:
        return float(v)
    v = getattr(job, "appointment_gate_time", None)
    return None if v is None else float(v)


# ------------------------------------------------------------------ 후보
def iter_time_sell_candidates(mbt, src: str, *, horizon_s: float = DEFER_WINDOW_S,
                              max_deferrals: int = MAX_ENTRY_DEFERRALS
                              ) -> list[tuple[str, float]]:
    """블록 src 가 지금 "다른 시간에 와 달라"고 요청할 수 있는 **반출** 트럭.

    (job_id, 공개 ETA) 목록. 공개 정보만 쓴다 — 통지된 gate-in(아직 미래)·공개 예측
    도착. 실현 미래값 미열람. 대상은 반출(GATE_OUT)뿐이다: 반입은 공간 판매가 담당하고
    (한 작업에 두 축을 겹치지 않는다 — 단일축 원칙), 양하·적하는 제외 확정.
    """
    now = mbt.now
    sim = mbt.blocks[src]
    out = []
    for jid in sorted(sim.jobs):
        rec = mbt.ledger.records.get(jid)
        if rec is None or rec.owner != src:
            continue
        if rec.flow != "GATE_OUT" or rec.entry_deferrals >= max_deferrals:
            continue
        j = sim.jobs[jid]
        gi = notified_gate_in(j)                       # 공개 통지 시각만 (누출 금지)
        if gi is None or gi <= now + 1e-6:
            continue                                   # 이미 진입 — 창 밖
        if j.status != JobStatus.PLANNED:
            continue
        eta = getattr(j, "estimated_block_arrival", None) or getattr(j, "provided_eta", None)
        if eta is None or not (0.0 < eta - now <= horizon_s):
            continue
        out.append((jid, eta))
    return out


def try_time_sell(mbt, job_id: str, *, delta_s: float = DEFER_DELTA_S,
                  max_deferrals: int = MAX_ENTRY_DEFERRALS,
                  t: float | None = None) -> bool:
    """재예약 요청 1건 실행 — 전원 수락 가정이므로 요청 = 확정. 실패는 KEEP.

    ★YR-171-A: 공개 예약 장부(`day_plan`)가 붙어 있으면 **장부를 먼저 검사**하고
    엔진 확정 뒤에 장부를 갱신한다(버전 상승). 순서가 중요하다 — 엔진이 먼저 확정한
    뒤 장부가 거부하면 둘이 어긋난 채로 남는다. 하루 격자를 넘기는 재예약은
    **거부**한다(명세: 날짜 변경 금지) — 거부는 KEEP 과 같다.
    """
    from .day_plan import get as _day_plan_get
    plan = _day_plan_get(mbt)
    new_t = None
    if plan is not None:
        from .slot_plan import DAY_S
        cur = plan.gate_in(job_id)
        if cur is None:
            return False
        new_t = cur + delta_s
        if not (0.0 <= new_t < DAY_S):
            return False                  # 날짜 넘김 — 재예약 불가
    ok = mbt.try_defer_admitted_entry(job_id, delta_s, max_deferrals=max_deferrals)
    if ok and plan is not None:
        plan.reschedule(job_id, new_t, t=t)
    return ok


# ------------------------------------------------------------------ 감사
def deferral_ledger(mbt) -> list[dict]:
    """이연 이력 전량 — 감사·비용 계상용. 원점(appointment) 대비 실제 진입의 차가
    기사 외부 대기이며, 이 값이 0 으로 보고되면 비용 은닉을 의심해야 한다."""
    rows = []
    for jid, rec in sorted(mbt.ledger.records.items()):
        if rec.entry_deferrals == 0:
            continue
        sim = mbt.blocks.get(rec.owner)
        j = sim.jobs.get(jid) if sim is not None else None
        appt = getattr(j, "appointment_gate_time", None) if j is not None else None
        rows.append({"job_id": jid, "block": rec.owner, "flow": rec.flow,
                     "n_deferrals": rec.entry_deferrals,
                     "deferred_total_s": rec.entry_deferred_s,
                     "original_appointment_s": appt,
                     "actual_gate_in_s": rec.a_gate_in,
                     "driver_outside_wait_s": (None if appt is None or rec.a_gate_in is None
                                               else rec.a_gate_in - appt)})
    return rows
