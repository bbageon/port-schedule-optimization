"""YR-161 — 시간 판매(재예약 요청) 후보·적용 계층 (사용자 제안 2026-08-08).

■ 무엇인가
반입(GATE_IN)의 판매는 **공간**(어느 블록으로)이고, 반출(GATE_OUT)은 컨테이너가 이미
특정 블록에 있어 목적지를 못 바꾼다 — 대신 **시간**(언제 올지)을 판다. 블록이 혼잡하면
기사에게 "다른 시간을 예약해서 들어와 주세요"라고 요청하고, 본 연구는 **전원 수락 가정**
(D5 assumed — 실제 수락률·보상 비용은 Level 3 자료 확보 전 미지)이다.

■ 두 메커니즘 (엔진 구현 위치)
  · 반출·반입 트럭: `MultiBlockTerminal.defer_admitted_entry` — 진입 전 재예약.
    **비용 원점 = `appointment_gate_time`(최초 통지)** 이 보존되므로 이연이 기사 대기를
    장부에서 지우지 못한다(퇴화 방지 — YR-052/119 무기한 WAIT 의 재판을 막는 축).
  · 적하(VESSEL_LOAD): `sim._load_embargo[jid] = until_t` — 후보 embargo. 지연 비용은
    본선 버퍼 고갈 → 본선 지연으로 **자기-계상**된다(원점 보존 불요).

■ 선결·한계 (정직 고지)
  · 진입 전 재예약은 **사전 통지 lead>0** 에서만 창이 열린다(walk-in lead 0 이면 후보 0).
    lead 설계는 공간 판매 0B 와 공유한다.
  · 적하 embargo 는 기사 서사가 없고(내부 작업) 본선 마감(ETD)에 직결된다 — 스펙 계약
    4항이 1차 제외를 권고했으나 사용자 제안 범위에 포함되어 기술 계층은 구현해 둔다.
    학습 행동으로 열지 여부는 별도 사전등록으로 결정한다.
■ 테스트 유예 (2026-08-08 빌드 우선 지시) — 디버깅 국면에서 계약 테스트 몰아서 작성.
"""
from __future__ import annotations

from ..domain.enums import JobFlow, JobStatus

DEFER_WINDOW_S = 1_800.0      # 재예약 검토 창 — 공개 예측 도착 30분 전 (공간 판매와 동일)
DEFER_DELTA_S = 900.0         # 1회 이연량 기본값 15분 (사전등록 동결 대상 — 튜닝 금지)
MAX_ENTRY_DEFERRALS = 1       # 작업당 이연 상한 (이송 1회 상한과 대칭·엔진 fail-closed)


# ------------------------------------------------------------------ 후보
def iter_time_sell_candidates(mbt, src: str, *, horizon_s: float = DEFER_WINDOW_S,
                              flows: tuple[str, ...] = ("GATE_OUT",),
                              max_deferrals: int = MAX_ENTRY_DEFERRALS) -> list[tuple[str, float]]:
    """블록 src 가 지금 "다른 시간에 와 달라"고 요청할 수 있는 트럭 — (job_id, 공개 ETA).

    공개 정보만 쓴다: 통지된 gate-in(아직 미래)·공개 예측 도착. 실현 미래값 미열람.
    기본 대상은 반출(GATE_OUT) — 반입은 공간 판매가 담당하고, 두 축을 같은 작업에
    겹쳐 학습하지 않는다(단일축 원칙). 엔진은 양쪽 다 허용하므로 flows 로 명시 확장 가능.
    """
    now = mbt.now
    sim = mbt.blocks[src]
    out = []
    for jid in sorted(sim.jobs):
        rec = mbt.ledger.records.get(jid)
        if rec is None or rec.owner != src:
            continue
        if rec.flow not in flows or rec.entry_deferrals >= max_deferrals:
            continue
        if rec.a_gate_in is None or rec.a_gate_in <= now + 1e-6:
            continue                                   # 이미 진입 — 창 밖
        j = sim.jobs[jid]
        if j.status != JobStatus.PLANNED:
            continue
        eta = getattr(j, "estimated_block_arrival", None) or getattr(j, "provided_eta", None)
        if eta is None or not (0.0 < eta - now <= horizon_s):
            continue
        out.append((jid, eta))
    return out


def try_time_sell(mbt, job_id: str, *, delta_s: float = DEFER_DELTA_S,
                  max_deferrals: int = MAX_ENTRY_DEFERRALS) -> bool:
    """재예약 요청 1건 실행 — 전원 수락 가정이므로 요청 = 확정. 실패는 KEEP."""
    return mbt.try_defer_admitted_entry(job_id, delta_s, max_deferrals=max_deferrals)


# ------------------------------------------------------------------ 적하 embargo
def embargo_vessel_load(sim, job_id: str, until_s: float) -> bool:
    """적하 작업 1건을 until 까지 SERVE 후보에서 제외한다 (opt-in — 기본 미사용).

    지연 비용은 본선 지연으로 자기-계상된다. 학습 행동으로 열려면 별도 사전등록 필요.
    """
    j = sim.jobs.get(job_id)
    if j is None or j.flow != JobFlow.VESSEL_LOAD or j.status != JobStatus.PLANNED:
        return False
    if until_s <= sim.now:
        return False
    if not hasattr(sim, "_load_embargo"):
        sim._load_embargo = {}
    sim._load_embargo[job_id] = float(until_s)
    return True


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
