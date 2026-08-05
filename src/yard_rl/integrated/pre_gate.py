"""YR-151 0A — 게이트 진입 전(PRE_GATE) 재배정 후보: **공개정보 전용** 생성기.

■ 정보 계약 (0A 핵심)
- 후보 자격·창 판정에 쓰는 시각은 **공개 예측**뿐이다: `estimated_block_arrival`(예약 기반)
  → 없으면 `appointment_gate_time` → 없으면 `provided_eta`.
- 실현값(`actual_gate_in`·`actual_block_arrival`)의 **값은 절대 읽지 않는다**.
  단 "게이트 진입 사건이 이미 일어났는가"는 **과거 사건의 관측**이므로 boolean 로만 쓴다
  (`gate_in_observed`) — 미래 시각을 읽는 것이 아니다.
- 이 계약은 `probe_public_info_only` 로 **실현 미래값을 흔들어도 후보가 불변**임을 검사한다.

■ 창: `0 < 공개예측 블록도착 − now <= horizon_s`(기본 1800s = 30분).
"""
from __future__ import annotations

from ..domain.enums import JobFlow, JobStatus

WINDOW_S = 1800.0          # 30분 (spec 동결)
MAX_TRANSFERS = 1          # 작업당 1회 상한 (PoC)


def public_block_eta(job) -> float | None:
    """공개 예측 블록도착 — 실현값 미열람. 없으면 None(= 후보 제외, fail-closed)."""
    for attr in ("estimated_block_arrival", "provided_eta"):
        v = getattr(job, attr, None)
        if v is not None:
            return float(v)
    return None


def gate_in_observed(mbt, job_id: str) -> bool:
    """게이트 진입 **사건이 이미 관측됐는가** — 과거 사건 유무만 반환(시각 미노출)."""
    rec = mbt.ledger.records.get(job_id)
    if rec is None or rec.a_gate_in is None:
        return False
    return rec.a_gate_in <= mbt.now + 1e-9


def iter_pre_gate_candidates(mbt, src: str, *, horizon_s: float = WINDOW_S,
                             max_transfers: int = MAX_TRANSFERS) -> list[tuple[str, float]]:
    """(job_id, 공개예측 블록도착) — 결정론(작업 id 순). 반입·미진입·창 안·이송상한 이내."""
    now = mbt.now
    src_sim = mbt.blocks[src]
    out: list[tuple[str, float]] = []
    for jid in sorted(src_sim.jobs):
        rec = mbt.ledger.records.get(jid)
        if rec is None or rec.owner != src:
            continue
        if not rec.reassignable or rec.transfer_count >= max_transfers:
            continue
        if gate_in_observed(mbt, jid):
            continue                                  # 이미 진입 = PRE_GATE 창 밖
        j = src_sim.jobs[jid]
        if j.flow != JobFlow.GATE_IN or j.status != JobStatus.PLANNED:
            continue                                  # 1차 범위 = 반입만
        eta = public_block_eta(j)
        if eta is None:
            continue                                  # 공개 예측 결측 = KEEP (fail-closed)
        dt = eta - now
        if 0.0 < dt <= horizon_s:
            out.append((jid, eta))
    return out


def probe_public_info_only(mbt, src: str, *, delta_s: float = 900.0) -> dict:
    """★누출 검사 — 미래 **실현값만** 흔들고 후보가 그대로인지 본다.

    공개 예측은 손대지 않고 아직 오지 않은 `actual_gate_in`·`actual_block_arrival` 을
    `delta_s` 만큼 뒤로 민다(미래 유지). 후보 집합·순서가 달라지면 실현 미래를 읽은 것이다.
    검사 후 원상복구한다.
    """
    before = iter_pre_gate_candidates(mbt, src)
    src_sim = mbt.blocks[src]
    saved: list[tuple[str, float | None, float | None, float | None]] = []
    now = mbt.now
    for jid, j in src_sim.jobs.items():
        a = getattr(j, "actual_gate_in", None)
        b = getattr(j, "actual_block_arrival", None)
        if a is None or a <= now:
            continue                                   # 이미 관측된 과거는 건드리지 않는다
        rec = mbt.ledger.records.get(jid)
        saved.append((jid, a, b, None if rec is None else rec.a_gate_in))
        j.actual_gate_in = a + delta_s
        if b is not None:
            j.actual_block_arrival = b + delta_s
        if rec is not None:
            rec.a_gate_in = rec.a_gate_in + delta_s
    try:
        after = iter_pre_gate_candidates(mbt, src)
    finally:
        for jid, a, b, ra in saved:
            j = src_sim.jobs[jid]
            j.actual_gate_in = a
            if b is not None:
                j.actual_block_arrival = b
            rec = mbt.ledger.records.get(jid)
            if rec is not None and ra is not None:
                rec.a_gate_in = ra
    return {"n_perturbed": len(saved), "identical": before == after,
            "n_before": len(before), "n_after": len(after)}
