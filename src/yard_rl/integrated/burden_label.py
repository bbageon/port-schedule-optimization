"""BUY 견적망의 **정답지** — "트럭 한 대를 블록 b 의 슬롯 t 에 넣으면 얼마나 부담인가".

■ 왜 가상 실행이 아니라 실현 부담인가 (설계 변경 — 근거 명시)
YR-171 명세는 각 후보 슬롯에 작업을 가상 삽입하고 동결 ExecutionHead 를 굴려 정답을
만들자고 한다. 그런데 그 경로는 두 가지 문제가 있다.

  ① **계산량**: OFFER 1건당 20블록 × 48슬롯 = 960회, 하루 약 1,380만 회 →
     에피소드당 약 173시간(YR-173 실측). 표본으로 줄여도 정답지 규모가 안 나온다.
  ② **정보 누출**: 복사본의 사건 큐에는 엔진이 **실현** 블록도착으로 밀어 넣은
     BLOCK_ARRIVAL 이 남는다(`baselines._rollout_cost` 경고와 같은 함정). 이걸
     공개 예약으로 갈아끼우는 수술은 엔진 계약을 건드리고, 틀려도 티가 안 난다.

**대안**: 매일 3,600대가 실제로 블록에 들어간다. 그 한 대 한 대가 곧 "블록 b 의 슬롯
t 에 트럭을 넣은 자연 실험"이다. 정답은 굴려서 만드는 것이 아니라 **이미 일어난 것을
장부에서 읽으면 된다**. 가상 실행보다 정확하고(근사가 아니라 실현), 공짜이며,
판매된 건뿐 아니라 **전건**을 덮는다(선택 편향 없음).

  · 정확도: 가상 실행 = 시뮬레이터의 근사 / 실현 = 그 시뮬레이터의 정답
  · 표본: 판매된 (b,t) 만 관측되는 문제가 사라진다 — 트럭은 하루 종일 모든 블록에 온다
  · 정합: YR-174 보상이 쓰는 부담 분해식과 **같은 식**이다(견적과 보상이 어긋나지 않음)

■ 부담의 정의 (YR-174 `realized_credit` 의 B_dst 항과 동일)

    부담(트럭 j) = j 자신의 비용
                 + (점유 동안 그 블록에서 밀린 다른 트럭 수) × 점유시간
                 + (점유 동안 본선이 밀린 몫)          ← 10배 가중

■ 한계 (정직 고지)
· **이 시뮬레이터의 정답**이지 실제 항만의 부담이 아니다(YR-173 §데이터의 지위).
· 실현 부담은 "그 블록이 그때 실제로 겪은 것"이라 **그 트럭이 없었을 때와의 차이**가
  아니다. 뒤 트럭이 밀린 몫을 더해 반사실에 가깝게 만들지만 근사다.
· 라벨은 사후에 장부에서 만든다 — 견적망의 **입력**은 공개 정보만 쓴다(정보 경계는
  `slot_plan`·`buy_estimator` 가 각자 보장). 라벨 경로와 입력 경로를 섞지 않는다.
"""
from __future__ import annotations

from .cost_curve_v2 import RHO_VESSEL_V2
from .slot_plan import N_SLOTS, SLOT_S
from .vessel import VesselWorkType

_BACKLOG_STATES = ("WAITING", "READY", "RUNNING")


def _truck_cost(rec, l_t: float) -> float:
    """그 트럭 자신의 실현 비용(비용시간). 미완이면 None."""
    a, o = rec.gate_in, rec.gate_out
    if a is None or o is None:
        return None
    return (o - a) / 3600.0 + max(0.0, (o - a) - l_t) / 3600.0


def _waiting_at(records, t: float) -> int:
    """시각 t 에 그 블록에서 **서비스를 기다리던** 트럭 수 (자기 자신 제외 전 단계)."""
    n = 0
    for r in records.values():
        b, s = r.block_arrival, r.service_start
        if b is None or b > t:
            continue
        if s is None or s > t:
            n += 1
    return n


def _vessel_pressure(sim, t: float, occupy_s: float) -> float:
    """점유 동안 본선이 밀린 비용(비용시간). YR-174 와 **같은 근사**를 쓴다.

    본선은 한 줄기라 대기 작업 수를 곱하지 않는다. 여유 안에 끝난 본선은 비용 0.
    """
    late = False
    for v in sim.vessels.values():
        if v.work_type != VesselWorkType.LOAD:
            continue
        p = v.plan.planned_completion_s
        f = getattr(getattr(v, "truth", None), "actual_completion_s", None)
        if p is not None and f is not None and f > p:
            late = True
            break
    if not late:
        return 0.0
    pending = any(
        j.is_vessel_linked and j.status.name != "DONE" and j.release_time <= t
        or (j.is_vessel_linked and j.status.name == "DONE"
            and j.service_end is not None and j.service_end > t
            and j.release_time <= t)
        for j in sim.jobs.values())
    if not pending:
        return 0.0
    return RHO_VESSEL_V2 * occupy_s / 3600.0


def realized_burden(mbt, bid: str, *, l_t: float) -> list[dict]:
    """블록 `bid` 에 들어온 **모든 트럭**의 실현 부담. 견적망 정답지 1블록분.

    반환 각 행:
      job_id · slot(0~47, 블록도착 기준) · burden(비용시간) · occupy_s ·
      blocked(점유 동안 밀린 트럭 수) · vessel(본선 몫) · own(자기 비용)
    """
    sim = mbt.blocks[bid]
    tl = getattr(sim, "time_ledger", None)
    if tl is None:
        return []
    recs = tl.records
    out: list[dict] = []
    for jid, rec in recs.items():
        if rec.service_start is None or rec.job_done is None:
            continue                      # 미완 — 정답을 만들 수 없다
        own = _truck_cost(rec, l_t)
        if own is None:
            continue
        arr = rec.block_arrival
        if arr is None or arr < 0.0 or arr >= N_SLOTS * SLOT_S:
            continue                      # 하루 격자 밖(배수 구간)
        occupy = rec.job_done - rec.service_start
        blocked = _waiting_at(recs, rec.service_start)
        vessel = _vessel_pressure(sim, rec.service_start, occupy)
        out.append({
            "job_id": jid, "block": bid, "slot": int(arr // SLOT_S),
            "block_arrival_s": arr, "occupy_s": occupy,
            "own": round(own, 6), "blocked": blocked,
            "blocked_cost": round(blocked * occupy / 3600.0, 6),
            "vessel": round(vessel, 6),
            "burden": round(own + blocked * occupy / 3600.0 + vessel, 6)})
    return out


def terminal_burden(mbt, *, l_t: float) -> list[dict]:
    """전 블록 실현 부담 — 하루 한 에피소드가 만드는 정답지 전량(약 3,600행)."""
    rows: list[dict] = []
    for bid in mbt.blocks:
        rows.extend(realized_burden(mbt, bid, l_t=l_t))
    return rows


def burden_stats(rows: list[dict]) -> dict:
    """정답지 요약 — 규모·분포·항별 기여. 학습 전에 눈으로 본다."""
    if not rows:
        return {"n": 0}
    from statistics import fmean, pstdev
    b = [r["burden"] for r in rows]
    slots = {}
    for r in rows:
        slots.setdefault(r["slot"], 0)
        slots[r["slot"]] += 1
    return {"n": len(rows),
            "burden_mean": round(fmean(b), 4), "burden_sd": round(pstdev(b), 4),
            "burden_min": round(min(b), 4), "burden_max": round(max(b), 4),
            "own_share": round(fmean(r["own"] for r in rows) / fmean(b), 4),
            "blocked_share": round(fmean(r["blocked_cost"] for r in rows) / fmean(b), 4),
            "vessel_share": round(fmean(r["vessel"] for r in rows) / fmean(b), 4),
            "n_slots_covered": len(slots),
            "n_blocks_covered": len({r["block"] for r in rows})}
