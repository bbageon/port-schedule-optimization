"""v1 — PPO 확률 정책 정책이 보는 특징 — **이 세대 전용 사본**.

■ 왜 세대마다 따로 두나 (사용자 지시 2026-08-20)
  v1·v2 는 **끝난 세대**다. 특징 코드를 각자 갖고 있으면 완전히 얼어붙어,
  v3 를 아무리 고쳐도 지난 판정이 흔들릴 수 없다. 공용으로 두면 한 줄 고칠 때마다
  "그럼 예전 결과는?"을 매번 확인해야 한다.

  세대 간 코드가 겹치는 것은 **의도된 중복**이다. 합치지 않는다.

■ 정보 경계 (지키지 않으면 반칙)
  결정 시점에 존재하는 값만 읽는다 — 통지 시각(`notified_gate_in`)은 되고
  실현 게이트인은 안 된다. 감사 치명 6(2026-08-09)에서 실제로 샜던 자리다.

■ 무대는 공유한다
  `yard_rl.integrated`(엔진·비용·레이아웃·배정기)는 세대별로 복제하지 않는다.
  판정이 **짝비교**라 세 세대가 같은 무대를 받아야 비교가 성립한다.
"""
from __future__ import annotations

from ..integrated.sell_review import block_inside, block_pipeline

BLOCK_DIM = 7


# ------------------------------------------------------------------ 특징 (전부 공개 정보)
def block_features(mbt, src: str, t: float, n_cands: int) -> list[float]:
    """블록 계획 요약 7차원 — "작업을 품은 계획"을 평가하기 위한 맥락."""
    sim = mbt.blocks[src]
    inside = block_inside(sim, t)
    pipeline = block_pipeline(mbt, src, t)
    crane_backlog = sum(max(0.0, sim.fleet.get(c.crane_id).state.available_at - t)
                        for c in sim.profile.cranes)
    g = sim.profile.block
    occupancy = len(sim.stacks.containers) / max(1, g.bay_count * g.row_count * g.tier_max)
    slack_vals = [v.slack_s(t) for v in sim.vessels.values()
                  if v.plan.planned_completion_s is not None and not v.done]
    vessel_slack = min([s for s in slack_vals if s is not None] + [2.0 * 3600.0])
    # 정보경계(감사 치명 6): 실현 a_gate_in 대신 **공개 통지 시각**만 읽는다.
    from .time_sell import notified_gate_in
    announced_30 = 0
    for jid, rec in mbt.ledger.records.items():
        if rec.owner != src:
            continue
        jj = sim.jobs.get(jid)
        gi = notified_gate_in(jj) if jj is not None else None
        if gi is not None and t < gi <= t + 1800.0:
            announced_30 += 1
    return [inside / 10.0, pipeline / 10.0, crane_backlog / 3600.0, occupancy,
            max(-2.0, min(2.0, vessel_slack / 3600.0)), announced_30 / 10.0,
            n_cands / 6.0]


def candidate_features(mbt, src: str, jid: str, t: float) -> list[float]:
    """후보 6차원 — 창 내 잔여시간(멈춤 문제의 시계)·이력 포함. 공개 정보만(감사 6)."""
    from .time_sell import notified_gate_in
    rec = mbt.ledger.records[jid]
    j = mbt.blocks[src].jobs[jid]
    eta = getattr(j, "estimated_block_arrival", None) or j.provided_eta or t
    gi = notified_gate_in(j)                       # 공개 통지 시각 (실현값 미열람)
    gate_remain = (gi - t) if gi is not None else 0.0
    is_out = 1.0 if rec.flow == "GATE_OUT" else 0.0
    size40 = 1.0 if str(getattr(j, "inbound_size", "")).endswith("40") else 0.0
    return [max(0.0, min(1.0, (eta - t) / 1800.0)), is_out, size40,
            max(0.0, min(1.0, gate_remain / 1800.0)),
            float(rec.transfer_count), float(rec.entry_deferrals)]
