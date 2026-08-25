"""블록 요약 — v3 는 **9차원**이다 (v2 는 7).

설계 정본: `.claude/docs/architecture/05-정보경계.md` §5 · [[YR-208]]

■ v2 에서 늘어난 둘 — 경계 안쪽인데 안 주던 것
  ⑧ **시각** `t / end` — 무대에 아침·저녁 이중 피크가 있는데 정책에 시계가 없었다.
     "지금 06시라 곧 붐빈다" 를 알 수 없었다.
  ⑨ **실제 대기 대수** — 채점은 `blockIn ≤ t < serviceStart`(줄 선 것만)로 하면서
     정책에는 `inside`(블록 안 전부)를 줬다. 같은 `inside=5` 라도 (줄 5·작업 0)과
     (줄 0·작업 1·이동 4)는 결과가 전혀 다르다.

  `inside` 는 지우지 않고 **병기**한다 — 총 대수 − 줄 선 대수 = 작업 중·이동 중이라
  그 차이도 정보다. 지우면 무엇이 효과를 냈는지 못 가른다.

■ 정보 경계
  결정 시점에 존재하는 값만 읽는다 — 통지 시각은 되고 실현 게이트인은 안 된다.
  감사 치명 6(2026-08-09)에서 실제로 샜던 자리다.
"""
from __future__ import annotations

#: 블록 요약 차원 — 계약(`block_dim_target = 9`)
BLOCK_DIM = 9

_ANNOUNCE_HORIZON_S = 1800.0     # "곧 올 통지분" 을 세는 창 (특징용 상수)


def _crane_backlog_s(sim, t: float) -> float:
    return sum(max(0.0, sim.fleet.get(c.crane_id).state.available_at - t)
               for c in sim.profile.cranes)


def _occupancy(sim) -> float:
    g = sim.profile.block
    return len(sim.stacks.containers) / max(1, g.bay_count * g.row_count * g.tier_max)


def _vessel_slack_s(sim, t: float) -> float:
    vals = [v.slack_s(t) for v in sim.vessels.values()
            if v.plan.planned_completion_s is not None and not v.done]
    return min([s for s in vals if s is not None] + [2.0 * 3600.0])


def waiting_count(mbt, bid: str, t: float, records) -> int:
    """★줄 선 대수 — `blockIn ≤ t < serviceStart`. 채점이 쓰는 정의와 **같다**.

    `records` 는 `{doc_key: ExecutionRecord}`. 그 블록 소속만 센다.
    """
    sim = mbt.blocks[bid]
    n = 0
    for jid in sim.jobs:
        rec = records.get(jid)
        if rec is not None and rec.waiting_at(t):
            n += 1
    return n


def inside_count(mbt, bid: str, t: float, records) -> int:
    """블록 안 전부 — `gateIn ≤ t < gateOut`. 이동 중·작업 중까지 포함한다."""
    sim = mbt.blocks[bid]
    n = 0
    for jid in sim.jobs:
        rec = records.get(jid)
        if rec is None or rec.gate_in_s is None or rec.gate_in_s > t:
            continue
        if rec.gate_out_s is None or rec.gate_out_s > t:
            n += 1
    return n


def pipeline_count(mbt, bid: str, t: float, records) -> int:
    """오는 중 — 게이트는 지났는데 블록에 아직 안 닿은 트럭."""
    sim = mbt.blocks[bid]
    n = 0
    for jid in sim.jobs:
        rec = records.get(jid)
        if rec is None or rec.gate_in_s is None or rec.gate_in_s > t:
            continue
        if rec.block_in_s is None or rec.block_in_s > t:
            n += 1
    return n


def announced_soon(mbt, bid: str, t: float, orders) -> int:
    """곧 올 **통지분** — 공개 정보만. 실현 게이트인은 안 읽는다."""
    sim = mbt.blocks[bid]
    n = 0
    for jid in sim.jobs:
        o = orders.get(jid)
        if o is None or o.con_loc != bid:
            continue
        eta = o.in_out_reserve_s          # 통지된 예정 — 실현이 아니다
        if t < eta <= t + _ANNOUNCE_HORIZON_S:
            n += 1
    return n


#: ★도착 시각 주변을 보는 반폭 (초) — [[YR-230]]. 앞뒤 30분 = 창 1시간.
#: `_ANNOUNCE_HORIZON_S` 와 같은 눈금으로 맞춘다. **동결 대상**이다.
_ARRIVAL_HALF_W_S = 1800.0


def announced_around(mbt, bid: str, center_s: float, orders,
                     half_w: float = _ARRIVAL_HALF_W_S) -> int:
    """`center_s ± half_w` 에 그 블록으로 **통지된** 물량 — 공개 정보만.

    ■ 왜 필요한가 ([[YR-230]])
      `announced_soon` 은 **지금부터 30분**만 센다. 그래서 오전 8시에 트럭을
      2시간 뒤로 미룰 때, 정책은 **미룬 그 시각(10시)이 얼마나 붐빌지를 못 본다**.

      실측([[YR-227]]·[[YR-231]]): 오전 7~11시 창의 이연은 거래 1건당
      **+147,779원 손해**(하루끝 기준)인데 라벨은 −2,192원 **이득**이라 말했다.
      원인이 둘인데(라벨 부호·정보 부재) 이 함수가 둘째를 막는다.

    ■ 정보 경계를 지킨다
      `in_out_reserve_s` 는 **통지된 예정**이지 실현 게이트인이 아니다.
      `announced_soon` 과 같은 출처라 새로 열리는 정보가 없다.
    """
    sim = mbt.blocks[bid]
    lo, hi = float(center_s) - half_w, float(center_s) + half_w
    n = 0
    for jid in sim.jobs:
        o = orders.get(jid)
        if o is None or o.con_loc != bid:
            continue
        if lo <= o.in_out_reserve_s <= hi:
            n += 1
    return n


def block_features(mbt, bid: str, t: float, *, n_cands: int,
                   records, orders, end_s: float) -> list[float]:
    """블록 요약 9차원 — 전부 공개 정보.

    v2 의 7 + **시각** + **줄 선 대수** ([[YR-208]]).
    """
    sim = mbt.blocks[bid]
    return [
        inside_count(mbt, bid, t, records) / 10.0,
        pipeline_count(mbt, bid, t, records) / 10.0,
        _crane_backlog_s(sim, t) / 3600.0,
        _occupancy(sim),
        max(-2.0, min(2.0, _vessel_slack_s(sim, t) / 3600.0)),
        announced_soon(mbt, bid, t, orders) / 10.0,
        n_cands / 6.0,
        # ── v3 신규 ─────────────────────────────────────────────
        min(1.0, t / max(1.0, end_s)),                       # ⑧ 지금 몇 시인가
        waiting_count(mbt, bid, t, records) / 10.0,          # ⑨ 실제 줄 선 대수
    ]
