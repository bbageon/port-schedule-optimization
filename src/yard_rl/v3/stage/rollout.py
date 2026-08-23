"""반사실 rollout — 같은 순간에서 **행동만 바꿔** H(=3시간) 을 다시 굴린다.

설계 정본: `.claude/docs/architecture/04b-학습-잣대.md` §3 · [[YR-215]] ④⑤

■ 어떻게 "같은 순간" 을 만드나
  결정이 일어나는 곳은 `review_fn` 안이고, 그때 **전 블록이 같은 t 에 park** 해 있다
  (공용 시계 계약). 그 시점의 터미널을 통째로 복제하면 사건 큐·적재·장부·시계가
  전부 같은 분기점이 된다. 복제본의 `end` 를 `t + H` 로 낮춰 H 만큼만 굴린다.

  ★분기점은 **결정 전**이어야 한다. 결정 뒤에 뜨면 그 오더가 이미 `decided` 라
  분기 세계가 다시 결정하지 않고, 거래도 이미 확정돼 "안 팔았다면" 을 만들 수 없다.

  ★분기 epoch 은 **손으로 한 번 연다**. 엔진은 `review_fn` 을 부르기 직전에 park
  기록을 비우므로, 복제본을 그냥 굴리면 t 를 건너뛰고 t+60 으로 간다.

■ ★세계는 셋이다 — factual 도 굴린다 (2026-08-22 정정)
  한때 "factual 은 실제 궤적에서 읽으면 공짜" 라고 짰다. **틀렸다.**

  분기 세계는 `sim.end = t+H` 라 창 끝 직전에 새 작업을 시작하지 않는다. 실제
  궤적은 계속 굴러가므로 경계 근처 트럭의 서비스 시각이 어긋난다. 실측(2026-08-22,
  부하 3,500 · 굴린 factual vs 같은 창을 실제 궤적에서 읽은 값):

      t= 0.0h  +15.74%      t=11.1h  +0.31%
      t=13.9h   +0.05%      t=16.7h  +0.08%

  새벽(창 안 Φ 가 27만원)에서 크고 한낮(수천만원)에서 작다 — **절대량은 비슷한데
  분모가 작아서** 비율이 커진다. 지평을 15·30·60분 더 주면 대기 항 차이가
  15,369 → 2,456 → −127 → 0 으로 수렴하므로 원인은 **지평 절단**이 맞다.

  대안 세계도 같은 절단을 겪으므로, 셋을 모두 같은 방식으로 굴리면 절단분이
  상쇄되고 **행동의 효과만** 남는다. 한쪽만 절단하지 않은 값을 쓰면 그 편향이
  라벨에 그대로 들어간다 — 새벽 표본에서는 라벨보다 편향이 클 수도 있다.

  → 읽기 최적화는 폐기한다. 결정 1건 = **rollout 3회**(문서 설계 그대로).

■ ⚠️ H=1시간은 **먼 이연을 못 본다** (2026-08-22 실측)
  Seller 가 `TIME@8`(2시간 뒤 칸)을 팔았는데 KEEP 대안과 Φ 가 **정확히 같았다** —
  이연 효과가 창 밖에 있어 창 안에서는 아무 일도 안 일어난다. 라벨이 0 이면
  "그 행동은 무의미" 로 배운다. 이연 칸을 H 안으로 제한하든 H 를 늘리든 결정이
  필요하다 → [[YR-216]].

■ 동일성 불변식은 **결정**을 본다
  Φ 는 위 이유로 실제 궤적과 같을 수 없다(절단). 대신 factual 가지가 실제 궤적과
  **같은 결정**을 내는지 본다 — 판매 행동·좌표·구매 응답이 전부 일치해야 한다.
  하나라도 어긋나면 분기 재조립이 상태를 제대로 복원하지 못한 것이다.
  절단 폭은 진단으로 함께 보고한다(라벨 편향이 아니라 **공통 상수**임을 보이려고).

■ ⚠️ 비용 실측 (2026-08-22, 부하 3,500)
      복제 0.6초 + H=1시간 재생 1.4~7.9초 (하루 늦을수록 비쌈) → 세계 1개 ≈ 4.8초
  전 결정(3,500건 × 3세계)을 라벨링하면 **회차당 14시간**이라 불가능하다.
  `RolloutBudget` 으로 **회차마다 K건만 표본**한다 — 메커니즘은 그대로 두고 표본
  수만 줄이는 것이라 우회가 아니다.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from ..reward.counterfactual import _count_rollout
from ..reward.phi import terminal_cost_krw
from .orders import EPOCH_S


@dataclass
class RolloutBudget:
    """회차당 라벨링할 결정 수. 전수는 물리적으로 불가능하다(모듈 머리 참조).

    고르는 방법은 **결정론 등간격**이다 — 난수를 쓰면 같은 시드가 같은 라벨 집합을
    못 만들어 재현이 깨진다.
    """

    max_labels: int = 64
    identity_checks: int = 2          # 그중 동일성 불변식을 실제로 검사할 건수
    _seen: int = field(default=0, repr=False)
    _taken: int = field(default=0, repr=False)
    #: 표본 간격. 0 = 자동(에피소드가 부하에 맞춰 정한다).
    stride: int = 0

    def take(self) -> bool:
        """이 결정을 라벨링할까. 부른 순서대로 세고 `stride` 마다 하나씩."""
        idx = self._seen
        self._seen += 1
        if self._taken >= self.max_labels or idx % max(1, self.stride) != 0:
            return False
        self._taken += 1
        return True

    def peek(self, n_ahead: int) -> bool:
        """★복제 **전에** 묻는다 — 다음 `n_ahead` 건 중 하나라도 뽑힐까.

        스냅샷 복제가 0.6초라 매 epoch 뜨면 하루에 14분이 사라진다. 뽑을 일이
        없는 epoch 은 복제 자체를 건너뛴다. 계수기는 건드리지 않는다(`take` 만 센다).
        """
        if self._taken >= self.max_labels:
            return False
        st = max(1, self.stride)
        return any((self._seen + k) % st == 0 for k in range(max(0, n_ahead)))

    @property
    def taken(self) -> int:
        return self._taken

    @property
    def seen(self) -> int:
        return self._seen


@dataclass
class BranchResult:
    """분기 세계 하나의 결과 — Φ 와 **그 세계가 내린 결정**."""

    phi_krw: float
    seller_action: str | None = None
    seller_coord: str | None = None
    buyer_action: str | None = None
    traded_edges: int = 0
    decisions: int = 0
    admitted: int = 0


class SnapshotRollout:
    """분기점 하나에서 세계를 굴린다.

    `ctx` 는 에피소드가 넘겨주는 재조립 재료다 — 실행 정책 공장, 공고기, 시장을
    다시 만드는 함수. **망(weights)은 복제하지 않고 공유**한다: 복제하면 학습 중인
    가중치가 갈라져 라벨이 다른 정책의 것이 된다.
    """

    def __init__(self, ctx, *, horizon_s: float, grid_s: float = EPOCH_S):
        self.ctx = ctx
        self.horizon_s = float(horizon_s)
        self.grid_s = float(grid_s)
        self.n_worlds = 0

    def branch(self, mbt, t: float, *, orders, records, decided, doc_key: str,
               force_seller=None, force_buyer=None) -> BranchResult:
        """`t` 에서 갈라 H 만큼 굴리고 **그 창의 Φ(원화)** 와 결정을 돌려준다.

        `force_seller` / `force_buyer` = (docKey, 행동). 분기 직후 첫 결정에서만
        먹고 그 뒤로는 정책이 평소대로 판단한다 — "상대를 가정하지 않는다"(04b §3).
        """
        _count_rollout()
        self.n_worlds += 1
        end = t + self.horizon_s

        snap = copy.deepcopy(mbt)
        for sim in snap.blocks.values():
            sim.end = min(sim.end, end)

        o2 = dict(orders)
        r2 = copy.deepcopy(records)
        market2 = self.ctx.make_market(snap, decided=set(decided))
        if force_seller is not None:
            market2.seller.force_once[force_seller[0]] = force_seller[1]
        if force_buyer is not None:
            market2.buyer.force_once[force_buyer[0]] = force_buyer[1]

        # ★`end_s` 는 넘기지 않는다 — 분기 세계도 **하루의 끝**을 그대로 받아야
        #   같은 선택지·같은 시계 특징을 본다. 절단은 `sim.end` 로만 건다.
        bridge2 = self.ctx.make_bridge(market2, orders=o2, records=r2,
                                       on_decision=None)      # ★교사 재귀 금지
        ann2 = self.ctx.announcer.clone_fresh()

        def review(m, tt):
            ann2.review(m, tt)
            bridge2.review(m, tt)

        bridge2.review(snap, t)          # ★분기 epoch 을 손으로 연다 (머리 참조)
        snap.run(self.ctx.make_exec_policy(), review_fn=review)

        phi = terminal_cost_krw(r2, end_s=end,
                                vessel_idle=self.ctx.vessel_idle(snap, end),
                                yc_extra_move_s=self.ctx.yc_extra_move_s(snap),
                                rehandles=self.ctx.rehandles(snap))
        s_e = next((e for e in market2.seller.trail if e["doc_key"] == doc_key), None)
        b_e = next((e for e in market2.buyer.trail if e["doc_key"] == doc_key), None)
        return BranchResult(
            phi_krw=phi.total,
            seller_action=(s_e["action"] if s_e else None),
            seller_coord=(s_e["coord"] if s_e else None),
            buyer_action=(b_e["action"] if b_e else None),
            traded_edges=bridge2.traded_edges,
            decisions=len(market2.seller.trail),
            admitted=ann2.n_admitted)


def identity_check(*, factual: BranchResult, seller_entry, buyer_entry,
                   phi_read: float) -> dict:
    """★동일성 불변식 — factual 가지가 실제 궤적과 **같은 결정**을 내는가.

    Φ 로는 못 잰다. 분기 세계는 `t+H` 에서 세상이 끝나므로 경계 근처 서비스가
    실제와 달라지고, 그 차이는 라벨 편향이 아니라 **세 세계가 공유하는 상수**다
    (모듈 머리의 pad 수렴 실측). 그래서 검사 대상은 **결정 자체**다.

    같이 돌려주는 `phi_gap`·`phi_gap_ratio` 는 그 절단 폭이 얼마나 되는지를 보이는
    진단이다 — 판정 기준이 아니다.
    """
    want_s = seller_entry["action"]
    want_c = seller_entry.get("coord")
    want_b = buyer_entry["action"] if buyer_entry else None
    ok_s = factual.seller_action == want_s
    ok_c = factual.seller_coord == want_c
    ok_b = factual.buyer_action == want_b
    return {"ok": bool(ok_s and ok_c and ok_b),
            "seller": [want_s, factual.seller_action, ok_s],
            "coord": [want_c, factual.seller_coord, ok_c],
            "buyer": [want_b, factual.buyer_action, ok_b],
            "phi_rolled": factual.phi_krw, "phi_read": float(phi_read),
            "phi_gap": factual.phi_krw - float(phi_read),
            "phi_gap_ratio": ((factual.phi_krw - float(phi_read))
                              / max(1e-9, abs(factual.phi_krw)))}
