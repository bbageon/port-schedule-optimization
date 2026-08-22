"""Buyer — offer 를 보고 BUY / REJECT 를 **스스로** 고른다.

설계 정본: `.claude/docs/architecture/03-결정층.md` §3-1

■ 이게 v3 의 핵심 변화다
  v2 는 받는 쪽에 **거부권이 없어** 중앙이 붐비는 블록에 밀어넣을 수 있었다.
  거절권이 생기면 그 떠넘기기가 **구조적으로** 막힌다.

■ 다른 Buyer 의 응답은 못 본다
  동시 결정이라 구조적으로 불가능하다. 경쟁 입찰이 아니라 **각자 자기 값만 보고**
  응답한다.

■ 시간 축의 Buyer 는 "목표 슬롯" 이다
  블록이 안 바뀌어 상대가 없어 보이지만, 실제 VBS 예약 슬롯에는 용량이 있어
  차면 거절한다. 그래야 "상호 동의가 있어야 거래" 가 **양 축에서** 성립한다.

■ 두 세계를 어떻게 가르나
  같은 망으로 두 행을 채점한다 — `BUY` 행은 offer 특징을 그대로 담고, `REJECT`
  행은 **offer 특징을 0 으로 지운다**(그 offer 를 안 받은 나). 수신 부담이 전부
  offer 특징(주행 차이·출발 혼잡·슬롯 적재)에 들어 있으므로 이 대비가 곧
  "받았을 때 vs 안 받았을 때" 다.
"""
from __future__ import annotations

import torch

from ..features.block import block_features
from ..features.candidate import (BUYER_OFFER_DIM, buyer_offer_features,
                                  candidate_features)
from .nets import BuyerNet, from_scaled
from .offer import BUY, REJECT, SPACE, Offer, Response


class Buyer:
    """수신 측 행위자. 공간이면 블록, 시간이면 슬롯이 이 역할을 맡는다."""

    def __init__(self, net: BuyerNet, *, explore: float = 0.0, rng=None):
        self.net = net
        self.explore = float(explore)
        self.rng = rng
        self.trail: list[dict] = []

    def respond(self, mbt, offer: Offer, *, order, rec, t: float,
                records, orders, end_s: float,
                slot_capacity_left: int | None = None) -> Response:
        """offer 하나에 독립 응답한다.

        `slot_capacity_left` 가 0 이하면 **용량이 차서 거절**한다 — 값 판단이 아니라
        물리 제약이라 망을 부르지 않는다([[YR-176]] 이 이 용량을 정본으로 만든다).
        """
        is_time = offer.coord.kind != SPACE

        if slot_capacity_left is not None and slot_capacity_left <= 0:
            return Response(offer=offer, action=REJECT)

        # 공간이면 목적지 블록의 눈으로, 시간이면 출발 블록의 눈으로 본다
        # (시간 이연은 블록이 안 바뀌므로 그 블록이 곧 수신 측이다).
        own = offer.src_block if is_time else offer.coord.block
        if own is None or own not in mbt.blocks:
            return Response(offer=offer, action=REJECT)

        bf = block_features(mbt, own, t, n_cands=1, records=records,
                            orders=orders, end_s=end_s)
        cf = candidate_features(order, rec, t, transfer_count=0, defer_count=0)
        of = buyer_offer_features(is_time=is_time,
                                  route_delta_s=offer.coord.route_delta_s,
                                  src_load=offer.src_load,
                                  slot_load=offer.slot_load)

        row_buy = bf + cf + of
        row_reject = bf + cf + [0.0] * BUYER_OFFER_DIM

        with torch.no_grad():
            cost = self.net(torch.tensor([row_buy, row_reject],
                                         dtype=torch.float32))
        phi_buy, phi_reject = float(cost[0].item()), float(cost[1].item())

        action = BUY if phi_buy <= phi_reject else REJECT
        if self.explore > 0.0 and self.rng is not None and self.rng.random() < self.explore:
            action = BUY if self.rng.random() < 0.5 else REJECT

        self.trail.append({
            "t": t, "doc_key": offer.doc_key, "buyer": offer.buyer_id,
            "action": action, "row_buy": row_buy, "row_reject": row_reject,
            "phi_buy": from_scaled(phi_buy), "phi_reject": from_scaled(phi_reject),
        })
        return Response(offer=offer, action=action,
                        predicted_phi=from_scaled(
                            phi_buy if action == BUY else phi_reject))
