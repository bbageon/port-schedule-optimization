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

from ..features.block import (_ARRIVAL_HALF_W_S, announced_around,
                              block_features)
from ..features.candidate import (BUYER_OFFER_DIM, buyer_offer_features,
                                  candidate_features)
from .explore import draw
from .nets import BuyerNet, from_advantage
from .offer import BUY, REJECT, SPACE, Offer, Response


class Buyer:
    """수신 측 행위자. 공간이면 블록, 시간이면 슬롯이 이 역할을 맡는다."""

    def __init__(self, net: BuyerNet, *, explore: float = 0.0, seed: int = 0):
        self.net = net
        self.explore = float(explore)
        #: Seller 와 같은 이유로 좌표 기반이다 — `explore.py` 참조.
        self.seed = int(seed)
        self.trail: list[dict] = []
        #: 반사실 분기용 **1회성 강제 응답** — {docKey: "BUY"|"REJECT"}.
        self.force_once: dict[str, str] = {}
        #: ★절제 실험용 — 켜면 **수신 측 판단을 없앤다**([[YR-254]]).
        #:  용량 검사(물리)는 그대로 살아 있으므로, 이 팔은 상호 동의 중
        #:  **학습된 거절**만 덜어낸 것이지 제약 위반이 아니다.
        self.always_buy = False

    def respond(self, mbt, offer: Offer, *, order, rec, t: float,
                records, orders, end_s: float,
                slot_capacity_left: int | None = None) -> Response:
        """offer 하나에 독립 응답한다.

        `slot_capacity_left` 가 0 이하면 **용량이 차서 거절**한다 — 값 판단이 아니라
        물리 제약이라 망을 부르지 않는다([[YR-176]] 이 이 용량을 정본으로 만든다).
        """
        is_time = offer.coord.kind != SPACE

        if slot_capacity_left is not None and slot_capacity_left <= 0:
            self.force_once.pop(offer.doc_key, None)
            return Response(offer=offer, action=REJECT)   # 용량 = 물리, 강제보다 세다

        # 공간이면 목적지 블록의 눈으로, 시간이면 출발 블록의 눈으로 본다
        # (시간 이연은 블록이 안 바뀌므로 그 블록이 곧 수신 측이다).
        own = offer.src_block if is_time else offer.coord.block
        if own is None or own not in mbt.blocks:
            return Response(offer=offer, action=REJECT)

        bf = block_features(mbt, own, t, n_cands=None, records=records,
                            orders=orders, end_s=end_s)   # ★후보수 없음
        cf = candidate_features(order, rec, t)
        # ★[[YR-235]] A8 — Seller 와 **같은 잣대로 같은 시각**을 본다.
        #   시간 이연이면 도착이 밀리고, 공간 이동이면 시각은 그대로 블록만 바뀐다.
        arrive_s = order.in_out_reserve_s + (offer.coord.defer_s if is_time else 0.0)
        pressure = float(announced_around(mbt, own, arrive_s, orders,
                                          half_w=_ARRIVAL_HALF_W_S))
        of = buyer_offer_features(is_time=is_time,
                                  route_delta_s=offer.coord.route_delta_s,
                                  src_load=offer.src_load,
                                  slot_load=offer.slot_load,
                                  arrival_pressure=pressure)

        row_buy = bf + cf + of
        row_reject = bf + cf + [0.0] * BUYER_OFFER_DIM

        with torch.no_grad():
            cost = self.net(torch.tensor([row_buy, row_reject],
                                         dtype=torch.float32))
        phi_buy, phi_reject = float(cost[0].item()), float(cost[1].item())

        forced = self.force_once.pop(offer.doc_key, None)
        if forced in (BUY, REJECT):
            action = forced                    # 반사실: 반대로 응답했다면
        elif self.always_buy:
            action = BUY                       # ★절제 — 제안만으로 확정
        else:
            action = BUY if phi_buy <= phi_reject else REJECT
            dk = offer.doc_key
            if self.explore > 0.0 and draw(self.seed, dk, t, "buy:on") < self.explore:
                action = BUY if draw(self.seed, dk, t, "buy:which") < 0.5 else REJECT

        self.trail.append({
            "t": t, "doc_key": offer.doc_key, "buyer": offer.buyer_id,
            "action": action, "row_buy": row_buy, "row_reject": row_reject,
            "adv_buy": from_advantage(phi_buy),
            "adv_reject": from_advantage(phi_reject),
        })
        return Response(offer=offer, action=action,
                        predicted_phi=from_advantage(
                            phi_buy if action == BUY else phi_reject))
