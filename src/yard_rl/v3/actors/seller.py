"""Seller — 자기 블록만 보고 KEEP / SELL 을 **스스로** 고른다.

설계 정본: `.claude/docs/architecture/03-결정층.md` §3-1 · §4

■ v2 와 무엇이 다른가
  v2 는 중앙 배정기가 다 정했고 Seller 는 없었다. v3 의 Seller 는 **행위자**다 —
  자기가 고르고, 그 선택의 반사실로 배운다.

■ 자격은 게이트 진입 전까지다
  `gate_in` 이 찍혔으면 이미 그 블록을 향해 달리고 있어 목적지를 못 바꾼다.
  판단은 공개 ETA 가 창에 **처음 들어온** epoch 에서 **한 번**뿐이다([[YR-203]]).

■ 정보 경계
  자기 블록 상태 + 자기 후보 작업만. 실현 미래값은 한 줄도 안 읽는다.
"""
from __future__ import annotations

import torch

from ..features.block import block_features, inside_count
from ..features.candidate import (candidate_features, seller_action_features)
from .explore import draw, pick
from .nets import SellerNet, from_scaled
from .offer import KEEP, SELL, SPACE, TIME, Coord, Offer


class Seller:
    """블록 하나의 매도 행위자. 가중치는 21블록이 **1벌을 공유**한다."""

    def __init__(self, net: SellerNet, layout, *, explore: float = 0.0,
                 seed: int = 0):
        self.net = net
        self.layout = layout
        self.explore = float(explore)
        #: 탐색 난수는 **좌표로 뽑는다**(`explore.py`) — 순차 난수를 쓰면 분기
        #: 세계가 실제 궤적과 다른 탐색을 해 동일성 불변식이 깨진다.
        self.seed = int(seed)
        self.trail: list[dict] = []
        #: 반사실 분기용 **1회성 강제 행동** — {docKey: "KEEP"|"SELL"}.
        #: 교사가 "이 행위자가 반대로 했다면" 세계를 만들 때만 채운다. 평소엔 비어 있다.
        self.force_once: dict[str, str] = {}

    # ------------------------------------------------------------------ 후보 좌표
    def _space_coords(self, mbt, src: str, quay_of) -> list[Coord]:
        out = []
        for dst in sorted(b for b in mbt.blocks if b != src):
            if mbt.free_slots(dst) <= 0:
                continue                       # 물리적으로 불가능 = 후보에서 제거
            out.append(Coord(kind=SPACE, block=dst,
                             route_delta_s=self.layout.pre_gate_route_delta_s(src, dst)))
        return out

    @staticmethod
    def _time_coords(slots) -> list[Coord]:
        """이연 후보 슬롯 — **공개 예측으로 미리 좁힌 것**만 들어온다.

        값 판단이 아니라 실현 가능성 필터라 규칙 계산이 정당하다(03 §6 과 같은 성격).
        """
        return [Coord(kind=TIME, slot=k, slot_start_s=start, defer_s=defer)
                for k, start, defer in slots]

    # ------------------------------------------------------------------ 결정
    def decide(self, mbt, src: str, *, doc_key: str, order, rec, t: float,
               records, orders, end_s: float, n_cands: int,
               time_slots=(), quay_of=None, transfer_count: int = 0,
               defer_count: int = 0) -> Offer | None:
        """KEEP 이면 `None`, SELL 이면 방송할 `Offer` 를 돌려준다."""
        if rec.gate_in_s is not None:
            return None                        # 게이트를 지났다 = 자격 없음

        forced = self.force_once.pop(doc_key, None)
        if forced == KEEP:
            return None                        # 반사실: 안 팔았다면

        bf = block_features(mbt, src, t, n_cands=n_cands, records=records,
                            orders=orders, end_s=end_s)
        cf = candidate_features(order, rec, t, transfer_count=transfer_count,
                                defer_count=defer_count)

        coords: list[Coord | None] = [None]     # None = KEEP
        if order.is_inbound:
            coords += self._space_coords(mbt, src, quay_of)
        coords += self._time_coords(time_slots)

        rows, meta = [], []
        for c in coords:
            if c is None:
                af = seller_action_features(kind=KEEP)
            elif c.kind == SPACE:
                af = seller_action_features(
                    kind=SPACE,
                    dst_load=float(inside_count(mbt, c.block, t, records)),
                    dst_free=float(mbt.free_slots(c.block)),
                    route_delta_s=c.route_delta_s,
                    dst_quay_s=(quay_of(c.block) if quay_of else 0.0))
            else:
                af = seller_action_features(kind=TIME, defer_s=c.defer_s)
            rows.append(bf + cf + af)
            meta.append(c)

        x = torch.tensor(rows, dtype=torch.float32)
        with torch.no_grad():
            cost = self.net(x)
        if forced == SELL and len(meta) > 1:
            # 반사실: 팔았다면 — KEEP(0번 행)을 빼고 **가장 싼 좌표**를 고른다.
            idx = 1 + int(torch.argmin(cost[1:]).item())
        else:
            idx = int(torch.argmin(cost).item())

        if forced is None and self.explore > 0.0:
            if draw(self.seed, doc_key, t, "sell:on") < self.explore:
                idx = pick(self.seed, doc_key, t, "sell:which", len(meta))

        chosen = meta[idx]
        self.trail.append({
            "t": t, "doc_key": doc_key, "src": src,
            "action": KEEP if chosen is None else SELL,
            "coord": None if chosen is None else chosen.key(),
            # ★후보 좌표 목록 — 반사실 대안 세계가 고른 좌표를 **행 번호로** 되찾는 데 쓴다.
            #   교사는 굴린 두 행에만 라벨을 붙이므로(04b §3) 어느 행인지 알아야 한다.
            "coord_keys": [None if c is None else c.key() for c in meta],
            "rows": x, "picked": idx,
            "predicted_phi": from_scaled(float(cost[idx].item())),
        })
        if chosen is None:
            return None
        return Offer(doc_key=doc_key, src_block=src, coord=chosen,
                     src_load=float(inside_count(mbt, src, t, records)))
