"""Resolver — **동의된 것만** 모아 매칭한다.

설계 정본: `.claude/docs/architecture/03-결정층.md` §3-1 · §4

    거래 후보 (A, B, j)  ⇔  SELL_A(j)  ∧  BUY_B(A, j)

**동의가 없는 edge 를 새로 만들거나 강제할 수 없다.** 동의된 것들 중에서
용량·소유권·동시오더 제약을 지키며 **매칭만** 한다. 같은 작업을 여럿이 원하면
하나만, 한 Buyer 가 여럿을 원하면 배치 용량 안에서만.

`SELL` 인데 동의 edge 가 없거나 배치에서 안 뽑히면 `RESOLVER_KEEP` 으로 잠근다.

■ 순열 불변
  동점은 `(cost, doc_key, coord)` 사전순으로 깬다. 입력 순서가 결과를 못 바꾼다.

■ 최초 통지 1회
  결정이 끝난 작업은 `decided` 에 들어가 **다시 거래되지 않는다**([[YR-203]]).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .offer import BUY, RESOLVER_KEEP, SPACE, Offer, Response


@dataclass
class Trade:
    """확정된 거래 하나."""

    doc_key: str
    src_block: str
    coord_key: str
    kind: str                 # SPACE | TIME
    dst_block: str | None
    slot: int | None
    route_delta_s: float
    defer_s: float


@dataclass
class ResolveResult:
    trades: list[Trade] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)       # RESOLVER_KEEP 된 doc_key
    consented_edges: int = 0

    @property
    def traded_edges(self) -> int:
        """확정된 거래 수 — **판정 하드가드**가 0 인지 본다(거래 0 이면 학습 신호 없음)."""
        return len(self.trades)


class Resolver:
    """중앙은 매칭만 한다 — 값을 만들지 않는다."""

    def __init__(self, mbt, *, capacity_margin: int = 0,
                 slot_capacity: dict[int, int] | None = None):
        self.mbt = mbt
        self.capacity_margin = int(capacity_margin)
        self.slot_capacity = dict(slot_capacity or {})

    def resolve(self, responses: list[Response], *,
                offers: list[Offer]) -> ResolveResult:
        """동의된 edge 만 모아 batch 로 확정한다."""
        out = ResolveResult()

        consented = [r for r in responses if r.action == BUY]
        out.consented_edges = len(consented)

        # 결정론 — 예측 비용 오름차순, 동점은 사전순
        consented.sort(key=lambda r: (r.predicted_phi, r.offer.sort_key()))

        vcap: dict[str, int] = {}          # 블록별 가상 수신 예약
        slot_used: dict[int, int] = {}
        taken: set[str] = set()            # 이미 확정된 작업

        for r in consented:
            o = r.offer
            if o.doc_key in taken:
                continue                   # 같은 작업을 여럿이 원하면 하나만
            c = o.coord
            if c.kind == SPACE:
                dst = c.block
                used = vcap.get(dst, 0)
                free = self.mbt.free_slots(dst) - used
                if free <= self.capacity_margin:
                    continue               # 물리 용량 초과 = 확정 불가
                vcap[dst] = used + 1
            else:
                k = int(c.slot)
                cap = self.slot_capacity.get(k)
                if cap is not None and slot_used.get(k, 0) >= cap:
                    continue               # 슬롯 용량 초과 (VBS 예약 정본 — YR-176)
                slot_used[k] = slot_used.get(k, 0) + 1

            taken.add(o.doc_key)
            out.trades.append(Trade(
                doc_key=o.doc_key, src_block=o.src_block, coord_key=c.key(),
                kind=c.kind, dst_block=c.block, slot=c.slot,
                route_delta_s=c.route_delta_s, defer_s=c.defer_s))

        # SELL 을 냈는데 확정 안 된 것은 RESOLVER_KEEP 으로 잠근다
        for o in offers:
            if o.doc_key not in taken:
                out.kept.append(o.doc_key)
        return out

    @staticmethod
    def keep_reason() -> str:
        return RESOLVER_KEEP
