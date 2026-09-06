"""행위자 사이에 오가는 것 — offer 와 응답.

설계 정본: `.claude/docs/architecture/03-결정층.md` §3-1

    Seller A :  a ∈ {KEEP, SELL(공간 b), SELL(시간 슬롯 k)}
    Buyer    :  a ∈ {REJECT, BUY}        공간 → 목적지 블록 · 시간 → 목표 슬롯
    거래 후보 (A, B, j)  ⇔  SELL_A(j) ∧ BUY_B(A, j)

**Resolver 는 동의 없는 edge 를 만들거나 강제할 수 없다.**

■ offer message 는 공개된 것만 담는다
  어떤 작업을, 어디서, 어느 좌표로. Seller 의 내부 사정(자기 블록이 얼마나 급한지)
  은 안 담는다 — Buyer 가 그걸 보면 정보 경계가 비대칭이 된다.
"""
from __future__ import annotations

from dataclasses import dataclass

KEEP = "KEEP"
SELL = "SELL"
REJECT = "REJECT"
BUY = "BUY"

SPACE = "SPACE"     # 좌표: 다른 블록
TIME = "TIME"       # 좌표: 같은 블록, 다른 예약 슬롯

#: Resolver 가 동의 edge 를 못 찾았을 때 찍는 사유
RESOLVER_KEEP = "RESOLVER_KEEP"


@dataclass(frozen=True)
class Coord:
    """좌표 하나 — 공간이면 목적지 블록, 시간이면 목표 슬롯."""

    kind: str                      # SPACE | TIME
    block: str | None = None       # SPACE 일 때 목적지
    slot: int | None = None        # TIME 일 때 절대 칸 번호
    slot_start_s: float = 0.0      # TIME 일 때 그 칸의 시작 시각
    route_delta_s: float = 0.0     # 주행 차이 (SPACE) — 음수 가능
    defer_s: float = 0.0           # 이연량 (TIME)

    def key(self) -> str:
        """결정론 정렬용 — 동점은 사전순으로 깬다."""
        return f"{self.kind}@{self.block if self.kind == SPACE else self.slot}"


@dataclass(frozen=True)
class Offer:
    """Seller 가 방송하는 것. **공개 message 만** 담는다."""

    doc_key: str
    src_block: str
    coord: Coord
    src_load: float = 0.0          # 출발 블록 혼잡도 (공개 관측치)
    slot_load: float = 0.0         # TIME 일 때 그 칸의 예약 적재율

    @property
    def buyer_id(self) -> str:
        """이 offer 를 받는 쪽 — 공간이면 목적지 블록, 시간이면 목표 슬롯."""
        return (self.coord.block if self.coord.kind == SPACE
                else f"SLOT@{self.coord.slot}")

    def sort_key(self) -> tuple:
        return (self.doc_key, self.src_block, self.coord.key())


@dataclass(frozen=True)
class Response:
    """Buyer 의 독립 응답. 다른 Buyer 의 응답은 못 보고 낸 값이다."""

    offer: Offer
    action: str                    # BUY | REJECT
    predicted_phi: float = 0.0     # 학생 망이 낸 값 (진단·정렬용)

    @property
    def consented(self) -> bool:
        return self.action == BUY
