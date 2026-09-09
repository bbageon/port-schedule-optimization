"""① 오더 — 코피노 1통으로 확정되는 여섯 가지.

설계 정본: `.claude/docs/architecture/01-오더-스키마.md` §1

**코피노 1건 = 오더 1건 = 컨테이너 1개.** 20피트 2개를 실었으면 오더 2건이다.

■ 여기에 미래가 한 조각도 없다
  전부 **수신 시점에 확정**되는 값이라 정책이 통째로 읽어도 정보 경계가 안 샌다.
  실현 시각은 오더가 아니라 **실행 기록**(record.py)이고, 그건 이벤트가 와야 찬다.

■ v2 에서 버린 것
  `requested_flow`·`fallback_reason` (합성 생성기 인공물 — 실데이터에 없는 개념) ·
  `size_class`(오더 1건 = 컨테이너 1개라 불필요) · `travel_base_s`(레이아웃에서 계산) ·
  `travel_s`·`exit_travel_s`(시뮬레이터 생성 재료지 오더가 아니다) ·
  `schema`(행마다 반복할 이유 없음 — 파일 머리에 한 번).
"""
from __future__ import annotations

from dataclasses import dataclass

#: 파일 머리에 한 번 적는 규격 이름 (행마다 반복하지 않는다)
SCHEMA_VERSION = "v3-order-1"

#: 오더 필드 수 — 계약(`order_fields_target = 6`)
ORDER_FIELDS = 6

INOUT_OUT = 0      # 반출 (야드 → 트럭)
INOUT_IN = 1       # 반입 (트럭 → 야드)


@dataclass(frozen=True)
class Order:
    """코피노 1통으로 확정되는 작업 지시. **정책 전부 가시.**

    `copino_notice_s` 와 `in_out_reserve_s` 는 **함께 들어온다** — 코피노 한 통에
    "지금 접수했고, 이때 들어가겠다" 가 같이 담긴다.
    """

    doc_key: str            # 방문 고유키. **블록명 접두 없음**
    in_out: int             # 반출 0 · 반입 1
    copino_notice_s: float  # 코피노 접수 — 터미널이 알게 된 순간
    in_out_reserve_s: float # 기사가 신고한 반출입 예정 시각
    con_loc: str            # 배정 블록 (기정) — 재배치가 확정되면 여기가 바뀐다
    con_no: str             # 컨테이너 번호 (기정)

    def __post_init__(self) -> None:
        if self.in_out not in (INOUT_OUT, INOUT_IN):
            raise ValueError(f"{self.doc_key}: in_out 은 0(반출)/1(반입) — {self.in_out}")
        if ":" in self.doc_key:
            raise ValueError(f"docKey 에 블록 접두가 붙었다: {self.doc_key}")

    @property
    def is_inbound(self) -> bool:
        """반입인가 — **공간 재배치는 반입만** 가능하다(반출은 컨테이너 위치 고정)."""
        return self.in_out == INOUT_IN


def relocate(order: Order, new_block: str) -> Order:
    """공간 재배치 확정 — `conLoc` 을 새 블록으로. 옛 값은 기록이 보관한다.

    `moveLoc` 은 두지 않는다(2026-08-20 결정) — `conLoc` 이 **항상 현재 블록**을
    가리키고, 바뀔 때 옛 값이 기록의 `prev_con_loc` 으로 간다. 필드 둘이
    "어느 게 지금인지" 를 헷갈리게 만드는 것보다 낫다.
    """
    if not order.is_inbound:
        raise ValueError(f"{order.doc_key}: 반출은 공간 재배치 불가")
    return Order(doc_key=order.doc_key, in_out=order.in_out,
                 copino_notice_s=order.copino_notice_s,
                 in_out_reserve_s=order.in_out_reserve_s,
                 con_loc=new_block, con_no=order.con_no)


def defer(order: Order, new_reserve_s: float) -> Order:
    """시간 이연 확정 — 예약 슬롯을 옮긴다. 블록은 안 바뀐다.

    양은 고정이 아니라 **어느 칸으로 갈까**의 결과다
    ([03 결정층](../../../.claude/docs/architecture/03-결정층.md) §5).
    """
    if new_reserve_s < order.copino_notice_s:
        raise ValueError(f"{order.doc_key}: 이연 목표가 통지보다 이르다")
    return Order(doc_key=order.doc_key, in_out=order.in_out,
                 copino_notice_s=order.copino_notice_s,
                 in_out_reserve_s=new_reserve_s,
                 con_loc=order.con_loc, con_no=order.con_no)
