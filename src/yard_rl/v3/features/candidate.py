"""후보(작업) 특징과 행동 특징 — 행위자마다 입력이 다르다.

설계 정본: `.claude/docs/architecture/03-결정층.md` §3-1 · `05-정보경계.md`

■ 누가 무엇을 보나
  | 행위자 | 볼 수 있는 것 |
  |---|---|
  | Seller | 자기 블록 상태 + 자기 후보 작업 |
  | Buyer  | 자기 블록 상태 + **Seller 가 공개한 offer message** |

  Buyer 는 다른 Buyer 의 응답을 못 본다(동시 결정이라 구조적으로 불가) —
  경쟁 입찰이 아니라 각자 자기 값만 보고 응답한다.

■ 전부 공개 정보
  통지된 예정 시각·현재 상태·지도 거리·이력만. 실현 미래값(`gate_in_s` 가 아직
  안 찍힌 트럭의 진짜 도착 등)은 **한 줄도 안 읽는다.**
"""
from __future__ import annotations

#: 후보(작업) 특징 차원
CANDIDATE_DIM = 6

#: Seller 행동 특징 차원 — KEEP / 공간(블록) / 시간(슬롯) 을 한 벌로 표현
SELLER_ACTION_DIM = 8

#: Buyer 가 offer 에서 읽는 특징 차원
BUYER_OFFER_DIM = 4

#: Seller · Buyer 입력 행 차원 (block 9 + 후보 6 + 행동/offer)
from .block import BLOCK_DIM  # noqa: E402

SELLER_ROW_DIM = BLOCK_DIM + CANDIDATE_DIM + SELLER_ACTION_DIM   # 23
BUYER_ROW_DIM = BLOCK_DIM + CANDIDATE_DIM + BUYER_OFFER_DIM      # 19

_WINDOW_REF_S = 1800.0     # 정규화 기준 (창 길이가 바뀌어도 눈금은 고정)
_ROUTE_REF_S = 600.0
_DEFER_REF_S = 3600.0


def candidate_features(order, rec, t: float, *, transfer_count: int,
                       defer_count: int) -> list[float]:
    """후보 6차원 — 그 작업이 어떤 물건인가. 공개 정보만."""
    eta_remain = order.in_out_reserve_s - t          # 통지된 예정까지 남은 시간
    lead = order.in_out_reserve_s - order.copino_notice_s
    return [
        max(0.0, min(1.0, eta_remain / _WINDOW_REF_S)),
        0.0 if order.is_inbound else 1.0,            # is_out
        max(0.0, min(2.0, lead / _DEFER_REF_S)),     # 얼마나 미리 알았나
        1.0 if rec.gate_in_s is not None else 0.0,   # 이미 게이트를 지났나
        float(transfer_count),
        float(defer_count),
    ]


def seller_action_features(*, kind: str, dst_load: float = 0.0,
                           dst_free: float = 0.0, route_delta_s: float = 0.0,
                           defer_s: float = 0.0, dst_quay_s: float = 0.0,
                           slot_load: float = 0.0) -> list[float]:
    """Seller 행동 8차원 — `KEEP` / `SPACE(블록 b)` / `TIME(슬롯 k)`.

    `dst_quay_s` 는 목적지 블록의 **안벽까지 거리**다. 본선 스트림이 붙은 블록은
    YT 왕복이 길어 처리율이 깎이므로, 목적지 선택에 이 값이 필요하다
    ([02b 본선](../../../../.claude/docs/architecture/02b-본선.md)).
    """
    if kind not in ("KEEP", "SPACE", "TIME"):
        raise ValueError(f"알 수 없는 Seller 행동: {kind!r}")
    return [
        1.0 if kind == "KEEP" else 0.0,
        1.0 if kind == "SPACE" else 0.0,
        1.0 if kind == "TIME" else 0.0,
        dst_load / 10.0,
        dst_free / 1000.0,
        route_delta_s / _ROUTE_REF_S,
        defer_s / _DEFER_REF_S,
        dst_quay_s / _ROUTE_REF_S,
    ]


def buyer_offer_features(*, is_time: bool, route_delta_s: float,
                         src_load: float, slot_load: float) -> list[float]:
    """Buyer 가 offer 에서 읽는 4차원 — **Seller 가 공개한 것만**.

    Seller 의 내부 사정(자기 블록이 얼마나 급한지)은 안 준다. 공개 message 는
    "어떤 작업을, 어디서, 어느 좌표로" 뿐이다.
    """
    return [
        1.0 if is_time else 0.0,
        route_delta_s / _ROUTE_REF_S,
        src_load / 10.0,
        slot_load / 10.0,
    ]
