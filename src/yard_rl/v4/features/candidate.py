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
CANDIDATE_DIM = 3

#: Seller 행동 특징 차원 — KEEP / 공간(블록) / 시간(슬롯) 을 한 벌로 표현
SELLER_ACTION_DIM = 9

#: Buyer 가 offer 에서 읽는 특징 차원
BUYER_OFFER_DIM = 5

#: Seller · Buyer 입력 행 차원 (block 9 + 후보 6 + 행동/offer)
from .block import BLOCK_DIM, BLOCK_DIM_BUYER  # noqa: E402

SELLER_ROW_DIM = BLOCK_DIM + CANDIDATE_DIM + SELLER_ACTION_DIM   # 21
BUYER_ROW_DIM = BLOCK_DIM_BUYER + CANDIDATE_DIM + BUYER_OFFER_DIM   # 16

_WINDOW_REF_S = 1800.0     # 정규화 기준 (창 길이가 바뀌어도 눈금은 고정)
_ROUTE_REF_S = 600.0
_DEFER_REF_S = 3600.0


def candidate_features(order, rec, t: float) -> list[float]:
    """후보 3차원 — 그 작업이 어떤 물건인가. 공개 정보만.

    ★[[YR-235]] (2026-08-26) — 6칸에서 **3칸으로 줄였다**. 동적 감사가 잡았다:

      · `게이트통과` — 자격 조건이 *"게이트를 안 지났다"* 라 **구조상 항상 0**
      · `이동횟수`·`이연횟수` — **재결정이 한 번뿐**이므로(사용자 확인 2026-08-26)
        결정 시점에 그 오더는 아직 한 번도 안 옮겨졌다. **구조상 항상 0**

    셋 다 값이 맞는데 **정보가 없다.** 살아 있는 칸이 적은 편이 죽은 칸이 많은
    것보다 낫다(설계원칙 2 — 핵심 정보 우선).
    """
    eta_remain = order.in_out_reserve_s - t          # 통지된 예정까지 남은 시간
    lead = order.in_out_reserve_s - order.copino_notice_s
    return [
        max(0.0, min(1.0, eta_remain / _WINDOW_REF_S)),
        0.0 if order.is_inbound else 1.0,            # is_out
        max(0.0, min(2.0, lead / _DEFER_REF_S)),     # 얼마나 미리 알았나
    ]


def seller_action_features(*, kind: str, dst_load: float = 0.0,
                           dst_free: float = 0.0, route_delta_s: float = 0.0,
                           defer_s: float = 0.0, dst_quay_s: float = 0.0,
                           arrival_pressure: float = 0.0) -> list[float]:
    """Seller 행동 9차원 — `KEEP` / `SPACE(블록 b)` / `TIME(슬롯 k)`.

    ★`arrival_pressure` ([[YR-230]]) — **이 행동을 고르면 트럭이 도착할 시각·블록에
    이미 몇 대가 예약돼 있나.** 이게 없으면 정책은 *"몇 분 미루나"* 만 알 뿐
    **미룬 곳이 붐빌지를 모른다.** 세 행동 모두에 값이 있어야 서로 견줄 수 있다:

        KEEP    예정대로 도착 → `in_out_reserve_s` 시각·현재 블록
        SPACE   시각은 그대로 → `in_out_reserve_s` 시각·**목적 블록**
        TIME    블록은 그대로 → `in_out_reserve_s + defer_s` 시각·현재 블록

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
        arrival_pressure / 10.0,          # ★[[YR-230]] — 도착 시각·블록의 예약 밀도
    ]


def buyer_offer_features(*, is_time: bool, route_delta_s: float,
                         src_load: float, slot_load: float,
                         arrival_pressure: float = 0.0) -> list[float]:
    """Buyer 가 offer 에서 읽는 5차원 — **Seller 가 공개한 것만**.

    ★`arrival_pressure` ([[YR-235]] A8 · 2026-08-26) — **Seller 와 같은 잣대로
    같은 시각을 보게 한다.** 전에는 Seller 만 제안 시각을 ±30분으로 보고
    (도착 압력, [[YR-230]]) Buyer 는 ±7.5분(`slot_load`)만 봤다. 같은 제안을
    다른 잣대로 판단하면 거부권이 엇나간다.

    이 칸이 *"몇 분 미루는 제안인가"* 도 함께 담는다 — 도착 시각이 곧 중심이라
    `defer_s` 를 따로 안 줘도 된다.

    Seller 의 내부 사정(자기 블록이 얼마나 급한지)은 안 준다. 공개 message 는
    "어떤 작업을, 어디서, 어느 좌표로" 뿐이다.
    """
    return [
        1.0 if is_time else 0.0,
        route_delta_s / _ROUTE_REF_S,
        src_load / 10.0,
        slot_load / 10.0,
        arrival_pressure / 10.0,      # ★[[YR-235]] A8 — Seller 와 같은 ±30분
    ]
