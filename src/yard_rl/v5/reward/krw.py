"""원화 단가 — 하나만 법정 근거가 있다.

설계 정본: `.claude/docs/architecture/04-비용과-보상.md` §1-1 · [[YR-207]]

■ 트럭만 인용 가능한 법정값이다
  2026년 화물자동차 안전운임 고시(국토교통부 제2026-402호, 2026-08-01 시행)
  별표 1 제24호 — 항만부두 대기가 1시간을 초과하면 30분당 20,000원.
      v_truck = 20,000원 / 0.5시간 = 40,000원/(트럭·시간), VAT 별도

  **법정 청구서 재현이 아니다.** 첫 1시간 면제·30분 반올림을 따르지 않고
  **시간가치만 가져와** 연속 목적함수로 쓴다. 법정 재현값은 진단 열로만 둔다.

■ 나머지는 동결 대상 설계 파라미터다
  판정 시작 전에 동결하고 런 중 조정 금지 — 결과를 보고 바꾸면 판정이 무효다.
"""
from __future__ import annotations

#: 트럭 시간가치 (원/트럭·시간) — 2026 안전운임 고시. **인용 가능**
KRW_TRUCK_HOUR = 40_000.0

#: 초과 할증 (원/트럭·시간) — 기본과 동률이라 문턱을 넘으면 **한계요율 2배**.
#: 고시가 1시간 초과분부터 청구한다는 것을 "그 지점부터 실제 비용" 으로 읽었다.
KRW_TRUCK_OVER_HOUR = 40_000.0

#: 초과 할증이 시작되는 지점 (초) — 고시의 첫 1시간.
#: ⚠️ 무대의 트럭 SLA(`long_wait_sla_s = 1800`)와 별개 값이다. [[YR-207]] 에서 정리한다.
KRW_OVERTIME_START_S = 3600.0

#: YC 추가 이동 (원/시간) — 전력·정비·운영 환산
KRW_YC_MOVE_HOUR = 60_000.0

#: 재조작 (원/회) — 추가 lift 2분 × YC 분당 단가
KRW_REHANDLE_EACH = 2_000.0

#: 본선 유휴 (원/GT·시간) — 선박 톤수에 비례한다
KRW_VESSEL_GT_HOUR = 2.99

#: 선급 3종 — (이름, GT, TEU). [02b 본선](../../../../.claude/docs/architecture/02b-본선.md)
VESSEL_CLASSES: tuple[tuple[str, int, int], ...] = (
    ("SMALL", 50_000, 3_000),
    ("MEDIUM", 100_000, 7_500),
    ("LARGE", 150_000, 14_000),
)

#: 선급별 STS 스트림 수 (실측 참조) — 스트림당 25~30 moves/h
VESSEL_STS_STREAMS: dict[str, int] = {"SMALL": 2, "MEDIUM": 4, "LARGE": 6}

#: 비용 항 수 — 계약(`cost_terms_target = 4`)
COST_TERMS = 4


def vessel_krw_per_hour(gt: float) -> float:
    """선박 유휴 1시간의 원화 비용. 대형선이 소형선의 3배다."""
    return KRW_VESSEL_GT_HOUR * float(gt)


def vessel_rho(gt: float) -> float:
    """트럭 대비 가중치 — 구 단일 `RHO_VESSEL_V2 = 10.0` 이 3.74~11.21 로 갈린다."""
    return vessel_krw_per_hour(gt) / KRW_TRUCK_HOUR


def truck_wait_krw(turn_time_s: float | None) -> float:
    """항 1 — 트럭 대기. **턴타임(O−A) 기준**이고 1시간 초과분은 할증한다.

    선형이면 재배치 이득이 정의상 0 에 가까워진다(대기를 없애는 게 아니라 남에게
    넘기는 것뿐). 꺾이는 지점이 고시와 일치하므로 볼록성이 **법정 요금 구조 그
    자체**다 — 04 §1-3.
    """
    if turn_time_s is None:
        return 0.0
    tt = max(0.0, float(turn_time_s))
    base = KRW_TRUCK_HOUR * tt / 3600.0
    over = KRW_TRUCK_OVER_HOUR * max(0.0, tt - KRW_OVERTIME_START_S) / 3600.0
    return base + over


def yc_move_krw(extra_move_s: float) -> float:
    """항 2 — YC **추가** 이동. 생산 사이클은 세지 않는다.

    서비스 시간까지 넣으면 "아무것도 안 하기" 가 최적이 된다.
    """
    return KRW_YC_MOVE_HOUR * max(0.0, float(extra_move_s)) / 3600.0


def rehandle_krw(n: int) -> float:
    """항 3 — 재조작 건당. **야드 배치 품질이 목적함수에 들어오는 유일한 통로**다."""
    return KRW_REHANDLE_EACH * max(0, int(n))


def vessel_idle_krw(gt: float, idle_s: float) -> float:
    """항 4 — 본선 유휴. `T_idle` 은 **선박 유휴**(붙은 STS 가 전부 멈춘 시간)다.

    2.99원/GT·시간이 선박 단위 요율이라 "STS 1대 멈춤" 과 "배가 멈춤" 이 다르다.
    """
    return vessel_krw_per_hour(gt) * max(0.0, float(idle_s)) / 3600.0
