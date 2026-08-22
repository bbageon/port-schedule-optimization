"""v3 ② 보상 — 반사실 교사 — 안 했을 때의 세계를 실제로 굴린다.

■ 담는 것
  - 그 거래만 취소한 세계를 같은 시드로 H=1시간 굴려 총비용 차이를 산출
  - `R_src`(판 쪽 절감) · `B_dst`(산 쪽 부담)를 **따로** 낸다 → actors 의 두 망
  - 원화 환산 목적함수 (안전운임 40,000원/트럭·시간)

■ 왜 바꾸나
  현행 `realized_credit` 은 "밀렸을 것이다"라는 **추정**이다. 반사실은 두 세계를
  실제로 굴려 빼므로 추측이 없고, 재조작·크레인 이동의 간접 효과가
  **1시간 안이면 저절로 잡힌다**.

■ 하지 말 것
  **배포·판정 경로에서 부르지 않는다.** 학습 전용이다. 판정 실행에서
  rollout 호출 수는 0이어야 한다 (하드가드).
■ 설계 문서: `.claude/docs/architecture/04b-학습-잣대.md`
"""

from .counterfactual import (ActorLabel, CounterfactualTeacher, TeacherResult,
                             reset_rollout_calls, rollout_calls)
from .krw import (COST_TERMS, KRW_OVERTIME_START_S, KRW_TRUCK_HOUR,
                  VESSEL_CLASSES, VESSEL_STS_STREAMS, rehandle_krw,
                  truck_wait_krw, vessel_idle_krw, vessel_krw_per_hour,
                  vessel_rho, yc_move_krw)
from .phi import PhiBreakdown, terminal_cost_krw

__all__ = ["terminal_cost_krw", "PhiBreakdown", "CounterfactualTeacher",
           "ActorLabel", "TeacherResult", "rollout_calls",
           "reset_rollout_calls", "KRW_TRUCK_HOUR", "KRW_OVERTIME_START_S",
           "COST_TERMS", "VESSEL_CLASSES", "VESSEL_STS_STREAMS",
           "truck_wait_krw", "yc_move_krw", "rehandle_krw", "vessel_idle_krw",
           "vessel_krw_per_hour", "vessel_rho"]
