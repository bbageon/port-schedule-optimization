"""v3 ① 데이터 — 오더 레코드 규격 — 시각 셋을 가른다.

■ 담는 것
  데이터는 **둘**이다 — 언제 생기느냐가 다르다.

  ① 오더 (코피노 접수 1회 · 정책 전부 가시) — 6필드
      docKey · inOut(반출0/반입1) · copinoNoticeTime · inOutReserveTime
      · conLoc · conNo
  ② 실행 기록 (일어난 뒤에 생김 · 지난 것만 가시) — 7필드
      터미널이 전송: gateIn · blockIn · jobDone · gateOut
      터미널이 안 보냄(시뮬레이터만 앎): serviceStart   ← 실데이터엔 없다
      재배치 시 갱신: prevConLoc · conSwapReason

  ★네 이벤트는 **터미널이 실제로 전송해 주는 값**이다(사용자 지적 2026-08-20).
  "숨겨둔 미래"가 아니다 — 정책이 못 보는 이유는 **아직 안 왔기 때문**이다.
  시뮬레이터가 그 이벤트를 만들어내는 재료(도착 시각·주행 소요)는 데이터 층이
  아니라 **구현 사항**이며 레코드에 노출하지 않는다.

■ 지금 구현과 다른 점
  현행은 통지·예정·실현 **셋이 같은 값**(`arrival_s`)이라 예약 준수·리드타임
  축이 소멸해 있다. 여기서 가른다.

■ 하지 말 것
  파생값(turnTime·리드타임·예약 준수 오차)을 **저장하지 않는다** — 시각에서
  전부 유도된다. 중복 저장은 불일치의 원인.
■ 설계 문서: `.claude/docs/architecture/01-오더-스키마.md`
"""

from .lifecycle import (LIFECYCLE_STAGES, LifecycleError, Stage, adherence_error_s,
                        censored_turn_time_s, lead_time_s, reached, turn_time_s,
                        validate)
from .order import (INOUT_IN, INOUT_OUT, ORDER_FIELDS, SCHEMA_VERSION, Order,
                    defer, relocate)
from .record import (RECORD_FIELDS, SWAP_SPACE, SWAP_TIME, TRANSMITTED_FIELDS,
                     ExecutionRecord)

__all__ = ["Order", "ExecutionRecord", "Stage", "LifecycleError", "validate",
           "turn_time_s", "lead_time_s", "adherence_error_s",
           "censored_turn_time_s", "reached", "relocate", "defer",
           "INOUT_IN", "INOUT_OUT", "SWAP_SPACE", "SWAP_TIME",
           "ORDER_FIELDS", "RECORD_FIELDS", "TRANSMITTED_FIELDS",
           "LIFECYCLE_STAGES", "SCHEMA_VERSION"]
