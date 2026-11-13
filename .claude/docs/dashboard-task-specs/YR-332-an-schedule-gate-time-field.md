# YR-332-an — 보정된 도착 시각과 명단 보조 필드(`gate_in_s`)의 불일치

- **상태**: backlog
- 등록: 2026-11-13. 상위 [[YR-332]], 발견 작업 [[YR-332-ak]] 코드 대조.
- 잠재 기록 결함 한 축. 현재 결과를 바꾸지 않는다는 점을 먼저 확인한 뒤 고친다.

## 무엇과 왜

하루 명단을 만들 때 `order_schema.attach`가 BNCT 형식 보조 필드 `gate_in_s`·`reserved_s`를
그 시점의 `arrival_s`로 채운다. 그 뒤 aj의 시간대 보정과 ak의 시간대 안 보정이 `arrival_s`만 바꾼다.
`reserved_s`는 예약 계약이 다시 쓰지만, 명단의 `gate_in_s`는 보정 전 시각으로 남는다.

엔진은 `arrival_s`만 읽고(`orders.py`의 `V3Announcer.review`), 실현된 진입 시각은 기록 장부의 별도 값이다.
따라서 aj·ak 결과에는 영향이 없다. 다만 입력 지문에는 이 필드가 들어가고, 이 필드를 읽는 미래 도구는 틀린 시각을 받는다.

## 완료 기준

1. 이 필드를 읽는 모든 경로를 나열한다(현재 v7 엔진·평가 경로에는 없음).
2. 보정 뒤 보조 필드를 다시 맞추는 수정과 회귀 시험을 추가한다. aj·ak 저장 입력의 재현 방법을 함께 남긴다.
3. 결과·사건 해시가 바뀌지 않음을 확인한다. 입력 지문이 바뀌는 범위는 따로 보고한다.

근거: [ak 보고서](../../../outputs/reports/yr332_visit_arrivals/README.md) · `src/yard_rl/v7/world/integrated/order_schema.py:bnct_view`.
