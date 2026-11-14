# YR-332-an — 보정된 도착 시각과 명단 보조 필드(`gate_in_s`)의 불일치

- **상태**: done
- 등록: 2026-11-13. 상위 [[YR-332]], 발견 작업 [[YR-332-ak]] 코드 대조.
- 잠재 기록 결함 한 축. 현재 결과를 바꾸지 않는다는 점을 먼저 확인한 뒤 고친다.
- **3대 게이트 보정 대상**: scenario_validity(만든 입력의 기록·정보 계약이 실제 실행 값과 맞는가).

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

## 착수 — 2026-11-14

- 사용자 지시 “다음 사항 진행해볼래? 모든 사항 진행하고 학습 다시 해서 감소율 알려줘”로 착수했다.
- 착수 허가: ak의 `c7dc4e25` 저장 판정(신뢰성 PASS·현실성 FAIL·성능 INCONCLUSIVE)에서 scenario_validity 보정만 허용됨을 확인했다.
  [허가 기록](../../../outputs/reports/yr332_bnct_time_field/authorization.json).
- 순서: 읽는 경로 나열 → 보정 뒤 다시 맞추기와 회귀 시험 → 저장 입력 재현 방법 → 결과·사건 해시 불변 확인.

## 완료 — 2026-11-14

- 수정: `build_stage`가 모든 도착 보정 뒤 `gate_in_s`를 `arrival_s`로 다시 맞춘다(`sync_realized_gate_in`). `reserved_s`는 예약 계약 소관이라 그대로 둔다.
- 재현 방법: `build_stage(..., bnct_time_sync=False)`가 aj·ak 저장 입력 지문 9개를 그대로 만든다.
- 읽는 경로: v7 코드·스크립트의 `gate_in_s` 접근 92곳은 모두 실행 기록·GPU 배열·실자료 행이다. 명단 보조 필드를 읽는 엔진·평가 경로는 없다.
- 결과 불변: 저장된 ak 실행 4건과 보조 필드에 엉뚱한 값을 넣은 1건을 재실행했다. 결과·트럭 행·후보·선택·사건·작업·기록 해시가 모두 같다.
- 입력 지문: 보정 없는 ag 입력 9개는 바이트 동일. 보정 입력 9개(78,000건)는 `gate_in_s`만 바뀌어 명단·입력 지문이 달라진다. 시험 92건 통과.

결과 커밋 `31100a15`. [보고서](../../../outputs/reports/yr332_bnct_time_field/README.md) · [검증](../../../outputs/reports/yr332_bnct_time_field/verification.json)
