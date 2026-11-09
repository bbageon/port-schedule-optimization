# YR-332-k — 구 PPO 보상 시험의 현재 계약 대조

- **상태**: done
- 상위: [[YR-332]]. 파생: [[YR-332-h]] 확장 회귀 검사, 2026-11-05.
- 보정 축: 신뢰성(reliability). 실패를 숨기거나 보상 공식을 시험에 맞춰 바꾸지 않는다.

## 무엇과 왜

`test_full_yard_retains_waiting_order_and_continues_charging_ppo_reward`가
현재 기본값이 None인 `reward_scale_krw`로 비용을 나누려다 실패한다.
양하 보정 전 `5cde800c`의 격리된 소스에서도 같은 줄·같은 오류가 재현됐다.
현재 PPO 보상 계약과 시험 기대값이 일치하는지 확인해야 전체 회귀 판정을 정리할 수 있다.

## 완료 기준

1. 현재 정규화 방식·기준 통계·누적 보상 정의를 코드와 기존 명세에서 확인한다.
2. 구현과 시험 중 계약에서 벗어난 부분만 고친다. 낡은 분모 상수를 복구하지 않는다.
3. 실제 누적 보상과 기준 계산을 대조하고 관련 PPO 회귀 검사를 수행한다.

## 증거

- `outputs/reports/yr332_discharge_latency/cargo-tests.xml`: 20 통과, 1 실패.
- `outputs/reports/yr332_discharge_latency/baseline-cargo-check.json`: 변경 전 동일 실패.
- `outputs/reports/yr332_discharge_latency/baseline-cargo-test.xml`

## 보정 결과 — 2026-11-09

실행 코드 `5fbe8314`. 원화 비용을 None으로 나누던 오래된 기대값을 현재 운영 보상 계약으로 수정했다.
만차에서 기다린 트럭의 누적 시간은 2,460초이고 다른 세 물리 항목은 0이다.
고정 기준 통계로 나눈 이 시간에 트럭 항 가중치 0.25를 적용한 값이 실제 누적 보상과 일치한다.
100만원 등 임의 분모를 복구하거나 보상 구현을 시험에 맞춰 바꾸지 않았다.

관련 회귀 63건 통과(기존 보상·분기·물리 대기 검사 포함).
이는 구 보상 시험의 오류 수정이며 새 학습 효과를 뜻하지 않는다.
[시험](../../../tests/v7/test_cargo_capacity_wait.py) ·
[검증 기록](../../../outputs/reports/yr332_demand_preservation/preflight.xml).
