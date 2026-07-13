# Exp-1 외부트럭 Direct-Job Cost-Q 결과

> ⚠ **assumed 프로파일 + 합성 시나리오** 결과다. 실측 도착·서비스 자료와
> CURRENT_RULE 검증 전이므로 실제 부산항 운영 대비 개선율로 해석할 수 없다.

## 실험 정의

- 대상: 외부트럭-only, 선박 작업 제외(`n_vessel=0`).
- 의사결정 시점: 실제 블록 진입인 `BLOCK_ENTRY`.
- 정책: 직접 feasible job 선택형 Cost-Q(`argmin`); `SLA_OFF`가 primary, `SLA_ON`이 secondary arm.
- State: `['operation_phase', 'queue_length_bucket']`; candidate: `['transfer_direction', 'estimated_service_time_bucket', 'end_crane_zone']`.
- 실행 모드: `full`; manifest n_vessel: `0`.

## Validation 선택

- `SLA_OFF`: Cost-Q `p=1.0`, checkpoint `300` (validation mean 8.72 min); selected baseline `SHORTEST_ESTIMATED_SERVICE_TIME`.
- `SLA_ON`: Cost-Q `p=1.0`, checkpoint `50` (validation mean 11.13 min); selected baseline `SHORTEST_ESTIMATED_SERVICE_TIME`.

## Test 요약 (공통 test seed)

| Arm | Policy | Mean wait (min) | P50 (min) | P95 (min) | SLA 초과율 | Completion | Backlog | Fallback |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SLA_OFF | CostQ+GreedyFallback | 8.98 | 3.15 | 44.69 | 6.2% | 100.0% | 0.00 | 0.0% |
| SLA_OFF | FIFO | 13.78 | 11.29 | 34.92 | 15.7% | 100.0% | 0.00 | 0.0% |
| SLA_OFF | IMMEDIATE_COST_GREEDY | 7.79 | 2.95 | 30.28 | 4.5% | 100.0% | 0.00 | 0.0% |
| SLA_OFF | LONGEST_WAIT | 13.78 | 11.29 | 34.92 | 15.7% | 100.0% | 0.00 | 0.0% |
| SLA_OFF | MIN_BLOCKER | 8.03 | 3.67 | 32.64 | 4.8% | 100.0% | 0.00 | 0.0% |
| SLA_OFF | NEAREST_JOB | 11.28 | 3.39 | 52.29 | 9.1% | 100.0% | 0.00 | 0.0% |
| SLA_OFF | SHORTEST_ESTIMATED_SERVICE_TIME | 7.79 | 2.95 | 30.28 | 4.5% | 100.0% | 0.00 | 0.0% |
| SLA_ON | CostQ+GreedyFallback | 11.32 | 7.26 | 33.51 | 18.1% | 100.0% | 0.00 | 0.4% |
| SLA_ON | FIFO | 13.78 | 11.29 | 34.92 | 15.7% | 100.0% | 0.00 | 0.0% |
| SLA_ON | IMMEDIATE_COST_GREEDY | 10.88 | 6.93 | 31.86 | 17.8% | 100.0% | 0.00 | 0.0% |
| SLA_ON | LONGEST_WAIT | 13.78 | 11.29 | 34.92 | 15.7% | 100.0% | 0.00 | 0.0% |
| SLA_ON | MIN_BLOCKER | 11.28 | 7.96 | 31.38 | 16.9% | 100.0% | 0.00 | 0.0% |
| SLA_ON | NEAREST_JOB | 12.54 | 7.96 | 38.02 | 20.8% | 100.0% | 0.00 | 0.0% |
| SLA_ON | SHORTEST_ESTIMATED_SERVICE_TIME | 10.88 | 6.93 | 31.86 | 17.8% | 100.0% | 0.00 | 0.0% |

## Primary paired bootstrap (`SLA_OFF`)

- 비교: `CostQ+GreedyFallback` − validation-selected baseline `SHORTEST_ESTIMATED_SERVICE_TIME`.
- Mean wait 차이: +1.20 min (95% bootstrap CI [+0.96, +1.44]).
- P95 wait 변화: +47.57% (95% bootstrap percent CI [+35.87%, +59.96%]).

## Fallback coverage 해석

- 0% fallback은 **pure Cost-Q**, 0% 초과 5% 이하는 **hybrid Cost-Q + fallback**, 5% 초과는 **coverage insufficient**다.
- `SLA_OFF/CostQ+GreedyFallback`: 0.0% — hybrid Cost-Q + fallback.
- `SLA_ON/CostQ+GreedyFallback`: 0.4% — hybrid Cost-Q + fallback.
- runner primary coverage class: `HYBRID_ACCEPTABLE`.

## 판정과 한계

- Full run 판정: `FAIL`.
- 기준별 결과: 평균대기 개선 `False`, P95 guardrail `False`, completion `True`, backlog `True`, coverage `True`.
- assumed 터미널 프로파일과 합성 외부트럭 흐름의 상대 비교만 가능하다.
- 선박·본선 deadline·실제 운영 배차 규칙은 이 Exp-1의 범위 밖이다.
- paired 수치, completion/backlog, fallback coverage를 함께 보지 않은 단일 평균 비교는 금지한다.

*원자료: exp1_direct_results.json*
