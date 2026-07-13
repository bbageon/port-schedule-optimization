# YR-027 — 외부트럭 Direct-Job Cost-Q 구현·평가

- **Epic**: RL / **Priority**: 🟠 / **등록일**: 2026-07-13
- **배경**: YR-025 는 비용함수만 교체하고 휴리스틱 rule action을 유지해 유의차가 없었다. 개별 외부트럭 작업을 직접 고르는 비용 최소화 정책을 별도로 검증한다.
- **목표**: `BLOCK_ENTRY` 이후 실행 가능한 외부트럭 Job을 action으로 삼고, 평균 블록대기시간을 직접 최소화하는 Cost-Q와 즉시비용 greedy를 같은 외부트럭 전용 시나리오에서 paired 평가한다.
- **수용 기준**: `n_vessel=0`, `TRUCK_TO_YARD/YARD_TO_TRUCK` 용어, queue-area 단일 cost, `gamma=1`, 동적 후보 action, 미방문 fallback, SLA OFF primary/ON secondary, 물리 invariant 검증이 코드·테스트·리포트에 박제된다.
- **회귀 가드**: FIFO의 `BLOCK_ENTRY` 기준과 LONGEST_WAIT 동일성, ImmediateCostGreedy와 shortest-service 동일성, gate/deadline/숨은 SLA tie-break 미사용을 검증한다.
- **범위 밖**: 선박·본선 deadline, ETA·미래정보, 사전 포지셔닝·선재조작, 다중 YC, 실운영 개선 주장.
- **비교군**: FIFO, LONGEST_WAIT, NEAREST_JOB, MIN_BLOCKER, SHORTEST_ESTIMATED_SERVICE_TIME, ImmediateCostGreedy, Direct-Job Cost-Q.
- **주 지표**: `mean_wait_min`; guardrail은 P95·30분 SLA 초과율·backlog·완료율·이동·재조작.
- **전략 원본**: [v1 5+5 상태](../strategy-history/2026-07-13-YR-027-exp1-direct-job-cost-q.md) · [v2 최소상태](../strategy-history/2026-07-13-YR-027-exp1-direct-job-cost-q-minimal-state.md).

## 구현 경계

- 기존 `run-exp1-cost`와 YR-025 산출물은 유지하고, 별도 command·환경·artifact 이름을 쓴다.
- arm별 외부트럭 수 `N_config`를 고정 cost 분모로 쓰고, 실현 건수 일치·`n_vessel=0`·전 Job 외부트럭·metadata `v0`를 assert한다.
- 기존 `dispatchable_jobs()`와 실행 직전 `validate_assignment()`의 2중 물리 제약은 유지한다.
- train/validation/test의 bucket·seed·checkpoint·fallback coverage를 재현 가능하게 저장한다.

## 필수 검증

1. `BLOCK_ENTRY` 직전/정각 후보 경계와 gate 시각·ETA·deadline 변경 불변성을 검사한다.
2. 후보 0/1/다수, infeasible 저가 Q 무시, 완료 Job 재선택 거부, 동점 결정론을 검사한다.
3. 모든 feasible next key의 비종료 `c+min Q`, terminal `c`, 경과시간과 무관한 `gamma=1` update를 검사한다.
4. step cost 합과 `queue_area_s/(60N_config)=mean_wait_min` 항등식을 검사한다.
5. 미방문 우선탐색, 평가 fallback, 방문·미방문 혼합, Q-table 저장·복원을 검사한다.
6. primary SLA OFF와 secondary ON, `1800초` 경계, mask가 infeasible Job을 되살리지 않음을 검사한다.
7. FIFO=LONGEST_WAIT 및 ImmediateCostGreedy=shortest-service 동일성을 paired run에서 검사한다.
8. 기존 `run-exp1`, `run-exp1-cost`, golden 결과가 변하지 않는 회귀 검증을 수행한다.

## 평가 결과 (2026-07-13)

- HJNC assumed 프로파일, train 1,000 / validation 30 / locked test 100으로 동결 설정을 실행했다.
- Primary 비교군은 validation-selected shortest-service였고 Cost-Q 평균대기 차이는 `+0.039`분(95% bootstrap CI `[+0.006,+0.072]`)이었다.
- P95 변화율 CI 상한 `+7.13%`, fallback `55.04%`로 개선·guardrail·coverage 기준을 통과하지 못했다.
- Shortest-service는 FIFO `13.782→7.789`분(`-43.5%`)이었지만 NEAREST는 `11.285`분이었다. 따라서 효과는 이동거리 최소화가 아니라 접근·취급·positioning·blocker 재조작을 합친 총 예상 cycle time 최소화로 해석한다.
- 게이트 진입 후 `BLOCK_ENTRY` 전 차량도 YC 관점의 미래차량이며 Exp-1 정책에는 비공개다. 따라서 이 결과는 현재 블록 대기열만 본 조건이며 게이트 미래정보의 효과를 검증한 것이 아니다.
- completion 100%·backlog 0·물리 invariant·alias 회귀는 통과했다. [report](../../../outputs/reports/exp1_direct_costq_hjnc/exp1_direct_costq_report.md)

## 최소상태 v2 재평가 (2026-07-13)

- 사용자 결정에 따라 `GlobalState=(operation_phase, queue_length_bucket)`으로 축소했다.
- `CandidateFeature=(transfer_direction, estimated_service_time_bucket, end_crane_zone)`으로 바꾸고, 작업 완료 후 실제 YC bay 구간을 key에 추가했다.
- queue/service bucket만 train의 SLA_OFF FIFO 관측으로 fit하며 v1 산출물은 보존한다.
- 전체 동결설정 재실행 결과는 `outputs/reports/exp1_direct_costq_minimal_hjnc/`에 별도 기록한다.
