# YR-027 — 외부트럭 Direct-Job Cost-Q 구현·평가

- **Epic**: RL / **Priority**: 🟠 / **등록일**: 2026-07-13
- **배경**: YR-025 는 비용함수만 교체하고 휴리스틱 rule action을 유지해 유의차가 없었다. 개별 외부트럭 작업을 직접 고르는 비용 최소화 정책을 별도로 검증한다.
- **목표**: `BLOCK_ENTRY` 이후 실행 가능한 외부트럭 Job을 action으로 삼고, 평균 블록대기시간을 직접 최소화하는 Cost-Q와 즉시비용 greedy를 같은 외부트럭 전용 시나리오에서 paired 평가한다.
- **수용 기준**: `n_vessel=0`, `TRUCK_TO_YARD/YARD_TO_TRUCK` 용어, queue-area 단일 cost, `gamma=1`, 동적 후보 action, 미방문 fallback, SLA OFF primary/ON secondary, 물리 invariant 검증이 코드·테스트·리포트에 박제된다.
- **회귀 가드**: FIFO의 `BLOCK_ENTRY` 기준과 LONGEST_WAIT 동일성, ImmediateCostGreedy와 shortest-service 동일성, gate/deadline/숨은 SLA tie-break 미사용을 검증한다.
- **범위 밖**: 선박·본선 deadline, ETA·미래정보, 사전 포지셔닝·선재조작, 다중 YC, 실운영 개선 주장.
- **비교군**: FIFO, LONGEST_WAIT, NEAREST_JOB, MIN_BLOCKER, SHORTEST_ESTIMATED_SERVICE_TIME, ImmediateCostGreedy, Direct-Job Cost-Q.
- **주 지표**: `mean_wait_min`; guardrail은 P95·30분 SLA 초과율·backlog·완료율·이동·재조작.
- **전략 원본**: [2026-07-13 전략 히스토리](../strategy-history/2026-07-13-YR-027-exp1-direct-job-cost-q.md).

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
