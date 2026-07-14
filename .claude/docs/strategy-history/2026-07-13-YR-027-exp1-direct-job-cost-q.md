# YR-027 — Exp-1 외부트럭 Direct-Job Cost-Q 전략

> 기록일: 2026-07-13 · 상태: **구현·고정설정 평가 완료 — primary 미통과**
> 구현·평가 상태: [Dashboard YR-027](../../Dashboard/in-progress.md)

## 0. 이번 수정에서 동결한 결정

- 선박·본선 작업과 `deadline/slack`, 선박비용을 Exp-1에서 전부 제거한다.
- `GATE_IN/GATE_OUT`을 작업유형으로 쓰지 않고 `TRUCK_TO_YARD/YARD_TO_TRUCK`으로 분리한다.
- `BLOCK_ENTRY`는 차량 이벤트이자 **이 Exp-1의 후보 공개시점**이며 작업유형이 아니다.
- action은 휴리스틱 rule이 아니라 현재 실행 가능한 외부트럭 Job 하나다.
- reward 최대화가 아닌, 평균 블록대기의 queue-area cost를 직접 `argmin`한다.
- 0 초기화 Cost-Q의 미방문 낙관 편향을 평가 fallback과 coverage 지표로 통제한다.

## 1. 실험 질문과 범위

단일 야드블록·단일 YC에서 이미 블록에 도착한 외부트럭 중 다음 Job을 직접 고르는
정책이 비학습 정책보다 기사 평균 블록대기시간을 줄이는가?

| 포함 | 제외 |
|---|---|
| 외부트럭 반입·하차와 반출·상차 | 선박·본선 연계 Job, deadline/slack, `VESSEL_PRIORITY` |
| `BLOCK_ENTRY` 후 현재 대기열 순서 선택 | 게이트·COPINO·ETA 미래정보, 미도착 차량 |
| 단일 블록·단일 YC·non-preemptive 처리 | 사전 포지셔닝·선재조작, 다중 YC 협조 |

모든 train/validation/test 시나리오는 `n_vessel=0`이며, 생성 후에도 선박 Job 0건을
assert한다. 이 문서는 부산항 실운영 개선을 주장하지 않는 합성 시뮬레이션 전략이다.

## 2. 부산항 용어와 정책 경계

| 내부 표준명 | 의미 | 방향 |
|---|---|---|
| `TRUCK_TO_YARD` | 반입·하차 | 외부트럭 → 야드 슬롯 |
| `YARD_TO_TRUCK` | 반출·상차 | 야드 슬롯 → 외부트럭 |

두 이름은 공식 코드라고 주장하지 않는, 방향을 명확히 한 내부 alias다. 차량 진행 이벤트는
`GATE_ENTRY → BLOCK_ENTRY → SERVICE_START → SERVICE_COMPLETE → BLOCK_EXIT → GATE_EXIT`로
별도 관리한다. 정책 대기는 `SERVICE_START - BLOCK_ENTRY`다.

부산항만공사 공개 흐름도는 게이트 진입·블록 진입·상하차·게이트 진출을 구분하고,
[체인포털 기사 매뉴얼](https://www.chainportal.co.kr/manual/download?filename=ChainPortal_APP_Manual.pdf)은
오더의 반입/반출과 차량 진행상태를 별개로 표시한다. [BPA 프로세스](https://www.busanpa.com/board/view.do?boardId=BBS_0000031&dataSid=29935&menuCd=DOM_000000105002001000)

사전 오더·컨테이너·블록 배정은 COPINO 또는 게이트 단계에 일어날 수 있다. 따라서
`BLOCK_ENTRY`는 부산항 전체 TOS의 배정시점이라는 주장이 아니라, 실제 YC 대기열에 들어온
Job만 선택한다는 **Exp-1 정보 경계**다. 현재 코드의 `BLOCK_ARRIVAL`은 구현 시 호환 alias로 본다.

## 3. Episode와 의사결정 시점

- arm별 고정 `N_config`대가 8시간 동안 도착하고, 이후 모든 Job을 처리하는 clear-out이다.
- terminal은 모든 외부트럭 서비스 완료다. 안전 상한에 걸리거나 backlog가 남으면 그 run은 실패다.
- YC가 idle이고, `WAITING`이며 물리적으로 실행 가능한 Job이 하나 이상일 때만 결정한다.
- Job 실행 후 다음 결정까지 진행하며 그 구간의 실제 queue-area 증가를 반환한다.
- 작업별 경과시간이 다르므로 event-driven SMDP로 취급한다.

## 4. 최적화 목적과 step cost

arm을 시작하기 전에 외부트럭 수 `N_config`를 고정하고, 모든 scenario의 실현 건수가 같은지
assert한다. 부하 수준이 다르면 별도 arm·정책으로 학습한다. 트럭 `i`의 블록 진입을 `A_i`,
서비스 시작을 `B_i`라 하면 `W_i=B_i-A_i`이고,

`J(π) = Eπ[(1/N_config) Σ_i W_i] = Eπ[(1/N_config) ∫ Q_external(u)du]`.

결정 `t`와 다음 결정 `t+1` 사이 비용은 다음 하나만 사용한다.

`c_t = (queue_area_{t+1} - queue_area_t) / (60N_config) ≥ 0`.

모든 Job을 drain하므로 episode의 `Σ_t c_t`는 정확히 `mean_wait_min`이다. 이동·재조작은
별도 가중치로 더하지 않지만 처리시간을 늘리는 만큼 queue-area를 통해 간접 반영된다.
원화비용, tail penalty, 선박비용은 이 학습 cost에 넣지 않는다.

## 5. State와 후보 표현

> 2026-07-14 사용자 용어 확정에 따라 이름만 교체했다. tuple 값·순서·Q-key는 불변이다.
>
> | 원 이름 | 현재 이름 | 원 이름 | 현재 이름 |
> |---|---|---|---|
> | `time_period` | `work_phase` | `crane_position_zone` | `crane_area` |
> | `queue_length_bucket` | `waiting_truck_level` | `oldest_wait_bucket` | `longest_wait_level` |
> | `sla_over_count_bucket` | `over_30min_truck_count` |  |  |

```text
YardState = (
    work_phase, crane_area, waiting_truck_level,
    longest_wait_level, over_30min_truck_count
)

CandidateFeature(j) = (
    transfer_direction, own_wait_bucket, reach_time_bucket,
    service_time_bucket, blocker_bucket
)
```

- `work_phase`: 운영 초반·전반·후반·막판과 도착 종료 후 처리 구간이다.
- `crane_area`: 크레인이 현재 위치한 서비스 bay 구역(1~4)이다.
- `waiting_truck_level`: 기다리는 차량이 적음·보통·많음·매우 많음 중 어디인지 나타낸다.
- `longest_wait_level`: 가장 오래 기다린 차량의 대기시간 수준이며 30분 경계를 보존한다.
- `over_30min_truck_count`: 30분 넘게 기다린 차량 수 `0/1/2/3+`다.
- `transfer_direction`: `TRUCK_TO_YARD` 또는 `YARD_TO_TRUCK`이다.
- `own_wait`: train 분위수와 30분 hard edge로 해당 트럭의 대기를 구분한다.
- `reach_time`: 현재 YC 위치에서 작업 시작점까지의 프로파일 기반 예상시간이다.
- `service_time`: 접근·상하차·positioning·현재 필요한 재조작의 예상 총시간이다.
- `blocker`: 반출 대상 위 blocker `0/1/2/3+`; 반입은 방향 feature와 함께 0으로 둔다.

모든 bucket edge는 train에서만 fit한 뒤 고정한다. Job ID는 실행·결정론 동점처리에만 쓰며,
학습 key는 `K(j)=(YardState, CandidateFeature(j))`로 Job 사이에 값을 공유한다.

## 6. 후보집합과 SLA action mask

후보 `A_t`는 외부트럭, `actual_block_entry≤t`, `WAITING`, hold 없음, 단일 YC 서비스영역,
현 시점 정보만으로 안전·물리 제약을 모두 통과한 Job이다. 반출은 대상 컨테이너와 합법적
재조작 슬롯, 반입은 합법적 장치 슬롯이 있어야 한다. 미도착·미공개 Job은 포함하지 않는다.

`M_t={j∈A_t | wait_j≥30분}`으로 두고, 비어 있지 않으면 허용 action을 `M_t`로 제한한다.
평균대기 목적에 tail 가중치를 섞지 않고 운영제약으로 분리한다. 순수 Cost-Q 효과를 보는
`SLA_OFF`가 primary이고 `SLA_ON`은 제약형 secondary다. 같은 arm의 모든 정책에 같은 mask를 쓴다.

## 7. Cost-Q update와 정책

```text
V(G') = 0                                                     if terminal
        min_{j' feasible} Q_C(G', CandidateFeature(j'))       otherwise

Q_C(K_t) ← Q_C(K_t) + α_n [c_t + V(G_{t+1}) - Q_C(K_t)]
π(G) = argmin_j Q_C(G, CandidateFeature(j))
```

`γ=1.0`이다. 실제 경과시간은 이미 `c_t`에 적분되므로 추가 시간할인을 하지 않는다. 표준 Cost-Q
backup처럼 다음 상태의 모든 feasible signature를 포함하며, 미방문 값은 `Q_0=0`, terminal은 0이다.

학습은 (1) 현재 global state에서 미방문 후보를 균등 우선탐색, (2) 모두 방문했으면 확률 `ε`로
무작위 feasible Job, (3) 나머지는 `argmin Q_C` 순서다. 평가는 다음처럼 고정한다.

- 모든 후보 signature가 방문됨: `argmin Q_C`.
- 하나라도 미방문: 그 결정 전체를 `ImmediateCostGreedy`로 처리하고 fallback을 기록.
- Q 동률: longest wait → shortest estimated service → earliest `BLOCK_ENTRY` → `job_id`.

`Q_0=0`은 학습 중 낙관적 탐색을 유도하지만 평가 `argmin`에는 미방문 값을 그대로 섞지 않는다.
평가 정책명은 `CostQ+GreedyFallback`으로 기록하고 `fallback_count/rate`와 coverage를 보고한다.
test fallback 0%일 때만 순수 Cost-Q 효과를 주장하며, 5% 초과면 **coverage 부족**으로 판정한다.

## 8. Greedy와 비교 정책

`ImmediateCostGreedy`는 미래 도착을 보지 않고 현재 보이는 queue와 후보 예상시간만 사용한다.

`ĉ_t(j) = (q_t-1) × estimated_service_time(j) / (60N_config)`,
`a_t = argmin_j ĉ_t(j)`.

여기서 `q_t`는 mask 후 후보 수가 아니라 현재 외부트럭 전체 대기대수다. 이 정의는 미래 도착을
보지 않을 때 `SHORTEST_ESTIMATED_SERVICE_TIME`과 같은 선택이므로 두 이름은 구현 교차검증용
alias로 남기고 결과 동일성을 assert한다. 비교군은 FIFO, LONGEST_WAIT, NEAREST_JOB,
MIN_BLOCKER, SHORTEST_ESTIMATED_SERVICE_TIME(=ImmediateCostGreedy), Direct-Job Cost-Q다.

FIFO는 반드시 earliest `BLOCK_ENTRY`로 정의한다. 그러면 LONGEST_WAIT와도 같은 순서이므로
둘의 결과 동일성을 assert한다. 기존처럼 `actual_gate_in`을 쓰면 Exp-1 정보누출이다. 각 비교군은
이름에 적힌 1차 기준만 쓰고 `deadline`이나 숨은 SLA 우선순위를 tie-break에 넣지 않는다.
`MIN_BLOCKER`는 현재 필요한 blocker 수가 가장 적은 Job을 고르는 강한 비교군이다.

## 9. 초기 학습·평가 설정

| 항목 | 동결값 |
|---|---|
| 알고리즘 / 초기값 | Tabular Cost-Q / `Q_0=0` |
| 할인 / terminal | `γ=1.0` / `0` |
| 학습률 | `α_n=1/n_K^p`, `p∈{0.6,0.8,1.0}`를 validation 선택 |
| 탐색률 | train `ε_e=1/√(e+1)`, evaluation `0` |
| 데이터 | train 최소 1,000 episode / validation 30 / locked test 100 |
| 선택 | 50 episode마다 validation, 최저 `mean_wait_min` checkpoint |

train/validation/test seed는 분리하고, 정책 간에는 같은 scenario seed를 써 paired 비교한다.
hyperparameter와 checkpoint는 validation에서만 정하고 test 확인 후 바꾸지 않는다.

Primary는 `SLA_OFF`에서 validation으로 미리 고른 최강 비학습 비교군 대비 paired
`mean_wait_min` 차이다. 95% paired bootstrap CI가 0보다 작아야 개선으로 판정한다. P95 대기의
paired 변화율 95% CI 상한은 비교군 대비 `+5%` 이하여야 한다. 완료율 100%·backlog 0,
fallback 5% 이하와 모든 물리 invariant도 충족해야 하며 P50·SLA 초과율·이동·재조작을
함께 보고한다. fallback이 남으면 hybrid 결과로만 해석한다.

## 10. 연구 근거와 해석 한계

- [Bradtke & Duff (1994)](https://papers.nips.cc/paper/1994/hash/07871915a8107172b3b5dc15a6574ad3-Abstract.html): 가변 경과시간 의사결정의 SMDP RL 근거.
- [Watkins & Dayan (1992)](https://doi.org/10.1007/BF00992698): 이산 Q-learning과 비할인 흡수 환경 확장의 이론 근거.
- [Fotuhi et al. (2013)](https://doi.org/10.1016/j.retrec.2012.11.001): YC가 다음 트럭을 고르는 Q-learning 연구의 직접 선행근거.

문헌은 문제형식과 알고리즘 선택의 근거이며 위 episode 수·bucket·SLA·fallback은 이 실험의
사전등록 설정이다. 전체 queue 구성을 생략한 후보별 집계 key에는 state aliasing이 있으므로,
유한 데이터에서 수렴정리를 충족했거나 전역 최적 정책을 얻었다고 간주하지 않는다.

## 11. 구현 및 1차 평가 결과

기존 `run-exp1-cost`는 그대로 유지하고, `run-exp1-direct-costq`에 별도 환경·동적 Job action·
Cost-Q min update·fallback·전용 baseline과 paired bootstrap을 구현했다. HJNC assumed 프로파일의
동결 설정 실행에서 `SLA_OFF`는 `p=1.0`, episode 50을 골랐고 최강 비교군은 shortest-service였다.
test 평균대기는 Cost-Q 7.828분, 비교군 7.789분으로 차이 `+0.039`분(95% CI `[+0.006,+0.072]`),
P95 변화율 CI 상한은 `+7.13%`, fallback은 `55.04%`였다. 완료율 100%·backlog 0은 충족했지만
개선·P95·coverage 기준을 통과하지 못해 순수 Cost-Q 효과를 확인하지 못했다.
[전체 결과](../../../outputs/reports/exp1_direct_costq_hjnc/exp1_direct_costq_report.md)
