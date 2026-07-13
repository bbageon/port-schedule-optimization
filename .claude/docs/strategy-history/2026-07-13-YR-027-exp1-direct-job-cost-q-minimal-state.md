# YR-027 v2 — Exp-1 Direct-Job Cost-Q 최소상태

> 기록일: 2026-07-13 · 상태: **전체 평가 완료 — coverage 통과, primary 미통과**
> v1 원본: [5+5 상태 전략](2026-07-13-YR-027-exp1-direct-job-cost-q.md)

## 1. 변경 이유

v1은 5개 GlobalState와 5개 CandidateFeature를 결합해 이론상 약 102만 개의
tabular signature를 만들었다. 선택된 50-episode checkpoint는 test 의사결정 중
55.04%에서 미방문 signature를 만나 shortest-service fallback을 사용했다.

평균대기 단일 목적에서 이미 발생한 최장대기는 sunk cost이고, `SLA_OFF`의 SLA 초과 수는
목적함수에 직접 들어가지 않는다. 현재 YC 위치·reach·blocker는 예상 총 service time과
중복되지만, action 이후 YC가 남는 위치는 v1 key에 없었다. 따라서 중복 feature를 제거하고
미래 전이에 직접 관련된 종료 위치를 추가한다.

## 2. v2 상태와 후보

```text
GlobalState = (
    operation_phase,       # OPERATING / CLEAR_OUT
    queue_length_bucket,
)

CandidateFeature(j) = (
    transfer_direction,    # TRUCK_TO_YARD / YARD_TO_TRUCK
    estimated_service_time_bucket,
    end_crane_zone,        # 작업 완료 후 YC bay 4구간
)
```

- `operation_phase`는 신규 블록 도착이 가능한 8시간 창과 이후 drain을 구분한다.
- `queue_length_bucket`은 현재 `BLOCK_ENTRY` 이후 대기 차량 수의 train 분위수다.
- `estimated_service_time`은 접근·취급·positioning·필요한 blocker 재조작을 포함한다.
- `end_crane_zone`은 반입이면 결정론적 장치 슬롯 bay, 반출이면 대상 컨테이너 bay다.
- bucket은 `SLA_OFF` FIFO train seed에서 queue/service 분위수만 fit하고 고정한다.

Job ID, 실제 대기시간, reach, blocker 수, `BLOCK_ENTRY` 시각은 실행·baseline·결정론적
tie-break 진단값으로 유지하되 Cost-Q key에는 넣지 않는다.

## 3. 유지되는 실험 계약

- 단일 블록·단일 YC, 외부트럭 100건, `n_vessel=0`, 정보 경계 `BLOCK_ENTRY`.
- action은 현재 물리적으로 실행 가능한 외부트럭 Job 하나다.
- cost는 `queue_area_delta_s/(60*N_config)`이며 episode 합은 평균대기시간(분)이다.
- `gamma=1`, `alpha_n=n^-p`, `p∈{0.6,0.8,1.0}`, `epsilon=1/sqrt(episode)`를 유지한다.
- train 1,000 / validation 30 / locked test 100과 50-episode checkpoint를 유지한다.
- `SLA_OFF` primary, `SLA_ON` secondary와 30분 action mask를 유지한다.
- 미방문 signature가 하나라도 있으면 decision 전체를 shortest-service fallback으로 처리한다.
- 비교군과 paired bootstrap, P95·completion·backlog·coverage 판정 기준을 유지한다.

## 4. 평가 분리

v1 산출물 `outputs/reports/exp1_direct_costq_hjnc/`은 덮어쓰지 않는다. v2는
`outputs/reports/exp1_direct_costq_minimal_hjnc/`에 별도 기록해 state 축소가 fallback과
평균대기 차이에 미친 영향을 같은 seed·설정으로 비교한다.

## 5. 전체 평가 결과

- clean source `687e5d5`, train 1,000 / validation 30 / test 100을 실행했다.
- SLA_OFF 선택값은 `p=1.0`, episode 300이며 validation 평균은 `8.716`분이다.
- test 평균은 Cost-Q `8.984`분, shortest-service `7.789`분으로 차이 `+1.195`분이다.
  paired bootstrap 95% CI는 `[+0.963,+1.438]`이므로 개선되지 않았다.
- P95는 `44.685`분 대 `30.280`분으로 `+47.57%`; 95% CI는
  `[+35.87%,+59.96%]`여서 guardrail도 통과하지 못했다.
- SLA_OFF fallback은 v1 `55.04%`에서 `0.01%`(1/10,000)로 감소했고 coverage 기준은
  통과했다. SLA_ON fallback도 `0.36%`였다.
- 모든 test episode의 completion은 100%, backlog는 0이며 물리·cost invariant는 통과했다.

따라서 state 희석은 해결됐지만 primary 실패 원인은 남았다. v2는 거의 순수 Cost-Q의 순서가
shortest-service보다 열세임을 보여준다. v1의 `+0.039`분 근접 결과는 55.04% decision에서
shortest-service fallback을 사용한 영향과 함께 해석해야 한다.

[전체 결과](../../../outputs/reports/exp1_direct_costq_minimal_hjnc/exp1_direct_costq_report.md)
