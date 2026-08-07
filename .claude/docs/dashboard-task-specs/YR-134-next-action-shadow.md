# YR-134 — 다음 작업 shadow 이중계산 검증
- **상태**: **backlog** (2026-08-07 중간점검 — board backlog.md 와 정합)

> 상태: backlog · 등록 2026-07-30
> 상세 결정: [사용자 결정 기록](../strategy-history/2026-07-30-YR-134-다음작업-shadow-이중계산-사용자결정.md)

## 목적

현재 공동 배정의 작업이 시작될 때, 다음 완료사건 뒤 예상 의사결정 상태에서 **다음
공동행동과 예비 공동후보를 미리 계산**한다. 다만 정책 성능이 검증되기 전에는 이 결과로
작업·자원·통로를 예약하거나 완료 뒤 선택을 강제하지 않는다. 완료 시점에는 기존 Block Q가
실제 최신 상태에서 자유롭게 다시 선택하고, 두 계산을 비교해 사전 예측의 가치만 측정한다.

현재 BlockQ-v2의 행동 단위는 크레인 한 대의 작업이 아니라, 그 의사결정에 참여한
`crane_ids` 전체의 **joint assignment(공동 배정)**다. 따라서 shadow도 크레인별 다음
작업이 아니라 `다음 완료사건 뒤 decision point + 예상 crane_ids + 공동후보 top-K`를
한 예측 단위로 삼는다.

## 두 계산의 역할

```text
A. 작업 시작 시점 t_start
   시작 시점에 관측 가능한 상태·공개 ETA
   + 현재 커밋된 공동 실행계획의 다음 완료사건 예상상태
   → 예상 decision crane_ids와 shadow 공동후보 1순위·top-K·예상점수 기록

B. 작업 완료 시점 t_complete
   동시각 물리사건을 모두 반영하고 행동 commit 전 실제 최신 상태 snapshot
   → 실제 decision crane_ids의 현행 공동후보 생성
   → Block Q·내부 resolver가 자유롭게 선택·실행
   → A와 B의 차이 및 사후 비용 차이 기록
```

A는 예측 답안이고 B는 완료 시점의 실제 의사결정이다. 같은 계산의 낭비가 아니라
**미래 상태를 미리 맞힐 수 있는지 검사하는 두 시점 측정**이다.

## 행동 불변 계약

- A의 결과는 `ShadowNextPlan` 로그에만 저장한다.
- `ReservationTable`에 넣지 않고 job token·slot·lane·corridor를 잠그지 않는다.
- 후보 mask·pruning·Q 입력·replay·reward·resolver 우선순위·owner/version을 바꾸지 않는다.
- 신규 도착·ETA 변경·취소·고장도 shadow를 이용해 실행을 바꾸지 않고, 사후 불일치
  원인으로만 기록한다.
- B에서는 shadow 1순위·예비후보를 우선하지 않는다. 기존 정책이 최신 상태에서 고른
  결과를 그대로 실행한다.
- shadow 진단 로그를 제외한 ON/OFF의 시뮬레이터 사건·행동·비용·완주·backlog 해시가
  같아야 한다.
- shadow 계산은 정책·시나리오·탐험 난수열을 소비하지 않는다. 표본예측이 필요하면 전용
  난수열을 쓰고, ON/OFF에서 본 실험 난수상태와 캐시·카운터가 같음을 검사한다.
- live 후보 생성기 대신 분리된 clone을 사용하고, 신경망은 `eval()`·`no_grad()`로
  추론한다. 전후의 정책 파라미터·모드, 생성기 counter/cache, Python·Torch·시나리오
  난수상태 digest가 같아야 한다.
- `ShadowNextPlan`은 실제 `event_log`가 아닌 별도 diagnostic sink에 저장한다.

즉 이 단계에는 hard invalidation, soft 예약, 예비후보 자동사용, 긴급 재계산 제약이 없다.
그 기능은 shadow 가치가 먼저 확인된 뒤 별도 단일축 작업으로만 검토한다.

## 예측에 허용되는 정보

- 작업 시작 시점에 이미 커밋된 현재 크레인 `JobPlan`과 형제 크레인 활성계획
- 커밋 계획으로 계산할 수 있는 완료 예상시각·종료 bay/row·예상 stack 이동
- 당시 공개된 예약·ETA·본선 계획과 known mask
- 당시 상태·작업·계획 version

금지:

- 실제 미래 `BLOCK_ARRIVAL`·고장·계획변경 이벤트
- 작업 시작 뒤에 도착한 정보
- 실제 이벤트 큐를 가진 엔진 사본의 미래 진행

현재 `_rollout_cost`는 실제 미래 도착을 가진 이벤트 큐를 복사하는 오라클이므로 A에
사용하지 않는다. 완료 예상 stack은 라이브 `observable_stacks()`를 미리 바꾸지 않고
별도 projection에서만 만든다.

## shadow 기록

최소 기록:

- `created_at`, `predicted_complete_at` 또는 예상 구간
- 예측을 연 active plan 묶음·예상 decision key·예상 `crane_ids`
- state/job/plan version
- predicted completion bay/row와 projection 요약
- primary와 backup top-K의 안정적인 **공동후보 fingerprint**
  (`crane_id + CandidateKind + canonical JobRef 전 필드 + JobPlan 물리효과 hash`의 정렬 묶음).
  결정 안에서만 유효한 `candidate_id`는 fingerprint로 쓰지 않는다. 물리효과 hash에는
  `JobPlan`의 moves·duration·end bay/row·corridor·slots·lane·재조작·이동량 등 모든
  실행결과 필드를 포함한다.
- 후보별 예측 점수·순위·feasible 사유
- 완료 시 실제 decision key·`crane_ids`·후보 존재 여부·새 순위·실제 공동선택
- 불일치 원인: 새 도착, ETA/계획 갱신, 형제 크레인 변화, stack 변화, 물리 불가,
  후보 소멸·신규 후보 등장, 예상과 다른 참여 크레인, decision 미개방

## 판정 지표

1. **G0 행동 불변**: 진단 로그를 제외한 shadow ON/OFF의 사건·행동·비용 해시 완전 일치.
2. **완료상태 예측오차**: 완료시각, 종료 위치, stack·queue 핵심 요약의 예측 대 실제.
3. **결정단위 일치**: 예상·실제 `crane_ids`와 decision key 일치율. 불일치·미개방은
   제외하지 않고 전체 예측 기준 실패율에 포함한다.
4. **후보 커버리지**: ①완료 시 정책 선택의 사전 top-K 포함률 ②C600 사후 참고최선의
   사전 top-K 포함률 ③사전·완료 후보집합 overlap을 각각 분리한다.
5. **순위 안정성**: top-1 생존율·top-K 순위상관·후보 신규/소멸률. 전체 예측 기준과
   decision 단위 일치 표본 조건부 결과를 함께 보고한다.
6. **선택 품질**: 완료시점의 짝지은 사후 후보비용 기준 선택 후 손실(regret).
7. **계산비용**: 추가 추론 횟수·벽시계 시간·로그 크기.

완료 시점의 현행 정책 선택은 **시간적 일치 기준**일 뿐 최적 정답은 아니다. 따라서
“미리 고른 후보와 실제 정책 선택이 같았다”만으로 성공 판정하지 않고, 같은 완료상태에서
후보별 사후 비용 또는 진단용 반사실 교사를 이용해 비용상 손실도 별도로 잰다. 이 교사는
결과 평가 전용이며 배포 입력이나 실제 행동에 쓰지 않는다.

사후 비용은 기존 YR-128 계보의 **C600**으로 고정한다: 동시각 사건을 모두 처리한 뒤
B 행동을 commit하기 전 하나의 완료 snapshot에서 모든 실행 가능 공동후보를 복사하고,
`RC_TRAIN` 비용·600초 고정창·SF-SPT 후속정책으로 짝평가한다. 후보끼리 같은 snapshot과
공통 외생난수를 공유한다. `regret_C600 = C600(shadow 후보) - min C600(완료시 후보집합)`로
정의한다. 이는 진단용 오라클이며 배포에는 쓰지 않고, 검사한 후보와 600초 창 안의
참고최선이지 전 에피소드의 전역 최적이라는 주장은 하지 않는다. YR-132에서 기각된
개수 기반 `D_H`는 이번 판정 표적으로 재사용하지 않는다.

shadow 후보가 완료 후보집합에서 소멸했거나 실행 불가능하면 C600 regret을 억지로
계산하지 않고 **커버리지 실패**로 집계한다. regret은 생존·실행 가능 후보 조건부로
별도 보고하며, 이 표본만으로 전체 예측 성능을 주장하지 않는다. soft cache 후속 자격은
전체 예측의 커버리지 문턱과 조건부 regret 문턱을 모두 통과해야 한다.

## 연구 순서와 경계

1. 실행·본 실험 난수 영향 0인 shadow 로그만 구현한다.
2. 결과를 보기 전에 표본수·top-K·성공 문턱을 사전등록하고 독립 시드로 판정한다.
3. 실패하면 사전계산을 폐기하고 완료 시점의 자유 재계산을 유지한다.
4. 통과하면 계산 재사용(soft cache)을 **별도 Dashboard row·단일축**으로 등록한다.
5. 자원 선점·예약·DEFER 같은 제약은 그 이후에도 별도 실증 없이는 추가하지 않는다.

현재 시뮬레이터는 정책 추론시간을 물리시간으로 모델링하지 않는다. 따라서 YR-134는
트럭 대기나 터미널 비용 감소를 직접 주장하는 성능 실험이 아니라, **예측 가능성·캐시
후보 가치 진단**이다.

## 코드 접점

- [engine.py](../../../src/yard_rl/integrated/engine.py): `observable_stacks`,
  `run_until_decision`, `dry_run_commit`, `_complete`. `dry_run_commit`은 현재 주어진 상태의
  공동 가능성 검사이므로 라이브 미래 projector로 쓰지 않고 별도 projected snapshot
  위에서만 재사용한다.
- [jobplan.py](../../../src/yard_rl/integrated/jobplan.py): canonical `JobRef`·`JobPlan`
  fingerprint 원본
- [reservation.py](../../../src/yard_rl/integrated/reservation.py):
  실제 물리 예약표 — shadow 저장소로 사용 금지
- [baselines.py](../../../src/yard_rl/integrated/baselines.py):
  `_rollout_cost`는 미래정보 오라클 — 시작시점 예측에 사용 금지

## 선결·관계

- YR-131은 순위손실로 순위가 생길 수 있음을 보였지만 top-1은 약 0.35~0.40이고,
  YR-132는 3개 초기화 중 1개에서 역효과가 나 최종 정책 안정성이 확정되지 않았다.
  따라서 지금 hard 예약을 넣지 않는 근거가 충분하다.
- YR-133의 블록 간 판매·견적과 독립인 **블록 내부 실행 예측 진단**이다. 두 작업의
  결정시점·로그·성능 주장을 섞지 않는다.
