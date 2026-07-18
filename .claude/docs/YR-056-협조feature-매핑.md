# YR-056 — COORD 협조 feature 매핑 (itc-v2 → itc-v3)

> 설계 질문: rollout 이 시뮬레이션으로 얻는 "상대가 무엇을 할지"를, RL 에게는
> **이미 commit 된 관측 사실**로 줄 수 있는 만큼만 준다 (미래 예측·진실값 없음 — 누출 0).

## 1. 신규 필드 5종 (전부 AblationGroup.COORD — off 시 v2 와 정보 동일)

| group | field | 값 | 원천 |
|---|---|---|---|
| yc | `neighbor_busy_kind` | 최근접 상대의 실행 중 행동 종류 (action_kind_idx 동일 정규화). idle=결측 | `sim.active_plan()` |
| yc | `neighbor_busy_target_bay` | 상대 실행 계획의 종료 bay. idle=결측 | 〃 |
| yc | `neighbor_available_in_s` | 상대 가용까지 남은 시간 (상대 없으면 결측) | fleet 상태 |
| yc | `recent_yield_count` | 에피소드 내 경합 패배(LOST_CONTENTION) 양보 누적 | resolver→engine 배관 |
| candidate | `contention_risk` | max(같은 작업 idle 상대 가능 1.0 / busy 상대 eligible 0.5 / 상대 실행 corridor 겹침 0.7) | eligible_crane_ids·corridor |

- 최근접 상대 = bay 거리 최소 (동률 crane_id) — N-크레인 일반화 대비.
- 채널은 각 그룹 **말미 추가** — 기존 채널 index 불변 (dims: yc 22→26, candidate 37→38).

## 2. 배관 (behavior 불변 — 행동 golden n_events 115·hash 970b45a2 유지 확인)

- `resolver.apply` 가 WAIT 의 `yield_reason`(기판별: NO_FEASIBLE|LOST_CONTENTION)을
  `CraneAssignment` 에 실어 전달 → `engine.assign` 이 LOST_CONTENTION 만
  `YcRuntime.recent_yield_count` 에 누적 (recent_* 관례 동일 — 에피소드 누적).
- `engine.active_plan(crane_id)` 공개 접근자 신설 — 관측 계층이 `_active_plans` 직접 접근 금지.
- `dqn_learner.run_episode(ablation_off=...)` 인자 신설 — arm 구성이 capture 까지 전달.

## 3. 버전 절차 (YR-050 관례)

- SCHEMA_VERSION itc-v2→**itc-v3**, golden 3종 재생성(스키마·전이·terminal record),
  v2 golden 은 제거. contract fixtures 에 신규 raw 포함(채널 직렬화 검증).
- 거동 불변 증명: 행동 golden(`test_golden_terminal` — 이벤트 수·해시·비용 raw) 무수정 통과.

## 4. 실험 하네스 (`experiments/yr056_coord_experiment.py`)

- dueling(YR-045 RL 최선) × {COORD, NO_COORD} 동일 예산 500ep, val 20 으로 checkpoint 선택,
  test 60 — 전부 신규 대역 (500k/510k/520k; 가드가 <250k·300k대·400k대 차단).
- 기준선 JointRollout(forbid_strategic_wait=True — YR-045 최강 조건). RL 은 YR-052 기본(제외).
- 판정 지표: total_cost·**interference**(YR-054 격차 주인)·WAIT 수·mean/p95 wait,
  paired bootstrap CI (bootstrap_seed 75_056).
- quick 검증 노트: 8작업·6ep 조건에선 결정이 mandatory 로 강제되어 두 arm 이 동일 —
  arm 분리 자체는 `test_coord_ablation_restores_v2_information` 이 계약 수준에서 증명.

## 5. 테스트 (`tests/integrated/test_yr056_coord.py`, torch 불필요 — 양 환경 실행)

스키마 v3 구조·busy/idle 의도 관측·경합 위험 발현(≥0.5)·counter==resolution_log 정합·
ablation off 시 5채널 known=0. Windows 302 / WSL 353 passed (기지 머신민감 1건 제외 — YR-058).
