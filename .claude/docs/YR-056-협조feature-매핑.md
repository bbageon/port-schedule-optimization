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

## 6. 결과 (2026-07-18 full run, 2216s — negative)

| 정책 | total_cost | interference | WAIT 수 | mean_wait(분) | 완료율 |
|---|---:|---:|---:|---:|---:|
| COORD | 91.73 | 24.88 | 34.7 | 1.57 | 1.000 |
| NO_COORD | 88.03 | 22.86 | 33.3 | 1.08 | 1.000 |
| JOINT_ROLLOUT | 70.57 | 10.28 | 11.5 | 0.73 | 1.000 |

- **판정: feature 만으로는 협조 격차가 줄지 않는다.** COORD vs NO_COORD (paired 60 seed)
  Δtotal +3.70 [+1.52, +5.93]·Δinterference +2.02 [+0.49, +3.56] — 개선 증거 0,
  점추정은 오히려 악화. JR 대비 interference 격차(+12.6~+14.6)는 양 arm 공히 잔존.
- **해석**: 동시 결정 구조에서 "상대의 현재 상태 관측"은 "상대가 지금 무엇을 고를지"를
  알려주지 못한다 — 독립 argmin 학습자의 조정 실패는 관측 추가가 아니라 **결합
  가치/절차(joint value·mixer) 구조**를 요구한다는 YR-054 해석을 재확인.
- **한계**: arm 간 학습기 초기화 seed 상이(56000/56001) + checkpoint 선택 노이즈(YR-057,
  val→test 격차 ±3~9 기지) → "유의 악화 +3.7"의 크기는 과대해석 금지. 결정에 충분한
  사실은 방향 — **개선이 전혀 없다**는 것.
- **처분**: COORD 채널은 itc-v3 에 유지 (QMIX/YR-013 이 동일 채널을 mixer 입력으로 재사용
  가능, off arm 으로 언제든 재검증). 원자료: `outputs/reports/yr056_coord/`.
