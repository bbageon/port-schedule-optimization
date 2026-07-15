# YR-037 — 동적 후보·중앙 joint resolver·Hard Constraint (single source: 코드)

> YR-035 계약 + YR-036 시뮬레이터 위의 RL 후보·공동배정 계층. single source 는
> `src/yard_rl/integrated/`. 배경: [최종전략 §8](부산항_레인_다중야드크레인_협조최적화_강화학습_최종전략.md) ·
> [spec](dashboard-task-specs/YR-037-joint-candidates-constraints.md) · [계약](YR-035-통합계약-매핑.md) · [시뮬레이터](YR-036-통합시뮬레이터-매핑.md)

## 1. 핵심 판정 (3렌즈 수렴)

- **D-FEASIBLE**: `feasible_mask` = **committed(직전 결정) 예약 대비 marginal 실행가능성**만 인코딩.
  같은 결정 내 형제 크레인 충돌은 mask 에 굽지 않고 중앙 resolver 가 담당 → mask 가 resolve 순서 무관·
  결정적. capture 는 assign 이전이라 이미 committed-only.
- **D-GATING(golden 안전)**: `_decision_cranes` 는 SERVE feasible 유무로만 판정(오늘과 byte-동일).
  PRE_REHANDLE/REPOSITION 은 SERVE 결정점의 co-후보 — 그 자체로 idle 크레인을 결정점 만들지 않음.
  → `candidates_for` compat shim 유지 → golden event hash `970b45a27b3da76e` 불변.
- **D-ORACLE**: resolver 의 joint feasibility 는 `engine.dry_run_commit` 이 판정 — `commit_decisions` 와
  **동일 경로**(_plan + reject_reason, crane_id 순) → "resolver 수용 = commit 통과" 항등식.
- **스키마 무변경**: 세 마스크·resolver_token·lane_id·mandatory 가 이미 계약에 존재 → SCHEMA_VERSION bump
  없음. audit 은 계약 밖 side-channel(`sim.resolution_log`).

## 2. 범위

- **한다(YR-037)**: 4종 후보 생성기(mandatory·padding·feasible 노출·score·정보시점 게이팅), 중앙 joint
  resolver(결정적 baseline·deadlock yield·audit), 비통과/공동 제약 강제, 0-위반 보장.
- **안 한다**: Q망 학습(YR-039 — `Preference` 를 `QValuePreference` 로 교체하는 seam만), 비용 가중치
  (YR-038 — resolver 는 비용최소 아님), 실측(YR-002/009). score·mandatory_frac(0.8)·k_max(12)·pre_window(600)
  은 assumed.

## 3. 신규/수정 모듈 (`src/yard_rl/integrated/`)

| 파일 | 책임 |
|---|---|
| `candidates.py` (신규) | `CandidateGenerator` — 4종 생성·mandatory·feasible/mask_reason·score·prune·padding order |
| `resolver.py` (신규) | `CentralResolver`·`BaselinePreference`·`DispatcherPreference` — 결정적 joint matching |
| `audit.py` (신규) | `JointResolution`·`CraneResolution`·`CandidateVerdict`·`resolution_stream_hash` (side-channel) |
| `reservation.py` (수정) | `reject_reason()` 공유 판정 — reserve/can_reserve/생성기 1차 mask 단일 소스 |
| `engine.py` (수정) | `info_level`·`_plan` kind 분기(PRE/REPO)+extra_exclude·`assign`/`_complete` kind 분기·`dry_run_commit`+`CommitProjection`·`_assert_pairwise_resources`·`resolution_log` |
| `jobplan.py` (수정) | `JobRef.reposition_target_bay`, `token: str\|None` |
| `adapter.py` (수정) | `_build_candidate_set`(생성기 소비·padding)·`capture`(gen_by_crane 반환)·`record_episode`(resolver 구동) |

## 4. 후보 생성기 (§8.2·8.4·8.5)

- **SERVE**: dispatchable job 전수. plan+`_committed_reason`(=reject_reason) 으로 feasible/mask_reason.
  오늘 조용히 드롭하던 LANE/CORRIDOR/SLOT 충돌 SERVE 를 **feasible=False 로 노출**(정책 관측 대상).
- **PRE_REHANDLE**(§8.4): PRE_ADVICE + provided_eta 에서만(누출 0). 미도착 GATE_OUT·blocker·시간창·합법슬롯.
  token=job_id 재사용(진행 중 동시 SERVE 를 token 예약이 차단). `_complete` 가 job 을 DONE/RUNNING 미변경.
- **REPOSITION**(§8.2): 임박 미래작업 근접(|Δbay|>1). sentinel job_id·token/lane=None·corridor=[cur,tb].
- **WAIT**: 항상 items 마지막 실후보(feasible=True). JointAction 에선 candidate_id=None(None⟺WAIT 불변식).
- **mandatory**(§8.2·YR-029): `cum_wait ≥ 0.8×SLA` 외부트럭 — pruning 절대 금지, feasible=False 여도 잔존.
  `_prune` budget 초과 시 `K_TOO_SMALL`(조용한 유실 금지). `_order_key`(kind·job_id·bay)로 canonical id.
- **정보시점**: `_visible_eta` 는 PRE_ADVICE 의 provided_eta 만 — actual_* 절대 미열람. 후보 존재/부재 자체가
  info-가시 결정함수 → 누출 0. score 는 pruning 전용, features·net 진입 금지.

## 5. 중앙 resolver (§8.6·§11.3)

- **알고리즘**: mandatory-우선 완전순서 제약 그리디 + dry_run 오라클. `pairs = feasible∧¬WAIT`, `_pair_key`
  = (mandatory, preference.rank, kind, crane_id, token, candidate_id) 완전순서. 각 pair 를 `dry_run_commit`
  으로 시험 — `set(proj.plans)==set(trial)`(전원 동시 feasible)면 수용. crane 당 1개.
- **0-위반**: 최종 chosen 은 마지막 수용의 dry_run 이 전원 feasible 판정 → apply(crane_id 순 commit)도
  전원 성공. 4겹 방어: 생성기 mask → resolver dry_run → commit `reserve()` → standing `_assert_pairwise`.
- **deadlock yield**: 경합 미수용 mandatory 는 드롭 아님 → `mandatory_deferred` 감사 + 다음 결정 재생성·
  최우선. feasible pair 존재 시 정렬 첫 pair 는 항상 수용 → 전원 WAIT 불가(기아 없음). 승자 완료가 패자
  yield 를 `_clear_yields` 로 해제 → 순차 서비스(교착 아님).
- **결정성**: pair 완전순서 + dry_run 순수(deepcopy·라이브 미변형) + membership-only set + RNG 없음
  → `resolution_stream_hash` 2회 동일·deepcopy 분기 동일.
- **Preference**: `rank` 만 교체하는 seam. Baseline/Dispatcher = 본선>최장대기>job_id(ReferenceDispatcher
  동일 key). YR-039 는 `QValuePreference` — joint action masking 골격 재사용.

## 6. 제약 강제 (4겹) · 계약 정합

| 위반 | 1차 생성기 | 2차 resolver dry_run | 3차 commit reserve() | 4차 standing invariant |
|---|---|---|---|---|
| 불가능 행동 | feasible=False+reason | feasible pair 만 | NOT_DISPATCHABLE raise | validate_joint INFEASIBLE_SELECTION |
| 중복 Job | reject_reason DUP_JOB | token 1:1 | DUP_JOB | `_assert_pairwise` TOKEN_DOUBLE |
| 레인 충돌 | LANE_CONFLICT | lane 배타 | LANE_CONFLICT | LANE_DOUBLE |
| 비통과·안전거리 | CRANE_INTERFERENCE | corridor 무겹침 | CRANE_INTERFERENCE | CORRIDOR_OVERLAP |

비통과=corridor envelope `[min,max touched bay]` + `safety_gap_bay` 보수적 표현(자료구조 무변경).
`_build_candidate_set`: 실후보 + padding_candidate(zero)·pad_mask/feasible_mask/mask_reason 정합 →
validate_candidates 통과. resolver.apply 의 chosen_candidate_id = CandidateSet items 인덱스 → validate_joint 정합.

## 7. 테스트

`tests/unit/test_candidate_generator.py`(9): 4종·padding zero·mandatory 보존+K_TOO_SMALL·feasibility=reservation
동치·PRE 정보게이팅·누출불변·결정론. `tests/integrated/test_resolver.py`(7): 완주 0위반·매 record validate_joint·
결정론·공유job DUP 0·mandatory 우선·단일크레인 dispatcher 동치·dry_run==commit. 회귀: golden event hash 불변,
record golden 1회 재생성(4종+padding), sim/ 무수정. 전체 259 tests pass.

## 8. 열린 리스크

- record feature 는 gc.plan(marginal, 생성시점)·committed plan(post-joint)은 다를 수 있음 — 관측=marginal(D-FEASIBLE)이 설계.
- 그리디 최대성 ≠ 최대매칭(카디널리티 sub-optimal 가능) — 제약위반 아님, KPI 는 YR-038/039 몫.
- REPOSITION corridor 정적예약은 보수적(YR-036 계승). PRE/REPO 완료회계(served_count 제외)는 YR-038 비용렌즈 합의 대상.
