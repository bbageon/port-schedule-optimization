# YR-038 — 정규화 터미널 Total Cost·Reward (single source: 코드)

> YR-035 계약 + YR-036 시뮬레이터 + YR-037 resolver 위의 정규화 비용/보상 계층. single source 는
> `src/yard_rl/integrated/{cost_config,ledger}.py`·`experiments/terminal_cost.py`·`configs/costs/terminal_cost_*.yaml`.
> 배경: [최종전략 §10](부산항_레인_다중야드크레인_협조최적화_강화학습_최종전략.md) ·
> [spec](dashboard-task-specs/YR-038-total-terminal-cost.md) · [시뮬레이터](YR-036-통합시뮬레이터-매핑.md)

## 1. 핵심 판정

- **계약 무변경**: `contract/cost.py`(make_cost·CostBreakdown·contributions)·`schema.py`(COST_TERMS·
  VESSEL_FAMILY·SCHEMA_VERSION) 그대로. config 는 scale/weight/λ **실수치만** 주입, ledger 는 raw 전용
  side-channel(TransitionRecord 직렬화 미침투 → SCHEMA bump 없음).
- **golden 불변**: `default_assumed_config()` 가 현 `ASSUMED_SCALE/WEIGHT`·`assumed_lambda_vessel` 을
  바이트 재현 + ledger 기본 off(`enable_cost_ledger=False`) → `test_golden_terminal`(event hash
  `970b45a27b3da76e`)·`test_record_serialization_frozen` 불변.
- **재시뮬 0**: raw 는 record 에 박제 → config 변주(민감도·λ 비교)는 `make_cost` 재채점(rescore) 후처리.
- **전 항목 assumed**: 탐색 전용, 가중치 확정 금지(report ⚠ 배너). 실측 scale 은 YR-002, validation 은 YR-009.

## 2. 모듈/파일

| 파일 | 책임 |
|---|---|
| `integrated/cost_config.py` (신규) | `Provenance`·`TermCost`·`LambdaVesselPolicy`(정적/동적 밴드)·`TerminalCostConfig`(validate·load/save·with_*)·`RewardCalculator`·`default_assumed_config` |
| `integrated/ledger.py` (신규) | `CostCause`·`RATE_CAUSE`·`TERM_CAUSES`(항별 화이트리스트)·`CostLedger`·`assert_ledger_identity`·`build_ledger_report` |
| `integrated/cost.py` (수정) | `CostAccumulator.ledger` 필드·`accrue(cause=,subject=)`·`advance(RATE_TERMS_ORDERED)`·`cut→seal` |
| `integrated/engine.py` (수정) | accrue 8지점 cause/subject 배선(숫자 불변)·`enable_cost_ledger` 플래그 |
| `integrated/adapter.py` (수정) | `_assemble` 이 `RewardCalculator` 사용·`record_episode(reward_calc=)` seam |
| `experiments/terminal_cost.py` (신규) | `generate_terminal_scenarios`·`fit_terminal_scale`·`freeze_fitted_config`·`rescore`·`compare_lambda`·`sensitivity_grid`·`episode_ledger_check`·`build_cost_report` |
| `configs/costs/terminal_cost_v1.yaml`·`_static.yaml`·`_v2_fitted.yaml` | dynamic 기본·static A/B·baseline-fit 동결 |

## 3. cost config (§10.2·10.3·10.6)

- 13항 각 `TermCost{scale, weight, scale_prov, weight_prov, unit}`. `Provenance{basis(ProvBasis:
  assumed/regulation/fitted_baseline/measured), source, note, to_be_validated, fit_stat}` — **확정순서 추적축**.
- `LambdaVesselPolicy`: `STATIC`(상수) 또는 `DYNAMIC`(risk_ge 내림차순 밴드, §10.6 1.0~6.0). `lam(risk)` 는
  `assumed_lambda_vessel` 과 값 동치. λ 는 계약 `contributions()` 이 VESSEL_FAMILY 4항에만 적용.
- `RewardCalculator.cost_for(raw, risk_max)` → `make_cost`(scale/weight/λ 주입, total=Σcontrib·reward=-total 항등식).
- `validate()`: schema_version·13항 폐쇄·scale>0(ZERO_SCALE 선제)·weight≥0·DYNAMIC 밴드 내림차순.
- `load/save` YAML 왕복 무손실. `default_assumed_config()`==YAML load==현 상수 (test 강제, 이중소스 drift 차단).

## 4. 인과 ledger (항목 중복계상 0)

- **단일 write-path**: 13항 raw 는 예외 없이 `accrue()` 통과(rate 항은 `advance→accrue`). ledger 를 accrue
  안에서만 기록 → **Σledger==episode_raw 구성상 성립**, 중복계상 0 자동. scale/weight/λ 미포함(raw 물리 전용).
- **cause 화이트리스트**(`TERM_CAUSES`): 항별 허용 cause. `vessel_delay` 만 2-cause(VESSEL_FINISH·CLEAROUT).
  engine 8지점: truck/long_wait←WAIT_INTEGRAL, crane/empty/rehandle←DISPATCH(subject=crane), vessel/depart_delay←
  VESSEL_FINISH(subject=vid), 종료 vessel_delay←CLEAROUT. rate 5항←`advance`(`RATE_CAUSE`).
- **검증**: `assert_ledger_identity`(Σledger==episode_raw·항폐쇄·cause 화이트리스트). `interval_term_totals(k)==cut()[k]`.
- **guardrail 분리**: 안전위반·mandatory 미수용은 accrue 미호출 → ledger entry 원천 0. `term∈COST_TERMS` 폐쇄가
  신규 안전항 혼입 차단(SCHEMA bump 사안). `enable_cost_ledger=False`(기본)·`RATE_TERMS_ORDERED`(append 순서 결정론).

## 5. scale-fit·λ 비교·민감도

- **train baseline scale**: `fit_terminal_scale` — 참조 디스패처 baseline 의 **per-interval raw 평균**(§10.3).
  엔진 RNG 없음 → 결정론. TRAIN seed(101+)만 입력(test 누출 금지). baseline 미발현 항은 `fallback=True` 박제
  (assumed scale 유지, 조용한 0 금지). `freeze_fitted_config` → `terminal_cost_v2_fitted.yaml`(FITTED provenance).
- **정적 vs 동적 λ**: `compare_lambda` — 시나리오당 baseline 1회 → 두 config 재채점 → `paired_diff`(alt=dynamic).
  합성 fixture 본선 정시완료면 dyn=static(위험도 0) — λ 효과는 지연 구간에서만(메커니즘은 test 로 검증).
- **민감도(YR-026 흡수)**: `sensitivity_grid` — weight 축·λ 축 독립변주 재채점, `paired_diff` CI. 지배항
  vessel_delay 강조. `build_cost_report` → `outputs/reports/terminal_cost_v1/`(md ≤200줄 + raw JSON).

## 6. 테스트·산출물

`tests/unit/test_cost_config.py`(7): 계약동치·YAML왕복·reward identity·λ VESSEL_FAMILY 한정·bad config 거부·
fit provenance. `tests/invariants/test_ledger.py`(7): episode/interval 항등식·cause 화이트리스트·guardrail 배제·
off parity·결정론·RATE_CAUSE 완전성. `tests/experiments/test_terminal_cost_fit.py`(6): fit 결정론·fallback 박제·
rescore 순수회계·정적/동적 λ 고위험·compare·민감도 단조. 전체 279 tests pass, sim/·golden 불변.
산출물: `terminal_cost_v2_fitted.yaml`, `outputs/reports/terminal_cost_v1/{terminal_cost_report.md,terminal_cost_raw.json}`.

## 7. 범위 밖·known gap

- **YR-039**: Q망 학습(raw→정규화 total·reward·감사까지만). **YR-002/009**: 실측 원가·validation(전 항목 assumed).
- 안전위반 벌점화: mask(YR-037) 유지 — cost 스키마에 안전항 부재를 `keys==COST_TERMS` 로 강제.
- §10.4 비선형 장기대기(제곱항)·§10.6 본선 6항(목표생산성 d·긴급 e) vs 계약 4항 gap: v1 근사, COST_TERMS
  frozen → SCHEMA bump 사안, 범위 밖·문서화. 동적 λ 는 RISK 본선만 반응(SYMPTOM 선적 미상승).
