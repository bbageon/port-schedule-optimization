# YR-030-c — Greedy 기반 잔차 Cost-Q 사전등록

> 기록일: 2026-07-14 · 상태: **완료 — 개선 없음 (단 격차 역대 최소 +0.216분·잔차 구조 유효 확정)** · 결과는 하단 append
> 전략 원문: 사용자 제출 (2026-07-14)
> 상위: [YR-030 계열 2 baseline 승격](2026-07-14-YR-030-series2-baseline-pivot.md) · 선행: [YR-030-b 결과](2026-07-14-YR-030-b-v1final-greedy-prior-prereg.md)
> **되돌림 조건 2회차**: 본 실험도 greedy 유의 미달 시 tabular 한계 판정 → YR-012/재정의 재논의.

## 1. 전략 (사용자 최종안 — 원문 요지, 본 문서가 계약)

**기본 원칙**: 정확히 계산 가능한 현재 작업비용은 greedy 가 담당, RL 은 그 작업 선택
**이후의 미래 영향만** 보정한다.

- `Q_total(s,j) = G(s,j) + ΔQ(z(s,j))`, `j* = argmin_j Q_total`
- **G(s,j)** = `(현재 대기차량 수 − 1) × 후보 정확한 총 소요시간 / (60 × N_config)`
  — 소요시간 = YC 이동 + 취급 + 재조작 + 트럭 포지셔닝, **bucket 아닌 정확한 초 단위**
  (기존 `prior_cost` 재사용 — 예: G(A)=181s, G(B)=227s 의 46s 차이는 항상 유지)
- **ΔQ 입력 z** = `future_situation` (§2) — 현재 작업비용을 재학습하지 않는다
- **초기화**: ΔQ(z)=0 전부 → 학습 전 정책 ≡ greedy. 방문 후에도 G 는 **절대 대체되지
  않음** (YR-030-b prior 방식과의 결정적 차이 — α₁=1 대체 없음)
- **학습식**: `Y = c + γ·min_j'[G(s',j') + ΔQ(z(s',j'))]` (종료 시 Y=c),
  `Y_Δ = Y − G(s,j)`, `ΔQ(z) ← ΔQ(z) + α[Y_Δ − ΔQ(z)]` — **ΔQ 는 음수 허용**
- **선택**: 학습 ε-greedy(무작위/argmin Q_total), 평가 ε=0. 동률 tie-break 는
  기존 등록 순서 (−wait, service, block_entry, job_id)
- **선택 반전 예** (§8): A: 181+30=211 vs B: 227−40=187 → B — 46s 차이를 기억한 채
  미래 이익이 그것을 능가할 때만 뒤집힘

## 2. future_situation (ΔQ 키) — bucket 정의 (숫자 경계 동결)

| 필드 | 단계 | 경계 (train 관측으로 fit 후 고정 — val/test 재조정 금지) |
|---|---|---|
| crane_location_after_job | 4구역 | 기존 `_bay_zone` (service range 4등분) |
| number_of_jobs_left_group | 없음/적음/보통/많음 | 0 / train (대기수−1) 3분위 2경계 |
| total_work_time_left_group | 없음/짧음/보통/김 | 0 / train Σ잔여 예상시간 3분위 2경계 |
| remaining_job_mix | 없음/짧은위주/혼합/긴위주 | 잔여 중 짧은작업(< train service 중앙값) 비율 ≥⅔ / ≤⅓ / 그 외 |
| distance_to_nearest_job_group | 같은구역/인접/먼/없음 | 종료 위치 구역 vs 잔여 후보 최근접 구역 차 0/1/≥2 (운영 기준) |

- "잔여" = 결정 시점 feasible 후보 − 선택 후보 (근사: WAITING 중 dispatch 불가분 제외 — 명시)
- 잔여 후보 위치 = 각 후보의 종료 bay (반출=대상 컨테이너 bay, 반입=장치 슬롯 bay)
- 신규 bucket 필드 3종(`jobs_left`·`work_left_s`·`short_service_s`)은
  DirectJobBucketConfig 에 추가·저장 — 기존 필드·스키마 키와 하위호환

## 3. 비교 실험 (3-arm, 동일 seed paired)

| arm | Q_total | ΔQ 키 | 목적 |
|---|---|---|---|
| **Greedy only** | G | — | baseline (기존 IMMEDIATE_COST_GREEDY) |
| **Residual only** | G + ΔQ | 기존 coarse state `(YardState, JobState)` v1_final | 잔차분해 자체의 효과 |
| **Residual + future** | G + ΔQ | `future_situation` **단독** | 최종 정책 — 미래 맥락 효과 |

**해석 명시**: 전략 원문 §3 이 ΔQ 입력을 future_situation 으로 정의하므로 최종 arm 의
키는 future_situation 단독으로 구현한다 (coarse state 와 결합 아님). 잔차 철학상 현재
작업 정보는 G 전담이며, 키가 작아(≤ 4·4·4·4·4=1024) 조밀 학습이 가능 — YR-027 파편화
교훈 반영. env 전역 상태 스키마는 두 RL arm 모두 v1_final (일관성 규칙 4건 유지).

## 4. 동결 설정

| 항목 | 값 |
|---|---|
| 프로파일 / arm | HJNC-ARMG (assumed) / `SLA_OFF` |
| seed band | train 110000+3000 / val 120000+30 / test 130000+100 — 기존 6개 대역(10k~90k)과 분리, 코드가 재사용 거부 |
| γ | **0.95 고정** — YR-030-b 에서 γ 축 소진(4점 유의차 0), 사용자 지정값 승계. 재탐색은 비목표 |
| α | n^−p, **p=1.0** (YR-028 R2 승계) · ε = 1/√ep · 3,000 ep/arm · ckpt 50 ep |
| 선택 | arm 별 validation mean_wait 최소 checkpoint. baseline 은 validation 최저 mean 휴리스틱 (6종) |
| 통계 | paired bootstrap 10,000 · mean CI + p95 변화율 CI |
| **판정** | 개선 = 해당 arm 의 mean Δ CI **상한 < 0** (vs 선택 baseline). guardrail: P95 Δ% CI 상한 ≤ +5%·completion 100%·backlog 0·invariant 0 동시 보고 |
| ablation 해석 | future arm 만 개선 → 맥락 효과 입증 / 둘 다 미달 → 되돌림 조건 2회차 발동 |

## 5. 검토한 대안과 기각 사유

| 대안 | 기각 사유 |
|---|---|
| future 키 = coarse state ⊕ future 결합 | 키 폭발(≥10⁴×) — YR-027 v1 파편화 재현 위험. §3 원문 정의와도 불일치 |
| γ grid 재실행 | YR-030-b 에서 축 소진 (4점 유의차 0) — 예산 낭비 |
| prior 방식(첫 방문 대체) 병행 arm | YR-030-b 가 이미 그 결과 — reference 로 인용이면 충분 |

## 6. 한계 (사전 명시)

- G 는 **즉시비용**(다음 구간 근사), Y 는 **누적 잔여비용** — Y_Δ 에는 "미래 잔여 총량"
  이 포함되어 ΔQ 절대값이 클 수 있음. argmin 비교에서 후보 간 공통분은 상쇄되나,
  γ<1 부트스트랩과의 상호작용은 관찰 항목.
- 잔여 작업 집합을 feasible 후보로 근사 (WAITING 이지만 dispatch 불가한 작업 제외).
- 합성 시나리오 + assumed 프로파일 — 실운영 주장 불가 (YR-009 전).

## 7. 비목표

γ/p 재탐색 · 선박·미래정보·다중 YC · YR-027/028/030-b 결론 재해석 · SLA_ON arm.

## 8. 산출물

`outputs/reports/costq_residual_hjnc/` — direct_buckets(확장 필드 포함)·seed_manifest·
checkpoint_curve·selections·test_results·residual_results·residual_report.md ·
agent_*.json. 구현: `policies/residual_cost_q.py`·`envs/direct_job_env.py`(future
feature)·`experiments/residual_costq.py`·CLI `run-costq-residual [--quick]`.

---

## 실행 결과 (2026-07-14 append — 사전등록 원문 §1~§7 불변)

- 실행: clean source `04104c8`, 소요 20.0분. [리포트](../../../outputs/reports/costq_residual_hjnc/residual_report.md)

### 판정: 개선 없음 — 두 arm 모두 baseline(IMMEDIATE_COST_GREEDY) 유의 열세

| arm | 선택 ep | mean Δ [95% CI] | p95 Δ% 상한 | guardrail |
|---|---|---|---|---|
| ResidualCostQ[**state_job**] | 2950 | **+0.216 [+0.149, +0.289]** | +9.0% | P95 ❌ / 완료·backlog·invariant ✅ |
| ResidualCostQ[**future**] | 50 | +0.778 [+0.656, +0.911] | +17.1% | 〃 |

### 확정 1 (positive) — 잔차 구조는 유효: 격차 역대 최소로 절반 축소

- state_job arm: 학습곡선 **병리 없는 단조 개선** (7.76→7.34, 선택 ep 2950 —
  마지막 checkpoint), 격차 +0.216분 = **YR-030-b prior(+0.454)의 절반, YR-028
  순수(+0.525~+1.283)의 1/2~1/6**. "G 를 절대 보존하고 Δ만 학습" 구조가
  초기화·병리·스케일 문제를 모두 제거하면서 순서품질을 실질 개선.

### 확정 2 (negative) — future_situation 단독 키는 유해

- future arm: **학습할수록 악화** (ep50 8.02 → ep2950 8.94), best = ep50 (사실상
  무학습). 키 215개 — 너무 조악해 서로 다른 상황들이 한 칸에 뭉개지고, 그 평균
  보정이 개별 결정에 해로움 (aggregation bias). §3 해석(단독 키)의 리스크가
  현실화 — 미래 맥락 정보가 무익한 게 아니라 **coarse 현재-상태 정보를 버린 것**
  이 원인일 가능성이 남음 (단독 키로는 판별 불가, §5 에서 결합 키는 폭발로 기각).

### 되돌림 조건 발동 — 사전등록 2회 연속 greedy 미달 (YR-030-b → YR-030-c)

상위 전략 §6 합의대로 **"후보 단위 tabular 스코어링의 한계" 판정** → YR-012
(함수근사 직행) 또는 문제 재정의를 재논의한다. 함수근사는 본 실험의 두 교훈과
정확히 정합: (a) G 를 연속 feature 로 주입(잔차 구조 유지 가능), (b) bucket 없이
연속 상태로 Δ 학습 — state_job 의 남은 격차(+0.216)가 bucket 해상도 손실이라는
가설을 직접 시험. **다음 단계는 사용자 승인 대기.**
