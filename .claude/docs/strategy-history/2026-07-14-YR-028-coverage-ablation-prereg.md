# YR-028 — coverage ablation 사전등록 (실행 전 동결)

> 기록일: 2026-07-14 · 상태: **사전등록 — 본 실행 전 동결** (spec: [YR-028](../dashboard-task-specs/YR-028-cost-q-coverage.md))
> 질문: YR-027 v1 selected-checkpoint 의 test fallback 55.04% 는 무엇 때문인가 —
> (a) checkpoint 선택 규칙, (b) v1 상태공간 크기, (c) 학습예산.

## 동결 설정

| 항목 | 값 |
|---|---|
| 프로파일 / arm | HJNC-ARMG (assumed) / `SLA_OFF` 단일 |
| 새 seed band | train 40000+3000 / validation 50000+30 / test 60000+100 — **YR-027 band(10k/20k/30k) 와 분리, 코드가 재사용을 거부** |
| 상태 축 | `v1_rich` (YR-027 v1 인코딩 `20a42cf` 복원 — 5+5 feature) vs `v2_minimal` (현행 2+3) |
| 학습 | v1: 3,000 ep (프로토콜 horizon 1,000 + 증량 축) / v2: 1,000 ep. checkpoint 50 ep 마다, α=n^-p, p∈{0.6,0.8,1.0}, ε=1/√ep |
| 선택 규칙 | **R1** = validation mean_wait 최소 (YR-027 재현; tie: 낮은 p → 이른 ep) / **R2** = validation fallback ≤ 5% 중 mean_wait 최소 |
| horizon | R1/R2 를 각각 ≤1,000(프로토콜 재현)·≤3,000(증량) 에 사후 적용 — 학습 1회 공유, 선택은 validation 만 사용 |
| bucket | FIFO train 관측 fit (v1 확장 edge 포함: own/oldest wait 에 SLA 30분 hard edge) — v1/v2 공유 |
| 비교군 | Direct baseline 6종 (FIFO=LONGEST alias·shortest=greedy alias assert 유지), validation 최저 mean 이 paired 기준 |
| 통계 | paired bootstrap 10,000 resamples, mean_wait 차이 CI + p95 변화율 CI 보고 |

## 판정 규칙 (사전 동결)

- 어떤 v1 checkpoint(≤3,000ep)도 validation fallback ≤5% 를 달성하지 못하면 → **STATE_SPACE**
- ≤1,000ep 에서 달성 가능한데 R1 이 fallback>5% checkpoint 를 선택하면 → **CHECKPOINT_RULE**
- 1,000ep 초과에서만 달성되면 → **BUDGET**
- R1 선택이 gate 를 이미 넘으면 (55% 미재현) → **NONE_REPRODUCED** (새 band 특이성 보고)

## 비목표

- YR-027 FAIL 의 사후 재해석·성능 재판정 금지 (원인 분리만).
- 선박·미래정보·다중 YC·새 feature 추가 금지 — 상태 v3 설계는 YR-030 에서.

## 산출물

`outputs/reports/costq_coverage_ablation_hjnc/` — checkpoint_curve.json (episode 별
val mean·fallback·table_keys), selections.json, test_results.json, ablation_results.json
(verdict 포함), coverage_ablation_report.md. CLI: `run-costq-ablation`.
