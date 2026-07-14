# YR-028 — coverage ablation 사전등록 (실행 전 동결)

> 기록일: 2026-07-14 · 상태: **완료 — 판정 CHECKPOINT_RULE** (사전등록 원문 §동결 설정~§산출물 은 실행 전 그대로, 결과는 하단 append)
> spec: [YR-028](../dashboard-task-specs/YR-028-cost-q-coverage.md)
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

---

## 실행 결과 (2026-07-14 append — 사전등록 원문 불변)

- 실행: clean source `c54c4cd`, 전 규칙 사전등록대로. 소요 약 42분.
  결과 커밋 `5ab640d` · [리포트](../../../outputs/reports/costq_coverage_ablation_hjnc/coverage_ablation_report.md)

### 판정: `CHECKPOINT_RULE`

- gate(val fallback ≤5%) 통과 checkpoint: **프로토콜 horizon(≤1,000ep) 내 15개**,
  전체(≤3,000ep) 135개. v1 fallback 최소 0.5% — **상태공간·예산은 주범 아님**.
- R1(YR-027 규칙 재현)은 p=0.6, **ep=50, val fallback 53.2%** 를 선택 — 55% 현상이
  새 seed band 에서 재현 (locked test fallback 54.9%).
- v1 도달가능 signature 는 **~17.6k 에서 포화** (ep50 ~3.0k → ep2850 ~17.6k) —
  이론 상한 102만의 1.7%. 3개 p 모두 동일 궤적 (ep50 fallback 53.2~53.6%,
  gate 첫 도달 ep850~1,050).

### 핵심 부가 발견 — fallback↓ = 성능↓ 단조 (locked test, paired vs shortest-service)

| variant | 선택 (p, ep) | test fallback | mean_wait Δ [95% CI] (분) | p95 Δ% CI 상한 |
|---|---|---|---|---|
| v1 R1@1000 | 0.6, 50 | 54.9% | +0.125 [+0.073, +0.182] | +6.6% |
| v1 R2@1000 | 1.0, 1000 | 3.2% | +0.732 [+0.603, +0.869] | +26.6% |
| v1 R2@3000 | 1.0, 2200 | 0.8% | +0.580 [+0.456, +0.712] | +19.6% |
| v2 R1@1000 | 1.0, 800 | 0.0% | +1.283 [+1.058, +1.530] | +53.1% |

- **순수 학습 Q 비중이 높을수록 일관되게 나쁘다** — R1 이 ep50 을 고른 이유는
  규칙 결함이 아니라 "greedy 혼합이 실제로 최선"이었기 때문. YR-027 v1 의
  +0.039분(당시 band)과 이번 +0.125분(새 band)은 같은 hybrid 현상.
- 상태 풍부화(v1 vs v2, 순수 비교 +0.58 vs +1.28)는 격차를 절반으로 줄이나
  **역전시키지 못함** — 상태만으로는 부족.
- 학습 곡선에서 val mean 은 ep50 (7.24~7.31) 이 최선이고 학습이 진행될수록 악화
  (p=0.6 은 ~8.2 까지) 후 정체 — "더 배울수록 나빠지는" 방향성이 3개 p 공통.

### 해석과 파생 결정

1. **coverage 는 해결된 문제** — R2(coverage-gate) 선택규칙 채택 + 예산이면 충분.
   이후 계열 2 실험의 선택규칙은 R2 로 고정한다.
2. 다음 실험 질문은 **"왜 순수 Q 의 순서가 greedy 에 지는가"** 로 좁혀짐. 용의자:
   (a) **Q0=0 낙관 초기화** — min-backup 이 미방문 키의 0 을 bootstrap 에 포함,
   학습이 덜 된 signature 로 정책을 끌어당김 → greedy 추정치 ĉ 를 Q0 prior 로
   대체하는 실험이 YR-030 1차 (greedy 보다 나빠질 수 없는 출발점).
   (b) **키의 맥락 결손(aliasing)** — (전역상태, 후보) 키가 대기열의 나머지 구성을
   못 보는데 greedy 의 ĉ=(q-1)×service 는 그것을 직접 사용 — 후보 맥락 feature 가
   상태 v3 후보.
3. 비목표 준수 확인: YR-027 FAIL 재해석 없음 — 본 판정은 원인 분리이며, "순수
   Cost-Q 가 shortest-service 열세" 결론은 오히려 새 band 에서 재확인됨.
4. board 반영: YR-028 done (`27b2cdc`), YR-030 1차 과제 = 순서품질로 갱신,
   Dashboard README overview 갱신.
