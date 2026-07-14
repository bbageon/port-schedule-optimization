# YR-030-c — Greedy 기반 잔차 Cost-Q (3-arm)

> ⚠ 가정 프로파일 + 합성 시나리오. Q_total = 정확한 greedy 비용 G + ΔQ — 잔차분해(state_job) vs 미래맥락(future) 키의 순서품질 효과 검증.

- baseline (validation 선택): **IMMEDIATE_COST_GREEDY**
- baseline 을 유의하게 이긴 arm: **없음**

## locked test — paired vs IMMEDIATE_COST_GREEDY

| arm | 선택 ep | val_mean | coverage | mean_wait Δ [95% CI] | p95 Δ% CI 상한 | guardrail (P95≤+5%/완료100%/backlog0/invariant) |
|---|---|---|---|---|---|---|
| ResidualCostQ[state_job] | 2950 | 7.34 | 99.7% | +0.216 [+0.149, +0.289] | +9.0% | ❌/✅/✅/✅ |
| ResidualCostQ[future] | 50 | 8.02 | 99.8% | +0.778 [+0.656, +0.911] | +17.1% | ❌/✅/✅/✅ |

## checkpoint 곡선 (validation)

| arm | episode | val_mean | signature coverage | table_keys |
|---|---|---|---|---|
| state_job | 50 | 7.76 | 63.2% | 2783 |
| state_job | 150 | 8.00 | 83.8% | 5485 |
| state_job | 250 | 8.04 | 90.5% | 6986 |
| state_job | 350 | 7.96 | 94.0% | 7924 |
| state_job | 450 | 7.94 | 95.5% | 8599 |
| state_job | 550 | 7.90 | 96.9% | 9106 |
| state_job | 650 | 7.65 | 97.9% | 9484 |
| state_job | 750 | 7.71 | 98.1% | 9765 |
| state_job | 850 | 7.70 | 98.2% | 10002 |
| state_job | 950 | 7.65 | 98.4% | 10220 |
| state_job | 1050 | 7.51 | 98.8% | 10420 |
| state_job | 1150 | 7.65 | 98.8% | 10572 |
| state_job | 1250 | 7.48 | 99.0% | 10704 |
| state_job | 1350 | 7.47 | 99.2% | 10816 |
| state_job | 1450 | 7.46 | 99.3% | 10932 |
| state_job | 1550 | 7.54 | 99.3% | 11017 |
| state_job | 1650 | 7.50 | 99.4% | 11086 |
| state_job | 1750 | 7.46 | 99.4% | 11166 |
| state_job | 1850 | 7.48 | 99.5% | 11247 |
| state_job | 1950 | 7.43 | 99.5% | 11304 |
| state_job | 2050 | 7.41 | 99.7% | 11369 |
| state_job | 2150 | 7.42 | 99.7% | 11433 |
| state_job | 2250 | 7.42 | 99.7% | 11489 |
| state_job | 2350 | 7.38 | 99.7% | 11525 |
| state_job | 2450 | 7.42 | 99.8% | 11572 |
| state_job | 2550 | 7.45 | 99.7% | 11611 |
| state_job | 2650 | 7.37 | 99.7% | 11667 |
| state_job | 2750 | 7.41 | 99.7% | 11712 |
| state_job | 2850 | 7.35 | 99.8% | 11746 |
| state_job | 2950 | 7.34 | 99.9% | 11788 |
| future | 50 | 8.02 | 99.9% | 173 |
| future | 150 | 8.03 | 99.9% | 196 |
| future | 250 | 8.14 | 99.9% | 200 |
| future | 350 | 8.22 | 100.0% | 207 |
| future | 450 | 8.30 | 100.0% | 208 |
| future | 550 | 8.37 | 100.0% | 209 |
| future | 650 | 8.41 | 100.0% | 210 |
| future | 750 | 8.40 | 100.0% | 210 |
| future | 850 | 8.51 | 100.0% | 212 |
| future | 950 | 8.49 | 100.0% | 212 |
| future | 1050 | 8.44 | 100.0% | 212 |
| future | 1150 | 8.49 | 100.0% | 212 |
| future | 1250 | 8.50 | 100.0% | 212 |
| future | 1350 | 8.52 | 100.0% | 212 |
| future | 1450 | 8.69 | 100.0% | 212 |
| future | 1550 | 8.63 | 100.0% | 212 |
| future | 1650 | 8.73 | 100.0% | 212 |
| future | 1750 | 8.67 | 100.0% | 213 |
| future | 1850 | 8.72 | 100.0% | 213 |
| future | 1950 | 8.79 | 100.0% | 215 |
| future | 2050 | 8.83 | 100.0% | 215 |
| future | 2150 | 8.85 | 100.0% | 215 |
| future | 2250 | 8.86 | 100.0% | 215 |
| future | 2350 | 8.83 | 100.0% | 215 |
| future | 2450 | 8.87 | 100.0% | 215 |
| future | 2550 | 8.90 | 100.0% | 215 |
| future | 2650 | 8.92 | 100.0% | 215 |
| future | 2750 | 8.94 | 100.0% | 215 |
| future | 2850 | 8.94 | 100.0% | 215 |
| future | 2950 | 8.87 | 100.0% | 215 |

*생성: yard_rl.experiments.residual_costq — 원자료 residual_results.json*