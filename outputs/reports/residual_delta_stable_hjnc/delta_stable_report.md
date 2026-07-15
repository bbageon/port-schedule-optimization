# YR-012-b — Δ-net 학습 안정화 (replay buffer + target network)

> ⚠ 가정 프로파일 + 합성 시나리오. YR-012 진동(7.97~10.29) 해소가 목적 —
> 정책·feature 불변, 학습 절차만 DQN 표준화 (per-step 1 update 예산 등가).

- baseline (validation 선택): **IMMEDIATE_COST_GREEDY**
- baseline 을 유의하게 이긴 정책: **없음**

## locked test — paired vs IMMEDIATE_COST_GREEDY

| 정책 | 선택 ep | val_mean | mean_wait Δ [95% CI] | p95 Δ% CI 상한 | guardrail |
|---|---|---|---|---|---|
| DeltaNet[replay|sync500] | 3000 | 7.78 | +0.317 [+0.221, +0.426] | +10.1% | ❌/✅/✅/✅ |
| DeltaNet[replay|sync2000] | 50 | 7.68 | +0.158 [+0.082, +0.245] | +1.4% | ✅/✅/✅/✅ |
| ResidualDeltaNet[online](YR-012 ref) | — | nan | +0.107 [+0.033, +0.195] | -2.6% | ✅/✅/✅/✅ |

## checkpoint 곡선 (validation, arm 별) — 진동 폭이 1차 관찰 대상

| arm | episode | val_mean |
|---|---|---|
| DeltaNet[replay|sync500] | 50 | 8.06 |
| DeltaNet[replay|sync500] | 150 | 9.13 |
| DeltaNet[replay|sync500] | 250 | 8.89 |
| DeltaNet[replay|sync500] | 350 | 8.23 |
| DeltaNet[replay|sync500] | 450 | 8.57 |
| DeltaNet[replay|sync500] | 550 | 8.78 |
| DeltaNet[replay|sync500] | 650 | 8.47 |
| DeltaNet[replay|sync500] | 750 | 8.43 |
| DeltaNet[replay|sync500] | 850 | 8.36 |
| DeltaNet[replay|sync500] | 950 | 8.16 |
| DeltaNet[replay|sync500] | 1050 | 8.15 |
| DeltaNet[replay|sync500] | 1150 | 8.06 |
| DeltaNet[replay|sync500] | 1250 | 8.00 |
| DeltaNet[replay|sync500] | 1350 | 7.92 |
| DeltaNet[replay|sync500] | 1450 | 8.19 |
| DeltaNet[replay|sync500] | 1550 | 8.14 |
| DeltaNet[replay|sync500] | 1650 | 8.37 |
| DeltaNet[replay|sync500] | 1750 | 8.08 |
| DeltaNet[replay|sync500] | 1850 | 8.23 |
| DeltaNet[replay|sync500] | 1950 | 8.15 |
| DeltaNet[replay|sync500] | 2050 | 8.10 |
| DeltaNet[replay|sync500] | 2150 | 8.24 |
| DeltaNet[replay|sync500] | 2250 | 7.78 |
| DeltaNet[replay|sync500] | 2350 | 7.84 |
| DeltaNet[replay|sync500] | 2450 | 7.95 |
| DeltaNet[replay|sync500] | 2550 | 8.20 |
| DeltaNet[replay|sync500] | 2650 | 7.80 |
| DeltaNet[replay|sync500] | 2750 | 7.92 |
| DeltaNet[replay|sync500] | 2850 | 7.95 |
| DeltaNet[replay|sync500] | 2950 | 7.84 |
| DeltaNet[replay|sync2000] | 50 | 7.68 |
| DeltaNet[replay|sync2000] | 150 | 8.12 |
| DeltaNet[replay|sync2000] | 250 | 8.10 |
| DeltaNet[replay|sync2000] | 350 | 8.42 |
| DeltaNet[replay|sync2000] | 450 | 8.27 |
| DeltaNet[replay|sync2000] | 550 | 8.37 |
| DeltaNet[replay|sync2000] | 650 | 8.31 |
| DeltaNet[replay|sync2000] | 750 | 8.25 |
| DeltaNet[replay|sync2000] | 850 | 8.26 |
| DeltaNet[replay|sync2000] | 950 | 7.93 |
| DeltaNet[replay|sync2000] | 1050 | 8.00 |
| DeltaNet[replay|sync2000] | 1150 | 8.43 |
| DeltaNet[replay|sync2000] | 1250 | 8.16 |
| DeltaNet[replay|sync2000] | 1350 | 8.22 |
| DeltaNet[replay|sync2000] | 1450 | 8.27 |
| DeltaNet[replay|sync2000] | 1550 | 8.45 |
| DeltaNet[replay|sync2000] | 1650 | 8.03 |
| DeltaNet[replay|sync2000] | 1750 | 7.96 |
| DeltaNet[replay|sync2000] | 1850 | 8.25 |
| DeltaNet[replay|sync2000] | 1950 | 8.12 |
| DeltaNet[replay|sync2000] | 2050 | 7.90 |
| DeltaNet[replay|sync2000] | 2150 | 8.27 |
| DeltaNet[replay|sync2000] | 2250 | 7.93 |
| DeltaNet[replay|sync2000] | 2350 | 7.94 |
| DeltaNet[replay|sync2000] | 2450 | 7.87 |
| DeltaNet[replay|sync2000] | 2550 | 8.07 |
| DeltaNet[replay|sync2000] | 2650 | 7.92 |
| DeltaNet[replay|sync2000] | 2750 | 7.91 |
| DeltaNet[replay|sync2000] | 2850 | 7.94 |
| DeltaNet[replay|sync2000] | 2950 | 8.05 |

*생성: yard_rl.experiments.residual_delta_stable — 원자료 delta_stable_results.json*