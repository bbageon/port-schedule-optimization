# YR-068 — 차분 표적 QMIX 본 시나리오 확전 판정

> ⚠ 합성·가정 조건 (주장 게이트 YR-002/009). 승자 처방(YR-013c) 무조정 확전 —
> G1: vs INDEP@2000(norm) / G2: vs JOINT_ROLLOUT / G3: vs SF_SPT.

| arm | total_cost | mean_wait_min | p95_wait_min | completion_rate | backlog |
|---|---|---|---|---|---|
| DIFF_QMIX | 125.789 | 3.111 | — | 1.000 | 0.000 |
| INDEP@2000 | 80.511 | 0.748 | 4.051 | 1.000 | 0.000 |
| JOINT_ROLLOUT | 68.264 | 0.604 | 3.357 | 1.000 | 0.000 |
| QMIX@2000 | 133.304 | 4.434 | 19.724 | 1.000 | 0.000 |
| SF_SPT | 83.077 | 0.511 | — | 1.000 | 0.000 |

- **DIFF_QMIX_vs_INDEP@2000**: Δtotal=+45.28 [+41.73, +48.83]
- **DIFF_QMIX_vs_JOINT_ROLLOUT**: Δtotal=+57.52 [+54.38, +60.75]
- **DIFF_QMIX_vs_SF_SPT**: Δtotal=+42.71 [+39.61, +45.76]

*원자료: yr068_results.json · test_results.json (seed별)*