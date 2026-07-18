# YR-063 — 차분 귀속 1-step Q 판정 결과

> ⚠ 합성·가정 조건 (주장 게이트 YR-002/009). 페널티는 학습 표적 전용 —
> 아래 total_cost 는 전 arm 실제(비페널티) 비용. 판정 축: 퇴화 해소 여부
> (serve_when_available ≥ 0.25 — YR-044 건전성 계약) + 완료율 + 실제비용.

| arm | total_cost | 완료율 | backlog | serve_share | serve_when_avail | mean_wait(분) |
|---|---|---|---|---|---|---|
| DIFF | 85.58 | 1.000 | 0.0 | 0.306 | 0.322 | 1.51 |
| CONTROL_TD | 70.11 | 1.000 | 0.0 | 0.112 | 0.094 | 1.59 |
| BC | 56.25 | 1.000 | 0.0 | 0.413 | 0.491 | 0.46 |
| SF_SPT | 53.12 | 1.000 | 0.0 | 0.434 | 0.540 | 0.45 |
| FIFO | 65.43 | 1.000 | 0.0 | 0.438 | 0.526 | 0.40 |

- **DIFF_vs_CONTROL_TD**: Δtotal=+15.47 [+8.53, +22.90]
- **DIFF_vs_BC**: Δtotal=+29.32 [+24.01, +34.93]
- **DIFF_vs_SF_SPT**: Δtotal=+32.46 [+27.39, +37.63]

*원자료: results json · test_results.json (seed별)*