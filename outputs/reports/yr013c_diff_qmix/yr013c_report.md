# YR-013c — 차분 표적 QMIX 판정 결과

> ⚠ 합성·가정 조건 (주장 게이트 YR-002/009). 페널티는 학습 표적 전용 —
> 아래 total_cost 는 전 arm 실제(비페널티) 비용. 판정 축: 퇴화 해소 여부
> (serve_when_available ≥ 0.25 — YR-044 건전성 계약) + 완료율 + 실제비용.

| arm | total_cost | 완료율 | backlog | serve_share | serve_when_avail | mean_wait(분) |
|---|---|---|---|---|---|---|
| DIFF_QMIX | 65.97 | 1.000 | 0.0 | 0.368 | 0.424 | 0.47 |
| DIFF2400_NORM | 75.38 | 1.000 | 0.0 | 0.406 | 0.475 | 0.64 |
| CONTROL_TD | 70.11 | 1.000 | 0.0 | 0.112 | 0.094 | 1.59 |
| SF_SPT | 53.12 | 1.000 | 0.0 | 0.434 | 0.540 | 0.45 |
| FIFO | 65.43 | 1.000 | 0.0 | 0.438 | 0.526 | 0.40 |

- **DIFF_QMIX_vs_DIFF2400_NORM**: Δtotal=-9.41 [-12.81, -5.86]
- **DIFF_QMIX_vs_SF_SPT**: Δtotal=+12.85 [+10.06, +15.62]
- **DIFF_QMIX_vs_CONTROL_TD**: Δtotal=-4.14 [-9.09, +0.91]

*원자료: results json · test_results.json (seed별)*