# YR-067 — 상태 정규화 결합 판정 결과 (BC 제외)

> ⚠ 합성·가정 조건 (주장 게이트 YR-002/009). 페널티는 학습 표적 전용 —
> 아래 total_cost 는 전 arm 실제(비페널티) 비용. 판정 축: 퇴화 해소 여부
> (serve_when_available ≥ 0.25 — YR-044 건전성 계약) + 완료율 + 실제비용.

| arm | total_cost | 완료율 | backlog | serve_share | serve_when_avail | mean_wait(분) |
|---|---|---|---|---|---|---|
| TD_NORM | 84.59 | 1.000 | 0.0 | 0.074 | 0.064 | 4.62 |
| DIFF2400_NORM | 75.38 | 1.000 | 0.0 | 0.406 | 0.475 | 0.64 |
| CONTROL_TD | 70.11 | 1.000 | 0.0 | 0.112 | 0.094 | 1.59 |
| DIFF2400 | 78.91 | 1.000 | 0.0 | 0.351 | 0.400 | 1.44 |
| SF_SPT | 53.12 | 1.000 | 0.0 | 0.434 | 0.540 | 0.45 |
| FIFO | 65.43 | 1.000 | 0.0 | 0.438 | 0.526 | 0.40 |

- **TD_NORM_vs_CONTROL_TD**: Δtotal=+14.48 [+7.47, +22.18]
- **TD_NORM_vs_SF_SPT**: Δtotal=+31.47 [+24.55, +39.31]
- **DIFF2400_NORM_vs_DIFF2400**: Δtotal=-3.53 [-6.34, -0.69]
- **DIFF2400_NORM_vs_SF_SPT**: Δtotal=+22.26 [+19.82, +24.87]

*원자료: results json · test_results.json (seed별)*