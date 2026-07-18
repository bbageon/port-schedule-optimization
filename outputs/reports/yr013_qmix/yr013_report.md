# YR-013 — QMIX vs 독립 학습자 판정 결과

> ⚠ 합성·가정 조건 (주장 게이트 YR-002/009). 정보(COORD)·예산·실행경로 동일 —
> 차이는 학습 구조(CTDE mixer)뿐. 질문: 협조를 '학습 구조'로 가르치면 격차가 주는가.

| 정책 | total_cost | interference | WAIT 수 | mean_wait(분) | 완료율 |
|---|---|---|---|---|---|
| QMIX | 110.03 | 40.48 | 71.4 | 1.76 | 1.000 |
| INDEP | 85.25 | 24.87 | 31.2 | 0.71 | 1.000 |
| JOINT_ROLLOUT | 68.26 | 10.79 | 11.3 | 0.60 | 1.000 |

- **QMIX_vs_INDEP**: Δtotal=+24.77 [+21.30, +28.28] · Δinterference=+15.61 [+12.79, +18.30]
- **QMIX_vs_JR**: Δtotal=+41.76 [+38.70, +44.91] · Δinterference=+29.68 [+27.46, +32.02]
- **INDEP_vs_JR**: Δtotal=+16.99 [+15.45, +18.51] · Δinterference=+14.08 [+12.81, +15.34]

*원자료: yr013_results.json · test_results.json (seed별)*