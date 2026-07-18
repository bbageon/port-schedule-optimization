# YR-056 — COORD 협조 feature 경량 실험 결과

> ⚠ 합성·가정 조건 (주장 게이트 YR-002/009). 질문: 상대 의도·경합 feature 만으로
> RL 의 interference 격차(YR-054: 85%)가 줄어드는가 — QMIX 착수 판단 재료.

| 정책 | total_cost | interference | WAIT 수 | mean_wait(분) | 완료율 |
|---|---|---|---|---|---|
| COORD | 91.73 | 24.88 | 34.7 | 1.57 | 1.000 |
| NO_COORD | 88.03 | 22.86 | 33.3 | 1.08 | 1.000 |
| JOINT_ROLLOUT | 70.57 | 10.28 | 11.5 | 0.73 | 1.000 |

- **COORD_vs_NO_COORD**: Δtotal=+3.70 [+1.52, +5.93] · Δinterference=+2.02 [+0.49, +3.56]
- **COORD_vs_JR**: Δtotal=+21.15 [+19.19, +23.12] · Δinterference=+14.60 [+13.22, +15.98]
- **NO_COORD_vs_JR**: Δtotal=+17.45 [+15.72, +19.22] · Δinterference=+12.58 [+11.38, +13.86]

*원자료: yr056_results.json · test_results.json (seed별)*