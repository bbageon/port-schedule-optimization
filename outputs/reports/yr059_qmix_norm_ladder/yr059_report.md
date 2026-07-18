# YR-059 — 상태 정규화(scale-only·P90 동결) 후 QMIX 사다리 재실행

> OFF 대조 = 기존 yr013_qmix_ladder (같은 seed·프로토콜 — 차이는 정규화뿐).
> ⚠ 합성·가정 조건 (주장 게이트 YR-002/009).

| 정책(ON) | test 총비용 평균 |
|---|---|
| INDEP@1000 | 80.51 |
| INDEP@2000 | 80.51 |
| INDEP@500 | 81.74 |
| JOINT_ROLLOUT | 68.26 |
| QMIX@1000 | 139.25 |
| QMIX@2000 | 133.30 |
| QMIX@500 | 137.64 |

- **INDEP@1000: ON_vs_OFF**: Δtotal=-5.61 [-8.80, -2.74] (음수 = ON 개선)
- **INDEP@2000: ON_vs_OFF**: Δtotal=-5.06 [-7.16, -2.84] (음수 = ON 개선)
- **INDEP@500: ON_vs_OFF**: Δtotal=-4.39 [-7.44, -1.60] (음수 = ON 개선)
- **QMIX@1000: ON_vs_OFF**: Δtotal=+22.45 [+17.42, +27.41] (음수 = ON 개선)
- **QMIX@2000: ON_vs_OFF**: Δtotal=+28.51 [+24.33, +32.81] (음수 = ON 개선)
- **QMIX@500: ON_vs_OFF**: Δtotal=+20.84 [+15.28, +26.40] (음수 = ON 개선)