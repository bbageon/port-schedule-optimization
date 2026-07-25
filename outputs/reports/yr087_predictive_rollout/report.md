# YR-087 — 오라클 vs 현실형 예측 rollout (원자료)

전략·해석: [strategy-history/2026-07-25-오라클vs현실형-예측rollout.md](../../../.claude/docs/strategy-history/2026-07-25-오라클vs현실형-예측rollout.md)

## 방법

- **오라클** = `JointRolloutGreedy`(기존, deepcopy→실제엔진 = 진짜 미래 도착 사용, 상한).
- **현실형** = `PredictiveRollout`(미래 트럭 도착을 `provided_eta ± Uniform(300)` 예측 K=4표본 교체).
- horizon 1800s · numeraire_v1 · 가드 없음 · 2셀(high-loose·high-tight) × 5 seed (`BASE+500+i`).

## 선석초과(berth, 분) 원자료 — 낮을수록 본선 우수

| cell/seed | SF | oracle | realistic |
|---|---|---|---|
| high-loose/830600 | 63.5 | 45.0 | 43.6 |
| high-loose/830601 | 101.2 | 85.5 | 78.1 |
| high-loose/830602 | 46.7 | 45.6 | 47.5 |
| high-loose/830603 | 131.1 | 111.3 | 122.4 |
| high-loose/830604 | 109.8 | 97.6 | 81.9 |
| high-tight/830800 | 127.3 | 107.3 | 127.0 |
| high-tight/830801 | 151.5 | 139.7 | 128.0 |
| high-tight/830802 | 170.7 | 166.7 | 155.3 |
| high-tight/830803 | 154.7 | 130.5 | 123.7 |
| high-tight/830804 | 163.4 | 147.0 | 148.1 |
| **평균** | **122.0** | **107.6** | **105.6** |

## 짝지은 95% CI (n=10, 음수=현실형이 나음)

| 비교 | 점추정 [CI] | 판정 |
|---|---|---|
| 현실형 berth vs SF | −16.41 [−24.2, −8.61] | 유의 개선 |
| 현실형 berth vs 오라클 | −2.05 [−9.93, +5.83] | ≈ (구분 안 됨) |
| 현실형 트럭평균 vs SF | −5.03 [−6.64, −3.43] | 유의 개선 |
| 현실형 총비용 절감 vs SF | +16.79 = −18.6% [10.87, 22.7] | 유의 절감 |

## 판정·한계

**본선 이득이 현실형 예측(공개정보만)에서 유의 유지·오라클과 구분 안 됨 → 하이브리드 근거.** 한계:
도착시각 불확실성(ETA±5분)만 모델링(노쇼·물량구성 미검=YR-019)·2셀×5seed·K=4·eta_error 300 고정·
numeraire assumed(D5). 재현: `PredictiveRollout`(src/yard_rl/integrated/predictive_rollout.py)·
tests/integrated/test_yr087_predictive_rollout.py.
