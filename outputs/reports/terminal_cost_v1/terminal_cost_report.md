# 터미널 비용 identity·민감도 리포트 (terminal-cost-report-v1)

> ⚠ 전 항목 **assumed·synthetic** — 탐색 전용, 가중치 확정 금지. 실측 scale·검증은 YR-002/YR-009.
> 생성: YR-038 (build_cost_report). config=TERMINAL-COST-V1(dynamic λ), baseline=ReferenceDispatcher.

## 1. 비용 인과 identity (ledger)

| term | raw | ledger | residual |
|---|---:|---:|---:|
| truck_wait | 184.877778 | 184.877778 | 0.0 |
| long_wait | 0.0 | 0.0 | 0.0 |
| crane_travel | 0.0 | 0.0 | 0.0 |
| empty_travel | 513.5 | 513.5 | 0.0 |
| rehandle | 1.0 | 1.0 | 0.0 |
| sts_wait | 768.0 | 768.0 | 0.0 |
| transfer_wait | 10620.0 | 10620.0 | 0.0 |
| vessel_delay | 0.0 | 0.0 | 0.0 |
| depart_delay | 0.0 | 0.0 | 0.0 |
| lane_cong | 527.713889 | 527.713889 | 0.0 |
| interference | 420.377778 | 420.377778 | 0.0 |
| resequence | 0.0 | 0.0 | 0.0 |
| imbalance | 14779.430556 | 14779.430556 | 0.0 |

inactive terms(baseline 미발현): long_wait, crane_travel, vessel_delay, depart_delay, resequence
항목 중복계상 0 — 단일 accrue write-path 로 Σledger==episode_raw 구성상 성립.

## 2. train baseline scale fit (per-interval 평균)

| term | episode_raw_sum | n_int | per_interval | fallback |
|---|---:|---:|---:|:--:|
| truck_wait | 1866.27 | 25 | 74.651 |  |
| long_wait | 0.0 | 25 | 0.0 | ✓ |
| crane_travel | 0.0 | 25 | 0.0 | ✓ |
| empty_travel | 1456.0 | 25 | 58.24 |  |
| rehandle | 5.0 | 25 | 0.2 |  |
| sts_wait | 3840.0 | 25 | 153.6 |  |
| transfer_wait | 53100.0 | 25 | 2124.0 |  |
| vessel_delay | 0.0 | 25 | 0.0 | ✓ |
| depart_delay | 0.0 | 25 | 0.0 | ✓ |
| lane_cong | 2312.12 | 25 | 92.485 |  |
| interference | 2420.74 | 25 | 96.829 |  |
| resequence | 0.0 | 25 | 0.0 | ✓ |
| imbalance | 82839.32 | 25 | 3313.573 |  |

fallback=✓ 항은 baseline 미발현 → assumed scale 유지(문서화). fit scale 은 합성 proxy·잠정.

## 3. 정적 vs 동적 λ_vessel (paired, alt=dynamic, n=5 VAL)

- 총비용 diff(dyn-static): mean 0.0 CI[0.0,0.0] sig=False
- vessel_delay 기여 diff: mean 0.0 CI[0.0,0.0]
- 주: 합성 fixture 본선이 정시 완료(위험도 0)면 dyn=static — λ 효과는 지연 발생 구간에서만. 메커니즘 검증은 test_static_vs_dynamic_lambda_high_risk.

## 4. weight/λ 민감도 (YR-026 흡수)

| 축 | mean_diff(total) | CI |
|---|---:|---|
| vessel_delayx0.5 | 0.0 | [0.0,0.0] |
| vessel_delayx1.0 | 0.0 | [0.0,0.0] |
| vessel_delayx2.0 | 0.0 | [0.0,0.0] |
| vessel_delayx4.0 | 0.0 | [0.0,0.0] |
| truck_waitx0.5 | -0.2395 | [-0.3758,-0.1031] |
| truck_waitx1.0 | 0.0 | [0.0,0.0] |
| truck_waitx2.0 | 0.4789 | [0.2063,0.7516] |
| truck_waitx4.0 | 1.4368 | [0.6188,2.2548] |
| lam_x0.5 | -9.49 | [-9.49,-9.49] |
| lam_x1.0 | 0.0 | [0.0,0.0] |
| lam_x2.0 | 18.98 | [18.98,18.98] |

## 5. guardrail 분리
- 안전위반·mandatory 미수용은 **cost 아님** — mask(YR-037 JointResolution.mandatory_deferred) 유지. ledger 는 13 cost 항만(term∈COST_TERMS 폐쇄).
- 비선형 장기대기(§10.4 제곱항)·본선 6항(§10.6 d·e) 은 v1 미표현 (COST_TERMS frozen → SCHEMA bump 사안, 범위 밖).
