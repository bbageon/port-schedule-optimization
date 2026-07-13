# Exp-1 예비 PoC 결과 (합성 시나리오)

> ⚠ **가정 프로파일(assumed) + 합성 시나리오** 기반 예비 PoC.
> 실측 자료·CURRENT_RULE 미확보 상태로, 어떤 수치도 실제 운영 대비
> 개선율이 아니다. 시뮬레이터 실측 validation(YR-009) 전의 알고리즘
> 동작 검증 목적으로만 해석한다.

## 실행 조건

- **실험**: Exp-1 (정보=블록 도착 이후, sequence_only, 단일 YC)
- **프로파일**: DGT-ARMG (assumed=True)
- **시나리오**: 합성 v1 — GenParams(n_external=100, gate_out_share=0.6, n_vessel=8, fill_ratio=0.45, rehandle_risk=0.35, peak=False, horizon_s=28800.0, drain_window_s=7200.0, gate_offset_range_s=(300.0, 900.0), size_mix_ft40=0.7, eta_error_s=300.0)
- **seeds**: train 101..130 ×4epoch / test 301..312 (paired)
- **reward**: 정규화 Core Cost (탄소 미포함), w=(1,.3,.1,.1,.3) assumed, Scale=train FIFO fit 고정
- **주의**: CURRENT_RULE 미확보 — 휴리스틱 대비 비교만 유효

## 정책별 평균 (test seeds, 공통난수)

| 지표 | FIFO | LONGEST_WAIT | NEAREST_JOB | MIN_REHANDLE | QL_EXP1 |
|---|---|---|---|---|---|
| mean_wait_min | 22.78 | 23.65 | 17.81 | 15.40 | 20.41 |
| p95_wait_min | 49.08 | 50.48 | 91.59 | 54.80 | 57.61 |
| queue_area_h | 37.96 | 39.41 | 29.68 | 25.67 | 34.02 |
| tail_area_h | 11.70 | 12.28 | 15.01 | 9.35 | 10.36 |
| travel_km | 3.79 | 3.86 | 2.53 | 3.84 | 3.36 |
| rehandles | 58.67 | 58.50 | 58.33 | 56.42 | 58.17 |
| pre_rehandles | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| positionings | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| vessel_delay_min | 7.17 | 3.66 | 24.09 | 40.33 | 3.17 |
| completed_external | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| backlog | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| sla_exceed_count | 28.83 | 29.33 | 13.75 | 14.67 | 23.42 |

## Paired 비교 — 기준: FIFO (도착순 Baseline)

| 정책 | 지표 | 기준평균 | 대안평균 | Δ (95% CI) | 변화% | 방향일치 seed | 유의 |
|---|---|---|---|---|---|---|---|
| LONGEST_WAIT | mean_wait_min | 22.78 | 23.65 | +0.87 [+0.09, +1.64] | +3.8% | 9/12 | ✔ |
| LONGEST_WAIT | p95_wait_min | 49.08 | 50.48 | +1.40 [+0.12, +2.68] | +2.8% | 9/12 | ✔ |
| LONGEST_WAIT | queue_area_h | 37.96 | 39.41 | +1.45 [+0.16, +2.74] | +3.8% | 9/12 | ✔ |
| LONGEST_WAIT | travel_km | 3.79 | 3.86 | +0.07 [-0.18, +0.31] | +1.7% | 8/12 | — |
| LONGEST_WAIT | rehandles | 58.67 | 58.50 | -0.17 [-0.53, +0.20] | -0.3% | 1/12 | — |
| LONGEST_WAIT | vessel_delay_min | 7.17 | 3.66 | -3.51 [-7.04, +0.01] | -49.0% | 5/12 | — |
| NEAREST_JOB | mean_wait_min | 22.78 | 17.81 | -4.97 [-8.56, -1.38] | -21.8% | 10/12 | ✔ |
| NEAREST_JOB | p95_wait_min | 49.08 | 91.59 | +42.50 [+18.61, +66.39] | +86.6% | 11/12 | ✔ |
| NEAREST_JOB | queue_area_h | 37.96 | 29.68 | -8.29 [-14.27, -2.31] | -21.8% | 10/12 | ✔ |
| NEAREST_JOB | travel_km | 3.79 | 2.53 | -1.26 [-1.54, -0.99] | -33.4% | 12/12 | ✔ |
| NEAREST_JOB | rehandles | 58.67 | 58.33 | -0.33 [-1.16, +0.49] | -0.6% | 3/12 | — |
| NEAREST_JOB | vessel_delay_min | 7.17 | 24.09 | +16.91 [-22.34, +56.16] | +235.7% | 8/12 | — |
| MIN_REHANDLE | mean_wait_min | 22.78 | 15.40 | -7.38 [-10.95, -3.80] | -32.4% | 12/12 | ✔ |
| MIN_REHANDLE | p95_wait_min | 49.08 | 54.80 | +5.71 [-7.04, +18.46] | +11.6% | 6/12 | — |
| MIN_REHANDLE | queue_area_h | 37.96 | 25.67 | -12.29 [-18.25, -6.34] | -32.4% | 12/12 | ✔ |
| MIN_REHANDLE | travel_km | 3.79 | 3.84 | +0.05 [-0.15, +0.24] | +1.2% | 7/12 | — |
| MIN_REHANDLE | rehandles | 58.67 | 56.42 | -2.25 [-3.66, -0.84] | -3.8% | 8/12 | ✔ |
| MIN_REHANDLE | vessel_delay_min | 7.17 | 40.33 | +33.16 [-5.52, +71.84] | +462.2% | 9/12 | — |
| QL_EXP1 | mean_wait_min | 22.78 | 20.41 | -2.36 [-4.37, -0.36] | -10.4% | 9/12 | ✔ |
| QL_EXP1 | p95_wait_min | 49.08 | 57.61 | +8.53 [+1.37, +15.69] | +17.4% | 9/12 | ✔ |
| QL_EXP1 | queue_area_h | 37.96 | 34.02 | -3.94 [-7.28, -0.60] | -10.4% | 9/12 | ✔ |
| QL_EXP1 | travel_km | 3.79 | 3.36 | -0.43 [-0.67, -0.18] | -11.3% | 11/12 | ✔ |
| QL_EXP1 | rehandles | 58.67 | 58.17 | -0.50 [-1.63, +0.63] | -0.9% | 2/12 | — |
| QL_EXP1 | vessel_delay_min | 7.17 | 3.17 | -4.00 [-7.23, -0.78] | -55.8% | 5/12 | ✔ |

## 잠정 합격기준 점검 (03 §5.1 — 예비 PoC 버전)

- 안전·물리 제약위반 0 (invariant 상시 검사): 충족 — 위반 시 실행이 중단됨
- 처리량 ≥ 기준 99% (실측 100.0%): 충족
- 평균대기 5% 이상 감소 (실측 -10.4%): 충족
- P95 대기 악화 없음 (실측 +17.4%): 확인 필요
- 복수 seed 방향 일관 (9/12): 충족

*생성: yard_rl.experiments.report — 원자료 exp1_results.json*