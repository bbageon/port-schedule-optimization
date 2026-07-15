# YR-012-c — Δ-net feature 확장 14→22 (집합 맥락 8) 사전등록

> 기록일: 2026-07-15 · 상태: **완료 — 격차 역대 최소 +0.035·greedy 통계적 동률 첫 달성, 형식 승리 미달** · 사용자 승인: "8개 추가해서 한번 해볼래?" · 사전등록 원문(§1~§6) 불변, 결과는 하단 append
> 트랙 명시: **단일 야드 이동거리/greedy 격차 추격** (YR-034 최종 통합전략과 별개 — 사용자 확인)
> 상위: [YR-012 결과](2026-07-14-YR-012-residual-delta-net-prereg.md) · 근거: [YR-031-b 결과](2026-07-15-YR-031-b-oracle-pattern-prereg.md)

## 1. 결정 경위

1. YR-031-b 판정으로 처방이 데이터에서 직접 도출됨:
   - **H-A 지지 (AUC 0.852)**: oracle 이탈 시점은 관측 feature 로 예측 가능. 예측
     신호 **상위 5개 중 4개가 집합 맥락**(svc_mean −1.41·짧은작업비율 +1.31·
     반출비율 +1.19·svc_max +0.94) — **현 Δ-net 14-input 에 없는 정보**.
   - **H-B 기각 (집합 이득 +0.000)**: 이탈 선택은 후보쌍 차이만으로 재구성(0.993).
     후보 독립 argmin 골격 충분 — 구조 변경 불필요.
2. 이탈의 정체: 근소 동급 작업 미세 스왑 (anti-SPT 90%, service 차이 중앙값
   +9.5초). greedy 의 약점은 "동급 구간 순서 무감각" — 이를 감지하려면 후보
   집합의 작업시간 분포(min/mean/max)를 봐야 함.
3. 사용자 지시: "8개 추가해서 한번 해볼래?" — 처방 그대로 실행.

## 2. 설계 — 입력만 확장, 구조·학습 절차 불변

| 불변 (YR-012 승계) | 값 |
|---|---|
| 정책 | `Q_total = G + Δθ(x)`, argmin. G = 정확한 greedy 즉시비용, 비대체 |
| 구조 | MLP …→64→64→1, 출력층 **zero-init** (미학습 ≡ greedy) |
| 학습 | online TD (replay·target 없음 — YR-012-b 에서 잔차 Δ 는 online 상성 확인), γ0.95·lr1e-3·clip1.0·ε1/√ep·3,000ep·ckpt50·val 최소 선택 |

| 신규 (YR-012-c) | 값 |
|---|---|
| 입력 | 14 → **22** (14 base + 집합 8) |
| 집합 8 (env `set_raw`, 결정시 1회 산출·전 후보 공통) | 후보수·svc min/mean/max·reach min/mean·반출비율·짧은작업비율 — **oracle_pattern._set_aggregates 와 동일 정의** (H-A 신호 재현) |
| z-score 대상 | 14~19 (count·svc·reach) · 비율 20,21 passthrough. train FIFO fit 동결 |
| seed band | train 200000+3000 / val 210000+30 / test 220000+100 — 기존 전 band(10k~190k) disjoint, 코드 거부 |
| 비교군 | 휴리스틱 6종(val 선택 baseline) + **YR-012 online 14-dim 모델 동일 test 재평가** (feature 확장 효과 직접 대조) |
| 판정 | paired bootstrap 10,000 · **개선 = mean Δ CI 상한 < 0** (greedy 최초 초과) · P95≤+5%·완료·backlog·invariant guardrail |

## 3. 검토한 대안과 기각 사유

| 대안 | 기각 사유 |
|---|---|
| 집합 인코더/attention (후보 집합 통째 학습) | H-B 기각 (집합 이득 0) — 표현력 과잉, 불필요 |
| 결정시 소형 beam (학습가치+탐색) | H-B 기각으로 근거 없음. 비용만 큼 |
| replay+target 재도입 | YR-012-b 에서 잔차 Δ 는 online TD 상성 확인 — 역효과 |
| 8개 중 일부만 선별 추가 | H-A 상위 신호가 4/5 집합이라 전량 추가가 최소 판정 위험. 사후 ablation 은 개선 확인 후 |
| γ/lr 재탐색 | 이번 목적은 feature 효과 단분리 — 학습 하이퍼 고정 |

## 4. 한계 (사전 명시)

- online TD + 비선형 근사 수렴 보장 없음 (YR-012 진동 재현 가능) — 곡선·guardrail
  로 보고. 선택 프로토콜 winner's curse(YR-012-b 발견, YR-032)는 미해결 잔존.
- 개선이 나와도 "집합 정보" 외 net 표현력이 섞임 — YR-012 online(14) reference
  와 3자 대조로 해석하되 완전 분리 불가.
- H-A AUC 는 시점 예측가능성 — 회수 상금 크기(oracle 대비)는 별개, 본 실험이
  실측. 합성 + assumed 프로파일 (실운영 주장 불가, YR-009 전).

## 5. 비목표

집합 8 ablation · 구조 변경(H-B 기각) · replay/target · γ/lr 탐색 · 다른 band ·
YR-031-b·YR-012 결론 재해석 · YR-034 통합전략 요소.

## 6. 산출물

`outputs/reports/residual_setfeat_hjnc/` — feature_scaler(22)·seed_manifest·
checkpoint_curve·selections·test_results·setfeat_results·setfeat_report.md·
model_SetFeatDeltaNet.pt. 구현: `envs/direct_job_env.py`(set_raw 부착)·
`policies/residual_delta_net.py`(extract_features_with_set·22 모드)·
`experiments/residual_setfeat_experiment.py`·CLI `run-delta-setfeat [--quick]`.
실행 전 5차원 적대 리뷰 워크플로우 검증(greedy 등가·집합 무결성·인과·재현성·배관).

**리뷰 반영 (실행 전)**: 9-agent 적대검증에서 확정결함 1건(LOW) — env `set_raw` 평균이
`sum/n` 인데 oracle `_set_aggregates` 는 `fmean`(정밀합)이라 ULP 불일치("동일 정의" 계약
위배). env 를 `fmean` 으로 교정해 비트 단위 일치 강제(테스트도 approx→정확일치). 인과
누출·재현성·차원배관 3차원 0건, greedy 등가 2건은 검증에서 오탐 기각.

---

## 실행 결과 (2026-07-15 append — 사전등록 원문 불변)

- 실행: clean source `8c9c08e`, 소요 21.7분. 결과 커밋 `7928662` ·
  [리포트](../../../outputs/reports/residual_setfeat_hjnc/setfeat_report.md)

### 판정: 개선 미달 (CI 상한 +0.105 > 0) — 그러나 격차 역대 최소·greedy 통계적 동률 첫 달성

| 정책 | test mean (분) | Δ vs greedy [95% CI] | test p95 | guardrail |
|---|---|---|---|---|
| greedy (baseline) | 7.472 | — | 29.00 | — |
| **SetFeatDeltaNet[22]** | 7.507 | **+0.035 [−0.034, +0.105]** | 28.42 | 4/4 ✅ |
| YR-012 online[14] (동일 test) | 7.552 | +0.080 [+0.024, +0.138] | 26.48 | 4/4 ✅ |
| FIFO (참고) | 12.985 | — | 34.23 | — |

### 확정 — 집합 feature 처방(H-A)이 방향 입증: 격차 반감 + 동률 도달

- **CI 가 0 을 포함한 첫 정책** — 프로젝트 전체에서 학습 정책이 greedy 와
  통계적으로 구분 불가에 도달한 최초 사례 (그간 전부 CI 하한>0 유의 열세).
- 동일 test band 에서 **YR-012 online[14] 은 +0.080 (CI 전부 양수, 여전히 유의
  열세)** — 순수 차이는 입력 8 feature. 집합 맥락이 격차를 **+0.080→+0.035 로
  절반** 절감 (점추정). YR-031-b H-A 처방의 정량 확인.
- p95: SetFeat 28.42·YR-012 26.48 둘 다 greedy(29.00) 초과 개선 — guardrail 4/4.
- 직접 paired(22 vs 14): −0.044분 (22 우세)이나 55/100 seed — **약한 우위**
  (점추정은 개선, 통계적 결정력은 부족).

### 미달의 잔여 용의자

1. **online TD 진동 지속** (곡선 8.16~11.73) + **선택 winner's curse** (60 ckpt ×
   val 30일, YR-012-b 발견·YR-032) — 선택 ep550(val 8.16)이 최적 checkpoint 인지
   불확실. 선택 프로토콜 보완이 남은 +0.035 를 0 아래로 넘길 가능성.
2. **검정력** — Δ가 이제 노이즈 규모(±0.07)라 100일로는 CI 폭이 부호 확정에 부족.
   test 확대가 저비용 대안.
3. H-B 기각대로 구조는 불변 유지 — 표현 구조 아닌 위 둘이 남은 문턱.

### 파생 결정

1. **YR-032 (선택 프로토콜 보완) 우선순위 승격** — 이제 격차가 winner's curse
   규모(±0.07) 라 선택 규칙이 승부에 직접 영향. val 확대·이중검증으로 재선택 시
   동률→승리 전환 가능성 (재학습 없이 checkpoint 재선택).
2. 집합 8 ablation (어느 feature 가 주효했나)은 승리 확정 후로 유보 (비목표 유지).
3. 스코어보드 갱신: +1.283→+0.525→+0.454→+0.216→+0.083→**+0.035** (동률).
