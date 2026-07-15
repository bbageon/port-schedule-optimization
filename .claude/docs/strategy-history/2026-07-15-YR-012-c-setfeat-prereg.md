# YR-012-c — Δ-net feature 확장 14→22 (집합 맥락 8) 사전등록

> 기록일: 2026-07-15 · 상태: **사전등록 — 본 실행 전 동결** · 사용자 승인: "8개 추가해서 한번 해볼래?"
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
