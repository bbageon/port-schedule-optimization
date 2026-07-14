# YR-012-b — Δ-net 학습 안정화 (replay buffer + target network) 사전등록

> 기록일: 2026-07-15 · 상태: **사전등록 — 본 실행 전 동결** · 사용자 승인: "응 진행하고" (2026-07-15)
> 상위: [YR-012 결과](2026-07-14-YR-012-residual-delta-net-prereg.md) · 참조: [YR-031 Oracle](2026-07-14-YR-031-oracle-gap-prereg.md)

## 1. 결정 경위

1. **YR-012 확정 3**: Δ-net 격차 +0.083분 [+0.020, +0.145] — CI 하한이 문턱 직전.
   checkpoint 곡선이 7.97~10.29 로 심하게 진동 (online 단일표본 TD 특성) —
   "남은 용의자 = 학습 안정성"으로 명시, replay+target 을 자연 후속으로 예고 (§5).
2. **YR-031**: 전지적 beam 이 greedy 대비 **상금 하한 +0.182분 실재** 증명 (90/100일,
   혼잡일 편중) — 안정화로 진동만 걷어내면 회수할 목표가 있음이 보장된 상태.
3. 사용자 논의 (2026-07-15): 목표망의 원리 질의 3건 — ① 순환 잔존 (맞음, 감쇠기임)
   ② target 가변은 본질 (맞음) ③ 결론 = "정보 없는 변화의 즉시 피드백"만 차단.
   이해 확인 후 착수 지시.

## 2. 설계 — 정책 불변, 학습 절차만 교체

| 불변 (YR-012 승계) | 값 |
|---|---|
| 정책 구조 | `Q_total(s,j) = G(s,j) + Δθ(x(s,j))`, `j* = argmin` — G 는 정확한 greedy 즉시비용, 절대 비대체 |
| 입력 | 14차원 연속 x(s,j) (전역 5·후보 5·미래 4) · z-score scaler train-FIFO fit 동결 |
| 네트워크 | MLP 14→64→64→1 ReLU · **출력층 zero-init (미학습 ≡ greedy 계약 — target net 도 zero-init 복사라 유지)** |
| γ / lr / clip | 0.95 / 1e-3 / 1.0 |
| ε / 에피소드 / ckpt / 선택 | 1/√ep / 3,000 / 50 / validation 최소 mean |

| 신규 (YR-012-b) | 값 | 근거 |
|---|---|---|
| replay buffer | capacity 100,000 · batch 64 · min_replay 1,000 (warmup) | 상관 깨기 + 재사용. warmup 전 gradient 0회 (Δ≡0 → greedy 로 행동) |
| target network | bootstrap 만 담당, **N gradient step 마다 hard sync** | "정보 없는 변화" 차단 — 구간 내 target 고정 |
| N (sync 주기) grid | **{500, 2000}** 2점 | 단일점 오선택 위험 회피. 500≈5 에피소드, 2000≈20 에피소드 분량 |
| 계산 예산 | **per-step gradient update 1회 유지** | YR-012 와 업데이트 수 등가 — 개선분이 "더 배워서"가 아니라 "안정해서"임을 분리 |
| seed band | train 170000+3000 / val 180000+30 / test 190000+100 — 기존 8대역과 분리, 코드가 재사용 거부 | |
| 비교군 | 휴리스틱 6종 (validation 선택 baseline) + **YR-012 online 모델(.pt)을 같은 test band 재평가** (안정화 효과 직접 대조) | |
| 통계·판정 | paired bootstrap 10,000 · **개선 = mean Δ CI 상한 < 0** (최초 greedy 초과 시도) · guardrail (P95≤+5%·완료·backlog·invariant) | |

## 3. 검토한 대안과 기각 사유

| 대안 | 기각 사유 |
|---|---|
| full DQN 세트 (PER·듀얼링·double-Q 동시) | 변경 폭 큼 — 효과의 원인 분리 불가. replay+target 이 진동의 정설 처방이고 최소 조합 |
| soft update (Polyak τ) | hard sync 와 실질 등가이나 하이퍼 하나 추가 — N grid 로 충분 |
| N 단일점 | YR-030-b 교훈 (γ 단일점이었으면 0.95 근처 무효과를 몰랐음) — 2점이 최소 방어 |
| batch 크기 grid | 진동 처방의 본질은 상관 제거·과녁 고정 — batch 는 표준값 64 고정 |
| updates_per_step 증가 (재사용 극대화) | 예산 등가성 파괴 — "안정화 효과"와 "추가 학습 효과"가 섞임 |

## 4. 한계 (사전 명시)

- deadly triad (함수근사+bootstrap+off-policy)는 잔존 — 목표망은 감쇠기이지 수렴
  증명이 아님 (사용자 질의 ①의 결론 그대로).
- warmup 1,000 transition (~10 에피소드) 동안 학습 0 — 초기 곡선은 YR-012 와 비교
  불가 구간.
- replay 는 off-policy 정도를 높임 (과거 ε 시절 경험 재사용) — γ=0.95 가 완화.
- 합성 + assumed 프로파일 — 실운영 주장 불가 (YR-009 전).

## 5. 비목표

PER·듀얼링·double-Q·noisy net · feature 변경 · γ/lr 재탐색 · YR-012 결론 재해석 ·
oracle 패턴 feature 반영 (YR-031-b 별도).

## 6. 산출물

`outputs/reports/residual_delta_stable_hjnc/` — feature_scaler·seed_manifest·
checkpoint_curve(진동 폭이 1차 관찰 대상)·selections·test_results·
delta_stable_results·delta_stable_report.md·model_*.pt. 구현:
`policies/residual_delta_net.py` (opt-in 확장 — 기본값은 YR-012 동작 보존)·
`experiments/residual_delta_stable.py`·CLI `run-delta-stable [--quick]`.
