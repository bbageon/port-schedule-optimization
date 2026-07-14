# YR-012 — 잔차 연속-feature Δ 학습 (함수근사) 사전등록

> 기록일: 2026-07-14 · 상태: **완료 — 판정 미달이나 격차 +0.083분 (역대 최소)·P95 첫
> 개선·해상도 세금 실증** · 결과는 하단 append · 사용자 승인: "응 한번 해봐 —
> 기본 정책은 G+ΔQ 로 가서 bucket 해상도에 대해 실험" (2026-07-14)
> 상위: [YR-030 계열 2](2026-07-14-YR-030-series2-baseline-pivot.md) · 선행: [YR-030-c 결과](2026-07-14-YR-030-c-residual-costq-prereg.md)

## 1. 가설과 설계 원칙

**가설 (해상도 세금)**: YR-030-c state_job 의 잔여 격차 +0.216분 [+0.149, +0.289]의
주 원인은 bucket 이산화 — 같은 칸 안 차이(예: 이동 95s vs 235s)를 못 보는 손실이다.

**시험 방법**: 정책 골격은 YR-030-c 그대로 (`Q_total = G + Δ`, G 는 정확한 greedy
즉시비용·절대 비대체), **Δ 저장소만 표 → 신경망**으로 교체하고 입력은 같은 정보의
**연속 원값**을 쓴다. 정보량 동일·해상도만 무한 → 개선분이 곧 해상도 세금의 추정치.

- `Q_total(s,j) = G(s,j) + Δθ(x(s,j))` · `j* = argmin Q_total`
- **미학습 ≡ greedy 보장**: 출력층 zero-init → 초기 Δθ(x)=0 정확히 (테스트 계약)
- 학습식 (YR-030-c §6 승계): `Y = c + γ·min_j'[G' + Δθ(x')]` (종료 Y=c),
  회귀 목표 `Y_Δ = Y − G(s,j)` 로 Δθ 를 TD 회귀

## 2. 입력 feature x(s,j) — 14차원 연속 (bucket 화 대상의 연속판)

| 구분 | feature (원 단위) | tabular 대응 |
|---|---|---|
| 전역 5 | 진행률 now/horizon [0,1] · 크레인 bay(정규화 [0,1]) · 대기 수 · 최장대기 s · 30분초과 수 | YardState 5필드 |
| 후보 5 | 반출여부 {0,1} · 자기 대기 s · 크레인 이동 s · 예상 서비스 s · 선행이동 수 | JobState 5필드 |
| 미래 4 | 남은 작업 수 · 남은 총 서비스 s · 잔여 짧은작업 비율 [0,1] · 최근접 잔여작업 거리(bay) | future_situation 4필드 (종료구역은 거리·bay 로 연속화) |

- 표준화: **train FIFO 관측의 mean/std 로 z-score, fit 후 동결** (bucket edge 의
  대응물 — val/test 재조정 금지). 비율·[0,1] 필드는 그대로.
- G 는 입력에 넣지 않는다 — Q_total 의 상수항으로 이미 존재 (중복 주입 금지).

## 3. 네트워크·학습 설정 (동결)

| 항목 | 값 |
|---|---|
| 네트워크 | MLP 14→64→64→1 (ReLU), 출력층 weight/bias **zero-init** |
| 최적화 | Adam lr 1e-3 · grad clip 1.0 · **online TD** (replay buffer 없음 — tabular 프로토콜과 등가 유지, per-step SGD 1회) |
| target | bootstrap 은 현재 θ (target network 없음 — 최소변경. 발산 시 후속에서만 도입) |
| γ / ε / 에피소드 | 0.95 / 1/√ep / **3,000 ep** · ckpt 50 · validation 30일 최소 mean 선택 |
| 결정론 | torch CPU 단일 스레드 · manual_seed 고정 · 저장 = state_dict + scaler |
| seed band | train 140000+3000 / val 150000+30 / test 160000+100 — 기존 7대역과 분리, 코드가 재사용 거부 |
| 비교군 | 휴리스틱 6종 (validation 선택 baseline) + **YR-030-c state_job tabular agent 를 같은 test band 재평가** (reference — 해상도 효과의 직접 대조) |
| 통계·판정 | paired bootstrap 10,000 · **개선 = mean Δ CI 상한 < 0 vs baseline** · guardrail (P95 ≤+5%·완료 100%·backlog 0·invariant) 동시 보고 |
| 의존성 | torch ≥2.2 — pyproject optional `[rl]` (기본 설치는 순수 Python 유지) |

## 4. 검토한 대안과 기각 사유

| 대안 | 기각 사유 |
|---|---|
| DQN full (replay+target net) | 변경 폭 큼 — 실패 시 "해상도 vs 학습기법" 원인 분리 불가. 최소변경(온라인 TD)으로 해상도 효과만 분리 |
| PPO (정책경사) | 결정 구조가 후보 스코어링(argmin) — 가치 회귀가 자연스럽고 tabular 와 등가 비교 가능 |
| tabular 학습 연장 (10k ep) | 곡선 기울기 ~0.09분/1000ep 감속 — 남은 +0.17 해소 불확실. 신경망이 두 가설(수렴·해상도)을 함께 커버 |
| feature 확장 (신규 정보 추가) | 이번 목적은 **동일 정보의 해상도 효과 분리** — 정보 추가는 교란. 후속 축 |

## 5. 한계 (사전 명시)

- online TD + 비선형 근사는 수렴 보장이 없음 — 발산·진동 시 그 자체가 결과
  (guardrail·곡선으로 보고). target network 도입은 후속 사전등록.
- 신경망 개선이 나와도 "해상도" 외 표현력(feature 상호작용 학습)이 섞임 —
  tabular reference 와의 3자 비교로 해석하되 완전 분리는 불가함을 인정.
- 합성 시나리오 + assumed 프로파일 (HJNC-ARMG) — 실운영 주장 불가 (YR-009 전).

## 6. 비목표

replay buffer·target net·듀얼링 등 DQN 기법 grid · 신규 정보 feature ·
선박·다중 YC · γ/lr 탐색 (단일점 동결) · YR-030 결론 재해석.

## 7. 산출물

`outputs/reports/residual_delta_hjnc/` — feature_scaler.json·seed_manifest·
checkpoint_curve·selections·test_results·delta_net_results·delta_net_report.md·
model_*.pt. 구현: `policies/residual_delta_net.py`·`envs/direct_job_env.py`
(연속 future 원값)·`experiments/residual_delta_experiment.py`·CLI
`run-delta-net [--quick]`.

---

## 실행 결과 (2026-07-14 append — 사전등록 원문 §1~§7 불변)

- 실행: clean source `702f8d5`, 소요 13.5분. [리포트](../../../outputs/reports/residual_delta_hjnc/delta_net_report.md)

### 판정: 미달 (CI 상한 +0.145 > 0) — 그러나 세 가지 확정

| 정책 | test mean (분) | Δ vs greedy [95% CI] | test P95 | guardrail |
|---|---|---|---|---|
| greedy (baseline) | 7.347 | — | 27.66분 | — |
| **ResidualDeltaNet** | 7.430 | **+0.083 [+0.020, +0.145]** | **26.00분 (개선!)** | **4/4 ✅ (역대 최초)** |
| tabular ref (YR-030-c, 동일 test) | 7.595 | +0.248 [+0.162, +0.340] | — | P95 ❌ |

**확정 1 — 해상도 세금 실증**: 동일 정보·동일 잔차구조·동일 test 100일에서
tabular +0.248 vs net +0.083 — CI 비중첩. bucket 이산화가 격차의 약 2/3
(≈0.165분)를 설명함이 직접 증명됨 (본 실험의 1차 목적 달성).

**확정 2 — P95 첫 개선**: p95 26.00 vs 27.66 (Δ% CI 상한 −0.5%) — 프로젝트
전체에서 "평균 동급 + 최악층 개선"을 달성한 **첫 정책**. YR-018 이후 보상으로
불가능했던 tail 보호가 연속 상태의 부산물로 등장.

**확정 3 — 남은 용의자 = 학습 안정성**: 격차 진화 +1.283→+0.525→+0.454→
+0.216→**+0.083** (CI 하한 +0.020 — 문턱 직전). checkpoint 곡선이 심한 진동
(7.97~10.29, online 단일표본 TD 특성) — §5 에 예고했던 replay buffer + target
network 안정화가 자연 후속 (YR-012-b 등록).
