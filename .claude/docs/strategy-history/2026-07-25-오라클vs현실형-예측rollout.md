# 오라클 vs 현실형 예측 rollout — 본선 이득이 공개정보만으로 유지 (하이브리드 근거)

> 사용자 지시(2026-07-25): 하이브리드를 바로 합치지 말고, YR-087 안에서 **기존 rollout 을 오라클로
> 재분류**하고 **공개정보 기반 현실형 예측 rollout** 을 먼저 검증하라. 현실형서도 본선 이득 유지 시 근거.
> **문헌보정 시뮬레이션(numeraire_v1)·D5 — 실 운영 개선 주장 아님.**

## 1. 오라클 확증 (코드)

[`_rollout_cost`](../../src/yard_rl/integrated/baselines.py#L163) 는 `copy.deepcopy(sim)` 후 **실제 엔진**으로
forward 진행(`scratch.run_until_decision`). deepcopy 큐엔 트럭 도착이 [`_seed_events`](../../src/yard_rl/integrated/engine.py#L158)
에서 **`actual_block_arrival`(진짜 도착시각)** 로 실려 있어, rollout 은 그 **진짜 미래를 소비** = **상한(오라클)**.
사용자 진단 정확. **단일 leak = 트럭 도착시각** — 본선은 이미 공개 cadence(`sts_move_interval`, 결정론)로
진행하지 truth 아님, STS 굶주림은 버퍼 동역학(시뮬)서 창발.

## 2. 현실형 예측 모델

ETA 오차 모델: [scenario_gen:238](../../src/yard_rl/integrated/scenario_gen.py#L238)
`eta = actual + Uniform(−eta_error_s, +eta_error_s)`, `eta_error_s=300`(±5분). 현실형 rollout =
deepcopy 후 **미래(미도착) 트럭의 도착 이벤트를 `provided_eta ± Uniform(300)` 예측으로 교체**(큐 재구성
+`actual_block_arrival` 갱신), **K=4 예측 시나리오** rollout 평가 평균. 후보 열거·feasibility·현재상태는
실제(알려진 것). `make_predicted_scratch`+`PredictiveRollout`(scratchpad).

## 3. 결과 (2셀 high-loose·high-tight × 5 seed = 10쌍, 짝지은 CI)

| 비교 | 점추정 [95% CI] | 판정 |
|---|---|---|
| **현실형 berth vs SF** | **−16.41 [−24.2, −8.61]** | **유의 개선** |
| **현실형 berth vs 오라클** | −2.05 [−9.93, +5.83] | **≈ (구분 안 됨)** |
| 현실형 트럭평균 vs SF | −5.03 [−6.64, −3.43] | 유의 개선 |
| 현실형 총비용 절감 vs SF | +16.79 = **−18.6%** [10.87, 22.7] | 유의 절감 |

평균 berth: SF 122.0 / 오라클 107.6 / **현실형 105.6**. 종종 현실형 < 오라클(K표본 평균이 단일미래보다 강건).

## 4. 판정

- **본선 이득이 현실형 예측에서 유의하게 유지**(−16.4 vs SF)되고 **오라클과 구분 안 됨** → rollout 의 본선
  개선은 **진짜 미래가 아니라 공개정보(제공 ETA±오차·본선 공개계획·현재 버퍼)만으로** 나온다. **하이브리드
  근거 성립.**
- **반응형 RL 과 정면 대비**: 반응형 Joint-Q 는 본선·P95·총비용 동시개선이 학습시드에 걸쳐 재현 안 됨(운) —
  현재 구조 한계를 강하게 시사하나 모든 반응형 불가능 증명은 아님. **rollout(오라클=상한)** 은 본선을 신뢰성
  있게 개선하고, **공개정보 예측 rollout 검증 후** 하이브리드 채택을 판단한다(사용자 최종 프레이밍).

## 5. 한계

- **도착시각 불확실성(ETA±5분)만** 모델링 — 오는 트럭 집합·목적지는 제공정보로 유지. **노쇼·물량구성
  불확실성(YR-019 BIASED/NO_SHOW/STALE)은 별개·미검.** eta_error 300 고정(더 크면 이득 침식 가능).
- 2셀×5seed·K=4·rollout 느림(~170초/에피소드, 배포는 분단위 결정이라 결정당 ~1-2s 무방). numeraire assumed.

## 6. 다음 (사용자 계획)

1. **[YR-089](../Dashboard/ready.md)**: 트럭 목표 S-B → 블록도착→작업완료 B-C 로 재정의 (행동공간 확정 선결).
2. 행동공간 확정 후 **하이브리드를 별도 Dashboard 작업으로 등록** — 최종구조: 평시 RL(트럭 빠른 선택) +
   본선위험 시 공개정보 예측 rollout(공동후보 재평가) + 동일 resolver(안전·물리·mandatory 제약 보장).
3. **[YR-041](../Dashboard/backlog.md)**: 목적계약 동결 후 본선·트럭 평균/P95·완주 재검정.
