# YR-032 — 계열 2 미래정보 잔차 Δ-net 단계 검증

- **Epic**: RL / **Priority**: 🟠 / **등록일**: 2026-07-14 / **상태**: Backlog·승인 대기
- **배경**: [YR-031](../../Dashboard/in-progress.md)은 외부트럭-only·순서선택 환경에서
  전 도착을 아는 beam도 greedy를 이길 상금이 있는지 측정한다. 현
  `DirectJobEnv`의 `future_raw`는 이미 도착한 잔여 후보만 뜻하며 제공 ETA는 아니다.
  [원래 H1/H2](../실험설계안-요약.md)는 제공 ETA를 이용한 사전 준비의 가치를 묻는다.
- **목표**: `Q_total=G+Δθ` 계열 2를 유지하면서 정보와 행동범위를 한 단계씩만 넓혀,
  (1) ETA 자체의 가치, (2) 사전 포지셔닝의 추가가치, (3) 선재조작의 추가가치,
  (4) 동일 정보·행동에서 RL의 추가가치를 각각 식별한다.

## 1. 정책 계약

\[
Q_{total}(s,a)=G(s,a)+\Delta_\theta(x(s,a)),\qquad
a^*=\arg\min_a Q_{total}(s,a)
\]

- `G`: 각 행동 때문에 즉시 늘어나는 정확한 queue-area 비용. 준비행동도 실제 소요시간
  동안 현재 대기열 비용을 발생시킨다. 이동·재조작은 새 가중치로 섞지 않고 KPI와
  guardrail로 기록해 YR-031의 평균대기 목적을 유지한다.
- `Δθ`: 미래 대기와 후속 작업효과만 보정한다. 출력층 zero-init으로 미학습 시 `Δ=0`.
- 학습 안정화: replay buffer + target network + mini-batch TD를 기본으로 사용한다.
  YR-012-b를 별도 평균축으로 실행할지는 YR-031 판정 후 결정하되 구현 계약은 재사용한다.
- 연속 feature를 사용하고 ETA를 bucket으로 뭉개지 않는다.

## 2. 정보 공개 계약

정책에 공개되는 미래정보는 `provided_at` 이후의 사전정보와 `provided_eta`뿐이다.
실제 도착시각은 도착 전 절대 공개하지 않는다.

```text
현재 정보: 현재시각, YC 위치, 현재 대기열, 작업시간, blocker
미래 정보: ETA까지 남은 시간, 5/15/30분 공개 작업 수,
           공개 작업량, 대상 bay, 예상 준비시간, 작업방향
```

- 제공 ETA는 외생 입력이며 예측모델을 학습하지 않는다.
- Exp-1/2 arm에는 ETA feature와 미래 Job ID가 노출되지 않는 자동검사를 둔다.
- 가변 후보는 `SERVE(current_job)`과 공개된 미래작업의 준비행동으로 표현한다.
- Action mask가 도착 전 `SERVE`를 금지하고, 물리적으로 불가능한 준비행동을 제거한다.

## 3. 단계별 실험

각 단계는 앞 단계 설정을 동결하고 행동 한 종류만 추가한다.

| 단계 | 공개정보 | 허용 행동 | 확인 질문 |
|---|---|---|---|
| 0 | 전 도착(oracle) | 현재 feasible 작업 선택 | YR-031: 순서게임 상금이 있는가? |
| A | 현재정보 vs 제공 ETA | `SERVE`만 | 같은 행동에서 ETA 자체가 유용한가? |
| B | 제공 ETA | A + `PRE_POSITION` | 도착 전 YC 이동의 순효과가 있는가? |
| C | 제공 ETA | B + `PRE_REHANDLE` | blocker 선처리의 순효과가 있는가? |

- YR-031이 `CLOSED`이면 A는 원래 H1의 정보 ablation으로만 보고, 평균개선을 위한
  대규모 튜닝은 하지 않는다. 실제 승부는 새로운 행동통로가 생기는 B/C에서 판단한다.
- `WAIT`는 A~C 핵심범위에서 제외한다. 운영규칙이 작업 존재 중 idle을 허용하고 B/C로
  ETA를 충분히 활용할 수 없다는 증거가 생길 때만 `WAIT_UNTIL_NEXT_ETA`를 별도
  사전등록한다. 임의시간 WAIT와 반복 WAIT는 허용하지 않는다.
- 각 단계 전에 perfect-information lookahead로 해당 행동공간의 상금을 점검한다.
  평균 상금 `<0.15분`이고 P95/SLA 이득도 없으면 그 단계의 RL 학습을 중단한다.

## 4. 공정 비교군

| 비교군 | 정보 | 행동범위 | 목적 |
|---|---|---|---|
| current greedy | 현재정보 | 단계별 현재정보 가능 행동 | 기존 기준 |
| ETA-aware heuristic/lookahead | 제공 ETA | 해당 단계와 동일 | 정보·행동이 같은 강한 기준 |
| Residual Δ-net (no ETA) | 현재정보 | A의 `SERVE` | RL 내 정보효과 대조 |
| Residual Δ-net (ETA) | 제공 ETA | 해당 단계와 동일 | 시험 정책 |

- `ETA Δ-net vs no-ETA Δ-net`: 정보효과.
- `ETA Δ-net vs ETA-aware heuristic`: RL 자체의 추가효과.
- `B vs A`, `C vs B`: 행동범위 추가효과.

## 5. 판정·검증

- train/validation/test는 기존 대역과 겹치지 않는 새 seed band로 동결한다.
- **정보효과 판정(A)**: 같은 학습기·`SERVE` 행동범위의 `ETA − no ETA`
  `mean_wait_min` paired 차이 95% CI 상한 `<0`.
- **RL 추가효과 판정**: 같은 ETA·행동범위의 `Δ-net − 최강 ETA-aware heuristic`
  paired 차이 95% CI 상한 `<0`.
- B/C는 ETA를 고정한 채 행동만 추가하고 각각 `B−A`, `C−B`로 판정한다. A의
  정보효과와 B/C의 행동효과는 서로 대신하지 않는다.
- guardrail: P95 변화율 CI 상한 `≤+5%`, completion 100%, backlog 0,
  invariant 위반 0. P95·30/60분 SLA 초과율은 별도 효과로 함께 보고한다.
- 정보효과와 행동효과도 동일 test day의 paired 비교로 보고한다.
- 선택 checkpoint와 모든 feature/action 공개시점을 기록해 미래정보 누출을 감사한다.

## 6. 범위 밖

- 본선·내부이송 작업, 다중 YC, 개별 YT/AGV fleet: [YR-013](YR-013-exp4-multi-yc.md).
- ETA 품질·오차·no-show 민감도: YR-019.
- 명시적 P95 후보필터: YR-029.
- ETA 생성·예측모델, 실제 항만 개선 주장: YR-009 게이트 전 금지.
- HJNC/DGT 작업분담 가정 추가: YR-002 운영자료 확인 전 금지.

## 7. 산출물·의존

- **코드**: 미래정보 후보/Action mask, 안정화된 residual Δ-net, ETA-aware 비교군.
- **리포트**: 단계 A/B/C별 oracle 상금·paired KPI·행동빈도·누출검사 결과.
- **의존**: YR-031 판정 후 착수 승인. 합성 PoC는 가능하나 운영 해석은 YR-009에 의존.
