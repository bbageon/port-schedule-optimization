# 06. 동적 후보 평가형 Deep Q와 다중 YC 협조

[← 상위 구현계획서](../../부산항_야드크레인_강화학습_구현계획서.md)

## 1. 결정 요약

최종 State는 차량·본선·컨테이너·레인·다중 YC의 연속값과 가변 후보를 포함하므로
`Q[state, action]` 표를 최종 정책으로 사용하지 않는다. 각 실행 가능 후보를 같은
신경망으로 평가하는 **동적 후보 평가형 Deep Cost-Q**를 사용한다.

```text
Q_cost(global, crane_i, candidate_j, candidate_context)
→ 후보 j를 선택했을 때의 예상 할인 누적 운영비용

action_i = argmin_j Q_cost(...),  j는 현재 mask를 통과한 후보
```

표준 reward 최대화 DQN과 수학적으로 동등하지만, 구현은 프로젝트의 비용함수와 맞춰
`Q_cost`와 `argmin`만 사용한다. 문서·코드·리포트에서 높은 Q와 낮은 Q의 의미를 섞지
않는다.

알고리즘 비교 순서는 다음 계약으로 고정한다.

1. Candidate DQN: 기본 Q-network 동작 확인용
2. **Candidate Double DQN: 최초 운영 후보이자 기본 학습기**
3. Dueling Candidate Double DQN: 가치·후보 이점 분리의 추가효과 검증
4. 다중 YC 중앙 공동할당 기준선
5. QMIX: 개별 후보 평가기가 검증된 뒤 붙이는 CTDE 협조 계층

QMIX를 처음부터 단일 거대모델로 만들지 않는다. 최종 State·Action·Reward 데이터
계약은 처음부터 통합형으로 두되, 로컬 후보 점수와 공동할당을 분리해 오류 원인을
추적한다.

## 2. 적용 범위

### 2.1 단일 YC 수직절편

현재 PoC는 외부트럭-only·단일 YC·`SERVE(job_id)`를 유지한다. 다음 계약부터 교체한다.

- bucket/Q-table 대신 연속 feature tensor
- 가변 Job 목록을 padding한 후보 tensor와 `candidate_valid_mask`
- 후보별 Q값을 한 번의 batch forward로 계산
- 기존 휴리스틱·oracle·YR-012 잔차망을 동일 시나리오에서 비교

이 단계는 항만 전체 최적화를 주장하지 않고 후보 평가기가 강한 greedy를 넘는지 확인한다.

### 2.2 통합 터미널 범위

동일 후보 계약에 외부트럭 반입·반출, 본선 적하·양하, 내부이송 인계, 재조작 선처리,
포지셔닝, 제한적 양보를 추가한다. 이후 같은 평가망을 YC별로 공유하고 공동할당 계층이
중복 Job·레인 충돌·비통과를 해결한다.

## 3. 관측 tensor 계약

```text
global_features       float32[B, G]
crane_features        float32[B, N_yc, F_crane]
candidate_features    float32[B, N_yc, K, F_job]
candidate_valid_mask  bool[B, N_yc, K]
candidate_job_ids     str[B, N_yc, K]       # 학습 tensor가 아닌 감사·실행용
elapsed_s             float32[B]
```

`K`는 고정 action 의미가 아니라 batch padding 상한이다. 실제 후보 수는 매 결정마다
달라지며 mask 밖의 padding은 선택·bootstrap·dueling 평균에서 모두 제외한다.

### 3.1 전역 feature

- 현재시각의 주기형 표현과 운영 진행률
- 외부트럭·본선·내부이송 대기량과 대기시간 분포 요약
- 본선 잔여량·Slack·STS 대기·목표 생산성 차이
- 레인별 점유·혼잡·예상 해소시간의 요약
- YC별 가용시각·위치·부하·간섭 가능성의 집합 요약

절대시각 하나만 주지 않고 마감·도착·해소까지 **남은 시간**을 함께 제공한다. 아직
공개되지 않은 ETA·미래 Job은 `InformationFilter` 뒤에서 tensor에 들어가지 못한다.

### 3.2 YC feature

- 정규화 위치, 가용 여부, 현재 작업 잔여시간
- 최근 처리량·빈 이동·간섭대기
- 할당 작업량과 예상 완료부하
- 인접 YC 위치·가용시각의 permutation-invariant 요약

### 3.3 후보 feature

- 작업유형과 작업원천: 트럭·본선·내부이송·야드정리
- 후보 대기시간, deadline/cutoff/ETA까지 남은 시간
- 접근시간·예상 서비스시간·종료 위치
- blocker·예상 취급횟수·재조작시간
- 연결 STS·YT/AGV/SC 대기와 본선 Slack
- 목적 레인 점유·혼잡 영향
- 선택 후 남는 후보의 작업량·시간분포·위치분포 요약

ID는 일반화 입력으로 쓰지 않는다. ID는 실제 Job 실행과 결정 추적에만 사용한다.

## 4. 후보와 Action

후보 생성기는 물리적으로 가능한 모든 작업을 만든 뒤 mandatory 후보를 먼저 보존한다.

```text
mandatory = {
  SLA 초과·임박,
  본선 마감 임박,
  이미 대기 중인 최장대기,
  실행 가능한 유일 작업
}
```

나머지는 결정론적 사전점수로 `K`까지 채운다. 사전점수는 학습정책이 아니라 계산량
제한용이며, train/validation/test에서 동일하다. mandatory가 `K`보다 많으면 모두
보존하고 해당 결정의 K를 확장하거나 계약 위반으로 중단한다. 조용히 잘라내지 않는다.

```text
CandidateAction = {
  SERVE(job_id),
  PRE_REHANDLE(job_id),
  REPOSITION(target_bay),
  WAIT_UNTIL(next_public_event)  # 운영상 허용될 때만
}
```

현재 PoC는 `SERVE`만 사용한다. 도착 전 실제 서비스, Hold, 안전거리·비통과 위반,
목적 슬롯 부재, 중복할당은 후보 생성과 dispatch 직전에 같은 `ConstraintEngine`으로
이중 차단한다.

## 5. 후보 평가 Q-network

```text
global_context    = GlobalEncoder(global_features)
crane_context     = CraneEncoder(crane_features_i)
candidate_embed_j = CandidateEncoder(candidate_features_ij)
set_context_i     = MaskedSetEncoder({candidate_embed_ik})

q_cost_ij = QHead([
  global_context,
  crane_context,
  candidate_embed_j,
  set_context_i
])
```

`MaskedSetEncoder`는 masked mean/max 또는 attention pooling으로 구현해 후보 순서에
불변이어야 한다. 이렇게 해야 현재 후보 하나의 특성뿐 아니라 대기열 구성도 볼 수 있다.
후보를 같은 네트워크로 평가하므로 작업 수가 달라도 파라미터 수는 변하지 않는다.

### 5.1 Double DQN target

비용 최소화 규약에서 online network가 다음 후보를 고르고 target network가 그 비용을
평가한다.

```text
a* = argmin_a Q_online(s', a)
y  = c_t + gamma_tau * Q_target(s', a*)
loss = Huber(Q_online(s, a_t), y)
```

terminal이면 `y=c_t`다. mask된 action은 online 선택과 target 평가에서 모두 제외한다.

### 5.2 Dueling 확장

```text
Q_cost(s,j) = V(s) + A(s,j) - masked_mean_j A(s,j)
```

후보가 한 개면 advantage 보정이 0이 되도록 한다. Dueling은 기본 Double DQN을 이긴다는
locked validation/test 증거가 있을 때만 채택한다.

## 6. 다중 YC 공동할당

각 YC가 독립 argmin한 결과를 그대로 실행하지 않는다. 동일 Job, 공통 레인, 비통과 때문에
공동 feasibility가 깨질 수 있다.

1. 각 YC-후보 쌍의 로컬 `Q_i`를 계산한다.
2. 중앙 resolver가 유효한 YC-Job matching만 구성한다.
3. 중앙집중 기준선은 로컬 비용 합이 최소인 matching을 선택한다.
4. QMIX arm은 선택된 로컬 utility와 global state를 monotonic mixer로 결합한다.
5. online joint action 선택과 target joint action 평가는 Double DQN 방식으로 분리한다.

```text
Q_tot = Mixer(Q_1, ..., Q_N, global_state)
```

resolver가 실행한 joint action을 replay에 저장한다. 독립 선택 후 사후 교체한 action을
원래 선택으로 기록하면 off-policy 오류가 생기므로 금지한다. 교착 시에는 환경이 정의한
결정론적 `YIELD/WAIT_UNTIL_EVENT`만 허용한다.

공동 제약이 있는 동안에는 완전한 독립 실행을 주장하지 않는다. QMIX의 개별 utility는
YC별 후보 점수 계산에 쓰지만, 실제 배정은 중앙 resolver가 최종 승인한다. 여기서 CTDE는
학습 시 전역정보를 쓰고 로컬 utility를 공유한다는 뜻이며, 운영 중 충돌검사를 각 YC가
제각기 생략한다는 뜻이 아니다.

## 7. 비용과 SMDP target

한 transition의 `cost_t`는 같은 구간에서 실제 증가한 정규화 운영비용이다.

```text
truck queue-area + long-wait area
+ YC loaded/empty travel + rehandle occupancy
+ STS·YT/AGV/SC wait + vessel/departure delay
+ lane congestion + YC interference + plan change
```

안전 위반은 비용항이 아니라 불가능한 action이다. 같은 결과를 여러 항으로 중복계상하지
않고, 모든 raw delta와 정규화 scale을 replay·리포트에 남긴다.

```text
gamma_tau = gamma_ref ** (elapsed_s / reference_s)
```

discounted 학습목표와 undiscounted 에피소드 총 운영비용을 모두 보고한다.

## 8. 학습 파이프라인

- Replay Buffer는 transition tensor·mask·실행된 Job ID·경과시간·비용 분해를 저장한다.
- 환경 warm-up 뒤 mini-batch 학습을 시작한다.
- target network는 고정 주기 hard sync 또는 사전등록된 soft update 중 하나만 사용한다.
- optimizer는 Adam/AdamW, learning rate는 `1e-4~1e-3` 범위에서 validation으로 선택한다.
- gradient clipping, NaN 검사, reward/cost scale drift 검사를 적용한다.
- epsilon은 학습에서만 감소시키며 평가·운영에서는 0이다.
- checkpoint는 확대 또는 이중 validation 계약을 사용하고 test를 보고 재선택하지 않는다.

YR-012-b에서 replay가 online 잔차학습을 악화시킨 사실은 보존한다. 여기서 replay·target은
일반 DQN의 비교 계약으로 다시 검증하며, 곡선 안정만으로 채택하지 않는다.

## 9. CPU·GPU 실행 계약

GPU는 필수가 아니다. `device=auto`는 CUDA가 있으면 학습 batch를 GPU로 보내고 없으면
CPU로 동일 코드를 실행한다.

- 이산사건 시뮬레이션과 후보 생성: CPU worker
- Replay Buffer: CPU 메모리
- Q-network batch forward/backward: GPU 권장
- validation/test: CPU 또는 GPU, 정책결과 동등성 검사
- 운영 추론: CPU 기본, 필요하면 GPU 서버 사용

GPU가 Python 이벤트 시뮬레이션 자체를 자동으로 빠르게 만들지는 않는다. 여러 CPU 환경이
경험을 만들고 learner가 batch를 GPU에서 학습하도록 분리한다. checkpoint는 device와
무관한 `state_dict + scaler + schema + config hash`로 저장하고 CPU에서 load 가능해야 한다.

## 10. 코드 배치

```text
src/yard_rl/policies/candidate_q_network.py
src/yard_rl/policies/candidate_double_dqn.py
src/yard_rl/policies/qmix.py
src/yard_rl/policies/replay_buffer.py
src/yard_rl/policies/joint_assignment.py
src/yard_rl/experiments/candidate_deep_q_runner.py
```

기존 `DirectJobEnv`, `DirectJobCandidate`, `ConstraintEngine`, 비용·paired 평가 코드는
재사용하되, 새 policy는 기존 `ResidualDeltaNetAgent`를 덮어쓰지 않고 별도 opt-in으로 둔다.

## 11. 필수 검증

- 후보 순서를 바꿔도 같은 Job을 선택하는 permutation 테스트
- 서로 다른 후보 수·padding·all-masked 오류 테스트
- mask된 action의 선택·bootstrap 0건
- Double DQN online 선택/target 평가 분리 단위테스트
- Dueling masked mean과 단일 후보 테스트
- 동일 Job 중복배정·레인충돌·비통과 0건
- `gamma_tau` 경과시간별 target 테스트
- 미공개 ETA·미래 Job 누출 0건
- CPU/GPU Q값·argmin 허용오차 내 일치
- checkpoint GPU 저장→CPU load round-trip
- 비용 구간합=에피소드 비용 항등식과 raw 비용 분해
- 같은 scenario manifest의 휴리스틱·DQN·Double DQN paired 비교

복잡한 모델의 채택 기준은 학습곡선의 매끄러움이 아니라 locked test의 총 운영비용 개선과
P95·본선·내부이송·backlog·안전 guardrail 동시 충족이다.
