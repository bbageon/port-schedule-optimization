# YR-099 — TOS 배정 후 반입 재배정·중앙 Transfer Resolver

- **Epic**: RL / **Priority**: ⚪ / **등록일**: 2026-07-26
- **상태**: 미래 다중 블록 확장 작업. 현재 단일 블록 성능 트랙을 바꾸지 않는다.
- **상위 작업**: [YR-081 가변 크레인·다중 블록 확장](YR-081-variable-crane-yard-scaling.md)
- **결정 원본**: [2026-07-26 사용자 확정 아키텍처](../strategy-history/2026-07-26-YR-081-배정후-반입재배정-중앙-resolver-사용자확정.md)

## 배경

TOS는 연구 통제영역 밖의 최초 배정자다. 최초 블록 경매나 TOS 대체 계획기는 만들지 않는다.
다중 블록 단계에서는 TOS가 A에 배정한 **재배정 가능한 반입 작업**을 source BlockQ가
판매 후보로 평가하고, receiver BlockQ의 수용비용과 함께 중앙 resolver가 실제 `KEEP/A→B`를
확정한다. 단순 판매 요청만 남기거나 source queue에서 먼저 지우는 구조는 금지한다.

현재 엔진은 GATE_IN 도착 전 결정 epoch, job의 블록 소유권/version, 블록 간 원자적 queue
이동이 없다. 현재 CandidateQNet도 YC 실행후보만 채점한다. 이 작업은 해당 계약을 다중 블록
환경에 추가하되 기존 단일 블록 실행 Q와 안전 resolver를 보존한다.

## 현재 코드 seam

- `integrated/engine.py:194-203`: 외부트럭은 `BLOCK_ARRIVAL`만 event queue에 등록.
- `integrated/engine.py:244-266`: 현재 SMDP decision loop.
- `integrated/engine.py:351-370`: `WAITING/RELEASED` job만 YC 실행후보가 됨.
- `integrated/qnet.py:33`: CandidateQNet은 현재 YC 물리후보 비용망 — raw Q를 quote로 사용 금지.
- `integrated/predictive_rollout.py:24-54`: 공개 ETA 표본 scratch 생성 seam.
- `integrated/baselines.py:146-185`: 고정 시간창 paired rollout 비용 계산 seam.
- `integrated/resolver.py:43`: 기존 `CentralResolver`는 블록 내부 YC 공동배정용.
  새 블록 간 class는 반드시 `TransferResolver`로 분리한다.
- `domain/models.py:28`: Job에 현재 block owner/version 필드가 없음.

## 목표

```text
TOS 최초 배정
  → BlockQ별 OutRelief/InBurden quote
  → deterministic TransferResolver
  → KEEP 또는 concrete TRANSFER(A→B)
  → atomic commit/rollback
  → 최종 블록의 기존 ExecutionQ 처리
```

QMIX·PPO·LLM 통신 없이 명시적 한계비용과 제약 resolver로 배정 후 반입 재배정의 상금과
안전성을 먼저 검증한다.

## 필수 데이터 계약

### Job

- `tos_assigned_block_id`: 최초 배정, 불변 감사값
- `execution_block_id`: 현재 실행 소유 블록
- `assignment_received_at`, `reassignable: bool`, `allowed_execution_blocks`
- `reassignable_until` 또는 동등한 `transfer_lock_at`
- `assignment_version`, `transfer_count`, `transfer_history`
- 기존 `job_id`, 예약·실제 A/B/C/O 시각, deadline은 이전 뒤에도 유지

### 이벤트

- `TOS_ASSIGNMENT_RECEIVED`: 외부 배정과 공개정보 수신
- `TRANSFER_REVIEW`: 계획창 진입 또는 허용된 상태 급변
- `TRANSFER_PREPARED`, `TRANSFER_COMMITTED`, `TRANSFER_ABORTED`

`reassignable_until`이 없거나 마감·YC 예약·RUNNING 이후면 transfer 후보를 fail-closed로 막는다.
수신 블록까지의 실제 이동으로 B가 달라지므로 이전 전 A용 `actual_block_arrival`을 재사용하지
않는다. 정책에는 receiver별 예상 경로시간만 공개하고 실제 B는 실행 경로에서 새로 확정한다.

다중 블록은 독립 simulator 복사본의 느슨한 묶음이 아니라 shared clock, canonical
`JobRegistry`, 블록별 stack/crane/queue로 구성한다. 이전 전 queue에 남은 낡은 도착 event는
`assignment_version` 불일치로 무효화한다.

## BlockQ 계약

호환 블록은 파라미터를 공유하고 local state는 분리한다.

```text
OutRelief(A,j) = J_A(with j) - J_A(without j)
InBurden(B,j)  = J_B(with j) - J_B(without j)
```

- 입력: block profile, 현재 스택·YC·queue·본선상태, 작업 규격·예약·예상 블록도착·마감.
- **J 분해 (YR-100)**: `J = J_계산식 + J_잔여`. 본선 지연항은 스케줄 기반 계산식(YR-100 `ΔC_vessel`)으로 직접 산출하고, rollout/증류 대상은 **J_잔여(트럭 도착·큐 상호작용)만**이다. 본선 항을 증류에서 빼면 YR-087 관측별칭 위험과 반응형 본선 미학습이 quote로 전파되지 않는다. 블록 내 ExecutionQ와 **같은 공식을 공유**한다.
- 정답: J_잔여만 동일 공개정보 예측표본을 쓴 paired counterfactual rollout, 본선 항은 계산식.
- 출력: 한계비용 평균, 불확실성, quote 생성 시각·상태 version·TTL.
- `SELL_INBOUND`는 YC 물리 action이 아니라 transfer review의 source quote다.
- OFFER 자체에는 완료·비용감소 보상을 주지 않는다. commit된 결과만 실행 replay에 기록한다.

## TransferResolver 계약

```text
Gain(A→B,j)
 = OutRelief(A,j) - InBurden(B,j)
   - RouteCost(A→B,j) - RiskMargin(A→B,j)
```

- 후보: `KEEP(A,j)`와 eligible receiver별 `TRANSFER(A→B,j)`.
- 최적화: 양의 Gain 최대화. 동시 다작업은 batch matching으로 receiver 용량 중복을 방지.
- 제약: job owner 정확히 1, 슬롯·규격·높이·용량·경로·마감·공유자원 예약, 중복 0.
- 이미 YC에 예약·배정·실행된 작업과 지정 반출·본선 적하(LOAD, 위치 고정)는 이전 금지. 본선 양하(DISCHARGE)는 store라 배치 가능하나 stowage/그룹 모델(YR-095) 후 별도 개방(현재 범위 밖). 재배정 taxonomy는 [YR-081](YR-081-variable-crane-yard-scaling.md) 참조.
- tie-break: `(job_id, source_block_id, receiver_block_id)` 완전순서.
- 같은 snapshot과 quote에는 같은 결과를 내는 순수·결정론적 resolver.
- 첫 headroom은 epoch당 transfer 최대 1건을 전수비교한다. 다작업 개방 전까지 ping-pong을
  막기 위해 `transfer_count≤1`을 적용하고, 이후 batch matching에서만 동시 이전을 연다.

## 원자적 transaction

1. source는 commit 전까지 owner와 queue entry를 유지한다.
2. receiver 슬롯·용량·경로를 임시예약한다.
3. job/blocks/ledger version을 다시 검사한다.
4. 한 transaction에서 owner·source queue·receiver queue·route를 변경한다.
5. receiver 도착 event를 새 version으로 발행하고 기존 event를 stale 처리한다.
6. 전부 성공하면 commit하고 임시예약을 실예약으로 승격한다.
7. 하나라도 실패하면 모든 임시변경을 되돌리고 owner=A를 유지한다.

`KEEP`, 수신자 없음, 순이득≤0은 정상 결과다. commit 중 stale/failure만 rollback으로 센다.

## 실험 설계

### 비교군

- A: TOS 최초 배정 고정 + 블록별 독립 실행
- B: 관측 가능한 단순 부하규칙 재배정
- C: 공개정보 paired-rollout quote + 중앙 resolver
- D: C의 quote를 학습한 공유 TransferQuoteQ + 같은 resolver
- 상한 참고군: 완벽 미래 quote. 배포·채택 금지

### 단계 게이트

- **G0 계약**: owner 유일성·장부연속성·commit/rollback·결정론·정보안전 tests 통과.
- **G1 상금**: 같은 local policy를 고정한 채 C가 A보다 terminal total cost를 paired CI로
  유의 개선하고 필수 guard 통과.
  실패하면 D를 만들지 않는다.
- **G2 분해 타당성**: local `OutRelief-InBurden-route`와 직접 다중블록 반사실
  `C_terminal(KEEP)-C_terminal(A→B)`의 부호·순위·오차를 비교. 불일치가 크면 D 금지.
- **G3 근사**: D가 C의 transfer 순위와 성능에 근사하고 A/B를 초과하며 결정시간을 줄임.
- **G4 조건 일반화**: 구조군·부하별 결과를 분리 보고. 하나의 터미널 평균으로 합치지 않음.

정량 비열등 margin은 실제 SLA 근거를 확보한 뒤 결과 열람 전에 사전등록한다.

## 필수 지표·불변식

- 최종 KPI: gate-in→gate-out `A→O` 평균·P95, terminal total cost.
- 통제 KPI: block-arrival→job-done `B→C`, berth overrun, rehandle, 추가 차량거리.
- 운영량: transfer/keep/reject/rollback/stale quote 수, decision latency, message 수.
- guard: 완료율 100%, backlog 0, owner 없음 0, 이중 owner 0, 중복 도착/event 0,
  transfer lock 이후 이전 0, 시간 reset 0, 규격·슬롯·크레인 물리위반 0.

평가창 종료의 RUNNING 작업과 transfer-pending 작업은 unfinished backlog로 센다.

## 구현 순서

1. 다중 블록 shared clock·canonical job registry·owner/version 계약. 기능 off일 때 기존
   단일 블록 golden byte 불변을 먼저 고정.
2. GATE_IN 배정통지·review epoch와 transfer eligibility mask.
3. atomic transaction·실패주입·불변식 tests.
4. paired predictive rollout의 source/receiver quote와 no-transfer headroom.
5. G1 통과 때만 공유 TransferQuoteQ를 증류.
6. batch resolver·공유 YT/도로 외부효과 stress와 G2/G3 평가.

## 의존

- YR-014: 현 단일 블록 정책과 비용계약 최종 판정
- YR-082/083: 실제 블록 구조·런타임 자격
- YR-081: 가변 크레인·다중 블록 환경과 독립 블록 기준선
- YR-089: A/B/C/O 시간장부
- YR-093: 공개정보 예측 rollout 정보안전
- YR-100: ExecutionQ와 공유하는 본선 비용 계산식 — J_계산식의 본선 항을 재계산 말고 재사용

## 범위 밖

- TOS 최초 배정·최초 경매·TOS 알고리즘 수정
- 선석·QC·YT 전체계획기 재구축
- 지정 반출 target 변경, 미지정 공컨 terminal request, 본선 job 이전
- YC 자체의 블록 간 이동
- QMIX/PPO/자유형 agent 통신/LLM 감독
- 운영 인터페이스 확인 전 실운영 자동 재배정 주장

## 재검토 조건

- 실제 TOS·게이트 경로가 실행 블록 변경을 받을 수 없음
- G1에서 공개정보 재배정 상금 없음
- scalar quote가 direct terminal 반사실의 부호·순위를 반복적으로 틀림
- transfer churn·추가 이동이 절감액을 상쇄

마지막 두 조건이 실증될 때만 중앙 joint scorer·coordination graph·QMIX를 별도 ablation으로
검토한다. 어떤 학습기를 쓰더라도 원자적 소유권 변경과 물리 제약은 resolver에 남긴다.

## 산출물

- multi-block ownership/event/schema와 TransferResolver
- quote rollout·TransferQuoteQ checkpoint
- 불변식·결정론·정보안전·실패주입 tests
- seed별 원자료, headroom/근사/조건 일반화 보고서
