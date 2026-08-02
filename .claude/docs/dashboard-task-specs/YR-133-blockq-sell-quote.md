# YR-133 — Block Q 재배정 발의(SELL 별칭)·수신부담 견적·중앙 원자 확정

> 상태: backlog · 실환경 계약 정정 2026-08-02
> 선결: 단일 블록 실행정책 YR-142·143 판정, YR-123·136 공통 비용계약
> 근거: [상세 결정이력](../strategy-history/2026-08-02-YR-133-판매수용-실환경-설계와-YR-143-무재배치-채택계약.md)

## 목적과 권한 경계

외부 TOS(터미널 운영 시스템)는 작업 생성·최초 블록 배정·허용 블록을 담당하며 연구가
대체하지 않는다. TOS가 작업을 최초 블록의 Block Policy에 직접 전달한 뒤, 아직 물리적으로
고정되지 않은 작업만 다음 구조로 재배정한다.

```text
외부 TOS ──최초 배정──> 원소유 Block Policy
                            │ 재배정 발의 + 빠졌을 때 절감액
                            ▼
                    TransferResolver
                      │          ▲
          수용부담 조회│          │수용불가/추가비용 견적
                      ▼          │
                  수신 Block Policy
                            │
               KEEP 또는 TOS/ECS 승인 뒤 A→B 원자 확정
```

- `tos_assigned_block_id`는 이력으로 보존하고 `execution_block_id`만 바꾼다.
- 실제 TOS/ECS(장비 제어 시스템)가 재배정 API·ACK를 제공하지 않으면 자동 확정하지 않고
  shadow 추천으로만 평가한다.
- 단일 블록 실행순서 정책의 성능을 먼저 확정한 뒤 본 작업을 연다. 실행정책과 재배정정책을
  동시에 바꾸지 않는다.

## 판매·구매라는 말의 정확한 의미

`SELL`과 `BUY`는 설명용 시장 비유다. 실행 계약은 다음처럼 고정한다.

1. **원소유 Block Policy**
   - `NO_OFFER` 또는 `OFFER_TRANSFER(job, OutRelief, version, expires_at)` 중 하나를 낸다.
   - 발의해도 작업을 자기 대기열에서 지우지 않으며 commit 전까지 유일한 owner다.
2. **수신 Block Policy**
   - 자유롭게 사는 행동을 하지 않는다.
   - Resolver가 조회한 작업에 `NO_BID(reason)` 또는
     `QUOTE_IN_BURDEN(feasible, InBurden, uncertainty, version, expires_at)`만 응답한다.
3. **TransferResolver**
   - 모든 응답을 비교해 `KEEP` 또는 구체적인 `TRANSFER(job, A→B)`를 결정한다.
   - PPO·DQN·QMIX가 아니라 결정론적 제약검사·매칭·트랜잭션 계층이다.

첫 구현의 source top-1 발의는 계산량을 제한하는 MVP다. 터미널 전체 최적 재배정이라고
주장하지 않는다. source 절감액 2등 작업이 receiver 부담까지 합치면 더 좋을 수 있기 때문이다.

## 실행정책과 재배정정책 분리

- **ExecutionHead**: 현재 PPO가 한 블록 안 두 크레인의 공동 실행순서를 고른다.
- **TransferHead**: 작업목록 검토 사건에서만 재배정 발의를 만든다.
- **ReceiveQuoteHead**: 외부 조회를 받았을 때만 수용부담을 계산한다.

SELL을 `SERVE·PRE_REHANDLE·PREPOSITION`과 같은 크레인 후보목록에 넣지 않는다. 작업을
지금 처리하는 것과 다른 미래 작업을 재배정하는 것은 동시에 가능하며 서로 다른 결정이다.

## 최초 자격과 판매 창

| 작업 | 최초 판정 | 이유 |
|---|---|---|
| 반입 `GATE_IN/STORE` | **1차 대상** | 실제 gate-in 뒤 블록 도착·경로지시 전 3~7분 창에 목적지 변경 가능 |
| 본선 양하 `VESSEL_DISCHARGE/STORE` | **2차 대상** | STS 인계 뒤, YT 목적지·배차가 잠기기 전 별도 사건 필요 |
| 지정 트럭 반출 | 제외 | 지정 컨테이너가 이미 특정 블록에 있음 |
| 본선 적하 | 제외 | 적하 대상이 이미 특정 블록에 있음 |
| 재조작·이미 적재된 컨테이너 | 제외 | 블록 간 물리 이송이 되어 연구 질문이 달라짐 |
| 미지정 공컨 반출 | 제외 | 재고·선사 조건을 다루는 상위 TOS 문제 |

반입은 `actual_gate_in ≤ now < route/block_instruction_lock`에서만 허용한다. 양하는 나중에
`handover_ready ≤ now < YT_DISPATCHED`로 따로 검증한다. 이미 YT가 출발했거나 블록 도착·
slot/YC hard reservation이 생겼으면 fail-closed로 KEEP한다.

추가 자격은 다음을 모두 만족해야 한다.

- current owner·assignment version 일치, 열린 transaction 없음, 첫 PoC 이전횟수 ≤1.
- TOS가 준 `allowed_execution_blocks` 안의 receiver만 조회한다. 연구가 허용범위를 넓히지 않는다.
- receiver에 규격·높이·용량상 호환 slot과 YC 작업 가능성이 있다.
- ETA·소유권·허용블록 등 필수정보가 결측이면 KEEP한다.
- 냉동·위험물·중량·선사·세관 규칙은 YR-095에서 실제 자료로 추가한다. 그전에는 최소
  물리 타당성 환경이라는 주장만 허용한다.

## 견적 상태와 계산

각 Block Policy는 같은 인코더로 다음 정보를 본다.

- 블록: 현재·예상 queue, 평균·최장 대기, YC 잔여부하·고장, 장치율, 호환 slot 여유,
  본선 flow/slack, 예정 유입량.
- 작업: flow, 규격, 공개 ETA와 불확실성, 예상 블록 도착, deadline, 잠금까지 남은 시간,
  owner/version/이전횟수.
- 수신 가상상태: 해당 작업을 더했을 때의 도착시각·경로시간·예상 부하.

미래 실제 도착, 미통지 고장·계획변경은 보지 않는다.

```text
OutRelief(A,j) = J_A(KEEP j) - J_A(REMOVE j)
InBurden(B,j)  = J_B(ADD j) - J_B(NO ADD)
TransferCost   = route(origin→B) - route(origin→A) + 실제로 모델링된 변경비용
NetGain        = OutRelief - InBurden - TransferCost
```

`J`는 YR-136 v2의 같은 시간·비용 단위를 쓰며 혼잡점수를 비용에 다시 더하지 않는다.
반입의 origin은 gate, 양하의 origin은 quay handover다. 아직 A에 도착하지 않은 컨테이너를
A→B로 옮기는 비용으로 잘못 계산하지 않는다. 초기 판정은 계산식·짝지은 반사실로 견적의
부호와 보정을 검증하며, 상금이 확인된 뒤에만 QuoteNet 학습을 검토한다.

## 검토 시점

1차 실험은 **이벤트 전용**으로 단순하게 시작한다.

- 실제 gate-in, 양하 handover-ready, 유의한 ETA·계획 갱신, 장비 down/up,
  owner·capacity 변화가 있을 때만 검토한다.
- 같은 시각의 정상 YC 결정과 물리 사건을 먼저 처리한 뒤 transfer review를 연다.
- 상태 변화가 없거나 계산이 운영 latency budget을 넘으면 KEEP한다.
- 견적기가 통과한 뒤에만 `T_refresh` 타이머를 별도 단일축으로 추가한다.

## 원자 확정과 rollback

```text
OFFERED → QUOTED → PREPARED → VALIDATED → TOS/ECS ACK → COMMITTED
                         └─ stale/NACK/timeout/오류 → KEEP + 전체 rollback
```

1. source owner를 유지한 채 receiver slot·route/YT를 임시예약한다.
2. job/source/receiver/route version과 quote TTL을 다시 검사한다.
3. TOS/ECS에 재배정을 요청하고 ACK와 새 version을 확인한다.
4. owner·양쪽 queue·arrival event·route·전역 A→O 시간장부를 한 transaction으로 바꾼다.
5. 실패하면 임시예약과 event를 복원하고 원 owner에 작업을 남긴다.

`txn_id`는 멱등이어야 하며 낡은 arrival event는 `(job_id, version)` 불일치로 무효화한다.
epoch당 최대 1건만 확정하고 다건 matching은 후속 단계로 분리한다.

## 현재 구현과 갭

- `MultiBlockTerminal`에는 shared clock, canonical job, owner/version, 용량예약,
  prepare/validate/commit/rollback의 기반이 있다.
- 현재 `review_fn`은 중앙 혼잡도 규칙으로 직접 이송하며 독립 TransferResolver와
  source/receiver quote head는 없다.
- 현재 자격은 반입·gate-in 중심의 근사이며 TOS ACK, allowed-block mask, route/YT lock,
  quote TTL·불확실성, 양하 handover 사건은 미구현이다.
- 따라서 현재 PPO는 내부 실행순서만 정하며 **판매·수용 견적은 아직 수행하지 않는다**.

## 검증 게이트

1. **G0 원자성**: owner 정확히 1, orphan/중복 event 0, stale 명령 0, 모든 실패지점 rollback,
   A→O 장부 연속, 정상 YC 결정 선점 0, 결정론.
2. **G1 견적**: 직접 KEEP/TRANSFER 반사실 대비 OutRelief·InBurden·NetGain 부호·순위·
   보정오차. 실패하면 학습 quote 금지.
3. **G2 효과**: no-transfer / 현 혼잡규칙 / 계산 quote resolver를 신규 paired seed로 비교.
   반입-only와 양하-only를 분리하고 완주·backlog·안전을 hard guard로 둔다.
4. **G3 갱신**: event-only 통과 뒤 timer 추가 이득과 계산량을 단일축 검증한다.
5. **G4 학습**: 계산 quote의 상금과 라벨 품질이 확인된 뒤에만 공유 QuoteNet을 검토한다.

P95(트럭 100대 중 오래 걸린 5대의 시간)는 보고용 진단이며 채택 veto로 쓰지 않는다.

## Evidence

2026-08-02에는 실환경 설계 계약만 정정했다. 구현·실험 결과는 착수 후 별도 evidence로
갱신하며, 자동 commit은 실제 TOS/ECS 인터페이스가 확인되기 전 주장하지 않는다.
