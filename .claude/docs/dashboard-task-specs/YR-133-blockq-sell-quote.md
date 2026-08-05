# YR-133 — 1차 착수 사전등록 (2026-08-04 동결 · 기반 조사 반영)

> **2026-08-05 범위 정정**: 실제 TOS/ECS 연동은 연구에서 제외하며, 원자 확정은 이후부터 시뮬레이터 내부 트랜잭션만 뜻한다.

## 1차 범위 (동결 — event-only·반입·1건/epoch)
- **재사용**: MultiBlockTerminal 원자 이송(prepare/validate/commit/rollback·txn 멱등·용량/규격 fail-closed·전역 원장·G0 불변식·계약 테스트 22종)·yr105 러너(SF 실행정책·gate-in 정확 시각 review epoch·비용 채널 분해)·κ 동결(kappa_fit_v2p).
- **유일 변경**: review 규칙 = 혼잡 격차 → **견적 프로토콜**:
  ① source: 그 epoch gate-in 재배정 가능 후보 중 **top-1 OFFER**(OutRelief 최대,
    tie 는 작업 id 사전순) — OutRelief = v2 예측 KEEP 비용 j_truck(Ô_src+bias, A, D_T, κ_T)
  ② receiver(허용 블록 = 타 블록 전부·최소 InBurden·tie 블록명순): **InBurden** =
    가상 도착(A+이동+우회) 기준 수신 블록 큐 proxy 로 Ô_dst 계산한 j_truck — 공개 정보만
    (predict_gate_out 미도착 분기 로직 재사용·실현 미열람)
  ③ **TransferQuoteResolver(결정론)**: NetGain = OutRelief − InBurden − route/3600 −
    GAIN_MARGIN(0.5 승계) > 0 ∧ 본선 가드(소스 LOAD 최소 slack ≥ 0) → try_transfer
    **epoch당 최대 1건**. 예측 결측 = KEEP(fail-closed). quote 는 발행 epoch 전용
    (이월 금지 — 만료 원천 차단)·**이송/작업 ≤ 1회 mask**(transfer_count 필터 신설).
  ④ 견적 원장 전량 저장(t·작업·OutRelief·InBurden·NetGain·결정·version — 감사 가능).
- **명시 한계(1차)**: InBurden의 수신 기존 작업 지연·본선 항 미포함(후속 축), 양하·타이머·N>2 후속, 실제 TOS/ECS 연동은 범위 밖, 실행정책은 SF(학습 정책 이식은 별도 축).
- **비교(파일럿 8쌍·판정 아님)**: QUOTE vs KEEP(gain_margin=∞ — 계산 경로 동일·확정만
  차단, yr113 패턴). 신규 대역 y133-pilot(커서 906000 — 900k 이송 계열·910k 판정 계열
  회피), 지표 = terminal total(route 포함)·A→O(보고). **1차 성공 = 기능 가드 전부**:
  완주·불변식·정책 예외 0·epoch 1건 제한·mask·만료 위반 0·KEEP arm 이송 0.
  효과 확증(δ·표본·혼합 통계)은 후속 단일축.
- **승계 함정 3**: review 는 결정·wake 소진 뒤(기준선 24~32% 부풀림)·time_ledger
  포인터 이관·provided_eta 동반 시프트 — 전부 기존 코드가 처리(불변).

### 1차 파일럿 미통과·정정 v2 (2026-08-04 — 원장 열람 후 정정 표기)
- **미통과**: 안전 가드 5종 전부 통과(완주·예외 0·epoch 1건·이송 상한·KEEP 무이송)이나
  **"이송 발생" 요건 실패(0건)**. 원장 진단: ①견적 교환 정상(242건 계산·결측 0)
  ②**여유폭 0.5 단위 맥락 불일치** — yr099 는 에피소드 전체 비용 차 단위, 1차는 작업
  1건 예측 비용(발의 중앙 0.358) 단위라 최대 순이득 −0.08 로 구조적 발화 불가
  ③본선 가드 차단 359건(소스 셀 high·0.5 배 마감의 상시 음수 slack — 계약대로 작동,
  가드 유지).
- **정정 v2**: 여유폭 = **κ_T 예측오차 1σ (383.7s ≈ 0.107 비용단위)** 원칙 도출
  (예측오차 이내 이득 무시 = churn 방지 목적 동일 — 파일럿 원장 수치에 맞춘 값 아님).
  원장 열람 후 정정이므로 재실행은 **보정 파일럿**으로 표기(1차 산출물 pilot_v1.json
  박제 보존). 그 외 계약 전부 불변.
- **후속 감사 정정(2026-08-05)**: 0.5와 작업 견적은 같은 정규화 비용단위지만 추정 대상·규모가 달랐다. `κ_T=383.7초`도 실제 잔차 1σ(695.9초)가 아니므로 0.107은 기능 보정값이며 최적 여유폭 주장은 금지한다.

# YR-133 — Block Q 재배정 발의(SELL 별칭)·수신부담 견적·중앙 원자 확정

> 상태: **done — 시뮬레이터 기능 통과·비용효과 미확증** · 범위 정정 2026-08-05
> 선결: YR-147 정책 최적화 → YR-143 행동공간 → YR-146 배포 안전, YR-123·136 공통 비용계약
> 근거: [상세 결정이력](../strategy-history/2026-08-02-YR-133-판매수용-실환경-설계와-YR-143-무재배치-채택계약.md)

## 목적과 권한 경계

연구 입력에는 외부 TOS(터미널 운영 시스템)가 작업 생성·최초 블록 배정·허용 블록을 이미
정한 스냅샷이 주어진다고 가정한다. 연구는 TOS와 실시간 통신하지 않으며, 그 입력 중 아직
물리적으로 고정되지 않은 작업만 다음 구조로 재배정한다.

```text
외부 최초배정 스냅샷 ──> 원소유 Block Policy
                            │ 재배정 발의 + 빠졌을 때 절감액
                            ▼
                    TransferResolver
                      │          ▲
          수용부담 조회│          │수용불가/추가비용 견적
                      ▼          │
                  수신 Block Policy
                            │
               KEEP 또는 시뮬레이터 내부 A→B 원자 확정
```

- `tos_assigned_block_id`는 이력으로 보존하고 `execution_block_id`만 바꾼다.
- 외부 TOS는 최초 배정이 이미 끝났다는 입력 배경으로만 둔다. 실제 API·ACK/NACK·상태
  동기화와 현장 배포 가능성은 구현·평가하지 않는다.
- 실험 설정의 허용 블록 안에서 `TransferResolver`가 owner·queue·event·시간장부를 직접
  원자 변경한다. 이는 연구용 시뮬레이터 추상이며 실제 TOS 제어 주장이 아니다.
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
- 시나리오 설정의 `allowed_execution_blocks` 안의 receiver만 조회한다. 결측이면 KEEP한다.
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

## 시뮬레이터 원자 확정과 rollback

```text
OFFERED → QUOTED → PREPARED → VALIDATED → COMMITTED
                         └─ stale/용량변경/오류 → KEEP + 전체 rollback
```

1. source owner를 유지한 채 receiver slot·route/YT를 임시예약한다.
2. job/source/receiver/route version과 quote TTL을 다시 검사한다.
3. owner·양쪽 queue·arrival event·route·전역 A→O 시간장부를 한 transaction으로 바꾼다.
4. 실패하면 임시예약과 event를 복원하고 원 owner에 작업을 남긴다.

`txn_id`는 멱등이어야 하며 낡은 arrival event는 `(job_id, version)` 불일치로 무효화한다.
epoch당 최대 1건만 확정하고 다건 matching은 후속 단계로 분리한다.

## 현재 구현과 갭

- `MultiBlockTerminal`에는 shared clock, canonical job, owner/version, 용량예약,
  prepare/validate/commit/rollback의 기반이 있다.
- YR-133 1차에서 독립 `TransferQuoteResolver`와 source/receiver 계산 견적이 발화했고
  14건의 시뮬레이터 이송이 완료됐다. 비용효과는 미확증이다.
- 현재 자격은 반입·실제 gate-in 중심의 근사이며 설정 기반 allowed-block mask,
  quote TTL·불확실성, 양하 handover 사건은 후속 범위다.
- 1차 실행정책은 SF-SPT 규칙이다. 채택 PPO 정책과의 결합 성능은 아직 검증하지 않았다.

## 검증 게이트

1. **G0 원자성**: owner 정확히 1, orphan/중복 event 0, stale 명령 0, 모든 실패지점 rollback,
   A→O 장부 연속, 정상 YC 결정 선점 0, 결정론.
2. **G1 견적**: 직접 KEEP/TRANSFER 반사실 대비 OutRelief·InBurden·NetGain 부호·순위·
   보정오차. 실패하면 학습 quote 금지.
3. **G2 효과**: no-transfer / 현 혼잡규칙 / 계산 quote resolver를 신규 paired seed로 비교.
   반입-only와 양하-only를 분리하고 완주·backlog·안전을 hard guard로 둔다.
4. **G3 갱신**: event-only 통과 뒤 timer 추가 이득과 계산량을 단일축 검증한다.
5. **G4 학습(후속 YR-151로 개정)**: PPO SELL head를 별도 실험하되 YR-149 정보시점·5셀
   데이터 자격을 먼저 고정하고 KEEP·계산견적을 모두 대조한다.
P95(트럭 100대 중 오래 걸린 5대의 시간)는 보고용 진단이며 채택 veto로 쓰지 않는다.

## Evidence

2026-08-04 기능 파일럿은 `648d0c9`→`b9b8fd9`→`76c62d1`에 박제했다. 허용 결론은
“2블록·반입·SF 조건에서 견적과 시뮬레이터 원자 이송 경로가 발화했다”까지다. 2026-08-05
사용자 결정으로 실제 TOS 연동은 범위에서 제외했고, 후속 비용효과 검증은 YR-149가 맡는다.
