# YR-133 — Block Q 발의 판매·양방향 견적

> 상태: backlog · 등록/설계 정정 2026-07-30
> 근거: [BlockQ-v2 로드맵](../strategy-history/2026-07-30-BlockQ-v2-로드맵-10단사다리.md)

## 목적

외부 TOS가 최초 배정한 작업은 해당 블록의 Block Q가 직접 받는다. 이후 재배정 가능한
미적재 작업에 한해, 원소유 Block Q가 “이 작업이 빠지면 내 비용이 얼마나 줄어드는가”를
계산해 판매를 발의한다. TransferResolver는 발의된 작업만 수신 블록들에 조회하고,
터미널 전체 비용이 줄 때만 원자적으로 재배정한다.

## 정책 소유권과 행동 공간

SELL은 **Block Q가 소유하는 행동**이다. 다만 크레인 물리 행동과 같은 평면 후보목록
`CandidateKind`에 넣지는 않는다. 같은 Block Q 안에서 결정 시점별 행동집합을 분리한다.

1. **ExecutionHead — 크레인 실행 결정시점**
   - 블록 안 공동 실행후보를 평가한다.
   - `SERVE·PRE_REHANDLE·REPOSITION·구조적 WAIT`와 기존 내부 안전 resolver를 사용한다.
2. **SellProposal/OutReliefHead — 작업목록 검토시점**
   - `NO_PROPOSAL` 또는 `PROPOSE_SELL(job, OutRelief, version, TTL)`을 고른다.
   - SELL의 실행 의미는 “견적을 붙여 TransferResolver에 발의”이며 곧바로 이송하는 것이 아니다.
3. **InBurdenQuoteHead — 수신 조회시점**
   - 수신 후보 Block Q가 `InBurden(job)`을 같은 비용 단위로 응답한다.

실행과 판매는 같은 시각에 함께 가능할 수 있다. 둘을 한 목록에 섞으면 “지금 작업할지,
다른 미래 작업을 팔지”라는 거짓 양자택일이 생기므로 분리한다. 소유 주체는 하나지만
`decision_kind`에 따라 행동집합이 달라지는 event-conditioned subpolicy다.

## 판매 판정

```text
OutRelief = J_source(KEEP j) - J_source(REMOVE j)
InBurden  = J_receiver(ADD j) - J_receiver(NO ADD)
NetGain   = OutRelief - InBurden - TransferCost
```

`NetGain > 0`인 최선 수신처가 있으면 `TRANSFER(A→B)`, 없으면 `KEEP`이다. 용량·소유권·
version·잠금·원자성은 기존 `MultiBlockTerminal`의 prepare→validate→commit/rollback으로
재검사한다. 실패하거나 수신자가 없으면 작업은 원블록에 남는다.

작업 종류와 스케줄 정보는 원인 정보로 사용할 수 있다. 다만 “본선이면 항상 우선” 같은
유형 고정 규칙은 두지 않고, 작업별 한계비용과 실행 가능성으로 비교한다.

## 갱신 시점 계약 — 이벤트 + 타이머

운송 이벤트는 생길 수도, 한동안 없을 수도 있으므로 다음 혼합 계약을 목표로 한다.

```text
review_due =
    material_event
    OR (now - last_review_time >= T_refresh)
```

- **이벤트 즉시 검토**: 실제 게이트 진입, 공개 ETA의 유의한 갱신, 본선 계획 변경,
  장비 고장·복구, 작업 완료·블록 도착·이송 완료처럼 견적 또는 자격이 달라지는 사건.
- **타이머 보완 검토**: 사건이 없어도 `T_refresh`가 지나면 합성 `ReviewEpoch`를 예약한다.
  타이머는 재평가 기회만 열며 판매나 이송을 강제하지 않는다.
- 상태·작업 version이 그대로면 `NO_PROPOSAL`로 빠르게 종료하고, 같은 작업의 반복 발의를
  막는 cooldown/TTL을 둔다.
- 첫 구현은 epoch당 `NO_PROPOSAL` 또는 **최대 1건**만 발의·확정한다. 확정 뒤 상태를
  갱신해 다시 견적한다. 작업 전체의 KEEP/SELL 이진 조합은 후속 상호작용 검증 전 금지한다.
- 같은 시각에는 물리 이벤트와 이미 열려야 할 크레인 결정/wake를 먼저 소진한 뒤 review를
  연다. review가 정상 작업 결정을 선점해 크레인을 놀리는 과거 결함을 재발시키지 않는다.
- 실제 블록 도착 또는 잠금 이후에는 판매 창을 닫는다.

`T_refresh`는 결과를 보고 바꾸지 않고 사전등록한다. 견적기가 먼저 검증된 뒤
`EVENT_ONLY`와 `EVENT+TIMER`를 **별도 단일축**으로 비교해 타이머의 순효과를 확인한다.

## 최초 범위와 현재 구현 갭

- 최초 범위는 현재 엔진이 실제로 지원하는 **미적재 `GATE_IN` 반입 작업**이다.
- `VESSEL_DISCHARGE`(본선 양하)는 gate-in 사건이 없으므로 안벽 인계/ETA 갱신 트리거를
  별도로 정의한 뒤 확장하며, 첫 결과에 섞지 않는다.
- 현재 `ReviewEpoch`는 `time`만 가지며, `MultiBlockTerminal`은 실제 gate-in 시각만
  전 블록에 예약한다. 중앙 `review_fn`이 혼잡도 차이로 이송 여부를 직접 정한다.
- 목표 구현은 `ReviewEpoch(time, reason, affected_jobs, state_version)`과 합성 타이머,
  Block Q의 두 견적 head, 결정론적 `TransferResolver`를 추가하고 기존 원자 집행을 재사용한다.
- 현재 코드에는 독립 `TransferResolver` 클래스가 없고 `MultiBlockTerminal`이 집행 계약만
  담당한다. 문서의 Resolver는 위 판정 서비스를 새로 구현할 목표 이름이다.

## 선결

- BlockQ-v2가 후보 **순위**를 구분하고, 판매 문턱에 필요한 비용 크기는 별도로 보정할 것.
- [YR-123](YR-123-common-cost-curve-api.md): OutRelief·InBurden·TransferCost를 같은
  numeraire·시간창·종결 규칙으로 계산할 공통 비용 API.
- [YR-136](YR-136-smooth-cost-contract-v2.md): 실제 정책 견적에는 YR-123 v1의
  계단형·본선 33이 아니라 트럭 `1→2`, 본선 `0→10` 점증곡선 v2를 연결한다.
- 학습 전에 계산식/짝지은 반사실로 다음 분해가 맞는지 확인할 것.

```text
분해오차 = 실제 터미널 Δ비용 - (-OutRelief + InBurden + TransferCost)
```

## 검증 게이트

1. **시간·원자성**: 기능 OFF 골든 불변, 결정시각 엄격 증가, review의 실행결정 선점 0,
   owner 공백·중복 0, 실패 시 KEEP.
2. **견적 타당성**: source/receiver 각각 부호 일치·보정오차·선택 후 regret을 새 대역에서
   측정한다.
3. **분해 타당성**: 위 분해오차가 사전 허용범위 안인지 확인한다.
4. **운영효과**: 현행 중앙 혼잡도 규칙(0.10) 대비 터미널 총비용과 평균 A→O를 공동 판정한다.
5. **갱신 방식**: 견적기 통과 후 EVENT_ONLY 대비 EVENT+TIMER의 추가 이득과 계산량을
   단일축으로 검정한다.

P95(트럭 100대 중 오래 걸린 5대의 시간)는 보고용 진단이며 채택 veto로 쓰지 않는다.

## Evidence

등록 시점은 설계 계약만 동결했다. 구현·실험 결과는 착수 후 별도 evidence로 갱신한다.
