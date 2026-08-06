# YR-081 — 가변 크레인 수·다중 블록 확장 게이트

- **Epic**: RL / **Priority**: ⚪ / **등록일**: 2026-07-20
- **사용자 범위 정정**: 2026-07-26
- **2026-08-06 순서 정정**: **H-21 동질 공유형**의 N블록 공용시계·원장·route·terminal-wide
  입력은 YR-150으로, **V-21 육·해측 역할분리형**의 mask·LSTP/WSTP·AGV 흐름은 YR-083으로
  앞당긴다. 가변 크레인 수·블록별 이질성·두 구조 밖 실제 터미널 일반화만 본 row에 남긴다.
- **결정 원본**: [TOS 배정 후 반입 재배정·중앙 resolver](../strategy-history/2026-07-26-YR-081-배정후-반입재배정-중앙-resolver-사용자확정.md)
- **세부 구현**: [YR-099](YR-099-post-tos-inbound-transfer-resolver.md)

## 배경

현재 블록 정책은 단일 블록·2크레인 입력과 자료구조를 전제로 한다. 크레인 1대·3대 이상이나
여러 블록은 단순한 성능 회귀가 아니라 실행 구조 변화다. 부산항 자료도 RMG/ATC-YT,
수직 ARMG/AGV, 수직 T/C/S/C, 북항 혼합형의 물리가 달라 하나의 환경으로 묶을 수 없음을
보였다.

초기 명세의 “상위 야드 관제기가 작업·차량·이송장비를 배분한다”는 범위는 과했다.
**TOS가 최초 작업과 담당 블록을 배정하며, 연구는 TOS를 수정·대체하지 않는다.**
다중 블록의 기본형은 같은 블록 정책을 여러 블록에 독립 배포하는 구조다.

블록 간 추가 결정권은 별도 상금이 확인되는 경우에만 연다. 현재 사용자 확정 대상은
TOS 배정 후, 아직 적재되지 않은 **재배정 가능한 반입·본선 양하(STORE)**를 블록 Q가
판매/수용 한계비용으로 평가하고 중앙 `TransferResolver`가 `KEEP` 또는 실제 `A→B`를
원자적으로 확정하는 기능이다.

## 목표 구조

```text
외부 TOS: 최초 작업·담당 블록 배정
       ├─ block A → 공유 BlockPolicy 인스턴스 A
       ├─ block B → 공유 BlockPolicy 인스턴스 B
       └─ block C → 공유 BlockPolicy 인스턴스 C
                         ↓
             독립 블록 실행이 기본 기준선

재배정 가능한 GATE_IN·DISCHARGE STORE만:
BlockQ OutRelief/InBurden → deterministic TransferResolver
→ KEEP 또는 concrete TRANSFER(A→B) → 최종 블록 ExecutionQ
```

블록끼리 자유형 신경망 메시지를 주고받지 않는다. 구조화된 quote와 versioned job 상태만
중앙 resolver에 전달한다. 최초 구조에는 QMIX·PPO·LLM 관제를 넣지 않는다.

## 확장 축

### A. 블록 내부 구조

- 크레인 수 `{1, 2, 3+}`와 이기종 속도·고정 작업구역.
- 고정형 ARMG의 블록 간 이동·상호통과는 기본 금지.
- 공동조합이 작으면 중앙 직접평가, 커지면 set/attention/GNN 정책을 비교.
- 안전·비통과·슬롯·레인·마감은 기존 mask와 resolver가 담당.

### B. 다중 블록 기본형

- 호환 블록은 하나의 정책 파라미터를 공유하되 상태·queue·stack은 독립.
- 구조가 다른 블록은 `block_profile` 조건 입력 또는 구조군별 정책으로 분리.
- TOS가 각 블록에 배정한 작업을 그대로 처리하는 독립 블록 정책을 첫 기준선으로 둠.
- TOS 최초 배정·선석/QC/YT 전역 계획을 연구 레이어가 재구축하지 않음.

### C. 배정 후 반입·양하 STORE 재배정

- source Q: 해당 STORE 작업을 뺄 때 감소하는 비용 `OutRelief`.
- receiver Q: 해당 작업을 받을 때 증가하는 비용 `InBurden`.
- 중앙 resolver: `KEEP/A→B/A→C`의 terminal 순이득과 제약을 비교.
- 성공 시에만 source→receiver owner/queue/route를 한 transaction으로 변경.
- 실패·수신자 없음·순이득≤0이면 source가 계속 처리.
- resolver가 균등화하는 "부하"는 작업 수가 아니라 **한계비용**이다. 한계비용을 맞추는 것이 곧 총비용 최소이고, 본선이 눌린(비용 높은) 블록이 자동으로 반입을 덜어낸다 — "본선 집중"이 아니라 **비용 기반 부하분포의 창발적 결과**.
- 본선 지연항은 **YR-100 계산식**으로, 블록 내 ExecutionQ와 이 transfer J가 공유한다. 물리·자격·공식 계산은 작업 흐름을 구분하고, 그 결과를 받은 **잔여 Q망만** 작업 라벨에 의존하지 않고 비용을 최소화한다.
- 자세한 상태·event·commit·검증 계약은 YR-099가 원본.

### D. 터미널 구조군

- YR-082의 구조군을 별도 환경으로 유지하고 평균 하나로 합치지 않는다.
- 연구 본환경은 먼저 H-21과 V-21을 같은 21블록·동일 master stream으로 통제한다. 구조별
  별도 정책이 자기 구조의 규칙을 이긴 뒤에만 공유 가중치 일반화를 별도 축으로 연다.
- DGT는 육측·해측 역할과 AGV/FMS, BNCT·BCT는 S/C 이동·인계,
  북항은 혼합장비·블록 간 이동 가능성부터 별도 모델링한다.
- 공개 Level 0~1 자료는 stress 조건일 뿐 실제 터미널 일반화 증거가 아니다.

## 재배정 taxonomy (재배정 불가 + 3유형)

작업을 "재배정 가능 여부"와 "매칭 방향"으로 나눈다. 재배정 불가 작업은 원래 블록이 실행하고, 아래 세 재배정 유형은 **블록 한계비용 입찰 + 결정론적 resolver**를 방향만 달리 쓴다.

| 유형 | 예 | 방향 | 메커니즘 |
|---|---|---|---|
| 재배정 불가 | 지정 반출·본선 적하(LOAD) | — | 대상 컨테이너·소유블록은 고정하되 블록 내 실행순서·rehandle은 최적화 |
| ① 장치 위치 | 반입·본선 양하(DISCHARGE) | 컨테이너→블록 | resolver가 블록 선택(InBurden bid); 블록 내 bay/row/tier는 블록 Q(find_slot) |
| ② 후보 선택 | 미지정 공컨·환적 그룹오더 | 주문→컨테이너 | resolver가 주문 등록, 조건 만족 후보 가진 블록이 구매(fulfillment 입찰); 차량 도착 시 조건 내 최저 rehandle 컨테이너 선택 |
| ③ 물리 재배치 | 이미 장치된 컨테이너 | 블록이 offload | source 블록이 판매, target이 InBurden 입찰; 실물 이동 고비용→높은 최소 순이득 gate |

- **①의 2계층**: 블록 선택=resolver(inter-block), 블록 내 슬롯=선택 블록의 allocator(현재 `find_slot`, 향후 슬롯정책). resolver가 bay/row/tier까지 정하지 않는다.
- `yard_handover_cap=None`에서 양하 배치는 해당 양하 선박의 완료를 직접 당기지 않는다. 다만 다른 블록의 YC 시간을 비워 LOAD를 앞당기는 간접효과는 YR-100의 `KEEP vs TRANSFER` 반사실이 입증할 때만 센다.
- **본선 긴급도(YR-100)** 는 현재 반입·양하 STORE가 만드는 LOAD 처리여유에 들어간다. 미래 ② 환적 후보를 열 때 선적순서 효과를 별도 재검토한다.
- 현재 반입·양하 PoC는 공간·높이·규격·pile 일관성 등 최소 물리만 사용한다. 중량·냉동·위험물·선사·항차 그룹 규칙은 메커니즘 검증을 막지 않고 YR-095 최종 현실화 단계에서 추가한다.

**착수 순서** (payoff·복잡도 순, 각 단계 상금 확인 후): ① 반입+양하([YR-099](YR-099-post-tos-inbound-transfer-resolver.md), 현재 단순화 PoC) → ② 후보 선택(자격 로직 신설) → ③ 물리 재배치(최고비용) → 실제 자료 기반 YR-095 현실 적재규칙(최종 실증). ②③은 현 YR-099 범위 밖이다.

### 요청 리스트 (order book) — resolver 의 핵심 자료구조 (2026-07-27 명시)

중앙 resolver 는 두 종류의 **요청 리스트**를 유지한다. 블록 간 상태를 아는 것은 이
리스트를 든 resolver 하나뿐 — 블록 Q 는 견적 질문에 답만 하며 다른 블록·주문을 모른다
(분업 불변식: 블록=계산, resolver=매칭·안전). 현실 대응물 = TOS 작업 대기열의 수신함.

| 리스트 | 항목 | 미정인 것 | 블록의 입찰 |
|---|---|---|---|
| **배치 요청** (①, 정방향) | 반입 트럭·양하 STORE | 블록 | 수용비용(InBurden) |
| **공급 주문** (②, 역방향) | 미지정 공컨 반출·환적 그룹오더 | 컨테이너 | 공급/불출 비용 |

- **항목 공통 필드**: 공개정보 스냅샷(예약·예상도착·규격·허용블록 또는 주문조건) ·
  **마감시각**(배치=transfer_lock/블록도착 전, 주문=차량 도착 전) · 상태(대기→견적→
  확정/KEEP) · `version`(낡은 견적 거부).
- **생명주기**: TOS 수신 → 등록 → review 에 견적 요청 → 낙찰 or KEEP → 원자 commit →
  제거. **마감 경과 = fail-closed 자동 KEEP/기본배정** (미확정 방치 금지).
- **구현 거리(정직)**: ①은 YR-099 데이터 계약(`TOS_ASSIGNMENT_RECEIVED`·`TRANSFER_REVIEW`·
  `reassignable_until`·`assignment_version`)이 이미 정의 — 현 MVP 는 t=0 일괄 review 라
  리스트가 자명하고, **실체화 시점 = 창중(run 중) review**(엔진 브리지 잔여작업). ②의
  주문 등록·자격 매칭은 후속 신설.

## 판정 원칙

1. 단일 블록 정책의 현재 계약과 성능을 먼저 동결한다.
2. 독립 다중 블록 기준선이 정상 작동하는지 검증한다.
3. 공개정보 paired rollout으로 배정 후 반입 재배정의 terminal 상금을 측정한다.
4. 상금이 없으면 TransferQuoteQ·QMIX·학습 통신을 만들지 않는다.
5. 상금이 있을 때만 rollout quote를 공유 TransferQuoteQ로 근사한다.
6. 공유 YT·도로의 비가산 효과로 scalar quote가 실패할 때만 중앙 joint scorer를 검토한다.

QMIX는 central resolver가 아니다. 여러 Q의 학습 교차항 보정이 필요하다는 별도 증거가
있을 때만 비교하며, 사용하더라도 소유권·예약·commit·rollback은 결정론적 resolver가 맡는다.

## 산출물

- 가변 agent·block profile 계약과 독립 다중 블록 simulator
- 구조군별 공유/분리 정책 판정
- YR-099의 block ownership·quote·TransferResolver
- 독립 블록 대 재배정 resolver headroom/근사/조건 일반화 보고서

## 의존

- YR-014 단일 블록 최종 판정
- YR-082 터미널 구조군·프로파일 자격
- YR-083 Level 2 런타임 계약
- YR-042 다른 단일 블록 구조 일반화 분류
- YR-089 시간장부, YR-093 공개정보 rollout 안전
- YR-086 크레인 수 선택 컴포넌트는 선행 발판으로 재사용
- YR-100 본선 비용 계산식 — ExecutionQ·TransferResolver 공유 원료(블록 Q type-agnostic 전제)
- YR-103 게이트→블록 3~7분·동적 블록혼잡 관측 — 재배정 quote의 공개정보 원료
- YR-095 현실 적재규칙은 반입·양하 PoC 선결이 아니라 최종 실증 게이트

## 범위 밖

- TOS 최초 배정 경매·TOS 알고리즘 변경
- 독자적인 터미널 통합계획기 구축
- 지정 반출 컨테이너 변경·본선 적하(LOAD) 블록 이전·이미 적재된 양하 컨테이너 물리이동
- 실제 지원 근거 없는 YC 블록 간 이동
- LLM의 실시간 배정·장비제어
- 단일 블록 결과를 다중 블록·실운영 성과로 확대 주장
