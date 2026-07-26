# YR-081 — 가변 크레인 수·다중 블록 확장 게이트

- **Epic**: RL / **Priority**: ⚪ / **등록일**: 2026-07-20
- **사용자 범위 정정**: 2026-07-26
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
TOS 배정 후, 재배정 가능한 반입 작업을 블록 Q가 판매/수용 한계비용으로 평가하고 중앙
`TransferResolver`가 `KEEP` 또는 실제 `A→B`를 원자적으로 확정하는 기능이다.

## 목표 구조

```text
외부 TOS: 최초 작업·담당 블록 배정
       ├─ block A → 공유 BlockPolicy 인스턴스 A
       ├─ block B → 공유 BlockPolicy 인스턴스 B
       └─ block C → 공유 BlockPolicy 인스턴스 C
                         ↓
             독립 블록 실행이 기본 기준선

재배정 가능한 GATE_IN만:
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

### C. 배정 후 반입 재배정

- source Q: 해당 반입 작업을 뺄 때 감소하는 비용 `OutRelief`.
- receiver Q: 해당 작업을 받을 때 증가하는 비용 `InBurden`.
- 중앙 resolver: `KEEP/A→B/A→C`의 terminal 순이득과 제약을 비교.
- 성공 시에만 source→receiver owner/queue/route를 한 transaction으로 변경.
- 실패·수신자 없음·순이득≤0이면 source가 계속 처리.
- resolver가 균등화하는 "부하"는 작업 수가 아니라 **한계비용**이다. 한계비용을 맞추는 것이 곧 총비용 최소이고, 본선이 눌린(비용 높은) 블록이 자동으로 반입을 덜어낸다 — "본선 집중"이 아니라 **비용 기반 부하분포의 창발적 결과**.
- 본선 지연항은 **YR-100 계산식**으로, 블록 내 ExecutionQ와 이 transfer J가 공유한다. 블록 Q는 본선/트럭을 구분하지 않고 비용만 최소화한다(type-agnostic). 이 type-agnostic은 본선 긴급도가 학습이 아니라 계산으로 비용에 들어가 있기에 성립한다.
- 자세한 상태·event·commit·검증 계약은 YR-099가 원본.

### D. 터미널 구조군

- YR-082의 구조군을 별도 환경으로 유지하고 평균 하나로 합치지 않는다.
- DGT는 육측·해측 역할과 AGV/FMS, BNCT·BCT는 S/C 이동·인계,
  북항은 혼합장비·블록 간 이동 가능성부터 별도 모델링한다.
- 공개 Level 0~1 자료는 stress 조건일 뿐 실제 터미널 일반화 증거가 아니다.

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

## 범위 밖

- TOS 최초 배정 경매·TOS 알고리즘 변경
- 독자적인 터미널 통합계획기 구축
- 지정 반출 컨테이너 변경·본선 job 블록 이전
- 실제 지원 근거 없는 YC 블록 간 이동
- LLM의 실시간 배정·장비제어
- 단일 블록 결과를 다중 블록·실운영 성과로 확대 주장
