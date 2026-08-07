# YR-069 — 차분 표적 QMIX 민감도 — λ_mix·창 그리드 (별도 사전등록)

- **Epic**: RL / **Priority**: ⚪ / **상태**: **backlog**
- **생성**: 2026-08-07 중간점검 — board row 는 있었으나 spec 파일이 없어 감사 하네스가
  검사조차 못 하던 항목이다(AGENTS.md: 모든 row 는 spec 필수). 아래 내용은 **board row 의
  Note 를 그대로 옮긴 것**이며 새로 지어낸 서술이 아니다.

## 현재 등록 내용 (board row Note 원문)

YR-013c prereg 가 금지한 knob 탐색의 정식 경로. **2026-07-26 조건 축소**: 다중 블록이라는 이유만으로 재개하지 않음. YR-099의 local 한계비용 합이 direct terminal 반사실의 부호·순위를 반복적으로 틀리고, 결정론적 공동평가도 규모상 불가능하다는 증거가 있을 때만 비교. QMIX를 써도 작업매칭·예약·commit/rollback은 TransferResolver가 담당

## 착수 전 확인

- 이 spec 은 row Note 를 옮겼을 뿐 **연구 설계가 정제된 상태가 아니다.** 실제 착수 시
  질문·판정계약·되돌릴 조건을 먼저 채우고, 그때 이 문서를 갱신한다.
- 3대 게이트 보정 대상 축을 착수 시 명시하고 YR-153 `authorize-next` 를 통과한다.

## 참조

- board: [backlog](../../Dashboard/backlog.md)
