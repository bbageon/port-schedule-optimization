# YR-060 — QMIX 타깃 PopArt 보완 (return 크기 비정상성)

- **Epic**: RL / **Priority**: 🟡 / **상태**: **backlog**
- **생성**: 2026-08-07 중간점검 — board row 는 있었으나 spec 파일이 없어 감사 하네스가
  검사조차 못 하던 항목이다(AGENTS.md: 모든 row 는 spec 필수). 아래 내용은 **board row 의
  Note 를 그대로 옮긴 것**이며 새로 지어낸 서술이 아니다.

## 현재 등록 내용 (board row Note 원문)

[적용전략](../docs/상태정규화-보상가중치-적용전략.md) §6-2 · **조건 발동 (2026-07-19, YR-059: 입력 정규화로 미해결)** — 단 신용 축(YR-061~065)이 1차 용의자로 승격돼 순서는 그 종합 뒤. state_norm 결합 필수

## 착수 전 확인

- 이 spec 은 row Note 를 옮겼을 뿐 **연구 설계가 정제된 상태가 아니다.** 실제 착수 시
  질문·판정계약·되돌릴 조건을 먼저 채우고, 그때 이 문서를 갱신한다.
- 3대 게이트 보정 대상 축을 착수 시 명시하고 YR-153 `authorize-next` 를 통과한다.

## 참조

- board: [backlog](../../Dashboard/backlog.md)
