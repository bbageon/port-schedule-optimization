# YR-066 — 차분 신호 개량 — on-policy counterfactual base·창 확대·amortized-JR 회귀

- **Epic**: RL / **Priority**: 🟠 / **상태**: **backlog**
- **생성**: 2026-08-07 중간점검 — board row 는 있었으나 spec 파일이 없어 감사 하네스가
  검사조차 못 하던 항목이다(AGENTS.md: 모든 row 는 spec 필수). 아래 내용은 **board row 의
  Note 를 그대로 옮긴 것**이며 새로 지어낸 서술이 아니다.

## 현재 등록 내용 (board row Note 원문)

**우선순위 상향 (2026-07-19, YR-068 기각 근거)**: 본 시나리오에서 D_mean 급감(−5~−7→−1.6~−2.5) — 결정 밀도 상승이 창 내 반사실 차이를 희석. **규모에서 살아남는 차분 신호 설계가 협조 트랙의 선결 병목**. rollout 비용 주의 · **개정 전략 재정의 (2026-07-19)**: OLD 비용 기준 차분 개량은 철회 — 대체 경로 = YR-073(JR_NEW 교사·NEW 목적), 반사실 기법은 YR-074 미세조정 표적으로 흡수

## 착수 전 확인

- 이 spec 은 row Note 를 옮겼을 뿐 **연구 설계가 정제된 상태가 아니다.** 실제 착수 시
  질문·판정계약·되돌릴 조건을 먼저 채우고, 그때 이 문서를 갱신한다.
- 3대 게이트 보정 대상 축을 착수 시 명시하고 YR-153 `authorize-next` 를 통과한다.

## 참조

- board: [backlog](../../Dashboard/backlog.md)
