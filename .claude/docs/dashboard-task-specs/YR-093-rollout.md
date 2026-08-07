# YR-093 — 예측 rollout 정보안전 — 미통지 고장·계획변경 이벤트 마스킹

- **Epic**: RL / **Priority**: 🟠 / **상태**: **backlog**
- **생성**: 2026-08-07 중간점검 — board row 는 있었으나 spec 파일이 없어 감사 하네스가
  검사조차 못 하던 항목이다(AGENTS.md: 모든 row 는 spec 필수). 아래 내용은 **board row 의
  Note 를 그대로 옮긴 것**이며 새로 지어낸 서술이 아니다.

## 현재 등록 내용 (board row Note 원문)

**외부감사 결함4 (2026-07-26)**: 현실형 예측 rollout 이 deepcopy 로 시뮬 전체를 복사해 **미래의 미통지 고장(EQUIPMENT_DOWN)·계획변경(PLAN_CHANGE) 이벤트를 알고** rollout — 조건부 미래정보 누출. ETA 결측 트럭의 실제도착 대체 누출은 fail-closed 로 즉시 정정(`4b44737` 후속 커밋). 잔여 수정: scratch 에서 미공지 injected/PLAN_CHANGE 이벤트 제거 또는 공지분포 대체. **현 YR-087 결과는 돌발 없는·ETA 완비 실험이라 영향 없음(주장 한정 유지)** — 고장·계획변경·ETA 결측 있는 하이브리드 실증 전 필수. [predictive_rollout.py](../../src/yard_rl/integrated/predictive_rollout.py)

## 착수 전 확인

- 이 spec 은 row Note 를 옮겼을 뿐 **연구 설계가 정제된 상태가 아니다.** 실제 착수 시
  질문·판정계약·되돌릴 조건을 먼저 채우고, 그때 이 문서를 갱신한다.
- 3대 게이트 보정 대상 축을 착수 시 명시하고 YR-153 `authorize-next` 를 통과한다.

## 참조

- board: [backlog](../../Dashboard/backlog.md)
