# YR-094 — 엔진 견고성 소갭 묶음 — 감사 지적 부수결함

- **Epic**: Sim / **Priority**: 🟡 / **상태**: **backlog**
- **생성**: 2026-08-07 중간점검 — board row 는 있었으나 spec 파일이 없어 감사 하네스가
  검사조차 못 하던 항목이다(AGENTS.md: 모든 row 는 spec 필수). 아래 내용은 **board row 의
  Note 를 그대로 옮긴 것**이며 새로 지어낸 서술이 아니다.

## 현재 등록 내용 (board row Note 원문)

외부감사 (2026-07-26) 부수: ①assign() 외부 API 가 위조·낡은 후보 완전 재검증 안 함(정상 resolver 경로는 안전 — fail-closed 화) ②다중 배정 중 후속 실패 시 선배정 자동 rollback 없음 ③죽은 상태필드(last_move_dir·recent_empty_travel_s 미갱신, recent_throughput 은 누적값) — 제거 또는 실갱신 ④겹친 고장구간에서 첫 UP 이 조기 복구(boolean → 카운트). **⑤ ✅정정됨(감사 2차)**: unfinished_backlog 가 ASSIGNED·RUNNING 미집계(runner·direct_job_env 소비) — 양 엔진 DONE·CANCELLED 외 전부 집계로 수정+테스트 (AUDIT-0726 2차 보완 commit). 잔여 ①~④ 각각 소규모·행동 영향 낮음이나 계약 정직성 건

## 착수 전 확인

- 이 spec 은 row Note 를 옮겼을 뿐 **연구 설계가 정제된 상태가 아니다.** 실제 착수 시
  질문·판정계약·되돌릴 조건을 먼저 채우고, 그때 이 문서를 갱신한다.
- 3대 게이트 보정 대상 축을 착수 시 명시하고 YR-153 `authorize-next` 를 통과한다.

## 참조

- board: [backlog](../../Dashboard/backlog.md)
