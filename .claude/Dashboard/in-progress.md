# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-099-b | RL | **창중 재배정 브리지 계약 보정 — 이벤트·공용시계·전역 A→O 장부·owner/version·2단계 commit·용량검사** | 🟠 | 2026-07-27 | **사용자 순서 결정 (2026-07-27)**: YR-105 임계 탐색보다 **브리지 계약 보정이 먼저** — 기반이 틀리면 임계를 잘 찾아도 실제 중앙 resolver 에서 재현되지 않는다. 범위: ①`GATE_IN_EVENT` 정확 결정시점(현 lockstep 근사 제거) ②공용 시계 ③전역 `A→O` 장부(터미널 단일 원장) ④`owner/version` 필드 ⑤prepare→validate→commit/rollback 2단계 ⑥**수신 블록 용량 검사**(현재 없음 — 무한수용 가정). MVP(yr099_midrun_review, 기각)는 관측·발화 자산으로만 재사용. [spec](../docs/dashboard-task-specs/YR-099-post-tos-inbound-transfer-resolver.md) |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
