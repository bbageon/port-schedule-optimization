# Skill: Dashboard 운영 절차

> 대상: `.claude/Dashboard/` board. 규약 원본: [dashboard-board.md](../../dashboard-board.md).
> 트리거: 세션 시작 / 작업 착수·종료(commit) / 후속 작업 발견 / 작업 폐기 시.

## 세션 시작

1. `Dashboard/README.md` 현황 overview → `in-progress.md` → `ready.md` 순으로 읽는다.
2. in-progress 에 row 가 있으면 그 작업을 잇는다. `⏸ 외부 대기` 표기 row 는 WIP 로 세지 않는다.

## 작업 착수 (pull)

1. `ready.md` 의 row 를 `in-progress.md` 로 **이동**한다 (잘라내기 — 복사 금지) + 착수일 기록.
2. WIP 는 한 번에 1개 권장. 예외는 `⏸ 외부 대기` row 뿐 — 그 외 in-progress 2개면 하나는 거짓이다.
3. Agent 작업이 끝나고 사용자/외부 응답만 남으면 Note 에 `⏸ 외부 대기` 를 명시한다.

## 작업 종료 (commit + push)

1. 관련 테스트·lint·문서 검증을 실행하고 결과를 확인한다.
2. 현재 작업에 속한 파일만 stage 하여 범위가 명확한 commit 을 만든다. 다른 작업자의 미커밋 변경을 포함하지 않는다.
3. `in-progress.md` → `done.md` 로 row 를 이동하고 Evidence 열에 해당하는 것 전부 박제한다: `<commit 해시/링크> · [report](경로) · [snapshot](경로) · [일지](경로)`. Evidence 갱신이 commit 뒤에 이뤄지면 별도 bookkeeping commit 으로 남긴다.
4. upstream 이 없으면 `git push -u origin HEAD`, 있으면 `git push` 를 실행하고 원격 반영을 확인한다. 원격은 `https://github.com/bbageon/port-schedule-optimization.git` 이다.
5. push 실패 시 작업을 완료로 보고하지 않는다. 인증·권한·non-fast-forward 등 원인을 기록하고 사용자에게 필요한 조치를 알린다. 사용자가 명시적으로 로컬 전용 또는 push 금지를 요청한 경우만 예외다.
6. Title 에 결론을 굵게 1구절 포함 (예: "**X 가설 지지 (p<1e-4)**") — done 목록 스캔만으로 확정사항이 보이게.
7. 파생 후속 작업은 **같은 turn 에** `backlog.md` 에 등록한다 — row 가 아니면 증발한다.

## 다음 순서 제시 (후보 안내)

- 후보는 board row 기준으로만 제시하되, **ID·제목 나열로 끝내지 않는다**. 후보마다 쉬운 말로 두 가지를 붙인다:
  - **무엇**: 구체적으로 뭘 하는 작업인지 — 전문용어는 풀어서 1줄 (예: "w_tail grid" ✗ → "보상에서 '최악 대기 벌점'의 세기를 4단계로 바꿔가며 재학습" ✓)
  - **왜**: 왜 지금 하는지 — 어떤 관찰·결과에서 파생됐고, 끝나면 무엇을 알게/결정하게 되는지 1줄
- 우선순위의 전제가 바뀐 경우 (예: 새 실험 결과로 시급해짐) 그 변화도 한 줄 언급한다.

## 폐기

- `cancelled.md` 로 이동 + 폐기 사유 박제 (negative result 도 자산). 닫힌 ID 는 영구 폐번.

## ID·Priority 체계

- ID: `YR-NNN` (yard_rl 기반), 하위 분할은 `YR-NNN-a/b/c`. 재사용 금지.
- Priority: 🔴 Urgent / 🟠 High / 🟡 Medium / ⚪ Low. 실험 결과가 기존 row 의 전제를 뒤집으면 그 자리에서 재정렬/폐기.

## 유지보수

- done 20+ 누적 시 최근 ~20개만 본표에 남기고 같은 파일 하단 `## Archive` 절로 이동.
- README 현황 overview(3~5줄)는 핵심 결론이 바뀔 때마다 갱신 — 세션 시작 재설명 비용 제거용.
- row 명세가 길어지면 `.claude/docs/dashboard-task-specs/<ID>-<slug>.md` 로 분리하고 row 에는 링크만.
- 사용자 승인이 필요한 결정은 board 에 "승인 대기"만 표기하고 단독 진행하지 않는다.
