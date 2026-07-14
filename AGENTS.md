여기는 claude.md 의 진입점이다.

claude.md 에 대해 수정하지 않는다.

## 전역 규칙

- 모든 md 파일은 **200줄을 넘지 않는다**. 초과가 예상되면 분할한다 (사용자 소유 설계문서 — 루트 2건과 `docs/구현계획/` — 는 예외).
- 예정사항·다음 순서·우선순위는 `.claude/Dashboard/` 에 등록된 row 기준으로만 제시하고, 없는 작업은 backlog 에 먼저 등록한다. 후보 제시 시 각 후보에 "무엇을 하는 작업인지 / 왜 지금 하는지"를 쉬운 말로 1–2줄 붙인다 (절차: dashboard-ops).
- Dashboard row 는 1줄 index 로 유지한다 — 상세 명세는 `.claude/docs/dashboard-task-specs/<ID>-*.md` 로 분리.
- 작업 완료 시 관련 검증과 Dashboard evidence 갱신 후 범위가 명확한 commit 을 만들고 `origin` 으로 push 한다. upstream 이 없으면 `git push -u origin HEAD`, 있으면 `git push` 를 사용한다. push 실패 시 완료로 보고하지 말고 원인을 명시한다.
- 원격 저장소 single source 는 `origin = https://github.com/bbageon/port-schedule-optimization.git` 이다. 사용자가 명시적으로 로컬 전용 또는 push 금지를 요청한 작업만 예외로 한다.
- 본 문서(AGENTS.md)에는 전역 규칙과 1~2줄 명세만 둔다. 구체적 기술·방법론·절차는 `.claude/skills/` 에 두고, `.claude/skills.md` 가 그 index 다.
- 참조 요약은 `.claude/docs/`, 작업 상태는 `.claude/Dashboard/` (index: README.md). board 규약 원본은 루트 `dashboard-board.md`.

## 원본 문서 (single source)

- `부산항_야드크레인_강화학습_실험설계안_업데이트.md` — 연구 설계 (무엇을 왜 검증, 가설 H1~H5). 요약: `.claude/docs/실험설계안-요약.md`
- `부산항_야드크레인_강화학습_구현계획서.md` — 기술 명세 상위 index (상세: `docs/구현계획/01~05`, Phase 0~9). 요약: `.claude/docs/구현계획서-요약.md`
