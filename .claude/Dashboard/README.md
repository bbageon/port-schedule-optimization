# `.claude/Dashboard/` — 작업 board (index)

> Jira/Plane 스타일 작업 board. **상태별로 파일 분리** (각 state = 한 파일). 본 README 가 **index**
> (운영 규약 + state 링크 + epic + 현황 overview). 상세 evidence 는 설계문서/산출물/git commit 이
> single source — board 는 그 위의 index 일 뿐 중복 서술하지 않는다.
> board 규약 원본: [dashboard-board.md](../../dashboard-board.md)

## 상태별 파일

| State | 파일 | 현황 |
|---|---|---|
| 🗂️ Backlog | [backlog.md](backlog.md) | 미착수·미래 |
| 📋 Ready | [ready.md](ready.md) | 착수 준비 |
| 🟢 In Progress | [in-progress.md](in-progress.md) | 진행 중 (한 번에 1개) |
| ✅ Done | [done.md](done.md) | 완료 (evidence 박제) |
| 🚫 Cancelled | [cancelled.md](cancelled.md) | 폐기 (사유 박제) |

흐름: `Backlog → Ready → In Progress → Done / Cancelled`.

## 사용 규약 (요지)

- **Issue ID**: `YR-NNN` (yard_rl 패키지명 기반). 닫힌 ID 재사용 금지. **Priority**: 🔴/🟠/🟡/⚪. **Epic**: 아래 표.
- 상태 이동(pull/done/폐기)·evidence 박제·grooming **절차 상세**: [dashboard-ops skill](../skills/dashboard-ops.md) (index: [skills.md](../skills.md)).
- **순서 제시 원칙·1줄 index 원칙** 등 전역 규칙: [AGENTS.md](../../AGENTS.md). row 상세 명세는 [task-specs](../docs/dashboard-task-specs/) 의 `<ID>-<slug>.md` (spec 이 원본 문서 § 를 링크).

## 🧭 Epics

| Epic | 의미 | 상태 |
|---|---|---|
| Infra | 하네스·환경·도구·프로젝트 스캐폴드 | 상시 |
| Data | 자료 확보·TerminalProfile·도메인 모델·전처리 (Phase 0~1) | active |
| Sim | 이벤트 시뮬레이터·Hard Constraint·실측 validation (Phase 2) | 예정 |
| Baseline | Baseline 정책·KPI·paired runner (Phase 3) | 예정 |
| RL | Tabular Q-learning → Masked DQN/PPO (Phase 4·7) | 예정 |
| Exp | Exp-1~4 요인실험·최종평가·ablation (Phase 5·8·9) | 예정 |
| UI | 검증·시연용 읽기 전용 replay UI (Phase 6) | 예정 |

## 📌 현재 상태 overview (한눈에)

- **설계 단계 완료, 코드 미착수**: [실험설계안](../../부산항_야드크레인_강화학습_실험설계안_업데이트.md)(연구 설계) + [구현계획서](../../부산항_야드크레인_강화학습_구현계획서.md)(index) → `docs/구현계획/01~05` 분할 5문서. 요약은 [../docs/](../docs/), row 명세는 [task-specs](../docs/dashboard-task-specs/).
- **핵심 가설(H1~H5)**: "차량 정보를 더 이른 시점에 확보할수록 YC 작업순서 최적화 효과가 커지는가" — 사전 결론이 아닌 검증가설.
- **2026-07-12 구현계획 재편**: Phase 0~9 (검증 UI Phase 6 신설 → YR-015, 실측 validation 은 Phase 2 게이트로 이동).
- **병목**: 대상 터미널·운영자료 미확보([ready](ready.md) YR-002, 사용자/운영사 결정 필요). 단 스캐폴드·도메인 모델·시뮬레이터는 가정 프로파일(`assumed: true`)로 선착수 가능.
- **Git 원격 연결 완료** — 현재 저장소의 `origin` 은 [bbageon/port-schedule-optimization](https://github.com/bbageon/port-schedule-optimization) 이다. 완료 작업은 검증·commit·Dashboard evidence 갱신 후 push 성공까지 확인한다.
