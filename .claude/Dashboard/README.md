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

- **Phase 0~5 예비 PoC 완주 (2026-07-12, 합성 데이터)**: 시뮬레이터+제약(YR-006/007)·Baseline 4종(YR-008)·Tabular Q(YR-010)·Exp-1~3C matrix(YR-011-a/b/c) 구현·실행 완료. 테스트 46건. 리포트: `outputs/reports/exp_matrix/`.
- **핵심 결과(합성·가정 조건)**: QL_EXP1 이 FIFO 대비 **평균대기 -15.2% (12/12 seed 유의)**·본선지연 0 달성. 단 **정보 선행(Exp-2/3)이 오히려 열세** — 상태공간 희석 의심(YR-020), P95 는 전 QL 에서 악화 경향(YR-018 w_tail 탐색 필요). H1~H2 는 이 조건에서 **미지지** — 사전 결론 금지 원칙 그대로 유효.
- **67-agent 리뷰 워크플로우**로 확정 결함 9건 수정 완료 (`24b095a`) — KPI 적분창·본선 방치 무벌점·검열 편향 등.
- **프로파일 v2 (2026-07-13, YR-022/023)**: HJNC·DGT ARMG 초안 2벌로 Exp-1 재실행 — 방향 유지(평균대기 -10.4%)·개선폭 축소·P95 악화 재현. 공개정보만으론 두 케이스 수치 동일 수렴 (차별화는 🤝 협약 또는 YR-024 확률화).
- **비용 최소화 RL (2026-07-13, YR-025/027)**: Direct-Job Cost-Q는 shortest-service 대비 평균 +0.039분(95% CI +0.006~+0.072), fallback 55.0%로 primary 미통과. 반면 shortest-service는 FIFO 13.782→7.789분(-43.5%)이며 NEAREST는 11.285분 — 이동거리 단독이 아니라 이동·취급·재조작을 합친 총 cycle time 효과다.
- **병목 불변**: 실측자료·CURRENT_RULE 미확보(YR-002) — 모든 수치는 실운영 대비 아님. YR-009 validation 게이트 전까지 연구 주장 불가.
- **Git**: `origin` = [bbageon/port-schedule-optimization](https://github.com/bbageon/port-schedule-optimization). 완료 작업은 검증·commit·evidence 갱신 후 push 성공까지 확인.
