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
| Data | TOS·VBS·ETA·본선·장비·레인 통합 schema와 실자료 매핑 | active |
| Sim | 통합 이벤트 시뮬레이터·Hard Constraint·실측 validation | active |
| Baseline | 강한 동정보 휴리스틱·중앙 matching·paired runner | active |
| RL | 동적 후보 Double DQN·Total Cost·QMIX 협조학습 | active |
| Exp | 동일 통합정책의 locked 평가·ablation·민감도 | active |
| UI | 정책 replay·설명·운영자 승인/반려 피드백 | active |

## 📌 현재 상태 overview (한눈에)

- **최종 목표 전환 (2026-07-15, 사용자 결정)**: 별도 Exp 정책이 아니라 차량·본선·이송장비·레인·다중 YC를 처음부터 같은 State·Action·Total Cost 계약으로 다루는 단일 통합정책. [최종전략](../docs/부산항_레인_다중야드크레인_협조최적화_강화학습_최종전략.md) · [결정 이력](../docs/strategy-history/2026-07-15-YR-034-final-integrated-strategy-pivot.md).
- **정책 구조**: 가변 후보 `Q_cost`를 Candidate Double DQN으로 평가하고 중앙 resolver가 공동제약을 보장한 뒤 QMIX 추가효과를 검증한다. 안전·물리·마감 위반은 보상이 아니라 mask다.
- **보존된 PoC 증거 (단일 야드 트랙)**: 격차를 +1.195→**+0.035분(YR-012-c)** 으로 축소, oracle 상금 하한 +0.182분 확인. YR-031-b가 이탈 AUC 0.852·집합구조 이득 0.000으로 **입력 요약 부족·후보별 골격 충분**을 판정 → YR-012-c 가 집합 8 feature 추가로 **greedy 통계적 동률(CI 0 포함) 첫 달성** (형식 승리는 미달 — 선택 winner's curse·검정력 잔존, YR-032). 이 트랙은 YR-034 통합전략과 별개로 계속.
- **현재 실행 순서**: YR-035 통합 MDP·데이터 계약 → YR-036 시뮬레이터 → YR-037 후보·공동제약 + YR-038 Total Cost → YR-039 Candidate DDQN → YR-013 공동배정·QMIX → YR-014 locked ablation.
- **주장 게이트**: 실측자료·CURRENT_RULE은 YR-002, 실측 validation은 YR-009가 담당한다. 두 게이트 전 모든 결과는 합성·가정 조건의 구현 증거이며 부산항 실운영 개선 주장이 아니다.
