# 🗂️ Backlog (미래)

> 미착수·미래 작업. 방향이 자주 바뀌면 재정렬. [index](README.md) · 다음 상태: [ready](ready.md).
> Phase 는 [구현계획서](../../부산항_야드크레인_강화학습_구현계획서.md) 분할문서 [05 §4](../../docs/구현계획/05_테스트_로드맵_산출물.md) 기준 (0~9).

| ID | Epic | Title | Priority | Note |
|---|---|---|---|---|
| YR-005 | Data | Phase 1 후반: 원천자료 loader·익명화·품질 플래그·날짜 split | 🟡 | [spec](../docs/dashboard-task-specs/YR-005-data-pipeline.md) · 실자료는 YR-002 후 |
| YR-006 | Sim | Phase 2: 단일 YC 이벤트 시뮬레이터 | 🟠 | [spec](../docs/dashboard-task-specs/YR-006-single-yc-simulator.md) · YR-004 후속 |
| YR-007 | Sim | Phase 2: SafetyConstraintEngine + invariant 테스트 | 🟠 | [spec](../docs/dashboard-task-specs/YR-007-constraint-engine.md) · YR-006 과 동반 |
| YR-009 | Sim | Phase 2 게이트: 시뮬레이터 실측 validation | 🟡 | [spec](../docs/dashboard-task-specs/YR-009-simulator-validation.md) · 실자료 의존, 미충족 시 RL 평가 금지 |
| YR-008 | Baseline | Phase 3: Baseline 정책 + KPI·paired runner | 🟠 | [spec](../docs/dashboard-task-specs/YR-008-baseline-policies.md) |
| YR-010 | RL | Phase 4: Tabular Q-learning PoC | 🟡 | [spec](../docs/dashboard-task-specs/YR-010-tabular-q-poc.md) |
| YR-011-a | Exp | Phase 5: Exp-1 (블록 도착 이후 정보, sequence_only) | 🟡 | [spec](../docs/dashboard-task-specs/YR-011-exp1-3-experiments.md) · 예비 PoC 는 합성 데이터로 |
| YR-011-b | Exp | Phase 5: Exp-2 (게이트 진입 이후 정보) | 🟡 | [spec](../docs/dashboard-task-specs/YR-011-exp1-3-experiments.md) · YR-011-a 후 |
| YR-011-c | Exp | Phase 5: Exp-3 (사전 반출입정보+제공 ETA, 3A/B/C) | 🟡 | [spec](../docs/dashboard-task-specs/YR-011-exp1-3-experiments.md) · YR-011-b 후 |
| YR-017 | Data | 합성 시나리오 생성기 (가정 프로파일 기반, 실자료 확보 전 PoC 구동용) | 🟠 | 실자료 대체 아님 — YR-005·009 로 대체·검증 예정. 도착분포·초기장치·재조작위험 파라미터화 |
| YR-015 | UI | Phase 6: 검증 UI MVP (recorder·replay·정책설명·동기비교) | 🟡 | [spec](../docs/dashboard-task-specs/YR-015-verification-ui.md) · 신규 범위 (04 문서) |
| YR-012 | RL | Phase 7: Masked DQN/PPO 함수근사 | ⚪ | [spec](../docs/dashboard-task-specs/YR-012-dqn-ppo.md) · YR-010 에서 전환조건 확인 시만 |
| YR-013 | Exp | Phase 8: Exp-4 다중 YC 협조 | ⚪ | [spec](../docs/dashboard-task-specs/YR-013-exp4-multi-yc.md) |
| YR-014 | Exp | Phase 9: 최종평가·ablation·탄소 사후평가 | ⚪ | [spec](../docs/dashboard-task-specs/YR-014-final-evaluation.md) |

---

운영: 우선순위 오르면 [ready.md](ready.md) 로 승격. 폐기 시 [cancelled.md](cancelled.md).
