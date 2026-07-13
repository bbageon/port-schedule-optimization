# 🗂️ Backlog (미래)

> 미착수·미래 작업. 방향이 자주 바뀌면 재정렬. [index](README.md) · 다음 상태: [ready](ready.md).
> Phase 는 [구현계획서](../../부산항_야드크레인_강화학습_구현계획서.md) 분할문서 [05 §4](../../docs/구현계획/05_테스트_로드맵_산출물.md) 기준 (0~9).

| ID | Epic | Title | Priority | Note |
|---|---|---|---|---|
| YR-005 | Data | Phase 1 후반: 원천자료 loader·익명화·품질 플래그·날짜 split | 🟡 | [spec](../docs/dashboard-task-specs/YR-005-data-pipeline.md) · 실자료는 YR-002 후 |
| YR-009 | Sim | Phase 2 게이트: 시뮬레이터 실측 validation | 🟡 | [spec](../docs/dashboard-task-specs/YR-009-simulator-validation.md) · 실자료 의존, 미충족 시 RL 평가 금지 |
| YR-018 | RL | reward weight 민감도 — {0,.1,.3,1} grid, P95(tail) 보호 중심 | 🟠 | Exp-1 에서 평균↓·P95↑ trade-off 관찰 → w_tail 탐색 필요 (03 §1.2 weight 원칙) |
| YR-019 | Exp | ETA 품질 시나리오 매트릭스 (PERFECT/BIASED/NO_SHOW/STALE) | 🟡 | §18.2 — 현재는 EMPIRICAL(±300s)만 구현. Exp-3 결과의 강건성 확인용 |
| YR-020 | RL | Exp-2/3 열세 원인 분석 — 상태공간 희석 vs 정보 무익 판별 | 🟠 | 학습예산·상태 축소 실험. 함수근사(YR-012) 전환조건 §16.3 판단 재료 |
| YR-021 | Exp | 부하조건별 정보효과 — peak·고장치율·고재조작 시나리오 재실험 | 🟡 | 정보 선행 편익은 혼잡 조건 의존 가설 — §18.3 운영부하 축 |
| YR-026 | RL | 비용계수 민감도 + tail 60분 임계 KPI 확장 (안전운임 제도 정합) | 🟡 | YR-025 후속: 계수 4/5가 assumed·tail 은 30분 proxy — 본선·tail 계수 grid 로 negative 결과의 강건성 판별. YR-018 과 통합 검토 |
| YR-024 | Sim | 취급시간 확률화 — DGT 육측 원격 인계 분산(PEMA) 반영, 결정적 모델 확장 | 🟡 | YR-023 발견: 공개정보만으론 HJNC·DGT 프로파일이 수치 동일 — 케이스 차별화의 유일한 문헌 경로 |
| YR-015-b | UI | Phase 6 후반: UI-3 정책설명 패널 고도화·UI-4 동기비교·검증 테스트 확충 | 🟡 | [spec](../docs/dashboard-task-specs/YR-015-verification-ui.md) · UI-1/2 는 YR-015-a 로 분할 착수 (2026-07-13) |
| YR-012 | RL | Phase 7: Masked DQN/PPO 함수근사 | ⚪ | [spec](../docs/dashboard-task-specs/YR-012-dqn-ppo.md) · YR-010 에서 전환조건 확인 시만 |
| YR-013 | Exp | Phase 8: Exp-4 다중 YC 협조 | ⚪ | [spec](../docs/dashboard-task-specs/YR-013-exp4-multi-yc.md) |
| YR-014 | Exp | Phase 9: 최종평가·ablation·탄소 사후평가 | ⚪ | [spec](../docs/dashboard-task-specs/YR-014-final-evaluation.md) |

---

운영: 우선순위 오르면 [ready.md](ready.md) 로 승격. 폐기 시 [cancelled.md](cancelled.md).
