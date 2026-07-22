# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-080 | RL | 목적함수 문헌정합 재설계와 채택 정책 재판정 — **1단계·2a 완료, 2b(학생층 증류 bake-off) 대기** | 🟠 | 2026-07-21 | [spec](../docs/dashboard-task-specs/YR-080-objective-contract-redesign.md) · **1단계 완료** (2026-07-22, `827bba4`~`9ed4923` — [전략 v5 §9](../docs/strategy-history/2026-07-21-본선처리-전략.md)): 양하 역전 수정·인과 연결·비용==KPI 등식·기준재 config·manifest 재동결·적재 seam. · **2a 완료** (2026-07-22, [보고서](../../outputs/reports/yr080_readjudicate/yr080_2a_report.md)): 교사층 목적 재판정. **교락 해소(15-seed 같은-창 짝지음): LEX는 창 늘려도 배 못 지킴(무의), 같은 창서 기준재 목적이 배 유의 보호(@5400 −20.8★/−52.0★) → 사전식 회귀 확정 기각, 목적 재설계는 실제 효과.** 정직한 정정: 초기 CI 버그(pstdev+z)·P95 누락 → 표본sd+t·P95검정 교정(적대 3관점 검증). 배↔트럭 P95 상충 → **P95는 채택 게이트로(목적 밖, 사용자 결정)**. · **2b 완료 (2026-07-22, negative — [2b 보고서](../../outputs/reports/yr080_distill/yr080_2b_report.md))**: 학생 재현성 우선(사용자 결정)으로 관측가능 교사(numeraire sts5@1800) 증류. **학생 건전성 OK(yr073 붕괴와 달리 action mix 정상)이나 교사 배 보호 미재현 — 학생 berth 155~224 vs 교사 31~152(2~5배 악화)·전 셀 완주 실패·SF보다도 나쁜 뭉개진 정책·top1_disagree 0.167.** 결론: **2a 교사층 목적 실효는 배포 학생으로 전이 안 됨. 병목=증류/관측**(feed-forward 학생이 교사 1800s rollout 미래 미관측, 검증 예측 적중). 후속=YR-087(학생 본선 feature 강화). · **재판정 verdict**: 기존 채택 체크포인트 기각(목적 바뀜)·재학습(증류)은 현 구조서 실패 → 배포 정책은 증류/관측 축(별개) 해결 선결 |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
