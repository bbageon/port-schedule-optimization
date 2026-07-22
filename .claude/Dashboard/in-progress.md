# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-087 | RL | 증류 격차 해소 — 학생 관측 본선 feature 강화 / 룩어헤드 (YR-080 2b negative 파생) | 🟠 | 2026-07-22 | **착수 (사용자 결정 2026-07-22)**: YR-080 2b 진단 = 기준재 교사 배 보호가 feed-forward 학생으로 전이 안 됨(학생 berth 2~5배 악화·완주 실패·top1_disagree 0.167), 원인 = 교사 우위가 1800s rollout **미래**에 있는데 학생 관측은 현재 집계만·본선 그룹 rich feature(여유시간·위험·예상지연) 부재. **1단계 완료 — 관측 강화 가설 기각 (negative)**: ①진단 = encode_observation 이 본선 그룹 8필드(slack/risk/잔여작업/버퍼)를 통째 드롭·후보 훅 2개 죽음(vessel_risk_delta 항상 None·deadline_slack naive) 확인. ②v1(후보 훅 2개 관측값 교정, 214불변) → 개선 없음(희소·후행·후보한정). ③v2(본선 그룹 전역 인코딩 top-K, 214→246, use_vessel opt-in 하위호환) — 본선 블록 실신호 확인(slack −1344s·risk 0.75)에도 **개선 없음**. 3판본 학생 berth 교사 대비 2~4배 악화·완주 실패·top1_disagree ~0.16 동일. **결론: 병목은 관측 아님** — feed-forward 가 교사 1800s rollout 을 단일 forward 로 amortize 못하는 근본 한계(방법론 검증 강형 예측 적중). **미해결 fork**: 학생 룩어헤드(shallow rollout/beam)·DAgger/FT·배포정책이 rollout 유지·train-fit 진단(미학습 vs 근본 구분) 중 택. 사용자 방향 대기. 관측 수정은 correct enrichment 로 보존(future 룩어헤드 학생용) |
 | 🟠 | 2026-07-21 | [spec](../docs/dashboard-task-specs/YR-080-objective-contract-redesign.md) · **1단계 완료** (2026-07-22, `827bba4`~`9ed4923` — [전략 v5 §9](../docs/strategy-history/2026-07-21-본선처리-전략.md)): 양하 역전 수정·인과 연결·비용==KPI 등식·기준재 config·manifest 재동결·적재 seam. · **2a 완료** (2026-07-22, [보고서](../../outputs/reports/yr080_readjudicate/yr080_2a_report.md)): 교사층 목적 재판정. **교락 해소(15-seed 같은-창 짝지음): LEX는 창 늘려도 배 못 지킴(무의), 같은 창서 기준재 목적이 배 유의 보호(@5400 −20.8★/−52.0★) → 사전식 회귀 확정 기각, 목적 재설계는 실제 효과.** 정직한 정정: 초기 CI 버그(pstdev+z)·P95 누락 → 표본sd+t·P95검정 교정(적대 3관점 검증). 배↔트럭 P95 상충 → **P95는 채택 게이트로(목적 밖, 사용자 결정)**. · **2b 완료 (2026-07-22, negative — [2b 보고서](../../outputs/reports/yr080_distill/yr080_2b_report.md))**: 학생 재현성 우선(사용자 결정)으로 관측가능 교사(numeraire sts5@1800) 증류. **학생 건전성 OK(yr073 붕괴와 달리 action mix 정상)이나 교사 배 보호 미재현 — 학생 berth 155~224 vs 교사 31~152(2~5배 악화)·전 셀 완주 실패·SF보다도 나쁜 뭉개진 정책·top1_disagree 0.167.** 결론: **2a 교사층 목적 실효는 배포 학생으로 전이 안 됨. 병목=증류/관측**(feed-forward 학생이 교사 1800s rollout 미래 미관측, 검증 예측 적중). 후속=YR-087(학생 본선 feature 강화). · **재판정 verdict**: 기존 채택 체크포인트 기각(목적 바뀜)·재학습(증류)은 현 구조서 실패 → 배포 정책은 증류/관측 축(별개) 해결 선결 |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
