# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-125 | RL | **차분 신용 이식 — 2단계 학습 중 (학습 표적 단일축, 계약 7항 동결)** | 🔴 | 2026-07-29 | 1단계 Q값 진단 완료(H-A) → YR-127(갱신×20) 기각·YR-128(순위결함 확정) 경유로 재정렬 → **2단계 착수 (사용자 지시 2026-07-29, 결과 미열람 동결)**: base = WAITON recipe 전부 동일, **유일 차이 = 학습 표적** — TD 부트스트랩 절대비용 → 1-step 반사실 차분 `D = C600(행동) − C600(전략 WAIT 기준행동)` 회귀 (표적 교체는 shaping·UNSERVED·γ 항 제거를 포함 — 표적 정의의 일부로 고지). **계약 7항 동결**(외부 피드백 2026-07-29): ①차분 정의·부호 ②공동 기준행동 = 결정 참여 크레인만 WAIT ③강제/구조적 WAIT 표본 제외 ④동일조건 반사실(같은 상태·이벤트·후속정책 SF·종료시각) ⑤창 절단 공통 상쇄 + 300/1200s 민감도는 후속 진단 ⑥C600 은 CTDE 학습 교사정보만(실행망은 현재 관측만 — YR-107) ⑦판정 = **P1** 순위 회복(3/3 시드 P(cw|qw)≥0.5·top1≥0.35·ρ≥0.30) ∧ **P2** 전략WAIT<0.479·WAIT장악 0 ∧ **P3** 짝 총비용 upper95<+10·A→O<+1분 ∧ **P4** REPO≤0.15·REPO장악 0 (풍선 감시) ∧ **P5** 완주·backlog·미건전 guard. 스모크 통과(162표본/ep·11.6s·D<0 비율 0.44). 성공해도 채택 아님(별도 후보평가) · [spec](../docs/dashboard-task-specs/YR-125-diff-credit-port.md)·[하네스](../../src/yard_rl/experiments/yr125_diff_credit.py) |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
