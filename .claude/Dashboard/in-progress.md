# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).
>
> **사후 비용계약 변경(2026-07-30)**: YR-135의 현재 라벨은 계단형 트럭비용·본선 33의
> v1 계약이다. 결과는 V/A 구조 진단으로만 보존하며 새 정책 채택 근거로 승계하지 않는다.
> 구조가 통과하면 YR-136 점증비용 v2로 신규 라벨을 생성해야 한다.

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-135 | RL | **공동 Advantage-Q 1단계 — V/A 분리 구조 (BlockQ-v3)** | 🔴 | 2026-07-30 | **7차 피드백·사용자 지시로 v2 마감 후 착수**: 유일 변경 = 점수망 구조 `Q = V(상태) + A(후보) − mean A`. 표적 = 후보별 절대 C600(차분·잔여 없음), SF-SPT = 동등 경쟁자. 라벨 3대역 신규(judge = BASE+1100/1101 미열람). **판정 동결**: J1 3/3 ρ≥0.30∧top1≥0.35 · J2 3/3 r(Q̂,C600)≥0.5 · J3 regret ≤ 131-b 참조. 성공→에피소드 판정 / J1만 미달→2단계 순위 보조손실 / J2 미달→구조 재검 · [spec](../docs/dashboard-task-specs/YR-135-advantage-q.md)·[하네스](../../src/yard_rl/experiments/yr135_advantage_q.py)·[전환 기록](../docs/strategy-history/2026-07-30-BlockQ-v3-AdvantageQ-전환.md) |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
