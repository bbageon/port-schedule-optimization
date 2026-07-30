# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-131 | RL | **K-후보 커버리지 단일축 (a: 후보 확대만 / b: 실패 시 순위손실)** | 🔴 | 2026-07-30 | **5차 피드백 축 분리 반영 후 착수**: 131-a = 상태당 전 후보 라벨(YR-129 live 2.1만~4.1만/시드, 신규 rollout 0)·Huber·YR-130 프로토콜 유지 — 유일 차이 표본 구성. 131-b(a 실패 시에만) = 손실만 pairwise margin 순위로 교체. **J1 동결**: 3/3 held-out r≥0.5 ∧ ρ≥0.30 ∧ top1≥0.35 (참조 YR-130: 0.29~0.39/≈0/0.16~0.18). **기준행동 SF-SPT 교체는 이 축에서 제외** — 같은 상태 전 후보에서 같은 기준값을 빼므로 순위 불변(개입 게이트·해석용 → 사다리 6단). 관찰 부속 = 혼잡도 5분위 신호 분석(600s 변별 소멸 가설 검증). a 통과→5단(종결 잔여)·모두 실패→4단(결정 분해·상태 별칭 검사) · [spec](../docs/dashboard-task-specs/YR-131-k-candidate-ranking.md)·[하네스](../../src/yard_rl/experiments/yr131_k_candidates.py) |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
