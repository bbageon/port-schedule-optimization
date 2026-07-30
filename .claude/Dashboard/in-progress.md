# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-132 | RL | **종결 잔여 표적 단일축 — D_H = D + λ·Δ잔여 (λ=1.0 동결)** | 🔴 | 2026-07-30 | **6차 피드백 승인·조건 5 반영 착수**: 표적만 D→D_H(잔여 = 창 종료 미완 수, λ=1.0 가격앵커 — **개수 잔여는 진단용 지위**, 성공 시 YR-123 비용가중 잔여로 승계). 학습 = 131-b 프로토콜·rollout 0. **판정 = 새 대역 BASE+900/901(미열람) 라벨**: J1 3/3 ρ_H≥0.30 ∧ top1≥0.35 · J2 3/3 선택 후 손실 regret(132) < regret(131-b) 같은 결정 짝. 성공→혼합손실→에피소드 검증 / 실패→상태 별칭·비동기 정보 검사. 구조 정정(양방향 Block Q↔Resolver·TransferQuoteQ 미구현 갭)은 로드맵 문서에 박제 · [spec](../docs/dashboard-task-specs/YR-132-terminal-residual-target.md)·[하네스](../../src/yard_rl/experiments/yr132_terminal_residual.py) |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
