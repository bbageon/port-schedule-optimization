# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-127 | RL | **학습 예산 단일축 — 경사 갱신 ×20 (YR-125 1단계 후속)** | 🔴 | 2026-07-29 | **사전등록 동결 후 학습 중**: base = YR-119 WAITON recipe 전부 동일, 유일 차이 = `y90.UPDATE_MULT = 20`(에피소드당 갱신 루프 횟수만 배수 — 내용물 불변, 실수행 갱신 수 계수 박제). 판정(동결) = **N1** 보정 회복(3/3 시드 bias_ratio = |mean(G−Q̂)|/mean|G| ≤ 0.5 — 참조 WAITON ≈ 0.99) ∧ **N2** 전략적 WAIT ≤ 0.5×WAITON 재평가·WAIT 점유>60% 없음 ∧ **N3** 짝지은 총비용 upper95 < +10·미건전 비증가. 실패 시 YR-125 2단계(차분 1-step)로 · [spec](../docs/dashboard-task-specs/YR-127-training-budget-axis.md)·[하네스](../../src/yard_rl/experiments/yr127_training_budget.py) |
| YR-125 | RL | **1단계: Q값 진단 (무학습)** | 🔴 | 2026-07-29 | **1단계 완료 — H-A 확정(예상보다 근본적)**: 12 체크포인트 전부 보정오차(G−Q̂) ≈ 실현수익 전체(+138~+215 vs G 140~218) = **Q 가 비용-투-고를 사실상 미학습**. 원인 역산 = 경사 갱신 총 ~1,000회 vs 지평 150+ 결정(부트스트랩 예산 절대 부족). 세 기각(119/121/122)은 "빈 Q 위의 개입"으로 재해석. **2단계 분기 진행 중**: (i) 학습 예산 단일축 = **YR-127 착수(2026-07-29)** → 실패 시 (ii) 차분 1-step 이식(YR-125 2단계) · [spec](../docs/dashboard-task-specs/YR-125-diff-credit-port.md)·[1단계 report](../../outputs/reports/yr125_qvalue_diagnosis/report.md) |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
