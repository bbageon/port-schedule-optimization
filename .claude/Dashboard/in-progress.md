# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-125 | RL | **1단계: Q값 진단 (무학습)** | 🔴 | 2026-07-29 | **1단계 완료 — H-A 확정(예상보다 근본적)**: 12 체크포인트 전부 보정오차(G−Q̂) ≈ 실현수익 전체(+138~+215 vs G 140~218) = **Q 가 비용-투-고를 사실상 미학습**. 원인 역산 = 경사 갱신 총 ~1,000회 vs 지평 150+ 결정(부트스트랩 예산 절대 부족). 세 기각(119/121/122)은 "빈 Q 위의 개입"으로 재해석. **2단계 선택 대기**: (i) 학습 예산 단일축 = **[YR-127](ready.md) 등록**(권고 — 최소 수정 직접 검증) → 실패 시 (ii) 차분 1-step 이식(YR-125 2단계) · [spec](../docs/dashboard-task-specs/YR-125-diff-credit-port.md)·[1단계 report](../../outputs/reports/yr125_qvalue_diagnosis/report.md) |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
