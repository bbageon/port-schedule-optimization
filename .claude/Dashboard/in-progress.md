# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-129 | RL | **차분 실패 판별 진단 — D̂ 적합도 4축 (무학습)** | 🔴 | 2026-07-30 | **외부 3차 피드백 설계로 보강 후 착수 (사용자 지시)**: A 훈련분포 적합(r·부호·MAE·기울기) / B 신규상태 적합 / C 미선택 후보 적합(전 후보 라벨·결정 내 ρ·top-1·0-앵커) / D 종결 잔여 상관. **분기표 동결**: A r<0.5→고정 데이터셋 오프라인 epoch(생성·갱신 혼합 금지) · B 낮음→YR-124 · 미선택만 낮음→K-후보 학습 · 전부 양호+운영실패→종결비용/창 · REPO MAE 2배→유형별 검사. **결과 전에는 학습량 확대·종결비용·YR-124 어느 것도 착수 금지** (피드백 지시). 부산물 = 고정 라벨 데이터셋(dataset_s*.pt) · [spec](../docs/dashboard-task-specs/YR-129-diff-fit-diagnosis.md)·[하네스](../../src/yard_rl/experiments/yr129_diff_fit.py) |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
