# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-130 | RL | **고정 데이터셋 오프라인 epoch 학습 (YR-129 R-미적합 처방)** | 🔴 | 2026-07-30 | **YR-129 분기표의 동결 처방 — 표본 재생성 금지**: YR-129 dataset_s*.pt 고정(train = 훈련 대역 실행 표본만 ~0.9~1.7k — K-후보 혼입 금지 / val = 신규 대역), fresh init·base 하이퍼 동일·최대 2000 epoch·val r patience 100. **J1 동결**: 3/3 val r≥0.5 → 갱신 병목 확정 / 정체 <0.5 → 표본량·표현 한계(K-후보 or YR-124). **J2 관찰**: 순위 재진단(top-1·ρ·P(cw|qw)) — 적합 회복이 순위로 이어지는지. 정직 고지: 데이터가 원 81k 보다 작아 "고정 표본에서 epoch 만으로 사는가"의 판별임 · [spec](../docs/dashboard-task-specs/YR-130-offline-epochs.md)·[하네스](../../src/yard_rl/experiments/yr130_offline_epochs.py) |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
