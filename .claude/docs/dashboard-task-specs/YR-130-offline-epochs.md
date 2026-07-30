# YR-130 — 고정 데이터셋 오프라인 epoch 학습 (YR-129 R-미적합 처방)
> 상태: **in-progress (2026-07-30 — YR-129 분기표의 동결 처방, 즉시 착수)**
> 근거: [YR-129 판별](../../../outputs/reports/yr129_diff_fit/report.md) — A 훈련 적합
> r 0.14~0.45 (3/3 < 0.5)·기울기 4~21 = 회귀 미적합. 처방(외부 3차 피드백) = 표본
> 재생성 금지, 고정 데이터 train/val 분리·epoch 반복·val 정체 시 종료.

## 계약 ([하네스](../../../src/yard_rl/experiments/yr130_offline_epochs.py) docstring 이 동결 정본)

- 데이터 고정 = YR-129 dataset_s*.pt: train = 훈련 대역 **실행 표본만**(상태당 1개 —
  미선택 라벨 혼입 금지, 그건 K-후보 분기), val = 신규 대역 실행 표본.
- fresh init·base 하이퍼 동일(Adam 5e-4·배치 64·Huber·clip 10)·최대 2000 epoch·
  val Pearson r patience 100·best-val 스냅샷.
- 정직 고지: 시드당 train ~0.9~1.7k 는 원 학습 81k 보다 작음 — 질문은 "갱신 확대
  충분성"이 아니라 **"고정 표본에서 epoch 만으로 적합이 사는가"**.
- **J1 (판별)**: 3/3 시드 best val r ≥ 0.5 → 갱신 병목 확정 / 정체 < 0.5 → 표본량·
  표현 한계 → K-후보(미선택 라벨 기성) 또는 YR-124. 일부 시드만이면 전수 명기.
- **J2 (관찰)**: best-val 망으로 YR-128 순위 재진단 — 적합 회복이 순위로 이어지는지.

## Evidence
사전동결: (커밋 예정) · 결과: outputs/reports/yr130_offline_epochs/
