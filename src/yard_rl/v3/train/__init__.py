"""v3 학습 진입점 — 교사가 라벨을 만들고 학생이 배운다.

■ 2단 구조
  교사 (reward/)  반사실 rollout → 정확한 라벨      학습 때만 · 느림
     ↓ 라벨
  학생 (actors/)  Seller · Buyer 망 → 즉시 추론     배포 · 빠름

  교사는 망이 아니다 — 시뮬레이터를 두 번 굴리는 절차이고 학습할 파라미터가 없다.

■ 계약
  - 라벨은 목표에만 쓴다. **입력에 섞지 않는다.**
  - 탐색(ε·σ)은 학습 중에만. 평가는 0.
  - 가중치는 **마지막 회차 고정** — 중간 지점 고르기 금지.

■ 설계 문서: `.claude/docs/architecture/06-학습과-판정.md` §1-1
"""

from .fit import FitReport, StudentTrainer, explore_sigma
from .labels import LabelCollector, LabelSet, Sample
from .loop import (DIAGNOSTIC_BASE, EXPLORE_END, EXPLORE_START, IterReport,
                   LABELS_PER_ITER, TrainState, explore_at, run_training)

__all__ = ["StudentTrainer", "FitReport", "explore_sigma", "LabelCollector",
           "LabelSet", "Sample",
           "run_training", "TrainState", "IterReport", "explore_at",
           "LABELS_PER_ITER", "EXPLORE_START", "EXPLORE_END", "DIAGNOSTIC_BASE"]
