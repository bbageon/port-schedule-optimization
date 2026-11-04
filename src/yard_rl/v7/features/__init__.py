"""v3 ④ 정보 — 정책 특징 — 경계 안쪽인데 안 주던 것.

■ 담는 것
  - 시각 (`t / sim_end`) — 무대에 이중 피크가 있는데 정책에 시계가 없다
  - 실제 대기 대수 (`block_arrival ≤ t < service_start`) — 채점은 이걸 쓰는데
    정책에는 `inside`(블록 안 전부)를 준다
  - 크레인 예상 서비스 시각 · 목적지 블록 크레인 부하

■ 규칙
  **결정 시점에 존재하지 않는 값은 읽지 않는다.** 위 넷은 전부 결정 시점에
  관측 가능하다 — 못 주는 게 아니라 안 주고 있던 것이다.
  교사가 만든 라벨을 **입력에 섞지 않는다** (그 순간 미래가 샌다).

■ 넣기 전에
  **잡음 하한을 먼저 잰다** (학습 불필요·20분). 안 내려가면 넣지 않는다.
■ 설계 문서: `.claude/docs/architecture/05-정보경계.md`
"""

from .block import (BLOCK_DIM, BLOCK_DIM_BUYER, announced_around, block_features,
                    inside_count, pipeline_count,
                    waiting_count)
from .candidate import (BUYER_OFFER_DIM, BUYER_ROW_DIM, CANDIDATE_DIM,
                        SELLER_ACTION_DIM, SELLER_ROW_DIM, buyer_offer_features,
                        candidate_features, seller_action_features)

__all__ = ["BLOCK_DIM", "BLOCK_DIM_BUYER", "CANDIDATE_DIM", "SELLER_ACTION_DIM", "BUYER_OFFER_DIM",
           "SELLER_ROW_DIM", "BUYER_ROW_DIM", "block_features",
           "candidate_features", "seller_action_features",
           "buyer_offer_features", "waiting_count", "inside_count",
           "announced_around",
           "pipeline_count"]
