"""v3 ③ 구조 — Seller · Buyer 두 망과 배정기.

■ 담는 것
  - Seller: "이 트럭을 내놓으면 우리 블록이 얼마나 편해지나" → `R_src`
  - Buyer:  "이 트럭을 받으면 우리 블록이 얼마나 힘들어지나" → `B_dst`
  - 배정기: `점수 = Seller(src) − Buyer(dst) − ΔC_route` 로 조립

■ 왜 나누나
  라벨이 처음부터 세 조각의 뺄셈인데 망 하나가 **결과만** 본다. 두 큰 수의 차라
  양쪽이 커도 차가 작으면 신호가 상쇄돼 사라지고, 어느 쪽이 틀렸는지도 모른다.
  나누면 신용 배분이 갈리고, 채점이 O(N²) → O(N) 으로 준다.

■ 하지 말 것
  두 망을 따로 학습시키지 않는다 — 눈금이 어긋난다. 같은 `Q_SCALE` 로 동시 학습.
■ 설계 문서: `.claude/docs/architecture/03-결정층.md`
"""

from .buyer import Buyer
from .explore import draw as explore_draw, pick as explore_pick
from .market import EpochResult, Market
from .nets import (ADV_SCALE, PHI_SCALE, BuyerNet, SellerNet, from_advantage,
                   from_scaled, to_advantage, to_scaled)
from .offer import (BUY, KEEP, REJECT, RESOLVER_KEEP, SELL, SPACE, TIME, Coord,
                    Offer, Response)
from .resolver import ResolveResult, Resolver, Trade
from .seller import Seller

__all__ = ["Seller", "Buyer", "Resolver", "Market", "SellerNet", "BuyerNet",
           "Offer", "Response", "Coord", "Trade", "ResolveResult",
           "EpochResult", "KEEP", "SELL", "BUY", "REJECT", "SPACE", "TIME",
           "RESOLVER_KEEP", "PHI_SCALE", "to_scaled", "from_scaled",
           "ADV_SCALE", "to_advantage", "from_advantage",
           "explore_draw", "explore_pick"]
