"""`slot_load` 를 채우면 Buyer 판단이 바뀌는가 — **v3 를 안 고치고** 재기만 한다.

■ 무엇이 문제인가 (2026-08-26 측정)
  `Offer.slot_load` 는 필드도 있고 주석도 있고(*"TIME 일 때 그 칸의 예약 적재율"*)
  Buyer 특징 4칸 중 한 자리도 차지한다. **그런데 Seller 가 안 채운다.**

      seller.py :  Offer(doc_key=..., src_block=src, coord=chosen,
                         src_load=inside_count(...))     # ← slot_load 누락

  실측: 시간 이연 offer **445건 전부 0.0**. 죽은 칸이다.

  그래서 Buyer 는 *"15분 미루자"* 와 *"60분 미루자"* 를, *"텅 빈 칸"* 과
  *"꽉 찬 칸"* 을 **똑같이** 본다. 그런데 [[YR-232]] 가 확인한 대로
  **재배치 이득의 전부가 시간 축**이다.

■ 이 꾸러미가 하는 일 — **고치지 않고 묻는다**
  *"만약 채웠다면 Buyer 의 결정이 몇 건이나 뒤집혔을까?"*

  ① v3 를 그대로 굴리면서 offer·응답·특징 행을 **함께** 기록한다
     (Buyer 를 감싸기만 한다 — `actors/buyer.py` 는 한 줄도 안 바뀐다)
  ② 각 시간 이연 offer 에 대해 `slot_load` 를 **사후 계산**한다
  ③ 그 값을 넣은 행을 **같은 망**에 다시 통과시켜 결정이 바뀌는지 본다

  바뀌는 게 없으면 고칠 이유가 없다. 많이 바뀌면 그때 고친다.

■ `slot_load` 를 무엇으로 잴 것인가
  *"그 칸에 이미 몇 대가 예약돼 있나"* 이므로, **새 도착 시각 ±7.5분** 안에
  같은 블록으로 통지된 물량 수로 잰다(칸 폭 15분과 맞춘다).

  `features.block.announced_around` 를 그대로 쓴다 — **통지값만** 읽으므로
  정보 경계가 이미 검증된 함수다([[YR-230]]).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..v3.actors.offer import BUY, REJECT, SPACE
from ..v3.features import BLOCK_DIM, BUYER_ROW_DIM, CANDIDATE_DIM
from ..v3.features.block import announced_around

#: Buyer 행에서 `slot_load` 가 앉는 칸 번호 (블록 9 + 후보 6 + offer 4 의 마지막)
SLOT_LOAD_IDX = BLOCK_DIM + CANDIDATE_DIM + 3
#: 칸 폭 15분의 절반 — "그 칸" 을 도착 시각 ±7.5분으로 읽는다
SLOT_HALF_W_S = 450.0
#: Buyer 특징의 대수 눈금 (`buyer_offer_features` 가 `/10.0` 한다)
LOAD_SCALE = 10.0


@dataclass
class BuyerTape:
    """Buyer 를 **감싸기만** 한다 — 원본은 한 줄도 안 바뀐다.

    `Market` 은 `buyer.respond(...)` 만 부르므로 같은 자리에 꽂힌다.
    """

    inner: object
    rows: list = field(default_factory=list)

    @property
    def trail(self):
        return self.inner.trail

    def respond(self, mbt, offer, **kw):
        r = self.inner.respond(mbt, offer, **kw)
        e = self.inner.trail[-1]
        self.rows.append({
            "t": e["t"], "doc_key": e["doc_key"], "action": e["action"],
            "is_time": offer.coord.kind != SPACE,
            "block": (offer.src_block if offer.coord.kind != SPACE
                      else offer.coord.block),
            "arrive_s": offer.coord.slot_start_s,
            "defer_s": offer.coord.defer_s,
            "row_buy": list(e["row_buy"]), "row_reject": list(e["row_reject"]),
            "adv_buy": e["adv_buy"], "adv_reject": e["adv_reject"],
        })
        return r

    def __getattr__(self, k):
        return getattr(self.inner, k)


def slot_load_of(mbt, block: str, arrive_s: float, orders) -> int:
    """그 칸에 이미 통지된 물량 — **공개 정보만**([[YR-230]] 와 같은 출처)."""
    if block not in mbt.blocks:
        return 0
    return announced_around(mbt, block, float(arrive_s), orders,
                            half_w=SLOT_HALF_W_S)


@torch.no_grad()
def replay(rows, *, buyer_net, mbt, orders) -> dict:
    """`slot_load` 를 채운 행을 **같은 망**에 다시 통과시킨다.

    ⚠️ 이건 **사후 재생**이다 — 결정이 바뀌면 그 뒤 세계도 달라지므로, 여기서
    세는 "뒤집힘" 은 *"첫 갈림길에서 몇 건이 달라지나"* 이지 하루 전체의 결과가
    아니다. 하루 효과는 실제로 고쳐서 굴려야 안다.
    """
    flips, filled, vals = 0, 0, []
    same_dir = 0
    for r in rows:
        if not r["is_time"]:
            continue
        sl = slot_load_of(mbt, r["block"], r["arrive_s"], orders)
        vals.append(sl)
        if sl == 0:
            continue
        filled += 1
        rb = list(r["row_buy"])
        rb[SLOT_LOAD_IDX] = sl / LOAD_SCALE
        x = torch.tensor([rb, r["row_reject"]], dtype=torch.float32)
        c = buyer_net(x)
        new = BUY if float(c[0]) <= float(c[1]) else REJECT
        if new != r["action"]:
            flips += 1
        else:
            same_dir += 1
    return {"n_time": len(vals), "n_nonzero": filled, "flips": flips,
            "kept": same_dir, "values": vals}


class SellerNoSlotLoad:
    """`slot_load` 를 **끈** Seller — 고치기 전 동작을 그대로 재현한다.

    v3 를 안 고치고 A/B 를 만들기 위한 껍데기다. `Market` 은 `seller.decide(...)`
    와 `seller.trail` 만 쓰므로 같은 자리에 꽂힌다.

    ⚠️ **결정 자체는 안 바꾼다** — Seller 는 원래도 `slot_load` 를 안 봤다
    (행동 특징 8칸에 없다). 바뀌는 것은 **Buyer 에게 전달되는 값**뿐이다.
    """

    def __init__(self, inner):
        self.inner = inner

    @property
    def trail(self):
        return self.inner.trail

    def decide(self, *a, **kw):
        import dataclasses
        off = self.inner.decide(*a, **kw)
        return None if off is None else dataclasses.replace(off, slot_load=0.0)

    def __getattr__(self, k):
        return getattr(self.inner, k)


class BuyerAlwaysAccept:
    """거부권을 **끈** Buyer — *"Buyer 가 값어치를 하는가"* 를 재는 팔.

    v3 의 연구 주장은 **독립 행위자 + 상호 동의 시장**이다. 그런데 실측상
    선택의 정보는 Seller 에 몰려 있고(도착 압력·`defer_s`·후보 6칸),
    Buyer 는 굵은 거름망에 가깝다.

    그러면 물어야 한다 — **거부권이 실제로 비용을 줄이나?**

        현행        Seller 고름 → Buyer 가 37.6% 거절
        이 팔       Seller 고름 → **전부 통과**(물리 거부만 남는다)

    둘을 같은 날에 붙이면 답이 나온다. `Market` 은 `buyer.respond(...)` 만
    부르므로 같은 자리에 꽂힌다 — v3 는 한 줄도 안 바뀐다.

    ⚠️ 물리 제약은 그대로 산다: 슬롯 용량이 0 이면 원래 코드가 먼저 거절하고,
       엔진이 못 받으면 `bridge._confirm` 이 되돌린다. 이 팔이 끄는 것은
       **정책적 거부권**뿐이다.
    """

    def __init__(self, inner):
        self.inner = inner

    @property
    def trail(self):
        return self.inner.trail

    def respond(self, mbt, offer, **kw):
        from ..v3.actors.offer import Response
        left = kw.get("slot_capacity_left")
        if left is not None and left <= 0:
            return Response(offer=offer, action=REJECT)   # 용량 = 물리
        self.inner.respond(mbt, offer, **kw)              # 기록은 남긴다
        self.inner.trail[-1]["action"] = BUY
        return Response(offer=offer, action=BUY)

    def __getattr__(self, k):
        return getattr(self.inner, k)
