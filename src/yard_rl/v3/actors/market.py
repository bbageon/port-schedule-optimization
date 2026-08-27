"""한 epoch 을 굴린다 — Seller 방송 → Buyer 독립 응답 → Resolver 매칭.

설계 정본: `.claude/docs/architecture/03-결정층.md` §4

    ① 이번 epoch 에 처음 자격을 얻은 작업 J_t 를 모은다
    ② 각 Seller 가 KEEP / SELL(공간 b · 시간 슬롯 k) 을 고른다   (SELL 은 방송)
    ③ 각 Buyer(블록 또는 슬롯)가 BUY / REJECT 로 독립 응답
    ④ Resolver 가 동의 edge 만 모아 batch 확정

■ 최초 통지 1회 ([[YR-203]])
  v2 의 "60초마다 재검토" 가 사라진다. 공개 ETA 가 창에 **처음 들어온** epoch 에서만
  판단하고 KEEP 이든 재배정이든 **즉시 잠근다** — 반복 제안·거절되면 학습 표본이
  중복되고 공로가 왜곡되기 때문이다.

■ 창은 계산량을 안 늘린다
  한 epoch 배치 = 그 60초에 자격을 얻은 건수(7,500대 기준 평균 5.2건·피크 17건)라
  **창 길이와 무관**하다. 창이 바꾸는 것은 "언제 결정하나" 다 → [[YR-190]] 이 스윕.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .offer import RESOLVER_KEEP, SPACE, TIME
from .resolver import ResolveResult, Resolver


@dataclass
class EpochResult:
    offers: list = field(default_factory=list)
    responses: list = field(default_factory=list)
    resolve: ResolveResult | None = None
    newly_eligible: list[str] = field(default_factory=list)

    @property
    def traded_edges(self) -> int:
        return self.resolve.traded_edges if self.resolve else 0


class Market:
    """재배치 층 한 판. 크레인 층은 이걸 기다리지 않는다 — 작업 목록만 바뀐다."""

    def __init__(self, seller, buyer, resolver: Resolver, *,
                 window_s: float = 1800.0, window_cap_s: float | None = None):
        self.seller = seller
        self.buyer = buyer
        self.resolver = resolver
        self.window_s = float(window_s)
        self.window_cap_s = window_cap_s
        self.decided: set[str] = set()          # 최초 1회 — 다시 거래하지 않는다
        #: ★예약 시각 순 색인 — `newly_eligible` 이 매번 전량을 훑지 않게 한다.
        #: 계약: `orders` 의 **키 집합은 무대를 세운 뒤 안 바뀐다**(개방 루프 —
        #: 트럭 명단이 미리 확정된다). 값은 바뀌지만(`relocate`·`defer`) 그건
        #: 이미 `decided` 라 자격 검사에서 걸러진다.
        self._by_reserve: list | None = None

    # ------------------------------------------------------------------ 자격
    def effective_window_s(self, lead_s: float | None) -> float:
        """검토 창 = `min(리드, WINDOW_CAP)` — 통지 안 된 트럭은 후보가 못 된다.

        리드가 고정이면 창을 늘려도 볼 게 없다. [[YR-190]] 이 이 겸직을 가른다.
        """
        w = self.window_s if lead_s is None else min(self.window_s, float(lead_s))
        if self.window_cap_s is not None:
            w = min(w, float(self.window_cap_s))
        return w

    def _reserve_index(self, orders) -> list:
        """★예약 시각으로 정렬한 색인 — **한 번만** 만든다.

        ■ 왜 필요한가 (2026-08-27 실측)
          옛 코드는 epoch 마다 `sorted(orders.items())` 로 **전량을 정렬**했다.
          하루 무대는 오더가 수천 개라 문제가 없었지만 30일 무대는 **21만 개**다:

              하루 1,440 epoch × 21만 정렬  +  분기 세계마다 180 epoch × 21만

          자격 검사는 `reserve ∈ (t, t + window_s]` 인 오더만 볼 수 있다
          (`effective_window_s` 가 `window_s` 를 절대 안 넘기므로). 부하 12,500 이면
          그 창에 **약 260건** — 21만 대신 260건만 보면 된다.

        ■ 값이 바뀌어도 색인은 안 상한다
          `defer` 가 예약 시각을 옮기지만, 그 오더는 그때 이미 `decided` 라
          자격 검사에서 먼저 걸러진다. `relocate` 는 블록만 바꾼다.
        """
        if self._by_reserve is None:
            self._by_reserve = sorted(
                (o.in_out_reserve_s, k) for k, o in orders.items())
        return self._by_reserve

    def newly_eligible(self, mbt, t: float, *, orders, records,
                       epoch_s: float = 60.0) -> list[str]:
        """이번 epoch 에 **처음** 자격을 얻은 작업. 결정된 것은 다시 안 본다.

        돌려주는 순서는 **docKey 오름차순** — 옛 구현(`sorted(orders.items())`)과
        같아야 한다. 순서가 바뀌면 Seller 가 다른 판단을 내려 세계가 갈린다.
        """
        import bisect

        idx = self._reserve_index(orders)
        lo = bisect.bisect_right(idx, (t, chr(0x10FFFF)))
        hi = bisect.bisect_right(idx, (t + self.window_s, chr(0x10FFFF)))
        out = []
        for _, doc_key in idx[lo:hi]:
            if doc_key in self.decided:
                continue
            rec = records.get(doc_key)
            if rec is None or rec.gate_in_s is not None:
                continue                          # 게이트를 지났다 = 자격 없음
            o = orders[doc_key]
            lead = o.in_out_reserve_s - o.copino_notice_s
            w = self.effective_window_s(lead)
            dt = o.in_out_reserve_s - t
            if 0.0 < dt <= w and dt > w - epoch_s:   # ← "처음" 들어온 epoch 만
                out.append(doc_key)
        out.sort()                                   # 옛 구현과 같은 순서
        return out

    # ------------------------------------------------------------------ 한 판
    def step(self, mbt, t: float, *, orders, records, end_s: float,
             time_slots_of=None, quay_of=None, slot_capacity_left=None,
             epoch_s: float = 60.0) -> EpochResult:
        res = EpochResult()
        res.newly_eligible = self.newly_eligible(
            mbt, t, orders=orders, records=records, epoch_s=epoch_s)
        if not res.newly_eligible:
            res.resolve = ResolveResult()
            return res

        # ② Seller 가 고른다 (SELL 은 방송된다)
        for doc_key in res.newly_eligible:
            o, rec = orders[doc_key], records[doc_key]
            src = o.con_loc
            if src not in mbt.blocks:
                continue
            slots = time_slots_of(doc_key, t) if time_slots_of else ()
            offer = self.seller.decide(
                mbt, src, doc_key=doc_key, order=o, rec=rec, t=t,
                records=records, orders=orders, end_s=end_s,
                n_cands=len(res.newly_eligible), time_slots=slots,
                quay_of=quay_of)
            if offer is None:
                self.decided.add(doc_key)         # KEEP — 즉시 잠근다
            else:
                res.offers.append(offer)

        # ③ Buyer 가 독립 응답한다 (다른 Buyer 의 응답은 못 본다)
        for offer in res.offers:
            left = (slot_capacity_left(offer.coord.slot)
                    if (slot_capacity_left and offer.coord.kind == TIME) else None)
            res.responses.append(self.buyer.respond(
                mbt, offer, order=orders[offer.doc_key],
                rec=records[offer.doc_key], t=t, records=records,
                orders=orders, end_s=end_s, slot_capacity_left=left))

        # ④ Resolver 가 동의 edge 만 확정한다
        res.resolve = self.resolver.resolve(res.responses, offers=res.offers)

        for tr in res.resolve.trades:
            self.decided.add(tr.doc_key)
            rec = records[tr.doc_key]
            if tr.kind == SPACE:
                rec.record_swap(prev_block=tr.src_block, reason="SPACE")
            else:
                rec.record_swap(prev_block=tr.src_block, reason="TIME")
        for doc_key in res.resolve.kept:
            self.decided.add(doc_key)             # RESOLVER_KEEP — 잠근다
            records[doc_key].con_swap_reason = RESOLVER_KEEP

        return res
