"""엔진 ↔ v3 번역기. `review_fn` 으로 꽂힌다.

설계 정본: `.claude/docs/architecture/03-결정층.md` §4 · `05-정보경계.md`

■ 한 epoch 의 순서가 계약이다
      ① 사건 수집   엔진이 보낸 것을 기록에 옮긴다 (t 이전 것만)
      ② 시장        Seller 방송 → Buyer 독립 응답 → Resolver 매칭
      ③ 확정        동의된 거래만 엔진의 **원자 트랜잭션**으로 밀어 넣는다
  ①이 ② 앞이라야 정책이 그 시각의 진짜 상태를 본다.
  ③이 ② 뒤라야 한 epoch 안의 결정들이 서로의 결과를 못 본다(동시 결정).

■ ★정보 경계가 새기 가장 쉬운 자리
  엔진 장부에는 **미래가 이미 적혀 있다** — 트럭을 투입하는 순간
  `TruckTimes(gate_in=도착시각)` 이 들어가는데, 그 도착은 아직 오지 않았다.
  그래서 `_sync` 는 **`값 ≤ t` 인 것만** 기록으로 옮긴다. 이 한 줄이 없으면
  정책이 "이 트럭은 몇 시에 게이트를 통과할 것이다" 를 그냥 읽는다.

■ 실패는 KEEP 이다
  엔진 트랜잭션은 용량·상태·시각을 다시 검사하고, 하나라도 어긋나면 아무것도
  바꾸지 않는다(fail-closed). 그때는 거래를 취소하고 그대로 둔다 — 시장이 합의해도
  물리가 거부하면 물리가 이긴다.
"""
from __future__ import annotations

import copy

from ..world.integrated.terminal_stream import on_grid
from ..world.integrated.time_sell import MAX_ENTRY_DEFERRALS, try_time_sell
from ..actors.offer import RESOLVER_KEEP, SPACE
from ..schema import Stage, defer, relocate
from .orders import EPOCH_S


def epoch_on_grid(t: float, grid_s: float = EPOCH_S) -> bool:
    """60초 격자인가. 게이트인 시각에 열리는 epoch 에서는 시장을 안 연다."""
    return on_grid(t, grid_s)


class MarketBridge:
    """v3 재배치 층 ↔ 엔진. `mbt.run(..., review_fn=bridge.review)` 로 쓴다."""

    def __init__(self, market, layout, *, orders, records, end_s: float,
                 arm: str = "RL", grid_s: float = EPOCH_S,
                 slot_capacity: dict[int, int] | None = None,
                 cf_horizon_s: float | None = None,
                 on_decision=None):
        self.market = market
        self.layout = layout
        self.orders = orders
        self.records = records
        self.end_s = float(end_s)
        self.arm = arm
        self.grid_s = float(grid_s)
        self.slot_capacity = dict(slot_capacity or {})
        #: 이연 칸을 반사실 창 안으로 묶는 지평([[YR-216]]). None = 옛 동작(비교용).
        self.cf_horizon_s = None if cf_horizon_s is None else float(cf_horizon_s)
        self.slot_steps = (self.SLOT_STEPS_LEGACY if cf_horizon_s is None
                           else self.SLOT_STEPS_HORIZON)
        self.on_decision = on_decision      # 교사 훅 — 없으면 라벨을 안 만든다
        self.traded_edges = 0
        self.txn_failed = 0
        self.n_space = 0
        self.n_time = 0
        self.ledger: list[dict] = []

    # ------------------------------------------------------------------ ① 사건
    def _sync(self, mbt, t: float) -> None:
        """엔진이 보낸 사건을 기록에 옮긴다 — **이미 일어난 것만**."""
        for bid, sim in mbt.blocks.items():
            tl = getattr(sim, "time_ledger", None)
            if tl is None:
                continue
            for jid, tt in tl.records.items():
                rec = self.records.get(jid)
                if rec is None:
                    continue                       # 배경(본선) 작업 — 오더가 아니다
                self._stamp_past(rec, Stage.GATE_IN, tt.gate_in, t)
                self._stamp_past(rec, Stage.BLOCK_IN, tt.block_arrival, t)
                self._stamp_past(rec, Stage.JOB_DONE, tt.job_done, t)
                self._stamp_past(rec, Stage.GATE_OUT, tt.gate_out, t)
                if (tt.service_start is not None and rec.service_start_s is None
                        and tt.service_start <= t + 1e-9):
                    rec.observe_service_start(tt.service_start)
                j = sim.jobs.get(jid)
                if j is not None and j.rehandle_count:
                    rec.rehandles = int(j.rehandle_count)

    @staticmethod
    def _stamp_past(rec, stage: Stage, value, t: float) -> None:
        """★`값 ≤ t` 인 사건만 찍는다 — 미래 시각은 아직 사건이 아니다.

        60초마다 장부 전체를 훑으므로 **이미 찍은 단계는 건너뛴다.** 기록은 한 번뿐
        이라(00 §2) 두 번 찍으면 즉시 예외다 — 그 계약을 여기서 우회하지 않고
        `_stamped` 를 보고 넘긴다.
        """
        if value is None or value > t + 1e-9 or stage in rec._stamped:
            return
        rec.stamp(stage, float(value))

    # ------------------------------------------------------------------ 시간 좌표
    #: 이연 격자 — 15분 배수. 어느 칸이 좋은지는 Seller 가 고른다(03 §5).
    SLOT_GRID_S = 900.0
    #: 창 안으로 묶을 때는 **격자도 창에 맞춘다** — 15·30·45·60분.
    #: 옛 격자(15·30·60·120)를 그대로 두고 자르면 후보가 둘로 줄어 선택지가 빈약해진다.
    SLOT_STEPS_HORIZON = (1, 2, 3, 4)
    SLOT_STEPS_LEGACY = (1, 2, 4, 8)

    #: ★[[YR-232]] 진단 — True 면 이연 후보를 아예 안 낸다(공간 전용 RL).
    no_time = False

    def _time_slots(self, doc_key: str, t: float):
        if self.no_time:
            return ()
        """이연 후보 칸. 값 판단이 아니라 **실현 가능성 필터**다.

        ★[[YR-216]] — `cf_horizon_s` 가 있으면 **새 도착이 반사실 창 안에 남는
        칸만** 낸다:

            reserve + defer ≤ t + H

        창 밖으로 미루면 그 결정의 효과가 채점 창에서 안 보여 **라벨이 정확히 0**
        이 되고, 학습은 "그 행동은 무의미" 로 배운다(2026-08-22 실측 — 거래가
        성립했는데 Φ 가 factual 과 원 단위까지 같았다).

        고정 목록이 아니라 **오더마다 다르다.** 도착이 임박할수록 미룰 수 있는
        폭이 준다 — 물리적으로도 그게 맞다.
        """
        o = self.orders[doc_key]
        cap = None if self.cf_horizon_s is None else t + self.cf_horizon_s
        out = []
        for k in self.slot_steps:
            defer = self.SLOT_GRID_S * k
            start = o.in_out_reserve_s + defer
            if start >= self.end_s:
                break
            if cap is not None and start > cap + 1e-9:
                break                          # ★창 밖 — 채점할 수 없는 칸
            out.append((k, start, defer))
        return out

    def _slot_left(self, slot: int) -> int | None:
        cap = self.slot_capacity.get(int(slot))
        return None if cap is None else cap

    # ------------------------------------------------------------------ review
    def review(self, mbt, t: float) -> None:
        self._sync(mbt, t)
        if self.arm == "NO_REALLOC" or not epoch_on_grid(t, self.grid_s):
            return

        # ★분기점은 **결정 전**이다 (2026-08-22 동일성 불변식이 잡았다).
        #   결정 뒤에 뜨면 ① 그 오더가 이미 `decided` 라 분기 세계가 다시 결정하지
        #   않고 ② 거래가 이미 엔진에 확정돼 "안 팔았다면" 을 만들 수가 없다.
        #   증상은 강제한 세계와 안 한 세계의 Φ 가 **똑같이** 나오는 것이었다.
        pre = None
        if self.on_decision is not None and self._wants_snapshot(mbt, t):
            pre = {"mbt": copy.deepcopy(mbt),
                   "records": copy.deepcopy(self.records),
                   "orders": dict(self.orders),
                   "decided": set(self.market.decided)}

        res = self.market.step(
            mbt, t, orders=self.orders, records=self.records, end_s=self.end_s,
            time_slots_of=self._time_slots, quay_of=self.layout.quay_to_block_s,
            slot_capacity_left=self._slot_left, epoch_s=self.grid_s)
        applied = () if (res.resolve is None or not res.resolve.trades) \
            else self._confirm(mbt, t, res)
        self._note_decisions(mbt, t, res, applied=applied, pre=pre)

    def _wants_snapshot(self, mbt, t: float) -> bool:
        """이 epoch 에 라벨을 만들 일이 있는가 — **복제 전에** 싸게 판단한다.

        복제는 0.6초라 매 epoch 뜨면 하루에 14분이 그냥 사라진다. 자격자가 있고
        예산이 남았을 때만 뜬다. `newly_eligible` 은 부작용이 없어 두 번 불러도 된다.
        """
        want = getattr(self.on_decision, "wants_epoch", None)
        if want is None:
            return True
        elig = self.market.newly_eligible(
            mbt, t, orders=self.orders, records=self.records, epoch_s=self.grid_s)
        return bool(elig) and want(len(elig))

    # ------------------------------------------------------------------ ③ 확정
    def _confirm(self, mbt, t: float, res) -> tuple[str, ...]:
        """동의된 거래를 엔진에 밀어 넣는다. 실패하면 그 건만 KEEP 으로 되돌린다."""
        ok_keys = []
        for tr in res.resolve.trades:
            rec = self.records[tr.doc_key]
            if tr.kind == SPACE:
                ok = mbt.try_pre_gate_transfer(
                    tr.doc_key, tr.dst_block,
                    travel_s=self.layout.gate_to_block_s(tr.dst_block),
                    route_delta_s=tr.route_delta_s)
                if ok:
                    self.orders[tr.doc_key] = relocate(self.orders[tr.doc_key],
                                                       tr.dst_block)
                    self.n_space += 1
            else:
                ok = try_time_sell(mbt, tr.doc_key, delta_s=tr.defer_s,
                                   max_deferrals=MAX_ENTRY_DEFERRALS, t=t)
                if ok:
                    # ★오더의 예약 시각도 옮긴다. 안 옮기면 "곧 올 통지분"
                    #   특징이 오지 않을 트럭을 계속 세어 정책이 헛것을 본다.
                    o = self.orders[tr.doc_key]
                    self.orders[tr.doc_key] = defer(
                        o, o.in_out_reserve_s + tr.defer_s)
                    self.n_time += 1
            if ok:
                ok_keys.append(tr.doc_key)
                self.traded_edges += 1
            else:
                # ★물리가 거부했다 — 시장의 기록을 되돌린다(있지도 않은 거래를 남기지 않는다)
                self.txn_failed += 1
                rec.prev_con_loc = None
                rec.con_swap_reason = RESOLVER_KEEP
            self.ledger.append({
                "t": t, "doc_key": tr.doc_key, "kind": tr.kind,
                "dst": tr.dst_block, "slot": tr.slot,
                "defer_s": tr.defer_s, "ok": bool(ok)})
        return tuple(ok_keys)

    # ------------------------------------------------------------------ 교사 훅
    def _note_decisions(self, mbt, t: float, res, *, applied, pre=None) -> None:
        """이 epoch 의 결정을 교사에게 알린다. 훅이 없으면 아무 일도 안 한다.

        **판정 실행에서는 훅이 None 이라 반사실이 한 번도 안 돌아간다**
        (하드가드 `rollout_calls_during_eval == 0`).
        """
        if self.on_decision is None:
            return
        buyer_by_key = {e["doc_key"]: e for e in self.market.buyer.trail}
        for e in self.market.seller.trail:
            if e.get("_noted"):
                continue
            e["_noted"] = True
            self.on_decision(mbt, t, seller_entry=e,
                             buyer_entry=buyer_by_key.get(e["doc_key"]),
                             applied=applied, pre=pre)
