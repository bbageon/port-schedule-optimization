"""30일을 **한 시뮬로** 굴린다 — 하루가 끝날 때마다 평가한다 ([[YR-239]]).

사용자 지시 2026-08-26:
  *"30일치 시드 데이터를 만들어놓고 30일치를 학습할꺼야 … 첫날과 마지막날은
    사용하지않고 세계를 연결하는 용도로만 사용해서 나머지 28일을 학습해봐.
    단 학습 중간보고는 28일중에서 하루가 끝날때마다 평가를 하는거지"*

■ 하루 무대와 무엇이 다른가

| | 하루 무대 (`episode.run_episode`) | **30일 무대** |
|---|---|---|
| 시작 야드 | 날마다 인공 초기 적재 | **첫날만** — 그 뒤는 전날이 남긴 것 |
| 창 끝 | 도착 0 인 배수 2시간 | 없다 — 다음 날 새벽이 이어진다 |
| 본선 | t=0 에 전부 박아 둔다 | ★**그날 아침에 붙인다** (`month_engine`) |
| Φ | 하루치 한 개 | **날마다 한 개** (28일 + 연결용 2일) |
| 계수기 | 창 끝 한 번 | 날 경계마다 찍어 **차분** |

■ 하루의 Φ 를 어떻게 가르나
  · **항1 (트럭 대기)** — 트럭은 **게이트를 들어온 날**에 속한다. 그 날 트럭만
    모아 턴타임을 센다. 자정을 넘겨 끝나도 그 날 몫이다(세계가 이어지므로 실제로
    끝날 때까지 센다 — 하루 무대의 인공 검열이 사라진다).
  · **항2·3·4 (YC 빈주행·재조작·본선 유휴)** — 누적 계수기라 시각이 없다.
    날 경계에서 찍어 **오늘 값 − 어제 값**을 쓴다.

■ 잠정과 확정
  하루가 끝나는 순간에는 **그 날 트럭이 아직 안 나갔을 수 있다.** 그래서
    · 중간보고(그 자리)   = **잠정** — 그 시각에 검열한 값
    · 달이 끝난 뒤        = **확정** — 끝까지 기다린 값 ← 판정은 이쪽
  둘 다 남긴다. 잠정이 확정보다 크면(검열이 물린 것) 그만큼 미완료가 있었다는 뜻이다.
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field

import torch

from .. import CF_HORIZON_S
from ..actors import BuyerNet, SellerNet
from ..reward.phi import terminal_cost_krw
from ..world.integrated.multiblock import TransferError
from ..world.integrated.profiles import build_h21_profile
from ..world.integrated.terminal_stream import (DIURNAL_DRAIN_S, OBS_24H,
                                                ensure_time_ledger)
from ..world.integrated.yard_layout import terminal_layout
from .episode import (ARMS, DISPATCHERS_READY, _Ctx, _sf_spt_policy, _sim_from,
                      rehandles_of, yc_empty_travel_s)
from .month import (DAY_S, N_DAYS, build_month, ledger_load, make_retarget,
                    month_vessel_idle, plan_month, plan_month_vessels,
                    prune_completed, retire_done_vessels, truck_net_by_block,
                    vessel_meta)
from .branchpool import BranchJob, BranchPool, default_workers
from .month_engine import MonthTerminal, inject_vessel
from .orders import EPOCH_S, V3Announcer, orders_from_schedule
from .rollout import RolloutBudget

#: 계수기를 얼마나 자주 찍나. 날 경계는 **항상** 따로 찍는다.
#: 60초마다 찍으면 30일 = 43,200번 × 21블록이라 그 자체가 벽이 된다.
SNAP_S = 900.0


@dataclass
class DayReport:
    """하루치 결과 한 줄 — 중간보고가 이걸 그대로 읽는다."""

    index: int
    load: int
    label: str
    train: bool
    phi_krw: float = 0.0
    c_wait: float = 0.0
    c_move: float = 0.0
    c_rehandle: float = 0.0
    c_vessel: float = 0.0
    n_trucks: int = 0
    n_censored: int = 0
    mean_turn_time_s: float = 0.0
    p90_turn_time_s: float = 0.0
    over_ratio: float = 0.0
    traded: int = 0
    n_space: int = 0
    n_time: int = 0
    decisions: int = 0
    vessels: int = 0
    vessel_moves: int = 0
    vessel_skipped: int = 0
    #: 그날 **못 들어온 트럭** — 조용히 사라지면 부하가 저절로 줄어든다
    truck_skipped: int = 0
    #: 그날 반출 대상을 **다시 고른** 트럭 (30일이면 이름이 겹친다 — YR-239)
    retargeted: int = 0
    pruned: dict = field(default_factory=dict)
    load_after: dict = field(default_factory=dict)
    provisional: bool = True
    # -- 학습 (교사를 붙였을 때만 채워진다)
    explore: float = 0.0
    n_labels: int = 0
    worlds: int = 0
    fit: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class MonthResult:
    days: list = field(default_factory=list)          # DayReport (확정)
    live: list = field(default_factory=list)          # DayReport (잠정 — 중간보고 그대로)
    plan: list = field(default_factory=list)
    admitted: int = 0
    skipped: int = 0
    traded_edges: int = 0
    n_space: int = 0
    n_time: int = 0
    txn_failed: int = 0
    decisions: int = 0
    policy_exceptions: int = 0
    retargeted: int = 0
    vessel_admissions: list = field(default_factory=list)

    @property
    def train_days(self) -> list:
        return [d for d in self.days if d.train]

    def by_load(self) -> dict:
        """★부하별 판정용 — 학습에 쓴 날만 부하로 묶는다 (사용자 지시 2026-08-26)."""
        out: dict[int, list] = {}
        for d in self.train_days:
            out.setdefault(d.load, []).append(d)
        return dict(sorted(out.items()))


class _MonthTape:
    """날 경계의 **계수기 사진첩**. 하루치 항2·3·4 는 여기 차분에서 나온다."""

    def __init__(self, meta: dict, archive: dict):
        self.meta, self.archive = meta, archive
        self.at: dict[float, tuple[dict, float, int]] = {}

    def snap(self, mbt, t: float) -> None:
        self.at[round(t, 6)] = (month_vessel_idle(mbt, self.meta, self.archive),
                                yc_empty_travel_s(mbt), rehandles_of(mbt))

    def read(self, t: float) -> tuple[dict, float, int]:
        key = round(t, 6)
        if key in self.at:
            return self.at[key]
        keys = [k for k in self.at if k <= key]
        return self.at[max(keys)] if keys else ({}, 0.0, 0)

    def diff(self, t0: float, t1: float) -> tuple[dict, float, int]:
        """`t0 → t1` 사이에 **늘어난 만큼**. 이게 그 하루의 몫이다."""
        v0, yc0, rh0 = self.read(t0)
        v1, yc1, rh1 = self.read(t1)
        v = {}
        for k, (gt, idle) in v1.items():
            v[k] = (gt, max(0.0, idle - (v0.get(k, (gt, 0.0))[1])))
        return v, max(0.0, yc1 - yc0), max(0, rh1 - rh0)


class _MonthCtx(_Ctx):
    """분기 세계도 배를 **스트림 이름**으로 묶는다.

    하루 무대의 `_Ctx.vessel_idle` 은 블록으로 묶는데, 30일이면 한 블록에 날마다
    다른 배가 오므로 어제 배의 유휴가 오늘 배에 붙고 선급(GT·STS)도 어긋난다.
    """

    #: 스트림 이름 -> (배 이름, GT, STS). 무대가 채운다.
    vessel_meta: dict = {}

    def vessel_idle(self, mbt, end_s: float) -> dict:
        return month_vessel_idle(mbt, self.vessel_meta)


def _day_records(records: dict, day: int) -> dict:
    """그 날 **게이트를 들어온** 트럭의 기록만. `docKey` 에 날짜가 박혀 있다."""
    pre = f"D{day:02d}-"
    return {k: v for k, v in records.items() if k.startswith(pre)}


def _phi_of_day(records, day: int, *, end_s: float, tape: _MonthTape,
                t0: float, t1: float):
    v, yc, rh = tape.diff(t0, t1)
    return terminal_cost_krw(_day_records(records, day), end_s=end_s,
                             vessel_idle=v, yc_extra_move_s=yc, rehandles=rh)


def run_month(*, seed: int, arm: str = "RL", seller_net=None, buyer_net=None,
              n_days: int = N_DAYS, dispatcher: str = "SF_SPT",
              lead_mode: str = "DIST", window_s: float = 1800.0,
              explore: float = 0.0, horizon_s: float = CF_HORIZON_S,
              slot_mode: str = "HORIZON", trigger_top_k: float | None = None,
              days=None, on_day=None, labels_per_day: int | None = None,
              workers: int = 1, explore_of_day=None, on_fit=None,
              branch_days: int = 1) -> MonthResult:
    """30일을 한 번에 굴린다. `on_day(DayReport)` 가 **중간보고** 훅이다.

    ■ 교사를 붙이면 (`labels_per_day`) **하루가 곧 한 회차**가 된다
      하루를 굴리며 반사실 라벨을 K건 모으고, 하루가 끝나면 `on_fit(day, rows)` 로
      학생 망을 갱신한다. **갱신된 망으로 다음 날을 이어서 굴린다** — 세계가 안
      끊기므로 어제의 야드 위에서 오늘 정책이 판단한다.

      · 표본 예산·작업자 풀은 **날마다 새로** 만든다 — 작업자에는 망이 복사돼
        가므로, 학습으로 망이 바뀌면 풀도 새로 띄워야 옛 정책이 안 남는다.
      · ε(탐색)은 `explore_of_day(day)` 로 날마다 정한다.

    `branch_days` — 분기 세계에 넘길 기록을 **며칠치**로 자를까 (기본 1 = 앞뒤 하루씩
    더해 사흘). 20만 건을 통째로 복제하면 스냅샷 하나가 몇 초가 된다.
    """
    if arm not in ARMS:
        raise NotImplementedError(f"알 수 없는 재배치 팔 {arm!r} — 쓸 수 있는 팔: {ARMS}")
    if dispatcher not in DISPATCHERS_READY:
        raise NotImplementedError(f"배차 {dispatcher!r} 은 아직 구현 전이다 (YR-213)")

    prof, layout = build_h21_profile(), terminal_layout()
    days = list(days) if days else plan_month(seed, n_days=n_days)
    n_days = len(days)
    built = build_month(seed, days=days, profile=prof, layout=layout,
                        lead_mode=lead_mode)
    orders, records = orders_from_schedule(built)
    # ★본선 양하/적하를 **트럭 수지에 맞춘다** — 안 그러면 야드가 30일 동안 빈다.
    v_by_day = plan_month_vessels(days, layout, obs=OBS_24H,
                                  truck_net=truck_net_by_block(built["schedule"]))
    meta = {}
    for rows in v_by_day.values():
        meta.update(vessel_meta(rows))

    month_s = n_days * DAY_S
    sim_end = month_s + DIURNAL_DRAIN_S
    # ★배경은 **첫날 것만** 쓴다 — 초기 적재가 곧 출발 야드다.
    #   본선(`vessels`)과 그 야드작업(`jobs`)은 **넣지 않는다**: 그날 아침에 붙인다.
    scns = {b: dataclasses.replace(s, jobs=[], vessels=[], horizon_s=month_s,
                                   drain_window_s=DIURNAL_DRAIN_S)
            for b, s in built["day0"]["scenarios"].items()}
    mbt = MonthTerminal({b: ensure_time_ledger(_sim_from(s, prof))
                         for b, s in scns.items()},
                        extra_review_epochs=tuple(
                            i * EPOCH_S for i in range(int(month_s // EPOCH_S) + 1)))
    # ★반출 대상은 **투입 시각에** 다시 고른다 — 30일은 이름이 날마다 겹친다.
    ann = V3Announcer(built["schedule"], end_s=sim_end,
                      retarget=make_retarget(seed))

    if seller_net is None or buyer_net is None:
        torch.manual_seed(int(seed))
    ctx = _MonthCtx(seller_net=(seller_net if seller_net is not None else SellerNet()),
               buyer_net=(buyer_net if buyer_net is not None else BuyerNet()),
               layout=layout, announcer=ann, arm=arm, grid_s=EPOCH_S,
               window_s=window_s, explore=explore, seed=seed,
               episode_end_s=month_s,
               cf_horizon_s=(horizon_s if slot_mode == "HORIZON" else None))
    ctx.vessel_meta = meta
    if trigger_top_k is not None:
        ctx.trigger_top_k = float(trigger_top_k)
    market = ctx.make_market(mbt)

    # -- 교사 배선 — `labels_per_day` 가 없으면 훅이 통째로 None 이라
    #    반사실이 **한 번도** 안 돌아간다 (판정 하드가드가 이걸 검사한다).
    teach: dict = {"budget": None, "pool": None, "meta": {}}

    def on_decision(m, t, *, seller_entry, buyer_entry, applied, pre=None):
        b, pool = teach["budget"], teach["pool"]
        if pool is None or not b.take() or pre is None:
            return
        dk = seller_entry["doc_key"]
        s_alt = "KEEP" if seller_entry["action"] != "KEEP" else "SELL"
        b_alt = (None if buyer_entry is None else
                 ("REJECT" if buyer_entry["action"] == "BUY" else "BUY"))
        teach["meta"][(round(t, 6), dk)] = {
            "seller": seller_entry, "buyer": buyer_entry,
            "seller_alt": s_alt, "buyer_alt": b_alt}
        pool.submit(BranchJob(doc_key=dk, t=t, mbt=pre["mbt"],
                              orders=pre["orders"], records=pre["records"],
                              decided=pre["decided"],
                              seller_alt=s_alt, buyer_alt=b_alt))

    def wants_epoch(n_elig: int) -> bool:
        b = teach["budget"]
        return b is not None and b.peek(n_elig)

    on_decision.wants_epoch = wants_epoch
    bridge = ctx.make_bridge(market, orders=orders, records=records,
                             on_decision=(on_decision if labels_per_day else None))

    def _near_records(t: float) -> dict:
        """분기 세계에 넘길 기록 — **그 앞뒤 며칠치**만.

        H=3시간 안에 얽히는 트럭은 오늘과 (자정 근처면) 내일뿐이고, 어제 들어와
        아직 안 나간 트럭도 있다. 그 밖은 사실·대안 양쪽에서 똑같이 빠져 상쇄된다.
        """
        d = int(t // DAY_S)
        pre = tuple(f"D{i:02d}-" for i in range(max(0, d - branch_days),
                                                min(n_days, d + branch_days + 1)))
        return {k: v for k, v in records.items() if k.startswith(pre)}

    if labels_per_day:
        bridge.branch_records = _near_records

    res = MonthResult(plan=[d.as_dict() for d in days])
    archive: dict[str, float] = {}
    tape = _MonthTape(meta, archive)
    state = {"day": 0, "snap": 0.0, "traded": 0, "space": 0, "time": 0,
             "dec": 0, "skip": 0, "retgt": 0}

    def open_teacher(d) -> float:
        """그날의 표본 예산·작업자 풀·ε 를 새로 연다. 교사가 없으면 ε 만 정한다."""
        eps = explore if explore_of_day is None else float(explore_of_day(d))
        for who in (market.seller, market.buyer):
            if hasattr(who, "explore"):
                who.explore = eps
        if not labels_per_day:
            return eps
        b = RolloutBudget(max_labels=labels_per_day, identity_checks=0)
        if b.stride <= 0:                    # 하루에 고르게 흩는다
            b.stride = max(1, d.load // max(1, labels_per_day))
        # ★작업자에게는 **그날 명단만** 보낸다 — 30일치(20만 건)를 통째로 절이면
        #   작업자 하나당 수십 MB 다. 분기는 `[t, t+H]` 밖을 볼 일이 없다.
        pool_ctx = copy.copy(ctx)            # 얕은 복사 — 망·레이아웃은 공유
        pool_ctx.announcer = ann.window(d.t0 - EPOCH_S, d.t1 + horizon_s)
        pool = BranchPool(pool_ctx, horizon_s=horizon_s,
                          workers=(default_workers() if workers < 0 else workers))
        pool.__enter__()
        teach.update(budget=b, pool=pool, meta={})
        return eps

    def close_teacher(d, rep) -> None:
        """그날 라벨을 거둬 학생을 갱신한다 — **다음 날은 새 정책으로 간다**."""
        pool = teach["pool"]
        if pool is None:
            return
        rows = []
        for r in pool.results():
            mt = teach["meta"].get((round(r["t"], 6), r["doc_key"]))
            if mt is None:
                continue
            row = {"doc_key": r["doc_key"], "t": r["t"], "horizon_s": horizon_s,
                   "worlds": r["worlds"], "seller": mt["seller"],
                   "seller_alt": mt["seller_alt"],
                   "seller_alt_coord": r["seller_alt_coord"],
                   "phi_factual": r["phi_factual"],
                   "phi_seller_alt": r["phi_seller_alt"]}
            if "phi_buyer_alt" in r:
                row["buyer"] = mt["buyer"]
                row["buyer_alt"] = mt["buyer_alt"]
                row["phi_buyer_alt"] = r["phi_buyer_alt"]
            rows.append(row)
        rep.n_labels = len(rows)
        rep.worlds = pool.n_worlds
        pool.close()
        teach.update(pool=None, budget=None, meta={})
        if on_fit is not None:
            rep.fit = dict(on_fit(d, rows) or {})

    def open_day(m, d) -> tuple[int, int, int]:
        """그날 배를 붙인다. 못 붙인 배는 조용히 넘기지 않고 세어 둔다."""
        n = moves = skipped = 0
        for r in v_by_day.get(d.index, []):
            try:
                a = inject_vessel(m, r["block"], r, key=r["key"],
                                  size_seed=f"v3:month:{seed}:{r['key']}")
            except TransferError as ex:
                skipped += 1
                res.vessel_admissions.append({"key": r["key"], "day": d.index,
                                              "ok": False, "why": str(ex)})
                continue
            n += 1
            moves += a.moves
            res.vessel_admissions.append({"key": a.vessel_key, "day": d.index,
                                          "ok": True, "moves": a.moves,
                                          "asked": a.asked_moves,
                                          "why": a.reason})
        return n, moves, skipped

    def close_day(m, d, t: float, opened: tuple) -> None:
        """하루를 닫는다 — 잠정 Φ 를 내고, 끝난 것을 치운다."""
        bridge._sync(m, t)                    # 마지막 epoch 뒤 완료분까지 흡수
        phi = _phi_of_day(records, d.index, end_s=t, tape=tape, t0=d.t0, t1=t)
        rep = DayReport(index=d.index, load=d.load, label=d.label, train=d.is_train,
                        phi_krw=phi.total, c_wait=phi.wait, c_move=phi.move,
                        c_rehandle=phi.rehandle, c_vessel=phi.vessel,
                        n_trucks=phi.n_trucks, n_censored=phi.n_censored,
                        mean_turn_time_s=phi.mean_turn_time_s,
                        p90_turn_time_s=phi.p90_turn_time_s,
                        over_ratio=phi.over_ratio,
                        traded=bridge.traded_edges - state["traded"],
                        n_space=bridge.n_space - state["space"],
                        n_time=bridge.n_time - state["time"],
                        decisions=len(market.seller.trail) - state["dec"],
                        vessels=opened[0], vessel_moves=opened[1],
                        vessel_skipped=opened[2], provisional=True,
                        truck_skipped=ann.n_skipped - state["skip"],
                        retargeted=ann.n_retargeted - state["retgt"])
        state.update(traded=bridge.traded_edges, space=bridge.n_space,
                     time=bridge.n_time, dec=len(market.seller.trail),
                     skip=ann.n_skipped, retgt=ann.n_retargeted)
        rep.explore = state.get("eps", 0.0)
        close_teacher(d, rep)                 # 라벨 거두기 -> 학생 갱신
        # ★순서가 계약이다 — **먼저 job 을 치우고** 그 다음 배를 치운다.
        #   배 앞으로 남은 job 이 있는데 배를 치우면 그 job 이 영원히 안 풀린다
        #   (`retire_done_vessels` 머리말 참조 · 2026-08-26 실측 사고).
        rep.pruned = prune_completed(m, t)
        retire_done_vessels(m, archive, t=t)
        rep.load_after = ledger_load(m)
        res.live.append(rep)
        if on_day is not None:
            on_day(rep)

    opened = {"cur": (0, 0, 0)}

    def review(m, t: float) -> None:
        ann.review(m, t)
        bridge.review(m, t)
        if t >= state["snap"]:
            tape.snap(m, t)
            state["snap"] = t + SNAP_S
        while state["day"] < n_days and t >= days[state["day"]].t0 - 1e-9:
            d = days[state["day"]]
            tape.snap(m, d.t0)                # 경계는 **정확히** 찍는다
            if state["day"] > 0:
                close_day(m, days[state["day"] - 1], d.t0, opened["cur"])
            state["eps"] = open_teacher(d)
            opened["cur"] = open_day(m, d)
            state["day"] += 1
            state["snap"] = t + SNAP_S

    exec_policy, exc = _sf_spt_policy()
    mbt.run(exec_policy, review_fn=review)
    tape.snap(mbt, month_s)
    bridge._sync(mbt, month_s)
    close_day(mbt, days[-1], month_s, opened["cur"])

    # ★배수 구간(마지막 날 뒤 2시간)에 끝난 트럭까지 흡수한다. 이걸 안 하면
    #   그 트럭들이 게이트를 나간 기록이 없어 **검열된 것처럼** 계산된다.
    bridge._sync(mbt, sim_end)

    # ── 확정 — 끝까지 기다린 값으로 다시 낸다 (판정은 이쪽)
    for d in days:
        phi = _phi_of_day(records, d.index, end_s=sim_end, tape=tape,
                          t0=d.t0, t1=min(d.t1, month_s))
        live = res.live[d.index]
        fin = dataclasses.replace(
            live, phi_krw=phi.total, c_wait=phi.wait, c_move=phi.move,
            c_rehandle=phi.rehandle, c_vessel=phi.vessel,
            n_trucks=phi.n_trucks, n_censored=phi.n_censored,
            mean_turn_time_s=phi.mean_turn_time_s,
            p90_turn_time_s=phi.p90_turn_time_s, over_ratio=phi.over_ratio,
            provisional=False)
        res.days.append(fin)

    res.admitted = ann.n_admitted
    res.skipped = ann.n_skipped
    res.traded_edges = bridge.traded_edges
    res.n_space, res.n_time = bridge.n_space, bridge.n_time
    res.txn_failed = bridge.txn_failed
    res.decisions = len(market.seller.trail)
    res.policy_exceptions = exc["n"]
    res.retargeted = ann.n_retargeted
    return res
