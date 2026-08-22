"""에피소드 하나 — 무대를 세우고 하루를 굴려 Φ 를 낸다.

설계 정본: `.claude/docs/architecture/06-학습과-판정.md` · `06b-대조군.md`

■ `run_episode` 가 곧 `eval/run.py` 의 `episode_fn` 이다
  판정 셀 하나 = (부하 × 배차 × 재배치 팔 × 날). 여기서 그 한 셀을 돌린다.

■ 재배치 팔 (06b)
  | 팔 | 지금 |
  |---|---|
  | `NO_REALLOC` | ✅ 시장을 아예 안 연다 (기준선) |
  | `RL` | ✅ Seller·Buyer 학습 망 |
  | `FCFS`·`SPT`·`LEAST_SLACK`·`NEAREST`·`NETGAIN` | ⏳ [[YR-211]] |
  고전 규칙 팔은 **조용히 대체하지 않고 즉시 실패**한다 — 안 만든 팔이 다른 팔
  행세를 하면 대조표 전체가 거짓이 된다.

■ 교사는 학습에서만 붙는다
  `teacher=None`(기본)이면 반사실이 **한 번도** 안 돌아간다. 판정 하드가드
  `rollout_calls_during_eval == 0` 이 이걸 검사한다.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch

from ..world.domain.enums import InformationLevel
from ..world.integrated.baselines import (ResolverPolicy,
                                          ServiceFirstSPTPreference, _apply,
                                          _wait_of)
from ..world.integrated.candidates import CandidateGenerator
from ..world.integrated.engine import TerminalSimulator
from ..world.integrated.multiblock import MultiBlockTerminal
from ..world.integrated.policy_config import LEGACY_DEFAULT
from ..world.integrated.profiles import build_h21_profile
from ..world.integrated.terminal_stream import (OBS_24H, admission_epochs,
                                                ensure_time_ledger)
from ..world.integrated.vessel import VESSEL_CLASSES
from ..world.integrated.yard_layout import terminal_layout

#: 정책에 공개되는 차량 정보 시점 — 사전 반출입정보 + 제공 ETA.
#: v2 는 `experiments/yr088_joint_rl.LEVEL` 을 빌려 썼다. v3 는 자기 것을 갖는다.
INFO_LEVEL = InformationLevel.PRE_ADVICE


def _sim_from(scn, profile):
    """시나리오 → 시뮬레이터. **시나리오를 만든 프로파일과 같은 것**을 넘긴다.

    v2 의 `experiments/yr149_load_cells._sim_from` 과 같은 일을 하지만, v3 는
    실험 스크립트를 임포트하지 않는다(사용자 지시 2026-08-22).
    """
    sim = TerminalSimulator(profile, scn, check_invariants=True)
    sim.info_level = INFO_LEVEL
    return sim
from ..actors import Buyer, BuyerNet, Market, Resolver, Seller, SellerNet
from ..reward.phi import terminal_cost_krw
from .bridge import MarketBridge
from .orders import EPOCH_S, V3Announcer, build_stage, orders_from_schedule
from .rollout import RolloutBudget, SnapshotRollout

#: 구현된 재배치 팔. 나머지는 [[YR-211]].
ARMS = ("NO_REALLOC", "RL")
TODO_ARMS = ("FCFS", "SPT", "LEAST_SLACK", "NEAREST", "NETGAIN")

#: 배차 축 — 지금은 규칙 `SF-SPT` 만. 계획법·얼린 망은 [[YR-213]].
DISPATCHERS_READY = ("SF_SPT",)


@dataclass
class EpisodeResult:
    phi_krw: float = 0.0
    breakdown: dict = field(default_factory=dict)
    admitted: int = 0
    traded_edges: int = 0
    policy_exceptions: int = 0
    n_space: int = 0
    n_time: int = 0
    txn_failed: int = 0
    labels: list = field(default_factory=list)
    identity: list = field(default_factory=list)
    rollout_worlds: int = 0
    decisions: int = 0
    missed_labels: int = 0

    def as_dict(self) -> dict:
        d = {"phi_krw": self.phi_krw, "admitted": self.admitted,
             "traded_edges": self.traded_edges,
             "policy_exceptions": self.policy_exceptions,
             "n_space": self.n_space, "n_time": self.n_time,
             "txn_failed": self.txn_failed, "n_decisions": self.decisions,
             "n_labels": len(self.labels), "rollout_worlds": self.rollout_worlds,
             "missed_labels": self.missed_labels}
        d.update(self.breakdown)
        return d


class _Ctx:
    """분기 세계를 다시 조립하는 재료. 망은 **공유**하고 나머지만 새로 만든다."""

    def __init__(self, *, seller_net, buyer_net, layout, announcer, arm, grid_s,
                 window_s, explore, seed, episode_end_s):
        self.seller_net, self.buyer_net = seller_net, buyer_net
        self.layout, self.announcer = layout, announcer
        self.arm, self.grid_s, self.window_s = arm, grid_s, window_s
        self.explore, self.seed = explore, seed
        #: ★하루의 끝. 분기 세계도 **이 값을 그대로** 받는다 — rollout 지평(t+H)은
        #: 시뮬레이션을 어디서 멈출지만 정하고, 선택지 집합과 시계 특징은 하루 기준이다.
        #: 여기에 t+H 를 주면 분기 세계의 Seller 가 먼 이연 칸을 못 보고 다른 좌표를
        #: 고른다(2026-08-22 불변식이 잡았다).
        self.episode_end_s = float(episode_end_s)

    def make_market(self, mbt, *, decided=()):
        rng = random.Random(self.seed)
        seller = Seller(self.seller_net, self.layout, explore=self.explore, rng=rng)
        buyer = Buyer(self.buyer_net, explore=self.explore, rng=rng)
        m = Market(seller, buyer, Resolver(mbt), window_s=self.window_s)
        m.decided = set(decided)
        return m

    def make_bridge(self, market, *, orders, records, on_decision):
        return MarketBridge(market, self.layout, orders=orders, records=records,
                            end_s=self.episode_end_s, arm=self.arm,
                            grid_s=self.grid_s, on_decision=on_decision)

    def make_exec_policy(self):
        return _sf_spt_policy()[0]

    @staticmethod
    def vessel_idle(mbt, end_s: float) -> dict:
        return vessel_idle_of(mbt, end_s)

    @staticmethod
    def yc_extra_move_s(mbt) -> float:
        return yc_empty_travel_s(mbt)

    @staticmethod
    def rehandles(mbt) -> int:
        return rehandles_of(mbt)


def _sf_spt_policy():
    """규칙 배차 `SF-SPT` — 실행 정책과 예외 계수기를 함께 돌려준다."""
    gens: dict[int, CandidateGenerator] = {}
    exc = {"n": 0}
    pol = ResolverPolicy(ServiceFirstSPTPreference(), "SF")

    def exec_policy(sim, dp):
        g = gens.setdefault(id(sim), CandidateGenerator(config=LEGACY_DEFAULT))
        gb = {c: g.generate(sim, c, INFO_LEVEL) for c in dp.crane_ids}
        try:
            _apply(sim, pol.decide(sim, dp, gb))
        except Exception:
            exc["n"] += 1
            _apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})
    return exec_policy, exc


def yc_empty_travel_s(mbt) -> float:
    """항 2 의 원료 — YC **빈 갠트리** 주행 시간(초).

    적재 주행은 생산 사이클이라 세지 않는다(넣으면 "아무것도 안 하기" 가 최적).
    빈 주행이 곧 "재배치·재조작 때문에 생긴 추가 이동" 이다.
    """
    tot = 0.0
    for sim in mbt.blocks.values():
        k = getattr(sim, "kpis", None)
        if k is None:
            continue
        v = sim.profile.cranes[0].gantry_speed_mps if sim.profile.cranes else 1.0
        tot += float(getattr(k, "empty_gantry_m", 0.0)) / max(1e-9, v)
    return tot


def rehandles_of(mbt) -> int:
    """항 3 의 원료 — 터미널 전체 재조작 횟수(선재조작 포함)."""
    return sum(int(getattr(getattr(sim, "kpis", None), "rehandle_count", 0))
               for sim in mbt.blocks.values())


def vessel_idle_of(mbt, end_s: float) -> dict:
    """항 4 의 원료 — {선박: (GT, 유휴 초)}.

    ⚠️ **잠정**: 무대가 아직 선급을 배정하지 않는다([[YR-212]]). 지금은 기항 물량으로
    선급을 되짚는다 — 물량 범위가 선급을 정하므로(02b §1) 역추정이 성립한다.
    YR-212 가 선급을 무대에 박으면 이 되짚기는 지운다.
    """
    out = {}
    for bid, sim in sorted(mbt.blocks.items()):
        for vid, v in sorted(getattr(sim, "vessels", {}).items()):
            moves = int(v.plan.total_moves)
            cls = min(VESSEL_CLASSES,
                      key=lambda c: abs(sum(c.moves_range()) / 2.0 - moves))
            idle = float(getattr(v, "sts_wait_accum_s", 0.0))
            out[f"{bid}:{vid}"] = (float(cls.gt), idle)
    return out


class _CounterTape:
    """★엔진 계수기를 **epoch 마다 찍어 둔다.**

    항2(YC 빈 주행)·항4(본선 유휴)는 기록이 아니라 **엔진 누적 계수기**라, 하루가
    끝난 뒤에는 "그때 얼마였나" 를 되짚을 수 없다. 트럭 대기(항1·항3)는 기록에
    시각이 남아 `end_s` 검열로 잘리지만 계수기는 그렇게 못 자른다.

    분기 세계는 t+H 에서 멈추므로 계수기가 정확히 그 시점 값이다. 실제 궤적에서
    같은 시점 값을 읽으려면 **지나갈 때 찍어 두는 수밖에 없다.** 이걸 안 하면
    동일성 불변식이 항상 어긋난다(창 길이가 다른 값을 비교하게 된다).
    """

    def __init__(self):
        self.at: dict[float, tuple[dict, float, int]] = {}

    def snap(self, mbt, t: float) -> None:
        self.at[round(t, 6)] = (vessel_idle_of(mbt, t), yc_empty_travel_s(mbt),
                                rehandles_of(mbt))

    def read(self, t: float) -> tuple[dict, float, int]:
        """`t` 이하의 가장 늦은 기록. 창 끝이 격자 밖이면 직전 격자를 쓴다."""
        key = round(t, 6)
        if key in self.at:
            return self.at[key]
        keys = [k for k in self.at if k <= key]
        return self.at[max(keys)] if keys else ({}, 0.0, 0)


def run_episode(*, load: int, dispatcher: str = "SF_SPT", arm: str = "RL",
                seed: int, seller_net=None, buyer_net=None,
                lead_mode: str = "DIST", window_s: float = 1800.0,
                explore: float = 0.0, horizon_s: float = 3600.0,
                budget: RolloutBudget | None = None,
                obs=None) -> EpisodeResult:
    """한 셀을 돌린다. `budget` 이 있으면 교사가 라벨을 만든다(학습용)."""
    if arm not in ARMS:
        raise NotImplementedError(
            f"재배치 팔 {arm!r} 은 아직 구현 전이다 — {TODO_ARMS} 는 YR-211. "
            f"지금 쓸 수 있는 팔: {ARMS}")
    if dispatcher not in DISPATCHERS_READY:
        raise NotImplementedError(
            f"배차 {dispatcher!r} 은 아직 구현 전이다 — 계획법·얼린 망은 YR-213. "
            f"지금 쓸 수 있는 배차: {DISPATCHERS_READY}")

    obs = obs or OBS_24H
    prof, layout = build_h21_profile(), terminal_layout()
    built = build_stage(load=load, seed=seed, profile=prof, layout=layout,
                        obs=obs, lead_mode=lead_mode)
    orders, records = orders_from_schedule(built)

    mbt = MultiBlockTerminal(
        {b: ensure_time_ledger(_sim_from(s, prof)) for b, s in built["scenarios"].items()},
        extra_review_epochs=admission_epochs(obs))
    ann = V3Announcer(built["schedule"], end_s=built["sim_end_s"])

    # ★망 초기화에 시드를 건다. 안 걸면 학습 전 팔이 **실행마다 다른 정책**이 되어
    #   같은 시드로 돌려도 Φ 가 재현되지 않는다(2026-08-22 실측: 같은 셀이
    #   102,913,359 ↔ 136,388,850 로 갈렸다). 짝비교의 전제가 깨진다.
    if seller_net is None or buyer_net is None:
        torch.manual_seed(int(seed))
    s_net = seller_net if seller_net is not None else SellerNet()
    b_net = buyer_net if buyer_net is not None else BuyerNet()
    ctx = _Ctx(seller_net=s_net, buyer_net=b_net, layout=layout, announcer=ann,
               arm=arm, grid_s=EPOCH_S, window_s=window_s, explore=explore,
               seed=seed, episode_end_s=obs.observe_s)
    market = ctx.make_market(mbt)

    res = EpisodeResult()
    pending: list[dict] = []
    tape = _CounterTape()
    roll = SnapshotRollout(ctx, horizon_s=horizon_s) if budget else None
    if budget is not None and budget.stride <= 0:
        # 하루에 고르게 흩는다 — 앞부분만 뽑으면 새벽 표본만 남는다.
        budget.stride = max(1, load // max(1, budget.max_labels))

    def on_decision(m, t, *, seller_entry, buyer_entry, applied, pre=None):
        # ★계수기는 **모든 결정**에서 돈다. 라벨을 만든 것만 세면 `stride` 가
        #   영원히 다시 안 맞아 첫 한 건만 뽑힌다(2026-08-22 실측: 3,018 중 1건).
        if roll is None:
            return
        want = budget.take()
        if not want:
            return
        if pre is None:
            res.missed_labels += 1        # peek 이 헛짚었다 — 있으면 안 되는 일
            return
        dk = seller_entry["doc_key"]
        # ★분기는 **결정 전 스냅샷**에서 출발한다 — `m` 은 이미 거래가 확정된 뒤다.
        def br(**kw):
            return roll.branch(pre["mbt"], t, orders=pre["orders"],
                               records=pre["records"], decided=pre["decided"],
                               doc_key=dk, **kw)

        # ★세계 셋 — factual 도 굴린다. 절단(t+H 에서 세상이 끝남)을 셋이 똑같이
        #   겪어야 차이에서 절단분이 상쇄된다 (실측·근거는 rollout.py 머리).
        fact = br()
        s_alt = "KEEP" if seller_entry["action"] != "KEEP" else "SELL"
        row = {"doc_key": dk, "t": t, "horizon_s": horizon_s, "worlds": 2,
               "seller": seller_entry, "seller_alt": s_alt,
               "phi_factual": fact.phi_krw,
               "phi_seller_alt": br(force_seller=(dk, s_alt)).phi_krw}
        if buyer_entry is not None:
            b_alt = "REJECT" if buyer_entry["action"] == "BUY" else "BUY"
            row["buyer"] = buyer_entry
            row["buyer_alt"] = b_alt
            row["phi_buyer_alt"] = br(force_buyer=(dk, b_alt)).phi_krw
            row["worlds"] = 3
        row["_factual_branch"] = fact
        pending.append(row)

    def wants_epoch(n_elig: int) -> bool:
        """복제하기 **전에** 이 epoch 을 라벨링할지 정한다(bridge 가 묻는다)."""
        return budget.peek(n_elig)

    on_decision.wants_epoch = wants_epoch

    bridge = MarketBridge(market, layout, orders=orders, records=records,
                          end_s=obs.observe_s, arm=arm, grid_s=EPOCH_S,
                          on_decision=(on_decision if budget else None))
    exec_policy, exc = _sf_spt_policy()

    def review(m, t):
        ann.review(m, t)
        bridge.review(m, t)
        if budget is not None:
            tape.snap(m, t)                # ★계수기를 지나갈 때 찍는다(진단용)

    mbt.run(exec_policy, review_fn=review)

    # ── 사건을 끝까지 흡수 (마지막 epoch 이후 완료분)
    bridge._sync(mbt, obs.observe_s)
    tape.snap(mbt, obs.observe_s)

    phi = terminal_cost_krw(records, end_s=obs.observe_s,
                            vessel_idle=vessel_idle_of(mbt, obs.observe_s),
                            yc_extra_move_s=yc_empty_travel_s(mbt),
                            rehandles=rehandles_of(mbt))
    res.phi_krw = phi.total
    res.breakdown = phi.as_dict()
    res.admitted = ann.n_admitted
    res.traded_edges = bridge.traded_edges
    res.policy_exceptions = exc["n"]
    res.n_space, res.n_time = bridge.n_space, bridge.n_time
    res.txn_failed = bridge.txn_failed
    res.rollout_worlds = roll.n_worlds if roll else 0
    res.decisions = len(market.seller.trail)     # 교사 유무와 무관하게 센다

    # ── 동일성 불변식 — factual 가지가 실제 궤적과 **같은 결정**을 냈는가
    from .rollout import identity_check
    for i, row in enumerate(pending):
        w_end = row["t"] + row["horizon_s"]
        v_idle, yc_s, reh = tape.read(w_end)
        # 실제 궤적에서 같은 창을 읽은 값 — **라벨이 아니라 절단 폭 진단용**이다.
        row["phi_read_untruncated"] = terminal_cost_krw(
            records, end_s=w_end, vessel_idle=v_idle,
            yc_extra_move_s=yc_s, rehandles=reh).total
        fact = row.pop("_factual_branch")
        if i < budget.identity_checks:
            res.identity.append(identity_check(
                factual=fact, seller_entry=row["seller"],
                buyer_entry=row.get("buyer"),
                phi_read=row["phi_read_untruncated"]))
    res.labels = pending
    return res
