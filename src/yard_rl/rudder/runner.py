"""창 하나를 굴리며 **epoch 토큰**을 남긴다 ([[YR-223]] 1단계).

■ 왜 v3 의 `run_episode` 를 못 쓰나
  `run_episode` 는 결과를 **집계값으로만** 돌려준다(Φ·거래 수·라벨). epoch 별
  궤적은 `market.seller.trail` 과 `bridge` 안에 있는데 밖으로 안 나온다. 훅
  (`on_decision`)도 교사 예산이 있을 때만 꽂힌다.

  → v3 를 고쳐서 훅을 내는 대신 **여기서 새 러너를 쓴다.** v3 는 한 줄도 안 건드린다.

■ 그러면 v3 와 다른 세계를 재는 것 아닌가 — 그래서 **조립은 v3 것을 그대로 쓴다**
  무대·시장·다리·실행정책을 전부 `v3.stage` 의 함수로 만든다. 다르게 조립하면
  v3 아닌 것의 궤적을 재게 된다. `tests/rudder/test_runner_identity.py` 가
  같은 시드에서 v3 `run_episode` 와 **Φ·거래 수·결정 수가 일치**하는지 검사한다.

■ 창 짝 (pair)
      정책 창    t0 에서 갈라 H 시간 — 시장을 평소대로 연다 · **토큰 기록**
      안팔기 창  같은 t0 에서 갈라 H 시간 — 시장을 아예 안 연다
      Y = Φ(정책) − Φ(안팔기)        ← 맞혀야 하는 값 (원화)

  같은 스냅샷에서 갈라지므로 **공통난수**다 — 무대·도착·고장이 모두 같고 차이는
  오직 재배치뿐이다. 분산이 크게 줄어든다.

■ ⚠️ 창 절단은 양쪽에 똑같이 걸린다
  두 창 모두 `sim.end = t0+H` 다. v3 rollout 이 확인한 대로 절단분은 **공통 상수**라
  차이 `Y` 에서 상쇄된다. 한쪽만 절단하지 않으면 그 편향이 그대로 라벨에 들어간다.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch

from ..v3 import CF_HORIZON_S
from ..v3.actors import BuyerNet, SellerNet
from ..v3.reward.counterfactual import _count_rollout
from ..v3.reward.phi import terminal_cost_krw
from ..v3.stage.bridge import epoch_on_grid
from ..v3.stage.episode import _Ctx, _sf_spt_policy, _sim_from
from ..v3.stage.orders import EPOCH_S, V3Announcer, build_stage, orders_from_schedule
from ..v3.world.integrated.multiblock import MultiBlockTerminal
from ..v3.world.integrated.profiles import build_h21_profile
from ..v3.world.integrated.terminal_stream import (OBS_24H, admission_epochs,
                                                   ensure_time_ledger)
from ..v3.world.integrated.yard_layout import terminal_layout
from .tape import EpochTape


@dataclass
class Window:
    """창 하나의 궤적 — 학습에 들어가는 최소 단위."""

    seed: int
    load: int
    t0: float
    horizon_s: float
    tokens: list = field(default_factory=list)
    ts: list = field(default_factory=list)
    phi_policy: float = 0.0
    phi_keep: float = 0.0
    #: ★맞혀야 하는 값 — 이 창에서 재배치가 만든 원화 차이
    y_krw: float = 0.0
    traded: int = 0
    decisions: int = 0

    def as_dict(self) -> dict:
        return {"seed": self.seed, "load": self.load, "t0": self.t0,
                "horizon_s": self.horizon_s, "n_tokens": len(self.tokens),
                "phi_policy": self.phi_policy, "phi_keep": self.phi_keep,
                "y_krw": self.y_krw, "traded": self.traded,
                "decisions": self.decisions}


# --------------------------------------------------------------------- 창 하나
def run_branch(ctx, *, mbt, orders, records, decided, t0, horizon_s,
               freeze: bool, record: bool, freeze_at=None) -> dict:
    """스냅샷에서 갈라 H 만큼 굴린다.

    `freeze=True` 면 t0 부터 시장을 아예 안 연다(안팔기 창).
    `record=True` 면 60초 격자마다 토큰을 남긴다.
    `freeze_at` 은 ★**그 epoch 들만** 시장을 안 여는 개입이다 (관문 F).
        RUDDER 가 "이 epoch 이 −8만원 기여" 라고 하면, 실제로 그 epoch 만 막고
        다시 굴려 부호가 맞는지 본다. 다른 RUDDER 연구는 못 하는 검증이다 —
        우리에겐 반사실 시뮬레이터가 있다.

    ★분기 epoch 은 **손으로 연다**. 엔진이 `review_fn` 직전에 park 기록을 비우므로
    복제본을 그냥 굴리면 t0 를 건너뛴다 (v3 `SnapshotRollout` 과 같은 이유).
    """
    # ★v3 의 rollout 계수기를 **같이** 쓴다. 창 굴리기도 반사실 시뮬레이션이므로
    #   판정 하드가드(`rollout_calls_during_eval == 0`)가 이걸 못 보면 가드가
    #   거짓말을 하게 된다. 세는 것은 v3 를 고치는 것이 아니다 — 계약을 지키는 것이다.
    _count_rollout()
    end = t0 + horizon_s
    snap = copy.deepcopy(mbt)
    for sim in snap.blocks.values():
        sim.end = min(sim.end, end)

    o2 = dict(orders)
    r2 = copy.deepcopy(records)
    market2 = ctx.make_market(snap, decided=set(decided))
    # ★`end_s` 는 하루의 끝 그대로 — 절단은 `sim.end` 로만 건다. 여기에 t0+H 를
    #   주면 Seller 가 먼 이연 칸을 못 보고 **다른 정책**이 된다(v3 가 겪은 버그).
    bridge2 = ctx.make_bridge(market2, orders=o2, records=r2, on_decision=None)
    if freeze:
        bridge2.arm = "NO_REALLOC"
    ann2 = ctx.announcer.clone_fresh()

    tape = (EpochTape(t0=float(t0), horizon_s=float(horizon_s),
                      block_vessel=getattr(ctx, "block_vessel", {}))
            if record else None)

    base_arm = bridge2.arm
    hit = {float(v) for v in (freeze_at or ())}

    def review(m, tt):
        ann2.review(m, tt)
        # ★개입 — 지목된 epoch 에서만 시장을 닫는다. 나머지는 평소대로.
        bridge2.arm = "NO_REALLOC" if float(tt) in hit else base_arm
        bridge2.review(m, tt)
        if tape is not None and epoch_on_grid(tt, ctx.grid_s):
            tape.snap(m, tt, market=market2, records=r2, bridge=bridge2)

    review(snap, t0)                       # ★분기 epoch 을 손으로 연다
    snap.run(ctx.make_exec_policy(), review_fn=review)
    if tape is not None:                   # 창 끝의 잔여 사건까지 흡수
        bridge2._sync(snap, end)
        tape.snap(snap, end, market=market2, records=r2, bridge=bridge2)

    phi = terminal_cost_krw(r2, end_s=end, vessel_idle=ctx.vessel_idle(snap, end),
                            yc_extra_move_s=ctx.yc_extra_move_s(snap),
                            rehandles=ctx.rehandles(snap))
    return {"phi": float(phi.total), "tape": tape,
            "traded": bridge2.traded_edges, "decisions": len(market2.seller.trail)}


def run_window(ctx, *, mbt, orders, records, decided, t0, seed: int, load: int,
               horizon_s: float = CF_HORIZON_S) -> Window:
    """정책 창과 안팔기 창을 **같은 분기점에서** 굴려 짝을 만든다."""
    pol = run_branch(ctx, mbt=mbt, orders=orders, records=records, decided=decided,
                     t0=t0, horizon_s=horizon_s, freeze=False, record=True)
    keep = run_branch(ctx, mbt=mbt, orders=orders, records=records, decided=decided,
                      t0=t0, horizon_s=horizon_s, freeze=True, record=False)
    tp = pol["tape"]
    return Window(seed=seed, load=load, t0=float(t0), horizon_s=float(horizon_s),
                  tokens=tp.matrix(), ts=[tk.t for tk in tp.tokens],
                  phi_policy=pol["phi"], phi_keep=keep["phi"],
                  y_krw=pol["phi"] - keep["phi"],
                  traded=pol["traded"], decisions=pol["decisions"])


# --------------------------------------------------------------------- 하루
def build_ctx(*, load: int, seed: int, seller_net=None, buyer_net=None,
              lead_mode: str = "DIST", window_s: float = 1800.0,
              explore: float = 0.0, obs=None):
    """v3 `run_episode` 와 **같은 재료**로 무대를 세운다.

    한 줄이라도 다르면 v3 아닌 것을 재게 된다 — 동일성 시험이 이걸 검사한다.
    """
    obs = obs or OBS_24H
    prof, layout = build_h21_profile(), terminal_layout()
    built = build_stage(load=load, seed=seed, profile=prof, layout=layout,
                        obs=obs, lead_mode=lead_mode)
    orders, records = orders_from_schedule(built)
    mbt = MultiBlockTerminal(
        {b: ensure_time_ledger(_sim_from(s, prof))
         for b, s in built["scenarios"].items()},
        extra_review_epochs=admission_epochs(obs))
    ann = V3Announcer(built["schedule"], end_s=built["sim_end_s"])
    if seller_net is None or buyer_net is None:
        torch.manual_seed(int(seed))       # ★v3 와 같은 시드 규약
    ctx = _Ctx(seller_net=(seller_net if seller_net is not None else SellerNet()),
               buyer_net=(buyer_net if buyer_net is not None else BuyerNet()),
               layout=layout, announcer=ann, arm="RL", grid_s=EPOCH_S,
               window_s=window_s, explore=explore, seed=seed,
               episode_end_s=obs.observe_s, cf_horizon_s=CF_HORIZON_S)
    ctx.block_vessel = built.get("block_vessel", {})
    return ctx, mbt, orders, records, built, obs


def window_starts(*, obs_end_s: float, horizon_s: float, n: int,
                  first_s: float = 3_600.0) -> tuple:
    """창 시작 시각을 하루에 **고르게** 흩는다.

    앞부분만 뽑으면 새벽 표본만 남는다(v3 `RolloutBudget` 이 겪은 문제와 같다).
    난수를 안 쓰므로 같은 설정이면 **같은 시각**이 나온다 — 재현된다.
    """
    last = obs_end_s - horizon_s
    if n <= 1 or last <= first_s:
        return (first_s,)
    step = (last - first_s) / (n - 1)
    return tuple(round((first_s + i * step) / EPOCH_S) * EPOCH_S for i in range(n))


def collect_day(*, load: int, seed: int, n_windows: int,
                horizon_s: float = CF_HORIZON_S, seller_net=None, buyer_net=None,
                explore: float = 0.0, obs=None, on_window=None) -> list:
    """하루를 굴리며 예정된 시각마다 **그 자리에서** 창 짝을 굴린다.

    ★스냅샷을 모아 두지 않고 즉시 소비한다 — 창 20개를 들고 있으면 메모리가 터진다.
    창은 복제본에서 도므로 하루 궤적은 영향을 안 받는다.
    """
    ctx, mbt, orders, records, built, obs = build_ctx(
        load=load, seed=seed, seller_net=seller_net, buyer_net=buyer_net,
        explore=explore, obs=obs)
    starts = set(window_starts(obs_end_s=obs.observe_s, horizon_s=horizon_s,
                               n=n_windows))
    market = ctx.make_market(mbt)
    bridge = ctx.make_bridge(market, orders=orders, records=records,
                             on_decision=None)
    exec_policy, _exc = _sf_spt_policy()
    out = []

    def review(m, t):
        # ★순서가 v3 와 같아야 한다: 공고 → (분기) → 시장.
        #   공고 앞에서 뜨면 그 epoch 에 들어올 오더를 못 본 세계가 되고,
        #   시장 뒤에서 뜨면 거래가 이미 확정돼 "안 팔았다면" 을 만들 수 없다.
        ctx.announcer.review(m, t)
        if t in starts:
            w = run_window(ctx, mbt=m, orders=orders, records=records,
                           decided=market.decided, t0=t, seed=seed, load=load,
                           horizon_s=horizon_s)
            out.append(w)
            if on_window is not None:
                on_window(w)
        bridge.review(m, t)

    mbt.run(exec_policy, review_fn=review)
    return out


def run_day_phi(*, load: int, seed: int, seller_net=None, buyer_net=None,
                explore: float = 0.0, obs=None) -> dict:
    """창 없이 하루만 굴려 Φ 를 낸다 — **동일성 시험 전용**.

    v3 `run_episode` 와 같은 값이 나와야 이 꾸러미가 v3 를 재고 있다고 말할 수 있다.
    """
    ctx, mbt, orders, records, built, obs = build_ctx(
        load=load, seed=seed, seller_net=seller_net, buyer_net=buyer_net,
        explore=explore, obs=obs)
    market = ctx.make_market(mbt)
    bridge = ctx.make_bridge(market, orders=orders, records=records,
                             on_decision=None)
    exec_policy, exc = _sf_spt_policy()
    # ★[[YR-235]] A2 (2026-08-26) — v3 와 **같은 창**을 재야 한다.
    #   시뮬은 배수 2시간을 더 돌지만 Φ 항2/3/4 는 관측창까지만 센다. 여기서
    #   계수기를 찍어 두지 않으면 rudder 만 26시간을 세어 동일성이 깨진다.
    from ..v3.stage.episode import _CounterTape
    tape = _CounterTape(getattr(ctx, "block_vessel", {}))

    def review(m, t):
        ctx.announcer.review(m, t)
        bridge.review(m, t)
        if t <= obs.observe_s:
            tape.snap(m, t)

    mbt.run(exec_policy, review_fn=review)
    bridge._sync(mbt, obs.observe_s)
    v_idle, yc_s, reh = tape.read(obs.observe_s)
    phi = terminal_cost_krw(records, end_s=obs.observe_s, vessel_idle=v_idle,
                            yc_extra_move_s=yc_s, rehandles=reh)
    return {"phi_krw": float(phi.total), "traded_edges": bridge.traded_edges,
            "decisions": len(market.seller.trail), "admitted": ctx.announcer.n_admitted,
            "n_space": bridge.n_space, "n_time": bridge.n_time,
            "txn_failed": bridge.txn_failed, "policy_exceptions": exc["n"]}


def intervene(ctx, *, mbt, orders, records, decided, t0, horizon_s, epochs) -> dict:
    """★관문 F — 지목한 epoch 들만 막고 다시 굴려 **실제 효과**를 잰다.

    `d_krw = Φ(정책) − Φ(그 epoch 만 막은 정책)` 이다. RUDDER 가 그 epoch 에
    매긴 기여도와 **부호가 같아야** 기여도가 인과를 가리킨다고 말할 수 있다.
    """
    pol = run_branch(ctx, mbt=mbt, orders=orders, records=records, decided=decided,
                     t0=t0, horizon_s=horizon_s, freeze=False, record=False)
    blk = run_branch(ctx, mbt=mbt, orders=orders, records=records, decided=decided,
                     t0=t0, horizon_s=horizon_s, freeze=False, record=False,
                     freeze_at=epochs)
    return {"phi_policy": pol["phi"], "phi_blocked": blk["phi"],
            "d_krw": pol["phi"] - blk["phi"], "epochs": [float(e) for e in epochs]}


def snapshot_at(*, load: int, seed: int, t0: float, seller_net=None,
                buyer_net=None, explore: float = 0.0, obs=None) -> dict:
    """하루를 굴리다 `t0` 에서 재료를 통째로 복제해 돌려준다 (관문 F 용).

    ⚠️ 하루를 끝까지 굴린다(169초). `t0` 에서 멈추는 길도 있지만 엔진의 종료
    경로를 건드려야 해서 **v3 를 안 고친다는 계약**과 충돌한다 — 시간을 낸다.
    """
    ctx, mbt, orders, records, built, obs = build_ctx(
        load=load, seed=seed, seller_net=seller_net, buyer_net=buyer_net,
        explore=explore, obs=obs)
    market = ctx.make_market(mbt)
    bridge = ctx.make_bridge(market, orders=orders, records=records,
                             on_decision=None)
    exec_policy, _exc = _sf_spt_policy()
    box = {}

    def review(m, t):
        ctx.announcer.review(m, t)
        if not box and float(t) >= float(t0):
            box.update(t0=float(t), mbt=copy.deepcopy(m),
                       orders=dict(orders), records=copy.deepcopy(records),
                       decided=set(market.decided))
        bridge.review(m, t)

    mbt.run(exec_policy, review_fn=review)
    if not box:
        raise RuntimeError(f"t0={t0} 에 닿지 못했다 — 관측창을 확인하라")
    box["ctx"] = ctx
    return box
