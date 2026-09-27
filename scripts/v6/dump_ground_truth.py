"""v5(=v6 사본) 세계를 규칙으로 굴려 **정답 궤적**을 남긴다 ([[YR-327]]).

배열 세계가 옮겨질 때마다 이 궤적과 대조한다 — 사건 하나하나의 (시각·종류·대상)이
같은 순서로 나와야 하고, 비용 항목·턴타임·재처리 수가 같아야 한다.

■ 두 가지 모드
  terminal  21블록 전체를 다중블록 루프로 굴린다 (게이트·이송 포함). 최종 목표.
  block     블록 **하나**를 엔진만으로 굴린다 (이송 없음). 1번 조각(단일 블록 엔진)의
            동등성 시험용. 다중블록 문맥이 없으니 게이트 승인은 시나리오에 적힌 대로.

    PYTHONPATH=src python scripts/v6/dump_ground_truth.py --load 30 --seed 9900777

■ 정책·정보수준 (조각 3·4 통합, 2026-09-26) — block 모드의 결정 규칙을 고른다
  --policy sf_spt     (기본) ResolverPolicy(ServiceFirstSPTPreference) + CandidateGenerator(LEGACY_DEFAULT) —
                      SERVE·PRE_REHANDLE·REPOSITION·WAIT 를 공동 resolver 가 고른다 (배열판: engine_step joint 규약 +
                      dispatch.make_resolver("sf_spt")). 기존 파일 이름은 이 조합일 때만 그대로다.
  --policy reference  ReferenceDispatcher.run 의미 — 크레인 순 live SERVE 후보 · 본선 우선→최장대기→job_id
                      (배열판: 순차 규약 + dispatch.policy_reference)
  --policy first      크레인 순 live SERVE 후보 중 job_id 가 가장 작은 것 (배열판: engine_step.first_by_id)
    ⚠️ **블록 Y01 에서 reference 와 first 는 같은 궤적을 낸다** (해시 760af9a960d07bf8) — 버그가 아니다.
       Y01 은 외부트럭이 0 이라 모든 후보가 본선연계이고 누적대기가 0 이므로 reference 의 정렬 3-튜플
       `(본선?0:1, −누적대기, job_id)` 이 **job_id 순**으로 줄어든다. 그 순서가 곧 `candidates_for` 가
       돌려주는 순서라 `cands[0]` 과 같아진다 — 240 크레인결정 전부에서 같은 후보를 골랐음을 직접 확인했다
       (2026-09-26). 정보수준(PRE_ADVICE·BLOCK_ARRIVAL)도 이 두 정책에는 영향이 없다(SERVE 만 본다).
       ⇒ 두 정책을 **구별**하려면 외부트럭이 있는 무대가 필요하다.
  --policy v5net      ★**학습 정책망** (조각 7) — v5 `ppo/model.BlockPolicy`(tanh·37특징) + `ppo/crane.CraneActor`
                      (순차 조건부 joint_mask · 최고점 선택). 배열판: `dispatch.make_v5net_policy`.
                      `--ckpt <경로>` 를 주면 그 체크포인트를, 없으면 `--net-seed`(기본 20260927)·`--hidden`(64) 로
                      **고정 시드 무작위 초기화** 망을 쓴다. 쓴 가중치는 결과 JSON 옆에 `.npz` 로 남긴다
                      (배열 쪽 시험이 **같은 수**를 싣도록 — torch 버전에 의존하지 않는다).
                      **두 모드 다 된다**: block(블록 하나) · terminal(21블록). 터미널에서는 명단(`Order`)·
                      기록(`ExecutionRecord`)을 `net_schedule_orders` 가 만들어 `MarketBridge._sync` 가 채운다 —
                      그래야 블록 요약의 네 칸(블록 안·오는 중·곧 올 통지·줄 선 대수)이 0 이 아니다
                      (블록 Y01 단독 무대는 외부트럭 0 이라 그 넷이 전부 0 이다).
                      ⚠️ 터미널 궤적은 아직 **배열판과 대조되지 않는다** — 배열 쪽에 블록축 `reserve_s`·`end_s`
                      와 그 네 칸의 원장 연결이 없다(조각 8). 재현성만 시험이 못박는다.
  --info-level        InformationLevel 이름 (기본 PRE_ADVICE; BLOCK_ARRIVAL 이면 ETA wake·PRE 후보가 없다)
                      ⚠️ v5net 은 `ppo/crane.py:74` 가 `stage/episode.INFO_LEVEL`(PRE_ADVICE) 를 직접 읽으므로
                      PRE_ADVICE 만 허용한다 (다른 값을 주면 조용히 어긋나지 않고 크게 실패한다).
  결과 파일: block_<블록>_load<L>_seed<S>[_<policy>_<level>[_<망표식>]].json · terminal_load<L>_seed<S>[...].json
            — sf_spt·PRE_ADVICE 조합만 접미사 없음(기존 이름 보존)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

#: ★numpy 를 torch 보다 **먼저** 들인다 — 거꾸로면 OMP #15(libiomp5md.dll 중복)로 프로세스가 그냥 죽는다
#:   (`.venv-jax` 는 아나콘다 numpy(MKL) + pip torch 라 OpenMP 런타임이 둘이다). `--policy v5net` 이 torch 를 쓴다.
import numpy  # noqa: F401

from yard_rl.v6.world.domain.enums import InformationLevel
from yard_rl.v6.world.integrated import (baselines as bl, candidates as cd,
                                          engine as eng, multiblock as mb,
                                          policy_config as pc, profiles as pr,
                                          terminal_stream as ts, time_sell, yard_layout as yl)

LVL = InformationLevel.PRE_ADVICE
#: 학습 정책망 이름 (조각 7) — 나머지는 규칙 정책 (조각 1~3)
NET = "v5net"
POLICIES = ("sf_spt", "reference", "first", NET)
#: 고정 시드 무작위 망의 기본 씨앗 — 결과 파일 이름에 들어간다
NET_SEED = 20_260_927
NET_HIDDEN = 64


def _build(load: int, seed: int):
    prof, layout = pr.build_h21_profile(), yl.terminal_layout()
    built = ts.build_diurnal(prof, seed, obs=ts.OBS_24H, layout=layout,
                             params=ts.TerminalStreamParams(load_4h=load),
                             day_total=load, background_seed=seed)
    return prof, built


def _sim(prof, scn, level=LVL):
    s = eng.TerminalSimulator(prof, scn, check_invariants=True)
    s.info_level = level
    return s


def _rule_policy(level=LVL):
    """규칙(작업 우선·최단작업) — v5 판정의 바닥. 망 없음, 난수 없음."""
    gens: dict[int, object] = {}
    pol = bl.ResolverPolicy(bl.ServiceFirstSPTPreference(), "SF")

    def exec_policy(sim, dp):
        g = gens.setdefault(id(sim), cd.CandidateGenerator(config=pc.LEGACY_DEFAULT))
        gb = {c: g.generate(sim, c, level) for c in dp.crane_ids}
        bl._apply(sim, pol.decide(sim, dp, gb))
    return exec_policy


def _serve_policy(select):
    """순차 규약 — ReferenceDispatcher.run (dispatcher.py:19-32): 크레인 순서대로 live SERVE 후보를 뽑아 하나씩 assign."""
    from yard_rl.v6.world.contract.schema import CandidateKind

    def exec_policy(sim, dp):
        for cid in dp.crane_ids:
            cands = sim.candidates_for(cid)
            if cands:
                sim.assign(cid, eng.CraneAssignment(cid, CandidateKind.SERVE, job_ref=select(sim, cid, cands)))
            else:
                sim.assign(cid, eng.CraneAssignment(cid, CandidateKind.WAIT))
        sim.close_decision()
    return exec_policy


# ───────────────────────────────────────────────── 학습 정책망 (조각 7)
def net_tag(ckpt: str | None, seed: int, hidden: int) -> str:
    """가중치를 한 줄 이름으로 — 결과 파일 이름·JSON 에 같이 들어간다 (다른 망의 궤적이 같은 이름을 못 쓰게)."""
    return f"ckpt-{Path(ckpt).stem}" if ckpt else f"seed{seed}h{hidden}"


def build_net(*, ckpt: str | None = None, seed: int = NET_SEED, hidden: int = NET_HIDDEN):
    """v5 `ppo/model.BlockPolicy` 하나. 체크포인트가 있으면 그것을, 없으면 **고정 시드 무작위 초기화**.

    무작위 망도 "결정이 갈리는가" 를 보는 데는 충분하다 — 오히려 점수가 고르게 퍼져 동점이 적다.
    """
    import torch
    from yard_rl.v6.ppo.model import BlockPolicy
    if ckpt:
        from yard_rl.v6.ppo.checkpoint import load_policy
        return load_policy(ckpt)
    torch.manual_seed(int(seed))                 # torch 기본 초기화(kaiming_uniform) 의 난수 흐름
    return BlockPolicy(hidden=int(hidden))


def save_net_npz(policy, path) -> None:
    """가중치를 numpy `.npz` 로 — 배열 쪽 시험이 **torch 버전에 의존하지 않고** 같은 수를 싣는다."""
    import numpy as np
    np.savez(str(path), **{k: v.detach().cpu().numpy() for k, v in policy.state_dict().items()})


def load_net_npz(path) -> dict:
    """`.npz` → torch `state_dict` 모양의 numpy 사전 (`gpu/v5net.load_v5_params` 가 그대로 받는다)."""
    import numpy as np
    with np.load(str(path)) as z:
        return {k: z[k] for k in z.files}


def net_from_npz(path):
    """`.npz` → v5 `BlockPolicy` (v5 쪽 재구동용 — 시험이 **같은 망**으로 v5 를 다시 굴린다)."""
    import torch
    from yard_rl.v6.ppo.model import BlockPolicy
    sd = load_net_npz(path)
    pol = BlockPolicy(hidden=int(sd["trunk.0.bias"].shape[0]))
    pol.load_state_dict({k: torch.as_tensor(v) for k, v in sd.items()}, strict=True)
    return pol


def net_terminal_inputs(sim, bid: str, end_s: float, *, reserve_of=None, lead_s: float = 1800.0):
    """블록 요약 8칸이 요구하는 **터미널 층 입력** — (mbt 대역, records, orders).

    v5 `ppo/runtime.block_state`(107-110행) 는 `MarketBridge` 의 `records`·`orders`·`end_s` 를 읽는다.
    블록 단독 무대에는 그 다리가 없으므로 명단 규약대로 만든다 (`stage/orders.py:88` = `round(도착예정,3)`).
    외부트럭만이 오더다 (본선 작업은 records·orders 어디에도 없다 — bridge.py:66-67) — 블록 Y01 은
    외부트럭 0 이라 둘 다 빈 사전이고, 그래서 블록 요약 네 칸(안·오는중·곧올통지·줄선대수)이 0 이다.

    ■ ★`reserve_s`(통지된 예정)를 어디서 얻나 — **정보 경계** (2026-09-26 검증 반박)
      `Order.in_out_reserve_s` 는 "통지된 예정" 이지 "실현된 게이트인" 이 아니다 (features/block.py:92,122).
      `reserve_of(jid) -> 도착예정 초` 를 주면 그 값을 쓴다 — 터미널 명단(`built["schedule"]` 의 `arrival_s`)이
      그것이고, `run_block` 이 그렇게 넘긴다.
      ⚠️ `reserve_of` 가 없으면 **실현 게이트인**(`j.actual_gate_in`)을 그 칸에 넣는다. 양쪽(v5·배열)이 같은
         값을 받으므로 **동등성 판정에는 무해**하지만, 칸 5(곧 올 통지분)가 미래를 아는 값으로 만들어진다 —
         즉 이 하네스는 **동등성 전용이고 정보 경계 검증에는 쓸 수 없다**. 명단이 있는 무대에서는 꼭 넘겨라.
    """
    from yard_rl.v6.schema.order import Order
    from yard_rl.v6.schema.record import ExecutionRecord
    from yard_rl.v6.world.domain.enums import JobFlow
    orders, records = {}, {}
    for jid, j in sim.jobs.items():
        if not j.is_external_truck:
            continue
        #: 명단이 있으면 **통지된 예정**, 없으면 실현 게이트인 (위 ⚠️)
        arr = float(j.actual_gate_in or 0.0) if reserve_of is None else float(reserve_of(jid))
        notice = round(max(0.0, arr - float(lead_s)), 3)
        orders[jid] = Order(doc_key=jid, in_out=(0 if j.flow == JobFlow.GATE_OUT else 1),
                            copino_notice_s=notice, in_out_reserve_s=round(arr, 3), con_loc=bid,
                            con_no=(j.target_container if j.target_container is not None else f"IN-{jid}"))
        records[jid] = ExecutionRecord(doc_key=jid, copino_notice_s=notice)
    return SimpleNamespace(blocks={bid: sim}), records, orders


def net_schedule_orders(built: dict, *, lead_s: float = 1800.0):
    """터미널(21블록) 명단 → (records, orders) — `stage/orders.orders_from_schedule` 과 같은 규약.

    `run_terminal` 이 쓰는 `ScheduledAnnouncer(lead_s=1800.0)` 와 **같은 리드**를 쓴다.
    ★사전은 **엔진 작업 번호**(블록 접두가 붙은 `Y01:D-00013`) 로 키를 매긴다 — `MarketBridge._sync` 와
      `features/block.py` 가 `sim.jobs` 의 키로 찾기 때문이다. `Order.doc_key` 는 접두를 금지하므로
      (order.py:53) 꼬리만 넣는다 — 블록 요약은 `doc_key` 를 읽지 않는다 (`con_loc`·`in_out_reserve_s` 만).
    ★기록은 **비어서** 시작한다 (orders.py:77) — 터미널이 사건을 보내야 찬다. 미리 적으면 미래가 샌다.
    """
    from yard_rl.v6.schema.order import Order
    from yard_rl.v6.schema.record import ExecutionRecord
    orders, records = {}, {}
    for e in built["schedule"]:
        jid = e["job_id"]
        notice = round(max(0.0, float(e["arrival_s"]) - float(lead_s)), 3)
        orders[jid] = Order(doc_key=jid.split(":")[-1],
                            in_out=(0 if e["flow"] == "GATE_OUT" else 1),
                            copino_notice_s=notice, in_out_reserve_s=round(float(e["arrival_s"]), 3),
                            con_loc=e["block"],
                            con_no=(e["target"] or f"IN-{jid.split(':')[-1]}"))
        records[jid] = ExecutionRecord(doc_key=jid.split(":")[-1], copino_notice_s=notice)
    return records, orders


class NetRuntime:
    """v5 `ppo/crane.CraneActor` 가 요구하는 **최소** 런타임.

    옮긴 것은 `PPORuntime.select`(runtime.py:130-136) 의 고정 운영 경로 세 줄과 `block_state`(107-110행)
    뿐이다 — 학습 버퍼·보상·업데이트는 결정에 영향이 없다. `sample_actions=False`(최고점 선택) 고정:
    추첨 경로는 `torch.multinomial` 의 Generator 흐름이라 배열판이 같은 표본을 낼 수 없다([[YR-319]]).
    """

    def __init__(self, policy, sim, bid: str, end_s: float, *, on_select=None,
                 mbt=None, records=None, orders=None, reserve_of=None):
        import torch
        self._torch = torch
        self.policy = policy
        if mbt is None:
            #: 블록 단독 — 터미널 다리가 없으므로 명단을 그 블록 것만 만든다 (`net_terminal_inputs`)
            self.mbt, records, orders = net_terminal_inputs(sim, bid, end_s, reserve_of=reserve_of)
            self.block_of = {id(sim): bid}
        else:
            #: 터미널(21블록) — `MultiBlockTerminal` 을 **그대로** 쓰고 명단은 밖에서 받는다
            self.mbt = mbt
            self.block_of = {id(s): b for b, s in mbt.blocks.items()}
        self.bridge = SimpleNamespace(records=records, orders=orders, end_s=float(end_s),
                                      _sync=self._sync)
        self.crane_actions: dict[str, int] = {}
        self.on_select = on_select              # 시험용 가로채기 — 결정마다 행렬·점수·가치·답을 남긴다
        self.n_select = 0

    def _sync(self, mbt, t: float, *, block_ids=None) -> None:
        """v5 `MarketBridge._sync` 를 **그대로** 부른다 (쓰는 속성은 records·_stamp_past 뿐)."""
        from yard_rl.v6.stage.bridge import MarketBridge
        MarketBridge._sync(SimpleNamespace(records=self.bridge.records,
                                           _stamp_past=MarketBridge._stamp_past),
                           mbt, t, block_ids=block_ids)

    def block_state(self, bid: str, t: float):
        from yard_rl.v6.features.block import block_features
        return block_features(self.mbt, bid, t, n_cands=None, records=self.bridge.records,
                              orders=self.bridge.orders, end_s=self.bridge.end_s)

    def select(self, role: str, bid: str, t: float, rows, mask=None) -> int:
        from yard_rl.v6.ppo.model import encode
        torch = self._torch
        x = encode(rows, role)                                   # runtime.py:130
        mask = torch.ones(len(x), dtype=torch.bool) if mask is None else mask.clone().bool()
        with torch.no_grad():
            dist = self.policy.distribution(x, mask)             # 133행
            action = int(dist.probs.argmax())                    # 135행 (sample_actions=False)
            logp = float(dist.log_prob(torch.tensor(action)))
            value = self.policy.value(x)                         # critic (49-50행)
        self.n_select += 1
        if self.on_select is not None:
            #: 시험만 쓰는 추가 순전파 — 마스크 **전** 원점수(actor 머리 그대로)를 따로 받는다.
            #: `dist.logits` 는 torch `Categorical` 이 `z − logsumexp(z)` 로 정규화해 든 값이라
            #: 배열판 `actor_scores` 와 바로 맞댈 수 없다.
            with torch.no_grad():
                raw = self.policy.actor(self.policy.trunk(x)).squeeze(-1)
            self.on_select(dict(role=role, bid=bid, t=float(t), rows=[list(r) for r in rows],
                                mask=[bool(m) for m in mask], x=x.numpy().copy(),
                                raw_logits=raw.numpy().copy(),
                                logits=dist.logits.numpy().copy(), probs=dist.probs.numpy().copy(),
                                value=value.numpy().copy(), action=action, log_prob=logp))
        return action


def _net_policy(policy, sim, bid: str, end_s: float, level=LVL, *, on_select=None, reserve_of=None):
    """학습 정책망 — v5 `ppo/crane.CraneActor` 를 **그대로** 부른다 (joint_mask·candidate_row·_apply 원본).

    ⚠️ `CraneActor` 는 `stage/episode.INFO_LEVEL`(PRE_ADVICE) 를 직접 읽는다 (crane.py:14,74) —
       다른 정보수준은 조용히 어긋나므로 크게 실패시킨다.
    예외 대체(`stage/episode.py:212-216` "한 크레인 실패 = 전원 WAIT")는 `CraneActor` 자체에는 **없다**
    (crane.py:81 주석: 예외는 일부러 전파한다). 여기서는 계수하며 감싸 두고, 0 이 아니면 보고에 드러난다.
    ⚠️ **그 감싸기는 v5 의 실제 학습 드라이버보다 관대하다** (2026-09-26 검증 반박) — v5 학습 경로는
       `stage/month_run.py:541-544` 에서 `exec_policy = ppo.execute` 로 갈아끼우므로 `_rule_policy` 의
       try/except 를 거치지 않고 예외가 그대로 전파된다. 지금까지 예외 0 건이라 궤적에 차이가 없지만,
       조각 8 이 이 하네스를 정본으로 쓸 것이므로 차이를 여기 적어 둔다 (예외가 나면 v5 는 죽고 이 하네스는
       전원 WAIT 로 계속 간다).
    """
    from yard_rl.v6.stage.episode import INFO_LEVEL
    if level != INFO_LEVEL:
        raise ValueError(f"v5net 은 {INFO_LEVEL.name} 만 된다 (ppo/crane.py:74) — 받은 값 {level.name}")
    return _net_driver(NetRuntime(policy, sim, bid, end_s, on_select=on_select, reserve_of=reserve_of))


def _net_driver(rt: "NetRuntime"):
    """`NetRuntime` → (exec_policy, rt, 예외 계수기). 블록 단독·터미널이 같은 몸통을 쓴다."""
    from yard_rl.v6.ppo.crane import CraneActor
    from yard_rl.v6.stage.episode import INFO_LEVEL
    actor = CraneActor(rt)
    exc = {"n": 0}

    def exec_policy(s, dp):
        try:
            actor(s, dp)
        except Exception:                                        # stage/episode.py:212-216
            exc["n"] += 1
            g = actor.generators[rt.block_of[id(s)]]
            bl._apply(s, {c: bl._wait_of(g.generate(s, c, INFO_LEVEL)) for c in dp.crane_ids})
    return exec_policy, rt, exc


def make_policy(name: str, level=LVL, *, sim=None, end_s: float | None = None,
                block: str | None = None, net=None, on_select=None, reserve_of=None):
    """--policy 이름 → exec_policy(sim, dp). v5net 은 `sim`·`end_s`·`block`·`net` 을 함께 받는다."""
    if name == "sf_spt":
        return _rule_policy(level)
    if name == "reference":
        from yard_rl.v6.world.integrated.dispatcher import ReferenceDispatcher
        return _serve_policy(ReferenceDispatcher().select)
    if name == "first":
        return _serve_policy(lambda sim_, cid, cands: cands[0])
    if name == NET:
        if sim is None or end_s is None or block is None or net is None:
            raise ValueError("v5net 은 sim·end_s·block·net 이 필요하다 (블록 요약이 터미널 층 값을 읽는다)")
        return _net_policy(net, sim, block, end_s, level, on_select=on_select, reserve_of=reserve_of)
    raise ValueError(f"모르는 정책 {name!r} — {POLICIES}")


def _block_dump(bid: str, s) -> dict:
    """블록 하나의 궤적 — 배열판이 **그대로** 재현해야 하는 것."""
    k = s.kpis
    return {
        "block": bid, "n_jobs": len(s.jobs), "end_s": s.end, "clock_s": s.now,
        "n_cranes": len(list(s.fleet.all())),
        "event_hash": s.event_stream_hash(),
        "n_events": len(s.event_log),
        "events": [[round(t, 6), k_, p] for (t, k_, p) in s.event_log],
        "cost_raw": s.cost.episode_raw(),
        "kpis": {"queue_area_s": k.queue_area_s, "tail_area_s": k.tail_area_s,
                 "loaded_gantry_m": k.loaded_gantry_m, "empty_gantry_m": k.empty_gantry_m,
                 "rehandle_count": k.rehandle_count, "pre_rehandle_count": k.pre_rehandle_count,
                 "completed_external": k.completed_external, "completed_vessel": k.completed_vessel,
                 "vessel_delay_s": k.vessel_delay_s, "positioning_count": k.positioning_count},
        "deadlock_escapes": s.deadlock_escape_count,
        "unfinished": s.unfinished_backlog(),
    }


def _job_rows(s) -> dict:
    """블록 하나의 **오더별 진실** — 배열판 `from_terminal_world` 의 blocks[b]["jobs"] 와 같은 키.

    시각 셋(gate_in·block_arrival·actual_gate_out)의 출처는 **시간 장부 기록**이다 (v5 `time_ledger.records`) —
    Job 의 actual_* 가 아니라. 배열판도 같은 곳(장부 열)에서 읽는다.
    """
    tl = getattr(s, "time_ledger", None)
    out: dict[str, dict] = {}
    for jid, j in s.jobs.items():
        r = tl.records.get(jid) if tl is not None else None
        out[jid] = {"status": j.status.name, "assigned_crane": j.assigned_crane,
                    "rehandle_count": j.rehandle_count,
                    "service_start": j.service_start, "service_end": j.service_end,
                    "gate_in": (r.gate_in if r else None), "block_arrival": (r.block_arrival if r else None),
                    "actual_gate_out": (r.gate_out if r else None)}
    return out


def _ledger_areas(s) -> dict:
    """v5 `TimeLedger` 적분 3항 — 배열판 blocks[b]["ledger"] 와 같은 키."""
    tl = getattr(s, "time_ledger", None)
    if tl is None:
        return {}
    return {"terminal_area_s": tl.terminal_area_s, "block_area_s": tl.block_area_s,
            "block_tail_area_s": tl.block_tail_area_s}


def run_terminal(load: int, seed: int, policy: str = "sf_spt", *, net=None) -> dict:
    """터미널 21블록 — 기본은 규칙 정책. `net` 이 있으면 **학습 정책망**(`--policy v5net`) 이다.

    ★학습 정책망 경로는 `MarketBridge` 대신 **최소 다리**(`NetRuntime`)를 쓴다 — 명단(`Order`)·기록
      (`ExecutionRecord`)은 `net_schedule_orders` 가 `stage/orders.orders_from_schedule` 과 같은 규약으로
      만들고, 기록을 채우는 것은 v5 자신의 `MarketBridge._sync` 다. 시장(공간·시간 판매)은 안 연다 —
      `ScheduledAnnouncer` 는 개방 루프라 이송이 없고, 그래서 오더의 `con_loc` 이 런 중에 바뀌지 않는다.
    """
    prof, built = _build(load, seed)
    mbt = mb.MultiBlockTerminal(
        {b: ts.ensure_time_ledger(_sim(prof, s)) for b, s in built["scenarios"].items()},
        extra_review_epochs=ts.admission_epochs(ts.OBS_24H))
    ann = ts.ScheduledAnnouncer(built["schedule"], lead_s=1800.0, end_s=built["sim_end_s"])
    rt = exc = None
    if policy == NET:
        if net is None:
            raise ValueError("v5net 은 net 이 필요하다")
        recs, ords = net_schedule_orders(built)
        pol, rt, exc = _net_driver(NetRuntime(net, None, None, float(built["sim_end_s"]),
                                              mbt=mbt, records=recs, orders=ords))
    elif policy == "sf_spt":
        pol = _rule_policy()
    else:
        #: ★조용히 규칙 궤적을 내놓지 않는다 — 전에는 터미널이 `--policy` 를 무시하고 `_rule_policy()` 를
        #:   못박아 써서, `--policy reference`(기본 `--mode both`) 가 "reference 라고 이름 붙은 규칙 궤적" 을
        #:   냈다. reference·first 는 순차 규약(`_serve_policy`) 이라 `sim.close_decision()` 을 직접 부르는데
        #:   조정자 루프의 review/park 와 맞물린 적이 없다 — 옮기기 전에는 크게 실패하는 쪽이 옳다.
        raise ValueError(f"터미널 모드는 sf_spt·{NET} 만 된다 — 받은 값 {policy!r}. "
                         f"reference·first 는 `--mode block` 으로 굴린다.")
    t0 = time.perf_counter()
    out = mbt.run(pol, review_fn=ann.review)
    secs = time.perf_counter() - t0
    turns = sorted(mbt.ledger.a_to_o_samples_s(ts.OBS_24H.observe_s))
    truck_ids = {e["job_id"] for e in built["schedule"]}
    # ★두꺼운 정답 (YR-327 검증 지적) — 21블록 규모에서도 투입 원장·locked·오더 전열·장부 적분을 대조할 수 있게.
    #   배열판은 v5 를 살아 있는 채로 부르지 못하는 세션이 많아(WSL 80초), 이 세 가지가 정답에 없으면 구조적으로
    #   대조가 불가능했다. 여기 넣어 두면 배열 쪽 시험은 JSON 만 읽고도 같은 항목을 볼 수 있다.
    blocks = {}
    for b, sim in mbt.blocks.items():
        d = _block_dump(b, sim)
        d["jobs"] = _job_rows(sim)
        d["ledger"] = _ledger_areas(sim)
        blocks[b] = d
    ann_rows = []
    for row in ann.ledger:
        r = {"t": row["t"], "event": row["event"], "job_id": row.get("job_id")}
        if row["event"] == "ADMIT":
            r.update({"block": row["block"], "flow": row["flow"], "arrival_s": row["arrival_s"]})
        elif row["event"] == "SKIP":
            r["reason"] = row["reason"]
        ann_rows.append(r)
    d = {
        "mode": "terminal", "load": load, "seed": seed, "wall_s": round(secs, 3),
        "terminal_total": out["terminal_total"], "route_cost_s": out["route_cost_s"],
        "end": out["end"], "admitted": ann.n_admitted,
        "n_turns": len(turns), "turn_sum_s": round(sum(turns), 6),
        "turn_samples_s": [round(t, 6) for t in turns],
        "totals": out["totals"],
        "ann_ledger": ann_rows,
        "locked": {j: r.locked for j, r in sorted(mbt.ledger.records.items()) if j in truck_ids},
        "deferrals": time_sell.deferral_ledger(mbt),
        "blocks": blocks,
    }
    if rt is not None:
        d.update({"policy": policy, "info_level": LVL.name, "bridge_end_s": float(built["sim_end_s"]),
                  "n_net_selects": rt.n_select, "crane_actions": dict(sorted(rt.crane_actions.items())),
                  "policy_exceptions": exc["n"], "net_hidden": int(rt.policy.hidden)})
    return d


def block_end_s(scn) -> float:
    """블록 단독 무대의 `MarketBridge.end_s` 대역 — 지평 + 배수창 (`world.end_s` 와 다른 값이다)."""
    return float(scn.horizon_s) + float(scn.drain_window_s)


def run_block(load: int, seed: int, block: str | None, policy: str = "sf_spt",
              level: InformationLevel = LVL, *, net=None, on_select=None) -> dict:
    """블록 하나를 **엔진만으로** — 다중블록 루프 없이 결정 루프를 직접 돈다.

    `net` 이 있으면 `--policy v5net` 경로다 (v5 `BlockPolicy` + `CraneActor`).
    """
    prof, built = _build(load, seed)
    scns = built["scenarios"]
    bid = block or max(scns, key=lambda b: len(scns[b].jobs))   # 기본: 일이 제일 많은 블록
    s = _sim(prof, scns[bid], level)
    end_s = block_end_s(scns[bid])
    rt = exc = None
    if policy == NET:
        #: ★명단(`built["schedule"]`)의 `arrival_s` = **통지된 예정**을 블록 요약 칸 5 에 쓴다 —
        #:   실현 게이트인을 쓰면 정보 경계가 샌다 (`net_terminal_inputs` 머리말 ★).
        sched = {e["job_id"]: float(e["arrival_s"]) for e in built["schedule"] if e["block"] == bid}
        pol, rt, exc = make_policy(policy, level, sim=s, end_s=end_s, block=bid, net=net,
                                   on_select=on_select,
                                   reserve_of=(sched.get if sched else None))
    else:
        pol = make_policy(policy, level)
    t0 = time.perf_counter()
    n_dec = n_rev = 0
    while True:
        out = s.run_until_decision()
        if out is None:
            break
        if isinstance(out, eng.ReviewEpoch):
            n_rev += 1
            continue
        n_dec += 1
        pol(s, out)
    secs = time.perf_counter() - t0
    d = _block_dump(bid, s)
    d.update({"mode": "block", "load": load, "seed": seed, "wall_s": round(secs, 3),
              "n_decisions": n_dec, "n_review_epochs": n_rev,
              "scenario_jobs": len(scns[bid].jobs),
              "policy": policy, "info_level": level.name,
              "bridge_end_s": end_s})
    if rt is not None:
        #: 학습 정책망 궤적의 추가 증거 — 망 호출 수(= 크레인-결정 수)·행동 종류별·예외 계수
        d.update({"n_net_selects": rt.n_select, "crane_actions": dict(sorted(rt.crane_actions.items())),
                  "policy_exceptions": exc["n"], "net_hidden": int(rt.policy.hidden)})
    return d


def block_truth_name(block: str, load: int, seed: int, policy: str = "sf_spt", level=LVL,
                     tag: str | None = None) -> str:
    """결과 파일 이름 — sf_spt·PRE_ADVICE 조합은 기존 이름 그대로 (block_Y01_load30_seed9900777.json).

    `tag` 는 학습 정책망의 가중치 표식(`net_tag`) 이다 — 다른 망의 궤적이 같은 이름을 못 쓰게 한다.
    """
    suffix = "" if (policy == "sf_spt" and level == InformationLevel.PRE_ADVICE) else f"_{policy}_{level.name}"
    if tag:
        suffix += f"_{tag}"
    return f"block_{block}_load{load}_seed{seed}{suffix}.json"


def net_weights_name(block: str, load: int, seed: int, level=LVL, tag: str = "") -> str:
    """정답 JSON 옆에 남기는 가중치 파일 이름 — 배열 쪽 시험이 같은 수를 싣는다."""
    return block_truth_name(block, load, seed, NET, level, tag)[:-len(".json")] + ".npz"


def terminal_truth_name(load: int, seed: int, policy: str = "sf_spt", level=LVL,
                        tag: str | None = None) -> str:
    """터미널 결과 파일 이름 — 규칙(sf_spt) 은 기존 이름 그대로 (terminal_load30_seed9900777.json)."""
    suffix = "" if policy == "sf_spt" else f"_{policy}_{level.name}" + (f"_{tag}" if tag else "")
    return f"terminal_load{load}_seed{seed}{suffix}.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="v5 정답 궤적 덤프")
    ap.add_argument("--load", type=int, default=30)
    ap.add_argument("--seed", type=int, default=9_900_777)
    ap.add_argument("--mode", choices=("terminal", "block", "both"), default="both")
    ap.add_argument("--block", default=None, help="block 모드에서 굴릴 블록 (기본: 일 제일 많은 곳)")
    ap.add_argument("--out", default="outputs/v6/ground_truth")
    ap.add_argument("--policy", choices=POLICIES, default="sf_spt",
                    help="결정 규칙 (머리말) — terminal 모드는 sf_spt·v5net 만, block 모드는 넷 다")
    ap.add_argument("--info-level", default="PRE_ADVICE", choices=[m.name for m in InformationLevel],
                    help="block 모드의 정보수준 (InformationLevel 이름)")
    ap.add_argument("--ckpt", default=None, help="--policy v5net 이 쓸 학습된 체크포인트 (.pt · 없으면 고정 시드 무작위)")
    ap.add_argument("--net-seed", type=int, default=NET_SEED, help="--policy v5net 무작위 초기화 씨앗")
    ap.add_argument("--hidden", type=int, default=NET_HIDDEN, help="--policy v5net 은닉 폭 (v5 기본 64)")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    level = InformationLevel[a.info_level]
    net = tag = None
    if a.policy == NET:
        net = build_net(ckpt=a.ckpt, seed=a.net_seed, hidden=a.hidden)
        tag = net_tag(a.ckpt, a.net_seed, a.hidden)

    if a.mode in ("terminal", "both"):
        r = run_terminal(a.load, a.seed, a.policy, net=net)
        if net is not None:
            #: ★가중치를 궤적 **옆에** 남긴다 (블록 모드와 같은 규약) — 배열 쪽이 torch 초기화 난수에
            #:   의존하지 않고 같은 수를 싣는다
            wp = out / (terminal_truth_name(a.load, a.seed, a.policy, level, tag)[:-len(".json")] + ".npz")
            save_net_npz(net, wp)
            r.update({"net_tag": tag, "net_weights": wp.name, "net_ckpt": a.ckpt,
                      "net_seed": (None if a.ckpt else a.net_seed)})
        p = out / terminal_truth_name(a.load, a.seed, a.policy, level, tag)
        p.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        ev = sum(b["n_events"] for b in r["blocks"].values())
        print(f"■ terminal  load={a.load} seed={a.seed}  정책 {a.policy}  {r['wall_s']}s  "
              f"비용 {r['terminal_total']:,.0f}  승인 {r['admitted']}  턴 {r['n_turns']}  "
              f"사건 {ev}  → {p}")
        if net is not None:
            print(f"   망 {r['net_tag']} 은닉 {r['net_hidden']}  망 호출 {r['n_net_selects']}  "
                  f"행동 {r['crane_actions']}  예외 {r['policy_exceptions']}  가중치 → {r['net_weights']}")
    if a.mode in ("block", "both"):
        r = run_block(a.load, a.seed, a.block, a.policy, level, net=net)
        if net is not None:
            #: ★가중치를 궤적 **옆에** 남긴다 — 배열 쪽이 torch 초기화 난수에 의존하지 않고 같은 수를 싣는다
            wp = out / net_weights_name(r["block"], a.load, a.seed, level, tag)
            save_net_npz(net, wp)
            r.update({"net_tag": tag, "net_weights": wp.name, "net_ckpt": a.ckpt,
                      "net_seed": (None if a.ckpt else a.net_seed)})
        p = out / block_truth_name(r["block"], a.load, a.seed, a.policy, level, tag)
        p.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        print(f"■ block {r['block']}  정책 {a.policy}/{level.name}  일 {r['n_jobs']}  크레인 {r['n_cranes']}  {r['wall_s']}s  "
              f"결정 {r['n_decisions']}  사건 {r['n_events']}  해시 {r['event_hash']}  "
              f"재처리 {r['kpis']['rehandle_count']}  미완 {r['unfinished']}  → {p}")
        if net is not None:
            print(f"   망 {r['net_tag']} 은닉 {r['net_hidden']}  망 호출 {r['n_net_selects']}  "
                  f"행동 {r['crane_actions']}  예외 {r['policy_exceptions']}  가중치 → {r['net_weights']}")
        kinds: dict[str, int] = {}
        for _, k, _ in r["events"]:
            kinds[k] = kinds.get(k, 0) + 1
        print("   사건 종류별:", dict(sorted(kinds.items(), key=lambda kv: -kv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
