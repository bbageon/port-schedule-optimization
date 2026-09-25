"""v5(=v6 사본) 세계를 규칙으로 굴려 **정답 궤적**을 남긴다 ([[YR-327]]).

배열 세계가 옮겨질 때마다 이 궤적과 대조한다 — 사건 하나하나의 (시각·종류·대상)이
같은 순서로 나와야 하고, 비용 항목·턴타임·재처리 수가 같아야 한다.

■ 두 가지 모드
  terminal  21블록 전체를 다중블록 루프로 굴린다 (게이트·이송 포함). 최종 목표.
  block     블록 **하나**를 엔진만으로 굴린다 (이송 없음). 1번 조각(단일 블록 엔진)의
            동등성 시험용. 다중블록 문맥이 없으니 게이트 승인은 시나리오에 적힌 대로.

    PYTHONPATH=src python scripts/v6/dump_ground_truth.py --load 30 --seed 9900777
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from yard_rl.v6.world.domain.enums import InformationLevel
from yard_rl.v6.world.integrated import (baselines as bl, candidates as cd,
                                          engine as eng, multiblock as mb,
                                          policy_config as pc, profiles as pr,
                                          terminal_stream as ts, yard_layout as yl)

LVL = InformationLevel.PRE_ADVICE


def _build(load: int, seed: int):
    prof, layout = pr.build_h21_profile(), yl.terminal_layout()
    built = ts.build_diurnal(prof, seed, obs=ts.OBS_24H, layout=layout,
                             params=ts.TerminalStreamParams(load_4h=load),
                             day_total=load, background_seed=seed)
    return prof, built


def _sim(prof, scn):
    s = eng.TerminalSimulator(prof, scn, check_invariants=True)
    s.info_level = LVL
    return s


def _rule_policy():
    """규칙(작업 우선·최단작업) — v5 판정의 바닥. 망 없음, 난수 없음."""
    gens: dict[int, object] = {}
    pol = bl.ResolverPolicy(bl.ServiceFirstSPTPreference(), "SF")

    def exec_policy(sim, dp):
        g = gens.setdefault(id(sim), cd.CandidateGenerator(config=pc.LEGACY_DEFAULT))
        gb = {c: g.generate(sim, c, LVL) for c in dp.crane_ids}
        bl._apply(sim, pol.decide(sim, dp, gb))
    return exec_policy


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


def run_terminal(load: int, seed: int) -> dict:
    prof, built = _build(load, seed)
    mbt = mb.MultiBlockTerminal(
        {b: ts.ensure_time_ledger(_sim(prof, s)) for b, s in built["scenarios"].items()},
        extra_review_epochs=ts.admission_epochs(ts.OBS_24H))
    ann = ts.ScheduledAnnouncer(built["schedule"], lead_s=1800.0, end_s=built["sim_end_s"])
    t0 = time.perf_counter()
    out = mbt.run(_rule_policy(), review_fn=ann.review)
    secs = time.perf_counter() - t0
    turns = sorted(mbt.ledger.a_to_o_samples_s(ts.OBS_24H.observe_s))
    return {
        "mode": "terminal", "load": load, "seed": seed, "wall_s": round(secs, 3),
        "terminal_total": out["terminal_total"], "route_cost_s": out["route_cost_s"],
        "end": out["end"], "admitted": ann.n_admitted,
        "n_turns": len(turns), "turn_sum_s": round(sum(turns), 6),
        "turn_samples_s": [round(t, 6) for t in turns],
        "blocks": {b: _block_dump(b, s) for b, s in mbt.blocks.items()},
    }


def run_block(load: int, seed: int, block: str | None) -> dict:
    """블록 하나를 **엔진만으로** — 다중블록 루프 없이 결정 루프를 직접 돈다."""
    prof, built = _build(load, seed)
    scns = built["scenarios"]
    bid = block or max(scns, key=lambda b: len(scns[b].jobs))   # 기본: 일이 제일 많은 블록
    s = _sim(prof, scns[bid])
    pol = _rule_policy()
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
              "scenario_jobs": len(scns[bid].jobs)})
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="v5 정답 궤적 덤프")
    ap.add_argument("--load", type=int, default=30)
    ap.add_argument("--seed", type=int, default=9_900_777)
    ap.add_argument("--mode", choices=("terminal", "block", "both"), default="both")
    ap.add_argument("--block", default=None, help="block 모드에서 굴릴 블록 (기본: 일 제일 많은 곳)")
    ap.add_argument("--out", default="outputs/v6/ground_truth")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    if a.mode in ("terminal", "both"):
        r = run_terminal(a.load, a.seed)
        p = out / f"terminal_load{a.load}_seed{a.seed}.json"
        p.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        ev = sum(b["n_events"] for b in r["blocks"].values())
        print(f"■ terminal  load={a.load} seed={a.seed}  {r['wall_s']}s  "
              f"비용 {r['terminal_total']:,.0f}  승인 {r['admitted']}  턴 {r['n_turns']}  "
              f"사건 {ev}  → {p}")
    if a.mode in ("block", "both"):
        r = run_block(a.load, a.seed, a.block)
        p = out / f"block_{r['block']}_load{a.load}_seed{a.seed}.json"
        p.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        print(f"■ block {r['block']}  일 {r['n_jobs']}  크레인 {r['n_cranes']}  {r['wall_s']}s  "
              f"결정 {r['n_decisions']}  사건 {r['n_events']}  해시 {r['event_hash']}  "
              f"재처리 {r['kpis']['rehandle_count']}  미완 {r['unfinished']}  → {p}")
        kinds: dict[str, int] = {}
        for _, k, _ in r["events"]:
            kinds[k] = kinds.get(k, 0) + 1
        print("   사건 종류별:", dict(sorted(kinds.items(), key=lambda kv: -kv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
