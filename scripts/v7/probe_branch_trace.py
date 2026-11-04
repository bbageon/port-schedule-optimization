"""Factual-branch vs main-trajectory decision traces in processing order (YR-332-b evidence).

    PYTHONPATH=src python scripts/v7/probe_branch_trace.py 1500 10800 8 outputs/reports/yr332_v7_scaffold/branch_trace_1500.json
"""
import json
import sys
from yard_rl.v7.crane import collect as C
from yard_rl.v7.crane import rollout as R
from yard_rl.v7.stage import episode as E
from yard_rl.v7.stage.rollout import RolloutBudget

LOAD = int(sys.argv[1])
H = float(sys.argv[2])
NLAB = int(sys.argv[3])
OUT = sys.argv[4]
SEED = 9_900_335
main_trace, start, box = [], {}, {}
orig_hook, orig_finish = C.CraneCollector.hook, C.CraneCollector.finish


def hook(self, mbt, market, orders, records):
    inner = orig_hook(self, mbt, market, orders, records)
    bmap = {id(s): b for b, s in mbt.blocks.items()}

    def on_crane(sim, dp, gb, assign):
        k = len(main_trace)
        for c in sorted(assign):
            ref = getattr(assign[c], "job_ref", None)
            main_trace.append((round(sim.clock, 6), bmap[id(sim)], c, getattr(ref, "job_id", None)))
        n = len(self.rows)
        out = inner(sim, dp, gb, assign)
        if len(self.rows) > n:
            start[list(self.rows)[-1]] = k
        return out
    return on_crane


def finish(self):
    out = orig_finish(self)
    box["c"] = self
    return out


C.CraneCollector.hook, C.CraneCollector.finish = hook, finish
R.TRACE_DECISIONS = True
res = E.run_episode(load=LOAD, seed=SEED, dispatcher="RL_CRANE",
                    crane_budget=RolloutBudget(max_labels=NLAB, identity_checks=0), horizon_s=H, workers=1)
rows = []
for r in box["c"].raw:
    t = r["t"]
    a = [e for e in main_trace[start[r["key"]]:] if e[0] < t + H]
    b = [e for e in r["trace_factual"] if e[0] < t + H]
    i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if i is None and len(a) != len(b):
        i = min(len(a), len(b))
    rows.append({"t": t, "block": r["key"][1], "crane": r["crane"], "main_n": len(a), "fact_n": len(b),
                 "first_diff_after_s": (None if i is None else round(min(a[i][0] if i < len(a) else 9e9,
                                                                          b[i][0] if i < len(b) else 9e9) - t, 1)),
                 "first_diff": (None if i is None else [a[i] if i < len(a) else None, b[i] if i < len(b) else None]),
                 "phi_factual": r["phi_factual"], "phi_alt": r["phi_alt"]})
out = {"load": LOAD, "seed": SEED, "horizon_s": H, "labels": NLAB, "crane_stats": res.crane_stats, "branches": rows}
open(OUT, "w").write(json.dumps(out, ensure_ascii=False, indent=1))
for x in rows:
    print(x["t"], x["block"], x["main_n"], x["fact_n"], x["first_diff_after_s"], x["first_diff"])
