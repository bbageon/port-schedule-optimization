"""YR-151 0B — **반사실** 판매 신호 예선 (사전등록 + 3·4차 추록 동결본 이행).

계약(동결): 주 무대 w3-L100·개발 시드 5(+10M 대역)·smoke 5런 × 8건(4분위별 2·
job_id 사전순) = 표본 40 · 짝 런 = (전 KEEP) vs (그 제안 1건만 집행) 같은 시드 ·
실현 이득 G = Φ_base − Φ_treat · 판정 = 런(군집) 단위 — 축1: 런별 ΣG > 0 이 5/5
(부호검정 p=1/32) · 축2: 런별 Spearman(계산 −ΔJ, G) 중앙값 ≥ 0.3 · 하드 가드:
처리량 감소 표본 = 실패 계수·장부/불변식 위반 = 실격·(불가+실격) > 10/40 = 무효.
인가: authorize-next allowed=true (966c20a). 성능 주장 없음 — GO/STOP 만.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

import torch

from ..integrated.baselines import _apply, _wait_of
from ..integrated.candidates import CandidateGenerator
from ..integrated.multiblock import MultiBlockTerminal
from ..integrated.policy_config import ADOPTED_C0_GUARD
from ..integrated.profiles import build_h21_profile
from ..integrated.repro import code_dirty
from ..integrated.sell_review import (ANNOUNCE_LEAD_S, UnifiedSellOrchestrator)
from ..integrated.terminal_stream import (ObservationContract,
                                          TerminalStreamParams,
                                          WipAdmissionController,
                                          WIP_FILL_SPAN_S, admission_epochs,
                                          build_fixed_wip,
                                          hotspot_rotation)
from ..integrated.scenario_gen import GATE_BLOCK_MAX_S
from ..integrated.time_sell import try_time_sell
from ..integrated.transfer_head import (PpoSellPolicy, TransferActor,
                                        TransferCritic)
from ..integrated.yard_layout import terminal_layout
from .yr088_joint_rl import LEVEL
from .yr149_load_cells import _sim_from
from .yr150_h21_pilot import _git, _sha256
from .yr151_transfer_ppo import (AdoptedExecFleet, load_adopted_execution_head,
                                 load_kf, phi_terminal)
from .yr157_band_qual import (N_HOTSPOT, SEED as BAND_SEED, hotspot_seed,
                              pair_seed)

OUT = Path("outputs/reports/yr151_0b_counterfactual")
PREREG = Path(".claude/docs/strategy-history/"
              "2026-08-09-YR-151-0B-반사실-재설계-사전등록.md")
W, LOAD, MASTER = 3.0, 100, 150          # 동결 — 주 무대(짝 체계 정본)
DEV_BASE, DEV_REPS = 10_000_000, 5       # 동결 — 개발 대역
NET_INIT = 7_000_000                     # 동결 — 미학습 초기화
PER_RUN, RHO_MIN, MAX_BAD = 8, 0.3, 10   # 동결 — 런당 표본·구분력 임계·무효 상한
WIP_TOL_FRAC = 0.05


def dev_seed(rep: int) -> int:
    return pair_seed(W, 0) + DEV_BASE + rep * 1_000_000


def _env(rep: int):
    obs = ObservationContract()
    layout = terminal_layout()
    hs = hotspot_rotation(layout, hotspot_seed(W, 0), N_HOTSPOT)
    params = TerminalStreamParams(load_4h=LOAD, hotspot_blocks=hs, hotspot_weight=W)
    built = build_fixed_wip(build_h21_profile(), dev_seed(rep), wip_target=LOAD,
                            obs=obs, layout=layout, params=params,
                            background_seed=BAND_SEED + DEV_BASE + rep * 1_000_000,
                            master_load=MASTER)
    mbt = MultiBlockTerminal({b: _sim_from(s) for b, s in built["scenarios"].items()},
                             extra_review_epochs=admission_epochs(obs))
    ctrl = WipAdmissionController(built["pool"], wip_target=LOAD,
                                  lead_s=ANNOUNCE_LEAD_S, end_s=obs.observe_s)
    return obs, layout, built, mbt, ctrl


def _state_digest(mbt, ctrl) -> str:
    """Hash the observable/physical state immediately before a paired action."""
    blocks = []
    for bid, sim in sorted(mbt.blocks.items()):
        jobs = []
        for jid, job in sorted(sim.jobs.items()):
            jobs.append((jid, str(job.status), job.actual_gate_in,
                         job.actual_block_arrival, job.estimated_block_arrival,
                         job.provided_eta, str(job.flow)))
        cranes = []
        for cid in sorted(sim.fleet.ids):
            yc = sim.fleet.get(cid)
            cranes.append((cid, str(yc.state.status), yc.state.position_bay,
                           getattr(yc.state, "current_job_id", None)))
        blocks.append((bid, sim.clock, jobs, cranes))
    owners = [(jid, rec.owner, rec.version, rec.transfer_count)
              for jid, rec in sorted(mbt.ledger.records.items())]
    payload = {"now": mbt.now, "blocks": blocks, "owners": owners,
               "admission": {"cursor": ctrl.cursor,
                             "n_admitted": ctrl.n_admitted}}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                     default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def _run(rep: int, review_extra=None):
    """공통 러너 — 채택 PPO 실행 + 60초 투입 + (옵션) 추가 review 훅.

    YR-160 본체 API: 전역 없음 — config 를 fleet·generator 에 명시 주입한다.
    """
    obs, layout, built, mbt, ctrl = _env(rep)
    exec_actor, exec_norm = load_adopted_execution_head()
    fleet = AdoptedExecFleet(exec_actor, exec_norm, config=ADOPTED_C0_GUARD)
    gens: dict[int, CandidateGenerator] = {}

    def exec_policy(sim, dp):
        g = gens.setdefault(id(sim),
                            CandidateGenerator(config=ADOPTED_C0_GUARD))
        gb = {c: g.generate(sim, c, LEVEL) for c in dp.crane_ids}
        _apply(sim, fleet.get(sim).decide(sim, dp, gb))   # 예외 = 즉시 실격(전파)

    def review(mbt_, t):
        ctrl.review(mbt_, t)
        if review_extra is not None:
            review_extra(mbt_, t, ctrl)

    mbt.run(exec_policy, review_fn=review)
    n_rows = sum(len(getattr(s, "time_ledger").records)
                 for s in mbt.blocks.values()
                 if getattr(s, "time_ledger", None) is not None)
    if n_rows != len(built["fill"]) + ctrl.n_admitted:
        raise RuntimeError("장부 불일치 — 런 실격")
    mbt.check_invariants()
    done = sum(1 for s in mbt.blocks.values()
               if getattr(s, "time_ledger", None) is not None
               for r in s.time_ledger.records.values() if r.gate_out is not None)
    hold_lo = max(obs.warmup_s, WIP_FILL_SPAN_S)
    hold_hi = obs.observe_s - (ANNOUNCE_LEAD_S + GATE_BLOCK_MAX_S)
    held = [e["inside"] + e["pipeline"] + e["admitted"]
            for e in ctrl.ledger
            if e["event"] == "EPOCH" and hold_lo <= e["t"] <= hold_hi]
    wip_ok = bool(held) and all(
        LOAD * (1.0 - WIP_TOL_FRAC) <= value <= LOAD * (1.0 + WIP_TOL_FRAC)
        for value in held)
    return {"phi": phi_terminal(mbt, obs.observe_s), "n_done": done,
            "wip_contract_ok": wip_ok,
            "mbt": mbt, "layout": layout, "obs": obs}


def smoke(rep: int) -> Path:
    """shadow smoke 1런 — would-commit 원장 수집 (표본 원천)."""
    obs, layout, built, mbt, ctrl = _env(rep)
    torch.manual_seed(NET_INIT)
    pol = PpoSellPolicy(TransferActor(), TransferCritic(), mode="shadow",
                        sample=True, seed=NET_INIT + rep, layout=layout)
    orch = UnifiedSellOrchestrator(pol, layout, load_kf(), dry_run=True)
    exec_actor, exec_norm = load_adopted_execution_head()
    fleet = AdoptedExecFleet(exec_actor, exec_norm, config=ADOPTED_C0_GUARD)
    gens: dict[int, CandidateGenerator] = {}

    def exec_policy(sim, dp):
        g = gens.setdefault(id(sim),
                            CandidateGenerator(config=ADOPTED_C0_GUARD))
        gb = {c: g.generate(sim, c, LEVEL) for c in dp.crane_ids}
        _apply(sim, fleet.get(sim).decide(sim, dp, gb))

    def review(mbt_, t):
        ctrl.review(mbt_, t)
        orch.review(mbt_, t)

    mbt.run(exec_policy, review_fn=review)
    wc = [{"t": e["t"], "src": e["src"], "job_id": e["job_id"], "axis": e["axis"],
           "dst": e.get("dst"), "delta_j": e["delta_j"]}
          for e in orch.ledger if e["decision"] == "DRY_WOULD_COMMIT"]
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"smoke_rep{rep}.json"
    p.write_text(json.dumps({"rep": rep, "would": wc}, ensure_ascii=False,
                            indent=1), encoding="utf-8")
    print(json.dumps({"rep": rep, "n_would": len(wc)}, ensure_ascii=False))
    return p


def sample() -> Path:
    """동결 규칙 표본 추출 — 런별 이득 4분위마다 2건(job_id 사전순) = 8건 × 5런."""
    samples = []
    for rep in range(DEV_REPS):
        wc = json.loads((OUT / f"smoke_rep{rep}.json").read_text(
            encoding="utf-8"))["would"]
        # Repeated reviews can quote the same coordinate more than once.  A
        # counterfactual sample is one exact decision coordinate, not a job ID.
        unique = {}
        for e in wc:
            key = (round(e["t"], 6), e["src"], e["job_id"], e["axis"], e.get("dst"))
            unique.setdefault(key, e)
        wc = sorted(unique.values(), key=lambda e: (e["delta_j"], e["job_id"],
                                                     e["t"], str(e.get("dst"))))
        n = len(wc)
        if n < PER_RUN:
            raise RuntimeError(f"rep {rep}: unique proposals {n} < {PER_RUN}")
        selected = []
        for q in range(4):
            seg = wc[q * n // 4:(q + 1) * n // 4]
            if len(seg) < 2:
                raise RuntimeError(f"rep {rep}: quartile {q} has <2 proposals")
            seg = sorted(seg, key=lambda e: e["job_id"])[:2]
            for e in seg:
                selected.append({**e, "rep": rep, "quartile": q})
        if len(selected) != PER_RUN:
            raise RuntimeError(f"rep {rep}: selected {len(selected)} != {PER_RUN}")
        samples.extend(selected)
    if len(samples) != DEV_REPS * PER_RUN:
        raise RuntimeError("counterfactual manifest must contain exactly 40 samples")
    p = OUT / "samples.json"
    p.write_text(json.dumps({"n": len(samples), "samples": samples},
                            ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"n_samples": len(samples)}, ensure_ascii=False))
    return p


def pair(idx: int) -> Path:
    """표본 1건의 반사실 짝 — 기준(전 KEEP) vs 처치(그 1건만 집행)."""
    s = json.loads((OUT / "samples.json").read_text(encoding="utf-8"))["samples"][idx]
    rep, t_star, jid = s["rep"], s["t"], s["job_id"]
    base_state = {}

    def capture_base(mbt_, t, ctrl):
        if "digest" not in base_state and abs(t - t_star) < 1e-6:
            base_state["digest"] = _state_digest(mbt_, ctrl)

    base = _run(rep, review_extra=capture_base)             # 기준 세계
    if "digest" not in base_state:
        raise RuntimeError("paired decision epoch missing in KEEP world")

    state = {"tried": False, "ok": False, "equal": False}

    def treat(mbt_, t, ctrl):
        if not state["tried"] and abs(t - t_star) < 1e-6:
            state["tried"] = True
            digest = _state_digest(mbt_, ctrl)
            state["equal"] = digest == base_state["digest"]
            if not state["equal"]:
                raise RuntimeError("paired worlds differ before treatment")
            rec = mbt_.ledger.records.get(jid)
            if rec is None or rec.owner != s["src"]:
                raise RuntimeError("proposal owner/source changed before treatment")
            layout = terminal_layout()
            if s["axis"] == "TIME":
                state["ok"] = bool(try_time_sell(mbt_, jid, delta_s=900.0,
                                                 max_deferrals=1))
            else:
                state["ok"] = bool(mbt_.try_pre_gate_transfer(
                    jid, s["dst"], travel_s=layout.gate_to_block_s(s["dst"]),
                    route_delta_s=layout.pre_gate_route_delta_s(s["src"], s["dst"])))

    tr = _run(rep, review_extra=treat)                      # 처치 세계
    res = {"idx": idx, "rep": rep, "axis": s["axis"], "quartile": s["quartile"],
           "delta_j": s["delta_j"], "executed": state["ok"],
           "paired_state_equal": state["equal"],
           "gain_realized": round(base["phi"] - tr["phi"], 6),
           "throughput_base": base["n_done"], "throughput_treat": tr["n_done"],
           "throughput_ok": tr["n_done"] >= base["n_done"],
           "wip_contract_base": base["wip_contract_ok"],
           "wip_contract_treat": tr["wip_contract_ok"]}
    p = OUT / f"pair_{idx:02d}.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False))
    return p


def _spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx > 0 and vy > 0 else 0.0


def verdict() -> dict:
    manifest = json.loads((OUT / "samples.json").read_text(encoding="utf-8"))
    if manifest.get("n") != DEV_REPS * PER_RUN:
        raise RuntimeError("invalid counterfactual sample manifest")
    pairs_by_idx = {
        i: json.loads((OUT / f"pair_{i:02d}.json").read_text(encoding="utf-8"))
        for i in range(DEV_REPS * PER_RUN)
        if (OUT / f"pair_{i:02d}.json").exists()
    }
    pairs = []
    for i, s in enumerate(manifest["samples"]):
        p = pairs_by_idx.get(i)
        if p is None:
            p = {"idx": i, "rep": s["rep"], "axis": s["axis"],
                 "quartile": s["quartile"], "delta_j": s["delta_j"],
                 "executed": False, "paired_state_equal": False,
                 "gain_realized": 0.0, "throughput_ok": False,
                 "wip_contract_base": False, "wip_contract_treat": False}
        pairs.append(p)
    bad = [p for p in pairs
           if (not p["executed"] or not p.get("paired_state_equal", False)
               or not p.get("wip_contract_base", False)
               or not p.get("wip_contract_treat", False))]
    ok = [p for p in pairs if p not in bad]
    # 하드 가드: 처리량 감소 표본은 이득 판정에서 실패로 계수(이득값을 0 이하 취급)
    eff = [{**p, "g_eff": (-1e-9 if p in bad else
                            (p["gain_realized"] if p["throughput_ok"]
                             else min(0.0, p["gain_realized"]) - 1e-9))}
           for p in pairs]
    by_rep = {}
    for p in eff:
        by_rep.setdefault(p["rep"], []).append(p)
    run_sums = {r: sum(p["g_eff"] for p in ps) for r, ps in by_rep.items()}
    axis1 = len(run_sums) == DEV_REPS and all(v > 0 for v in run_sums.values())
    rhos = {r: _spearman([-p["delta_j"] for p in ps], [p["g_eff"] for p in ps])
            for r, ps in by_rep.items() if len(ps) >= 3}
    med_rho = median(rhos.values()) if rhos else 0.0
    axis2 = med_rho >= RHO_MIN
    invalid = len(bad) > MAX_BAD
    v = {"signal_go": bool(axis1 and axis2 and not invalid),
         "invalid": invalid, "axis1_run_sums_all_positive": axis1,
         "axis2_median_run_rho": round(med_rho, 4), "rho_min": RHO_MIN,
         "run_sums": {str(k): round(vv, 4) for k, vv in run_sums.items()},
         "run_rhos": {str(k): round(vv, 4) for k, vv in rhos.items()},
         "n_pairs": len(pairs), "n_invalid_or_unexecuted": len(bad),
         "n_throughput_fail": sum(1 for p in ok if not p["throughput_ok"]),
         "median_gain_all": round(median([p["gain_realized"] for p in ok]), 6)
                            if ok else None,
         "note": "반사실 실현 이득 판정 — 런(군집) 단위. 성능 주장 없음. "
                 "무오차 조건 한정(YR-162B 전)."}
    dirty = bool(code_dirty())
    res = {"task": "YR-151-0B-counterfactual",
           "runtime": {"commit": _git("rev-parse", "HEAD"), "git_dirty": dirty,
                       "prereg_file": str(PREREG),
                       "prereg_sha256": _sha256(PREREG) if PREREG.exists() else None,
                       "stage": f"w{W}-L{LOAD}·master{MASTER}",
                       "seeds": {"dev": [dev_seed(r) for r in range(DEV_REPS)],
                                 "net_init": NET_INIT}},
           "verdict": v}
    p = OUT / "ob_counterfactual.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "ob_counterfactual.json.sha256").write_text(_sha256(p) + "\n",
                                                       encoding="utf-8")
    print(json.dumps({"verdict": v, "dirty": dirty}, ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=None)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--pair", type=int, default=None)
    ap.add_argument("--verdict", action="store_true")
    a = ap.parse_args()
    if a.smoke is not None:
        smoke(a.smoke)
    elif a.sample:
        sample()
    elif a.pair is not None:
        pair(a.pair)
    elif a.verdict:
        verdict()
    print("DONE")
