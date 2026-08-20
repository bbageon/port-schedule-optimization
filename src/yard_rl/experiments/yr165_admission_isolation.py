"""YR-165 — 공간 판매 "전역 효과" 분리검증 (투입 고정 대조·사전등록 동결본 이행).

가설(38차 감사): YR-164 의 T_other(+16.52)는 물리적 파급이 아니라 **중앙 투입
컨트롤러**가 만든 장치 효과다 — 판매로 진출 시각이 바뀌면 전역 재공량 계수가
달라지고 단일 pool 커서가 어긋나 **이후 전 블록의 트럭 명단이 통째로 바뀐다**.

검사: 기준 세계의 **투입 계획(epoch·job_id)** 을 그대로 재생하는 고정 투입기로
처치 세계를 돌린다 → 두 세계의 미래 트럭이 완전히 같아진다. 이때
  A) |T_other| < 1e-9 (전 표본) → 장치 효과 확정
  B) 비영 잔존 → 투입기 외 숨은 결합 (원인 규명 선행)
성능 주장 없음 — 장치 타당성 검사다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..integrated.baselines import _apply
from ..integrated.candidates import CandidateGenerator
from ..integrated.multiblock import MultiBlockTerminal, TransferError
from ..integrated.policy_config import ADOPTED_C0_GUARD
from ..integrated.profiles import build_h21_profile
from ..integrated.repro import code_dirty
from ..integrated.sell_review import (ANNOUNCE_LEAD_S, UnifiedSellOrchestrator)
from ..integrated.terminal_stream import (WIP_ADMISSION_PERIOD_S,
                                          ObservationContract,
                                          TerminalStreamParams,
                                          WipAdmissionController,
                                          _job_from_entry, admission_epochs,
                                          build_fixed_wip, hotspot_rotation)
from ..integrated.time_grid import on_grid
from ..v1.ppo_policy import PpoSellPolicy, TransferActor, TransferCritic
from ..integrated.yard_layout import terminal_layout
from .yr088_joint_rl import LEVEL
from .yr149_load_cells import _sim_from
from .yr150_h21_pilot import _git, _sha256
from .yr151_transfer_ppo import (AdoptedExecFleet, load_adopted_execution_head,
                                 load_kf, phi_terminal)
from .yr157_band_qual import (N_HOTSPOT, SEED as BAND_SEED, hotspot_seed,
                              pair_seed)
from .yr164_cost_decomposition import _phi_parts

OUT = Path("outputs/reports/yr165_admission_isolation")
PREREG = Path(".claude/docs/dashboard-task-specs/"
              "YR-165-space-sell-global-effect-isolation.md")
W, LOAD, MASTER = 3.0, 100, 150      # 동결 — 주 무대(짝 체계 정본)
ISO_BASE, ISO_REPS, PER_RUN = 20_000_000, 2, 4   # 동결 — 새 시드 대역·2런×4건
NET_INIT = 7_000_000


def iso_seed(rep: int) -> int:
    return pair_seed(W, 0) + ISO_BASE + rep * 1_000_000


class FrozenAdmission:
    """기준 세계의 투입 계획을 **그대로 재생** — 전역 재공량 계수를 보지 않는다."""

    def __init__(self, pool: list[dict], plan: list[dict], lead_s: float):
        by_id = {e["job_id"]: e for e in pool}
        self.by_t: dict[float, list[dict]] = {}
        for rec in plan:
            self.by_t.setdefault(round(rec["t"], 6), []).append(by_id[rec["job_id"]])
        self.lead_s = lead_s
        self.n_admitted = 0
        self.cursor = 0
        self.ledger: list[dict] = []

    def review(self, mbt, t: float) -> None:
        if not on_grid(t, WIP_ADMISSION_PERIOD_S):
            return
        for e in self.by_t.get(round(t, 6), []):
            notified = t + self.lead_s
            job = _job_from_entry(e, notified)
            try:
                mbt.admit_external_job(e["block"], job, gate_in_s=notified,
                                       travel_s=e["travel_s"])
                self.n_admitted += 1
                self.cursor += 1
                self.ledger.append({"t": t, "event": "ADMIT",
                                    "job_id": e["job_id"], "block": e["block"],
                                    "flow": e["flow"]})
            except TransferError as ex:
                self.ledger.append({"t": t, "event": "SKIP",
                                    "job_id": e["job_id"], "reason": str(ex)})
        self.ledger.append({"t": t, "event": "EPOCH", "frozen": True})


def _env(rep: int):
    obs = ObservationContract()
    layout = terminal_layout()
    hs = hotspot_rotation(layout, hotspot_seed(W, 0), N_HOTSPOT)
    params = TerminalStreamParams(load_4h=LOAD, hotspot_blocks=hs, hotspot_weight=W)
    built = build_fixed_wip(build_h21_profile(), iso_seed(rep), wip_target=LOAD,
                            obs=obs, layout=layout, params=params,
                            background_seed=BAND_SEED + ISO_BASE + rep * 1_000_000,
                            master_load=MASTER)
    mbt = MultiBlockTerminal({b: _sim_from(s) for b, s in built["scenarios"].items()},
                             extra_review_epochs=admission_epochs(obs))
    return obs, layout, built, mbt


def _run(rep: int, *, ctrl, sell=None, orch=None):
    """1 에피소드 — ctrl(투입기)·sell(판매 훅)·orch(shadow) 를 주입받는다."""
    obs, layout, built, mbt = _env(rep)
    exec_actor, exec_norm = load_adopted_execution_head()
    fleet = AdoptedExecFleet(exec_actor, exec_norm, config=ADOPTED_C0_GUARD)
    gens: dict[int, CandidateGenerator] = {}

    def exec_policy(sim, dp):
        g = gens.setdefault(id(sim), CandidateGenerator(config=ADOPTED_C0_GUARD))
        gb = {c: g.generate(sim, c, LEVEL) for c in dp.crane_ids}
        _apply(sim, fleet.get(sim).decide(sim, dp, gb))

    def review(mbt_, t):
        ctrl.review(mbt_, t)
        if orch is not None:
            orch.review(mbt_, t)
        if sell is not None:
            sell(mbt_, t)

    mbt.run(exec_policy, review_fn=review)
    n_rows = sum(len(s.time_ledger.records) for s in mbt.blocks.values()
                 if getattr(s, "time_ledger", None) is not None)
    if n_rows != len(built["fill"]) + ctrl.n_admitted:
        raise RuntimeError("장부 불일치")
    mbt.check_invariants()
    return {"mbt": mbt, "obs": obs, "built": built, "ctrl": ctrl,
            "phi": phi_terminal(mbt, obs.observe_s)}


def _fresh_ctrl(built, obs):
    return WipAdmissionController(built["pool"], wip_target=LOAD,
                                  lead_s=ANNOUNCE_LEAD_S, end_s=obs.observe_s)


def smoke(rep: int) -> Path:
    """새 시드 shadow smoke — 판매 후보 수집(투입은 정상)."""
    obs, layout, built, mbt = _env(rep)
    torch.manual_seed(NET_INIT)
    pol = PpoSellPolicy(TransferActor(), TransferCritic(), mode="shadow",
                        sample=True, seed=NET_INIT + rep, layout=layout)
    orch = UnifiedSellOrchestrator(pol, layout, load_kf(), dry_run=True)
    ctrl = _fresh_ctrl(built, obs)
    r = _run(rep, ctrl=ctrl, orch=orch)
    wc = [{"t": e["t"], "src": e["src"], "job_id": e["job_id"], "axis": e["axis"],
           "dst": e.get("dst"), "delta_j": e["delta_j"]}
          for e in orch.ledger
          if e["decision"] == "DRY_WOULD_COMMIT" and e["axis"] == "SPACE"]
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"smoke_rep{rep}.json"
    p.write_text(json.dumps({"rep": rep, "n_would": len(wc), "would": wc},
                            ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"rep": rep, "n_would": len(wc)}, ensure_ascii=False))
    return p


def sample() -> Path:
    """동결 규칙 — 런당 4분위 각 1건(job_id 사전순) = 8표본."""
    out = []
    for rep in range(ISO_REPS):
        wc = json.loads((OUT / f"smoke_rep{rep}.json").read_text(
            encoding="utf-8"))["would"]
        uniq = {}
        for e in wc:
            uniq.setdefault((round(e["t"], 6), e["src"], e["job_id"],
                             e.get("dst")), e)
        wc = sorted(uniq.values(), key=lambda e: (e["delta_j"], e["job_id"]))
        n = len(wc)
        for q in range(4):
            seg = wc[q * n // 4:(q + 1) * n // 4]
            if not seg:
                raise RuntimeError(f"rep {rep} quartile {q} 비어 있음")
            e = sorted(seg, key=lambda x: x["job_id"])[0]
            out.append({**e, "rep": rep, "quartile": q})
    if len(out) != ISO_REPS * PER_RUN:
        raise RuntimeError("표본 수 계약 위반")
    p = OUT / "samples.json"
    p.write_text(json.dumps({"n": len(out), "samples": out}, ensure_ascii=False,
                            indent=1), encoding="utf-8")
    print(json.dumps({"n_samples": len(out)}, ensure_ascii=False))
    return p


def pair(idx: int) -> Path:
    """표본 1건 — 기준(정상 투입) vs 처치(기준 투입 계획 **고정 재생** + 판매 1건)."""
    s = json.loads((OUT / "samples.json").read_text(encoding="utf-8"))["samples"][idx]
    rep, t_star, jid, src, dst = (s["rep"], s["t"], s["job_id"], s["src"], s["dst"])

    obs0, _, built0, _ = _env(rep)
    base = _run(rep, ctrl=_fresh_ctrl(built0, obs0))
    plan = [{"t": e["t"], "job_id": e["job_id"]}
            for e in base["ctrl"].ledger if e["event"] == "ADMIT"]
    base_parts = _phi_parts(base["mbt"], base["obs"].observe_s)

    st = {"ok": False, "tried": False}

    def sell(mbt_, t):
        if not st["tried"] and abs(t - t_star) < 1e-6:
            st["tried"] = True
            layout = terminal_layout()
            st["ok"] = bool(mbt_.try_pre_gate_transfer(
                jid, dst, travel_s=layout.gate_to_block_s(dst),
                route_delta_s=layout.pre_gate_route_delta_s(src, dst)))

    obs1, _, built1, _ = _env(rep)
    frozen = FrozenAdmission(built1["pool"], plan, ANNOUNCE_LEAD_S)
    tr = _run(rep, ctrl=frozen, sell=sell)
    if not st["ok"]:
        raise RuntimeError("처치 세계 집행 실패")
    tr_parts = _phi_parts(tr["mbt"], tr["obs"].observe_s)

    dd = lambda k: base_parts["truck"].get(k, 0.0) - tr_parts["truck"].get(k, 0.0)
    t_src, t_dst = dd(src), dd(dst)
    t_other = sum(base_parts["truck"][k] - tr_parts["truck"][k]
                  for k in base_parts["truck"] if k not in (src, dst))
    v_all = sum(base_parts["vessel"][k] - tr_parts["vessel"][k]
                for k in base_parts["vessel"])
    r_wait = base_parts["r_wait"] - tr_parts["r_wait"]
    g = base["phi"] - tr["phi"]
    res = {"idx": idx, "rep": rep, "src": src, "dst": dst,
           "delta_j": s["delta_j"], "gain_realized": round(g, 9),
           "parts": {"T_src": round(t_src, 9), "T_dst": round(t_dst, 9),
                     "T_other": round(t_other, 9), "V_all": round(v_all, 9),
                     "R_wait": round(r_wait, 9)},
           "identity_gap": round(t_src + t_dst + t_other + v_all + r_wait - g, 12),
           "n_admitted_base": base["ctrl"].n_admitted,
           "n_admitted_treat": frozen.n_admitted,
           "admission_identical": base["ctrl"].n_admitted == frozen.n_admitted,
           "stamp": {"code_commit": _git("rev-parse", "HEAD"),
                     "code_dirty": bool(code_dirty()),
                     "prereg_sha256": _sha256(PREREG) if PREREG.exists() else None}}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"iso_{idx:02d}.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("idx", "gain_realized", "parts",
                                          "admission_identical")},
                     ensure_ascii=False))
    return p


def verdict() -> dict:
    rows = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(OUT.glob("iso_*.json"))]
    if len(rows) != ISO_REPS * PER_RUN:
        raise RuntimeError(f"표본 {len(rows)}/{ISO_REPS*PER_RUN} — 판정 미실행")
    others = [abs(r["parts"]["T_other"]) for r in rows]
    device_effect = all(x < 1e-9 for x in others)
    g_frozen = [r["gain_realized"] for r in rows]
    v = {"verdict": ("A_device_effect" if device_effect else "B_hidden_coupling"),
         "n": len(rows), "max_abs_T_other": max(others),
         "admission_identical_all": all(r["admission_identical"] for r in rows),
         "identity_max_gap": max(abs(r["identity_gap"]) for r in rows),
         "gain_frozen": [round(x, 6) for x in g_frozen],
         "gain_frozen_positive": sum(1 for x in g_frozen if x > 0),
         "note": "장치 타당성 검사 — 성능·GO 주장 없음. 고정 투입 하 이득은 "
                 "예비 신호이며 정식 판정은 재판정 0B 에서만."}
    res = {"task": "YR-165", "runtime": {
        "commit": _git("rev-parse", "HEAD"), "git_dirty": bool(code_dirty()),
        "prereg_sha256": _sha256(PREREG) if PREREG.exists() else None,
        "seeds": [iso_seed(r) for r in range(ISO_REPS)]}, "result": v}
    p = OUT / "isolation.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "isolation.json.sha256").write_text(_sha256(p) + "\n", encoding="utf-8")
    print(json.dumps(v, ensure_ascii=False))
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
