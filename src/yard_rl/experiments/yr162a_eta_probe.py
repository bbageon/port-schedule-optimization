"""YR-162A — 예약 준수오차 **방어 경로 발화** probe (A/B 분할의 A — 0B 전 의무).

■ 동결 (결과 열람 전)
  · σ = 600초(±2σ 절단) — 1~3차 유입량 계약의 준수오차 관행값 승계
    (`TerminalStreamParams.appt_adherence_sigma_s` 기본 600). 문헌 재조사값 대신
    저장소 선례를 채택(등록 시점 확정 — 결과 보고 바꾸지 않는다).
  · 셀 = w5·L100 (YR-157 확정 안정 BUSY 무대) · lead 30분 · 배경 = YR-157 rep0 동일.
  · 오차는 **교체 투입분에만** 주입(초기 채움은 t=0 사전 배치라 준수오차 개념 없음).
■ 판정 A1~A5 — **발화·정합만** 본다. "결론이 뒤집히는가"는 YR-162B(0B 후).
  A1 오차 실재: 투입분 중 |실현−통지| > 1초 비율 ≥ 50%
  A2 정보경계: 전 외부트럭 공개 예측 = 통지 + 기대주행 (실현 미참조 — 누출 0 유지)
  A3 방어 발화: 원자 확정 거부(KEEP_TXN_FAIL) ≥ 1 — 공개값으로는 진입 전으로 보이나
     실현은 이미 진입한(조기 도착) 트럭의 이연·이송 시도를 거부 경로가 실제로 막는가.
     0 이면 FAIL(죽은 경로 그대로) — 그 사실을 박제한다.
  A4 결정론: 같은 시드 재실행 시 admission·resolver 원장 완전 동일
  A5 물리 불변식 + 장부 보존(등록 = 채움 + 투입)
■ 실행 정책 = SF(자격 관행 — 성능 무관), 판매 정책 = OfferFirst(항상 첫 후보 제안 —
  방어 경로를 최대로 눌러보는 막대 probe. 성능 정책 아님).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ..integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference,
                                    _apply, _wait_of)
from ..integrated.candidates import CandidateGenerator
from ..integrated.multiblock import MultiBlockTerminal
from ..integrated.profiles import build_h21_profile
from ..integrated.repro import code_dirty
from ..integrated.sell_review import (ANNOUNCE_LEAD_S, UnifiedSellOrchestrator)
from ..integrated.terminal_stream import (ObservationContract,
                                          TerminalStreamParams,
                                          WipAdmissionController,
                                          admission_epochs, build_fixed_wip,
                                          hotspot_rotation)
from ..integrated.yard_layout import terminal_layout
from .yr088_joint_rl import LEVEL
from .yr149_load_cells import _sim_from
from .yr150_h21_pilot import _git, _sha256
from .yr151_transfer_ppo import load_kf
from .yr157_band_qual import N_HOTSPOT, SEED as BAND_SEED, cell_seed

OUT = Path("outputs/reports/yr162a_eta_probe")
PREREG = Path(".claude/docs/dashboard-task-specs/YR-162-eta-error-probe.md")
SIGMA_S = 600.0                 # 동결 — 준수오차 σ (저장소 선례 승계)
W, LOAD = 5.0, 100              # YR-157 확정 안정 BUSY 셀


class OfferFirst:
    """probe 전용 — 항상 첫 후보를 OFFER. 방어 경로를 눌러보는 막대(성능 정책 아님)."""

    mode = "live"

    def decide(self, mbt, src: str, cands: list, t: float) -> str | None:
        return cands[0][0] if cands else None


def _run_once() -> dict:
    obs = ObservationContract()
    prof = build_h21_profile()
    layout = terminal_layout()
    seed = cell_seed(W, LOAD)
    hs = hotspot_rotation(layout, seed, N_HOTSPOT)
    params = TerminalStreamParams(load_4h=LOAD, hotspot_blocks=hs, hotspot_weight=W)
    built = build_fixed_wip(prof, seed, wip_target=LOAD, obs=obs, layout=layout,
                            params=params, background_seed=BAND_SEED)
    mbt = MultiBlockTerminal({b: _sim_from(s) for b, s in built["scenarios"].items()},
                             extra_review_epochs=admission_epochs(obs))
    ctrl = WipAdmissionController(built["pool"], wip_target=LOAD,
                                  lead_s=ANNOUNCE_LEAD_S, end_s=obs.observe_s,
                                  adherence_sigma_s=SIGMA_S)
    orch = UnifiedSellOrchestrator(OfferFirst(), layout, load_kf())

    pol = ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    gens: dict[int, CandidateGenerator] = {}
    exc = {"n": 0}

    def policy(sim, dp):
        g = gens.setdefault(id(sim), CandidateGenerator())
        gb = {c: g.generate(sim, c, LEVEL) for c in dp.crane_ids}
        try:
            _apply(sim, pol.decide(sim, dp, gb))
        except Exception:
            exc["n"] += 1
            _apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})

    def review(mbt_, t):
        ctrl.review(mbt_, t)
        orch.review(mbt_, t)

    mbt.run(policy, review_fn=review)
    return {"mbt": mbt, "ctrl": ctrl, "orch": orch, "built": built,
            "obs": obs, "layout": layout, "exc": exc["n"]}


def run() -> dict:
    r1 = _run_once()
    mbt, ctrl, orch = r1["mbt"], r1["ctrl"], r1["orch"]
    obs, layout, built = r1["obs"], r1["layout"], r1["built"]

    admits = [e for e in ctrl.ledger if e["event"] == "ADMIT"]
    eps = [e["eps_s"] for e in admits]
    diverged = [x for x in eps if abs(x) > 1.0]
    a1 = bool(admits) and len(diverged) / max(1, len(admits)) >= 0.5

    # A2 검사식 정정(1차 실행 발견): 판매(SELL)·이연(DEFER)으로 **정당하게 갱신된**
    # 트럭은 원식(통지+원블록 주행)이 성립하지 않는 게 규약이다(이송·이연 갱신 규약은
    # 0A 검증 소관). 누출 검사는 resolver 가 건드리지 않은 트럭 한정이 정확하다.
    touched = {e["job_id"] for e in orch.ledger
               if e.get("decision") in ("SELL", "DEFER")}
    a2 = True
    for b, sim in mbt.blocks.items():
        base = layout.gate_to_block_s(b)
        for j in sim.jobs.values():
            if (j.is_external_truck and j.job_id not in touched
                    and getattr(j, "notified_gate_in_s", None) is not None):
                if abs(j.estimated_block_arrival
                       - (j.notified_gate_in_s + base)) > 1e-6:
                    a2 = False

    decisions = Counter(f"{e['axis']}:{e['decision']}" for e in orch.ledger)
    txn_fail = sum(v for k, v in decisions.items() if k.endswith("KEEP_TXN_FAIL"))
    a3 = txn_fail >= 1

    r2 = _run_once()                                   # A4 결정론 — 전체 재실행 대조
    a4 = (r2["ctrl"].ledger == ctrl.ledger and r2["orch"].ledger == orch.ledger)

    try:
        mbt.check_invariants()
        inv = True
    except Exception:
        inv = False
    n_rows = sum(len(getattr(s, "time_ledger").records)
                 for s in mbt.blocks.values() if getattr(s, "time_ledger", None))
    a5 = inv and n_rows == len(built["fill"]) + ctrl.n_admitted

    checks = {"A1_error_present": a1, "A2_no_leak_public_prediction": a2,
              "A3_defense_txn_reject_fired": a3, "A4_deterministic": a4,
              "A5_invariants_and_ledger": a5,
              "no_exec_policy_exceptions": r1["exc"] == 0}
    verdict = {
        "probe_all_pass": all(checks.values()), "checks": checks,
        "n_admitted": len(admits), "n_diverged_gt1s": len(diverged),
        "eps_abs_mean_s": round(sum(abs(x) for x in eps) / max(1, len(eps)), 1),
        "resolver_decisions": dict(decisions),
        "n_skips": len([e for e in ctrl.ledger if e["event"] == "SKIP"]),
        "note": "A = 방어 발화·정합만 판정. 결론 안정성(GO/STOP 유지)은 YR-162B(0B 후). "
                "성능 주장 없음 — OfferFirst 는 막대 probe.",
    }
    dirty = bool(code_dirty())
    res = {"task": "YR-162A", "cell": f"w{W}-L{LOAD}", "sigma_s": SIGMA_S,
           "runtime": {"commit": _git("rev-parse", "HEAD"), "git_dirty": dirty,
                       "prereg_file": str(PREREG),
                       "prereg_sha256": _sha256(PREREG) if PREREG.exists() else None,
                       "seeds": {"cell": cell_seed(W, LOAD),
                                 "background": BAND_SEED}},
           "verdict": verdict}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "probe_a.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "probe_a.json.sha256").write_text(_sha256(p) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "dirty": dirty}, ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.run:
        run()
    print("DONE")
