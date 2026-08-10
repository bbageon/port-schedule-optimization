"""YR-164 ② — 공간 판매 견적 진단: **비용 분해 재생** (사전등록 동결본 이행).

사전등록: .claude/docs/strategy-history/2026-08-10-YR-164-비용분해-사전등록.md
  Φ = Σ블록 phi_v2(트럭 대기 + 적하 지연) + route/3600 + 기사 외부 대기/3600 를
  5항으로 분해: T_src·T_dst·T_other·V_all·R_wait (전부 기준−처치 부호).
  · 항등 검사: 5항 합 == 실현 이득 G (오차 1e-6 초과 시 그 표본 진단 제외)
  · 잔차 4종: res_src·res_dst·res_route·res_unmodeled (양수 = 견적 과대평가)
  · 지배 원인: |중앙값| 최대 항 — 상위 두 항 차 10% 미만이면 "복합"
  · 처리량 감소 10건은 **분리** 집계(가설 생성용)
판정 재실행이 아니라 **세부 장부 수집**이다 — 0B STOP 판정은 불변.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

from ..integrated.cost_curve_v2 import j_truck_realized, j_vessel_realized
from ..integrated.repro import code_dirty
from ..integrated.time_sell import deferral_ledger
from ..integrated.vessel import VesselWorkType
from .yr139_blockq_v4_ppo import SLA_ANCHOR_S
from .yr150_h21_pilot import _git, _sha256
from .yr151_0b_counterfactual import (OUT as CF_OUT, PREREG as CF_PREREG, _run,
                                      _state_digest)
from ..integrated.yard_layout import terminal_layout
from ..integrated.time_sell import try_time_sell

OUT = Path("outputs/reports/yr164_cost_decomposition")
PREREG = Path(".claude/docs/strategy-history/2026-08-10-YR-164-비용분해-사전등록.md")


def _phi_parts(mbt, t: float) -> dict:
    """Φ 를 블록별 트럭 대기 / 블록별 적하 지연 / route+외부대기 로 분해.

    합계는 phi_terminal 과 동일해야 한다(같은 원시 함수·같은 검열 규칙 사용).
    """
    truck: dict[str, float] = {}
    vessel: dict[str, float] = {}
    for bid, sim in mbt.blocks.items():
        sla = float(sim.profile.long_wait_sla_s)
        l_t = SLA_ANCHOR_S + sla
        s = 0.0
        tl = getattr(sim, "time_ledger", None)
        if tl is not None:
            for r in tl.records.values():
                a = getattr(r, "gate_in", None)
                if a is None or a > t + 1e-9:
                    continue
                o = getattr(r, "gate_out", None)
                o_eff = o if (o is not None and o <= t + 1e-9) else t
                s += j_truck_realized(o_eff, a, a + l_t)
        truck[bid] = s
        v_sum = 0.0
        for v in sim.vessels.values():
            if v.work_type != VesselWorkType.LOAD:
                continue
            p = v.plan.planned_completion_s
            if p is None:
                continue
            f = getattr(getattr(v, "truth", None), "actual_completion_s", None)
            f_eff = f if (f is not None and f <= t + 1e-9) else t
            v_sum += j_vessel_realized(f_eff, p)
        vessel[bid] = v_sum
    r_wait = mbt.route_cost_s / 3600.0
    route_only = r_wait
    for row in deferral_ledger(mbt):
        appt, a = row["original_appointment_s"], row["actual_gate_in_s"]
        if appt is not None and a is not None:
            r_wait += max(0.0, min(a, t) - appt) / 3600.0
    return {"truck": truck, "vessel": vessel, "r_wait": r_wait,
            "route_only": route_only}


def _done_in_window(mbt, obs) -> int:
    return sum(1 for s in mbt.blocks.values()
               if getattr(s, "time_ledger", None) is not None
               for r in s.time_ledger.records.values()
               if r.gate_out is not None
               and obs.warmup_s <= r.gate_out < obs.observe_s)


def replay(idx: int) -> Path:
    """표본 1건 진단 재생 — 두 세계를 다시 돌려 **세부 장부만** 수집."""
    s = json.loads((CF_OUT / "samples.json").read_text(encoding="utf-8"))["samples"][idx]
    rep, t_star, jid = s["rep"], s["t"], s["job_id"]
    if s["axis"] != "SPACE":
        raise RuntimeError("사전등록 범위: 공간 판매 표본만")
    src, dst = s["src"], s["dst"]

    base_state: dict = {}

    def capture(mbt_, t, ctrl):
        if "d" not in base_state and abs(t - t_star) < 1e-6:
            base_state["d"] = _state_digest(mbt_, ctrl)

    b = _run(rep, review_extra=capture)
    if "d" not in base_state:
        raise RuntimeError("기준 세계에 결정 epoch 없음")
    base_parts = _phi_parts(b["mbt"], b["obs"].observe_s)

    st = {"tried": False, "ok": False}

    def treat(mbt_, t, ctrl):
        if not st["tried"] and abs(t - t_star) < 1e-6:
            st["tried"] = True
            if _state_digest(mbt_, ctrl) != base_state["d"]:
                raise RuntimeError("짝 세계가 처치 전에 다름")
            layout = terminal_layout()
            st["ok"] = bool(mbt_.try_pre_gate_transfer(
                jid, dst, travel_s=layout.gate_to_block_s(dst),
                route_delta_s=layout.pre_gate_route_delta_s(src, dst)))

    tr = _run(rep, review_extra=treat)
    if not st["ok"]:
        raise RuntimeError("처치 세계에서 집행 실패 — 재생 불가")
    tr_parts = _phi_parts(tr["mbt"], tr["obs"].observe_s)

    # 5항 분해 (기준 − 처치 = 이득 방향)
    d = lambda m1, m2, k: m1.get(k, 0.0) - m2.get(k, 0.0)
    t_src = d(base_parts["truck"], tr_parts["truck"], src)
    t_dst = d(base_parts["truck"], tr_parts["truck"], dst)
    t_other = sum(base_parts["truck"][k] - tr_parts["truck"][k]
                  for k in base_parts["truck"] if k not in (src, dst))
    v_all = sum(base_parts["vessel"][k] - tr_parts["vessel"][k]
                for k in base_parts["vessel"])
    r_wait = base_parts["r_wait"] - tr_parts["r_wait"]
    g = b["phi"] - tr["phi"]
    total = t_src + t_dst + t_other + v_all + r_wait
    identity_ok = abs(total - g) < 1e-6

    # 견적 대응 잔차 (사전등록 정의 — 양수 = 견적이 이득을 과대평가)
    layout = terminal_layout()
    route_est = layout.pre_gate_route_delta_s(src, dst) / 3600.0
    res = {"res_src": None, "res_dst": None,
           "res_route": (base_parts["route_only"] - tr_parts["route_only"]) * -1.0
                        - route_est,
           "res_unmodeled": -(t_other + v_all)}

    out = {"idx": idx, "rep": rep, "src": src, "dst": dst,
           "delta_j": s["delta_j"], "gain_realized": round(g, 6),
           "parts": {"T_src": round(t_src, 6), "T_dst": round(t_dst, 6),
                     "T_other": round(t_other, 6), "V_all": round(v_all, 6),
                     "R_wait": round(r_wait, 6)},
           "identity_ok": identity_ok,
           "identity_gap": round(total - g, 9),
           "residuals": {k: (None if v is None else round(v, 6))
                         for k, v in res.items()},
           "throughput_base": _done_in_window(b["mbt"], b["obs"]),
           "throughput_treat": _done_in_window(tr["mbt"], tr["obs"]),
           "stamp": {"code_commit": _git("rev-parse", "HEAD"),
                     "code_dirty": bool(code_dirty()),
                     "prereg_sha256": _sha256(PREREG) if PREREG.exists() else None,
                     "samples_sha256": _sha256(CF_OUT / "samples.json")}}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"decomp_{idx:02d}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("idx", "gain_realized", "parts",
                                          "identity_ok")}, ensure_ascii=False))
    return p


def summarize() -> dict:
    files = sorted(OUT.glob("decomp_*.json"))
    rows = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    valid = [r for r in rows if r["identity_ok"]]
    excluded = [r["idx"] for r in rows if not r["identity_ok"]]
    parts_med = {k: round(median([r["parts"][k] for r in valid]), 6)
                 for k in ("T_src", "T_dst", "T_other", "V_all", "R_wait")} \
        if valid else {}
    res_keys = ("res_route", "res_unmodeled")
    res_med = {k: round(median([r["residuals"][k] for r in valid
                                if r["residuals"].get(k) is not None]), 6)
               for k in res_keys} if valid else {}
    ranked = sorted(((abs(v), k) for k, v in res_med.items()), reverse=True)
    dominant = None
    if len(ranked) >= 2 and ranked[0][0] > 0:
        dominant = ("복합" if (ranked[0][0] - ranked[1][0]) / ranked[0][0] < 0.10
                    else ranked[0][1])
    elif ranked:
        dominant = ranked[0][1]
    res = {"task": "YR-164-decomposition", "n_replayed": len(rows),
           "n_valid": len(valid), "excluded_idx": excluded,
           "parts_median": parts_med, "residual_median": res_med,
           "dominant_by_rule": dominant,
           "note": "진단 재생 — 판정 재실행 아님. 원인 가설까지만. 수정식·조건 "
                   "검증은 새 시드 별도 사전등록.",
           "runtime": {"commit": _git("rev-parse", "HEAD"),
                       "git_dirty": bool(code_dirty()),
                       "prereg_sha256": _sha256(PREREG) if PREREG.exists() else None,
                       "cf_prereg_sha256": (_sha256(CF_PREREG)
                                            if CF_PREREG.exists() else None)}}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "decomposition.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "decomposition.json.sha256").write_text(_sha256(p) + "\n",
                                                   encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("n_valid", "parts_median",
                                          "residual_median", "dominant_by_rule")},
                     ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=int, default=None)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.replay is not None:
        replay(a.replay)
    elif a.summarize:
        summarize()
    print("DONE")
