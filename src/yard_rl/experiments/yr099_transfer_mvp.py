"""YR-099 MVP — TOS 배정수신(t=0) 반입 재배정 headroom: A0(독립) vs C(quote+resolver).

■ 사전등록 (결과 미열람 동결)
- 범위: **반입(GATE_IN STORE)** 의 t=0 review 1회 — spec 결정시점 ①(TOS_ASSIGNMENT_RECEIVED).
  엔진 무변경(시나리오 수준 이동) → G0 소유권·보존·결정론이 구조적으로 성립.
  양하(DISCHARGE) 재배정·창중(in-flight) review 는 cross-sim 이벤트 브리지 필요 — 잔여작업.
- 터미널 = 블록 2: A=high-tight(본선압박) + B=mid-loose(여유). local policy = SF 고정(G1 계약).
- quote 정보안전: **예측 시나리오**(예약 준수 가정 actual:=appointment/estimated — 실현 draw
  미참조)에서 SF full-episode paired 반사실. 실현은 true 시나리오로만.
  OutRelief(A,j)=J_A(with)−J_A(without) · InBurden(B,j)=J_B(with j′)−J_B(base)
  Gain=OutRelief−InBurden−ROUTE_S/3600−MARGIN. 전수 비교, epoch 당 transfer ≤1 (spec).
- G1: paired Δ[terminal(C)−terminal(A0)] over seeds — CI 상한<0 이면 반입 재배정 상금 실재.
- G2-lite: 예측 Gain 부호 vs 실현 Δ 부호 일치율 (scalar quote 타당성 예비).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
from pathlib import Path
from statistics import fmean

from ..domain.enums import InformationLevel, JobFlow
from ..integrated import TerminalSimulator
from ..integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference,
                                    run_joint_episode)
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.profiles import build_calibrated_profile
from ..integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario
from .yr090_dense_vessel import _ci

RC = RewardCalculator.numeraire_v1()
OUT = Path("outputs/reports/yr099_transfer_mvp")
CELL_A, CELL_B = ("high", 0.5), ("mid", 2.0)      # 압박블록 / 여유블록
ROUTE_S = 180.0          # 블록 재라우팅 추가 주행 (assumed)
GAIN_MARGIN = 0.5        # 최소 순이득 (numeraire) — churn 방지 (spec RiskMargin)
REVIEW_MARGIN_S = 900.0  # 예상 블록도착이 이보다 임박하면 fail-closed
N_SEEDS = 8
BASE_A, BASE_B = 833_000, 834_000    # candidate_eval(+700 대역)과 불겹침 — 독립 측정


def _scen(seed: int, cell):
    prof = build_calibrated_profile()
    p = dataclasses.replace(calibrated_load_params(cell[0], vessel_deadline_mult=cell[1]),
                            time_contract_v2=True)
    return generate_terminal_scenario(prof, seed, p)


def _sim_of(scen):
    prof = build_calibrated_profile()
    s = TerminalSimulator(prof, scen, check_invariants=True)
    s.info_level = InformationLevel.PRE_ADVICE
    return s


def sf_run(scen) -> dict:
    r = run_joint_episode(_sim_of(scen), ResolverPolicy(ServiceFirstSPTPreference(), "SF"),
                          RC, generator=CandidateGenerator())
    return {"total": r["total_cost"], "berth": r["berth_overrun_min"],
            "wait": r["mean_wait_min"], "p95": r["p95_wait_min"],
            "compl": r["completion_rate"]}


def to_predicted(scen):
    """정보안전 quote 용 예측 시나리오 — 실현 draw 를 예약 준수 가정으로 대체 (누출 0)."""
    s = copy.deepcopy(scen)
    for j in s.jobs:
        appt = getattr(j, "appointment_gate_time", None)
        est = getattr(j, "estimated_block_arrival", None)
        if appt is not None and est is not None:
            j.actual_gate_in = appt
            j.actual_block_arrival = est
    return s


def eligible_inbound(scen) -> list[str]:
    """반입(GATE_IN STORE) + 공개 예상도착 존재 + 여유 REVIEW_MARGIN 이상 — fail-closed."""
    out = []
    for j in scen.jobs:
        if j.flow != JobFlow.GATE_IN:
            continue
        est = getattr(j, "estimated_block_arrival", None)
        if est is None or est < REVIEW_MARGIN_S:
            continue
        out.append(j.job_id)
    return sorted(out)


def move_job(scen_from, scen_to, jid: str, route_s: float = ROUTE_S):
    """원자적 이동(시나리오 수준): from 에서 제거 + to 에 도착시각 +route_s 로 삽입.

    반환 = (새 from, 새 to). 실패(미존재·중복 id) 시 ValueError — 호출부 KEEP 유지.
    """
    sf_, st_ = copy.deepcopy(scen_from), copy.deepcopy(scen_to)
    src = [j for j in sf_.jobs if j.job_id == jid]
    if len(src) != 1:
        raise ValueError(f"source 에 {jid} 정확히 1개 아님")
    j = src[0]
    new_id = f"{jid}@X"                          # 수신블록 내 id 충돌 방지 (registry 는 매핑 보존)
    if any(x.job_id == new_id for x in st_.jobs):
        raise ValueError(f"receiver 에 {new_id} 중복")
    sf_.jobs.remove(j)
    j.job_id = new_id
    if j.actual_block_arrival is not None:
        j.actual_block_arrival += route_s
    if getattr(j, "estimated_block_arrival", None) is not None:
        j.estimated_block_arrival += route_s
    if j.provided_eta is not None:
        j.provided_eta += route_s
    st_.jobs.append(j)
    return sf_, st_


def quote_and_pick(scen_a, scen_b, *, oracle: bool = False) -> dict | None:
    """예측 시나리오에서 전수 quote — 최대 양의 Gain 1건 (없으면 None=KEEP).

    oracle=True: **진단 전용** — 참(실현) 시나리오로 quote (정보누출·배포 금지).
    예측-quote 실패의 원인 분리용: 오라클도 실패 = marginal 카오스 자체 /
    오라클 성공 = 예측↔실현 정보 한계.
    """
    if oracle:
        pa, pb = copy.deepcopy(scen_a), copy.deepcopy(scen_b)
    else:
        pa, pb = to_predicted(scen_a), to_predicted(scen_b)
    ja, jb = sf_run(pa)["total"], sf_run(pb)["total"]
    best = None
    for src, dst, j_src, j_dst, tag in ((pa, pb, ja, jb, "A->B"), (pb, pa, jb, ja, "B->A")):
        for jid in eligible_inbound(src):
            try:
                s_wo, d_w = move_job(src, dst, jid)
            except ValueError:
                continue
            out_relief = j_src - sf_run(s_wo)["total"]
            in_burden = sf_run(d_w)["total"] - j_dst
            gain = out_relief - in_burden - ROUTE_S / 3600.0 - GAIN_MARGIN
            if gain > 0 and (best is None or gain > best["gain"] or
                             (gain == best["gain"] and (jid, tag) < (best["jid"], best["dir"]))):
                best = {"jid": jid, "dir": tag, "gain": round(gain, 3),
                        "out_relief": round(out_relief, 3), "in_burden": round(in_burden, 3)}
    return best


def run_seed(i: int, *, oracle: bool = False) -> dict:
    sa, sb = _scen(BASE_A + i, CELL_A), _scen(BASE_B + i, CELL_B)
    ra0, rb0 = sf_run(sa), sf_run(sb)
    a0 = {"total": ra0["total"] + rb0["total"], "berth": ra0["berth"] + rb0["berth"],
          "compl": min(ra0["compl"], rb0["compl"])}
    pick = quote_and_pick(sa, sb, oracle=oracle)
    if pick is None:
        c = dict(a0)
        n_moved = 0
    else:
        if pick["dir"] == "A->B":
            sa2, sb2 = move_job(sa, sb, pick["jid"])
        else:
            sb2, sa2 = move_job(sb, sa, pick["jid"])
        # G0 불변식: 보존·블록내 유일·이동작업 단일 소유 (블록 간 id 네임스페이스는 독립)
        assert len(sa2.jobs) + len(sb2.jobs) == len(sa.jobs) + len(sb.jobs)
        for blk in (sa2, sb2):
            ids = [j.job_id for j in blk.jobs]
            assert len(ids) == len(set(ids))
        moved_id = f"{pick['jid']}@X"
        n_moved_total = sum(1 for j in list(sa2.jobs) + list(sb2.jobs)
                            if j.job_id == moved_id)
        assert n_moved_total == 1
        ra, rb = sf_run(sa2), sf_run(sb2)
        c = {"total": ra["total"] + rb["total"], "berth": ra["berth"] + rb["berth"],
             "compl": min(ra["compl"], rb["compl"])}
        n_moved = 1
    return {"seed": i, "a0": a0, "c": c, "pick": pick, "n_moved": n_moved,
            "d_total": round(c["total"] - a0["total"], 3),
            "gain_pred": pick["gain"] if pick else 0.0}


def run(*, oracle: bool = False) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [run_seed(i, oracle=oracle) for i in range(N_SEEDS)]
    for r in rows:
        print(f"[seed {r['seed']}] A0={r['a0']['total']:.2f} C={r['c']['total']:.2f} "
              f"d={r['d_total']:+.2f} pick={r['pick']}", flush=True)
    d = [r["d_total"] for r in rows]
    moved = [r for r in rows if r["n_moved"]]
    sign_ok = sum(1 for r in moved if (r["d_total"] < 0) == (r["gain_pred"] > 0))
    res = {"rows": rows, "g1_d_total_ci": _ci(d), "oracle": oracle,
           "n_transfer": len(moved), "n_keep": len(rows) - len(moved),
           "g2lite_sign_agree": f"{sign_ok}/{len(moved)}" if moved else "n/a",
           "compl_min": min(min(r["a0"]["compl"], r["c"]["compl"]) for r in rows),
           "prereg": "G1: d_total CI 상한<0 → 반입 재배정 상금 실재 (t=0 review·SF 고정·"
                     "반입-only). 양하·창중 review 는 잔여작업."}
    name = "results_oracle.json" if oracle else "results.json"
    (OUT / name).write_text(json.dumps(res, ensure_ascii=False, indent=1),
                            encoding="utf-8")
    print(f"\nG1 d_total CI={res['g1_d_total_ci']} transfers={res['n_transfer']}/{N_SEEDS} "
          f"g2lite={res['g2lite_sign_agree']}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--oracle", action="store_true", help="진단 전용 — 참 시나리오 quote")
    a = ap.parse_args()
    N_SEEDS = a.seeds
    run(oracle=a.oracle)
    print("DONE")
