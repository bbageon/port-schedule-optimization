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
import random
from pathlib import Path
from statistics import fmean

from ..domain.enums import InformationLevel, JobFlow
from ..integrated import TerminalSimulator
from ..integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference,
                                    run_joint_episode)
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.profiles import build_calibrated_profile
from ..integrated.scenario_gen import (calibrated_load_params, generate_terminal_scenario,
                                       trunc_normal)
from .yr090_dense_vessel import _ci

RC = RewardCalculator.numeraire_v1()
OUT = Path("outputs/reports/yr099_transfer_mvp")
CELL_A, CELL_B = ("high", 0.5), ("mid", 2.0)      # 압박블록 / 여유블록
ROUTE_S = 180.0          # 블록 재라우팅 추가 주행 (assumed)
GAIN_MARGIN = 0.5        # 최소 순이득 (numeraire) — churn 방지 (spec RiskMargin)
REVIEW_MARGIN_S = 900.0  # 예상 블록도착이 이보다 임박하면 fail-closed
N_SEEDS = 8
BASE_A, BASE_B = 833_000, 834_000    # candidate_eval(+700 대역)과 불겹침 — 독립 측정


def _params(cell):
    return dataclasses.replace(calibrated_load_params(cell[0], vessel_deadline_mult=cell[1]),
                               time_contract_v2=True)


def _scen(seed: int, cell):
    return generate_terminal_scenario(build_calibrated_profile(), seed, _params(cell))


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


# ---------------------------------------------------------------- YR-101 ensemble quote
def to_predicted_sample(scen, params, sample_key: str):
    """공개 오차분포 **표본** 예측 시나리오 (YR-101) — 실현 draw 미참조 (누출 0).

    표본 = 예약 + clip(N(0,준수σ),±2σ) 진입, + trunc_normal(주행 μ,σfrac) 도착 —
    생성기와 같은 분포族·**전용 RNG 스트림**(ens-*: 실현 열 adh:/gtx: 와 완전 분리).
    """
    s = copy.deepcopy(scen)
    adh = random.Random(f"ens-adh:{sample_key}")
    gtx = random.Random(f"ens-gtx:{sample_key}")
    sig = params.appt_adherence_sigma_s
    for j in s.jobs:
        appt = getattr(j, "appointment_gate_time", None)
        est = getattr(j, "estimated_block_arrival", None)
        if appt is None or est is None:
            continue
        delta = max(-2.0 * sig, min(2.0 * sig, adh.gauss(0.0, sig))) if sig > 0 else 0.0
        a_in = max(0.0, appt + delta)
        j.actual_gate_in = a_in
        j.actual_block_arrival = a_in + trunc_normal(gtx, params.gate_travel_mu_s,
                                                     params.sigma_frac or 0.12, lo=60.0)
    return s


def quote_and_pick_ensemble(scen_a, scen_b, params_a, params_b, k_samples: int,
                            tag: str) -> dict | None:
    """K표본 평균 marginal quote (prereg: mean-Gain 최대 & > MARGIN 1건. pos_frac 은 보고용)."""
    samples = []
    for k in range(k_samples):
        sa_k = to_predicted_sample(scen_a, params_a, f"{tag}:A:{k}")
        sb_k = to_predicted_sample(scen_b, params_b, f"{tag}:B:{k}")
        samples.append((sa_k, sb_k, sf_run(sa_k)["total"], sf_run(sb_k)["total"]))
    best = None
    for direction, src_scen in (("A->B", scen_a), ("B->A", scen_b)):
        for jid in eligible_inbound(src_scen):            # eligibility = 원 시나리오(결정론)
            gains = []
            for sa_k, sb_k, ja_k, jb_k in samples:
                src, dst = (sa_k, sb_k) if direction == "A->B" else (sb_k, sa_k)
                j_src, j_dst = (ja_k, jb_k) if direction == "A->B" else (jb_k, ja_k)
                try:
                    s_wo, d_w = move_job(src, dst, jid)
                except ValueError:
                    gains = None
                    break
                out_relief = j_src - sf_run(s_wo)["total"]
                in_burden = sf_run(d_w)["total"] - j_dst
                gains.append(out_relief - in_burden - ROUTE_S / 3600.0 - GAIN_MARGIN)
            if not gains:
                continue
            mg = fmean(gains)
            if mg > 0 and (best is None or mg > best["gain"]
                           or (mg == best["gain"] and (jid, direction) < (best["jid"], best["dir"]))):
                best = {"jid": jid, "dir": direction, "gain": round(mg, 3),
                        "pos_frac": round(sum(1 for g in gains if g > 0) / len(gains), 3),
                        "gains_k": [round(g, 2) for g in gains]}
    return best


def run_seed_ens(i: int, k_samples: int) -> dict:
    sa, sb = _scen(BASE_A + i, CELL_A), _scen(BASE_B + i, CELL_B)
    ra0, rb0 = sf_run(sa), sf_run(sb)
    a0 = {"total": ra0["total"] + rb0["total"], "berth": ra0["berth"] + rb0["berth"],
          "compl": min(ra0["compl"], rb0["compl"])}
    pick = quote_and_pick_ensemble(sa, sb, _params(CELL_A), _params(CELL_B),
                                   k_samples, f"s{i}")
    if pick is None:
        c, n_moved = dict(a0), 0
    else:
        if pick["dir"] == "A->B":
            sa2, sb2 = move_job(sa, sb, pick["jid"])
        else:
            sb2, sa2 = move_job(sb, sa, pick["jid"])
        assert len(sa2.jobs) + len(sb2.jobs) == len(sa.jobs) + len(sb.jobs)
        for blk in (sa2, sb2):
            ids = [j.job_id for j in blk.jobs]
            assert len(ids) == len(set(ids))
        moved_id = f"{pick['jid']}@X"                    # G0 패리티: 이동작업 단일 소유
        assert sum(1 for j in list(sa2.jobs) + list(sb2.jobs)
                   if j.job_id == moved_id) == 1
        ra, rb = sf_run(sa2), sf_run(sb2)
        c = {"total": ra["total"] + rb["total"], "berth": ra["berth"] + rb["berth"],
             "compl": min(ra["compl"], rb["compl"])}
        n_moved = 1
    return {"seed": i, "a0": a0, "c": c, "pick": pick, "n_moved": n_moved,
            "d_total": round(c["total"] - a0["total"], 3),
            "gain_pred": pick["gain"] if pick else 0.0}


def run_ens(k_samples: int, shard: int, n_shards: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(N_SEEDS):
        if i % n_shards != shard:
            continue
        r = run_seed_ens(i, k_samples)
        rows.append(r)
        print(f"[ens K{k_samples} seed {r['seed']}] A0={r['a0']['total']:.2f} "
              f"C={r['c']['total']:.2f} d={r['d_total']:+.2f} pick={r['pick']}", flush=True)
    p = OUT / f"results_ens_K{k_samples}_shard{shard}.json"
    p.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {p}")


def merge_ens(k_samples: int, n_shards: int) -> dict:
    rows = []
    for s in range(n_shards):
        p = OUT / f"results_ens_K{k_samples}_shard{s}.json"
        rows.extend(json.loads(p.read_text(encoding="utf-8")))
    rows.sort(key=lambda r: r["seed"])
    # 검증 지적: 시드 커버리지 — 부분/중복 병합의 조용한 CI 오염 차단
    assert [r["seed"] for r in rows] == list(range(N_SEEDS)), "shard 병합 불완전/중복"
    d = [r["d_total"] for r in rows]
    moved = [r for r in rows if r["n_moved"]]
    # 검증 지적: 부호판정은 반올림된 gain_pred 아닌 n_moved(=pick 존재) 기준
    sign_ok = sum(1 for r in moved if r["d_total"] < 0)
    # 검증 지적(winner's curse 사후진단-lite): pick 의 K표본 LOO 안정성 —
    # 한 표본 빼도 mean-Gain>0 유지 비율 (전 후보 재-argmax 아님 — 한계 명시)
    loo = []
    for r in moved:
        gk = r["pick"].get("gains_k", [])
        if len(gk) >= 2:
            loo.append(sum(1 for i in range(len(gk))
                           if fmean(gk[:i] + gk[i + 1:]) > 0) / len(gk))
    res = {"rows": rows, "k_samples": k_samples, "g1_d_total_ci": _ci(d),
           "n_transfer": len(moved), "n_keep": len(rows) - len(moved),
           "sign_agree": f"{sign_ok}/{len(moved)} (transfer 조건부 — 분모=transfer 수)"
                         if moved else "n/a",
           "pick_loo_stability": round(fmean(loo), 3) if loo else None,
           "oracle_recovery": round(fmean(d) / -22.41, 3),
           "compl_min": min(min(r["a0"]["compl"], r["c"]["compl"]) for r in rows)}
    (OUT / f"results_ens_K{k_samples}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"ENS K{k_samples}: d_total CI={res['g1_d_total_ci']} sign={res['sign_agree']} "
          f"transfers={res['n_transfer']}/{len(rows)} recovery={res['oracle_recovery']}")
    return res


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
    ap.add_argument("--ensemble", type=int, help="YR-101: K표본 ensemble quote 실행")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--merge-ens", type=int, help="YR-101: 샤드 병합·판정")
    a = ap.parse_args()
    N_SEEDS = a.seeds
    if a.merge_ens:
        merge_ens(a.merge_ens, a.n_shards)
    elif a.ensemble:
        run_ens(a.ensemble, a.shard, a.n_shards)
    else:
        run(oracle=a.oracle)
    print("DONE")
