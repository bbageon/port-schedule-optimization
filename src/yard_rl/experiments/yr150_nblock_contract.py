"""YR-150 0단계 — N블록 엔진 계약 검사 (N=3, 성능 비교·정책 학습 없음).

■ 검사만 한다 (spec 실험 사다리 0단계)
  N1 목적지별 주행 matrix 가 **비영**이고 대칭·삼각부등식·앵커(평균 300초·범위 180~420초)를 지킴
  N2 수신 후보가 **둘 이상**인 결정에서 resolver 가 부담+주행 합이 최소인 목적지를 고름
  N3 한 epoch 에 **여러 소스가 동시에** 확정될 수 있음(2블록 터미널 1건 상한이 아님)
  N4 소스당 1건·작업당 이송 1회 상한 유지
  N5 수신 블록 **용량 부족**이면 확정 거절(엔진 fail-closed)
  N6 rollback 완전 복원 + 닫힌 txn 재commit 거절
  N7 결정론 — 같은 입력 2회 실행의 견적 원장·비용이 동일
  N8 소유권 정확히 1·orphan 0·A→O 장부 등록 수 보존
  N9 PRE_GATE 재배정의 route 차이가 **더 이상 0 이 아님**(0A 발견 ② 해소)
■ **금지**: 성능 비교·정책 군 비교·21블록 본시험 판정. 여기서 나오는 어떤 수치도
  "비용이 줄었다"는 근거로 쓰지 않는다(그 권한은 현실성 PASS 뒤 별도 인가).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

from ..integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference, _apply,
                                    _wait_of)
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_curve_v2 import KappaFit
from ..integrated.load_cells import generate_block
from ..integrated.multiblock import MultiBlockTerminal, TransferError
from ..integrated.repro import code_dirty
from ..integrated.transfer_quote import TransferQuoteResolver
from ..integrated.yard_layout import terminal_layout
from .yr088_joint_rl import LEVEL
from .yr149_load_cells import _sim_from

OUT = Path("outputs/reports/yr150_nblock_contract")
PREREG = Path(".claude/docs/dashboard-task-specs/YR-150-continuous-inflow-steady-state.md")
KAPPA = Path("outputs/reports/yr136_softplus_contract/kappa_fit_v2p.json")

# N=3 회귀는 **21블록 배치의 양 끝과 중앙**을 쓴다 — 인접 3블록만 쓰면 주행 차이가
# 최소가 되어 "목적지가 달라도 사실상 같다"는 구판 상태를 그대로 재현하게 된다.
SUB_IDS = ("Y01", "Y11", "Y21")
N_EXTERNAL = 50                     # 블록당 4시간 유입 (0단계 계약 검사용 규모)
SEEDS = {"Y01": 5100100, "Y11": 5100200, "Y21": 5100300}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _layout():
    return terminal_layout().subset(SUB_IDS)


def _build() -> MultiBlockTerminal:
    return MultiBlockTerminal({b: _sim_from(generate_block(SEEDS[b], N_EXTERNAL))
                               for b in SUB_IDS})


def _resolver(kf: KappaFit, layout, *, cap: int | None = None,
              gain_margin: float | None = None) -> TransferQuoteResolver:
    return TransferQuoteResolver(
        kf,
        travel_fn=lambda s, d, j: layout.gate_to_block_s(d),
        route_fn=layout.post_gate_route_s,
        gain_margin=gain_margin, terminal_epoch_cap=cap)


def _run(mbt: MultiBlockTerminal, review_fn) -> dict:
    """정책은 고정 규칙(SF-SPT)만 쓴다 — 학습 없음. review_fn 이 이송 확정자다."""
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

    res = mbt.run(policy, review_fn=review_fn)
    return {"run": res, "policy_exceptions": exc["n"]}


# ------------------------------------------------------------------ 개별 계약
def check_route_matrix(layout) -> dict:
    """N1 — 주행 matrix 가 비영이고 물리적으로 말이 되는가."""
    m = layout.matrix_s()
    pairs = [(a, b) for a in layout.ids for b in layout.ids if a != b]
    offdiag = [m[a][b] for a, b in pairs]
    gate = {b: layout.gate_to_block_s(b) for b in layout.ids}
    lo, hi = layout.gate_time_range_s()
    tri = all(m[a][c] <= m[a][b] + m[b][c] + 1e-9
              for a, b, c in itertools.product(layout.ids, repeat=3))
    full = terminal_layout()
    return {
        "nonzero_offdiagonal": bool(offdiag) and min(offdiag) > 0.0,
        "zero_diagonal": all(m[b][b] == 0.0 for b in layout.ids),
        "symmetric": all(abs(m[a][b] - m[b][a]) < 1e-9 for a, b in pairs),
        "triangle_inequality": tri,
        "gate_time_within_support": 180.0 <= lo and hi <= 420.0,
        "full_terminal_mean_is_anchor": abs(full.mean_gate_time_s() - 300.0) < 1e-6,
        "gate_to_block_s": {b: round(v, 3) for b, v in gate.items()},
        "block_to_block_s": m,
        "max_pair_gap_s": round(max(offdiag), 3),
    }


def check_multi_receiver(kf, layout) -> dict:
    """N2 — 수신 후보가 둘 이상인 결정에서 최소 부담+주행 목적지를 골랐는가."""
    mbt, resolver = _build(), _resolver(kf, layout)
    out = _run(mbt, resolver.review)
    led = resolver.ledger
    quoted = [r for r in led if r.get("bids")]
    n_recv = len(layout.ids) - 1
    # ① 원장 route 가 배치 matrix 와 정확히 일치 = 목적지별 비용이 실제로 쓰였다
    route_ok = all(
        abs(b["route_s"] - layout.post_gate_route_s(r["src"], b["dst"])) < 1e-9
        for r in quoted for b in r["bids"])
    # ② 고른 목적지가 **후보 전량 중 최소(부담+주행)** 인가 — 원장으로 직접 재계산
    def _key(src, b):
        return (b["in_burden"] + b["route_s"] / 3600.0, b["dst"])
    argmin_ok = all(
        r["dst"] == min(r["bids"], key=lambda b: _key(r["src"], b))["dst"]
        for r in quoted)
    # ③ 후보가 실제로 여럿이었고 그들의 route 가 서로 달랐는가(선택이 무의미하지 않았는가)
    multi_bid = [r for r in quoted if len({b["dst"] for b in r["bids"]}) >= 2]
    differing = [r for r in multi_bid if len({b["route_s"] for b in r["bids"]}) >= 2]
    return {"n_receiver_candidates_per_decision": n_recv,
            "n_decisions_with_quote": len(quoted),
            "n_decisions_with_multiple_bids": len(multi_bid),
            "n_decisions_with_differing_route": len(differing),
            "route_matches_layout": bool(quoted) and route_ok,
            "chose_min_burden_plus_route": bool(quoted) and argmin_ok,
            "distinct_route_values_s": sorted({round(b["route_s"], 3)
                                               for r in quoted for b in r["bids"]}),
            "n_transferred": resolver.n_transferred,
            "policy_exceptions": out["policy_exceptions"],
            "route_cost_ledger_s": round(out["run"]["route_cost_s"], 3),
            "cost_not_computed": "0단계는 cost_fn 을 붙이지 않는다 — 비용 수치 없음(의도)",
            "decisions": {k: sum(1 for r in led if r.get("decision") == k)
                          for k in sorted({r.get("decision") for r in led
                                           if r.get("decision")})}}


def check_simultaneous(kf, layout) -> dict:
    """N3·N4 — 같은 epoch 에 여러 소스가 동시에 확정될 수 있고, 소스당 1건인가.

    자연 시나리오에서는 두 블록의 gate-in 시각이 정확히 겹치는 일이 사실상 없으므로,
    **계약 검사 전용으로 gate-in 시각을 맞춘 상태**를 만들어 동시 확정 경로를 발화시킨다.
    (성능 수치가 아니라 경로 도달 여부만 본다.)
    """
    mbt = _build()
    # 각 블록에서 재배정 가능한 반입 작업 하나씩 골라 gate-in 을 같은 시각으로 맞춘다.
    picked = []
    for b in SUB_IDS:
        cands = sorted(jid for jid, rec in mbt.ledger.records.items()
                       if rec.owner == b and rec.reassignable and rec.a_gate_in is not None)
        if cands:
            picked.append(cands[len(cands) // 2])
    t0 = min(mbt.ledger.records[j].a_gate_in for j in picked) if picked else None
    if t0 is None or len(picked) < 2:
        return {"exercised": False, "reason": "동시 확정용 후보 부족"}
    for j in picked:
        mbt.ledger.records[j].a_gate_in = t0
        sim = mbt.blocks[mbt.ledger.records[j].owner]
        sim.jobs[j].actual_gate_in = t0
    mbt._schedule_review_epochs()

    # gain_margin 을 음수로 두어 **경로를 확실히 발화**시킨다 — 이 값은 계약 경로 도달을
    # 보기 위한 것이고 성능 판정에 쓰지 않는다(0단계는 성능 주장 금지).
    resolver = _resolver(kf, layout, cap=None, gain_margin=-1e9)
    out = _run(mbt, resolver.review)
    led = resolver.ledger

    per_epoch: dict[float, list[str]] = {}
    for r in led:
        if r.get("decision") == "TRANSFER":
            per_epoch.setdefault(r["t"], []).append(r["src"])
    multi = {t: sorted(v) for t, v in per_epoch.items() if len(set(v)) > 1}
    caps_ok = all(rec.transfer_count <= 1 for rec in mbt.ledger.records.values())
    return {"exercised": True,
            "aligned_gate_in_s": round(t0, 3),
            "n_aligned_jobs": len(picked),
            "n_epochs_with_transfer": len(per_epoch),
            "max_sources_in_one_epoch": max((len(set(v)) for v in per_epoch.values()),
                                            default=0),
            "simultaneous_multi_source": bool(multi),
            "simultaneous_epochs": {str(round(t, 3)): v for t, v in multi.items()},
            "one_offer_per_source_per_epoch": all(
                len(v) == len(set(v)) for v in per_epoch.values()),
            "job_transfer_cap_respected": caps_ok,
            "n_transferred": resolver.n_transferred,
            "policy_exceptions": out["policy_exceptions"]}


def check_capacity_and_atomicity(kf, layout) -> dict:
    """N5·N6·N8 — 용량 거절·rollback 복원·장부 보존."""
    mbt = _build()
    n_reg0 = len(mbt.ledger.records)
    src = SUB_IDS[0]
    cands = sorted(jid for jid, rec in mbt.ledger.records.items()
                   if rec.owner == src and rec.reassignable and rec.a_gate_in is not None)
    if not cands:
        return {"exercised": False, "reason": "후보 없음"}
    jid = cands[0]
    dst = SUB_IDS[1]
    travel = layout.gate_to_block_s(dst)
    route = layout.post_gate_route_s(src, dst)

    # 용량 fail-closed — 여유를 0 으로 만들면 준비 단계에서 거절돼야 한다
    saved = mbt.capacity_margin
    mbt.capacity_margin = 10 ** 9
    try:
        mbt.prepare_pre_gate_transfer(jid, dst, travel_s=travel)
        cap_reject = False
    except TransferError:
        cap_reject = True
    mbt.capacity_margin = saved

    # rollback 완전 복원 + 닫힌 txn 재commit 거절
    txn = mbt.prepare_pre_gate_transfer(jid, dst, travel_s=travel,
                                        route_delta_s=layout.pre_gate_route_delta_s(src, dst))
    mbt.rollback(txn)
    restored = (mbt.ledger.records[jid].owner == src and jid in mbt.blocks[src].jobs
                and mbt.ledger.records[jid].transfer_count == 0
                and mbt._reserved_inbound[dst] == 0)
    try:
        mbt.commit(txn)
        reclosed = False
    except TransferError:
        reclosed = True

    ok = mbt.try_pre_gate_transfer(jid, dst, travel_s=travel,
                                   route_delta_s=layout.pre_gate_route_delta_s(src, dst))
    try:
        mbt.check_invariants()
        inv = True
    except Exception:
        inv = False
    rec = mbt.ledger.records[jid]
    return {"exercised": True, "job": jid, "src": src, "dst": dst,
            "capacity_fail_closed": cap_reject,
            "rollback_restored": bool(restored), "closed_txn_rejected": bool(reclosed),
            "committed": bool(ok), "invariants": inv,
            "owner_after": rec.owner, "origin_preserved": rec.origin_block == src,
            "ledger_registration_preserved": len(mbt.ledger.records) == n_reg0,
            "second_transfer_rejected": not mbt.try_pre_gate_transfer(
                jid, SUB_IDS[2], travel_s=layout.gate_to_block_s(SUB_IDS[2]))}


def check_pre_gate_route_delta(layout) -> dict:
    """N9 — PRE_GATE 재배정의 주행 차이가 더 이상 0 이 아닌가 (0A 발견 ② 해소)."""
    deltas = {f"{a}->{b}": round(layout.pre_gate_route_delta_s(a, b), 3)
              for a in layout.ids for b in layout.ids if a != b}
    vals = list(deltas.values())
    return {"deltas_s": deltas,
            "any_nonzero": any(abs(v) > 1e-9 for v in vals),
            "has_negative": any(v < 0 for v in vals),      # 가까운 블록행 = 주행 절감
            "antisymmetric": all(
                abs(layout.pre_gate_route_delta_s(a, b)
                    + layout.pre_gate_route_delta_s(b, a)) < 1e-9
                for a in layout.ids for b in layout.ids)}


def check_determinism(kf, layout) -> dict:
    """N7 — 같은 입력 2회 실행이 완전히 같은 원장·비용을 낸다."""
    outs = []
    for _ in range(2):
        mbt, resolver = _build(), _resolver(kf, layout)
        out = _run(mbt, resolver.review)
        outs.append({"route_cost_s": round(out["run"]["route_cost_s"], 6),
                     "n": resolver.n_transferred,
                     "digest": hashlib.sha256(
                         json.dumps(resolver.ledger, sort_keys=True, default=str)
                         .encode()).hexdigest()[:16]})
    return {"identical": outs[0] == outs[1], "runs": outs}


# ------------------------------------------------------------------ 실행
def run() -> dict:
    kf = KappaFit(**{k: v for k, v in
                     json.loads(KAPPA.read_text(encoding="utf-8")).items()
                     if k in KappaFit.__dataclass_fields__})
    layout = _layout()
    checks = {
        "N1_route_matrix": check_route_matrix(layout),
        "N2_multi_receiver": check_multi_receiver(kf, layout),
        "N3N4_simultaneous": check_simultaneous(kf, layout),
        "N5N6N8_capacity_atomicity": check_capacity_and_atomicity(kf, layout),
        "N9_pre_gate_route_delta": check_pre_gate_route_delta(layout),
        "N7_determinism": check_determinism(kf, layout),
    }
    flags = {
        "N1": all(checks["N1_route_matrix"][k] for k in
                  ("nonzero_offdiagonal", "zero_diagonal", "symmetric",
                   "triangle_inequality", "gate_time_within_support",
                   "full_terminal_mean_is_anchor")),
        "N2": (checks["N2_multi_receiver"]["route_matches_layout"]
               and checks["N2_multi_receiver"]["chose_min_burden_plus_route"]
               and checks["N2_multi_receiver"]["n_decisions_with_differing_route"] > 0),
        "N3": bool(checks["N3N4_simultaneous"].get("simultaneous_multi_source")),
        "N4": bool(checks["N3N4_simultaneous"].get("one_offer_per_source_per_epoch")),
        "N5": bool(checks["N5N6N8_capacity_atomicity"].get("capacity_fail_closed")),
        "N6": bool(checks["N5N6N8_capacity_atomicity"].get("rollback_restored")
                   and checks["N5N6N8_capacity_atomicity"].get("closed_txn_rejected")),
        "N7": checks["N7_determinism"]["identical"],
        "N8": bool(checks["N5N6N8_capacity_atomicity"].get("invariants")
                   and checks["N5N6N8_capacity_atomicity"]
                   .get("ledger_registration_preserved")),
        "N9": bool(checks["N9_pre_gate_route_delta"]["any_nonzero"]
                   and checks["N9_pre_gate_route_delta"]["has_negative"]),
    }
    verdict = {"contract_all_pass": all(flags.values()), "flags": flags,
               "note": "0단계 계약 검사 전용 — 성능·학습 판정 없음. 어떤 수치도 "
                       "비용 개선 근거로 쓰지 않는다(현실성 PASS 뒤 별도 인가)."}
    # ★미추적 신규 파일까지 본다 (2026-08-06 정정 — 아래 주석 참조).
    dirty = bool(code_dirty())
    res = {"stage": "0", "task": "YR-150",
           "runtime": {"commit": _git("rev-parse", "HEAD"), "git_dirty": dirty,
                       "remote_ref": "origin/master",
                       "remote_head": _git("rev-parse", "origin/master"),
                       "prereg_file": str(PREREG),
                       "prereg_sha256": _sha256(PREREG) if PREREG.exists() else None,
                       "params": {"SUB_IDS": list(SUB_IDS), "N_EXTERNAL": N_EXTERNAL,
                                  "layout": layout.as_dict(),
                                  "terminal_blocks": 21,
                                  "terminal_epoch_cap": None},
                       "seeds": {"blocks": [SEEDS[b] for b in SUB_IDS]}},
           "verdict": verdict, "checks": checks}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "contract_n3.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "contract_n3.json.sha256").write_text(_sha256(p) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "dirty": dirty}, ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.run:
        run()
    print("DONE")
