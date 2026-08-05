"""YR-151 0A — 기술 계약 검사 (성능 비교·PPO 학습 없음, 동결 2026-08-06).

■ 검사만 한다 (31차 계약)
  C1 후보가 전부 PRE_GATE 창(공개 예측 30분) 안·미진입      C2 실현 미래 누출 0(교란 검사)
  C3 원자 commit(소유자 정확히 1·orphan 0·최초배정 이력 보존) C4 A 불변·도착은 게이트→새 블록
  C5 작업당 이송 1회 상한(엔진 fail-closed)                   C6 stale version → KEEP
  C7 rollback 완전 복원·닫힌 txn 재commit 거절                C8 A→O 장부 등록 수 보존
■ **금지**: 성능 비교·PPO 학습·"후보가 적다"를 이유로 한 학습 중단 판정(그 권한은 0B).
■ 신뢰성 게이트를 닫기 위한 재현 스탬프: clean commit·구체 설정값·사전등록 파일 경로·
  절대 시드·산출물 sha256 을 결과 JSON 에 남긴다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from ..integrated.load_cells import CELL_SIZES, make_a_cell, make_b_cell
from ..integrated.multiblock import MultiBlockTerminal, TransferError
from ..integrated.pre_gate import (MAX_TRANSFERS, WINDOW_S, iter_pre_gate_candidates,
                                   probe_public_info_only, public_block_eta)
from .yr149_load_cells import _sim_from

OUT = Path("outputs/reports/yr151_pre_gate_0a")
PREREG = Path(".claude/docs/dashboard-task-specs/YR-151-block-ppo-sell-head.md")
BAND = Path("outputs/reports/yr149_load_cells/band.json")
TRAVEL_S = 300.0            # 게이트→블록 기대 주행 (계약 물리 중심값) — 구체 설정값 박제
ROUTE_DELTA_S = 0.0         # 게이트→새 블록 − 게이트→기존 블록 **예측** 차이


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_cell(a_seed: int, b_seed: int, m: int) -> dict:
    """한 셀에서 계약 8종을 검사 — 실행은 t=0 상태에서 결정론적으로 수행."""
    mbt = MultiBlockTerminal({"A": _sim_from(make_a_cell(a_seed, m)),
                              "B": _sim_from(make_b_cell(b_seed))})
    n_reg0 = len(mbt.ledger.records)
    res: dict = {"cell": f"A{m}", "a_seed": a_seed, "b_seed": b_seed}

    cands = {b: iter_pre_gate_candidates(mbt, b) for b in ("A", "B")}
    res["n_candidates"] = {b: len(v) for b, v in cands.items()}      # 보고 전용(판정 아님)

    # C1 창·미진입
    c1 = True
    for b, lst in cands.items():
        for jid, eta in lst:
            rec = mbt.ledger.records[jid]
            if not (rec.a_gate_in > mbt.now and 0.0 < eta - mbt.now <= WINDOW_S):
                c1 = False
            if public_block_eta(mbt.blocks[b].jobs[jid]) != eta:
                c1 = False
    res["C1_window_and_pre_gate"] = c1

    # C2 실현 미래 누출 0
    probes = {b: probe_public_info_only(mbt, b) for b in ("A", "B")}
    res["C2_no_future_leak"] = all(p["identical"] for p in probes.values())
    res["leak_probe"] = probes

    if not cands["A"]:
        res.update({"C3_atomic_commit": None, "C4_gate_in_invariant": None,
                    "C5_transfer_cap": None, "C6_stale_rejected": None,
                    "C7_rollback_restore": None, "C8_ledger_registration": None,
                    "note": "A 블록 후보 없음 — 실행 계약 검사 미수행(0A 에서 STOP 사유 아님)"})
        return res

    jid = cands["A"][0][0]
    a0 = mbt.ledger.records[jid].a_gate_in

    # C6 stale version → KEEP (commit 전 version 변경)
    txn = mbt.prepare_pre_gate_transfer(jid, "B", travel_s=TRAVEL_S,
                                        route_delta_s=ROUTE_DELTA_S)
    mbt.ledger.records[jid].version += 1
    try:
        mbt.commit(txn)
        res["C6_stale_rejected"] = False
    except TransferError:
        res["C6_stale_rejected"] = True
    mbt.rollback(txn)
    mbt.ledger.records[jid].version -= 1

    # C7 rollback 완전 복원 + 닫힌 txn 재commit 거절
    txn2 = mbt.prepare_pre_gate_transfer(jid, "B", travel_s=TRAVEL_S,
                                         route_delta_s=ROUTE_DELTA_S)
    mbt.rollback(txn2)
    restored = (mbt.ledger.records[jid].owner == "A" and jid in mbt.blocks["A"].jobs
                and mbt.ledger.records[jid].transfer_count == 0)
    try:
        mbt.commit(txn2)
        reclosed = False
    except TransferError:
        reclosed = True
    res["C7_rollback_restore"] = bool(restored and reclosed)

    # C3·C4 실제 원자 commit
    ok = mbt.try_pre_gate_transfer(jid, "B", travel_s=TRAVEL_S,
                                   route_delta_s=ROUTE_DELTA_S)
    rec = mbt.ledger.records[jid]
    try:
        mbt.check_invariants()
        inv = True
    except Exception:
        inv = False
    res["C3_atomic_commit"] = bool(
        ok and inv and rec.owner == "B" and rec.origin_block == "A"
        and jid in mbt.blocks["B"].jobs and jid not in mbt.blocks["A"].jobs)
    res["C4_gate_in_invariant"] = bool(
        rec.a_gate_in == a0
        and abs(mbt.blocks["B"].jobs[jid].actual_block_arrival - (a0 + TRAVEL_S)) < 1e-6)
    res["route_cost_s"] = mbt.route_cost_s

    # C5 상한 (엔진 fail-closed)
    res["C5_transfer_cap"] = not mbt.try_pre_gate_transfer(jid, "A", travel_s=TRAVEL_S)

    # C8 전역 장부 등록 수 보존 (이송이 트럭을 증발시키지 않는다)
    res["C8_ledger_registration"] = (len(mbt.ledger.records) == n_reg0)
    res["transferred_job"] = jid
    return res


def run() -> dict:
    band = json.loads(BAND.read_text(encoding="utf-8"))
    pairs = list(zip(band["seeds"]["A"], band["seeds"]["B"]))
    rows = [check_cell(sa, sb, m) for sa, sb in pairs for m in CELL_SIZES]
    keys = ("C1_window_and_pre_gate", "C2_no_future_leak", "C3_atomic_commit",
            "C4_gate_in_invariant", "C5_transfer_cap", "C6_stale_rejected",
            "C7_rollback_restore", "C8_ledger_registration")
    summary = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        summary[k] = {"checked": len(vals), "pass": sum(1 for v in vals if v),
                      "all_pass": bool(vals) and all(vals)}
    n_exercised = sum(1 for r in rows if r.get("C3_atomic_commit") is not None)
    verdict = {"contract_all_pass": all(v["all_pass"] for v in summary.values()),
               "n_cells": len(rows), "n_cells_exercised": n_exercised,
               "note": "계약 검사 전용 — 성능·학습 판정 없음. 후보 수는 보고 전용이며 "
                       "적다는 이유로 학습을 중단하지 않는다(0B 권한)."}
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    res = {"stage": "0A",
           "runtime": {"commit": _git("rev-parse", "HEAD"),
                       "git_dirty": dirty,
                       "remote_ref": "origin/master",
                       "remote_head": _git("rev-parse", "origin/master"),
                       "prereg_file": str(PREREG),
                       "prereg_sha256": _sha256(PREREG) if PREREG.exists() else None,
                       "band_file": str(BAND), "band_sha256": _sha256(BAND),
                       "params": {"WINDOW_S": WINDOW_S, "MAX_TRANSFERS": MAX_TRANSFERS,
                                  "TRAVEL_S": TRAVEL_S,
                                  "ROUTE_DELTA_S": ROUTE_DELTA_S,
                                  "CELL_SIZES": list(CELL_SIZES)},
                       "seeds": {"A": band["seeds"]["A"], "B": band["seeds"]["B"]}},
           "verdict": verdict, "summary": summary, "cells": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "contract_0a.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    res["self_sha256"] = _sha256(p)
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "dirty": dirty,
                      "summary": {k: v["all_pass"] for k, v in summary.items()},
                      "n_candidates_A": [r["n_candidates"]["A"] for r in rows]},
                     ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.run:
        run()
    print("DONE")
