"""YR-168 ① — **관찰 모드**: 판매를 실행하지 않고 "기회가 존재하는지"만 측정한다.

40차 감사 ④: *혼잡한 것과 판매 기회가 있는 것은 다르다.* YR-167 은 터미널 전체
혼잡만 확인했고, **같은 시각에 바쁜 판매 블록과 여유 있는 수신 블록이 함께 있는지**
는 확인하지 않았다. 판매가 애초에 성립할 수 없는 무대라면 본 실험은 무의미하다.

이 하네스는 YR-167 자격 무대(같은 시드·같은 코드 경로)를 **전건 KEEP** 으로 돌리며
5분 격자마다 다음을 읽는다 — 정책 결정에 개입하지 않는다(읽기 전용 관측):
  · `iter_pre_gate_candidates` 로 블록별 **실제 판매 후보 수**(해석적 근사 아님)
  · 블록별 대기열·재공·여유칸 → **혼잡 격차**(최대−최소, 상위/하위 분위)
  · 후보를 가진 혼잡 블록과 여유 수신 블록이 **동시에** 존재하는 시각의 비율
  · 크레인 사이클당 서비스 시간(현실성 게이트 재제출용 — 5차 계약 환경 실측)

판정 없음 — 노출량 보고다. 임계는 YR-168 판정 계약에서 따로 동결한다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, median

from ..integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference,
                                    _apply, _wait_of)
from ..integrated.candidates import CandidateGenerator
from ..integrated.multiblock import MultiBlockTerminal
from ..integrated.policy_config import LEGACY_DEFAULT
from ..integrated.pre_gate import iter_pre_gate_candidates
from ..integrated.profiles import build_h21_profile
from ..integrated.repro import code_dirty
from ..integrated.sell_review import ANNOUNCE_LEAD_S
from ..integrated.terminal_stream import (DIURNAL_DAY_TOTAL, OBS_24H,
                                          ScheduledAnnouncer, TerminalStreamParams,
                                          admission_epochs, build_diurnal,
                                          ensure_time_ledger)
from ..integrated.yard_layout import terminal_layout
from .yr088_joint_rl import LEVEL
from .yr149_load_cells import _sim_from
from .yr150_h21_pilot import _git, _sha256
from .yr167_diurnal_qual import SEED
from .yr167_observers import SnapshotCollector

OUT = Path("outputs/reports/yr168_observe")
PEAK_H = (11, 16)      # 피크대 (사전등록 §A 봉우리 11h·15h 를 포함하는 구간)
NIGHT_H = (2, 8)       # 야간대 — 측정창이 2h 부터라 1시는 관측 불가(40차 감사 ②)


def _crane_service_samples(mbt) -> list[float]:
    """사이클당 서비스 시간 = 작업 소요 ÷ (1 + 재취급) — 문헌 단위와 정합(YR-158)."""
    out = []
    for sim in mbt.blocks.values():
        for j in sim.jobs.values():
            if (j.status.name == "DONE" and j.service_start is not None
                    and j.service_end is not None):
                out.append((j.service_end - j.service_start)
                           / (1 + max(0, j.rehandle_count)))
    return out


def run_cell(rep: int) -> dict:
    obs, prof = OBS_24H, build_h21_profile()
    layout = terminal_layout()
    params = TerminalStreamParams(load_4h=DIURNAL_DAY_TOTAL)
    built = build_diurnal(prof, SEED + rep, obs=obs, layout=layout, params=params,
                          background_seed=SEED)
    mbt = MultiBlockTerminal(
        {b: ensure_time_ledger(_sim_from(s, prof))
         for b, s in built["scenarios"].items()},
        extra_review_epochs=admission_epochs(obs))
    ann = ScheduledAnnouncer(built["schedule"], lead_s=ANNOUNCE_LEAD_S,
                             end_s=built["sim_end_s"])
    snap = SnapshotCollector(obs.snapshot_times())
    pol = ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    gens: dict[int, CandidateGenerator] = {}
    want = {round(t, 6) for t in obs.snapshot_times()}
    seen: set[float] = set()
    frames: list[dict] = []          # 시각별 블록 상태 (판매 기회 판정 원자료)

    def policy(sim, dp):
        g = gens.setdefault(id(sim), CandidateGenerator(config=LEGACY_DEFAULT))
        gb = {c: g.generate(sim, c, LEVEL) for c in dp.crane_ids}
        try:
            _apply(sim, pol.decide(sim, dp, gb))
        except Exception:
            _apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})

    def review(m, t):
        ann.review(m, t)             # 투입만 — 판매는 하지 않는다(전건 KEEP)
        snap.observe(m, t)
        k = round(t, 6)
        if k not in want or k in seen:
            return
        seen.add(k)
        rows = []
        for bid, sim in m.blocks.items():
            tl = sim.time_ledger
            q = sum(1 for r in tl.records.values()
                    if r.block_arrival is not None and r.block_arrival <= t
                    and (r.service_start is None or r.service_start > t))
            rows.append({"b": bid, "q": q,
                         "cand": len(iter_pre_gate_candidates(m, bid)),
                         "free": m.free_slots(bid)})
        frames.append({"t": k, "rows": rows})

    mbt.run(policy, review_fn=review)

    # ---- 판매 기회 노출 ----
    # 기회 = 같은 시각에 ①후보를 가진 블록이 있고 ②그 블록보다 대기열이 낮은 블록이
    # 있다. 임계는 두지 않고 **격차 분포**를 보고한다(판정은 YR-168 계약에서).
    gaps, cands, with_cand, opp_any = [], [], [], []
    per_hour: dict[int, list[float]] = {}
    for fr in frames:
        qs = sorted(r["q"] for r in fr["rows"])
        gap = qs[-1] - qs[0]
        gaps.append(gap)
        c_tot = sum(r["cand"] for r in fr["rows"])
        cands.append(c_tot)
        blocks_with = [r for r in fr["rows"] if r["cand"] > 0]
        with_cand.append(len(blocks_with))
        # 후보 보유 블록의 최대 대기열 vs 전체 최소 대기열
        opp = (max((r["q"] for r in blocks_with), default=0) - qs[0]
               if blocks_with else 0)
        opp_any.append(opp)
        per_hour.setdefault(int(fr["t"] // 3600), []).append(opp)

    def band(lo, hi):
        v = [x for h, xs in per_hour.items() if lo <= h < hi for x in xs]
        return {"n_frames": len(v), "mean_gap": round(fmean(v), 2) if v else None,
                "max_gap": max(v) if v else None,
                "share_gap_ge_3": round(sum(1 for x in v if x >= 3) / len(v), 4) if v else None,
                "share_gap_ge_5": round(sum(1 for x in v if x >= 5) / len(v), 4) if v else None,
                "share_gap_ge_10": round(sum(1 for x in v if x >= 10) / len(v), 4) if v else None}

    cs = _crane_service_samples(mbt)
    return {
        "rep": rep, "seed": SEED + rep, "mode": "observe_only_all_keep",
        "n_frames": len(frames),
        "sale_candidates": {
            "total_frames_with_any": sum(1 for c in cands if c > 0),
            "mean_per_frame": round(fmean(cands), 2) if cands else None,
            "max_per_frame": max(cands) if cands else None,
            "mean_blocks_with_candidate": round(fmean(with_cand), 2) if with_cand else None,
            "max_blocks_with_candidate": max(with_cand) if with_cand else None},
        "queue_gap_all_blocks": {
            "mean": round(fmean(gaps), 2), "median": median(gaps), "max": max(gaps)},
        "opportunity_gap": {
            "definition": "후보 보유 블록의 최대 대기열 − 전체 블록 최소 대기열",
            "mean": round(fmean(opp_any), 2), "max": max(opp_any),
            "share_ge_3": round(sum(1 for x in opp_any if x >= 3) / len(opp_any), 4),
            "share_ge_5": round(sum(1 for x in opp_any if x >= 5) / len(opp_any), 4),
            "share_ge_10": round(sum(1 for x in opp_any if x >= 10) / len(opp_any), 4)},
        "peak_band": {"hours": list(PEAK_H), **band(*PEAK_H)},
        "night_band": {"hours": list(NIGHT_H), **band(*NIGHT_H)},
        "crane_service_s": {
            "n": len(cs), "mean": round(fmean(cs), 1) if cs else None,
            "min": round(min(cs), 1) if cs else None,
            "max": round(max(cs), 1) if cs else None,
            "note": "사이클당 = 작업 소요 ÷ (1+재취급) — 문헌 단위(YR-158 정의)"},
        "gate_throughput_per_h": round(
            sum(1 for sim in mbt.blocks.values() for r in sim.time_ledger.records.values()
                if obs.warmup_s <= r.gate_in < obs.observe_s)
            / (obs.measure_s / 3600.0), 1),
        "snapshot_summary": snap.summary(),
    }


def run() -> dict:
    cells = [json.loads((OUT / f"cell_rep{r}.json").read_text(encoding="utf-8"))
             for r in range(3)]
    res = {"task": "YR-168-observe",
           "runtime": {"commit": _git("rev-parse", "HEAD"),
                       "git_dirty": bool(code_dirty()),
                       "seeds": [SEED + r for r in range(3)],
                       "cell_sha256": {f"cell_rep{r}.json": _sha256(OUT / f"cell_rep{r}.json")
                                       for r in range(3)}},
           "summary": {
               "opportunity_gap_mean": [c["opportunity_gap"]["mean"] for c in cells],
               "opportunity_share_ge_5": [c["opportunity_gap"]["share_ge_5"] for c in cells],
               "peak_share_ge_5": [c["peak_band"]["share_gap_ge_5"] for c in cells],
               "night_share_ge_5": [c["night_band"]["share_gap_ge_5"] for c in cells],
               "crane_service_mean_s": [c["crane_service_s"]["mean"] for c in cells],
               "gate_throughput_per_h": [c["gate_throughput_per_h"] for c in cells]},
           "note": "관찰 전용 — 판매 미실행(전건 KEEP). 판정 임계 없음.",
           "cells": cells}
    p = OUT / "observe.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "observe.json.sha256").write_text(_sha256(p) + "\n", encoding="utf-8")
    print(json.dumps(res["summary"], ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int, default=None)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        run()
    elif a.rep is not None:
        c = run_cell(a.rep)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"cell_rep{a.rep}.json").write_text(
            json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in c.items()
                          if k != "snapshot_summary"}, ensure_ascii=False))
    print("DONE")
