"""YR-105 — 조건부 창중 이송: ①평가계약 보완(결정별 로깅) → ②본선 악화 방지 필터 단일축.

■ 사용자 순서 결정 (2026-07-27)
- 첫 정책변경은 `C_HIGH`(혼잡 임계)가 **아니라 본선 악화 방지 필터 하나**다.
  근거: YR-099-mid 에서 비본선(트럭·블록) 비용은 이미 개선 방향이었고 **실패가 본선항에서
  발생**했다. 그 다음에 `C_HIGH`·지속 혼잡을 **각각 따로** 시험한다(한 번에 한 축).
- 선결 = 평가계약 보완: 결정마다 `Q/W/L/F/C·본선 여유·이송시각·A→O·B→C·본선·주행비` 저장.

■ 사전등록 (결과 미열람 동결)
- 기반: YR-099-b 브리지(공용 시계·정확 gate-in epoch·전역 A→O 장부·2단계 commit·용량검사).
- 공통 base 규칙 = YR-099-mid 와 동일한 혼잡 격차 `C_zero(src) − C_zero(dst) ≥ 0.10`
  (기각된 규칙 — 여기서는 **단일축 변경의 대조군**이지 재도전이 아니다).
- **arm 3개, 유일 차이 = 이송 차단 규칙**:
    · `BASE`   : 격차 조건만 (YR-099-mid 재현 — 브리지 정정판)
    · `VGUARD` : 격차 조건 **AND 본선 보호** — 소스 블록의 LOAD 본선 최소 여유
      `min slack_s(v) < 0`(= 계획 대비 **이미 밀린** 본선)이면 **이송 금지**.
      임계 0 은 튜닝값이 아니라 `slack_s` 의 의미 경계다(음수 = 현 작업률로
      계획완료 미달). 선택 대역 보정 관측: 발화 결정의 본선 여유 중앙값 −3096s
      (deciles −5607…+2198) — 대부분의 이송이 **본선이 이미 밀린 상태**에서 발화.
    · `RGUARD` : **볼륨 매칭 대조군** — VGUARD 와 *같은 차단 비율*을 본선상태와
      무관한 전용 난수로 차단. "이송을 줄여서 좋아진 것"과 "본선 인지 덕에
      좋아진 것"을 분리한다(교락 통제).
- 판정: ①주판정 paired `Δ = terminal(VGUARD) − terminal(BASE)` CI(t df=n−1)
  **상한 < 0** → 본선 보호 필터가 창중 이송의 손실원을 막음. ②**교락 통제**
  `Δ' = terminal(VGUARD) − terminal(RGUARD)` 상한 < 0 → 그 이득이 **볼륨 감소가
  아니라 본선 인지** 덕. ①만 통과하고 ②가 실패하면 "이송량 축소 효과"로 정직 보고.
  부지표: 본선(berth 합)·A→O 평균·B→C 평균·주행비·이송 수·완주.
- 대역: **선택 847k/848k(N=8) → 확증 849k/850k(N=8)** 2대역 (YR-098/099-mid 관행).
- 금지: 이 실험에서 `C_HIGH`·지속 혼잡·quote 정교화를 함께 바꾸지 않는다(축 혼입).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
from pathlib import Path
from statistics import fmean

from ..domain.enums import InformationLevel, JobFlow
from ..integrated import TerminalSimulator
from ..integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference, _apply, _wait_of)
from ..integrated.block_congestion import block_congestion
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.evalkit import CHANNELS, channel_split, check_guards, paired_by_channel
from ..integrated.multiblock import MultiBlockTerminal
from ..integrated.profiles import build_calibrated_profile
from ..integrated.scenario_gen import (GATE_BLOCK_MAX_S, GATE_BLOCK_MEAN_S, GATE_BLOCK_MIN_S,
                                       GATE_BLOCK_SIGMA_S, calibrated_load_params,
                                       generate_terminal_scenario, trunc_normal)
from ..integrated.vessel import VesselWorkType
from .yr103_info_sufficiency import _ci_t

RC = RewardCalculator.numeraire_v1()
OUT = Path("outputs/reports/yr105_conditional_transfer")
LEVEL = InformationLevel.PRE_ADVICE
CELL_A, CELL_B = ("high", 0.5), ("mid", 2.0)
BANDS = {"select": (847_000, 848_000), "confirm": (849_000, 850_000)}
N_SEEDS = 8
THRESH, ROUTE_S = 0.10, 180.0
# 본선 보호 경계 = slack_s 의 의미 경계(0 = 계획완료 미달 시작). 튜닝값 아님.
VESSEL_SLACK_MIN = 0.0


def _sim(cell, seed):
    prof = build_calibrated_profile()
    p = dataclasses.replace(calibrated_load_params(cell[0], vessel_deadline_mult=cell[1]),
                            time_contract_v2=True, gate_block_contract=True)
    s = TerminalSimulator(prof, generate_terminal_scenario(prof, seed, p),
                          check_invariants=True)
    s.info_level = LEVEL
    return s


def vessel_slack_min_s(sim) -> float | None:
    """소스 블록의 **LOAD 본선 최소 여유** — 없으면 None (본선 무관 블록)."""
    vals = []
    for v in sim.vessels.values():
        if v.work_type != VesselWorkType.LOAD or v.done:
            continue
        s = v.slack_s(sim.now)
        if s is not None:
            vals.append(s)
    return min(vals) if vals else None


def run_arm(seed_i: int, band: str, *, vessel_guard: bool, log: list | None = None,
            random_block_p: float | None = None) -> dict:
    """arm 실행. vessel_guard=본선 보호 / random_block_p=볼륨 매칭 무작위 차단(대조군)."""
    base_a, base_b = BANDS[band]
    rblock = random.Random(f"y105-rg:{band}:{seed_i}") if random_block_p else None
    mbt = MultiBlockTerminal({"A": _sim(CELL_A, base_a + seed_i),
                              "B": _sim(CELL_B, base_b + seed_i)})
    pol = ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    gens: dict[int, CandidateGenerator] = {}
    stats = {"A->B": 0, "B->A": 0, "blocked_vessel": 0, "rejected": 0, "fired": 0}

    def policy(sim, dp):
        g = gens.setdefault(id(sim), CandidateGenerator())
        gb = {c: g.generate(sim, c, LEVEL) for c in dp.crane_ids}
        try:
            _apply(sim, pol.decide(sim, dp, gb))
        except Exception:
            _apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})

    def review(m: MultiBlockTerminal, t: float):
        cong = {b: block_congestion(s) for b, s in m.blocks.items()}
        vslack = {b: vessel_slack_min_s(s) for b, s in m.blocks.items()}
        for src in list(m.blocks):
            dst = "B" if src == "A" else "A"
            gap = cong[src]["C_zero"] - cong[dst]["C_zero"]
            for jid, rec in list(m.ledger.records.items()):
                if rec.owner != src or not rec.reassignable:
                    continue
                if rec.a_gate_in is None or abs(rec.a_gate_in - t) > 1e-6:
                    continue                       # 이 epoch 의 gate-in 만 (정확 창)
                fire = gap >= THRESH
                if fire:
                    stats["fired"] += 1          # 차단 전 발화 수 (볼륨 매칭 분모)
                vs = vslack[src]
                blocked = bool(vessel_guard and fire and vs is not None
                               and vs < VESSEL_SLACK_MIN)
                if rblock is not None and fire and not blocked:      # 볼륨 매칭 대조군
                    blocked = rblock.random() < random_block_p
                if log is not None:
                    log.append({"t": round(t, 1), "job": jid, "src": src,
                                "gap": round(gap, 4), "vessel_slack_s": None if vs is None
                                else round(vs, 1),
                                **{k: round(v, 4) for k, v in cong[src].items()},
                                "fired": bool(fire and not blocked),
                                "blocked_by_vessel": blocked})
                if blocked:
                    stats["blocked_vessel"] += 1
                    continue
                if not fire:
                    continue
                rng = random.Random(f"y105:{band}:{seed_i}:{src}:{jid}")
                travel = trunc_normal(rng, GATE_BLOCK_MEAN_S,
                                      GATE_BLOCK_SIGMA_S / GATE_BLOCK_MEAN_S,
                                      lo=GATE_BLOCK_MIN_S, hi=GATE_BLOCK_MAX_S)
                if m.try_transfer(jid, dst, route_s=ROUTE_S, travel_s=travel):
                    stats[f"{src}->{dst}"] += 1
                else:
                    stats["rejected"] += 1

    chan = {k: 0.0 for k in CHANNELS}                # YR-106: 채널 분해 누적

    def cost_fn(sim, t0, t1, raw):
        cb = RC.cost_for(interval_start_s=t0, interval_end_s=t1, raw=raw, risk_max=0.0)
        for k, v in channel_split(cb.contributions()).items():
            chan[k] += v
        return cb.total_normalized

    res = mbt.run(policy, review, cost_fn)
    mbt.check_invariants()
    n_moved = stats["A->B"] + stats["B->A"]
    route_cost = res["route_cost_s"] / 3600.0        # 브리지가 계상한 실제 추가주행
    end = res["end"]
    a2o = mbt.ledger.a_to_o_samples_s(end)
    berth = sum(getattr(s.kpis, "berth_overrun_s", 0.0) for s in mbt.blocks.values()) / 60.0
    b2c = [x for s in mbt.blocks.values()
           for x in (s.time_ledger.block_turntime_samples_s() if s.time_ledger else [])]
    done = sum(1 for s in mbt.blocks.values()
               for j in s.jobs.values() if j.status.name == "DONE")
    chan_out = {k: round(v, 4) for k, v in chan.items()}
    chan_out["move"] = round(chan_out["move"] + route_cost, 4)   # 이송 주행은 move 채널
    chan_out["total"] = round(sum(chan_out.values()), 4)
    return {"total": round(res["terminal_total"] + route_cost, 3), "chan": chan_out,
            "backlog": sum(s.unfinished_backlog() for s in mbt.blocks.values()),
            "route_cost": round(route_cost, 3), "berth_min": round(berth, 1),
            "a2o_mean_min": round(fmean(a2o) / 60.0, 2) if a2o else None,
            "b2c_mean_min": round(fmean(b2c) / 60.0, 2) if b2c else None,
            "n_moved": n_moved, "n_blocked_vessel": stats["blocked_vessel"],
            "n_rejected": stats["rejected"], "n_fired": stats["fired"],
            "compl": round(done / max(1, len(mbt.ledger.records)), 4),
            "n_jobs": len(mbt.ledger.records)}


def run(band: str, n_seeds: int = N_SEEDS) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, logs = [], []
    for i in range(n_seeds):
        lg: list = []
        base = run_arm(i, band, vessel_guard=False)
        vg = run_arm(i, band, vessel_guard=True, log=lg)
        fired = vg["n_fired"]                # 차단 전 발화 수 (VGUARD·BASE 동일 규칙)
        p = (vg["n_blocked_vessel"] / fired) if fired else 0.0
        rg = run_arm(i, band, vessel_guard=False, random_block_p=min(1.0, p))
        assert base["n_jobs"] == vg["n_jobs"] == rg["n_jobs"]
        d = round(vg["total"] - base["total"], 3)
        d_r = round(vg["total"] - rg["total"], 3)
        rows.append({"seed": i, "base": base, "vguard": vg, "rguard": rg,
                     "block_p": round(p, 3), "d_total": d, "d_vs_rguard": d_r})
        logs.extend({"seed": i, **r} for r in lg)
        print(f"[{band} s{i}] BASE={base['total']:.2f}(mv {base['n_moved']}) "
              f"VG={vg['total']:.2f}(mv {vg['n_moved']}) RG={rg['total']:.2f}"
              f"(mv {rg['n_moved']}) d={d:+.2f} d_vs_RG={d_r:+.2f} "
              f"berth {base['berth_min']:.0f}→{vg['berth_min']:.0f}", flush=True)
    dts = [r["d_total"] for r in rows]
    # YR-106: 채널 분해 판정 + 하드 guard 기계 검사 + 관심효과 δ 사전지정
    guards = check_guards([{"compl": r[a]["compl"], "backlog": r[a]["backlog"]}
                           for r in rows for a in ("base", "vguard", "rguard")])
    delta = {"truck": 3.0, "vessel": 10.0, "move": 1.0, "other": 1.0, "total": 10.0}
    by_ch_vg = paired_by_channel([r["vguard"]["chan"] for r in rows],
                                 [r["base"]["chan"] for r in rows], delta_interest=delta)
    by_ch_rg = paired_by_channel([r["vguard"]["chan"] for r in rows],
                                 [r["rguard"]["chan"] for r in rows], delta_interest=delta)
    res = {"band": band, "rows": rows, "d_total_ci": _ci_t(dts),
           "yr106_channels_vs_base": by_ch_vg, "yr106_channels_vs_rguard": by_ch_rg,
           "yr106_guards": {"ok": guards.ok, "failures": guards.failures[:5]},
           "yr106_delta_interest": delta,
           "d_vs_rguard_ci": _ci_t([r["d_vs_rguard"] for r in rows]),
           "d_berth": _ci_t([r["vguard"]["berth_min"] - r["base"]["berth_min"] for r in rows]),
           "moved_base": round(fmean(r["base"]["n_moved"] for r in rows), 1),
           "moved_vguard": round(fmean(r["vguard"]["n_moved"] for r in rows), 1),
           "moved_rguard": round(fmean(r["rguard"]["n_moved"] for r in rows), 1),
           "blocked_mean": round(fmean(r["vguard"]["n_blocked_vessel"] for r in rows), 1),
           "compl_min": min(min(r["base"]["compl"], r["vguard"]["compl"], r["rguard"]["compl"])
                            for r in rows),
           "prereg": "①d_total 상한<0 → 본선 보호 필터가 손실원 차단. "
                     "②d_vs_rguard 상한<0 → 이득이 볼륨감소 아닌 본선인지 덕(교락 통제)"}
    (OUT / f"results_{band}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    (OUT / f"decisions_{band}.json").write_text(json.dumps(logs, ensure_ascii=False),
                                                encoding="utf-8")
    print(f"\nYR-105[{band}] d_total CI={res['d_total_ci']} d_berth={res['d_berth']} "
          f"moved {res['moved_base']}→{res['moved_vguard']} blocked={res['blocked_mean']}")
    print(f"[YR-106 채널분해 VGUARD−BASE] guards_ok={guards.ok}")
    for ch, v in by_ch_vg.items():
        print(f"   {ch:7s} {v['mean']:+8.3f} CI[{v['ci'][0]:+7.2f},{v['ci'][1]:+7.2f}] "
              f"MDE {v['mde80']:6.2f} · {v['label']}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="select", choices=list(BANDS))
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    a = ap.parse_args()
    run(a.band, a.seeds)
    print("DONE")
