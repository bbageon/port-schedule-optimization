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
from ..integrated.repro import repro_stamp, vessel_physics_rows
from ..integrated.seedbank import BandSpec, assign_band, independence_report, realization_hash
from ..integrated.scenario_gen import (GATE_BLOCK_MAX_S, GATE_BLOCK_MEAN_S, GATE_BLOCK_MIN_S,
                                       GATE_BLOCK_SIGMA_S, calibrated_load_params,
                                       generate_terminal_scenario, trunc_normal)
from ..integrated.vessel import VesselWorkType
from .yr103_info_sufficiency import _ci_t

RC = RewardCalculator.numeraire_v1()
OUT = Path("outputs/reports/yr105_conditional_transfer")
LEVEL = InformationLevel.PRE_ADVICE
CELL_A, CELL_B = ("high", 0.5), ("mid", 2.0)
# 구 대역 (YR-105-a 정본). **재현 전용** — 새 판정에는 쓰지 않는다:
# ①이미 결과를 열람했고 ②confirm 의 850k 는 yr088_generalize(850000..850004)와 정수 충돌한다.
LEGACY_BANDS = {"select": (847_000, 848_000), "confirm": (849_000, 850_000)}
BANDS = LEGACY_BANDS                   # 하위호환 별칭 (yr109 계측 하네스가 참조)
# 신규 판정 대역 = **실현 지문으로 배정**한다 (YR-108). 시드 산술 금지.
NEW_BAND_START = {"select": 901_000, "confirm": 902_000}
N_SEEDS = 8                            # 구 기본값 (신규 판정은 --seeds 로 24/48)
THRESH, ROUTE_S = 0.10, 180.0
# 본선 보호 경계 = slack_s 의 의미 경계(0 = 계획완료 미달 시작). 튜닝값 아님.
VESSEL_SLACK_MIN = 0.0


# YR-109/게이트A: True 면 마감을 YC→YT→STS 물리 하한 위로 올린다. **CLI 로만 켠다**
# (`--achievable-deadline`) — 모듈 상수 토글은 커밋된 코드로 재현이 안 돼 금지(게이트 B).
ACHIEVABLE_DEADLINE = False
# YR-106: 1차 판정 채널 **사전 고정**. 이걸 고정했으므로 나머지 채널은 탐색적이고
# 다중비교 보정이 필요 없다 — 구두 규약이 아니라 결과 JSON 에 박제된다.
PRIMARY_CHANNEL = "truck"
SD_CONF = 0.80                     # 필요 표본수는 파일럿 sd 의 80% 상측한계 기준


def _params(cell):
    return dataclasses.replace(calibrated_load_params(cell[0], vessel_deadline_mult=cell[1]),
                               time_contract_v2=True, gate_block_contract=True,
                               vessel_deadline_achievable=ACHIEVABLE_DEADLINE)


def _sim(cell, seed):
    prof = build_calibrated_profile()
    s = TerminalSimulator(prof, generate_terminal_scenario(prof, seed, _params(cell)),
                          check_invariants=True)
    s.info_level = LEVEL
    return s


_CELLS = {"A": CELL_A, "B": CELL_B}


def _gen_for_band(key, cell, seed):
    return generate_terminal_scenario(build_calibrated_profile(), seed, _params(cell))


def _legacy_hashes(n: int = 8) -> set[str]:
    """구 대역(선택·확증)의 실현 지문 — 이미 **결과를 열람**했으므로 재사용 금지."""
    out = set()
    for a, b in LEGACY_BANDS.values():
        for key, base in (("A", a), ("B", b)):
            for i in range(n):
                out.add(realization_hash(_gen_for_band(key, _CELLS[key], base + i)))
    return out


def resolve_band(band: str, n_seeds: int) -> tuple[dict[str, list[int]], dict]:
    """대역 → (셀별 절대 시드, 사전등록에 박제할 독립성 블록).

    `legacy-*` 는 구 판정 재현 전용(산술 배정). 그 외는 **실현 지문 배정**이다.
    """
    if band.startswith("legacy-"):
        a, b = LEGACY_BANDS[band.split("-", 1)[1]]
        seeds = {"A": [a + i for i in range(n_seeds)], "B": [b + i for i in range(n_seeds)]}
        spec = BandSpec(family=band, seeds=seeds,
                        hashes={k: [realization_hash(_gen_for_band(k, _CELLS[k], s))
                                    for s in v] for k, v in seeds.items()})
        rep = independence_report(spec)
        return seeds, {**spec.freeze_json(), "independence": rep,
                       "note": "구 대역 재현 — 실현 독립성 미보장(YR-108 이전 관행)"}
    excl = _legacy_hashes()
    prior = {}
    if band == "confirm":               # 확증은 선택 대역과 **실현이 겹치면 안 된다**
        sel, _ = resolve_band("select", n_seeds // 2 if n_seeds >= 2 else 1)
        prior = {k: {realization_hash(_gen_for_band(k, _CELLS[k], s)) for s in v}
                 for k, v in sel.items()}
        excl |= {h for hs in prior.values() for h in hs}
    spec = assign_band(family=f"y105-{band}", cells=_CELLS, n=n_seeds,
                       generate=_gen_for_band, exclude=excl,
                       start_seed=NEW_BAND_START[band])
    rep = independence_report(spec, forbidden={"구 대역(열람됨)": _legacy_hashes(),
                                               **{f"선택대역": v for v in prior.values()}}
                              if prior else {"구 대역(열람됨)": _legacy_hashes()})
    if not rep["ok"]:
        raise AssertionError(f"실현 독립성 위반: {rep}")
    return spec.seeds, {**spec.freeze_json(), "independence": rep}


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
            random_block_p: float | None = None, seeds: dict[str, int] | None = None,
            gap_threshold: float | None = None) -> dict:
    """arm 실행. vessel_guard=본선 보호 / random_block_p=볼륨 매칭 무작위 차단(대조군).

    seeds: 셀별 **절대 시드**. 생략하면 구 대역 산술(재현 전용).
    """
    if seeds is None:
        base_a, base_b = LEGACY_BANDS[band]
        seeds = {"A": base_a + seed_i, "B": base_b + seed_i}
    threshold = THRESH if gap_threshold is None else float(gap_threshold)
    rblock = random.Random(f"y105-rg:{band}:{seed_i}") if random_block_p else None
    mbt = MultiBlockTerminal({"A": _sim(CELL_A, seeds["A"]),
                              "B": _sim(CELL_B, seeds["B"])})
    pol = ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    gens: dict[int, CandidateGenerator] = {}
    stats = {"A->B": 0, "B->A": 0, "blocked_vessel": 0, "rejected": 0, "fired": 0,
             # 게이트 D/110: 정책 예외를 WAIT 로 삼키면서 세지 않으면, 정책이 매 결정
             # 실패해도 런은 정상 종료하고 결과 JSON 이 조용히 완성된다(평가계약 0순위의
             # "검열 회피"가 겨냥한 실패 양식인데 측정 수단이 없었다).
             "policy_exceptions": 0, "decisions": 0}

    def policy(sim, dp):
        g = gens.setdefault(id(sim), CandidateGenerator())
        gb = {c: g.generate(sim, c, LEVEL) for c in dp.crane_ids}
        stats["decisions"] += 1
        try:
            _apply(sim, pol.decide(sim, dp, gb))
        except Exception:
            stats["policy_exceptions"] += 1
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
                fire = gap >= threshold
                if fire:
                    stats["fired"] += 1          # 차단 전 발화 수 (볼륨 매칭 분모)
                vs = vslack[src]
                blocked = bool(vessel_guard and fire and vs is not None
                               and vs < VESSEL_SLACK_MIN)
                if rblock is not None and fire and not blocked:      # 볼륨 매칭 대조군
                    blocked = rblock.random() < random_block_p
                log_rec = None
                if log is not None:
                    log_rec = {"t": round(t, 1), "job": jid, "src": src,
                               "gap": round(gap, 4), "gap_threshold": threshold,
                               "vessel_slack_s": None if vs is None else round(vs, 1),
                               **{k: round(v, 4) for k, v in cong[src].items()},
                               "fired": bool(fire and not blocked),
                               "blocked_by_vessel": blocked, "transferred": False}
                    log.append(log_rec)
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
                    if log_rec is not None:
                        log_rec["transferred"] = True
                        log_rec["dst"] = dst
                else:
                    stats["rejected"] += 1
                    if log_rec is not None:
                        log_rec["rejected"] = True

    chan = {k: 0.0 for k in CHANNELS}                # YR-106: 채널 분해 누적
    # 게이트 B: 블록별로도 나눠 둔다 — 구판은 A·B 를 한 dict 에 합산해 블록별 채널 분해가
    # 사후에 불가능했다(어느 블록에서 이득이 났는지 원자료로 확인할 수 없었다).
    chan_by_block = {b: {k: 0.0 for k in CHANNELS} for b in ("A", "B")}
    _bid = {id(s): b for b, s in mbt.blocks.items()}

    def cost_fn(sim, t0, t1, raw):
        cb = RC.cost_for(interval_start_s=t0, interval_end_s=t1, raw=raw, risk_max=0.0)
        b = _bid.get(id(sim))
        for k, v in channel_split(cb.contributions()).items():
            chan[k] += v
            if b is not None:
                chan_by_block[b][k] += v
        return cb.total_normalized

    res = mbt.run(policy, review, cost_fn)
    mbt.check_invariants()
    n_moved = stats["A->B"] + stats["B->A"]
    route_cost = res["route_cost_s"] / 3600.0        # 브리지가 계상한 실제 추가주행
    end = res["end"]
    a2o = mbt.ledger.a_to_o_samples_s(end)
    n_a2o_expected = sum(
        1 for rec in mbt.ledger.records.values() if rec.a_gate_in is not None)
    berth = sum(getattr(s.kpis, "berth_overrun_s", 0.0) for s in mbt.blocks.values()) / 60.0
    b2c = [x for s in mbt.blocks.values()
           for x in (s.time_ledger.block_turntime_samples_s() if s.time_ledger else [])]
    done = sum(1 for s in mbt.blocks.values()
               for j in s.jobs.values() if j.status.name == "DONE")
    chan_out = {k: round(v, 4) for k, v in chan.items()}
    chan_out["move"] = round(chan_out["move"] + route_cost, 4)   # 이송 주행은 move 채널
    chan_out["total"] = round(sum(chan_out.values()), 4)
    total_raw = float(res["terminal_total"] + route_cost)
    a2o_mean_min_raw = (fmean(a2o) / 60.0) if a2o else None
    return {"total": round(total_raw, 3), "total_raw": total_raw, "chan": chan_out,
            "chan_by_block": {b: {k: round(v, 4) for k, v in d.items()}
                              for b, d in chan_by_block.items()},
            "policy_exceptions": stats["policy_exceptions"], "decisions": stats["decisions"],
            # YR-112: 간섭 교착 탈출이 이 런에 개입했는지 — arm 마다 저장해 감사 가능하게 한다
            # (0 이면 결과는 탈출 로직과 무관하다는 뜻).
            "deadlock_escapes": sum(s.deadlock_escape_count for s in mbt.blocks.values()),
            "backlog": sum(s.unfinished_backlog() for s in mbt.blocks.values()),
            "route_cost": round(route_cost, 3), "berth_min": round(berth, 1),
            # 판정에는 full precision을 쓰고, 사람이 보는 요약만 반올림한다.
            "a2o_mean_min": round(a2o_mean_min_raw, 2)
            if a2o_mean_min_raw is not None else None,
            "a2o_mean_min_raw": a2o_mean_min_raw, "n_a2o": len(a2o),
            # 전체 원장에는 본선 작업도 있으므로 n_jobs와 직접 비교하면 안 된다.
            # A→O 계약은 실제 gate-in 기록이 있는 원장 건수와 따로 맞춘다.
            "n_a2o_expected": n_a2o_expected,
            "b2c_mean_min": round(fmean(b2c) / 60.0, 2) if b2c else None,
            "n_moved": n_moved, "n_blocked_vessel": stats["blocked_vessel"],
            "n_rejected": stats["rejected"], "n_fired": stats["fired"],
            "compl": round(done / max(1, len(mbt.ledger.records)), 4),
            "n_jobs": len(mbt.ledger.records), "gap_threshold": threshold}


def run(band: str, n_seeds: int = N_SEEDS) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, logs = [], []
    band_seeds, band_block = resolve_band(band, n_seeds)
    for i in range(n_seeds):
        lg: list = []
        lg_r: list = []
        sd = {"A": band_seeds["A"][i], "B": band_seeds["B"][i]}
        base = run_arm(i, band, vessel_guard=False, seeds=sd)
        vg = run_arm(i, band, vessel_guard=True, log=lg, seeds=sd)
        fired = vg["n_fired"]                # 차단 전 발화 수 (VGUARD·BASE 동일 규칙)
        p = (vg["n_blocked_vessel"] / fired) if fired else 0.0
        # 게이트 B: 대조군 결정도 로깅한다 — 두 번째 주판정이 교락 통제(RGUARD 대비)인데
        # 정작 RGUARD 가 무엇을 막았는지 기록이 없었다.
        rg = run_arm(i, band, vessel_guard=False, random_block_p=min(1.0, p), log=lg_r,
                     seeds=sd)
        assert base["n_jobs"] == vg["n_jobs"] == rg["n_jobs"]
        d = round(vg["total"] - base["total"], 3)
        d_r = round(vg["total"] - rg["total"], 3)
        rows.append({"seed": i, "seed_A": sd["A"], "seed_B": sd["B"],
                     "base": base, "vguard": vg, "rguard": rg,
                     "physics_A": vessel_physics_rows(
                         generate_terminal_scenario(build_calibrated_profile(),
                                                    sd["A"], _params(CELL_A))),
                     "physics_B": vessel_physics_rows(
                         generate_terminal_scenario(build_calibrated_profile(),
                                                    sd["B"], _params(CELL_B))),
                     "block_p": round(p, 3), "d_total": d, "d_vs_rguard": d_r})
        logs.extend({"seed": i, "arm": "VGUARD", **r} for r in lg)
        logs.extend({"seed": i, "arm": "RGUARD", **r} for r in lg_r)
        print(f"[{band} s{i}] BASE={base['total']:.2f}(mv {base['n_moved']}) "
              f"VG={vg['total']:.2f}(mv {vg['n_moved']}) RG={rg['total']:.2f}"
              f"(mv {rg['n_moved']}) d={d:+.2f} d_vs_RG={d_r:+.2f} "
              f"berth {base['berth_min']:.0f}→{vg['berth_min']:.0f}", flush=True)
    dts = [r["d_total"] for r in rows]
    # YR-106: 채널 분해 판정 + 하드 guard 기계 검사 + 관심효과 δ 사전지정
    guards = check_guards([{"compl": r[a]["compl"], "backlog": r[a]["backlog"]}
                           for r in rows for a in ("base", "vguard", "rguard")])
    # 게이트 110: 정책 예외가 하나라도 있으면 판정 무효 — 결과는 지우지 않고 표기만 한다
    # (시드 제외는 평가계약 0순위가 금지한 "검열 회피" 그 자체이므로 쓰지 않는다).
    n_exc = sum(r[a]["policy_exceptions"] for r in rows for a in ("base", "vguard", "rguard"))
    if n_exc:
        guards.ok = False
        guards.failures.append(f"정책 예외 {n_exc}건 — 결정이 WAIT 로 대체됨(판정 무효)")
    delta = {"truck": 3.0, "vessel": 10.0, "move": 1.0, "other": 1.0, "total": 10.0}
    by_ch_vg = paired_by_channel([r["vguard"]["chan"] for r in rows],
                                 [r["base"]["chan"] for r in rows], delta_interest=delta,
                                 primary=PRIMARY_CHANNEL, sd_conf=SD_CONF)
    by_ch_rg = paired_by_channel([r["vguard"]["chan"] for r in rows],
                                 [r["rguard"]["chan"] for r in rows], delta_interest=delta,
                                 primary=PRIMARY_CHANNEL, sd_conf=SD_CONF)
    res = {"repro": repro_stamp(
               experiment="YR-105 조건부 창중 이송 — 본선 악화 방지 필터 단일축",
               seeds=band_seeds,
               params={"cell_A": _params(CELL_A), "cell_B": _params(CELL_B)},
               profile_id=build_calibrated_profile().terminal_id,
               prereg=f"1차 판정 채널 = {PRIMARY_CHANNEL} (사전 고정 → 다중비교 보정 불요). "
                      "①VGUARD−BASE 트럭 채널 상한<0 ②교락 통제 VGUARD−RGUARD.",
               extra={"delta_interest": delta, "sd_conf": SD_CONF, "band": band,
                      "achievable_deadline": ACHIEVABLE_DEADLINE,
                      "primary_channel": PRIMARY_CHANNEL,
                      "band_spec": band_block}),
           "verdict_valid": guards.ok,
           "band": band, "rows": rows, "d_total_ci": _ci_t(dts),
           "yr106_channels_vs_base": by_ch_vg, "yr106_channels_vs_rguard": by_ch_rg,
           "yr106_guards": {"ok": guards.ok, "failures": guards.failures[:5]},
           "yr106_delta_interest": delta,
           "d_vs_rguard_ci": _ci_t([r["d_vs_rguard"] for r in rows]),
           "d_berth": _ci_t([r["vguard"]["berth_min"] - r["base"]["berth_min"] for r in rows]),
           # YR-112 감사: 탈출이 한 번도 발화하지 않았으면 판정은 탈출 로직과 무관하다
           "deadlock_escapes_total": sum(r[a]["deadlock_escapes"]
                                         for r in rows for a in ("base", "vguard", "rguard")),
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
    print(f"[YR-106 채널분해 VGUARD−BASE] guards_ok={guards.ok} "
          f"(1차 판정 채널 = {PRIMARY_CHANNEL})")
    for ch, v in by_ch_vg.items():
        print(f"   {ch:7s} {v['mean']:+8.3f} CI[{v['ci'][0]:+7.2f},{v['ci'][1]:+7.2f}] "
              f"MDE {v['mde80']:6.2f} 필요n {v['required_n_for_delta']} "
              f"[{v.get('role', '-')}] · {v['label']}")
    if not guards.ok:
        print("\n*** 판정 무효 (하드 guard 실패) — 원자료는 보존, 판정만 사용 금지 ***")
        for f in guards.failures[:5]:
            print("   ·", f)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="select",
                    choices=["select", "confirm", "legacy-select", "legacy-confirm"])
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--achievable-deadline", action="store_true",
                    help="본선 마감을 YC→YT→STS 물리 하한 위로 (YR-109/게이트A)")
    a = ap.parse_args()
    ACHIEVABLE_DEADLINE = a.achievable_deadline      # 모듈 전역 재바인딩 (CLI 가 유일한 토글)
    r = run(a.band, a.seeds)
    print("DONE" if r["verdict_valid"] else "DONE (INVALID)")
    raise SystemExit(0 if r["verdict_valid"] else 2)
