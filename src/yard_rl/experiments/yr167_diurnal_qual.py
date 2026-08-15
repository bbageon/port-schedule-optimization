"""YR-167 — 5차 계약(도착률·24시간 이중 피크) **자격 시험**.

39차 감사 보정판: 1차 자격의 "11/11 PASS" 는 **허위 통과**였다. 꼬리 2대 누락을
통과시켰고, 엔진이 실제로 쓰는 프로파일을 검사하지 않았고, W4·W5·W8′ 이 이름만
있는 검사였다. 이 판은 전부 실측으로 바꾼다:

  W1′ 명단 준수 — 계획=투입=등록=3,600 **하드**, SKIP=0·SKIP_TAIL=0, 시각 전건 일치
  W2 장부 보존 — 수 일치 + 중복 0 + 완료분 A≤B≤S≤C≤O
  W3 결정론 — 빌드(명단·시나리오) + **런 digest**(같은 시드 반복 셀과 대조)
  W4 본선 — 30스트림 분산 + **동시 활성 실측**(설계 c=6 은 평균) + 실현 작업률 앵커
  W5 스냅샷 — 5분 격자 × 21블록 **실제 수집**(재공·대기열·backlog·재고·여유칸)
  W6 프로파일 — **엔진 실물**이 H-21 YT 인지 (시나리오 생성 프로파일과 동일)
  W7 내부타당성 4종 / W8′ 재고 소진 — **런타임** 재고 하한·취소 0 (W9 와 분리)
  W9 생성단계 flow fallback 0 / W11 피크 도달(계획 시간대별 최대 ±10%)

성능 주장 없음 — 환경 자격이다. 측정창은 **실행 24h 중 22h**(워밍업 2h 제외,
사전등록 §E 동결)이며 부수 관측의 분모를 명시한다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from ..integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference,
                                    _apply, _wait_of)
from ..integrated.candidates import CandidateGenerator
from ..integrated.multiblock import MultiBlockTerminal
from ..integrated.policy_config import LEGACY_DEFAULT
from ..integrated.profiles import build_h21_profile
from ..integrated.repro import code_dirty
from ..integrated.sell_review import ANNOUNCE_LEAD_S
from ..integrated.terminal_stream import (DIURNAL_DAY_TOTAL,
                                          DIURNAL_VESSEL_CONCURRENCY,
                                          DIURNAL_VESSEL_STREAMS, OBS_24H,
                                          ObservationContract, ScheduledAnnouncer,
                                          TerminalStreamParams, admission_epochs,
                                          build_diurnal, diurnal_rate,
                                          ensure_time_ledger)
from ..integrated.yard_layout import terminal_layout
from .yr088_joint_rl import LEVEL
from .yr149_load_cells import _sim_from
from .yr150_h21_pilot import _git, _sha256
from .yr167_observers import (SnapshotCollector, cancelled_external, run_digest,
                              stock_conservation, vessel_concurrency,
                              vessel_realized)

OUT = Path("outputs/reports/yr167_diurnal_qual")
PREREG = Path(".claude/docs/strategy-history/"
              "2026-08-11-24시간-이중피크-상수-유도-사전등록.md")
SEED = 8_100_000                 # 5차 자격 전용 시드 대역
REPS = 3                         # 독립 시드 3
SLA_PENALTY_S = 2_580.0          # 벌금 임계(= SLA_ANCHOR 780 + long_wait 1800)
W11_TOL = 0.10
VESSEL_RATE_ANCHOR = (145.0, 170.0)   # 외부 앵커 vessel_workload (moves/h)
CONCURRENCY_TOL = 0.5                 # 평균 동시 활성 허용 오차 (설계값 c=6)


def planned_hourly_peak(obs: ObservationContract, total: int) -> float:
    """계획 시간대별 최대 도착 — 동결 λ(t) 에서 유도(사전등록 A7)."""
    hourly = [sum(diurnal_rate((h + i / 60) * 3600, day_s=obs.observe_s,
                               total=total) * 60 for i in range(60))
              for h in range(int(obs.observe_s // 3600))]
    return max(hourly)


def _w12_day_plan(plan, schedule: list[dict]) -> dict:
    """공개 예약 장부가 **명단에서 왔고 실현에 반응하지 않았는가**.

    장부를 안 붙인 구 계약 런에서는 `attached=False` 로만 남긴다(판정 제외).
    """
    if plan is None:
        return {"attached": False}
    mismatch = sum(1 for e in schedule
                   if plan.gate_in(e["job_id"]) != float(e["arrival_s"]))
    return {"attached": True,
            "n_jobs": plan.n_jobs,
            "n_schedule": len(schedule),
            "count_matches": plan.n_jobs == len(schedule),
            # SF 자격런은 판매를 하지 않는다 — 장부가 스스로 움직이면 결함이다
            "n_reschedules": len(plan.reschedules),
            "no_self_reschedule": len(plan.reschedules) == 0,
            # 재예약이 0이므로 장부 = 명단이어야 한다(실현 진입이 아니라 예약에서 옴)
            "gate_in_mismatch_n": mismatch,
            "matches_schedule": mismatch == 0,
            "plan_version": plan.plan_version,
            "version_zero": plan.plan_version == 0}


def run_cell(rep: int, *, day_plan_public: bool = False) -> dict:
    obs = OBS_24H
    prof = build_h21_profile()
    layout = terminal_layout()
    params = TerminalStreamParams(load_4h=DIURNAL_DAY_TOTAL)
    built = build_diurnal(prof, SEED + rep, obs=obs, layout=layout, params=params,
                          background_seed=SEED)
    # 5차 계약: 초기 트럭 0 → v2 장부를 런너에서 활성화(엔진 골든 경로 불변).
    # 프로파일은 **시나리오를 생성한 것과 같은 것**을 넘긴다 (39차 감사 — 구판은
    # _sim_from 이 AGV 보정 프로파일을 다시 로드해 H-21 계약과 어긋났다).
    mbt = MultiBlockTerminal(
        {b: ensure_time_ledger(_sim_from(s, prof))
         for b, s in built["scenarios"].items()},
        extra_review_epochs=admission_epochs(obs))
    # ★YR-171-A 재자격 — 하루 공개 예약 장부. 정보 계약이 바뀌므로 자격을 다시 받는다.
    # 장부는 **읽기 전용**이라 엔진에 영향이 없어야 한다: W12 가 그것을 판정한다.
    day_plan = None
    if day_plan_public:
        from ..integrated.day_plan import attach as _attach_day_plan
        day_plan = _attach_day_plan(mbt, built["schedule"])
    ann = ScheduledAnnouncer(built["schedule"], lead_s=ANNOUNCE_LEAD_S,
                             end_s=built["sim_end_s"])
    snap = SnapshotCollector(obs.snapshot_times())
    pol = ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    gens: dict[int, CandidateGenerator] = {}
    exc = {"n": 0}

    def policy(sim, dp):
        g = gens.setdefault(id(sim), CandidateGenerator(config=LEGACY_DEFAULT))
        gb = {c: g.generate(sim, c, LEVEL) for c in dp.crane_ids}
        try:
            _apply(sim, pol.decide(sim, dp, gb))
        except Exception:
            exc["n"] += 1
            _apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})

    def review(m, t):
        ann.review(m, t)
        snap.observe(m, t)

    mbt.run(policy, review_fn=review)
    try:
        mbt.check_invariants()          # 소유권·개수 보존 (터미널 전역 원장)
        invariants = True
    except Exception:
        invariants = False
    # **물리 제약** — 엔진 자체 검사(단 적재 초과·컨테이너 분실·통로/순서 위반).
    # 런 중 자동 호출되므로 위반 시 셀이 죽지만, "통과했다"를 증거로 남기려면
    # 종료 상태에서 명시적으로 한 번 더 부른다 (게이트 internal_checks 계약 항목).
    physical = True
    for _sim in mbt.blocks.values():
        try:
            _sim.check_invariants()
        except Exception:
            physical = False

    # ---- 장부 수집 ----
    rows = []
    for bid, sim in mbt.blocks.items():
        tl = getattr(sim, "time_ledger", None)
        if tl is None:
            continue
        for jid, r in tl.records.items():
            rows.append({"job": jid, "block": bid, "a": r.gate_in,
                         "b": r.block_arrival, "s": r.service_start,
                         "c": r.job_done, "o": r.gate_out})
    plan = {e["job_id"]: e["arrival_s"] for e in built["schedule"]}
    admitted = [e["job_id"] for e in ann.ledger if e["event"] == "ADMIT"]
    tail = [e for e in ann.ledger if e["event"] == "SKIP_TAIL"]
    skips = [e for e in ann.ledger if e["event"] == "SKIP"]
    n_plan = len(built["schedule"])

    # ---- W1′ 명단 준수 (하드) ----
    ids = {r["job"] for r in rows}
    w1_counts = (n_plan == built["day_total"] == ann.n_admitted == len(rows)
                 and len(skips) == 0 and len(tail) == 0
                 and ids == set(plan) == set(admitted))
    w1_times = all(r["a"] is not None and abs(r["a"] - plan[r["job"]]) < 1e-6
                   for r in rows if r["job"] in plan)
    w1 = bool(w1_counts and w1_times)

    # ---- W2 장부 보존 ----
    def _ordered(r):
        seq = [r["a"], r["b"], r["s"], r["c"], r["o"]]
        seq = [x for x in seq if x is not None]
        return all(seq[i] <= seq[i + 1] + 1e-6 for i in range(len(seq) - 1))
    n_done = sum(1 for r in rows if r["o"] is not None)
    # 완료(O)가 있으면 B·S·C 도 반드시 있어야 한다 — "블록 도착 없이 출문" 같은 모순 차단
    complete_chain = all(not (r["o"] is not None
                              and (r["b"] is None or r["s"] is None or r["c"] is None))
                         for r in rows)
    # **면적 항등식** — 장부의 존재 이유. 런 중 수기 삽입(_a_sorted/_n_inside 포인터
    # 보정)이 틀어지면 여기서 드러난다. 등록 수 대조만으로는 절대 못 잡는다.
    area_err = 0.0
    for sim in mbt.blocks.values():
        tl = sim.time_ledger
        if tl is None:
            continue
        expect = sum(tl.terminal_turntime_samples_s())
        area_err = max(area_err, abs(tl.terminal_area_s - expect))
    w2 = bool(len(rows) == ann.n_admitted == len(ids)      # 중복 등록 없음
              and all(_ordered(r) for r in rows) and complete_chain
              and area_err <= 1e-6)

    # ---- W4 본선 ----
    vs = built["vessel_schedule"]
    dur = params.vessel_moves * params.sts_move_interval_s
    conc = vessel_concurrency(vs, dur, obs.observe_s)
    vreal = vessel_realized(mbt, params.vessel_moves, obs.observe_s)
    # 평탄 상한 = 동결 배치(등간격)에서 유도 — 사후 조정값이 아니다.
    # **평균은 판별력이 없다**(총 본선시간÷창 이라 배치를 바꿔도 불변):
    # 30척을 한 시각에 몰아도 평균 6.0 이 나온다 → min(끊김)·max(상한)가 실제 검사다.
    spacing = (obs.observe_s - dur) / max(1, DIURNAL_VESSEL_STREAMS - 1)
    conc_max_expect = int(dur // spacing) + 1
    w4 = bool(len(vs) == DIURNAL_VESSEL_STREAMS
              and len({v["block"] for v in vs}) == len(layout.ids)
              and conc["min"] >= 1                      # 본선 작업이 끊기지 않음
              and conc["max"] <= conc_max_expect        # 몰아주기 배치 탐지
              and abs(conc["mean"] - DIURNAL_VESSEL_CONCURRENCY) <= CONCURRENCY_TOL
              and VESSEL_RATE_ANCHOR[0] <= vreal["realized_moves_per_h"]
              <= VESSEL_RATE_ANCHOR[1]
              and vreal["completion_ratio"] is not None
              and vreal["completion_ratio"] >= 0.99)

    # ---- W5 스냅샷 (실측) ----
    ssum = snap.summary()
    w5 = bool(ssum.get("complete")
              and ssum.get("n_rows") == len(obs.snapshot_times()) * len(mbt.blocks)
              and not ssum.get("degenerate"))

    # ---- W6 프로파일 (엔진 실물) ----
    engine_profiles = {(s.profile.terminal_id, s.profile.transfer.kind)
                       for s in mbt.blocks.values()}
    w6 = engine_profiles == {(prof.terminal_id, prof.transfer.kind)} and \
        prof.transfer.kind == "YT"

    # ---- W8′ 재고 소진 (런타임) + 재고 보존 항등식 ----
    n_cancel = cancelled_external(mbt)
    cons = stock_conservation(mbt, snap)
    w8 = bool(ssum.get("block_stock_min", 0) > 0
              and ssum.get("block_free_min", 0) > mbt.capacity_margin
              and n_cancel == 0
              and cons["ok"]
              and built["flow_fallbacks_total"] == 0)

    # ---- W11 피크 도달 ----
    hourly = [0] * int(obs.observe_s // 3600)
    for r in rows:
        h = int(r["a"] // 3600)
        if 0 <= h < len(hourly):
            hourly[h] += 1
    peak_obs = max(hourly)
    peak_plan = planned_hourly_peak(obs, built["day_total"])
    w11 = abs(peak_obs - peak_plan) <= W11_TOL * peak_plan

    # ---- W3 빌드 결정론 (런 결정론은 반복 셀 digest 대조) ----
    again = build_diurnal(prof, SEED + rep, obs=obs, layout=layout, params=params,
                          background_seed=SEED)
    det_build = (again["schedule"] == built["schedule"]
                 and again["vessel_schedule"] == built["vessel_schedule"])

    # ---- W7 내부타당성 ----
    # 공개 예측 = 통지 + 기대 주행 (구성상 항등) **그리고** 공개 예측이 실현 도착과
    # 일치하지 않는 트럭이 존재 = 진실(주행 잔차)이 정책에 새지 않았다는 실측 증거.
    # 정책이 읽는 값은 provided_eta 이므로 그것도 함께 본다. 필드 결측을 면제로
    # 취급하던 truthy 가드(fail-open)를 is not None 으로 바꾼다.
    leak_n = leak_exact = leak_missing = 0
    leak_free = True
    for bid, sim in mbt.blocks.items():
        base = layout.gate_to_block_s(bid)
        for j in sim.jobs.values():
            if not j.is_external_truck:
                continue
            nt = getattr(j, "notified_gate_in_s", None)
            if nt is None or j.estimated_block_arrival is None:
                leak_missing += 1
                continue
            leak_n += 1
            if (abs(j.estimated_block_arrival - (nt + base)) > 1e-6
                    or j.provided_eta is None
                    or abs(j.provided_eta - j.estimated_block_arrival) > 1e-6):
                leak_free = False
            if (j.actual_block_arrival is not None
                    and abs(j.estimated_block_arrival - j.actual_block_arrival) < 1e-9):
                leak_exact += 1
    # 정책 자체가 미래 정보(실현 진실값) 사용을 선언했는지 — 선언 정책은 자격 무효.
    # 정보경계 검사는 데이터만 보므로, 진실값을 읽는 정책(FIFO·오라클)을 못 잡는다.
    def _future(o) -> bool:
        return bool(getattr(o, "USES_FUTURE_INFORMATION", False))
    uses_future = (_future(pol) or _future(getattr(pol, "resolver", None))
                   or _future(getattr(getattr(pol, "resolver", None),
                                      "preference", None)))
    achievable = all(
        v.plan.phys_min_completion_s is not None
        and v.plan.planned_completion_s is not None
        and v.plan.planned_completion_s >= v.plan.phys_min_completion_s - 1e-6
        for s in built["scenarios"].values() for v in s.vessels)

    # ---- 부수 관측 (판정 아님) — 분모를 명시한다 ----
    # **도착 코호트**가 정본이다: 측정창 안에 게이트인한 트럭 중 완료분(배수 구간에서
    # 끝난 것 포함). 구판은 출문 시각으로 골라 워밍업 도착분이 섞이고 워밍업 안에서
    # 끝난 트럭이 빠졌다 (39차 감사 주장 E). 두 분모를 모두 보고한다.
    coh = [r for r in rows if r["a"] is not None
           and obs.warmup_s <= r["a"] < obs.observe_s]
    fin = [r for r in coh if r["o"] is not None]
    stay = [r["o"] - r["a"] for r in fin]
    penal = [s for s in stay if s > SLA_PENALTY_S]
    old_fin = [r for r in rows if r["o"] is not None and r["a"] is not None
               and obs.warmup_s <= r["o"] < obs.observe_s]
    old_stay = [r["o"] - r["a"] for r in old_fin]
    by_hour: dict[int, list[float]] = {}
    for r in fin:
        by_hour.setdefault(int(r["a"] // 3600), []).append(r["o"] - r["a"])

    return {
        "rep": rep, "seed": SEED + rep,
        "profile_built": {"terminal_id": prof.terminal_id,
                          "transfer_kind": prof.transfer.kind},
        "profile_engine": sorted(f"{a}/{b}" for a, b in engine_profiles),
        "window": {"run_s": built["sim_end_s"], "observe_s": obs.observe_s,
                   "warmup_s": obs.warmup_s, "measure_s": obs.measure_s,
                   "note": "실행 24h + 배수 2h, 측정은 [2h, 24h) 22시간"},
        "n_plan": n_plan, "n_admitted": ann.n_admitted,
        "n_tail_skipped": len(tail), "n_skips": len(skips),
        "n_registered": len(rows), "n_completed": n_done,
        "n_cancelled_external": n_cancel,
        "W1p_schedule_honored": w1,
        "W1p_detail": {"counts_ok": bool(w1_counts), "times_ok": bool(w1_times)},
        "W2_ledger_conserved": w2,
        "W3_deterministic_build": bool(det_build),
        "W4_vessels": w4,
        "W4_detail": {"concurrency": conc, "realized": vreal,
                      "concurrency_max_expected": conc_max_expect,
                      "mean_is_invariant_note": "평균은 배치 불변량 — min·max 가 판별항"},
        "W5_snapshots": w5, "W5_detail": ssum,
        "W6_profile_h21_yt_engine": bool(w6),
        "W7_internal": {"event_time_order": all(_ordered(r) for r in rows),
                        "record_chain_complete": bool(complete_chain),
                        "information_boundary": bool(leak_free and leak_n > 0
                                                     and leak_missing == 0),
                        "no_truth_leak": bool(leak_n > 0 and leak_exact < leak_n),
                        "policy_declares_no_future_info": not uses_future,
                        "information_boundary_n": leak_n,
                        "information_boundary_missing_n": leak_missing,
                        "estimate_equals_truth_n": leak_exact,
                        "ownership_conservation": invariants,
                        "physical_constraints": physical,
                        "achievable_vessel_deadline": achievable},
        "W2_area_error_s": round(area_err, 9),
        "W8p_no_stock_exhaustion": w8, "W8p_conservation": cons,
        "W9_flow_fallback_zero": built["flow_fallbacks_total"] == 0,
        "W11_peak_reached": bool(w11),
        "peak_observed": peak_obs, "peak_planned": round(peak_plan, 1),
        "policy_exceptions": exc["n"],
        "run_digest": run_digest(mbt),
        # ---- W12 (YR-171-A 신규) — 공개 예약 장부의 정보 경계 ----
        # ①장부가 명단과 정확히 일치하는가(실현값이 아니라 예약에서 왔는가)
        # ②SF 자격런은 판매를 하지 않으므로 재예약이 0이어야 한다(장부 무개입 증명)
        # ③엔진 영향 없음은 `run_digest` 를 구 계약 자격런과 대조해 판정한다(합산 단계)
        "W12_day_plan": _w12_day_plan(day_plan, built["schedule"]),
        # 부수 관측 (판정 아님)
        "obs_denominator": "측정창 [2h,24h) 안에 게이트인한 트럭 중 완료분(도착 코호트)",
        "obs_n_cohort": len(coh), "obs_n_sample": len(stay),
        "obs_legacy_gateout_n": len(old_stay),
        "obs_legacy_gateout_mean_stay_s": (round(fmean(old_stay), 1)
                                           if old_stay else None),
        "obs_legacy_gateout_penalty_share": (
            round(sum(1 for s in old_stay if s > SLA_PENALTY_S) / len(old_stay), 4)
            if old_stay else None),
        "obs_mean_stay_s": round(fmean(stay), 1) if stay else None,
        "obs_max_stay_s": round(max(stay), 1) if stay else None,
        "obs_penalty_share": round(len(penal) / len(stay), 4) if stay else None,
        "obs_hourly_arrivals": hourly,
        "obs_hourly_mean_stay_s": {h: round(fmean(v), 1)
                                   for h, v in sorted(by_hour.items())},
        "obs_hourly_terminal_wip": snap.hourly_terminal_wip(),
    }


CHECKS = ("W1p_schedule_honored", "W2_ledger_conserved", "W3_deterministic_build",
          "W4_vessels", "W5_snapshots", "W6_profile_h21_yt_engine",
          "W8p_no_stock_exhaustion", "W9_flow_fallback_zero", "W11_peak_reached")


def run(cells: list[dict] | None = None, repeat: dict | None = None,
        *, out_dir: Path | None = None, day_plan_public: bool = False) -> dict:
    """cells 가 주어지면 병렬 셀 산출물을 합산(재실행 없음), 아니면 순차 실행."""
    out_dir = out_dir or OUT
    from_files = cells is not None
    cells = cells if from_files else [run_cell(r) for r in range(REPS)]
    checks = {k: all(c[k] for c in cells) for k in CHECKS}
    checks["W7_internal_checks_pass"] = all(
        all(v for k, v in c["W7_internal"].items() if isinstance(v, bool))
        for c in cells)
    checks["no_policy_exceptions"] = all(c["policy_exceptions"] == 0 for c in cells)
    # ---- W12 (YR-171-A) — 공개 예약 장부의 정보 경계 ----
    # ①장부 = 명단(실현이 아니라 예약에서 옴) ②자격런은 판매가 없으므로 재예약 0
    # ③**엔진 무영향** — 구 계약 자격런의 run_digest 와 셀별로 완전 일치해야 한다.
    #   이것이 "읽기 전용 정보 채널"의 유일한 실측 증명이다.
    if day_plan_public:
        w12 = [c.get("W12_day_plan", {"attached": False}) for c in cells]
        checks["W12_day_plan_matches_schedule"] = all(
            d.get("attached") and d.get("matches_schedule") and d.get("count_matches")
            for d in w12)
        checks["W12_day_plan_no_self_reschedule"] = all(
            d.get("no_self_reschedule") and d.get("version_zero") for d in w12)
        base = OUT / "diurnal_qual.json"
        same = None
        if base.exists():
            old_cells = json.loads(base.read_text(encoding="utf-8"))["cells"]
            if len(old_cells) == len(cells):
                same = all(o["run_digest"] == n["run_digest"]
                           for o, n in zip(old_cells, cells))
        checks["W12_engine_unaffected"] = bool(same)
        checks["W12_engine_unaffected_checked"] = same is not None
    # W3 런 결정론 — 같은 시드 반복 셀의 digest 대조 (없으면 미판정)
    det_run = None
    if repeat is not None:
        det_run = repeat["run_digest"] == cells[repeat["rep"]]["run_digest"]
        checks["W3_deterministic_run"] = bool(det_run)
    verdict = {
        "qualification_all_pass": all(checks.values()) and det_run is True,
        "checks": checks,
        "run_determinism_checked": det_run is not None,
        "observed_penalty_share": [c["obs_penalty_share"] for c in cells],
        "observed_mean_stay_s": [c["obs_mean_stay_s"] for c in cells],
        "observed_concurrency": [c["W4_detail"]["concurrency"] for c in cells][:1],
        "observed_vessel_rate": [c["W4_detail"]["realized"]["realized_moves_per_h"]
                                 for c in cells],
        "note": "5차 계약 자격 — 성능 주장 없음. 벌금 구간 도달률·체류 곡선은 "
                "부수 관측이며 판정 임계가 아니다. 측정창은 실행 24h 중 22h.",
    }
    res = {"task": "YR-167",
           "contract": ("diurnal_24h+day_plan_public" if day_plan_public
                        else "diurnal_24h"),
           "runtime": {"commit": _git("rev-parse", "HEAD"),
                       "git_dirty": bool(code_dirty()),
                       "prereg_file": str(PREREG),
                       "prereg_sha256": _sha256(PREREG) if PREREG.exists() else None,
                       "observation": OBS_24H.as_dict(),
                       "seeds": [SEED + r for r in range(REPS)],
                       "cells_from_files": from_files,
                       "cell_sha256": ({f"cell_rep{r}.json":
                                        _sha256(out_dir / f"cell_rep{r}.json")
                                        for r in range(REPS)} if from_files else None)},
           "verdict": verdict, "cells": cells}
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "diurnal_qual.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "diurnal_qual.json.sha256").write_text(_sha256(p) + "\n",
                                                      encoding="utf-8")
    print(json.dumps({"verdict": verdict}, ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int, default=None)
    ap.add_argument("--tag", default="", help="반복 셀 표식 (예: repeat)")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summarize", action="store_true",
                    help="병렬로 산출된 cell_rep*.json 을 합산(재실행 없음)")
    ap.add_argument("--day-plan-public", action="store_true",
                    help="YR-171-A 재자격 — 하루 공개 예약 장부를 붙여 자격을 다시 받는다. "
                         "산출은 별도 디렉터리(구 계약 자격 증거를 덮지 않는다).")
    a = ap.parse_args()
    if a.day_plan_public:
        # 정보 계약이 다르면 자격 증거도 분리 보관한다 — 나중에 어느 계약의 자격인지
        # 구분할 수 없으면 둘 다 못 쓴다.
        OUT = Path(str(OUT) + "_public")
    if a.summarize:
        rp = OUT / "cell_rep0_repeat.json"
        run([json.loads((OUT / f"cell_rep{r}.json").read_text(encoding="utf-8"))
             for r in range(REPS)],
            repeat=(json.loads(rp.read_text(encoding="utf-8"))
                    if rp.exists() else None),
            out_dir=OUT, day_plan_public=a.day_plan_public)
    elif a.rep is not None:
        c = run_cell(a.rep, day_plan_public=a.day_plan_public)
        OUT.mkdir(parents=True, exist_ok=True)
        suffix = f"_{a.tag}" if a.tag else ""
        (OUT / f"cell_rep{a.rep}{suffix}.json").write_text(
            json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in c.items()
                          if not k.startswith("obs_hourly")}, ensure_ascii=False))
    elif a.run:
        run()
    print("DONE")
