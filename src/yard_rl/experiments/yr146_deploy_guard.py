"""YR-146 — 배포용 교착 탈출 안전장치 (발동 3조건 한정·OFF/ON 분리 검증, 동결).

■ 발동 (24차 협소화 — 정상적 기다림 불간섭):
  ①재개방 부재: 전원 대기 선택인데 미래 재검토(사건·wake·defer 만료) 전무 → 정책 평가
    최선 진행으로 대체  ②무진전 반복: 재개방 후 상태 변화(실사건·완료 수) 없이 다시
    전원 대기 → 동일 대체  ③간섭 교착: 진행 후보 전무 ∧ 간섭 술어 참 → 최소 escape 확정.
■ 검증 4군 (재학습 없음 — YR-143 확증 체크포인트 재사용): C0/C1 × OFF/ON,
  신규 대역 16판(셀 4)·초기화 8쌍. guard 이득을 학습 성과로 합산 금지.
■ 성공(동결): ON 양군 완주 100%·backlog 0 ∧ 개입률 ≤ 1%(공동결정 대비) ∧
  ON−OFF v2 악화 ≤ δ_v2 1.0 (양군). 불필요 개입(OFF 완주판에서의 개입)은 분리 보고.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from statistics import fmean

import torch

from ..integrated import candidates as cand_mod
from ..integrated.baselines import JointRolloutGreedy
from ..integrated.candidates import CandidateGenerator
from ..integrated.encoding import StateNorm
from ..integrated.joint_distill import JointPairNet
from ..integrated.seedbank import (BandSpec, assign_band, independence_report,
                                   realization_hash)
from . import yr088_joint_rl as y88
from .yr090_dense_vessel import CELLS
from .yr100_candidate_eval import RC_EVAL
from .yr136_softplus_contract import _sim_contract
from .yr141_bound_prepo import _PrepoRecorder
from .yr143_no_repo import ARM_FLAGS, NORM_TS, CONFIRM_TS
from .yr143_no_repo import OUT as OUT143

OUT = Path("outputs/reports/yr146_deploy_guard")
# 2차(허가증 v2) 대역 — 1차 band.json(910300대)·smoke 910200 은 열람 편입, 커서 이동.
BAND_PATH = OUT / "band2.json"
BAND_START, BAND_N = 910_400, 4
CAP_RATE = 0.01                     # 개입률 허용치 (공동결정 대비 — 사전 동결)
DELTA_V2 = 1.0                      # ON−OFF v2 악화 상한 (YR-143 δ 승계)
PROG = ("SERVE", "PRE_REHANDLE")


IV_KEYS = ("iv_no_target", "iv_deadline", "iv_escape")


def classify_wait(*, has_progress: bool, interference: bool, has_escape: bool,
                  deadline_pressure: bool, triggered: bool,
                  prev_untriggered_same_snap: bool) -> str:
    """대기 허가증 판정 (25차 동결 — 우선순위 g→f→e, 그 외 허용).

    triggered(관측 trigger 존재)는 생성 계약상 항상 미래 시각 — (a)600초 점검 재대기·
    (b)ETA 갱신 모두 여기로 허용된다. (c)실제 사건 도래는 스냅샷 변화로 반복 조건이
    풀려 허용된다."""
    if not has_progress:
        return "IV_ESCAPE" if (interference and has_escape) else "ALLOW"
    if deadline_pressure:
        return "IV_DEADLINE"
    if not triggered and prev_untriggered_same_snap:
        return "IV_NO_TARGET"
    return "ALLOW"


class DeployGuard:
    """정책 래퍼 v2 — 대기 허가증 (25차): 무엇을 기다리는지 기록하고, 기다릴 대상이
    있는 대기는 불간섭. 개입은 (e)무대상 반복·(f)마감 압박·(g)간섭 탈출뿐."""

    def __init__(self, inner, actor, norm, jr):
        self.inner, self.actor, self.norm, self.jr = inner, actor, norm, jr
        self.stats = {"dec": 0, "joint": 0, "iv_no_target": 0, "iv_deadline": 0,
                      "iv_escape": 0, "allow_future_trigger": 0,
                      "allow_state_change": 0, "allow_first_untrig": 0}
        self.permits = []                     # 허가증 원장 — 원자료 (25차 해석 한계 해소)
        self._prev = None                     # (untriggered, snap)

    @staticmethod
    def _snap(sim, gen_by):
        """진행 관련 상태 — (완료 수, 실행 가능 실작업 후보 수). 전체 사건 수 아님
        (무관 이벤트가 반복 카운터를 오초기화하지 않게 — 25차 필수 테스트 ④)."""
        done = sum(1 for j in sim.jobs.values() if j.status.name == "DONE")
        n_work = sum(1 for c in gen_by.values() for g in c.items
                     if g.kind.name in PROG and g.feasible)
        return (done, n_work)

    def decide(self, sim, dp, gen_by):
        assign = self.inner.decide(sim, dp, gen_by)
        self.stats["dec"] += 1
        if len(dp.crane_ids) >= 2:
            self.stats["joint"] += 1
        if not all(g.kind.name == "WAIT" for g in assign.values()):
            self._prev = None
            return assign
        rows, assigns = y88.build_rows(sim, dp, gen_by, self.norm, self.jr, 0)
        if not assigns:
            return assign                                   # 구조적 대기 — 불간섭
        # 대체 후보 = 위치조정 미포함 진행 조합만 (C0/C1 공통 행동 — 효과 교락 방지)
        prog = [i for i, a in enumerate(assigns)
                if any(a[c].kind.name in PROG for c in a)
                and all(a[c].kind.name != "REPOSITION" for c in a)]
        esc = [i for i, a in enumerate(assigns)
               if any(a[c].kind.name == "REPOSITION" and a[c].job_ref is not None
                      and a[c].job_ref.job_id.startswith("REPO:") for c in a)]
        triggered = [(g.defer_trigger, float(g.defer_until),
                      getattr(g, "defer_trigger_jid", None))
                     for g in assign.values()
                     if getattr(g, "defer_trigger", None) is not None]
        snap = self._snap(sim, gen_by)
        pending = any(j.status.name != "DONE" for j in sim.jobs.values())
        deadline_pressure = pending and (sim.end - sim.now) < cand_mod.DEFER_T_MAX
        prev_same = (self._prev is not None and self._prev[0]
                     and self._prev[1] == snap)
        case = classify_wait(
            has_progress=bool(prog),
            interference=bool(sim.interference_deadlock_corridors()) if not prog
            else False,
            has_escape=bool(esc), deadline_pressure=deadline_pressure,
            triggered=bool(triggered), prev_untriggered_same_snap=prev_same)
        rec = {"t": float(sim.now), "snap": list(snap),
               "triggered": triggered, "case": case}
        if case != "ALLOW":                     # 26차: 개입 정당성 감사 가능 원장
            last_ev = sim.event_log[-1] if sim.event_log else None
            rec["wake_src"] = (list(last_ev) if last_ev
                              and abs(last_ev[0] - sim.now) < 1e-6 else None)
            rec["orig"] = {c: [assign[c].kind.name,
                               getattr(assign[c].job_ref, "job_id", None)
                               if assign[c].job_ref else None]
                           for c in sorted(assign)}
        self.permits.append(rec)
        if case == "ALLOW":
            if triggered:
                self.stats["allow_future_trigger"] += 1
            elif self._prev is not None and self._prev[0] and self._prev[1] != snap:
                self.stats["allow_state_change"] += 1
            else:
                self.stats["allow_first_untrig"] += 1
            self._prev = (not triggered, snap)
            return assign
        pool = esc if case == "IV_ESCAPE" else prog
        with torch.no_grad():
            cost, _ = self.actor(torch.tensor(rows, dtype=torch.float32))
        self.stats[case.lower()] += 1
        self._prev = None
        best = assigns[min(pool, key=lambda i: float(cost[i]))]
        rec["repl"] = {c: [best[c].kind.name,
                           getattr(best[c].job_ref, "job_id", None)
                           if best[c].job_ref else None] for c in sorted(best)}
        return best


def _collect_hashes() -> set[str]:
    import re
    pat = re.compile(r"rz1:[0-9a-f]{16}")
    got: set[str] = set()
    for p in Path("outputs/reports").rglob("*.json"):
        if OUT in p.parents:
            continue
        try:
            got |= set(pat.findall(p.read_text(encoding="utf-8", errors="ignore")))
        except OSError:
            continue
    return got


def make_band():
    exclude = _collect_hashes()
    band = assign_band(family="yr146-validate", cells={c: None for c in CELLS},
                       n=BAND_N,
                       generate=lambda key, _p, seed: _sim_contract(key, seed).scenario,
                       exclude=exclude, start_seed=BAND_START)
    rep = independence_report(band, forbidden={"past-recorded": exclude})
    assert rep["ok"], rep
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    OUT.mkdir(parents=True, exist_ok=True)
    BAND_PATH.write_text(json.dumps(
        {**band.freeze_json(), "independence": rep, "n_excluded_hashes": len(exclude),
         "created_commit": head}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[band] {sum(len(v) for v in band.seeds.values())} seeds frozen")


def _episode_guard(cell, seed, arm, ts, guard_on: bool):
    from ..integrated.baselines import (ActionMixError, assert_healthy_action_mix,
                                        run_joint_episode)
    from .yr138_episode_pilot import _v2_hard_total
    f = ARM_FLAGS[arm]
    prev = (cand_mod.WAIT_MODE, cand_mod.SAFETY_ONLY, cand_mod.BOUND_REPO,
            cand_mod.PREPO_ONE_SHOT)
    cand_mod.WAIT_MODE = "DEFER_ALL"
    cand_mod.SAFETY_ONLY = f["safety_only"]
    cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT = f["bound"], True
    try:
        ck = torch.load(OUT143 / "confirm" / arm / f"ppo_s{ts}" / "net.pt",
                        map_location="cpu")
        actor = JointPairNet(250); actor.load_state_dict(ck["actor"]); actor.eval()
        ck0 = torch.load(Path("outputs/reports/yr125_diff_credit")
                         / f"diff1_s{NORM_TS}" / "rl_net.pt", map_location="cpu")
        norm = StateNorm(refs=ck0["norm_refs"])
        y88.FORBID_WAIT = True
        policy = y88.RLPolicy(actor, norm, name=f"{arm}:{ts}")
        guard = None
        if guard_on:
            jr = JointRolloutGreedy(RC_EVAL, horizon_s=1800.0,
                                    generator=CandidateGenerator(),
                                    forbid_strategic_wait=True)
            policy = guard = DeployGuard(policy, actor, norm, jr)
        sim = _sim_contract(cell, seed)
        rec = _PrepoRecorder(policy)
        r = run_joint_episode(sim, rec, RC_EVAL, generator=CandidateGenerator())
        healthy = True
        try:
            assert_healthy_action_mix(r["_mix"], label=f"{cell}/s{seed}")
        except ActionMixError:
            healthy = False
        out = {"cell": cell, "seed": seed, "compl": r["completion_rate"],
               "backlog": r["backlog"], "healthy": healthy,
               "v2_total": _v2_hard_total(sim), "v1_total": r["total_cost"],
               "berth_over_min": r["berth_overrun_min"]}
        if guard is not None:
            out["guard"] = dict(guard.stats)
            out["guard_permits"] = list(guard.permits)   # 26차: 감사 가능 원장
        return out
    finally:
        (cand_mod.WAIT_MODE, cand_mod.SAFETY_ONLY, cand_mod.BOUND_REPO,
         cand_mod.PREPO_ONE_SHOT) = prev


def validate() -> dict:
    from ..integrated.repro import repro_stamp
    d = json.loads(BAND_PATH.read_text(encoding="utf-8"))
    band = BandSpec(family=d["family"], seeds=d["seeds"], hashes=d["realization_hashes"])
    for cell, ss in band.seeds.items():
        for s, h in zip(ss, band.hashes[cell]):
            assert realization_hash(_sim_contract(cell, s).scenario) == h, f"{cell}:{s}"
    eval_eps = [(c, s) for c in CELLS for s in band.seeds[c]]
    rows = {}
    for arm in ("c0", "c1"):
        for on in (False, True):
            key = f"{arm}:{'on' if on else 'off'}"
            print(f"[validate] {key}", flush=True)
            rows[key] = {ts: [_episode_guard(c, s, arm, ts, on) for c, s in eval_eps]
                         for ts in CONFIRM_TS}
    summ = {}
    for key, by_ts in rows.items():
        eps = [e for ts in CONFIRM_TS for e in by_ts[ts]]
        s = {"n": len(eps), "compl_min": min(e["compl"] for e in eps),
             "n_incomplete": sum(1 for e in eps if e["compl"] < 1.0),
             "backlog_max": max(e["backlog"] for e in eps),
             "healthy_all": all(e["healthy"] for e in eps),
             "v2_mean": fmean(e["v2_total"] for e in eps),
             "berth_mean": fmean(e["berth_over_min"] for e in eps)}
        if ":on" in key:
            g = [e["guard"] for e in eps]
            joint = sum(x["joint"] for x in g)
            iv = sum(sum(x[k] for k in IV_KEYS) for x in g)
            s.update({"joint_dec": joint, "interventions": iv,
                      "iv_rate": iv / joint if joint else 0.0,
                      "iv_by_case": {k: sum(x[k] for x in g) for k in IV_KEYS},
                      "allow_by_cause": {k: sum(x[k] for x in g)
                                         for k in ("allow_future_trigger",
                                                   "allow_state_change",
                                                   "allow_first_untrig")}})
        summ[key] = s
    # OFF-완주 판 개입 (★25차 명칭 — 안전장치 없이도 완주했던 판에서의 개입)
    needless = {}
    for arm in ("c0", "c1"):
        cnt = 0
        for ts in CONFIRM_TS:
            for j, e_off in enumerate(rows[f"{arm}:off"][ts]):
                e_on = rows[f"{arm}:on"][ts][j]
                g = e_on["guard"]
                if e_off["compl"] >= 1.0 and sum(g[k] for k in IV_KEYS) > 0:
                    cnt += 1
        needless[arm] = cnt
    j = {}
    for arm in ("c0", "c1"):
        on, off = summ[f"{arm}:on"], summ[f"{arm}:off"]
        j[arm] = {"on_completion_all": on["compl_min"] >= 1.0 and on["backlog_max"] == 0,
                  "iv_rate_ok": on["iv_rate"] <= CAP_RATE,
                  "cost_ok": (on["v2_mean"] - off["v2_mean"]) <= DELTA_V2,
                  "on_minus_off_v2": on["v2_mean"] - off["v2_mean"]}
    j["success"] = all(v["on_completion_all"] and v["iv_rate_ok"] and v["cost_ok"]
                       for v in j.values() if isinstance(v, dict))
    res = {"repro": repro_stamp(
               experiment="YR-146 배포 안전장치 2차 — 대기 허가증 v2 · 4군 OFF/ON",
               seeds={"train": list(CONFIRM_TS), **{c: band.seeds[c] for c in CELLS}},
               profile_id="calibrated",
               prereg="허가증 v2(25차): 관측 trigger 있는 대기 불간섭·개입 = 무대상 "
                      "반복/마감 압박/간섭 탈출·대체 = 위치조정 미포함 공통 조합·"
                      "개입률 ≤1% 유지·ON−OFF v2 악화 ≤1.0·ON 완주 전판·재학습 없음",
               extra={"band_digest": d["digest"], "cap_rate": CAP_RATE}),
           "summary": summ, "off_complete_interventions": needless, "judgment": j,
           "arms": rows}
    (OUT / "results2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print(json.dumps({"judgment": j, "needless": needless,
                      **{k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                             for kk, vv in s.items() if kk != "iv_by_case"}
                         for k, s in summ.items()}}, ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-band", action="store_true")
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    if a.make_band:
        make_band()
    if a.validate:
        validate()
    print("DONE")
