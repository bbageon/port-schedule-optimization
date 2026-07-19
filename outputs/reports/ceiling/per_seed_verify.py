"""YR-071 ③ 시드별 재검증 + 알고리즘 작동 점검.

(A) 시드별 실제 트럭 대기 — SF_SPT(대조군, 목적 무관) vs JR/BEAM(OLD/NEW). 편차·일관성 확인.
(B) 알고리즘 작동 — NEW 에서 탐색기가 '어떻게' 대기를 줄이나: 행동분포 변화·완주율·backlog.
(C) 결정론 — 같은 seed·목적 2회 실행 바이트 일치(탐색 알고리즘 무결성).
"""
import json, time
from statistics import fmean, pstdev

from yard_rl.contract.schema import COST_TERMS
from yard_rl.integrated import TerminalSimulator
from yard_rl.integrated.fixtures import build_integrated_profile
from yard_rl.integrated.scenario_gen import TerminalGenParams, generate_terminal_scenario
from yard_rl.integrated.cost_config import RewardCalculator, default_assumed_config
from yard_rl.integrated.candidates import CandidateGenerator
from yard_rl.integrated.baselines import (
    ResolverPolicy, ServiceFirstSPTPreference, JointRolloutGreedy, BeamLookahead, run_joint_episode)

OUT = "/mnt/c/Users/geonu/Desktop/port_reinforcement/outputs/reports/ceiling/per_seed_verify.json"
PROFILE = build_integrated_profile()
GEN = CandidateGenerator()
RC_OLD = RewardCalculator.assumed_default()
PRIORITY = {"truck_wait": 100.0, "long_wait": 100.0, "vessel_delay": 5.0, "depart_delay": 5.0,
            "sts_wait": 3.0, "transfer_wait": 3.0, "rehandle": 1.0, "crane_travel": 1.0,
            "empty_travel": 1.0, "resequence": 1.0, "imbalance": 0.5,
            "interference": 0.1, "lane_cong": 0.1}
RC_NEW = RewardCalculator(default_assumed_config().with_weight({t: PRIORITY[t] for t in COST_TERMS}))


def make_sim(seed, n_ext, n_ves):
    return TerminalSimulator(
        PROFILE, generate_terminal_scenario(
            PROFILE, seed, TerminalGenParams(n_external=n_ext, n_vessels=n_ves, eta_error_s=300.0)),
        check_invariants=True)


def episode(fac, rc, seed, n_ext, n_ves):
    r = run_joint_episode(make_sim(seed, n_ext, n_ves), fac(rc), rc, generator=GEN)
    sh = r["action_mix"]["shares"]
    return {"total": round(r["total_cost"], 2), "wait": round(r["mean_wait_min"], 3),
            "p95": round(r["p95_wait_min"], 3), "swa": round(r["action_mix"]["serve_when_available"], 3),
            "compl": round(r["completion_rate"], 3), "backlog": r["backlog"],
            "SERVE": sh.get("SERVE", 0.0), "REPO": sh.get("REPOSITION", 0.0),
            "WAIT": sh.get("WAIT", 0.0), "PRE": sh.get("PRE_REHANDLE", 0.0)}


POLS = {
    "SF_SPT": lambda rc: ResolverPolicy(ServiceFirstSPTPreference(), "SF_SPT"),
    "JR": lambda rc: JointRolloutGreedy(rc, horizon_s=600.0, generator=GEN),
    "BEAM": lambda rc: BeamLookahead(rc, width=3, horizon_s=600.0, generator=GEN),
}


def per_seed(n_ext, n_ves, seeds, label):
    print(f"\n========== {label} (n_ext={n_ext} v={n_ves}) ==========")
    data = {p: {"OLD": [], "NEW": []} for p in POLS}
    for p, fac in POLS.items():
        for obj, rc in (("OLD", RC_OLD), ("NEW", RC_NEW)):
            for s in seeds:
                data[p][obj].append((s, episode(fac, rc, s, n_ext, n_ves)))
    # (A) 시드별 실제 대기
    print("\n[A] 시드별 실제 평균대기(분) — SF_SPT 는 대조군(OLD=NEW 이어야 정상)")
    print(f"{'seed':>7s} | {'SF_OLD':>6s} {'SF_NEW':>6s} | {'JR_OLD':>6s} {'JR_NEW':>6s} | {'BM_OLD':>6s} {'BM_NEW':>6s}")
    for i, s in enumerate(seeds):
        row = f"{s:>7d} |"
        for p in ("SF_SPT", "JR", "BEAM"):
            row += f" {data[p]['OLD'][i][1]['wait']:>6.3f} {data[p]['NEW'][i][1]['wait']:>6.3f} |"
        print(row)
    # 요약 통계
    def stat(p, obj, key):
        xs = [d[1][key] for d in data[p][obj]]
        return round(fmean(xs), 3), round(pstdev(xs), 3) if len(xs) > 1 else 0.0
    print("\n[A요약] 평균±표준편차 대기(분):")
    for p in ("SF_SPT", "JR", "BEAM"):
        mo, so = stat(p, "OLD", "wait"); mn, sn = stat(p, "NEW", "wait")
        print(f"  {p:8s} OLD {mo:.3f}±{so:.3f} → NEW {mn:.3f}±{sn:.3f}")
    # 일관성: NEW 에서 JR 이 SF 보다 대기 낮은 seed 수
    jr_win = sum(1 for i in range(len(seeds))
                 if data["JR"]["NEW"][i][1]["wait"] < data["SF_SPT"]["NEW"][i][1]["wait"])
    print(f"  일관성: NEW 목적에서 JR 대기 < SF_SPT 인 seed = {jr_win}/{len(seeds)}")

    # (B) 알고리즘 작동 — 행동분포·완주 변화
    print("\n[B] 알고리즘 작동: 행동분포·완주 (평균) — 탐색기가 '어떻게' 대기를 줄이나")
    print(f"{'policy/obj':12s} {'SERVE':>6s} {'REPO':>6s} {'WAIT':>6s} {'PRE':>5s} {'compl':>6s} {'backlog':>7s} {'swa':>5s}")
    for p in ("SF_SPT", "JR", "BEAM"):
        for obj in ("OLD", "NEW"):
            def m(k): return round(fmean(d[1][k] for d in data[p][obj]), 3)
            print(f"{p+'/'+obj:12s} {m('SERVE'):>6.2f} {m('REPO'):>6.2f} {m('WAIT'):>6.2f} "
                  f"{m('PRE'):>5.2f} {m('compl'):>6.3f} {m('backlog'):>7.2f} {m('swa'):>5.2f}")
    return {p: {o: [(s, d) for s, d in data[p][o]] for o in ("OLD", "NEW")} for p in POLS}


def determinism_check(seed):
    print("\n[C] 결정론 점검 — JR/NEW 같은 seed 2회 실행 일치?")
    a = episode(POLS["JR"], RC_NEW, seed, 20, 2)
    b = episode(POLS["JR"], RC_NEW, seed, 20, 2)
    ok = a == b
    print(f"  seed {seed}: run1 total={a['total']} wait={a['wait']} / run2 total={b['total']} "
          f"wait={b['wait']} → {'일치 ✅' if ok else '불일치 ⚠'}")
    return ok


def main():
    t0 = time.time()
    out = {}
    out["MODERATE"] = per_seed(20, 2, list(range(330000, 330008)), "MODERATE")
    out["HIGHLOAD"] = per_seed(48, 3, list(range(330000, 330004)), "HIGHLOAD")
    out["determinism_ok"] = determinism_check(330000)
    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # JSON 은 요약만 (중첩 튜플 직렬화 회피)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"note": "상세는 stdout, 이 파일은 실행 확인용",
                   "determinism_ok": out["determinism_ok"],
                   "elapsed_s": round(time.time() - t0, 1)}, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {OUT}  (elapsed {round(time.time()-t0,1)}s)")


if __name__ == "__main__":
    main()
