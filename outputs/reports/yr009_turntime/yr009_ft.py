"""YR-009 진행 — 채택본(FT+H1) turn-time 재확인 + 기준환경 manifest 동결.

spec: 환경 수정 시 FT+H1 재실행해 성능·완주 변화 확인(4항) + 환경 프로파일·생성기·공개
근거 해시 동결(5항). 정책 bundle 최종동결은 YR-080/075-c 뒤 — FT 는 여기선 '현 정책 진단'.
⚠ FT 는 OLD(간섭 지배) 목적 하 증류·미세조정본 → YR-080 재설계 대상. 본 실행은 환경 동결 +
정책 영향 진단이지 정책 채택 확정이 아니다.
커밋 하네스(yr009_turntime.py) 미변경 — 헬퍼만 재사용. 새 파일 저장.
"""
import sys, json, time, hashlib, dataclasses
from pathlib import Path
from statistics import fmean

sys.path.insert(0, "/mnt/c/Users/geonu/Desktop/port_reinforcement/src")
from yard_rl.experiments.yr009_turntime import _pcts, PUBLIC_MIN, PEMA_MOVE_MIN, LEVELS, SEED_BASE, N_SEEDS
from yard_rl.integrated import TerminalSimulator
from yard_rl.integrated.profiles import build_calibrated_profile
from yard_rl.integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario
from yard_rl.integrated.candidates import CandidateGenerator
from yard_rl.integrated.cost_config import RewardCalculator
from yard_rl.integrated.baselines import run_joint_episode, ResolverPolicy, ServiceFirstSPTPreference
from yard_rl.integrated.joint_distill import CentralJointValuePolicy, load_student, adopted_slot_selector
from yard_rl.domain.enums import JobFlow

OUT = Path("/mnt/c/Users/geonu/Desktop/port_reinforcement/outputs/reports/yr009_turntime")
FT_PATH = "/mnt/c/Users/geonu/Desktop/port_reinforcement/outputs/reports/yr074_finetune/student_ft.pt"
SRC = "/mnt/c/Users/geonu/Desktop/port_reinforcement/src/yard_rl"
FREEZE_FILES = [f"{SRC}/integrated/profiles.py", f"{SRC}/integrated/scenario_gen.py",
                f"{SRC}/sim/travel_time.py", f"{SRC}/integrated/fixtures.py",
                f"{SRC}/integrated/engine.py", f"{SRC}/contract/schema.py"]

PROFILE = build_calibrated_profile()
SLOTS = tuple(sorted(c.crane_id for c in PROFILE.cranes))
NET, NORM = load_student(FT_PATH)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _asdict(o):
    try:
        return dataclasses.asdict(o)
    except Exception:
        return {k: v for k, v in vars(o).items() if isinstance(v, (int, float, str, bool))}


def episode(level, seed, kind):
    params = calibrated_load_params(level)
    sim = TerminalSimulator(PROFILE, generate_terminal_scenario(PROFILE, seed, params),
                            check_invariants=True)   # 위반 시 예외 → 완주+무예외 = 위반 0
    gen = CandidateGenerator()
    if kind == "FT":
        sim.slot_selector = adopted_slot_selector()
        pol = CentralJointValuePolicy(NET, NORM, gen, SLOTS, name="FT")
    else:
        pol = ResolverPolicy(ServiceFirstSPTPreference(), "SF_SPT")
    row = run_joint_episode(sim, pol, RewardCalculator.assumed_default(), generator=gen)
    gto = PROFILE.gate_travel_estimate_s
    turns = {"IN": [], "OUT": []}
    svc, cens = [], 0
    for j in sim.jobs.values():
        if not j.is_external_truck:
            continue
        if j.service_end is None or j.actual_gate_in is None:
            cens += 1
            continue
        k = "IN" if j.flow == JobFlow.GATE_IN else "OUT"
        turns[k].append((j.service_end - j.actual_gate_in + gto) / 60.0)
        if j.service_start is not None:
            svc.append((j.service_end - j.service_start) / 60.0)
    return {"row": row, "turns": turns, "svc": svc, "cens": cens}


def run_policy(level, kind):
    eps = [episode(level, SEED_BASE[level] + i, kind) for i in range(N_SEEDS)]
    pooled = {k: [x for e in eps for x in e["turns"][k]] for k in ("IN", "OUT")}
    svc = [x for e in eps for x in e["svc"]]
    st = {"turn_in": _pcts(pooled["IN"]), "turn_out": _pcts(pooled["OUT"]),
          "service_move_min": _pcts(svc),
          "censored_total": sum(e["cens"] for e in eps),
          "completion": round(fmean(e["row"]["completion_rate"] for e in eps), 3),
          "backlog": round(fmean(e["row"]["backlog"] for e in eps), 2),
          "mean_wait_min": round(fmean(e["row"]["mean_wait_min"] for e in eps), 2),
          "p95_wait_min": round(fmean(e["row"]["p95_wait_min"] for e in eps), 2)}
    for key, jk in (("IN", "turn_in"), ("OUT", "turn_out")):
        lo, hi = PUBLIC_MIN[key]
        p50 = st[jk].get("p50")
        st[jk]["in_public_range"] = (p50 is not None and lo <= p50 <= hi)
    return st


def build_manifest():
    m = {
        "terminal_id": PROFILE.terminal_id,
        "block": _asdict(PROFILE.block),
        "cranes": [_asdict(c) for c in PROFILE.cranes],
        "gate_travel_estimate_s": getattr(PROFILE, "gate_travel_estimate_s", None),
        "long_wait_sla_s": getattr(PROFILE, "long_wait_sla_s", None),
        "load_params": {lv: _asdict(calibrated_load_params(lv)) for lv in LEVELS},
        "public_anchor_min": PUBLIC_MIN, "pema_move_min": PEMA_MOVE_MIN,
        "src_sha256_16": {Path(f).name: sha(f) for f in FREEZE_FILES},
        "ft_model_sha256_16": sha(FT_PATH),
    }
    canon = json.dumps(m, ensure_ascii=False, sort_keys=True)
    m["manifest_sha256"] = hashlib.sha256(canon.encode()).hexdigest()
    return m


def main():
    t0 = time.time()
    out = {"profile": PROFILE.terminal_id, "n_seeds": N_SEEDS, "slots": SLOTS,
           "note": "FT=채택 진단(OLD 목적 증류본, YR-080 재설계 대상)·환경 동결용",
           "public_anchor_min": PUBLIC_MIN, "levels": {}}
    print(f"{'lvl/pol':14s} {'IN p50/p90':>12s} {'OUT p50/p90':>12s} {'svc':>5s} "
          f"{'compl':>6s} {'bklog':>5s} {'wait':>5s}", flush=True)
    for lv in LEVELS:
        out["levels"][lv] = {"n_external_mu": calibrated_load_params(lv).n_external, "policies": {}}
        for kind in ("SF_SPT", "FT"):
            st = run_policy(lv, kind)
            out["levels"][lv]["policies"][kind] = st
            ti, to = st["turn_in"], st["turn_out"]
            print(f"{lv+'/'+kind:14s} {str(ti.get('p50'))+'/'+str(ti.get('p90')):>12s} "
                  f"{str(to.get('p50'))+'/'+str(to.get('p90')):>12s} "
                  f"{st['service_move_min'].get('mean'):>5} {st['completion']:>6} "
                  f"{st['backlog']:>5} {st['mean_wait_min']:>5}", flush=True)
    out["manifest"] = build_manifest()
    out["elapsed_s"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "yr009_ft.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "yr009_frozen_manifest.json").write_text(
        json.dumps(out["manifest"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n환경 manifest 동결 해시: {out['manifest']['manifest_sha256'][:32]}...", flush=True)
    print(f"저장: {OUT}/yr009_ft.json · yr009_frozen_manifest.json ({out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
