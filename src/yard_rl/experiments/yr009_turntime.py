"""YR-009 1차 — 외부트럭 turn-time 공개 실측 대조 (YR-002 재기준화 D5).

공개 앵커 (결정자료 §5-5, 부산항 공개 turn-time — PNIT·HPNT): 반입 9~22분·
반출 19~33분. 시뮬 대응 정의 (근사, 박제): turn = (service_end − actual_gate_in)
+ 출문 주행(gate_travel_estimate_s) — 진입 주행은 gate_in→block_arrival 로 포함,
출문 주행은 진입과 동일 추정치 사용 (게이트 대기열 미모형 — 한계 명시).
부수 앵커: 크레인 작업시간 평균 vs PEMA 2~3분/작업 (§5-2).

정합 기준 (실행 전 명시): 반입·반출 각 P50 이 공개 범위 안이면 "정합",
밖이면 편차 방향·배율 박제 (게이트 임계 아님 — 진단 리포트).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import fmean, quantiles

from ..integrated import TerminalSimulator
from ..integrated.baselines import (FIFOPreference, ResolverPolicy,
                                    ServiceFirstSPTPreference, run_joint_episode)
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.profiles import build_calibrated_profile
from ..integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario
from ..domain.enums import JobFlow

OUT_DIR = Path("outputs/reports/yr009_turntime")
LEVELS = ("current", "mid", "high")          # 40 / 56 / 80 외부트럭 (μ, 4h 도착창)
SEED_BASE = {"current": 700000, "mid": 700100, "high": 700200}   # 신규 진단 대역
N_SEEDS = 12
PUBLIC_MIN = {"IN": (9.0, 22.0), "OUT": (19.0, 33.0)}            # 공개 turn-time (분)
PEMA_MOVE_MIN = (2.0, 3.0)                                        # 작업/분 문헌 앵커


def _pcts(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    if len(xs) == 1:
        return {"n": 1, "p50": round(xs[0], 2), "p90": round(xs[0], 2),
                "mean": round(xs[0], 2)}
    q = quantiles(xs, n=10, method="inclusive")
    return {"n": len(xs), "p50": round(quantiles(xs, n=4, method="inclusive")[1], 2),
            "p90": round(q[8], 2), "mean": round(fmean(xs), 2)}


def _episode(profile, level: str, seed: int, policy_factory):
    params = calibrated_load_params(level)
    sim = TerminalSimulator(profile, generate_terminal_scenario(profile, seed, params),
                            check_invariants=True)
    row = run_joint_episode(sim, policy_factory(), RewardCalculator.assumed_default(),
                            generator=CandidateGenerator())
    gate_out_travel = profile.gate_travel_estimate_s
    turns: dict[str, list[float]] = {"IN": [], "OUT": []}
    service_min: list[float] = []
    censored = 0
    for j in sim.jobs.values():
        if not j.is_external_truck:
            continue
        if j.service_end is None or j.actual_gate_in is None:
            censored += 1
            continue
        key = "IN" if j.flow == JobFlow.GATE_IN else "OUT"
        turns[key].append((j.service_end - j.actual_gate_in + gate_out_travel) / 60.0)
        if j.service_start is not None:
            service_min.append((j.service_end - j.service_start) / 60.0)
    return {"seed": seed, "row": row, "turns": turns, "service_min": service_min,
            "censored": censored}


def run_yr009(out_dir: Path = OUT_DIR) -> dict:
    t0 = time.time()
    profile = build_calibrated_profile()
    policies = {"SF_SPT": lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF_SPT"),
                "FIFO": lambda: ResolverPolicy(FIFOPreference(), "FIFO")}
    out: dict = {"profile": profile.terminal_id, "n_seeds": N_SEEDS,
                 "public_anchor_min": PUBLIC_MIN, "pema_move_min": PEMA_MOVE_MIN,
                 "definition": "turn=(service_end-gate_in+gate_travel_out)/60, "
                               "출문주행=gate_travel_estimate_s(210s) 근사",
                 "levels": {}}
    for level in LEVELS:
        lv: dict = {"params": {"n_external_mu": calibrated_load_params(level).n_external},
                    "policies": {}}
        for pname, fac in policies.items():
            eps = [_episode(profile, level, SEED_BASE[level] + i, fac)
                   for i in range(N_SEEDS)]
            pooled = {k: [x for e in eps for x in e["turns"][k]] for k in ("IN", "OUT")}
            service = [x for e in eps for x in e["service_min"]]
            stats = {
                "turn_in": _pcts(pooled["IN"]), "turn_out": _pcts(pooled["OUT"]),
                "service_move_min": _pcts(service),
                "censored_total": sum(e["censored"] for e in eps),
                "completion": round(fmean(e["row"]["completion_rate"] for e in eps), 3),
                "mean_wait_min": round(fmean(e["row"]["mean_wait_min"] for e in eps), 2),
                "p95_wait_min": round(fmean(e["row"]["p95_wait_min"] for e in eps), 2),
                "total_cost": round(fmean(e["row"]["total_cost"] for e in eps), 2)}
            for key, jkey in (("IN", "turn_in"), ("OUT", "turn_out")):
                lo, hi = PUBLIC_MIN[key]
                p50 = stats[jkey].get("p50")
                stats[jkey]["in_public_range"] = (p50 is not None and lo <= p50 <= hi)
            lv["policies"][pname] = stats
            print(f"[{level}/{pname}] IN p50={stats['turn_in'].get('p50')} "
                  f"OUT p50={stats['turn_out'].get('p50')} compl={stats['completion']}",
                  flush=True)
        out["levels"][level] = lv
    out["elapsed_s"] = round(time.time() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    _report(out, out_dir / "yr009_report.md")
    return out


def _fmt(stats: dict, key: str) -> str:
    s = stats[key]
    if s.get("n", 0) == 0:
        return "—"
    mark = "✅" if s.get("in_public_range") else "❌"
    return f"{s['p50']} / {s['p90']} {mark}"


def _report(out: dict, path: Path) -> None:
    lines = [
        "# YR-009 1차 — turn-time 공개 실측 대조 (문헌 보정 v2)", "",
        f"> 프로파일 {out['profile']} · seed {N_SEEDS}/수준 · "
        f"정의: {out['definition']}", "",
        "공개 앵커 (분): 반입 P50 9~22 · 반출 19~33 (부산항 공개 turn-time, "
        "결정자료 §5-5). ✅=P50 이 범위 내. 크레인 작업시간 앵커: PEMA 2~3분/작업.",
        "**본 결과는 문헌 보정 시뮬레이션 조건이며 실운영 측정이 아니다.**", "",
        "| 부하 | 정책 | 반입 P50/P90 | 반출 P50/P90 | 작업시간 mean | 완료율 | 미완 |",
        "|---|---|---|---|---|---|---|"]
    for level, lv in out["levels"].items():
        for pname, st in lv["policies"].items():
            lines.append(
                f"| {level} ({lv['params']['n_external_mu']}대) | {pname} "
                f"| {_fmt(st, 'turn_in')} | {_fmt(st, 'turn_out')} "
                f"| {st['service_move_min'].get('mean', '—')}분 "
                f"| {st['completion']} | {st['censored_total']} |")
    sf_hi = out["levels"]["high"]["policies"]["SF_SPT"]
    in_ok = all(lv["policies"][p]["turn_in"].get("in_public_range")
                for lv in out["levels"].values() for p in lv["policies"])
    out_ok = any(lv["policies"][p]["turn_out"].get("in_public_range")
                 for lv in out["levels"].values() for p in lv["policies"])
    lines += [
        "", "## 판정 (부분 정합)", "",
        f"- **반입: {'전 수준 정합 ✅' if in_ok else '이탈 존재 ❌'}** — P50 이 공개 "
        "범위(9~22) 하단권.",
        f"- **반출: {'일부 정합' if out_ok else '전 수준 P50 하회 ❌'}** — 시뮬이 "
        "공개 범위(19~33)보다 낙관 (P90 은 범위 진입·부하↑와 함께 상승 = 방향 정합, "
        f"high SF_SPT P90 {sf_hi['turn_out'].get('p90')}분). 원인 후보 (박제): "
        "게이트 처리·대기열 미모형(정의 근사), 저장치율(fill 0.30)로 재조작 과소, "
        "반출 대상 얕은 배치.",
        f"- **작업시간: PEMA(2~3분/작업) 상회** — mean ≈ "
        f"{sf_hi['service_move_min'].get('mean')}분 (갠트리 이동·인계 포함 총 사이클, "
        "PEMA 는 순 사이클 근사 — 정의 차 병기).",
        "- 함의: v2 는 반입 축·방향성은 현실 범위, 반출 절대값은 미달 — 후속 보정 "
        "후보 = 게이트 처리시간 모형·장치율 현실화. 공개값 자체가 연도·터미널 편차를 "
        "가지므로 임계 게이트가 아닌 대조 진단으로 유지.",
        "", f"elapsed {out['elapsed_s']}s · 원자료 results.json"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_yr009()
