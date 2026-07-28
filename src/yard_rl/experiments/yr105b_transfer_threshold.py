"""YR-105-b — 창중 이송의 상대 혼잡격차 임계값 단일축 최적화.

바꾸는 것은 하나뿐이다.

    C_zero(source) - C_zero(destination) >= tau

tau 후보는 0.05 / 0.10(현재 기준) / 0.20이다. 이 값은 이송 건수뿐 아니라 이송 시점과
대상도 함께 바꾸므로, 결과는 "볼륨만의 효과"가 아니라 "상대 혼잡격차 임계 정책"의
효과로 해석한다.

실행 순서:
  1. pilot n=16: 결과 평균은 후보 선택에 쓰지 않고 세 쌍의 truck 차이 분산만 산출한다.
  2. select: 보수 필요표본수 이상에서 평균 truck 비용이 가장 낮은 임계 하나를 고른다.
  3. winner_freeze.json을 별도 커밋한 뒤 confirm: 동결 승자와 0.10만 독립 대역에서 비교한다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from ..integrated.evalkit import (CHANNELS, check_guards, paired, paired_by_channel,
                                  required_n)
from ..integrated.profiles import build_calibrated_profile
from ..integrated.repro import repro_stamp
from ..integrated.scenario_gen import generate_terminal_scenario
from ..integrated.seedbank import assign_band, independence_report
from . import yr105_conditional_transfer as y5

OUT = Path("outputs/reports/yr105b_transfer_threshold")
GRID = (0.05, 0.10, 0.20)
BASE = 0.10
PRIMARY = "truck"
DELTA = {"truck": 3.0, "vessel": 10.0, "move": 1.0, "other": 1.0, "total": 10.0}
SD_CONF = 0.80
PILOT_N = 16
BAND_START = {"pilot": 920_000, "select": 930_000, "confirm": 950_000}


def _generate(_key: str, cell, seed: int):
    return generate_terminal_scenario(build_calibrated_profile(), seed, y5._params(cell))


def _band(stage: str, n: int, *, exclude: set[str] | None = None):
    spec = assign_band(
        family=f"yr105b-{stage}",
        cells={"A": y5.CELL_A, "B": y5.CELL_B},
        n=n,
        generate=_generate,
        exclude=exclude or set(),
        start_seed=BAND_START[stage],
    )
    rep = independence_report(spec, forbidden={"이전 대역": exclude or set()})
    if not rep["ok"]:
        raise AssertionError(f"실현 대역 독립성 위반: {rep}")
    return spec, rep


def pilot_hashes() -> set[str]:
    return set(_band("pilot", PILOT_N)[0].all_hashes)


def select_hashes(n: int) -> set[str]:
    return set(_band("select", n, exclude=pilot_hashes())[0].all_hashes)


def run_seed(i: int, stage: str, seeds: dict[str, int],
             thresholds: tuple[float, ...]) -> dict:
    arms = {}
    for tau in thresholds:
        arm = y5.run_arm(
            i,
            stage,
            vessel_guard=False,
            seeds=seeds,
            gap_threshold=tau,
        )
        arms[f"{tau:.2f}"] = arm
    jobs = {a["n_jobs"] for a in arms.values()}
    if len(jobs) != 1:
        raise AssertionError(f"arm별 작업 수 불일치: {jobs}")
    return {
        "seed": i,
        "seed_A": seeds["A"],
        "seed_B": seeds["B"],
        "arms": arms,
    }


def _guard(rows: list[dict], thresholds: tuple[float, ...]):
    keys = [f"{t:.2f}" for t in thresholds]
    rep = check_guards([
        {"compl": r["arms"][k]["compl"], "backlog": r["arms"][k]["backlog"]}
        for r in rows for k in keys
    ])
    exceptions = sum(r["arms"][k]["policy_exceptions"] for r in rows for k in keys)
    if exceptions:
        rep.ok = False
        rep.failures.append(f"정책 예외 {exceptions}건")
    return rep


def _run_rows(stage: str, n: int, thresholds: tuple[float, ...],
              *, exclude: set[str] | None = None) -> tuple[list[dict], dict]:
    spec, independence = _band(stage, n, exclude=exclude)
    rows = []
    for i in range(n):
        seeds = {"A": spec.seeds["A"][i], "B": spec.seeds["B"][i]}
        row = run_seed(i, stage, seeds, thresholds)
        rows.append(row)
        summary = " ".join(
            f"tau={tau:.2f}:{row['arms'][f'{tau:.2f}']['chan'][PRIMARY]:.2f}"
            f"/mv{row['arms'][f'{tau:.2f}']['n_moved']}"
            for tau in thresholds
        )
        print(f"[{stage} {i + 1}/{n}] {summary}", flush=True)
    return rows, {
        "band": spec.freeze_json(),
        "independence": independence,
    }


def _pair(rows: list[dict], left: float, right: float, channel: str) -> list[float]:
    lk, rk = f"{left:.2f}", f"{right:.2f}"
    return [
        r["arms"][lk]["chan"][channel] - r["arms"][rk]["chan"][channel]
        for r in rows
    ]


def run_pilot() -> dict:
    """격자 동결 뒤 분산만 열어 선택·확증 표본수를 정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows, band = _run_rows("pilot", PILOT_N, GRID)
    pairs = ((0.05, 0.10), (0.20, 0.10), (0.05, 0.20))
    power = {}
    needs = []
    for left, right in pairs:
        p = paired(_pair(rows, left, right, PRIMARY), delta_interest=DELTA[PRIMARY],
                   sd_conf=SD_CONF)
        need = required_n(
            p.sd,
            DELTA[PRIMARY],
            sd_conf=SD_CONF,
            sd_df=p.n - 1,
        )
        power[f"{left:.2f}-{right:.2f}"] = {**p.as_dict(), "conservative_n": need}
        if need is not None:
            needs.append(need)
    n_select = max([24, *needs])
    n_confirm = n_select * 2
    guards = _guard(rows, GRID)
    result = {
        "stage": "pilot",
        "contract": {
            "grid": GRID,
            "base": BASE,
            "primary": PRIMARY,
            "delta": DELTA,
            "sd_conf": SD_CONF,
            "achievable_deadline": True,
            "pilot_means_not_for_selection": True,
        },
        "band": band,
        "power_by_pair": power,
        "frozen_sample_plan": {
            "n_select": n_select,
            "n_confirm": n_confirm,
            "rule": "max(24, 세 쌍의 보수 필요 n), 확증=선택×2",
        },
        "guards": {"ok": guards.ok, "failures": guards.failures},
        "rows": rows,
    }
    (OUT / "power_note.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"표본 동결: select={n_select}, confirm={n_confirm}", flush=True)
    return result


def run_select(n: int) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, band = _run_rows("select", n, GRID, exclude=pilot_hashes())
    guards = _guard(rows, GRID)
    means = {
        f"{tau:.2f}": fmean(r["arms"][f"{tau:.2f}"]["chan"][PRIMARY] for r in rows)
        for tau in GRID
    }
    # 정확 동률이면 현행 0.10을 우선한다. 그 밖에는 평균 truck 비용 최소 arm 한 개만 고른다.
    winner_key = min(means, key=lambda k: (means[k], 0 if float(k) == BASE else 1, float(k)))
    winner = float(winner_key)
    mean_benefit = means[f"{BASE:.2f}"] - means[winner_key]
    candidate = bool(guards.ok and winner != BASE and mean_benefit > 0)
    result = {
        "stage": "select",
        "verdict_valid": guards.ok,
        "selection_rule": "검정 없이 평균 truck 비용 최소; 정확 동률은 현행 0.10 우선",
        "means_truck": {k: round(v, 6) for k, v in means.items()},
        "winner": winner,
        "mean_benefit_vs_010": round(mean_benefit, 6),
        "selection": "CANDIDATE" if candidate else "NO_CANDIDATE",
        "claim_limit": "후보 순위 선정 전용이며 유의성 주장을 하지 않는다",
        "band": band,
        "guards": {"ok": guards.ok, "failures": guards.failures},
        "rows": rows,
    }
    (OUT / "results_select.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"선택 결과: {result['selection']} winner={winner:.2f} "
          f"mean benefit={mean_benefit:+.3f}", flush=True)
    return result


def _classification(channels: dict[str, dict], guards_ok: bool) -> str:
    if not guards_ok:
        return "INVALID"
    truck = channels["truck"]
    if truck["ci"][1] < 0:
        return "HARMFUL"
    if truck["ci"][0] <= 0:
        return "EQUIVALENT" if truck.get("equivalent") else "INCONCLUSIVE"
    if channels["vessel"]["ci"][0] <= -DELTA["vessel"]:
        return "TRADEOFF_FAIL"
    if channels["total"]["ci"][0] <= -DELTA["total"]:
        return "TRADEOFF_FAIL"
    if truck["ci"][0] > DELTA["truck"]:
        return "PRACTICAL_IMPROVEMENT"
    return "SMALL_CONFIRMED"


def run_confirm(n: int, winner: float) -> dict:
    if winner not in GRID or winner == BASE:
        raise ValueError("확증 승자는 0.05 또는 0.20이어야 한다")
    frozen = OUT / "winner_freeze.json"
    if not frozen.exists():
        raise FileNotFoundError("winner_freeze.json이 없다 — 승자를 먼저 별도 커밋으로 동결해야 한다")
    freeze = json.loads(frozen.read_text(encoding="utf-8"))
    if float(freeze["winner"]) != winner or int(freeze["n_confirm"]) != n:
        raise ValueError("CLI 승자·표본수가 동결 파일과 다르다")
    exclude = pilot_hashes() | select_hashes(int(freeze["n_select"]))
    thresholds = (BASE, winner)
    rows, band = _run_rows("confirm", n, thresholds, exclude=exclude)
    guards = _guard(rows, thresholds)
    bk, wk = f"{BASE:.2f}", f"{winner:.2f}"
    # 편익 = 현행 비용 - 후보 비용. 양수면 후보가 더 싸다.
    channels = paired_by_channel(
        [r["arms"][bk]["chan"] for r in rows],
        [r["arms"][wk]["chan"] for r in rows],
        delta_interest=DELTA,
        primary=PRIMARY,
        sd_conf=SD_CONF,
    )
    verdict = _classification(channels, guards.ok)
    result = {
        "repro": repro_stamp(
            experiment="YR-105-b 상대 혼잡격차 임계 정책 독립 확증",
            seeds=band["band"]["seeds"],
            params={"grid": GRID, "base": BASE, "winner": winner,
                    "achievable_deadline": True},
            profile_id=build_calibrated_profile().terminal_id,
            prereg="편익=비용(0.10)-비용(승자), 1차 truck. "
                    "CI 하한>0 개선, >3 실무개선. vessel·total 하한>-10 비열등.",
        ),
        "stage": "confirm",
        "winner_freeze": freeze,
        "benefit_010_minus_winner": channels,
        "verdict": verdict,
        "verdict_valid": guards.ok,
        "claim_limit": "동결 승자가 현행 0.10보다 나은지만 확증; 격자 전체 최적 주장은 금지",
        "band": band,
        "guards": {"ok": guards.ok, "failures": guards.failures},
        "mean_moved": {
            bk: round(fmean(r["arms"][bk]["n_moved"] for r in rows), 3),
            wk: round(fmean(r["arms"][wk]["n_moved"] for r in rows), 3),
        },
        "deadlock_escapes": {
            bk: sum(r["arms"][bk]["deadlock_escapes"] for r in rows),
            wk: sum(r["arms"][wk]["deadlock_escapes"] for r in rows),
        },
        "rows": rows,
    }
    (OUT / "results_confirm.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"확증 판정: {verdict}", flush=True)
    for channel, value in channels.items():
        print(f"  {channel}: {value['mean']:+.3f} CI {value['ci']}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("pilot", "select", "confirm"))
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--winner", type=float)
    args = parser.parse_args()
    y5.ACHIEVABLE_DEADLINE = True
    if args.stage == "pilot":
        result = run_pilot()
    elif args.stage == "select":
        if args.seeds is None:
            raise SystemExit("--seeds 필요")
        result = run_select(args.seeds)
    else:
        if args.seeds is None or args.winner is None:
            raise SystemExit("--seeds와 --winner 필요")
        result = run_confirm(args.seeds, args.winner)
    valid = result.get("verdict_valid", result.get("guards", {}).get("ok", True))
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
