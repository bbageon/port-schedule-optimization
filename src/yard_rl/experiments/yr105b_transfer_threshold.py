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
import dataclasses
import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from statistics import fmean

from ..integrated.evalkit import CHANNELS, check_guards, paired, required_n
from ..integrated.profiles import build_calibrated_profile
from ..integrated.repro import git_dirty, repro_stamp
from ..integrated.scenario_gen import generate_terminal_scenario
from ..integrated.seedbank import assign_band, independence_report, realization_hash
from ..integrated.statfuncs import sd_upper_conf, t_ppf
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
    params = dataclasses.replace(y5._params(cell), vessel_deadline_achievable=True)
    return generate_terminal_scenario(build_calibrated_profile(), seed, params)


def _activate_contract() -> None:
    """공개 함수 직접 호출에서도 정합 마감이 조용히 꺼지지 않게 한다."""
    y5.ACHIEVABLE_DEADLINE = True


@lru_cache(maxsize=1)
def _historical_hashes() -> frozenset[str]:
    """이미 열어 본 YR-105/YR-113 실현을 신규 세 대역에서 배제한다."""
    from . import yr113_transfer_net_effect as y113

    _activate_contract()
    hashes = set(y5._legacy_hashes())
    select_105, _ = y5.resolve_band("select", 24)
    confirm_105, _ = y5.resolve_band("confirm", 48)
    for seeds in (select_105, confirm_105):
        for block, values in seeds.items():
            cell = y5.CELL_A if block == "A" else y5.CELL_B
            hashes |= {
                realization_hash(_generate(block, cell, seed))
                for seed in values
            }
    pilot_113 = y113._band("pilot", 8)
    select_113 = y113._band("select", 53, set(pilot_113.all_hashes))
    confirm_113 = y113._band(
        "confirm", 106, set(pilot_113.all_hashes) | set(select_113.all_hashes))
    hashes |= set(pilot_113.all_hashes) | set(select_113.all_hashes) | set(confirm_113.all_hashes)
    return frozenset(hashes)


def _require_clean() -> None:
    if git_dirty() is not False:
        raise RuntimeError("판정 실행은 추적 파일 변경이 없는 clean commit에서만 가능하다")


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
    return set(_band("pilot", PILOT_N, exclude=_historical_hashes())[0].all_hashes)


def select_hashes(n: int) -> set[str]:
    return set(_band(
        "select", n, exclude=_historical_hashes() | pilot_hashes())[0].all_hashes)


def run_seed(i: int, stage: str, seeds: dict[str, int],
             thresholds: tuple[float, ...]) -> dict:
    _activate_contract()
    arms = {}
    traces = {}
    for tau in thresholds:
        log: list[dict] = []
        arm = y5.run_arm(
            i,
            stage,
            vessel_guard=False,
            seeds=seeds,
            gap_threshold=tau,
            log=log,
        )
        key = f"{tau:.2f}"
        arms[key] = arm
        transferred = [
            {k: rec.get(k) for k in ("t", "job", "src", "gap", "gap_threshold")}
            for rec in log if rec.get("transferred")
        ]
        traces[key] = {
            "transfer_events": transferred,
            "decision_digest": hashlib.sha256(
                json.dumps(log, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16],
        }
    jobs = {a["n_jobs"] for a in arms.values()}
    if len(jobs) != 1:
        raise AssertionError(f"arm별 작업 수 불일치: {jobs}")
    return {
        "seed": i,
        "seed_A": seeds["A"],
        "seed_B": seeds["B"],
        "arms": arms,
        "traces": traces,
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
    for i, row in enumerate(rows):
        for key in keys:
            arm = row["arms"][key]
            if abs(arm["total"] - arm["chan"]["total"]) > 0.02:
                rep.ok = False
                rep.failures.append(
                    f"row{i}/tau{key}: 채널합 {arm['chan']['total']} != total {arm['total']}")
    if rows and len(keys) > 1:
        distinct = any(
            len({row["traces"][key]["decision_digest"] for key in keys}) > 1
            for row in rows
        )
        if not distinct:
            rep.ok = False
            rep.failures.append("모든 임계 arm의 결정 trace가 동일 — 조작 미발화")
    return rep


def _run_rows(stage: str, n: int, thresholds: tuple[float, ...],
              *, exclude: set[str] | None = None,
              reveal_metrics: bool = True) -> tuple[list[dict], dict]:
    spec, independence = _band(stage, n, exclude=exclude)
    rows = []
    for i in range(n):
        seeds = {"A": spec.seeds["A"][i], "B": spec.seeds["B"][i]}
        row = run_seed(i, stage, seeds, thresholds)
        rows.append(row)
        if reveal_metrics:
            summary = " ".join(
                f"tau={tau:.2f}:{row['arms'][f'{tau:.2f}']['chan'][PRIMARY]:.2f}"
                f"/mv{row['arms'][f'{tau:.2f}']['n_moved']}"
                for tau in thresholds
            )
            print(f"[{stage} {i + 1}/{n}] {summary}", flush=True)
        else:
            print(f"[{stage} {i + 1}/{n}] 분산 표본 수집", flush=True)
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
    _require_clean()
    OUT.mkdir(parents=True, exist_ok=True)
    _activate_contract()
    rows, band = _run_rows(
        "pilot", PILOT_N, GRID, exclude=set(_historical_hashes()), reveal_metrics=False)
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
        sd_upper = sd_upper_conf(p.sd, p.n - 1, SD_CONF)
        power[f"{left:.2f}-{right:.2f}"] = {
            "pilot_n": p.n,
            "pilot_sd": round(p.sd, 6),
            "pilot_sd_upper80": round(sd_upper, 6),
            "conservative_n": need,
        }
        if need is not None:
            needs.append(need)
    n_select = max([24, *needs])
    n_confirm = n_select * 2
    for item in power.values():
        df = n_select - 1
        planned_mde = (
            t_ppf(0.975, df) + t_ppf(0.80, df)
        ) * item["pilot_sd_upper80"] / n_select ** 0.5
        item["planned_select_mde80"] = round(planned_mde, 6)
        if planned_mde > DELTA[PRIMARY] + 1e-9:
            raise AssertionError(f"계획 MDE {planned_mde:.4f} > δ={DELTA[PRIMARY]}")
    guards = _guard(rows, GRID)
    result = {
        "repro": repro_stamp(
            experiment="YR-105-b 상대 혼잡격차 임계 검정력 파일럿",
            seeds=band["band"]["seeds"],
            params={"cell_A": y5._params(y5.CELL_A), "cell_B": y5._params(y5.CELL_B)},
            profile_id=build_calibrated_profile().terminal_id,
            prereg="격자 고정 후 pilot16; 평균은 선택 금지, 세 쌍의 분산만 표본수에 사용",
        ),
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
        "sealed": "pilot arm 평균·CI·raw row는 선택 전 누출 방지를 위해 저장하지 않음",
    }
    (OUT / "power_note.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"표본 동결: select={n_select}, confirm={n_confirm}", flush=True)
    return result


def run_select(n: int) -> dict:
    _require_clean()
    OUT.mkdir(parents=True, exist_ok=True)
    _activate_contract()
    power_path = OUT / "power_note.json"
    if not power_path.exists():
        raise FileNotFoundError("power_note.json이 없다 — pilot을 먼저 실행해야 한다")
    power = json.loads(power_path.read_text(encoding="utf-8"))
    expected_n = int(power["frozen_sample_plan"]["n_select"])
    if not power["guards"]["ok"] or n != expected_n:
        raise ValueError(f"pilot guard 또는 표본계약 위반: 요청 n={n}, 동결 n={expected_n}")
    excluded = _historical_hashes() | pilot_hashes()
    rows, band = _run_rows("select", n, GRID, exclude=excluded)
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
        "repro": repro_stamp(
            experiment="YR-105-b 상대 혼잡격차 임계 선택",
            seeds=band["band"]["seeds"],
            params={"cell_A": y5._params(y5.CELL_A), "cell_B": y5._params(y5.CELL_B)},
            profile_id=build_calibrated_profile().terminal_id,
            prereg="세 arm 평균 truck 비용 순위만 사용; 유의성 주장은 독립 확증에서만",
        ),
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_winner() -> dict:
    """선택 결과와 확증 대역을 파일 하나로 결박한다. 이 파일을 커밋한 뒤에만 확증한다."""
    _require_clean()
    select_path = OUT / "results_select.json"
    power_path = OUT / "power_note.json"
    if not select_path.exists() or not power_path.exists():
        raise FileNotFoundError("power_note.json과 results_select.json이 모두 필요하다")
    selection = json.loads(select_path.read_text(encoding="utf-8"))
    power = json.loads(power_path.read_text(encoding="utf-8"))
    if (selection["selection"] != "CANDIDATE" or not selection["verdict_valid"]
            or not selection["guards"]["ok"]):
        raise RuntimeError("유효한 CANDIDATE가 아니므로 승자를 동결할 수 없다")
    winner = float(selection["winner"])
    n_select = int(power["frozen_sample_plan"]["n_select"])
    n_confirm = int(power["frozen_sample_plan"]["n_confirm"])
    if len(selection["rows"]) != n_select:
        raise RuntimeError("선택 row 수가 pilot 동결 표본수와 다르다")
    exclude = set(_historical_hashes()) | pilot_hashes() | select_hashes(n_select)
    confirm_band, independence = _band("confirm", n_confirm, exclude=exclude)
    freeze = {
        "schema": "yr105b-winner-freeze-v1",
        "winner": winner,
        "base": BASE,
        "n_select": n_select,
        "n_confirm": n_confirm,
        "selection_sha256": _sha256(select_path),
        "power_note_sha256": _sha256(power_path),
        "selection_git": selection["repro"]["code"]["git_head"],
        "confirm_band": confirm_band.freeze_json(),
        "confirm_independence": independence,
        "rule": "이 파일을 별도 commit한 뒤 winner·n·confirm digest 변경 금지",
    }
    (OUT / "winner_freeze.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"승자 동결 파일 생성: tau={winner:.2f}, confirm n={n_confirm}", flush=True)
    return freeze


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


def _benefit_by_channel(rows: list[dict], base_key: str, winner_key: str) -> dict[str, dict]:
    out = {}
    for channel in list(CHANNELS) + ["total"]:
        diffs = [
            row["arms"][base_key]["chan"][channel]
            - row["arms"][winner_key]["chan"][channel]
            for row in rows
        ]
        item = paired(
            diffs, delta_interest=DELTA[channel], sd_conf=SD_CONF).as_dict()
        # 공용 evalkit의 양수=처리비용 악화 라벨은 여기의 양수=편익 부호와 반대다.
        item.pop("label", None)
        item["role"] = (
            "1차 확증(양수=후보 개선)" if channel == PRIMARY
            else "채택 비열등 guard(양수=후보 개선)"
            if channel in ("vessel", "total")
            else "진단(양수=후보 개선)"
        )
        out[channel] = item
    return out


def run_confirm(n: int, winner: float) -> dict:
    _require_clean()
    _activate_contract()
    if winner not in GRID or winner == BASE:
        raise ValueError("확증 승자는 0.05 또는 0.20이어야 한다")
    frozen = OUT / "winner_freeze.json"
    if not frozen.exists():
        raise FileNotFoundError("winner_freeze.json이 없다 — 승자를 먼저 별도 커밋으로 동결해야 한다")
    freeze = json.loads(frozen.read_text(encoding="utf-8"))
    if float(freeze["winner"]) != winner or int(freeze["n_confirm"]) != n:
        raise ValueError("CLI 승자·표본수가 동결 파일과 다르다")
    select_path = OUT / "results_select.json"
    power_path = OUT / "power_note.json"
    if (_sha256(select_path) != freeze["selection_sha256"]
            or _sha256(power_path) != freeze["power_note_sha256"]):
        raise RuntimeError("동결 뒤 선택·검정력 원자료가 바뀌었다")
    selection = json.loads(select_path.read_text(encoding="utf-8"))
    if (selection["selection"] != "CANDIDATE"
            or float(selection["winner"]) != winner
            or not selection["guards"]["ok"]):
        raise RuntimeError("동결 승자와 선택 판정이 일치하지 않는다")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(frozen)],
        capture_output=True, text=True)
    unchanged = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(frozen)]).returncode == 0
    if tracked.returncode != 0 or not unchanged:
        raise RuntimeError("winner_freeze.json이 현재 HEAD에 동결돼 있지 않다")
    exclude = _historical_hashes() | pilot_hashes() | select_hashes(int(freeze["n_select"]))
    thresholds = (BASE, winner)
    rows, band = _run_rows("confirm", n, thresholds, exclude=exclude)
    if band["band"] != freeze["confirm_band"]:
        raise RuntimeError("실행한 확증 대역이 동결된 seed·지문과 다르다")
    guards = _guard(rows, thresholds)
    bk, wk = f"{BASE:.2f}", f"{winner:.2f}"
    # 편익 = 현행 비용 - 후보 비용. 양수면 후보가 더 싸다.
    channels = _benefit_by_channel(rows, bk, wk)
    verdict = _classification(channels, guards.ok)
    result = {
        "repro": repro_stamp(
            experiment="YR-105-b 상대 혼잡격차 임계 정책 독립 확증",
            seeds=band["band"]["seeds"],
            params={"cell_A": y5._params(y5.CELL_A), "cell_B": y5._params(y5.CELL_B),
                    "grid": GRID, "base": BASE, "winner": winner},
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
        "exact_command": (
            f"python -m yard_rl.experiments.yr105b_transfer_threshold "
            f"--stage confirm --seeds {n} --winner {winner:.2f}"),
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
    parser.add_argument("--stage", required=True, choices=("pilot", "select", "freeze", "confirm"))
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
    elif args.stage == "freeze":
        result = freeze_winner()
    else:
        if args.seeds is None or args.winner is None:
            raise SystemExit("--seeds와 --winner 필요")
        result = run_confirm(args.seeds, args.winner)
    valid = result.get("verdict_valid", result.get("guards", {}).get("ok", True))
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
