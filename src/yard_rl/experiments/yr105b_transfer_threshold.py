"""YR-105-b — 창중 이송의 상대 혼잡격차 임계값 단일축 최적화.

유일한 정책 변화는 ``gap >= tau``의 ``tau``다. 후보는 0.05, 0.10(현행),
0.20이다. 총비용과 평균 게이트 진입→진출 시간(A→O)을 공동 주지표로 삼고,
두 지표가 함께 좋아야 성공으로 판정한다.

실행 단계는 반드시 별도 커밋으로 끊는다.

``manifest -> pilot -> select -> freeze -> confirm``
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
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
MANIFEST = OUT / "prereg_manifest.json"
POWER_NOTE = OUT / "power_note.json"
SELECT_RESULT = OUT / "results_select.json"
WINNER_FREEZE = OUT / "winner_freeze.json"
CONFIRM_RESULT = OUT / "results_confirm.json"

GRID = (0.05, 0.10, 0.20)
BASE = 0.10
CO_PRIMARY = ("total", "a2o_min")
DELTA = {
    "total": 10.0,
    "a2o_min": 1.0,
    "truck": 3.0,
    "vessel": 10.0,
    "move": 1.0,
    "other": 1.0,
}
SD_CONF = 0.80
ENDPOINT_POWER = 0.90
PILOT_N = 16
BAND_START = {"pilot": 920_000, "select": 930_000, "confirm": 950_000}

CONTRACT_PATHS = (
    "src/yard_rl/experiments/yr105b_transfer_threshold.py",
    "src/yard_rl/experiments/yr105_conditional_transfer.py",
    "src/yard_rl/experiments/yr113_transfer_net_effect.py",
    "src/yard_rl/integrated/block_congestion.py",
    "src/yard_rl/integrated/candidates.py",
    "src/yard_rl/integrated/engine.py",
    "src/yard_rl/integrated/evalkit.py",
    "src/yard_rl/integrated/multiblock.py",
    "src/yard_rl/integrated/profiles.py",
    "src/yard_rl/integrated/repro.py",
    "src/yard_rl/integrated/resolver.py",
    "src/yard_rl/integrated/scenario_gen.py",
    "src/yard_rl/integrated/seedbank.py",
    "src/yard_rl/integrated/statfuncs.py",
    "configs/costs/numeraire_v1.yaml",
    ".claude/docs/dashboard-task-specs/YR-105-b-transfer-threshold.md",
    ".claude/docs/strategy-history/2026-07-28-YR-105-b-상대혼잡격차-임계최적화-prereg.md",
)


def _generate(_key: str, cell, seed: int):
    params = dataclasses.replace(y5._params(cell), vessel_deadline_achievable=True)
    return generate_terminal_scenario(build_calibrated_profile(), seed, params)


def _activate_contract() -> None:
    y5.ACHIEVABLE_DEADLINE = True


@lru_cache(maxsize=1)
def _historical_hashes() -> frozenset[str]:
    """이미 열어 본 YR-105/YR-113 실현을 신규 대역에서 제외한다."""
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
    hashes |= set(pilot_113.all_hashes) | set(select_113.all_hashes)
    hashes |= set(confirm_113.all_hashes)
    return frozenset(hashes)


def _run_git(*args: str) -> bytes:
    p = subprocess.run(["git", *args], capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} 실패: {p.stderr.decode(errors='replace').strip()}")
    return p.stdout


def _require_clean() -> None:
    if git_dirty() is not False:
        raise RuntimeError("판정 실행은 추적 파일 변경이 없는 clean commit에서만 가능하다")
    # 실행 코드·설정·사전등록 문서 아래의 untracked 파일도 계약 우회를 만들 수 있다.
    watched = (
        "src", "configs",
        ".claude/docs/dashboard-task-specs/YR-105-b-transfer-threshold.md",
        ".claude/docs/strategy-history/"
        "2026-07-28-YR-105-b-상대혼잡격차-임계최적화-prereg.md",
    )
    p = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *watched],
        capture_output=True, text=True)
    if p.returncode != 0 or p.stdout.strip():
        raise RuntimeError(f"실행 계약 경로가 clean하지 않다: {p.stdout.strip()}")


def _head_blob(path: str) -> bytes:
    _run_git("ls-files", "--error-unmatch", path)
    return _run_git("show", f"HEAD:{path}")


def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _source_snapshot() -> dict:
    files = {path: _sha256_bytes(_head_blob(path)) for path in CONTRACT_PATHS}
    digest = _sha256_bytes(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode())
    return {"files": files, "digest": digest}


def _relative(path: Path) -> str:
    return path.resolve().relative_to(Path.cwd().resolve()).as_posix()


def _require_head_artifact(path: Path) -> dict:
    """산출물이 현재 HEAD에 추적·동결돼 있는지 확인하고 JSON을 읽는다."""
    rel = _relative(path)
    blob = _head_blob(rel)
    if path.read_bytes() != blob:
        raise RuntimeError(f"{rel}이 현재 HEAD와 다르다 — 별도 커밋으로 먼저 동결해야 한다")
    return json.loads(blob.decode("utf-8"))


def _require_source_contract(manifest: dict) -> None:
    current = _source_snapshot()
    frozen = manifest.get("source_contract", {})
    if current != frozen:
        changed = sorted(
            p for p in set(current.get("files", {})) | set(frozen.get("files", {}))
            if current.get("files", {}).get(p) != frozen.get("files", {}).get(p))
        raise RuntimeError(f"사전등록 뒤 정책·물리·설정 계약이 바뀌었다: {changed}")


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


def _manifest() -> dict:
    manifest = _require_head_artifact(MANIFEST)
    _require_source_contract(manifest)
    return manifest


def pilot_hashes() -> set[str]:
    if MANIFEST.exists():
        m = json.loads(MANIFEST.read_text(encoding="utf-8"))
        return {
            h for values in m["pilot_band"]["realization_hashes"].values()
            for h in values
        }
    return set(_band("pilot", PILOT_N, exclude=set(_historical_hashes()))[0].all_hashes)


def select_hashes(n: int) -> set[str]:
    return set(_band(
        "select", n, exclude=set(_historical_hashes()) | pilot_hashes())[0].all_hashes)


def build_manifest() -> dict:
    """결과를 만들기 전에 격자·지표·pilot 대역·소스 계약을 동결한다."""
    _require_clean()
    if MANIFEST.exists():
        raise RuntimeError("prereg_manifest.json이 이미 있다 — 사전등록을 덮어쓸 수 없다")
    _activate_contract()
    historical = sorted(_historical_hashes())
    pilot, independence = _band("pilot", PILOT_N, exclude=set(historical))
    result = {
        "schema": "yr105b-prereg-v2",
        "status": "RESULTS_UNSEEN_FROZEN",
        "question": "tau 0.05/0.20 중 현행 0.10보다 total과 A→O를 함께 낮추는 값이 있는가",
        "grid": list(GRID),
        "base": BASE,
        "co_primary": list(CO_PRIMARY),
        "delta_assumed": {"total": DELTA["total"], "a2o_min": DELTA["a2o_min"]},
        "pilot_n": PILOT_N,
        "power": {
            "endpoint_power": ENDPOINT_POWER,
            "sd_upper_confidence": SD_CONF,
            "alpha_two_sided": 0.05,
            "n_select_rule": "max(24, 3 pairs × 2 endpoints의 보수 필요 n)",
            "n_confirm_rule": "2 * n_select",
        },
        "selection_rule": (
            "두 편익 평균이 모두 양수인 후보 중 "
            "min(B_total/10, B_a2o/1) 최대; 동률은 0.20, 없으면 NO_CANDIDATE"
        ),
        "confirmation_rule": (
            "B=Metric(0.10)-Metric(winner); total과 A→O의 95% CI 하한이 모두 >0"
        ),
        "historical_hashes": {
            "count": len(historical),
            "digest": _sha256_bytes(
                json.dumps(historical, separators=(",", ":")).encode()),
        },
        "pilot_band": pilot.freeze_json(),
        "pilot_independence": independence,
        "source_contract": _source_snapshot(),
        "command": (
            "python -m yard_rl.experiments.yr105b_transfer_threshold --stage pilot"
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def _canonical_trace(log: list[dict]) -> list[dict]:
    """임계 숫자·혼잡 점수는 빼고 실제 결정·이송만 정준화한다."""
    fields = (
        "t", "job", "src", "dst", "fired", "blocked_by_vessel",
        "transferred", "rejected",
    )
    return [{k: rec.get(k) for k in fields} for rec in log]


def run_seed(i: int, stage: str, seeds: dict[str, int],
             thresholds: tuple[float, ...]) -> dict:
    _activate_contract()
    arms: dict[str, dict] = {}
    traces: dict[str, dict] = {}
    for tau in thresholds:
        log: list[dict] = []
        arm = y5.run_arm(
            i, stage, vessel_guard=False, seeds=seeds, gap_threshold=tau, log=log)
        key = f"{tau:.2f}"
        canonical = _canonical_trace(log)
        transferred = [rec for rec in canonical if rec.get("transferred")]
        rejected = [rec for rec in canonical if rec.get("rejected")]
        if len(transferred) != arm["n_moved"]:
            raise AssertionError(
                f"tau {key}: trace 이송 {len(transferred)} != arm {arm['n_moved']}")
        if len(rejected) != arm["n_rejected"]:
            raise AssertionError(
                f"tau {key}: trace reject {len(rejected)} != arm {arm['n_rejected']}")
        arms[key] = arm
        traces[key] = {
            "transfer_events": transferred,
            "rejected_events": rejected,
            "action_digest": _sha256_bytes(
                json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode())[:16],
        }
    jobs = {arm["n_jobs"] for arm in arms.values()}
    if len(jobs) != 1:
        raise AssertionError(f"arm별 작업 수 불일치: {jobs}")
    return {
        "seed": i,
        "seed_A": seeds["A"],
        "seed_B": seeds["B"],
        "arms": arms,
        "traces": traces,
    }


def _arm_metric(arm: dict, metric: str) -> float:
    if metric == "total":
        value = arm.get("total_raw", arm.get("total"))
    elif metric == "a2o_min":
        value = arm.get("a2o_mean_min_raw", arm.get("a2o_mean_min"))
    else:
        value = arm["chan"][metric]
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"{metric} 누락·비유한 값")
    return float(value)


def _guard(rows: list[dict], thresholds: tuple[float, ...]):
    keys = [f"{t:.2f}" for t in thresholds]
    rep = check_guards([
        {"compl": row["arms"][key]["compl"], "backlog": row["arms"][key]["backlog"]}
        for row in rows for key in keys
    ])
    exceptions = sum(
        row["arms"][key]["policy_exceptions"] for row in rows for key in keys)
    if exceptions:
        rep.ok = False
        rep.failures.append(f"정책 예외 {exceptions}건")
    for i, row in enumerate(rows):
        n_a2o: set[int] = set()
        for key in keys:
            arm = row["arms"][key]
            try:
                _arm_metric(arm, "total")
                _arm_metric(arm, "a2o_min")
            except ValueError as exc:
                rep.ok = False
                rep.failures.append(f"row{i}/tau{key}: {exc}")
            n_a2o.add(int(arm.get("n_a2o", 0)))
            if abs(arm["total"] - arm["chan"]["total"]) > 0.02:
                rep.ok = False
                rep.failures.append(
                    f"row{i}/tau{key}: 채널합 {arm['chan']['total']} != total {arm['total']}")
        if n_a2o == {0} or len(n_a2o) != 1:
            rep.ok = False
            rep.failures.append(f"row{i}: arm별 A→O 표본수 불일치·0 {sorted(n_a2o)}")
    if rows and len(keys) > 1:
        distinct = any(
            len({row["traces"][key]["action_digest"] for key in keys}) > 1
            for row in rows
        )
        if not distinct:
            rep.ok = False
            rep.failures.append("모든 임계 arm의 실제 행동 trace가 동일 — 조작 미발화")
    return rep


def _run_rows(stage: str, n: int, thresholds: tuple[float, ...], *,
              exclude: set[str], expected_band: dict,
              reveal_metrics: bool = True) -> tuple[list[dict], dict]:
    spec, independence = _band(stage, n, exclude=exclude)
    if spec.freeze_json() != expected_band:
        raise RuntimeError(f"{stage} 대역이 사전 동결 seed·실현지문과 다르다")
    rows = []
    for i in range(n):
        seeds = {"A": spec.seeds["A"][i], "B": spec.seeds["B"][i]}
        row = run_seed(i, stage, seeds, thresholds)
        rows.append(row)
        if reveal_metrics:
            summary = " ".join(
                f"tau={tau:.2f}:T{_arm_metric(row['arms'][f'{tau:.2f}'],'total'):.2f}"
                f"/A{_arm_metric(row['arms'][f'{tau:.2f}'],'a2o_min'):.2f}"
                f"/mv{row['arms'][f'{tau:.2f}']['n_moved']}"
                for tau in thresholds
            )
            print(f"[{stage} {i + 1}/{n}] {summary}", flush=True)
        else:
            print(f"[{stage} {i + 1}/{n}] 봉인 분산 표본 수집", flush=True)
    return rows, {"band": spec.freeze_json(), "independence": independence}


def _pair_metric(rows: list[dict], left: float, right: float,
                 metric: str) -> list[float]:
    lk, rk = f"{left:.2f}", f"{right:.2f}"
    return [
        _arm_metric(row["arms"][lk], metric)
        - _arm_metric(row["arms"][rk], metric)
        for row in rows
    ]


def _mde(sd: float, n: int, power: float = ENDPOINT_POWER) -> float:
    df = n - 1
    return (t_ppf(0.975, df) + t_ppf(power, df)) * sd / n ** 0.5


def run_pilot() -> dict:
    """평균은 봉인하고 3쌍×2지표 분산만 열어 표본수를 정한다."""
    _require_clean()
    manifest = _manifest()
    rows, band = _run_rows(
        "pilot", PILOT_N, GRID,
        exclude=set(_historical_hashes()),
        expected_band=manifest["pilot_band"],
        reveal_metrics=False,
    )
    guards = _guard(rows, GRID)
    pairs = ((0.05, 0.10), (0.20, 0.10), (0.05, 0.20))
    power_by_pair: dict[str, dict] = {}
    needs: list[int] = []
    for left, right in pairs:
        for metric in CO_PRIMARY:
            diffs = _pair_metric(rows, left, right, metric)
            p = paired(diffs, delta_interest=DELTA[metric], sd_conf=SD_CONF)
            need = required_n(
                p.sd, DELTA[metric], power=ENDPOINT_POWER,
                sd_conf=SD_CONF, sd_df=p.n - 1)
            upper = sd_upper_conf(p.sd, p.n - 1, SD_CONF)
            key = f"{left:.2f}-{right:.2f}/{metric}"
            power_by_pair[key] = {
                "metric": metric,
                "pilot_n": p.n,
                "pilot_sd": round(p.sd, 9),
                "pilot_sd_upper80": round(upper, 9),
                "conservative_n_power90": need,
            }
            if need is None:
                raise RuntimeError(f"{key}: 필요 표본수를 계산하지 못했다")
            needs.append(need)
    n_select = max(24, *needs)
    n_confirm = 2 * n_select
    for key, item in power_by_pair.items():
        planned = _mde(item["pilot_sd_upper80"], n_select)
        item["planned_select_mde90"] = round(planned, 9)
        if planned > DELTA[item["metric"]] + 1e-9:
            raise AssertionError(
                f"{key}: 계획 MDE90 {planned:.4f} > δ={DELTA[item['metric']]}")

    select_spec, select_independence = _band(
        "select", n_select,
        exclude=set(_historical_hashes()) | set(manifest["pilot_band"]
                                                ["realization_hashes"]["A"])
        | set(manifest["pilot_band"]["realization_hashes"]["B"]),
    )
    result = {
        "repro": repro_stamp(
            experiment="YR-105-b 공동 주지표 검정력 파일럿",
            seeds=band["band"]["seeds"],
            params={"cell_A": y5._params(y5.CELL_A), "cell_B": y5._params(y5.CELL_B)},
            profile_id=build_calibrated_profile().terminal_id,
            prereg="결과 미열람 manifest 뒤 pilot16; 평균 봉인, 6개 분산만 표본수에 사용",
        ),
        "schema": "yr105b-power-v2",
        "stage": "pilot",
        "manifest_sha256": _sha256(MANIFEST),
        "source_contract_digest": manifest["source_contract"]["digest"],
        "contract": {
            "grid": list(GRID),
            "base": BASE,
            "co_primary": list(CO_PRIMARY),
            "delta": {m: DELTA[m] for m in CO_PRIMARY},
            "endpoint_power": ENDPOINT_POWER,
            "sd_conf": SD_CONF,
            "pilot_means_not_for_selection": True,
        },
        "band": band,
        "power_by_pair": power_by_pair,
        "frozen_sample_plan": {
            "n_select": n_select,
            "n_confirm": n_confirm,
            "select_band": select_spec.freeze_json(),
            "select_independence": select_independence,
            "rule": "max(24, 3쌍×2지표 보수 필요 n), 확증=선택×2",
        },
        "guards": {"ok": guards.ok, "failures": guards.failures},
        "sealed": "pilot arm 평균·CI·raw row는 저장·출력하지 않음",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    POWER_NOTE.write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"표본 동결: select={n_select}, confirm={n_confirm}", flush=True)
    return result


def _benefit_means(rows: list[dict], candidate: float) -> dict[str, float]:
    ck, bk = f"{candidate:.2f}", f"{BASE:.2f}"
    return {
        metric: fmean(
            _arm_metric(row["arms"][bk], metric)
            - _arm_metric(row["arms"][ck], metric)
            for row in rows
        )
        for metric in CO_PRIMARY
    }


def _trace_diagnostics(rows: list[dict], thresholds: tuple[float, ...]) -> dict:
    keys = [f"{tau:.2f}" for tau in thresholds]
    out: dict[str, dict] = {}
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            out[f"{left}-{right}"] = {
                "different_seed_count": sum(
                    row["traces"][left]["action_digest"]
                    != row["traces"][right]["action_digest"] for row in rows),
                "mean_abs_transfer_count_diff": round(fmean(
                    abs(row["arms"][left]["n_moved"]
                        - row["arms"][right]["n_moved"]) for row in rows), 6),
            }
    return out


def _select_candidate(benefits: dict[str, dict[str, float]],
                      guards_ok: bool = True) -> tuple[float | None, str, dict[str, float]]:
    eligible = {
        key: value for key, value in benefits.items()
        if value["total"] > 0 and value["a2o_min"] > 0
    }
    scores = {
        key: min(value["total"] / DELTA["total"],
                 value["a2o_min"] / DELTA["a2o_min"])
        for key, value in eligible.items()
    }
    if guards_ok and scores:
        # 같은 score면 이송을 덜 여는 보수적 0.20을 우선한다.
        winner_key = sorted(scores, key=lambda k: (-scores[k], -float(k)))[0]
        return float(winner_key), "CANDIDATE", scores
    return None, ("NO_CANDIDATE" if guards_ok else "INVALID"), scores


def run_select() -> dict:
    _require_clean()
    manifest = _manifest()
    power = _require_head_artifact(POWER_NOTE)
    if _sha256(MANIFEST) != power["manifest_sha256"]:
        raise RuntimeError("pilot 뒤 사전등록 manifest가 바뀌었다")
    if power["source_contract_digest"] != manifest["source_contract"]["digest"]:
        raise RuntimeError("pilot과 현재 source contract가 다르다")
    if not power["guards"]["ok"]:
        raise RuntimeError("pilot guard 실패로 선택 대역을 열 수 없다")
    n = int(power["frozen_sample_plan"]["n_select"])
    excluded = set(_historical_hashes()) | pilot_hashes()
    rows, band = _run_rows(
        "select", n, GRID, exclude=excluded,
        expected_band=power["frozen_sample_plan"]["select_band"])
    guards = _guard(rows, GRID)
    benefits = {f"{tau:.2f}": _benefit_means(rows, tau)
                for tau in GRID if tau != BASE}
    winner, selection, scores = _select_candidate(benefits, guards.ok)
    result = {
        "repro": repro_stamp(
            experiment="YR-105-b 상대 혼잡격차 임계 선택",
            seeds=band["band"]["seeds"],
            params={"cell_A": y5._params(y5.CELL_A), "cell_B": y5._params(y5.CELL_B)},
            profile_id=build_calibrated_profile().terminal_id,
            prereg="total·A→O 편익 모두 양수인 후보의 정규화 maximin; 유의성 주장은 금지",
        ),
        "schema": "yr105b-select-v2",
        "stage": "select",
        "manifest_sha256": _sha256(MANIFEST),
        "power_note_sha256": _sha256(POWER_NOTE),
        "source_contract_digest": manifest["source_contract"]["digest"],
        "selection_rule": manifest["selection_rule"],
        "benefit_means_010_minus_candidate": benefits,
        "normalized_maximin_score": scores,
        "winner": winner,
        "selection": selection,
        "verdict_valid": guards.ok,
        "claim_limit": "승자 동결 전 순위 선정 전용; 유의성·효과 주장을 하지 않음",
        "band": band,
        "guards": {"ok": guards.ok, "failures": guards.failures},
        "manipulation": _trace_diagnostics(rows, GRID),
        "rows": rows,
    }
    SELECT_RESULT.write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"선택 결과: {selection} winner={winner}", flush=True)
    return result


def freeze_winner() -> dict:
    """선택 결과와 독립 확증 대역을 별도 파일로 결박한다."""
    _require_clean()
    manifest = _manifest()
    power = _require_head_artifact(POWER_NOTE)
    selection = _require_head_artifact(SELECT_RESULT)
    if selection["selection"] != "CANDIDATE" or not selection["guards"]["ok"]:
        raise RuntimeError("유효한 CANDIDATE가 없어 승자를 동결할 수 없다")
    if selection["power_note_sha256"] != _sha256(POWER_NOTE):
        raise RuntimeError("선택에 사용한 power note와 현재 파일이 다르다")
    winner = float(selection["winner"])
    n_select = int(power["frozen_sample_plan"]["n_select"])
    n_confirm = int(power["frozen_sample_plan"]["n_confirm"])
    if len(selection["rows"]) != n_select:
        raise RuntimeError("선택 row 수가 동결 표본수와 다르다")
    exclude = set(_historical_hashes()) | pilot_hashes() | select_hashes(n_select)
    confirm_band, independence = _band("confirm", n_confirm, exclude=exclude)
    freeze = {
        "schema": "yr105b-winner-freeze-v2",
        "winner": winner,
        "base": BASE,
        "n_select": n_select,
        "n_confirm": n_confirm,
        "manifest_sha256": _sha256(MANIFEST),
        "power_note_sha256": _sha256(POWER_NOTE),
        "selection_sha256": _sha256(SELECT_RESULT),
        "source_contract_digest": manifest["source_contract"]["digest"],
        "confirm_band": confirm_band.freeze_json(),
        "confirm_independence": independence,
        "verdict_rule": manifest["confirmation_rule"],
        "rule": "이 파일을 별도 commit한 뒤 winner·n·confirm 대역 변경 금지",
    }
    WINNER_FREEZE.write_text(
        json.dumps(freeze, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"승자 동결: tau={winner:.2f}, confirm n={n_confirm}", flush=True)
    return freeze


def _benefit_stats(rows: list[dict], winner: float) -> tuple[dict, dict]:
    bk, wk = f"{BASE:.2f}", f"{winner:.2f}"
    primary: dict[str, dict] = {}
    for metric in CO_PRIMARY:
        diffs = [
            _arm_metric(row["arms"][bk], metric)
            - _arm_metric(row["arms"][wk], metric)
            for row in rows
        ]
        p = paired(diffs, delta_interest=DELTA[metric], sd_conf=SD_CONF)
        item = p.as_dict()
        item.pop("label", None)
        item["mde90"] = round(_mde(p.sd, p.n), 6)
        item["delta_assumed"] = DELTA[metric]
        item["role"] = "공동 1차(양수=후보 개선)"
        primary[metric] = item
    diagnostic: dict[str, dict] = {}
    for channel in CHANNELS:
        diffs = [
            row["arms"][bk]["chan"][channel]
            - row["arms"][wk]["chan"][channel]
            for row in rows
        ]
        item = paired(
            diffs, delta_interest=DELTA[channel], sd_conf=SD_CONF).as_dict()
        item["role"] = "진단(양수=후보 개선; 다중비교 보정 없음)"
        diagnostic[channel] = item
    return primary, diagnostic


def _classification(primary: dict[str, dict], guards_ok: bool,
                    power_ok: bool = True) -> str:
    if not guards_ok:
        return "INVALID"
    if not power_ok:
        return "POWER_FAIL"
    total, a2o = primary["total"], primary["a2o_min"]
    lo = {"total": total["ci"][0], "a2o_min": a2o["ci"][0]}
    hi = {"total": total["ci"][1], "a2o_min": a2o["ci"][1]}
    if ((lo["total"] > 0 and hi["a2o_min"] < 0)
            or (lo["a2o_min"] > 0 and hi["total"] < 0)):
        return "TRADEOFF_FAIL"
    if any(hi[m] < 0 for m in CO_PRIMARY):
        return "HARMFUL"
    if all(lo[m] > DELTA[m] for m in CO_PRIMARY):
        return "JOINT_PRACTICAL_IMPROVEMENT"
    if all(lo[m] > 0 for m in CO_PRIMARY):
        return "JOINT_CONFIRMED_SMALL"
    if all(primary[m].get("equivalent") for m in CO_PRIMARY):
        return "EQUIVALENT"
    return "INCONCLUSIVE"


def run_confirm() -> dict:
    _require_clean()
    manifest = _manifest()
    power = _require_head_artifact(POWER_NOTE)
    selection = _require_head_artifact(SELECT_RESULT)
    freeze = _require_head_artifact(WINNER_FREEZE)
    for path, key in (
        (MANIFEST, "manifest_sha256"),
        (POWER_NOTE, "power_note_sha256"),
        (SELECT_RESULT, "selection_sha256"),
    ):
        if freeze[key] != _sha256(path):
            raise RuntimeError(f"동결 뒤 {path.name}이 바뀌었다")
    if freeze["source_contract_digest"] != manifest["source_contract"]["digest"]:
        raise RuntimeError("확증 source contract가 사전등록과 다르다")
    winner = float(freeze["winner"])
    if selection["selection"] != "CANDIDATE" or float(selection["winner"]) != winner:
        raise RuntimeError("동결 승자와 선택 결과가 다르다")
    n = int(freeze["n_confirm"])
    exclude = set(_historical_hashes()) | pilot_hashes() | select_hashes(
        int(freeze["n_select"]))
    thresholds = (BASE, winner)
    rows, band = _run_rows(
        "confirm", n, thresholds, exclude=exclude,
        expected_band=freeze["confirm_band"])
    guards = _guard(rows, thresholds)
    primary, diagnostic = _benefit_stats(rows, winner)
    power_ok = all(
        primary[metric]["mde90"] <= DELTA[metric] + 1e-9
        for metric in CO_PRIMARY
    )
    verdict = _classification(primary, guards.ok, power_ok)
    bk, wk = f"{BASE:.2f}", f"{winner:.2f}"
    result = {
        "repro": repro_stamp(
            experiment="YR-105-b 상대 혼잡격차 임계 독립 확증",
            seeds=band["band"]["seeds"],
            params={"cell_A": y5._params(y5.CELL_A), "cell_B": y5._params(y5.CELL_B),
                    "grid": GRID, "base": BASE, "winner": winner},
            profile_id=build_calibrated_profile().terminal_id,
            prereg="B=Metric(0.10)-Metric(winner); total AND A→O CI 하한>0 공동 확증",
        ),
        "schema": "yr105b-confirm-v2",
        "stage": "confirm",
        "winner_freeze": freeze,
        "co_primary_benefit_010_minus_winner": primary,
        "diagnostic_benefit_010_minus_winner": diagnostic,
        "power_ok": power_ok,
        "verdict": verdict,
        "verdict_valid": guards.ok and power_ok,
        "adoption_note": (
            "JOINT_CONFIRMED_SMALL은 연구적 개선, 운영 채택은 "
            "JOINT_PRACTICAL_IMPROVEMENT에서만 허용"
        ),
        "claim_limit": "동결 승자가 현행 0.10보다 나은지만 확증; 연속 임계 전역최적 금지",
        "exact_command": (
            "python -m yard_rl.experiments.yr105b_transfer_threshold --stage confirm"
        ),
        "band": band,
        "guards": {"ok": guards.ok, "failures": guards.failures},
        "manipulation": _trace_diagnostics(rows, thresholds),
        "mean_moved": {
            bk: round(fmean(row["arms"][bk]["n_moved"] for row in rows), 6),
            wk: round(fmean(row["arms"][wk]["n_moved"] for row in rows), 6),
        },
        "deadlock_escapes": {
            bk: sum(row["arms"][bk]["deadlock_escapes"] for row in rows),
            wk: sum(row["arms"][wk]["deadlock_escapes"] for row in rows),
        },
        "rows": rows,
    }
    CONFIRM_RESULT.write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"확증 판정: {verdict}", flush=True)
    for metric, item in primary.items():
        print(f"  {metric}: {item['mean']:+.3f} CI {item['ci']}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True,
        choices=("manifest", "pilot", "select", "freeze", "confirm"))
    args = parser.parse_args()
    _activate_contract()
    if args.stage == "manifest":
        result = build_manifest()
    elif args.stage == "pilot":
        result = run_pilot()
    elif args.stage == "select":
        result = run_select()
    elif args.stage == "freeze":
        result = freeze_winner()
    else:
        result = run_confirm()
    valid = result.get("verdict_valid", result.get("guards", {}).get("ok", True))
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
