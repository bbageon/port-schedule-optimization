"""YR-115 — 현행 0.10 창중 이송의 total+A→O 공동 순효과 확증."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import subprocess
from pathlib import Path
from statistics import fmean

from ..integrated.evalkit import (CHANNELS, check_guards, judge_primary, paired,
                                  paired_by_channel, required_n)
from ..integrated.profiles import build_calibrated_profile
from ..integrated.repro import repro_stamp
from ..integrated.scenario_gen import generate_terminal_scenario
from ..integrated.seedbank import assign_band, independence_report, realization_hash
from ..integrated.statfuncs import sd_upper_conf, t_ppf
from . import yr105_conditional_transfer as y5
from . import yr105b_transfer_threshold as y105b

OUT = Path("outputs/reports/yr115_transfer_joint_confirm")
MANIFEST = OUT / "prereg_manifest.json"
POWER_NOTE = OUT / "power_note.json"
RESULT = OUT / "results_confirm.json"

BASE = 0.10
NO_TRANSFER = float("inf")
METRICS = ("total", "a2o_min")
METRIC_KEYS = {
    "total": ("total_raw", "total"),
    "a2o_min": ("a2o_mean_min_raw", "a2o_mean_min"),
}
DELTA = {"total": 10.0, "a2o_min": 1.0}
PLAN_EFFECT = {"total": 3.70, "a2o_min": 0.82}
DIAG_DELTA = {"truck": 3.0, "vessel": 10.0, "move": 1.0, "other": 1.0,
              "total": 10.0}
PILOT_N = 16
SD_CONF = 0.80
ENDPOINT_POWER = 0.90
BAND_START = {"pilot": 970_000, "confirm": 980_000}

CONTRACT_FILES = (
    "src/yard_rl/experiments/yr115_transfer_joint_confirm.py",
    "src/yard_rl/experiments/yr105_conditional_transfer.py",
    "src/yard_rl/experiments/yr105b_transfer_threshold.py",
    "src/yard_rl/experiments/yr103_info_sufficiency.py",
    "src/yard_rl/experiments/yr113_transfer_net_effect.py",
    "configs/terminals/dgt_armg.yaml",
    "outputs/reports/yr105b_transfer_threshold/prereg_manifest.json",
    "outputs/reports/yr105b_transfer_threshold/power_note.json",
    "outputs/reports/yr105b_transfer_threshold/results_select.json",
    ".claude/docs/dashboard-task-specs/YR-115-transfer-benefit-joint-confirm.md",
    ".claude/docs/strategy-history/2026-07-28-YR-115-이송순효과-공동지표-prereg.md",
)
CONTRACT_TREES = (
    "src/yard_rl/integrated",
    "src/yard_rl/domain",
    "src/yard_rl/sim",
    "src/yard_rl/contract",
    "src/yard_rl/io",
    "configs/costs",
)
# 이 테스트 probe는 사전등록 전에 결과를 열어 본 개발 실현이다. YR-108 계약에 따라
# realization hash뿐 아니라 같은 정수 RNG seed를 반대 블록에서 재사용하는 것도 막는다.
DEVELOPMENT_PROBE_SEEDS = (970_000, 970_001)


def _generate(_key: str, cell, seed: int):
    params = dataclasses.replace(y5._params(cell), vessel_deadline_achievable=True)
    return generate_terminal_scenario(build_calibrated_profile(), seed, params)


def _activate_contract() -> None:
    y5.ACHIEVABLE_DEADLINE = True


def _run_git(*args: str) -> bytes:
    proc = subprocess.run(["git", *args], capture_output=True)
    if proc.returncode:
        raise RuntimeError(
            f"git {' '.join(args)} 실패: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout


def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _head_blob(path: str) -> bytes:
    _run_git("ls-files", "--error-unmatch", path)
    return _run_git("show", f"HEAD:{path}")


def _contract_paths() -> tuple[str, ...]:
    """실행에 쓰이는 core tree와 직접 import 파일을 HEAD 기준으로 완전 열거한다."""
    tracked = _run_git("ls-files", "--", *CONTRACT_TREES).decode().splitlines()
    paths = tuple(sorted(set(CONTRACT_FILES) | set(tracked)))
    if not tracked:
        raise RuntimeError("실행 core tree 추적 파일을 찾지 못했다")
    return paths


def _source_snapshot() -> dict:
    paths = _contract_paths()
    files = {path: _sha256_bytes(_head_blob(path)) for path in paths}
    return {
        "files": files,
        "digest": _sha256_bytes(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()),
        "scope": {
            "direct_files": list(CONTRACT_FILES),
            "tracked_trees": list(CONTRACT_TREES),
        },
    }


def _require_clean() -> None:
    # 공유 작업공간의 무관한 Dashboard/다른 실험 변경은 실행 계약이 아니다. 대신 실제
    # runtime tree·설정·본 사전등록 파일은 tracked/untracked 모두 깨끗해야 한다.
    watched = (*CONTRACT_TREES, *CONTRACT_FILES)
    proc = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *watched],
        capture_output=True, text=True)
    if proc.returncode or proc.stdout.strip():
        raise RuntimeError(f"실험 계약 경로가 clean하지 않다: {proc.stdout.strip()}")


def _relative(path: Path) -> str:
    return path.resolve().relative_to(Path.cwd().resolve()).as_posix()


def _require_head_artifact(path: Path) -> dict:
    blob = _head_blob(_relative(path))
    if path.read_bytes() != blob:
        raise RuntimeError(f"{path.name}이 현재 HEAD와 다르다")
    return json.loads(blob.decode("utf-8"))


def _require_source_contract(manifest: dict) -> None:
    current = _source_snapshot()
    if current != manifest.get("source_contract"):
        changed = sorted(
            path for path in set(current["files"]) | set(
                manifest.get("source_contract", {}).get("files", {}))
            if current["files"].get(path)
            != manifest.get("source_contract", {}).get("files", {}).get(path)
        )
        raise RuntimeError(f"사전등록 뒤 소스 계약 변경: {changed}")


def _hashes_from_band(band: dict) -> set[str]:
    return {
        value for values in band["realization_hashes"].values() for value in values
    }


def _prior_hashes() -> set[str]:
    """기열람 YR-105/113/105-b 대역을 모두 제외한다."""
    hashes = set(y105b._historical_hashes())
    for path, keys in (
        (y105b.MANIFEST, ("pilot_band",)),
        (y105b.POWER_NOTE, ("frozen_sample_plan", "select_band")),
    ):
        data = _require_head_artifact(path)
        obj = data
        for key in keys:
            obj = obj[key]
        hashes |= _hashes_from_band(obj)
    select = _require_head_artifact(y105b.SELECT_RESULT)
    hashes |= _hashes_from_band(select["band"]["band"])
    for seed in DEVELOPMENT_PROBE_SEEDS:
        for block, cell in (("A", y5.CELL_A), ("B", y5.CELL_B)):
            hashes.add(realization_hash(_generate(block, cell, seed)))
    return hashes


def _prior_summary(prior: set[str] | None = None) -> dict:
    values = sorted(_prior_hashes() if prior is None else prior)
    return {
        "count": len(values),
        "digest": _sha256_bytes(
            json.dumps(values, separators=(",", ":")).encode()),
        "development_probe_seeds": list(DEVELOPMENT_PROBE_SEEDS),
    }


def _band(stage: str, n: int, *, exclude: set[str]):
    spec = assign_band(
        family=f"yr115-{stage}",
        cells={"A": y5.CELL_A, "B": y5.CELL_B},
        n=n,
        generate=_generate,
        exclude=exclude,
        start_seed=BAND_START[stage],
    )
    rep = independence_report(spec, forbidden={"기열람": exclude})
    if not rep["ok"]:
        raise RuntimeError(f"실현 독립성 위반: {rep}")
    return spec, rep


def build_manifest() -> dict:
    _require_clean()
    if MANIFEST.exists():
        raise RuntimeError("manifest를 덮어쓸 수 없다")
    prior = sorted(_prior_hashes())
    pilot, independence = _band("pilot", PILOT_N, exclude=set(prior))
    result = {
        "schema": "yr115-prereg-v1",
        "status": "RESULTS_UNSEEN_FROZEN",
        "arms": {"adopted_threshold": BASE, "notransfer_threshold": "inf"},
        "benefit": "NOTRANSFER_MINUS_ADOPTED",
        "co_primary": list(METRICS),
        "delta_assumed": DELTA,
        "planning_effect": PLAN_EFFECT,
        "pilot_n": PILOT_N,
        "power": {
            "endpoint_power": ENDPOINT_POWER,
            "sd_upper_confidence": SD_CONF,
            "n_rule": "max(24, total@3.70 필요n, A2O@0.82 필요n)",
        },
        "decision_rule": "두 주지표 95% CI 하한 모두 >0",
        "prior_hashes": _prior_summary(set(prior)),
        "pilot_band": pilot.freeze_json(),
        "pilot_independence": independence,
        "source_contract": _source_snapshot(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def _manifest() -> dict:
    manifest = _require_head_artifact(MANIFEST)
    if (
        manifest.get("schema") != "yr115-prereg-v1"
        or manifest.get("status") != "RESULTS_UNSEEN_FROZEN"
        or manifest.get("arms")
        != {"adopted_threshold": BASE, "notransfer_threshold": "inf"}
        or manifest.get("benefit") != "NOTRANSFER_MINUS_ADOPTED"
        or tuple(manifest.get("co_primary", ())) != METRICS
        or manifest.get("delta_assumed") != DELTA
        or manifest.get("planning_effect") != PLAN_EFFECT
        or manifest.get("pilot_n") != PILOT_N
        or manifest.get("power") != {
            "endpoint_power": ENDPOINT_POWER,
            "sd_upper_confidence": SD_CONF,
            "n_rule": "max(24, total@3.70 필요n, A2O@0.82 필요n)",
        }
        or manifest.get("decision_rule") != "두 주지표 95% CI 하한 모두 >0"
    ):
        raise RuntimeError("manifest 상수·스키마가 실행 계약과 다르다")
    if manifest.get("prior_hashes") != _prior_summary():
        raise RuntimeError("기열람 실현 집합이 manifest 동결값과 다르다")
    _require_source_contract(manifest)
    return manifest


def _canonical(log: list[dict]) -> list[dict]:
    return y105b._canonical_trace(log)


def run_pair(index: int, stage: str, seeds: dict[str, int]) -> dict:
    _activate_contract()
    logs: dict[str, list[dict]] = {"adopted": [], "notransfer": []}
    adopted = y5.run_arm(
        index, stage, vessel_guard=False, seeds=seeds,
        gap_threshold=BASE, log=logs["adopted"])
    none = y5.run_arm(
        index, stage, vessel_guard=False, seeds=seeds,
        gap_threshold=NO_TRANSFER, log=logs["notransfer"])
    if none["n_moved"] != 0:
        raise AssertionError("NOTRANSFER arm에서 이송 발생")
    if adopted["n_jobs"] != none["n_jobs"]:
        raise AssertionError(
            f"arm별 작업 원장 수 불일치: {adopted['n_jobs']} != {none['n_jobs']}")
    if adopted["n_a2o_expected"] != none["n_a2o_expected"]:
        raise AssertionError(
            "arm별 gate-in 원장 수 불일치: "
            f"{adopted['n_a2o_expected']} != {none['n_a2o_expected']}")
    traces = {}
    for key, arm in (("adopted", adopted), ("notransfer", none)):
        canonical = _canonical(logs[key])
        moved = [record for record in canonical if record.get("transferred")]
        rejected = [record for record in canonical if record.get("rejected")]
        if len(moved) != arm["n_moved"] or len(rejected) != arm["n_rejected"]:
            raise AssertionError(f"{key}: trace count 불일치")
        traces[key] = {
            "action_digest": _sha256_bytes(
                json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode())[:16],
            "transfer_events": moved,
            "rejected_events": rejected,
        }
    return {
        "seed": index, "seed_A": seeds["A"], "seed_B": seeds["B"],
        "adopted": adopted, "notransfer": none, "traces": traces,
    }


def _metric(arm: dict, metric: str) -> float:
    if metric == "total":
        value = arm.get("total_raw", arm.get("total"))
    else:
        value = arm.get("a2o_mean_min_raw", arm.get("a2o_mean_min"))
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"{metric} 누락·비유한 값")
    return float(value)


def _guard(rows: list[dict]):
    rep = check_guards([
        {"compl": row[arm]["compl"], "backlog": row[arm]["backlog"]}
        for row in rows for arm in ("adopted", "notransfer")
    ])
    exceptions = sum(
        row[arm]["policy_exceptions"]
        for row in rows for arm in ("adopted", "notransfer"))
    if exceptions:
        rep.ok = False
        rep.failures.append(f"정책 예외 {exceptions}건")
    distinct = 0
    total_moved = 0
    for index, row in enumerate(rows):
        counts = set()
        expected_a2o = set()
        job_counts = set()
        for arm in ("adopted", "notransfer"):
            try:
                _metric(row[arm], "total")
                _metric(row[arm], "a2o_min")
            except ValueError as exc:
                rep.ok = False
                rep.failures.append(f"row{index}/{arm}: {exc}")
            counts.add(int(row[arm].get("n_a2o", 0)))
            expected_a2o.add(int(row[arm].get("n_a2o_expected", 0)))
            job_counts.add(int(row[arm].get("n_jobs", 0)))
            if abs(row[arm]["total"] - row[arm]["chan"]["total"]) > 0.02:
                rep.ok = False
                rep.failures.append(f"row{index}/{arm}: 채널합 불일치")
        if counts == {0} or len(counts) != 1:
            rep.ok = False
            rep.failures.append(f"row{index}: A→O 표본수 불일치 {counts}")
        if expected_a2o == {0} or len(expected_a2o) != 1 or counts != expected_a2o:
            rep.ok = False
            rep.failures.append(
                f"row{index}: A→O 완료 수 {counts} != gate-in 원장 수 {expected_a2o}")
        if job_counts == {0} or len(job_counts) != 1:
            rep.ok = False
            rep.failures.append(f"row{index}: 작업 원장 수 불일치·0 {job_counts}")
        if row["notransfer"].get("n_moved") != 0:
            rep.ok = False
            rep.failures.append(f"row{index}: NOTRANSFER 이송 발생")
        total_moved += int(row["adopted"].get("n_moved", 0))
        distinct += (
            row["traces"]["adopted"]["action_digest"]
            != row["traces"]["notransfer"]["action_digest"])
    if rows and total_moved == 0:
        rep.ok = False
        rep.failures.append("ADOPTED arm에서 실제 이송 0건 — 조작 미발화")
    if rows and distinct == 0:
        rep.ok = False
        rep.failures.append("ADOPTED와 NOTRANSFER 행동이 전부 동일")
    return rep


def _run_rows(stage: str, n: int, *, exclude: set[str],
              expected_band: dict, reveal: bool) -> tuple[list[dict], dict]:
    spec, independence = _band(stage, n, exclude=exclude)
    if spec.freeze_json() != expected_band:
        raise RuntimeError(f"{stage} 대역이 동결값과 다르다")
    rows = []
    for index in range(n):
        seeds = {"A": spec.seeds["A"][index], "B": spec.seeds["B"][index]}
        row = run_pair(index, stage, seeds)
        rows.append(row)
        if reveal:
            print(
                f"[{stage} {index+1}/{n}] "
                f"BASE T{_metric(row['adopted'],'total'):.2f}/"
                f"A{_metric(row['adopted'],'a2o_min'):.2f}/mv{row['adopted']['n_moved']} "
                f"NONE T{_metric(row['notransfer'],'total'):.2f}/"
                f"A{_metric(row['notransfer'],'a2o_min'):.2f}",
                flush=True)
        else:
            print(f"[{stage} {index+1}/{n}] 봉인 분산 표본 수집", flush=True)
    return rows, {"band": spec.freeze_json(), "independence": independence}


def _diffs(rows: list[dict], metric: str) -> list[float]:
    return [
        _metric(row["notransfer"], metric) - _metric(row["adopted"], metric)
        for row in rows
    ]


def _mde(sd: float, n: int) -> float:
    return (t_ppf(0.975, n - 1) + t_ppf(ENDPOINT_POWER, n - 1)) * sd / n ** 0.5


def run_pilot() -> dict:
    _require_clean()
    manifest = _manifest()
    rows, band = _run_rows(
        "pilot", PILOT_N,
        exclude=_prior_hashes(),
        expected_band=manifest["pilot_band"],
        reveal=False,
    )
    guards = _guard(rows)
    if not guards.ok:
        raise RuntimeError(
            "blind pilot guard 실패 — power note·확증 대역을 만들지 않는다: "
            + "; ".join(guards.failures))
    power = {}
    needs = []
    for metric in METRICS:
        p = paired(_diffs(rows, metric), delta_interest=DELTA[metric], sd_conf=SD_CONF)
        need = required_n(
            p.sd, PLAN_EFFECT[metric], power=ENDPOINT_POWER,
            sd_conf=SD_CONF, sd_df=p.n - 1)
        if need is None:
            raise RuntimeError(f"{metric}: 필요 표본수 계산 실패")
        upper = sd_upper_conf(p.sd, p.n - 1, SD_CONF)
        power[metric] = {
            "pilot_n": p.n,
            "pilot_sd": p.sd,
            "pilot_sd_upper80": upper,
            "planning_effect": PLAN_EFFECT[metric],
            "conservative_n_power90": need,
        }
        needs.append(need)
    n = max(24, *needs)
    for metric, item in power.items():
        item["planned_mde90"] = _mde(item["pilot_sd_upper80"], n)
        if item["planned_mde90"] > PLAN_EFFECT[metric] + 1e-9:
            raise RuntimeError(f"{metric}: 계획 MDE90 초과")
    confirm, independence = _band(
        "confirm", n, exclude=_prior_hashes() | _hashes_from_band(manifest["pilot_band"]))
    result = {
        "repro": repro_stamp(
            experiment="YR-115 공동지표 blind pilot",
            seeds=band["band"]["seeds"],
            params={"cell_A": y5._params(y5.CELL_A), "cell_B": y5._params(y5.CELL_B)},
            profile_id=build_calibrated_profile().terminal_id,
            prereg="평균 봉인; total/A→O SD만 열어 계획효과 3.70/0.82의 n 산출",
        ),
        "schema": "yr115-power-v2",
        "status": "PILOT_GUARDS_PASSED_PLAN_FROZEN",
        "manifest_sha256": _sha256(MANIFEST),
        "source_contract_digest": manifest["source_contract"]["digest"],
        "power": power,
        "frozen_plan": {
            "n_confirm": n,
            "confirm_band": confirm.freeze_json(),
            "confirm_independence": independence,
        },
        "guards": {"ok": guards.ok, "failures": guards.failures},
        "sealed": "pilot 평균·CI·raw row는 저장·출력하지 않음",
    }
    POWER_NOTE.write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"확증 표본 동결: n={n}", flush=True)
    return result


def _classification(joint: dict, guards_ok: bool, power_ok: bool) -> str:
    if not guards_ok:
        return "INVALID"
    if not power_ok:
        return "POWER_FAIL"
    lo = {
        metric: joint[metric].get("ci_raw", joint[metric]["ci"])[0]
        for metric in METRICS
    }
    hi = {
        metric: joint[metric].get("ci_raw", joint[metric]["ci"])[1]
        for metric in METRICS
    }
    if ((lo["total"] > 0 and hi["a2o_min"] < 0)
            or (lo["a2o_min"] > 0 and hi["total"] < 0)):
        return "TRADEOFF_FAIL"
    if any(hi[metric] < 0 for metric in METRICS):
        return "HARMFUL"
    if all(lo[metric] > DELTA[metric] for metric in METRICS):
        return "JOINT_PRACTICAL_IMPROVEMENT"
    if all(lo[metric] > 0 for metric in METRICS):
        return "JOINT_CONFIRMED_SMALL"
    if all(joint[metric].get("equivalent") for metric in METRICS):
        return "EQUIVALENT"
    return "INCONCLUSIVE"


def _validated_power_note(manifest: dict, power: dict) -> int:
    """pilot 산출물을 믿지 않고 상수·n·대역을 다시 계산해 확증 진입을 결정한다."""
    if (
        power.get("schema") != "yr115-power-v2"
        or power.get("status") != "PILOT_GUARDS_PASSED_PLAN_FROZEN"
        or power.get("manifest_sha256") != _sha256(MANIFEST)
        or power.get("source_contract_digest") != manifest["source_contract"]["digest"]
        or power.get("guards", {}).get("ok") is not True
        or power.get("guards", {}).get("failures") != []
    ):
        raise RuntimeError("pilot 상태·guard·상위 계약이 확증 조건을 만족하지 않는다")
    if set(power.get("power", {})) != set(METRICS):
        raise RuntimeError("pilot power 지표 집합이 다르다")
    needs = []
    for metric in METRICS:
        item = power["power"][metric]
        if (
            item.get("pilot_n") != PILOT_N
            or float(item.get("planning_effect", math.nan)) != PLAN_EFFECT[metric]
        ):
            raise RuntimeError(f"{metric}: pilot 고정 상수 불일치")
        sd = float(item.get("pilot_sd", math.nan))
        upper = sd_upper_conf(sd, PILOT_N - 1, SD_CONF)
        need = required_n(
            sd, PLAN_EFFECT[metric], power=ENDPOINT_POWER,
            sd_conf=SD_CONF, sd_df=PILOT_N - 1)
        if (
            not math.isfinite(sd)
            or sd <= 0
            or need is None
            or abs(float(item.get("pilot_sd_upper80", math.nan)) - upper) > 2e-9
            or int(item.get("conservative_n_power90", -1)) != need
        ):
            raise RuntimeError(f"{metric}: pilot SD·필요 n 재계산 불일치")
        needs.append(need)
    n = max(24, *needs)
    frozen = power.get("frozen_plan", {})
    if int(frozen.get("n_confirm", -1)) != n:
        raise RuntimeError("확증 n이 pilot SD 재계산값과 다르다")
    for metric in METRICS:
        expected_mde = _mde(
            float(power["power"][metric]["pilot_sd_upper80"]), n)
        if (
            abs(float(power["power"][metric].get("planned_mde90", math.nan))
                - expected_mde) > 2e-9
            or expected_mde > PLAN_EFFECT[metric] + 1e-9
        ):
            raise RuntimeError(f"{metric}: 확증 MDE90 계약 불일치")
    exclude = _prior_hashes() | _hashes_from_band(manifest["pilot_band"])
    expected, independence = _band("confirm", n, exclude=exclude)
    if (
        frozen.get("confirm_band") != expected.freeze_json()
        or frozen.get("confirm_independence") != independence
    ):
        raise RuntimeError("확증 대역이 pilot 뒤 재계산값과 다르다")
    return n


def run_confirm() -> dict:
    _require_clean()
    manifest = _manifest()
    power = _require_head_artifact(POWER_NOTE)
    n = _validated_power_note(manifest, power)
    rows, band = _run_rows(
        "confirm", n,
        exclude=_prior_hashes() | _hashes_from_band(manifest["pilot_band"]),
        expected_band=power["frozen_plan"]["confirm_band"],
        reveal=True,
    )
    guards = _guard(rows)
    joint = judge_primary(
        [row["notransfer"] for row in rows],
        [row["adopted"] for row in rows],
        metrics=METRICS,
        metric_keys=METRIC_KEYS,
        delta=DELTA,
        sd_conf=SD_CONF,
    )
    power_ok = True
    for metric in METRICS:
        p = paired(
            _diffs(rows, metric), delta_interest=DELTA[metric], sd_conf=SD_CONF)
        # evalkit 기본 MDE80은 이 실험의 검정력 계약이 아니다. 혼동을 막기 위해
        # 공동 주지표 결과에는 endpoint power 0.90의 MDE만 남긴다.
        for key in ("mde80", "required_n_for_mean", "required_n_for_delta"):
            joint[metric].pop(key, None)
        mde90 = _mde(p.sd, p.n)
        joint[metric]["ci_raw"] = [p.ci_lo, p.ci_hi]
        joint[metric]["mde90"] = round(mde90, 6)
        joint[metric]["planning_effect"] = PLAN_EFFECT[metric]
        joint[metric]["power_contract"] = "endpoint power 0.90; joint lower bound 0.80"
        joint[metric]["statistical_pass"] = p.ci_lo > 0
        joint[metric]["operational_delta_pass"] = p.ci_lo > DELTA[metric]
        if mde90 > PLAN_EFFECT[metric] + 1e-12:
            power_ok = False
    joint_pass = all(joint[metric]["ci_raw"][0] > 0 for metric in METRICS)
    operational_pass = all(
        joint[metric]["ci_raw"][0] > DELTA[metric] for metric in METRICS)
    joint["joint_and_pass"] = joint["joint_and_pass_raw"] = joint_pass
    joint["joint_operational_delta_pass"] = operational_pass
    joint["joint_operational_delta_pass_raw"] = operational_pass
    diagnostics = paired_by_channel(
        [row["notransfer"]["chan"] for row in rows],
        [row["adopted"]["chan"] for row in rows],
        delta_interest=DIAG_DELTA,
        sd_conf=SD_CONF,
    )
    for value in diagnostics.values():
        value["role"] = "진단(양수=현행 이송 이득; 다중비교 보정 없음)"
    outcome_diagnostics = {}
    for name, key in (("b2c_mean_min", "b2c_mean_min"),
                      ("berth_min", "berth_min")):
        item = paired([
            float(row["notransfer"][key]) - float(row["adopted"][key])
            for row in rows
        ]).as_dict()
        item["role"] = "진단(양수=현행 이송 이득; 다중비교 보정 없음)"
        outcome_diagnostics[name] = item
    verdict = _classification(joint, guards.ok, power_ok)
    result = {
        "repro": repro_stamp(
            experiment="YR-115 현행 0.10 이송 공동 순효과 확증",
            seeds=band["band"]["seeds"],
            params={"cell_A": y5._params(y5.CELL_A), "cell_B": y5._params(y5.CELL_B)},
            profile_id=build_calibrated_profile().terminal_id,
            prereg="B=NOTRANSFER-ADOPTED; total AND A→O 95% CI 하한>0",
        ),
        "schema": "yr115-confirm-v1",
        "contract_artifacts": {
            "manifest_sha256": _sha256(MANIFEST),
            "power_note_sha256": _sha256(POWER_NOTE),
            "source_contract_digest": manifest["source_contract"]["digest"],
            "command": (
                "PYTHONPATH=src python -m "
                "yard_rl.experiments.yr115_transfer_joint_confirm --stage confirm"
            ),
        },
        "co_primary": joint,
        "diagnostic_channels": diagnostics,
        "diagnostic_outcomes": outcome_diagnostics,
        "truck_channel_bias_cap": round(
            fmean(row["adopted"]["n_moved"] for row in rows) * y5.ROUTE_S / 3600.0, 6),
        "power_ok": power_ok,
        "verdict": verdict,
        "verdict_valid": guards.ok and power_ok,
        "guards": {"ok": guards.ok, "failures": guards.failures},
        "band": band,
        "mean_moved": round(
            fmean(row["adopted"]["n_moved"] for row in rows), 6),
        "different_action_seed_count": sum(
            row["traces"]["adopted"]["action_digest"]
            != row["traces"]["notransfer"]["action_digest"] for row in rows),
        "rows": rows,
    }
    RESULT.write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"YR-115 판정: {verdict}", flush=True)
    for metric in METRICS:
        value = joint[metric]
        print(f"  {metric}: {value['mean']:+.3f} CI {value['ci']}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("manifest", "pilot", "confirm"))
    args = parser.parse_args()
    _activate_contract()
    if args.stage == "manifest":
        result = build_manifest()
    elif args.stage == "pilot":
        result = run_pilot()
    else:
        result = run_confirm()
    valid = result.get("verdict_valid", result.get("guards", {}).get("ok", True))
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
