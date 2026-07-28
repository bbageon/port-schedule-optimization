from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from yard_rl.experiments import yr115_transfer_joint_confirm as y115
from yard_rl.integrated.evalkit import GuardReport, required_n
from yard_rl.integrated.statfuncs import sd_upper_conf


def test_prior_hashes_include_yr105b_pilot_and_select():
    prior = y115._prior_hashes()
    assert y115.y105b.pilot_hashes() <= prior
    select_n = int(json.loads(
        y115.y105b.POWER_NOTE.read_text(encoding="utf-8"))
        ["frozen_sample_plan"]["n_select"])
    assert y115.y105b.select_hashes(select_n) <= prior
    invalid = json.loads(
        y115.INVALID_MANIFEST_V1.read_text(encoding="utf-8"))
    assert y115._hashes_from_band(invalid["pilot_band"]) <= prior


def test_new_pilot_and_confirm_bands_are_disjoint_from_prior():
    prior = y115._prior_hashes()
    pilot, pilot_report = y115._band("pilot", 2, exclude=prior)
    confirm, confirm_report = y115._band(
        "confirm", 2, exclude=prior | pilot.all_hashes)
    assert pilot_report["ok"] and confirm_report["ok"]
    assert not (pilot.all_hashes & prior)
    assert not (confirm.all_hashes & prior)
    assert not (pilot.all_hashes & confirm.all_hashes)
    used_seeds = {
        seed for values in pilot.seeds.values() for seed in values
    } | {
        seed for values in confirm.seeds.values() for seed in values
    }
    assert not (used_seeds & set(y115.DEVELOPMENT_PROBE_SEEDS))


def test_source_contract_covers_runtime_trees_and_profile():
    paths = set(y115._contract_paths())
    assert "src/yard_rl/integrated/baselines.py" in paths
    assert "src/yard_rl/integrated/time_contract.py" in paths
    assert "src/yard_rl/io/profile_loader.py" in paths
    assert "src/yard_rl/sim/stack.py" in paths
    assert "src/yard_rl/contract/cost.py" in paths
    assert "src/yard_rl/envs/direct_job_env.py" in paths
    assert "src/yard_rl/policies/baselines.py" in paths
    assert "configs/terminals/dgt_armg.yaml" in paths
    assert "src/yard_rl/experiments/yr113_transfer_net_effect.py" in paths
    assert "outputs/reports/yr105b_transfer_threshold/power_note.json" in paths


def test_pair_contract_has_zero_notransfer_and_full_precision():
    row = y115.run_pair(0, "test", {"A": 970_000, "B": 970_001})
    assert row["notransfer"]["n_moved"] == 0
    assert row["adopted"]["a2o_mean_min_raw"] is not None
    assert row["notransfer"]["a2o_mean_min_raw"] is not None
    assert row["adopted"]["n_a2o"] == row["notransfer"]["n_a2o"]
    assert row["adopted"]["n_a2o_expected"] == row["notransfer"]["n_a2o_expected"]
    assert row["adopted"]["n_a2o_completed"] == row["adopted"]["n_a2o_expected"]
    assert row["notransfer"]["n_a2o_completed"] == row["notransfer"]["n_a2o_expected"]
    assert row["adopted"]["n_a2o_censored"] == 0
    assert row["notransfer"]["n_a2o_censored"] == 0
    assert row["adopted"]["n_jobs"] == row["notransfer"]["n_jobs"]


@pytest.mark.parametrize(
    ("total_ci", "a2o_ci", "power_ok", "expected"),
    [
        ((10.1, 20.0), (1.1, 2.0), True, "JOINT_PRACTICAL_IMPROVEMENT"),
        ((0.1, 5.0), (0.1, 0.8), True, "JOINT_CONFIRMED_SMALL"),
        ((0.1, 5.0), (-1.0, -0.1), True, "TRADEOFF_FAIL"),
        ((-2.0, 2.0), (-0.2, 0.2), True, "INCONCLUSIVE"),
        ((0.1, 5.0), (0.1, 0.8), False, "POWER_FAIL"),
    ],
)
def test_joint_classification(total_ci, a2o_ci, power_ok, expected):
    joint = {
        "total": {"ci": list(total_ci), "equivalent": False},
        "a2o_min": {"ci": list(a2o_ci), "equivalent": False},
    }
    assert y115._classification(joint, True, power_ok) == expected
    assert y115._classification(joint, False, power_ok) == "INVALID"


def test_joint_classification_prefers_raw_ci_at_rounding_boundary():
    joint = {
        "total": {
            "ci": [0.0, 1.0], "ci_raw": [-1e-8, 1.0], "equivalent": False},
        "a2o_min": {
            "ci": [0.1, 1.0], "ci_raw": [0.1, 1.0], "equivalent": False},
    }
    assert y115._classification(joint, True, True) == "INCONCLUSIVE"


def test_guard_fails_if_a2o_missing_or_notransfer_moves(monkeypatch):
    arm = {
        "compl": 1.0, "backlog": 0, "policy_exceptions": 0,
        "total": 1.0, "total_raw": 1.0,
        "a2o_mean_min_raw": None, "n_a2o": 0,
        "chan": {"truck": 1.0, "vessel": 0.0, "move": 0.0,
                 "other": 0.0, "total": 1.0},
    }
    row = {
        "adopted": {**arm, "n_moved": 1},
        "notransfer": {**arm, "n_moved": 1},
        "traces": {
            "adopted": {"action_digest": "a"},
            "notransfer": {"action_digest": "b"},
        },
    }
    # run_pair가 이송 0을 선검사하므로 guard에는 A→O 누락만 별도로 검증한다.
    guard = y115._guard([row])
    assert not guard.ok
    assert any("A→O" in failure for failure in guard.failures)


def _valid_arm(*, moved: int, n_jobs: int = 2) -> dict:
    return {
        "compl": 1.0, "backlog": 0, "policy_exceptions": 0,
        "total": 1.0, "total_raw": 1.0,
        "a2o_mean_min_raw": 1.0, "n_a2o": 2, "n_a2o_expected": 2,
        "n_a2o_completed": 2, "n_a2o_censored": 0,
        "n_jobs": n_jobs,
        "n_moved": moved,
        "chan": {"truck": 1.0, "vessel": 0.0, "move": 0.0,
                 "other": 0.0, "total": 1.0},
    }


def _row(*, moved: int = 1, adopted_jobs: int = 2, none_jobs: int = 2) -> dict:
    return {
        "adopted": _valid_arm(moved=moved, n_jobs=adopted_jobs),
        "notransfer": _valid_arm(moved=0, n_jobs=none_jobs),
        "traces": {
            "adopted": {"action_digest": "a"},
            "notransfer": {"action_digest": "b"},
        },
    }


def test_guard_requires_transfer_manipulation_and_same_job_ledger():
    no_transfer = y115._guard([_row(moved=0)])
    assert not no_transfer.ok
    assert any("조작 미발화" in failure for failure in no_transfer.failures)

    mismatch = y115._guard([_row(adopted_jobs=3, none_jobs=2)])
    assert not mismatch.ok
    assert any("작업 원장 수" in failure for failure in mismatch.failures)


def test_guard_rejects_censored_a2o():
    row = _row()
    row["adopted"]["n_a2o_completed"] = 1
    row["adopted"]["n_a2o_censored"] = 1
    guard = y115._guard([row])
    assert not guard.ok
    assert any("실제 gate-out" in failure for failure in guard.failures)


def test_failed_pilot_guard_cannot_write_power_note(monkeypatch, tmp_path):
    monkeypatch.setattr(y115, "_require_clean", lambda: None)
    monkeypatch.setattr(y115, "_manifest", lambda: {"pilot_band": {}})
    monkeypatch.setattr(y115, "_prior_hashes", lambda: set())
    monkeypatch.setattr(
        y115, "_run_rows",
        lambda *args, **kwargs: ([], {"band": {"seeds": {}}}),
    )
    monkeypatch.setattr(
        y115, "_guard", lambda rows: GuardReport(False, ["의도한 실패"]))
    note = tmp_path / "power.json"
    monkeypatch.setattr(y115, "POWER_NOTE", note)
    with pytest.raises(RuntimeError, match="power note"):
        y115.run_pilot()
    assert not note.exists()


def _valid_power_note(monkeypatch) -> tuple[dict, dict, int]:
    manifest = {
        "source_contract": {"digest": "source"},
        "pilot_band": {"realization_hashes": {"A": [], "B": []}},
    }
    power = {
        "schema": "yr115-power-v3",
        "status": "PILOT_GUARDS_PASSED_PLAN_FROZEN",
        "manifest_sha256": "manifest",
        "source_contract_digest": "source",
        "guards": {"ok": True, "failures": []},
        "power": {},
    }
    needs = []
    for metric, sd in (("total", 4.0), ("a2o_min", 0.8)):
        upper = sd_upper_conf(sd, y115.PILOT_N - 1, y115.SD_CONF)
        need = required_n(
            sd, y115.PLAN_EFFECT[metric], power=y115.ENDPOINT_POWER,
            sd_conf=y115.SD_CONF, sd_df=y115.PILOT_N - 1)
        power["power"][metric] = {
            "pilot_n": y115.PILOT_N,
            "pilot_sd": sd,
            "pilot_sd_upper80": upper,
            "planning_effect": y115.PLAN_EFFECT[metric],
            "conservative_n_power90": need,
        }
        needs.append(need)
    n = max(24, *needs)
    for metric in y115.METRICS:
        power["power"][metric]["planned_mde90"] = y115._mde(
            power["power"][metric]["pilot_sd_upper80"], n)
    frozen_band = {"frozen": True}
    independence = {"ok": True}
    power["frozen_plan"] = {
        "n_confirm": n,
        "confirm_band": frozen_band,
        "confirm_independence": independence,
    }
    monkeypatch.setattr(y115, "_sha256", lambda path: "manifest")
    monkeypatch.setattr(y115, "_prior_hashes", lambda: set())
    monkeypatch.setattr(y115, "_hashes_from_band", lambda band: set())
    monkeypatch.setattr(
        y115, "_band",
        lambda *args, **kwargs: (
            SimpleNamespace(freeze_json=lambda: frozen_band), independence),
    )
    return manifest, power, n


def test_power_note_is_recomputed_before_confirm(monkeypatch):
    manifest, power, n = _valid_power_note(monkeypatch)
    assert y115._validated_power_note(manifest, power) == n

    power["frozen_plan"]["n_confirm"] = n + 1
    with pytest.raises(RuntimeError, match="확증 n"):
        y115._validated_power_note(manifest, power)


def test_power_note_with_failed_guard_is_rejected(monkeypatch):
    manifest, power, _ = _valid_power_note(monkeypatch)
    power["guards"]["ok"] = False
    with pytest.raises(RuntimeError, match="pilot 상태"):
        y115._validated_power_note(manifest, power)
