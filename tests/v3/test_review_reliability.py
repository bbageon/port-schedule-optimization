"""Prevent false before/after comparisons and reuse of unrelated monthly runs."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from yard_rl.v3.actors import BuyerNet, SellerNet
from yard_rl.v3.eval import month_judge as mj
from yard_rl.v3.eval.__main__ import _load_nets
from yard_rl.v3.eval.contracts import digest, load_arm, network_identity, runtime_identity, save_arm
from yard_rl.v3.stage.month import plan_month
from yard_rl.v3.stage.month_run import DayReport, MonthResult
from yard_rl.v3.train import month_loop as ml


@pytest.fixture
def judge_case(monkeypatch, tmp_path):
    days = plan_month(9_900_700, n_days=4)
    calls = []
    def fake(job):
        calls.append(job["_label"])
        return mj.ArmMonth(arm=job["_label"], traded=50,
            phi_by_day={d.index: (10.0 if job["arm"] == "RL" else 20.0) for d in days},
            days=[{"index": d.index} for d in days])
    monkeypatch.setattr(mj, "_run_arm", fake)
    kwargs = dict(seed=9_900_700, days=days, arms=("NO_REALLOC",), workers=1,
                  ckpt_dir=tmp_path, log=lambda *_: None)
    return kwargs, calls


def test_identical_run_resumes_and_dependent_days_do_not_claim_significance(judge_case):
    kwargs, calls = judge_case
    out = mj.judge_month(**kwargs)
    again = mj.judge_month(**kwargs)
    assert calls == ["RL", "NO_REALLOC"]
    assert again == out
    assert out["inference"]["independent_runs"] == 1
    assert not out["inference"]["claim_eligible"]
    assert out["monthly_total_krw"] == {"RL": 20.0, "NO_REALLOC": 40.0}
    assert out["arms"]["NO_REALLOC"]["sum_difference_krw"] == -20.0
    assert "p" not in out["arms"]["NO_REALLOC"]
    assert "beaten" not in next(iter(out["by_load"].values()))


@pytest.mark.parametrize("change", ["seed", "plan", "window", "trigger", "weights", "runtime", "admission"])
def test_changed_conditions_refuse_cache_before_any_new_simulation(judge_case, monkeypatch, change):
    kwargs, calls = judge_case
    mj.judge_month(**kwargs)
    if change == "seed":
        kwargs["seed"] += 1
    elif change == "plan":
        kwargs["days"] = [replace(d, load=d.load + 1) for d in kwargs["days"]]
    elif change == "window":
        kwargs["window_s"] = 3600
    elif change == "trigger":
        kwargs["trigger_k"] = {"RL": 0.5}
    elif change == "weights":
        kwargs["seller_net"] = SellerNet()
    elif change == "admission":
        kwargs["admission_mode"] = "PRESERVE"
    else:
        identity = runtime_identity()
        identity["source_and_config_sha256"] = "changed"
        monkeypatch.setattr(mj, "runtime_identity", lambda: identity)
    with pytest.raises(ValueError, match="new output directory"):
        mj.judge_month(**kwargs)
    assert len(calls) == 2


@pytest.mark.parametrize("kind", ["legacy", "missing_guard", "tampered"])
def test_missing_or_corrupt_evidence_is_not_assumed_zero(judge_case, kind):
    kwargs, calls = judge_case
    mj.judge_month(**kwargs)
    path = kwargs["ckpt_dir"] / "arm_RL.json"
    data = json.loads(path.read_text())
    if kind == "legacy":
        for field in ("contract", "contract_sha256", "result_sha256"):
            data.pop(field)
    elif kind == "missing_guard":
        data.pop("policy_exceptions")
        body = {k: v for k, v in data.items()
                if k not in {"contract", "contract_sha256", "result_sha256"}}
        data["result_sha256"] = digest(body)
    else:
        data["phi_by_day"]["1"] = 9999
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        mj.judge_month(**kwargs)
    assert len(calls) == 2


def test_conflicting_extra_policy_name_is_rejected(judge_case):
    kwargs, calls = judge_case
    kwargs["extra_policies"] = {"RL": (SellerNet(), BuyerNet())}
    with pytest.raises(ValueError, match="Duplicate"):
        mj.judge_month(**kwargs)
    assert calls == []


def test_thirty_day_integer_keys_roundtrip_without_false_corruption(tmp_path):
    contract = {"label": "RL", "seed": 9_900_700}
    path = tmp_path / "arm_RL.json"
    result = mj.ArmMonth(arm="RL", phi_by_day={i: i * 0.5 for i in range(30)})
    save_arm(path, result, contract)
    loaded = load_arm(path, contract)
    assert {int(k): v for k, v in loaded["phi_by_day"].items()} == result.phi_by_day


def test_initial_checkpoint_exists_before_simulation_and_is_reproducible(tmp_path, monkeypatch):
    saved = []
    class StopBeforeSimulation(Exception):
        pass
    def inspect_initial(**kw):
        path = saved[-1]
        s, b, _ = _load_nets(str(path / "ckpt_init.pt"), require_untrained=True)
        assert network_identity(s) == network_identity(kw["seller_net"])
        assert network_identity(b) == network_identity(kw["buyer_net"])
        assert not (path / "ckpt_000.pt").exists()
        raise StopBeforeSimulation
    monkeypatch.setattr(ml, "run_month", inspect_initial)
    identities = []
    for name, init_seed in [("a", 7), ("b", 7), ("c", 8)]:
        saved.append(tmp_path / name)
        with pytest.raises(StopBeforeSimulation):
            ml.run_month_training(seed=9_900_700, n_days=3, init_seed=init_seed,
                                  out_dir=saved[-1], log=lambda *_: None)
        identities.append(json.loads((saved[-1] / "training_manifest.json").read_text())["initial_networks"])
    assert identities[0] == identities[1]
    assert identities[0] != identities[2]
    with pytest.raises(ValueError, match="preserved"):
        ml.run_month_training(out_dir=saved[0], log=lambda *_: None)


def test_post_day_checkpoint_cannot_be_named_untrained(tmp_path):
    path = tmp_path / "ckpt_000.pt"
    torch.save({"seller": SellerNet().state_dict(), "buyer": BuyerNet().state_dict(), "it": 0}, path)
    _load_nets(str(path))  # Legacy weights remain readable as an early model.
    with pytest.raises(ValueError, match="zero-update"):
        _load_nets(str(path), require_untrained=True)


def test_actual_fit_days_and_steps_are_recorded_including_boundary_days(tmp_path, monkeypatch):
    from yard_rl.v3.train.labels import LabelSet, Sample
    def fake_collect(rows):
        dim = next(SellerNet().parameters()).shape[1]
        labels = LabelSet(seller=[Sample([0.1] * dim, float(i), str(i), "KEEP")
                                 for i in range(4)])
        return SimpleNamespace(result=lambda: labels), 0
    monkeypatch.setattr(ml, "_collect", fake_collect)
    def fake_world(**kw):
        reports = []
        for day in kw["days"]:
            fit = kw["on_fit"](day, [{}] * 4)
            rep = DayReport(index=day.index, load=day.load, label=day.label,
                            train=day.is_train, fit=fit, n_labels=4)
            kw["on_day"](rep)
            reports.append(rep)
        return MonthResult(days=reports, live=reports, plan=[])
    monkeypatch.setattr(ml, "run_month", fake_world)
    ml.run_month_training(seed=9_900_700, n_days=3, out_dir=tmp_path, log=lambda *_: None)
    manifest = json.loads((tmp_path / "training_manifest.json").read_text())
    assert manifest["measurement_days"] == [1]
    assert manifest["fit_days"] == [0, 1, 2]
    assert manifest["optimizer_steps"] == 150
    first = torch.load(tmp_path / "ckpt_000.pt", weights_only=True)
    initial = torch.load(tmp_path / "ckpt_init.pt", weights_only=True)
    assert first["metadata"]["fit_days"] == [0]
    assert first["metadata"]["optimizer_steps"] == 50
    assert any(not torch.equal(initial["seller"][k], first["seller"][k]) for k in initial["seller"])
    with pytest.raises(ValueError, match="zero-update"):
        _load_nets(str(tmp_path / "ckpt_000.pt"), require_untrained=True)


def test_preserved_demand_reaches_every_evaluation_arm_and_cache(judge_case, monkeypatch):
    kwargs, calls = judge_case
    seen = []
    original = mj._run_arm

    def capture(job):
        seen.append((job["_label"], job["admission_mode"]))
        return original(job)

    monkeypatch.setattr(mj, "_run_arm", capture)
    kwargs.update(admission_mode="PRESERVE", extra_policies={"RL_INIT": (SellerNet(), BuyerNet())})
    out = mj.judge_month(**kwargs)
    assert out["admission_mode"] == "PRESERVE"
    assert seen == [(a, "PRESERVE") for a in ("RL", "NO_REALLOC", "RL_INIT")]
    for arm, mode in seen:
        saved = json.loads((kwargs["ckpt_dir"] / f"arm_{arm}.json").read_text())
        assert saved["contract"]["settings"]["admission_mode"] == mode
    assert mj.judge_month(**kwargs) == out
    assert len(calls) == 3


@pytest.mark.parametrize("mode", ["LEGACY", "PRESERVE"])
def test_training_records_environment_in_initial_final_weights_and_results(tmp_path, monkeypatch, mode):
    def fake_world(**kw):
        assert kw["admission_mode"] == mode
        reports = []
        for day in kw["days"]:
            rep = DayReport(index=day.index, load=day.load, label=day.label, train=day.is_train)
            kw["on_day"](rep)
            reports.append(rep)
        return MonthResult(plan=[], days=reports, live=reports)

    monkeypatch.setattr(ml, "run_month", fake_world)
    ml.run_month_training(seed=9_900_700, n_days=3, out_dir=tmp_path,
                          admission_mode=mode, log=lambda *_: None)
    for name in ("training_manifest.json", "month.json"):
        assert json.loads((tmp_path / name).read_text())["admission_mode"] == mode
    for name in ("ckpt_init.pt", "ckpt_002.pt"):
        ck = torch.load(tmp_path / name, weights_only=True)
        assert ck["metadata"]["admission_mode"] == mode


def test_unknown_admission_mode_fails_before_output_creation(tmp_path):
    out = tmp_path / "invalid"
    with pytest.raises(ValueError, match="admission_mode"):
        ml.run_month_training(admission_mode="typo", out_dir=out)
    with pytest.raises(ValueError, match="admission_mode"):
        mj.judge_month(seed=9_900_700, admission_mode="typo", ckpt_dir=out)
    assert not out.exists()


@pytest.mark.parametrize("kind", ["train", "eval"])
def test_cli_passes_explicit_preserved_demand_to_entry_function(tmp_path, monkeypatch, kind):
    from yard_rl.v3.train import __main__ as train_cli
    from yard_rl.v3.eval import __main__ as eval_cli
    seen = []
    args = ["--seed", "9900700", "--days", "3", "--admission-mode", "PRESERVE", "--out", str(tmp_path)]
    if kind == "train":
        monkeypatch.setattr(train_cli, "run_month_training", lambda **kw: seen.append(kw))
        assert train_cli.main(args) == 0
    else:
        monkeypatch.setattr(eval_cli, "_load_nets", lambda *a, **kw: (None, None, "test"))
        monkeypatch.setattr(eval_cli, "judge_month", lambda **kw: seen.append(kw) or {})
        assert eval_cli.main(args) == 0
    assert len(seen) == 1 and seen[0]["admission_mode"] == "PRESERVE"
