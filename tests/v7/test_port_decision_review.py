import gzip
import json

import pytest

from scripts.v7.review_port_decisions import audit_cases, changes_in_time, json_file, render


def test_early_and_late_effects_use_only_common_cohort_after_decision():
    base = {"early": (0, 150), "late": (2500, 4000)}
    branch = {"early": (0, 210), "late": (2500, 4120)}
    assert changes_in_time(base, branch, 100) == (1, 3)
    with pytest.raises(ValueError, match="before the intervention"):
        changes_in_time(base, branch, 200)
    with pytest.raises(ValueError, match="cohorts"):
        changes_in_time(base, {}, 100)


def test_job_choice_and_port_call_effect_can_have_different_signs(tmp_path):
    def archive(name, truck_end, stream_end):
        folder = tmp_path / name
        json_file(folder / "result.json", {"windows": {"all": {
            "trucks": 1, "ships": 1, "unfinished_trucks": 0, "unfinished_ships": 0}}})
        json_file(folder / "ships.json", [{"ship": "ship", "start": 0, "finish": 3000,
                                          "stream_finishes": [stream_end, 3000]}])
        with gzip.open(folder / "trucks.jsonl.gz", "wt", encoding="utf8") as stream:
            stream.write(json.dumps({"job": "truck", "gate": 0, "finish": truck_end}))
    archive("model", 1600, 2000)
    archive("day-3", 1500, 2200)
    path = tmp_path / "manifest.gz"
    with gzip.open(path, "wt", encoding="utf8") as stream:
        json.dump({"vessels_by_day": {"0": [{"key": k, "ship": "ship"} for k in ("v", "other")]}}, stream)
    rows = {"3": {"disagreement": True, "event": {"time": 1000},
                  "rule_minus_model": dict(truck_minutes=-100/60, ship_hours=0, weighted_minutes=-100/60)},
            "10": {"disagreement": False}}
    snapshot = {3: [{"crane": "A", "model": dict(kind="SERVE", job="load", stream="v", buffer=0, quay_blocked=True),
                     "rule": dict(kind="SERVE", job="truck")}]}
    cases = audit_cases(tmp_path, {"days": rows}, snapshot, path)
    assert cases["3"]["changes"][0]["model"]["stream_delta_min"] == pytest.approx(200/60)
    assert cases["3"]["changes"][0]["model"]["call_delta_min"] == 0
    assert cases["10"] == {"type": "선택 일치"}
    report = render({}, cases, smoke=True)
    assert "트럭 우선이 유리한 사례 1건" in report
    assert "해당 트럭이 절약한 시간은 +1.667분" in report
