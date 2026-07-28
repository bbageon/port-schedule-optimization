"""YR-117 — 사후 지표 재분석이 YR-113 사전등록을 덮지 않는지 검사."""
import json
from pathlib import Path


ROOT = Path(__file__).parents[2]
YR113_OUT = ROOT / "outputs" / "reports" / "yr113_transfer_net_effect"
YR117_OUT = ROOT / "outputs" / "reports" / "yr117_metric_reanalysis"


def test_yr113_stored_results_preserve_original_prereg_and_mark_posthoc():
    for band in ("select", "confirm"):
        result = json.loads((YR113_OUT / f"results_{band}.json").read_text(encoding="utf-8"))
        assert "1차 채널 truck" in result["repro"]["prereg"]
        assert "주판정 = total + a2o_min" not in result["repro"]["prereg"]
        assert result["repro"]["original_prereg_commit"].startswith("73ef07b")
        assert result["repro"]["yr117_analysis_status"] == (
            "retrospective_post_hoc_not_confirmatory"
        )
        assert result["repro"]["yr117_confirmatory"] is False
        assert result["primary"]["confirmatory"] is False
        assert all(
            "사후 재분석" in result["primary"][metric]["role"]
            for metric in ("total", "a2o_min")
        )


def test_yr117_reanalysis_is_joint_and_but_not_prospective_confirmation():
    result = json.loads((YR117_OUT / "reanalysis.json").read_text(encoding="utf-8"))
    assert result["analysis_status"] == "retrospective_post_hoc_not_confirmatory"
    assert result["confirmatory"] is False
    assert result["contract_examined"]["decision_rule"] == "AND"
    assert result["results"]["select"]["joint_and_pass"] is False
    assert result["results"]["confirm"]["joint_and_pass"] is True
    assert result["results"]["two_band_joint_replication"] is False
