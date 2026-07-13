import json

from yard_rl.experiments.direct_job_report import (
    REPORT_NAME,
    build_direct_job_report,
    write_direct_job_payload,
)


def _payload(*, quick=False):
    return {
        "manifest": {
            "mode": "quick" if quick else "full",
            "quick_run": quick,
            "scenario": {"n_external": 100, "n_vessel": 0},
        },
        "selection": {
            "SLA_OFF": {"cost_q": {"p": 0.8, "checkpoint_episode": 1000},
                        "baseline": {"policy": "FIFO", "validation_mean_wait_min": 12.0}},
            "SLA_ON": {"cost_q": {"p": 1.0, "checkpoint_episode": 950},
                       "baseline": {"policy": "MIN_BLOCKER", "validation_mean_wait_min": 11.5}},
        },
        "test_rows": [
            {"arm": "SLA_OFF", "policy": "Cost-Q", "mean_wait_min": 10.0,
             "p50_wait_min": 8.0, "p95_wait_min": 24.0, "sla_over_rate": 0.12,
             "completion_rate": 1.0, "backlog": 0, "fallback_rate": 0.0},
            {"arm": "SLA_OFF", "policy": "FIFO", "mean_wait_min": 12.0,
             "p50_wait_min": 9.0, "p95_wait_min": 25.0, "sla_rate": 0.15,
             "completion_rate": 1.0, "backlog": 0},
            {"arm": "SLA_ON", "policy": "Cost-Q", "metrics": {
                "mean_wait_min": 10.5, "p50_wait_min": 8.1, "p95_wait_min": 23.5,
                "sla_exceed_count": 10, "n_config": 100, "completed_external": 100,
                "backlog": 0, "fallback_decisions": 4, "n_decisions": 100}},
        ],
        "paired_statistics": {"SLA_OFF": {
            "baseline": "FIFO", "alternative": "CostQ+GreedyFallback",
            "mean_wait": {"estimate": -2.0, "ci_low": -2.8, "ci_high": -1.2},
            "p95_percent_change": {"estimate": -4.0, "ci_low": -7.0, "ci_high": 1.5},
        }},
        "summary": {},
        "acceptance": {"quick": quick, "primary_arm": "SLA_OFF",
                       "coverage_class": "pure", "overall": "PASS"},
    }


def test_build_full_report_contains_frozen_design_and_statistics(tmp_path):
    path = build_direct_job_report(_payload(), tmp_path)
    text = path.read_text(encoding="utf-8")

    assert path.name == REPORT_NAME
    assert "외부트럭-only" in text and "n_vessel=0" in text
    assert "BLOCK_ENTRY" in text
    assert "`SLA_OFF`가 primary" in text and "`SLA_ON`이 secondary" in text
    assert "selected baseline `FIFO`" in text
    assert "Cost-Q `p=0.8`, checkpoint `1000`" in text
    assert "-2.00 min" in text and "[-2.80, -1.20]" in text
    assert "-4.00%" in text and "[-7.00%, +1.50%]" in text
    assert "pure Cost-Q" in text
    assert "hybrid Cost-Q + fallback" in text
    assert "Full run 판정: `PASS`" in text
    assert "기준별 결과" in text
    assert len(text.splitlines()) <= 200


def test_quick_report_is_defensive_and_forbids_acceptance(tmp_path):
    payload = {"manifest": {"run_mode": "quick", "n_vessel": 0},
               "selection": {}, "test": {"rows": [
                   {"sla_arm": "SLA_OFF", "policy_name": "CostQ",
                    "fallback_rate": 8.0, "completed_external": 9,
                    "configured_external_jobs": 10}
               ]}, "summary": {"verdict": "PASS"}}
    text = build_direct_job_report(payload, tmp_path).read_text(encoding="utf-8")

    assert "합격/불합격 판정을 금지" in text
    assert "Full run 판정: `PASS`" not in text
    assert "8.0% — coverage insufficient" in text
    assert "—" in text
    assert "assumed 프로파일 + 합성 시나리오" in text


def test_write_payload_roundtrip(tmp_path):
    payload = _payload()
    path = write_direct_job_payload(payload, tmp_path)
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_report_caps_large_policy_matrix_below_200_lines(tmp_path):
    payload = {"test_rows": [
        {"arm": "SLA_OFF", "policy": f"Cost-Q-{i}", "fallback_rate": 0.0}
        for i in range(80)
    ]}
    text = build_direct_job_report(payload, tmp_path).read_text(encoding="utf-8")
    assert "30개 정책 행을 생략" in text
    assert len(text.splitlines()) <= 200
