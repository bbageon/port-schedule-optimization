"""YR-142 계약 고정 — 견고 파싱·중복 결속 조합 제외·대역 동결 파일."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from yard_rl.integrated.baselines import _feasible_joint
from yard_rl.integrated.candidates import prepo_bound_jid


def test_prepo_bound_jid_parsing():
    assert prepo_bound_jid("PREPO:J-OUT-000:12") == "J-OUT-000"
    # 다중 블록 'A:작업' — jid 안의 콜론에도 양끝 파싱으로 안전 (18차 감사)
    assert prepo_bound_jid("PREPO:A:J-OUT-000:7") == "A:J-OUT-000"
    assert prepo_bound_jid("REPO:YC1:7") is None
    assert prepo_bound_jid("") is None
    assert prepo_bound_jid(None) is None


class _NoDryRun:
    def dry_run_commit(self, refs):
        raise AssertionError("중복 결속은 dry_run 이전에 걸러져야 한다")


def _cand(job_id):
    return SimpleNamespace(job_ref=SimpleNamespace(job_id=job_id, token=None))


def test_feasible_joint_rejects_same_bound_job():
    sim = _NoDryRun()
    assign = {"YC1": _cand("PREPO:J-OUT-000:5"), "YC2": _cand("PREPO:J-OUT-000:9")}
    assert _feasible_joint(sim, assign) is False
    assert getattr(sim, "_prepo_dup_removed", 0) == 1


def test_band_file_frozen_integrity():
    p = Path("outputs/reports/yr142_prepo_enforce/band.json")
    if not p.exists():
        pytest.skip("대역 미생성")
    d = json.loads(p.read_text(encoding="utf-8"))
    seeds = [s for ss in d["seeds"].values() for s in ss]
    hashes = [h for hs in d["realization_hashes"].values() for h in hs]
    assert len(seeds) == 12 and len(hashes) == 12
    assert all(910_000 <= s < 920_000 for s in seeds)      # 미사용 정수 구간
    assert len(set(hashes)) == 12                           # 내부 중복 0
    assert d["independence"]["ok"] is True
