"""YR-088 파생 — 양하 인계버퍼 역압력 (yard_handover_cap) 회귀 테스트.

기본(None)=현행 골든 바이트 동일(별도 golden 테스트가 담당). 여기선 opt-in ON 의 계약:
파이프라인이 cap 을 넘지 않고 실제 사용되며, cap 을 조일수록 STS 역압력이 커진다.
"""
from __future__ import annotations

from statistics import mean

from yard_rl.domain.enums import InformationLevel
from yard_rl.integrated import TerminalSimulator
from yard_rl.integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference,
                                          run_joint_episode)
from yard_rl.integrated.candidates import CandidateGenerator
from yard_rl.integrated.cost_config import RewardCalculator
from yard_rl.integrated.profiles import build_calibrated_profile
from yard_rl.integrated.scenario_gen import (calibrated_load_params,
                                             generate_terminal_scenario)

RC = RewardCalculator.numeraire_v1()
PROF = build_calibrated_profile()
SEEDS = range(820000, 820004)


class _Traced(TerminalSimulator):
    """양하 파이프라인 피크를 기록 (불변식 검사용)."""
    peak = 0

    def _sts_move(self, vid):
        super()._sts_move(vid)
        if self._discharge_pipeline:
            self.peak = max(self.peak, max(self._discharge_pipeline.values()))


def _sim(seed, cap, cls=TerminalSimulator):
    sim = cls(PROF, generate_terminal_scenario(PROF, seed, calibrated_load_params("high")),
              check_invariants=True, yard_handover_cap=cap)
    sim.info_level = InformationLevel.PRE_ADVICE
    return sim


def _run(sim):
    return run_joint_episode(sim, ResolverPolicy(ServiceFirstSPTPreference(), "SF"),
                             RC, generator=CandidateGenerator())


def test_off_pipeline_unused():
    """cap=None(기본) → 파이프라인 dict 는 끝까지 빈다 (미사용·골든 불변 경로)."""
    sim = _sim(820000, None)
    _run(sim)
    assert sim._discharge_pipeline == {}


def test_pipeline_bounded_and_used():
    """ON → 파이프라인 피크가 0<peak<=cap (넘지 않고 실제 발동)."""
    used = False
    for seed in SEEDS:
        sim = _sim(seed, 2, cls=_Traced)
        _run(sim)
        assert sim.peak <= 2, f"seed {seed}: peak {sim.peak} > cap 2"
        used = used or sim.peak >= 1
    assert used, "어느 seed 에서도 파이프라인이 안 쓰임 — 역압력 미발동"


def test_tighter_cap_more_backpressure():
    """cap 을 조일수록(1<100) STS 대기(역압력)가 크거나 같다 — 단조."""
    def avg_sts_wait(cap):
        return mean(_run(_sim(s, cap))["sts_wait_s"] for s in SEEDS)
    tight = avg_sts_wait(1)
    loose = avg_sts_wait(100)      # cap 이 아주 크면 사실상 역압력 없음(≈OFF)
    assert tight >= loose, f"tight {tight} < loose {loose} — 역압력 단조성 위반"


def test_on_runs_valid():
    """ON 에서도 에피소드가 정상 완료(유효 metric)."""
    r = _run(_sim(820000, 2))
    assert 0.0 <= r["completion_rate"] <= 1.0
    assert r["berth_overrun_min"] >= 0.0
