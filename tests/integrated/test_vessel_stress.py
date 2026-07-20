"""본선 스트레스 시나리오 축 계약 (2026-07-20) — ETD 조임 손잡이.

① 기본(deadline_mult 2.0)은 기존 바이트 동일 ② 조이면 본선지연↑ ③ 프리셋 정합.
"""
import hashlib

import pytest

from yard_rl.integrated import TerminalSimulator
from yard_rl.integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference,
                                          run_joint_episode)
from yard_rl.integrated.candidates import CandidateGenerator
from yard_rl.integrated.cost_config import RewardCalculator
from yard_rl.integrated.profiles import build_calibrated_profile
from yard_rl.integrated.scenario_gen import (TerminalGenParams, busan_scenario_params,
                                             generate_terminal_scenario)
from yard_rl.domain.enums import InformationLevel

RC = RewardCalculator.assumed_default()


def _fp(params, seed=754100):
    sc = generate_terminal_scenario(build_calibrated_profile(), seed, params)
    parts = [f"{j.job_id}|{j.deadline}|{j.release_time}" for j in sc.jobs]
    for v in sc.vessels:
        parts.append(f"{v.vessel_id}|{v.plan.planned_completion_s}|{v.plan.etd_s}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def test_default_mult_byte_identical():
    """deadline_mult 기본 2.0 = 미지정과 동일 지문 (골든 보존)."""
    assert _fp(TerminalGenParams()) == _fp(TerminalGenParams(vessel_deadline_mult=2.0))


def test_tighter_deadline_changes_vessel_plan():
    loose = generate_terminal_scenario(build_calibrated_profile(), 754100,
                                       TerminalGenParams(vessel_deadline_mult=2.0))
    tight = generate_terminal_scenario(build_calibrated_profile(), 754100,
                                       TerminalGenParams(vessel_deadline_mult=1.15))
    lv = next(v for v in loose.vessels if v.plan.planned_completion_s is not None)
    tv = next(v for v in tight.vessels if v.plan.planned_completion_s is not None)
    assert tv.plan.planned_completion_s < lv.plan.planned_completion_s   # 마감 조여짐
    assert tight.meta.get("vessel_deadline_mult") == pytest.approx(1.15)


def test_tighter_deadline_raises_vessel_delay():
    def vdelay(mult):
        p = build_calibrated_profile()
        sc = generate_terminal_scenario(p, 754000,
                                        busan_scenario_params("vessel_rush",
                                                              vessel_deadline_mult=mult))
        s = TerminalSimulator(p, sc, check_invariants=True)
        s.info_level = InformationLevel.PRE_ADVICE
        return run_joint_episode(s, ResolverPolicy(ServiceFirstSPTPreference(), "SF"),
                                 RC, generator=CandidateGenerator())["vessel_delay_min"]
    assert vdelay(1.1) > vdelay(2.5)          # 조일수록 본선지연 증가


def test_preset_validation():
    with pytest.raises(ValueError):
        busan_scenario_params("bogus")
    with pytest.raises(ValueError):
        TerminalGenParams(vessel_deadline_mult=0.0)
    assert busan_scenario_params("vessel_rush").vessel_deadline_mult == pytest.approx(1.15)
    assert busan_scenario_params("normal").vessel_deadline_mult == pytest.approx(2.0)
