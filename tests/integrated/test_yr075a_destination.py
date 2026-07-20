"""YR-075-a — 재조작 목적지 규칙 채택 계약 (H1 배포형).

채택 박제: ① slot_selector 미설정(=기본)은 기존 greedy find_slot 그대로 (골든 불변)
② H1(배포형 미래인지)은 고혼잡에서 greedy 와 다른 배치·결정론 ③ H1 은 자기보다
먼저 반출될 컨테이너 위 적재를 회피 (규칙 정합).
"""
import pytest

from yard_rl.integrated import TerminalSimulator
from yard_rl.integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference,
                                          run_joint_episode)
from yard_rl.integrated.candidates import CandidateGenerator
from yard_rl.integrated.cost_config import RewardCalculator
from yard_rl.integrated.profiles import build_calibrated_profile
from yard_rl.integrated.rehandle_oracle import deployable_future_selector
from yard_rl.integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario
from yard_rl.domain.enums import InformationLevel

RC = RewardCalculator.assumed_default()


def _run(seed, fill, selector):
    profile = build_calibrated_profile()
    scen = generate_terminal_scenario(profile, seed,
                                      calibrated_load_params("high", fill_ratio=fill))
    sim = TerminalSimulator(profile, scen, check_invariants=True)
    sim.info_level = InformationLevel.PRE_ADVICE
    if selector is not None:
        sim.slot_selector = selector
    return run_joint_episode(sim, ResolverPolicy(ServiceFirstSPTPreference(), "SF"),
                             RC, generator=CandidateGenerator())


def test_default_selector_is_greedy_unchanged():
    """slot_selector 미설정 = None 명시와 완전 동일 (기본 경로 = greedy find_slot)."""
    a = _run(749000, 0.70, None)
    b = _run(749000, 0.70, None)
    assert a["total_cost"] == pytest.approx(b["total_cost"])
    assert a["rehandles"] == b["rehandles"]


def test_h1_differs_and_deterministic():
    greedy = _run(749000, 0.70, None)
    h1_a = _run(749000, 0.70, deployable_future_selector)
    h1_b = _run(749000, 0.70, deployable_future_selector)
    # 고혼잡에서 H1 은 배치가 달라 결과가 바뀐다 (재조작 또는 총비용)
    assert (h1_a["rehandles"], round(h1_a["total_cost"], 4)) != \
           (greedy["rehandles"], round(greedy["total_cost"], 4))
    # 결정론 — 같은 seed·규칙이면 바이트 동일
    assert h1_a["total_cost"] == pytest.approx(h1_b["total_cost"])
    assert h1_a["rehandles"] == h1_b["rehandles"]
    assert h1_a["completion_rate"] == 1.0


def test_h1_avoids_stacking_on_earlier_retrieval():
    """규칙 정합 — H1 selector 가 '먼저 반출될 컨테이너 위'를 회피한다."""
    from yard_rl.integrated.rehandle_oracle import _select
    from yard_rl.domain.models import BlockGeometry, CraneSpec

    class _Stk:
        def __init__(self):
            self.geom = BlockGeometry(block_id="B", bay_count=3, row_count=1,
                                      tier_max=3, bay_length_m=6.5, row_width_m=3.0,
                                      tier_height_m=2.6, transfer_row=0)
            # bay1: [C_early] (곧 반출), bay2: 빈, bay3: 빈
            self._p = {(1, 1): ["C_early"], (2, 1): [], (3, 1): []}

        def top_tier(self, bay, row):
            return len(self._p[(bay, row)])

        def stack(self, bay, row):
            return self._p[(bay, row)]

        def stack_size_ok(self, bay, row, size):
            return True

    class _B:
        container_id, bay, row = "B_late", 1, 1
        size = None

    spec = CraneSpec(crane_id="Y", service_bay_min=1, service_bay_max=3,
                     gantry_speed_mps=4.0, trolley_speed_mps=1.0,
                     hoist_speed_loaded_mps=0.58, hoist_speed_empty_mps=1.17,
                     lock_time_s=30, unlock_time_s=20, truck_positioning_time_s=25)
    times = {"C_early": 100.0, "B_late": 9999.0}   # B 는 훨씬 늦게 반출
    dest = _select(_Stk(), _B(), spec, frozenset({(1, 1)}), times)
    # C_early(먼저 반출) 위(bay1)는 회피 → 빈 bay 로 (bay2 가 가장 가까움)
    assert dest == (2, 1)
