"""YR-092 초기 스택 규격 정합 — 생성기가 실행 적재규칙(pile 동일규격)을 준수 (외부감사 결함3).

감사 실측: tier 별 독립 추첨으로 12 seed 전부 혼합규격 pile(mid/high 평균 ~75개) —
재조작 용량검사의 "blocker 동일규격" 가정 위반. 본 테스트가 생성·검증 양쪽을 고정한다.
"""
import pytest

from yard_rl.domain.enums import ContainerSize, LoadStatus
from yard_rl.domain.models import Container
from yard_rl.integrated import TerminalSimulator, build_integrated_profile
from yard_rl.integrated.scenario_gen import (TerminalGenParams, calibrated_load_params,
                                             generate_terminal_scenario)
from yard_rl.sim.constraints import ConstraintViolation

PROF = build_integrated_profile()


def _mixed_piles(containers) -> int:
    sizes: dict[tuple[int, int], set] = {}
    for c in containers.values():
        sizes.setdefault((c.bay, c.row), set()).add(c.size)
    return sum(1 for s in sizes.values() if len(s) > 1)


def test_generator_produces_uniform_piles_across_seeds():
    """감사 재현 seed 대역 포함 12 seed — 혼합규격 pile 0."""
    for seed in [310000 + i for i in range(4)] + [830000 + i for i in range(4)] \
                + [830100 + i for i in range(4)]:
        for params in (None, calibrated_load_params("mid"), calibrated_load_params("high")):
            sc = generate_terminal_scenario(PROF, seed, params)
            assert _mixed_piles(sc.containers) == 0, f"seed={seed} 혼합 pile 발생"


def test_both_sizes_still_present():
    """규격 혼합비(size_mix_ft40)가 pile 단위로 살아있다 — 한 규격으로 붕괴하지 않음."""
    sc = generate_terminal_scenario(PROF, 310000, calibrated_load_params("high"))
    sizes = {c.size for c in sc.containers.values()}
    assert sizes == {ContainerSize.FT20, ContainerSize.FT40}


def test_validator_rejects_mixed_initial_pile():
    sc = generate_terminal_scenario(PROF, 310000)
    # 인위적으로 혼합 pile 주입 — validator 가 거부해야 한다
    pile: dict[tuple[int, int], list] = {}
    for c in sc.containers.values():
        pile.setdefault((c.bay, c.row), []).append(c)
    target = next(cs for cs in pile.values() if len(cs) >= 2)
    top = max(target, key=lambda c: c.tier)
    flipped = (ContainerSize.FT20 if top.size == ContainerSize.FT40 else ContainerSize.FT40)
    sc.containers[top.container_id] = Container(
        container_id=top.container_id, size=flipped, load_status=LoadStatus.FULL,
        block=top.block, bay=top.bay, row=top.row, tier=top.tier)
    with pytest.raises(ConstraintViolation, match="PILE_SIZE_MIX"):
        TerminalSimulator(PROF, sc)


def test_rehandle_capacity_assumption_now_holds():
    """용량검사의 "blocker 동일규격" 가정이 초기상태에서 실제 성립."""
    sim = TerminalSimulator(PROF, generate_terminal_scenario(
        PROF, 830100, calibrated_load_params("high")))
    for (bay, row), pile in sim.stacks._stacks.items():
        if len(pile) >= 2:
            sizes = {sim.stacks.containers[cid].size for cid in pile}
            assert len(sizes) == 1, f"({bay},{row}) 혼합 pile"
