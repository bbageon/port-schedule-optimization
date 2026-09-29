"""Market isolation and faithful original input are required for causal comparison."""
from dataclasses import asdict, replace

import pytest
import torch

from yard_rl.v6.ppo.model import BlockPolicy
from yard_rl.v6.ppo.original_comparison import OriginalComparisonRuntime, assignment_audit
from yard_rl.v6.schema import Order


def runtime(reallocate, intervention_s=0):
    torch.manual_seed(17)
    rt = OriginalComparisonRuntime(BlockPolicy(), reallocate=reallocate,
                                   seed=19, intervention_s=intervention_s)
    rt.time_s = 0.
    return rt


def seller_rows():
    x = torch.zeros((4, 21))
    x[0, 12] = 1.
    return x


def test_market_draws_do_not_shift_crane_decisions_or_probabilities():
    original, active = runtime(False), runtime(True)
    rows = torch.arange(4 * 24, dtype=torch.float32).reshape(4, 24) / 100
    a, b = [], []
    for i in range(30):
        original.select('seller', 'b', i, seller_rows())
        active.select('seller', 'b', i, seller_rows())
        active.select('buyer', 'b', i, torch.zeros((2, 20)))
        a.append(original.select('crane', 'b', i, rows))
        b.append(active.select('crane', 'b', i, rows))
    assert a == b
    assert torch.equal(original.role_rng['crane'].get_state(), active.role_rng['crane'].get_state())
    assert not original.training and not active.training
    assert all(torch.equal(v, active.policy.state_dict()[k]) for k, v in original.policy.state_dict().items())


def test_both_arms_keep_warmup_and_original_always_preserves_market():
    original, active = runtime(False, 60), runtime(True, 60)
    for rt in (original, active):
        assert rt.select('seller', 'b', 59., seller_rows()) == 0
        assert rt.select('buyer', 'b', 59., torch.zeros((2, 20))) == 1
        assert dict(rt.forced_market) == dict(seller=1, buyer=1)
    original.select('seller', 'b', 60., seller_rows())
    active.select('seller', 'b', 60., seller_rows())
    assert original.forced_market['seller'] == 2
    assert active.forced_market['seller'] == 1


def test_original_control_cannot_silently_select_a_nonkeep_or_masked_action():
    rt = runtime(False)
    wrong = seller_rows()
    wrong[0, 12] = 0
    with pytest.raises(RuntimeError, match='KEEP encoding'):
        rt.select('seller', 'b', 0., wrong)
    with pytest.raises(RuntimeError, match='unavailable'):
        rt.select('seller', 'b', 0., seller_rows(), torch.tensor([False, True, True, True]))


def test_assignment_check_finds_time_location_and_identity_changes():
    order = Order('job', 1, 0., 120., 'Y01', 'container')
    raw = [asdict(order)]
    assert assignment_audit(raw, {'job': order})['original_fields_preserved']
    changed = replace(order, con_loc='Y02', in_out_reserve_s=180.)
    audit = assignment_audit(raw, {'job': changed})
    assert audit['changed_fields'] == dict(in_out_reserve_s=1, con_loc=1)
    with pytest.raises(RuntimeError, match='membership'):
        assignment_audit(raw, {})
