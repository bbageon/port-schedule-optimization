"""Counterexamples protecting the meaning of descriptive saved-experience statistics."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'/'v7'))
from ppo_credit_metrics import (between_group_share, choice_metrics, first_choice, flow_id,
                                mask_description, normalized_advantages)
from yard_rl.v7.crane.actor_critic import GraphState


def features(vessel=0, store=1, wait=0, serve=1):
    row = [0.]*10
    row[0], row[1], row[2], row[9] = vessel, serve, wait, store
    return row


def test_forced_prefix_is_not_counted_as_a_free_policy_choice():
    sample = SimpleNamespace(masks=[[7], [3, 4], [9]], choices=[7, 4, 9])
    assert first_choice(sample) == (1, [3, 4], 4)
    with pytest.raises(ValueError):
        first_choice(SimpleNamespace(masks=[[7]], choices=[7]))


def test_vessel_wait_and_preparation_are_not_external_truck_saturation():
    fs = torch.tensor([features(wait=2), features(store=0, wait=1), features(vessel=1, wait=2),
                       features(vessel=1, store=0), features(serve=0, wait=2)])
    assert [flow_id(f) for f in fs] == [0, 1, 2, 3, 4]
    state = GraphState(fs, torch.arange(5), torch.zeros(5, dtype=torch.long), .5)
    desc = mask_description(SimpleNamespace(state=state), [0, 1, 2, 3, 4], 0)
    assert desc['cap_counts'] == [1, 0]
    assert desc['flow_counts'] == [1, 1, 1, 1, 1]


def test_probabilities_use_legal_mask_and_ties_have_midrank():
    fs = torch.tensor([features(), features(store=0), features(vessel=1)])
    a, probabilities = choice_metrics(torch.tensor([1., 1., 100.]), [0, 1], 0, fs)
    b, _ = choice_metrics(torch.tensor([31., 31., -100.]), [0, 1], 0, fs)
    assert a == b
    assert a['chosen_rank'] == 1.5
    assert a['top_ties'] == 2
    assert a['mass'] == [.5, .5, 0., 0., 0.]
    assert sum(probabilities) == 1.


def test_time_stratification_removes_a_pure_time_composition_effect():
    rows = [dict(day_index=day, time_bin=t, raw_advantage=float(t))
            for day, t, count in [(0, 0, 9), (0, 1, 1), (1, 0, 1), (1, 1, 9)]
            for _ in range(count)]
    assert between_group_share(rows)['share'] == pytest.approx(.64)
    assert between_group_share(rows, strata='time_bin')['share'] is None


def test_exact_day_separation_is_not_individual_action_evidence():
    rows = [dict(day_index=d, raw_advantage=float(d)*10) for d in (0, 0, 1, 1)]
    assert between_group_share(rows)['share'] == 1.


def test_normalization_preserves_original_permutation_and_signs():
    samples = [SimpleNamespace(target=y, old_value=0.) for y in [-10., -9., -2., -1.]]
    values, order, _ = normalized_advantages(samples, 12)
    original = torch.tensor([samples[i].target for i in order])
    expected = (original-original.mean())/original.std(unbiased=False)
    assert torch.equal(torch.tensor(values)[order], expected)
    assert values[0] < 0 < values[-1]
