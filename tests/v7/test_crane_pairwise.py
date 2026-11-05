"""Pairwise target invariance, complete-pair validation, and fixed update budget."""
import copy

import pytest
import torch

from yard_rl.v7.crane.fit import CraneTrainer
from yard_rl.v7.crane.policy import CraneNet
from yard_rl.v7.crane.teacher import CraneLabelCollector
from yard_rl.v7.crane.rollout import CranePool


def labels():
    c = CraneLabelCollector()
    for i in range(24):
        a = [i/24, 1., 0., 0., 0., 0., 0., 0., 0.]
        b = [(24-i)/24, 0., 0., 0., 0., 0., 0., 0., 0.]
        c.add(picked_row=a, alt_row=b, phi_factual=i*1500., phi_alt=(24-i)*1500.,
              crane='A', job=str(i))
    return c.result()


def test_pair_loss_ignores_shared_target_offset_and_keeps_update_budget():
    torch.manual_seed(9913501)
    base = CraneNet()
    a, b = copy.deepcopy(base), copy.deepcopy(base)
    data = labels()
    shifted = copy.deepcopy(data)
    # Powers of two keep the exact target differences representable.
    for sample in data.samples:
        sample.target = round(sample.target*8)/8
    for dst, src in zip(shifted.samples, data.samples):
        dst.target = src.target+8
    ar = CraneTrainer(a,loss_kind='pairwise').fit(data,seed=23)
    br = CraneTrainer(b,loss_kind='pairwise').fit(shifted,seed=23)
    assert ar.steps == br.steps == 50
    for x, y in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(x,y,rtol=0,atol=0)
    pr = CraneTrainer(copy.deepcopy(base)).fit(data,seed=23)
    assert (ar.n,ar.n_val,ar.steps) == (pr.n,pr.n_val,pr.steps)
    assert ar.val_nonzero_pairs > 0


def test_incomplete_decision_pair_is_rejected():
    d = labels()
    d.samples.pop()
    with pytest.raises(ValueError,match='complete'):
        CraneTrainer(CraneNet(),loss_kind='pairwise').fit(d)


def test_worker_completion_order_cannot_change_label_order():
    p = CranePool(None,horizon_s=3)
    p._done = [dict(key=(1.,'B','A'),t=1.,crane='A',worlds=2),
               dict(key=(1.,'A','A'),t=1.,crane='A',worlds=2)]
    assert [r['key'] for r in p.results()] == [(1.,'A','A'),(1.,'B','A')]
