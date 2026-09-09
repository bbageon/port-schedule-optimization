"""Copied CF trainers must seed model creation, not just the subsequent world."""
import pytest
import torch

from yard_rl.v5.stage.month import DayPlan
from yard_rl.v5.train import loop, month_loop


class CapturedModels(Exception):
    pass


def initial_weights(kind, seed, ambient_seed, monkeypatch, tmp_path):
    torch.manual_seed(ambient_seed)
    captured = []

    def stop_before_world(**kwargs):
        for key in ("seller_net", "buyer_net"):
            captured.append(torch.cat([p.detach().flatten().clone()
                                       for p in kwargs[key].parameters()]))
        raise CapturedModels

    with pytest.raises(CapturedModels):
        if kind == "daily":
            monkeypatch.setattr(loop, "run_episode", stop_before_world)
            loop.run_training(iters=1, seed_base=seed, out_dir=tmp_path, log=lambda _: None)
        else:
            monkeypatch.setattr(month_loop, "run_month", stop_before_world)
            day = DayPlan(0, 300, "seed-debug", seed + 1000, 0, 1)
            month_loop.run_month_training(seed=seed, days=[day], out_dir=tmp_path,
                                           log=lambda _: None)
    assert len(captured) == 2
    assert not list(tmp_path.iterdir())  # No physical run or checkpoint was started.
    return captured


@pytest.mark.parametrize("kind", ["daily", "month"])
def test_training_seed_overrides_ambient_rng(kind, monkeypatch, tmp_path):
    a = initial_weights(kind, 9900303, 1, monkeypatch, tmp_path)
    b = initial_weights(kind, 9900303, 2, monkeypatch, tmp_path)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("kind", ["daily", "month"])
def test_different_training_seeds_change_both_initial_models(kind, monkeypatch, tmp_path):
    a = initial_weights(kind, 9900303, 1, monkeypatch, tmp_path)
    b = initial_weights(kind, 9900304, 1, monkeypatch, tmp_path)
    assert all(not torch.equal(x, y) for x, y in zip(a, b))
