"""YR-012-b — replay buffer·target network 계약 테스트."""
import json

import pytest

torch = pytest.importorskip("torch")

from yard_rl.experiments.residual_delta_stable import (StableExpConfig,  # noqa: E402
                                                       quick_stable_config,
                                                       run_stable_experiment)
from yard_rl.policies.residual_delta_net import (DeltaNetConfig,  # noqa: E402
                                                 FeatureScaler,
                                                 N_FEATURES,
                                                 ResidualDeltaNetAgent)

PROFILE = "configs/terminals/hjnc_armg.yaml"


def _scaler() -> FeatureScaler:
    return FeatureScaler.fit([[float(i + r) for i in range(N_FEATURES)]
                              for r in range(4)])


class _Cand:
    def __init__(self, jid, service, prior, g=None, f=None):
        self.job_id = jid
        self.transfer_direction = "YARD_TO_TRUCK"
        self.wait_s = 10.0
        self.reach_s = 5.0
        self.estimated_service_s = service
        self.blocker_count = 0
        self.block_entry_s = 0.0
        self.prior_cost = prior
        self.global_raw = g or (0.1, 0.5, 3.0, 100.0, 0.0)
        self.future_raw = f or (2.0, 400.0, 0.5, 3.0)
        self.feature = ("YARD_TO_TRUCK", 0, 0, 0, 0)


def _stab_cfg(**kw) -> DeltaNetConfig:
    base = dict(replay_capacity=100, batch_size=8, min_replay=10,
                target_sync_every=5)
    base.update(kw)
    return DeltaNetConfig(**base)


def test_config_validation_and_default_preserves_yr012():
    assert not DeltaNetConfig().stabilized  # 기본값 = YR-012 online TD 그대로
    assert _stab_cfg().stabilized
    with pytest.raises(ValueError):
        DeltaNetConfig(replay_capacity=100, min_replay=200)  # warmup > capacity
    with pytest.raises(ValueError):
        DeltaNetConfig(target_sync_every=-1)


def test_untrained_stabilized_agent_is_still_greedy_equivalent():
    agent = ResidualDeltaNetAgent(_stab_cfg(), scaler=_scaler(), seed=0)
    a, b = _Cand("a", 100.0, 2.0), _Cand("b", 200.0, 5.0)
    assert agent.act(None, [a, b]).job_id == "a"  # zero-init → argmin prior = greedy
    assert agent.q_totals([a, b]) == pytest.approx([2.0, 5.0])


def test_replay_warmup_defers_learning_until_min_replay():
    agent = ResidualDeltaNetAgent(_stab_cfg(min_replay=10), scaler=_scaler(), seed=0)
    a, b = _Cand("a", 100.0, 2.0), _Cand("b", 200.0, 5.0)
    for _ in range(9):  # warmup 미달 — gradient 없음, Δ ≡ 0 유지
        agent.update(None, a, 1.0, None, [b], done=False)
    assert agent._grad_steps == 0
    assert agent.q_totals([a])[0] == pytest.approx(2.0)
    agent.update(None, a, 1.0, None, [b], done=False)  # 10번째 → 학습 시작
    assert agent._grad_steps == 1


def test_target_network_syncs_every_n_gradient_steps():
    agent = ResidualDeltaNetAgent(_stab_cfg(min_replay=1, target_sync_every=3),
                                  scaler=_scaler(), seed=0)
    a, b = _Cand("a", 100.0, 2.0), _Cand("b", 200.0, 5.0)

    def target_equals_online() -> bool:
        return all(torch.equal(p, q) for p, q in zip(
            agent.target_net.state_dict().values(),
            agent.net.state_dict().values()))

    agent.update(None, a, 1.0, None, [b], done=False)   # step 1
    agent.update(None, a, 2.0, None, [b], done=False)   # step 2 — 아직 미동기
    assert not target_equals_online()
    agent.update(None, a, 3.0, None, [b], done=False)   # step 3 — 동기화
    assert target_equals_online()


def test_stable_config_rejects_prior_bands():
    with pytest.raises(ValueError):
        StableExpConfig(train_seed0=140_000)  # YR-012 band 재사용 금지


def test_stable_quick_smoke(tmp_path):
    out = tmp_path / "stable"
    report = run_stable_experiment(PROFILE, str(out), quick_stable_config(),
                                   progress=lambda _msg: None)
    text = report.read_text(encoding="utf-8")
    assert "안정화" in text and "locked test" in text
    payload = json.loads((out / "delta_stable_results.json").read_text(encoding="utf-8"))
    assert any(n.startswith("DeltaNet[replay|sync") for n in payload["paired"])
    assert isinstance(payload["improved_vs_baseline"], list)
