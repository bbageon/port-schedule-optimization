"""YR-062 계약 테스트 — BC warm-start 는 가중치를 승계하고 optimizer 는 새로 시작."""
import pytest

torch = pytest.importorskip("torch")

from yard_rl.experiments.yr062_bc_finetune import (BC_CHECKPOINT, Yr062Config,
                                                   warm_start)
from yard_rl.integrated.dqn_learner import LearnerConfig

CKPT = BC_CHECKPOINT  # 저장소에 커밋된 YR-061 phase-3 산출물 (원자료 재사용)


def test_warm_start_inherits_weights_resets_training_state():
    payload = torch.load(CKPT, map_location="cpu", weights_only=False)
    learner = warm_start(CKPT, LearnerConfig(variant="ddqn", lr=1e-4, cost_scale=2.0))
    for k, v in payload["online"].items():
        assert torch.equal(learner.online.state_dict()[k], v)
        assert torch.equal(learner.target.state_dict()[k], v)   # target=online=BC
    assert learner.grad_steps == 0                              # 학습 상태는 초기화
    assert learner.cfg.lr == 1e-4                               # 새 optimizer lr 적용
    assert len(learner.replay) == 0


def test_warm_start_rejects_wrong_format(tmp_path):
    bad = tmp_path / "bad.pt"
    torch.save({"format": "other"}, bad)
    with pytest.raises(ValueError):
        warm_start(str(bad), LearnerConfig())


def test_yr062_config_defaults():
    cfg = Yr062Config()
    assert cfg.finetune_lrs == (1e-4, 3e-4, 1e-3)
    assert cfg.base.test_seed0 == 620_000       # YR-061 과 동일 test 대역 (paired 비교)
