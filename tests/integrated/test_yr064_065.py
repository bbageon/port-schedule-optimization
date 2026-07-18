"""YR-064/065 계약 테스트 — config 가드·재사용 대역 정합."""
import pytest

torch = pytest.importorskip("torch")

from yard_rl.experiments.yr064_bc_diff_finetune import (Yr064Config,
                                                        quick_yr064_config)
from yard_rl.experiments.yr065_window_ladder import (Yr065Config,
                                                     quick_yr065_config)


def test_yr064_config_defaults():
    cfg = Yr064Config()
    assert cfg.window_s == 600.0                # YR-063 동결값 승계
    assert cfg.finetune_lrs == (1e-4, 3e-4)
    assert cfg.base.test_seed0 == 620_000       # 재사용 행과 paired 전제
    assert not quick_yr064_config().reuse       # quick 은 재사용 행 없음


def test_yr065_config_defaults():
    cfg = Yr065Config()
    assert cfg.windows == (1_200.0, 2_400.0)    # 600s 는 재사용
    assert cfg.base.test_seed0 == 620_000
    assert not quick_yr065_config().reuse
