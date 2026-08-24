"""★rudder 러너가 **v3 그 자체**를 굴리는가 ([[YR-223]]).

rudder 는 v3 의 `run_episode` 를 못 쓰고 자기 러너를 쓴다(epoch 궤적이 필요해서).
그러면 "조립이 조금 다른 세계" 를 잴 위험이 생긴다 — 여기서 막는다.

같은 시드·같은 부하에서 **Φ·거래 수·결정 수가 전부 같아야** 한다. 하나라도
어긋나면 RUDDER 가 재는 것은 v3 정책의 궤적이 아니다.
"""
from __future__ import annotations

import pytest

from yard_rl.rudder.runner import run_day_phi
from yard_rl.v3.reward import reset_rollout_calls, rollout_calls
from yard_rl.v3.stage import run_episode

LOAD, SEED = 400, 9_900_811


@pytest.mark.parametrize("arm_seed", [9_900_811, 9_900_812])
def test_day_matches_v3_episode(arm_seed):
    reset_rollout_calls()
    ref = run_episode(load=LOAD, arm="RL", seed=arm_seed).as_dict()
    got = run_day_phi(load=LOAD, seed=arm_seed)
    for k in ("phi_krw", "traded_edges", "n_space", "n_time", "txn_failed",
              "policy_exceptions"):
        assert got[k] == pytest.approx(ref[k]), (
            f"{k}: rudder {got[k]} vs v3 {ref[k]} — 조립이 다르다")
    assert got["decisions"] == ref["n_decisions"]


def test_day_run_makes_no_rollouts():
    """창 없이 하루만 굴리면 반사실이 **한 번도** 안 돌아간다."""
    reset_rollout_calls()
    run_day_phi(load=LOAD, seed=SEED)
    assert rollout_calls() == 0


def test_windows_are_counted_as_rollouts():
    """★창은 반사실이므로 v3 계수기에 잡혀야 한다 — 판정 가드가 거짓말하지 않게."""
    from yard_rl.rudder.runner import build_ctx, run_window
    reset_rollout_calls()
    ctx, mbt, orders, records, _b, _o = build_ctx(load=200, seed=SEED)
    run_window(ctx, mbt=mbt, orders=orders, records=records, decided=set(),
               t0=3600.0, horizon_s=900.0, seed=SEED, load=200)
    assert rollout_calls() == 2, "창 짝 = 세계 2개"
