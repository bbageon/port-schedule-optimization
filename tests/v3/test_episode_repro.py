"""같은 시드가 같은 하루를 내는가 — 짝비교의 전제.

판정이 **날 단위 짝비교**라, 같은 시드로 두 번 돌렸을 때 Φ 가 다르면 두 팔의
격차가 정책 차이인지 잡음인지 구분할 수 없다. 실제로 한 번 밟았다 —
학습 전 팔의 망 초기화에 시드를 안 걸어 같은 셀이 102,913,359 ↔ 136,388,850 로
갈렸다(2026-08-22). 그 뒤로 이 검사가 지킨다.

하루 물량을 400대로 줄였다 — 등식 검사라 규모가 크지 않아도 된다.
"""
from __future__ import annotations

import pytest

from yard_rl.v3.stage import run_episode

LOAD = 400
SEED = 9_900_778          # 비판정(진단) 대역


def _summary(r):
    d = r.as_dict()
    return {k: d[k] for k in ("phi_krw", "c_wait", "c_move", "c_rehandle",
                              "c_vessel", "admitted", "traded_edges",
                              "n_space", "n_time", "mean_turn_time_s")}


@pytest.mark.parametrize("arm", ["NO_REALLOC", "RL"])
def test_same_seed_same_day(arm):
    a = _summary(run_episode(load=LOAD, arm=arm, seed=SEED))
    b = _summary(run_episode(load=LOAD, arm=arm, seed=SEED))
    assert a == b, f"{arm}: 같은 시드가 다른 하루를 냈다\n  {a}\n  {b}"


def test_unimplemented_arm_fails_loudly():
    """안 만든 팔은 **조용히 대체되지 않는다** — 대조표가 거짓이 되면 안 된다."""
    with pytest.raises(NotImplementedError, match="YR-211"):
        run_episode(load=LOAD, arm="FCFS", seed=SEED)
    with pytest.raises(NotImplementedError, match="YR-213"):
        run_episode(load=LOAD, arm="RL", dispatcher="ROLLOUT_GREEDY", seed=SEED)


def test_eval_path_never_calls_the_teacher():
    """판정 경로에서 반사실 rollout 이 **한 번도** 안 불린다 — 06 하드가드."""
    from yard_rl.v3.reward import reset_rollout_calls, rollout_calls

    reset_rollout_calls()
    r = run_episode(load=LOAD, arm="RL", seed=SEED)      # budget 없음 = 교사 미부착
    assert rollout_calls() == 0, f"교사 누출 {rollout_calls()}회"
    assert r.rollout_worlds == 0
