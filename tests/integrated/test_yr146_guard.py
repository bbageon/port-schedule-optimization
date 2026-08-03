"""YR-146 계약 고정 — 안전장치 불간섭(개입 0 이면 OFF 와 결과 동일)·계수 배선."""
from __future__ import annotations

from yard_rl.experiments.yr090_dense_vessel import BASE
from yard_rl.experiments.yr146_deploy_guard import _episode_guard


def test_guard_passthrough_when_no_intervention():
    """개입 0 인 판에서 ON 은 OFF 와 완전 동일해야 한다 (불간섭 계약)."""
    cell, seed, arm, ts = "mid-tight", BASE["mid-tight"] + 16, "c1", 221_000
    off = _episode_guard(cell, seed, arm, ts, guard_on=False)
    on = _episode_guard(cell, seed, arm, ts, guard_on=True)
    g = on.pop("guard")
    assert g["dec"] > 0 and g["joint"] >= 0                  # 계수 배선 확인
    if g["iv_reopen"] + g["iv_stagnant"] + g["iv_escape"] == 0:
        assert on == off                                     # 바이트 동일 결과
