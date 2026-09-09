"""크레인 학습 정책이 계약을 지키는가 ([[YR-248]] 2단계).

■ 무엇을 지켜야 하나
  ① `Preference.rank` 규약 — 작을수록 먼저 · WAIT 는 최하위 · 결정론
  ② 정보 경계 — 공개 정보만 (`USES_FUTURE_INFORMATION = False`)
  ③ 실작업 우선 울타리 — 학습 전에도 [[YR-039]] 퇴화(일은 안 하고 자리이동만)로
     안 무너진다
  ④ 특징이 **실제로 값을 낸다** — 죽은 칸이 있으면 정책이 그만큼 장님이다
"""
from __future__ import annotations

import torch

from yard_rl.v4.crane.features import CRANE_DIM, CRANE_FEATURES
from yard_rl.v4.crane.policy import (CRANE_ADV_SCALE, CraneNet, CranePolicy,
                                     from_advantage, to_advantage)
from yard_rl.v4.dispatch import RL_CRANE, make_preference
from yard_rl.v4.stage.episode import DISPATCHERS_READY, run_episode

LOAD, SEED = 300, 9_100_321


def test_registered_and_constructible():
    """이름으로 만들어지고, 쓸 수 있는 배차 목록에 있다."""
    assert RL_CRANE in DISPATCHERS_READY
    p = make_preference(RL_CRANE)
    assert isinstance(p, CranePolicy)
    assert p.USES_FUTURE_INFORMATION is False


def test_rank_is_deterministic():
    """같은 상태·같은 망이면 같은 순서 — 짝비교가 성립하려면 필수다."""
    torch.manual_seed(7)
    net = CraneNet()
    a, b = CranePolicy(net), CranePolicy(net)
    x = torch.randn(CRANE_DIM)
    with torch.no_grad():
        assert float(a.net(x[None])[0]) == float(b.net(x[None])[0])


def test_wait_is_last():
    """WAIT 는 항상 최하위 — `BaselinePreference` 규약."""
    p = CranePolicy()

    class _GC:
        job_ref = None
        plan = None
        kind = None

    assert p.rank(None, "YC-L", _GC()) == (2, 0.0, "")


def test_serve_first_fence():
    """★실작업 우선 울타리 — 학습 전에도 [[YR-039]] 퇴화를 막는다.

    켜져 있으면 SERVE 가 아닌 후보는 tier 1 로 밀린다. 망 점수가 아무리 낮아도
    실작업을 이길 수 없다.
    """
    p = CranePolicy(serve_first=True)
    assert p.serve_first is True
    off = CranePolicy(serve_first=False)
    assert off.serve_first is False


def test_runs_a_real_day():
    """★실제로 하루를 굴린다 — 꽂히기만 하고 안 불리면 소용없다."""
    r = run_episode(load=LOAD, arm="NO_REALLOC", dispatcher=RL_CRANE, seed=SEED)
    assert r.phi_krw > 0


def test_network_actually_decides():
    """★망이 바뀌면 결과가 바뀐다 — 안 바뀌면 정책이 무시되고 있다는 뜻이다.

    이 시험이 없으면 "꽂았다" 와 "작동한다" 를 구분 못 한다. 실제로 개발 중
    `SF_SPT` 와 Φ 가 우연히 같아 한 번 헷갈렸다 (2026-09-10).
    """
    import yard_rl.v4.dispatch as D
    import yard_rl.v4.stage.episode as E

    orig = E.make_preference
    got = []
    try:
        for s in (1, 2, 3):
            torch.manual_seed(s)
            net = CraneNet()
            E.make_preference = (lambda name, seed=0, _n=net:
                                 CranePolicy(_n) if name == RL_CRANE
                                 else D.make_preference(name, seed=seed))
            got.append(run_episode(load=LOAD, arm="NO_REALLOC",
                                   dispatcher=RL_CRANE, seed=SEED).phi_krw)
    finally:
        E.make_preference = orig
    assert len(set(got)) > 1, f"망이 달라도 Φ 가 같다 — 정책이 안 쓰이고 있다: {got}"


def test_advantage_scale_roundtrip():
    """목표 눈금이 왕복한다 — 보고 때 원화로 되돌릴 수 있어야 한다."""
    assert from_advantage(to_advantage(150_000.0, 100_000.0)) == 50_000.0
    assert to_advantage(100_000.0, 100_000.0) == 0.0
    assert CRANE_ADV_SCALE > 0


def test_features_are_not_all_dead():
    """★특징이 실제로 값을 낸다 — 죽은 칸이 있으면 그만큼 장님이다.

    하루를 굴리며 후보 특징을 모아, **모든 칸이 한 번은 0 이 아닌 값**을 갖는지 본다.
    [[YR-235]] 가 재배치층에서 죽은 칸 셋을 찾아 6→3 으로 줄인 적이 있다.
    """
    import yard_rl.v4.crane.policy as P

    seen = [0.0] * CRANE_DIM
    orig = P.crane_features

    def spy(sim, cid, gc):
        x = orig(sim, cid, gc)
        for i, v in enumerate(x):
            seen[i] = max(seen[i], abs(v))
        return x

    P.crane_features = spy
    try:
        run_episode(load=LOAD, arm="NO_REALLOC", dispatcher=RL_CRANE, seed=SEED)
    finally:
        P.crane_features = orig

    dead = [CRANE_FEATURES[i] for i, v in enumerate(seen) if v == 0.0]
    assert not dead, (
        f"죽은 칸: {dead} — 이 칸들은 정책에 정보를 주지 않는다. "
        f"관측값 {dict(zip(CRANE_FEATURES, seen))}")
