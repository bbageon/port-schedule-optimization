"""YR-147 3단계 — 순위 라벨·손실 계약 고정 (23차: 사전식·전용 난수·PPO 비의존·방향)."""
from __future__ import annotations

import random
from types import SimpleNamespace

import torch

from yard_rl.experiments.yr147_defer import (
    EPS_COST, _lex_pref, _rank_rng, rank_pair_loss, select_progress_combos)


def test_lex_pref_order():
    # 미완 비율이 우선 — 비용이 아무리 좋아도 못 뒤집음 (22차 ④)
    assert _lex_pref((0.1, 2, 5.0), (0.0, 0, 999.0)) == "PROG"
    assert _lex_pref((0.0, 0, 5.0), (0.1, 2, 0.0)) == "WAIT"
    # 미완 동률 → backlog
    assert _lex_pref((0.0, 3, 5.0), (0.0, 1, 999.0)) == "PROG"
    # 앞 두 항 동률 → 비용, 동점폭 EPS_COST 이내는 None
    assert _lex_pref((0.0, 0, 5.0), (0.0, 0, 5.0 + EPS_COST)) is None
    assert _lex_pref((0.0, 0, 5.0), (0.0, 0, 4.0)) == "PROG"
    assert _lex_pref((0.0, 0, 4.0), (0.0, 0, 5.0)) == "WAIT"


def test_rank_rng_dedicated_stream():
    r1 = _rank_rng(88000, 3, 2, 17)
    r2 = _rank_rng(88000, 3, 2, 17)
    assert [r1.random() for _ in range(5)] == [r2.random() for _ in range(5)]  # 재현
    assert _rank_rng(88000, 3, 2, 18).random() != _rank_rng(88000, 3, 2, 17).random()
    main = random.Random(88000)
    before = main.getstate()
    _rank_rng(88000, 1, 1, 1).random()
    assert main.getstate() == before                     # PPO 난수 비소비


def test_rank_pair_loss_direction():
    cost = torch.tensor([1.0, 3.0, 2.0], requires_grad=True)   # [prog0, ww, prog2]
    # PROG 선호: 진행 비용이 이미 낮으면 손실 작아야
    l_good = rank_pair_loss(cost, 1, [(0, "PROG")])
    l_bad = rank_pair_loss(cost, 1, [(0, "WAIT")])
    assert float(l_good) < float(l_bad)
    l_good.backward()
    assert cost.grad[0] > 0 or cost.grad[1] < 0          # 선호 방향으로 기울기 존재


def test_select_progress_policy_independent():
    def mk(dur, kind="SERVE"):
        return {"YC1": SimpleNamespace(plan=SimpleNamespace(duration_s=dur),
                                       kind=SimpleNamespace(name=kind)),
                "YC2": SimpleNamespace(plan=None,
                                       kind=SimpleNamespace(name="WAIT"))}
    assigns = [mk(50.), mk(10.), mk(30., "PRE_REHANDLE"), mk(40.), mk(20.), mk(60.)]
    prog = list(range(6))
    s1 = select_progress_combos(assigns, prog, random.Random(7), 4)
    s2 = select_progress_combos(assigns, prog, random.Random(7), 4)
    assert s1 == s2                                      # 결정론 (같은 전용 rng)
    assert 1 in s1                                       # 총 계획시간 최소(SF-SPT 근사) 포함
    assert 2 in s1                                       # 행동유형 서명 층화 포함
    assert len(s1) == 4
