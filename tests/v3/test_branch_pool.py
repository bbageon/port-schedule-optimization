"""병렬로 굴려도 **순차와 같은 결과**가 나오는가 ([[YR-219]]).

반사실 세계를 프로세스에 나눠 굴린다. 세계들이 서로 독립이고 탐색이 좌표 기반이라
(`v3/actors/explore.py`) 순서에 안 의존하므로 **결과가 같아야** 한다.

같지 않다면 어딘가에 공유 상태나 순서 의존이 남은 것이고, 그러면 라벨이 실행마다
달라져 짝비교가 무너진다. 속도를 얻으려다 재현성을 잃는 셈이라 여기서 막는다.
"""
from __future__ import annotations

import pytest

from yard_rl.v3.reward import reset_rollout_calls, rollout_calls
from yard_rl.v3.stage import RolloutBudget, run_episode

LOAD, SEED = 400, 9_900_811
LABELS = 3


def _run(workers: int):
    reset_rollout_calls()
    r = run_episode(load=LOAD, arm="RL", seed=SEED, explore=0.5, workers=workers,
                    budget=RolloutBudget(max_labels=LABELS, identity_checks=0))
    rows = [{k: v for k, v in row.items()
             if k in ("doc_key", "t", "worlds", "seller_alt", "buyer_alt",
                      "seller_alt_coord", "phi_factual", "phi_seller_alt",
                      "phi_buyer_alt")} for row in r.labels]
    return {"phi": r.phi_krw, "traded": r.traded_edges, "worlds": r.rollout_worlds,
            "calls": rollout_calls(), "rows": rows}


def test_parallel_matches_serial():
    a = _run(1)
    b = _run(3)
    assert a["rows"] == b["rows"], (
        "병렬 라벨이 순차와 다르다 — 공유 상태나 순서 의존이 남아 있다")
    assert a["phi"] == b["phi"] and a["traded"] == b["traded"]
    assert a["worlds"] == b["worlds"] > 0


def test_rollout_counter_survives_process_split():
    """★작업자 계수기가 부모로 돌아오는가.

    안 돌아오면 판정 하드가드(`rollout_calls_during_eval == 0`)가 **교사 누출을
    통과시킨다** — 다른 프로세스에서 굴렸다는 이유로.
    """
    a, b = _run(1), _run(3)
    assert a["calls"] == b["calls"] == a["worlds"], (
        f"계수기 {a['calls']} / {b['calls']} · 세계 {a['worlds']}")


@pytest.mark.parametrize("workers", [1, 3])
def test_eval_path_stays_teacher_free(workers):
    """판정 경로(budget 없음)는 작업자 수와 무관하게 rollout 0 이다."""
    reset_rollout_calls()
    r = run_episode(load=LOAD, arm="RL", seed=SEED, workers=workers)
    assert rollout_calls() == 0 and r.rollout_worlds == 0
