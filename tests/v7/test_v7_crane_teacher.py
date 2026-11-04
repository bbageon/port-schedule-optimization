"""v7 크레인 교사 이식 시험 ([[YR-332-b]]) — v4 시험(`test_crane_*.py`)에 더해 v7 이 새로 묻는 것.

  ① 사실 분기가 실제 궤적을 **결정 단위로** 재현한다 (v4 는 첫 결정 하나만 봤다)
  ② 라벨이 실제로 나온다 — 짧은 실행·작은 부하. 차이0·강제실패를 센다
  ③ 줄 길이 칸(⑨)이 실제로 변한다 — 죽은 칸이 아니다
  ④ 앞 8칸이 v4 계산과 **같다** — 이식 중 식이 바뀌지 않았다
  ⑤ ⑧ `n_cands` 의 실체 — 줄 선 트럭 수를 18로 나눈 값과 같다(v4 에서부터 · features 머리말)
  ⑥ 망 가중치를 바꾸면 **결정**이 바뀌고, 같은 망이면 같은 결정이다
  ⑦ 판정 경로(교사 없음)는 반사실을 한 번도 안 부른다 · 연속 무대도 RL_CRANE 을 굴린다
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from yard_rl.v7.crane import collect as C
from yard_rl.v7.crane import rollout as R
from yard_rl.v7.crane.features import CRANE_DIM, CRANE_FEATURES, V4_CORE_DIM
from yard_rl.v7.crane.policy import CraneNet
from yard_rl.v7.dispatch import RL_CRANE
from yard_rl.v7.stage import episode as E
from yard_rl.v7.stage.rollout import RolloutBudget

LOAD, SEED = 300, 9_900_335          # 비판정(진단) 대역
HORIZON = 3600.0                      # 시험용 1시간 — 실제 학습은 3시간


def test_nine_cells_and_order():
    assert CRANE_DIM == 9 and V4_CORE_DIM == 8
    assert CRANE_FEATURES[-1] == "queue_trucks"
    from yard_rl.v4.crane.features import CRANE_FEATURES as V4
    assert CRANE_FEATURES[:8] == V4


# ─────────────────────────────────────────────────────────── ①② 교사 한 번 굴리기
@pytest.fixture(scope="module")
def taught():
    """RL_CRANE 하루 + 교사(라벨 4건). 본 궤적의 결정 전부와 분기 원본을 함께 모은다."""
    main_trace: list = []
    box: dict = {"start": {}}
    mp = pytest.MonkeyPatch()
    orig_hook = C.CraneCollector.hook
    orig_finish = C.CraneCollector.finish

    def hook(self, mbt, market, orders, records):
        inner = orig_hook(self, mbt, market, orders, records)
        bmap = {id(s): b for b, s in mbt.blocks.items()}

        def on_crane(sim, dp, gb, assign):
            start = len(main_trace)
            for c in sorted(assign):
                ref = getattr(assign[c], "job_ref", None)
                main_trace.append((round(sim.clock, 6), bmap[id(sim)], c,
                                   getattr(ref, "job_id", None)))
            n = len(self.rows)
            out = inner(sim, dp, gb, assign)
            if len(self.rows) > n:              # 이 결정이 표본으로 뽑혔다 = 분기점
                box["start"][list(self.rows)[-1]] = start
            return out
        return on_crane

    def finish(self):
        out = orig_finish(self)
        box["collector"] = self
        return out

    mp.setattr(C.CraneCollector, "hook", hook)
    mp.setattr(C.CraneCollector, "finish", finish)
    mp.setattr(R, "TRACE_DECISIONS", True)
    try:
        res = E.run_episode(load=LOAD, seed=SEED, dispatcher=RL_CRANE,
                            crane_budget=RolloutBudget(max_labels=4, identity_checks=0),
                            horizon_s=HORIZON, workers=1)
    finally:
        mp.undo()
    return res, main_trace, box["collector"], box["start"]


def test_labels_come_out_and_accounting_is_complete(taught):
    res = taught[0]
    cs = res.crane_stats
    assert cs["labeled"] > 0, cs
    assert len(res.crane_labels) == 2 * cs["labeled"]
    assert cs["factual_mismatch"] == 0, cs
    assert cs["hook_errors"] == 0
    assert res.policy_exceptions == 0
    assert res.traded_edges == 0                      # 크레인 전용 — 시장은 안 열린다
    assert cs["seen"] >= cs["usable"] >= cs["sampled"] >= cs["labeled"] + cs["force_failed"]
    assert 0 <= cs["zero"] <= cs["labeled"]           # 차이 0 결정은 세어 둔다
    assert all(len(s.row) == CRANE_DIM for s in res.crane_labels)
    assert cs["scale"] >= 1_000.0                     # 날별 눈금(하한 1,000원)


def test_real_labels_feed_the_trainer(taught):
    """하루에서 나온 **실제 라벨**이 학습기에 들어가 망을 움직인다(목표 눈금 = 그날 격차)."""
    from yard_rl.v7.crane.fit import CraneTrainer
    from yard_rl.v7.crane.train import _labelset
    res = taught[0]
    ls = _labelset(res.crane_labels, res.crane_stats)
    assert ls.scale == res.crane_stats["scale"]
    torch.manual_seed(0)
    net = CraneNet()
    before = [p.detach().clone() for p in net.parameters()]
    rep = CraneTrainer(net).fit(ls, seed=1)
    assert rep.steps > 0 and rep.n > 0 and rep.n_val > 0
    assert any(not torch.equal(b, p) for b, p in zip(before, net.parameters()))


def test_factual_branch_replays_the_real_trajectory(taught):
    """★사실 분기의 결정 **전부**가 실제 궤적과 같다 — 분기점 뒤 같은 순서로.

    대조는 **진행 순서**(벽시계 순)로 한다. 블록마다 시계가 따로 가서, 분기점보다 시각이
    늦은 다른 블록 결정이 분기점보다 **먼저** 처리돼 있을 수 있다(그 결정은 스냅샷에 이미
    들어 있어 분기 세계가 다시 내지 않는다). 시각으로 거르면 이것이 가짜 불일치로 보인다.

    창 끝 `[t+H−30분, t+H]` 은 뺀다 — 분기 세계는 `sim.end = t+H` 라 그 뒤 도착 트럭을
    안 받고(투입 거부) 결정 지평(30분) 안의 사전정보도 달라진다(v4 머리말의 절단 효과).
    """
    from yard_rl.v7.world.integrated.profiles import build_h21_profile
    _, main_trace, coll, start = taught
    margin = max(1800.0, build_h21_profile().decision_horizon_s)
    checked = 0
    for r in coll.raw:
        fact = r.get("trace_factual")
        assert fact is not None
        t, cut = r["t"], r["t"] + HORIZON - margin
        k = start[r["key"]]
        a = [e for e in main_trace[k:] if e[0] < cut]
        b = [e for e in fact if e[0] < cut]
        assert a and a[0][:3] == (round(t, 6), r["key"][1], a[0][2])
        assert a == b, (t, next(((i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y),
                                (len(a), len(b))))
        checked += len(a)
    assert coll.raw and checked > 0


# ─────────────────────────────────────────────────────────── ③④⑤ 특징
@pytest.fixture(scope="module")
def feature_rows():
    """하루 동안 정책이 본 행 전부(후보마다) + 같은 순간 v4 식으로 계산한 앞 8칸."""
    import yard_rl.v7.crane.policy as P
    from yard_rl.v4.crane.features import crane_features as v4_features

    rows, v4rows = [], []
    orig = P.crane_features

    def spy(sim, cid, gc):
        x = orig(sim, cid, gc)
        rows.append(x)
        v4rows.append(v4_features(sim, cid, gc))
        return x

    P.crane_features = spy
    try:
        E.run_episode(load=1_500, seed=SEED, dispatcher=RL_CRANE)
    finally:
        P.crane_features = orig
    return rows, v4rows


def test_queue_cell_is_alive(feature_rows):
    rows, _ = feature_rows
    q = [r[8] for r in rows]
    assert len(set(q)) >= 3, sorted(set(q))[:10]
    assert max(q) >= 0.2


def test_first_eight_cells_equal_v4(feature_rows):
    rows, v4rows = feature_rows
    assert rows and all(r[:8] == v for r, v in zip(rows, v4rows))


def test_n_cands_is_really_the_waiting_queue(feature_rows):
    """⑧ 의 실체를 못 박는다 — 이름은 후보 수지만 값은 줄 선 트럭 수/18(상한 1)이다."""
    rows, _ = feature_rows
    real = [r for r in rows if any(r[:7])]         # WAIT 행(전부 0) 제외
    same = sum(1 for r in real if abs(r[7] - min(r[8] * 10.0 / 18.0, 1.0)) < 1e-9)
    assert real and same == len(real), (same, len(real))


# ─────────────────────────────────────────────────────────── ⑥ 망이 결정을 바꾼다
def _decisions(net):
    seen = []
    orig = E._apply

    def spy(sim, assign):
        for c in sorted(assign):
            ref = getattr(assign[c], "job_ref", None)
            seen.append((round(sim.clock, 6), c, getattr(ref, "job_id", None)))
        return orig(sim, assign)

    E._apply = spy
    try:
        r = E.run_episode(load=LOAD, seed=SEED, dispatcher=RL_CRANE, crane_net=net)
    finally:
        E._apply = orig
    return seen, r


def test_weights_change_decisions_and_same_weights_repeat():
    torch.manual_seed(11)
    a = CraneNet()
    torch.manual_seed(12)
    b = CraneNet()
    da, ra = _decisions(a)
    da2, _ = _decisions(a)
    db, rb = _decisions(b)
    assert da == da2, "같은 망인데 결정이 갈렸다"
    assert da != db, "망을 바꿨는데 결정이 하나도 안 바뀌었다 — 정책이 안 쓰이고 있다"
    first = next(i for i, (x, y) in enumerate(zip(da, db)) if x != y)
    assert first < len(da)


def test_flipping_the_output_layer_changes_a_ranking():
    """같은 후보 둘의 순서가 출력층 부호를 뒤집으면 뒤집힌다 — 점수가 순서를 정한다."""
    torch.manual_seed(5)
    net = CraneNet()
    x = torch.tensor([[0, 1, 0.2, 0, .3, .2, 0, .1, .1], [0, 1, 1.5, 0, .1, .4, 0, .1, .1]],
                     dtype=torch.float32)
    with torch.no_grad():
        s1 = net(x)
        last = net.net[-1]
        last.weight.mul_(-1.0)
        last.bias.mul_(-1.0)
        s2 = net(x)
    assert (s1[0] < s1[1]) != (s2[0] < s2[1])


# ─────────────────────────────────────────────────────────── ⑦ 판정 경로 · 연속 무대
def test_eval_path_makes_no_counterfactual_worlds():
    from yard_rl.v7.reward import reset_rollout_calls, rollout_calls
    reset_rollout_calls()
    r = E.run_episode(load=LOAD, seed=SEED, dispatcher=RL_CRANE)
    assert rollout_calls() == 0 and r.crane_labels == [] and r.crane_stats == {}


def test_month_stage_runs_the_crane_policy(fixed_container_input):
    from yard_rl.v7.stage.month import plan_days
    from yard_rl.v7.stage.month_run import run_month
    days = plan_days(SEED, loads=[120])
    res = run_month(seed=SEED, days=days, dispatcher=RL_CRANE)
    assert res.policy_exceptions == 0 and res.traded_edges == 0
    assert res.skipped == 0 and res.days[0].n_trucks == 120
