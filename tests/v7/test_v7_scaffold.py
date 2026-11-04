"""v7 골격 시험 ([[YR-332-a]]) — 크레인 전용 · 본선 단가 · DGT 리드타임 · 행동 범위.

무엇을 못 박나
  ① 재배정을 끄면 **거래가 정확히 0** 이고 시장이 한 번도 안 불린다
  ② 본선 유휴 단가가 35.88원/GT·시간으로 비용에 들어간다
  ③ DGT 리드타임 — 반입·반출 분위가 원 표본과 맞고, 통지 전 도착은 리드 0
  ④ 수명주기 불변식(통지 ≤ 게이트 진입 ≤ … ≤ 게이트 아웃) 위반 0
  ⑤ 규칙 크레인(SF_SPT)으로 하루 완주 · 전건 투입 · 정책 예외 0
  ⑥ 짧은 연속 무대(3일 · 작은 부하)에서 전건 투입 · 거래 0
     ⚠️ 30일 완주가 아니다 — 무거워서 3일 스모크로 줄였다. 그리고 그 3일은
        `conftest.fixed_container_input`(작은 고정 화물 고정물)을 쓴다. 생성 입력 그대로면
        반출 대상 상자가 모자라 NO_TARGET 으로 못 들어오는 트럭이 생긴다([[YR-331-c]]).
  ⑦ 미리 자리잡는 빈 이동 후보를 안 만든다(교착 탈출·파내기·쉬기는 남는다)
"""
from __future__ import annotations

import random

import pytest

from yard_rl.v7 import GENERATION, V6_BASE_COMMIT
from yard_rl.v7.schema.lifecycle import validate
from yard_rl.v7.stage import dgt_lead
from yard_rl.v7.stage.dgt_lead import (DGT_IN_NEGATIVE_S, DGT_IN_POSITIVE_S,
                                       DGT_OUT_NEGATIVE_S, DGT_OUT_POSITIVE_S,
                                       no_notice_share, sample_dgt_lead_s)

SEED = 9_900_332            # 비판정(진단) 대역
DAY_LOAD = 300

H = 3600.0


# ─────────────────────────────────────────────────────────── 세대 표식
def test_generation_and_base_commit():
    assert GENERATION == "v7"
    assert len(V6_BASE_COMMIT) == 40 and int(V6_BASE_COMMIT, 16) >= 0


# ─────────────────────────────────────────────────────────── ② 본선 단가
def test_vessel_price_is_charter_based():
    from yard_rl.v7.reward.krw import (KRW_VESSEL_GT_HOUR, vessel_idle_krw,
                                       vessel_krw_per_hour)
    assert KRW_VESSEL_GT_HOUR == 35.88
    assert vessel_krw_per_hour(100_000) == pytest.approx(3_588_000.0)
    # 10만 GT 배가 30분 놀면 179.4만원
    assert vessel_idle_krw(100_000, 1800.0) == pytest.approx(1_794_000.0)


def test_vessel_price_reaches_the_cost_function():
    """Φ 의 본선 항이 단가를 **실제로** 쓴다 — 상수만 바꾸고 배선이 딴 데를 보면 안 된다."""
    from yard_rl.v7.reward.phi import terminal_cost_krw
    phi = terminal_cost_krw({}, end_s=86_400.0, vessel_idle={"S1": (150_000.0, 7200.0)},
                            yc_extra_move_s=0.0, rehandles=0)
    assert phi.vessel == pytest.approx(35.88 * 150_000 * 2.0)
    assert phi.total == pytest.approx(phi.vessel)


# ─────────────────────────────────────────────────────────── ③ DGT 리드타임
def _doc_quantile(xs, p):
    """대조 문서와 같은 분위 방식 — `x[int(n·p)]`."""
    xs = sorted(xs)
    return xs[int(len(xs) * p)]


@pytest.mark.parametrize("flow,pos,neg,doc", [
    ("GATE_IN", DGT_IN_POSITIVE_S, DGT_IN_NEGATIVE_S, (0.34, 0.95, 2.02)),
    ("GATE_OUT", DGT_OUT_POSITIVE_S, DGT_OUT_NEGATIVE_S, (0.16, 2.45, 91.2)),
])
def test_embedded_table_reproduces_the_handover_document(flow, pos, neg, doc):
    """박아 둔 상수표가 대조 문서(2026-08-19) 표의 p25·중앙·p75 를 그대로 낸다."""
    raw = list(neg) + list(pos)
    got = [_doc_quantile(raw, p) / H for p in (0.25, 0.5, 0.75)]
    assert got == pytest.approx(doc, abs=0.01, rel=1e-3)   # 문서는 소수 둘째(91.2 는 첫째)에서 반올림
    assert len(neg) == 5
    assert len(raw) == (45 if flow == "GATE_IN" else 26)


@pytest.mark.parametrize("flow,doc,tol", [
    ("GATE_IN", (0.34, 0.95, 2.02), (0.1, 0.1, 0.1)),
    ("GATE_OUT", (0.16, 2.45, 91.2), (0.1, 0.1, 1.0)),
])
def test_sampler_quantiles_match_the_sample(flow, doc, tol):
    """뽑는 분포의 분위(역함수에 분위를 직접 넣는다)가 표본 분위와 맞는다."""
    got = [sample_dgt_lead_s(p, flow) / H for p in (0.25, 0.5, 0.75)]
    for g, d, t in zip(got, doc, tol):
        assert abs(g - d) <= t, (flow, got, doc)


def test_inbound_and_outbound_are_really_different():
    """반입·반출을 **따로** 뽑는다 — 같은 분포면 이 축을 넣은 의미가 없다."""
    assert sample_dgt_lead_s(0.75, "GATE_OUT") > 20 * sample_dgt_lead_s(0.75, "GATE_IN")


@pytest.mark.parametrize("flow,share", [("GATE_IN", 5 / 45), ("GATE_OUT", 5 / 26)])
def test_no_notice_share_and_empirical_draws(flow, share):
    """통지 전 도착 = 리드 0. 2만 번 뽑아 비율·중앙을 잰다."""
    assert no_notice_share(flow) == pytest.approx(share)
    rng = random.Random(f"test:{flow}")
    xs = sorted(sample_dgt_lead_s(rng.random(), flow) for _ in range(20_000))
    zero = sum(1 for x in xs if x == 0.0) / len(xs)
    assert zero == pytest.approx(share, abs=0.01)
    assert all(x >= 0.0 for x in xs)
    med = xs[len(xs) // 2] / H
    assert med == pytest.approx(sample_dgt_lead_s(0.5, flow) / H, abs=0.1)


def test_unknown_flow_is_rejected():
    with pytest.raises(ValueError):
        sample_dgt_lead_s(0.5, "VESSEL")


def test_stage_uses_dgt_by_flow_and_keeps_notice_before_arrival():
    """무대 명단 — 통지 ≤ 도착, 흐름별 리드 0 비율, 30분 고정은 선택지로 남는다."""
    from yard_rl.v7.stage.orders import (DEFAULT_LEAD_MODE, build_stage,
                                         orders_from_schedule)
    from yard_rl.v7.world.integrated.profiles import build_h21_profile
    assert DEFAULT_LEAD_MODE == "DGT"
    prof = build_h21_profile()
    built = build_stage(load=3_500, seed=SEED, profile=prof)
    assert built["lead_mode"] == "DGT"
    orders, _ = orders_from_schedule(built)
    assert all(o.copino_notice_s <= o.in_out_reserve_s + 1e-6 for o in orders.values())
    for flow in ("GATE_IN", "GATE_OUT"):
        rows = [e for e in built["schedule"] if e["flow"] == flow]
        # 0 리드 = 통지 전 도착(표본 비율) + 0시 직후 도착이라 눌린 것. 표본 비율보다 작을 수 없다.
        zero = sum(1 for e in rows if e["lead_s"] == 0.0) / len(rows)
        assert zero >= no_notice_share(flow) - 0.03, (flow, zero)
    fixed = build_stage(load=3_500, seed=SEED, profile=prof, lead_mode="FIXED")
    assert {min(1800.0, e["arrival_s"]) for e in fixed["schedule"]} == \
        {e["lead_s"] for e in fixed["schedule"]}


# ─────────────────────────────────────────────────────────── ①④⑤ 하루 무대
@pytest.fixture(scope="module")
def sf_spt_day():
    """규칙 크레인 하루 — 시장이 불리면 즉시 터지게 해 둔다(재배정 끔 확인)."""
    from yard_rl.v7.actors.market import Market
    from yard_rl.v7.actors.classical import ClassicalMarket
    from yard_rl.v7.stage import episode

    calls = {"market": 0}
    captured = {}
    mp = pytest.MonkeyPatch()

    def boom(*a, **k):
        calls["market"] += 1
        raise AssertionError("크레인 전용 경로에서 시장이 불렸다")

    original = episode.orders_from_schedule

    def capture(built):
        orders, records = original(built)
        captured.update(orders=orders, records=records)
        return orders, records

    mp.setattr(Market, "step", boom)
    mp.setattr(ClassicalMarket, "step", boom, raising=False)
    mp.setattr(episode, "orders_from_schedule", capture)
    try:
        res = episode.run_episode(load=DAY_LOAD, seed=SEED)
    finally:
        mp.undo()
    return res, captured, calls


def test_crane_only_is_the_default_and_trades_nothing(sf_spt_day):
    from yard_rl.v7.stage.episode import CRANE_ONLY_ARM, run_episode
    import inspect
    assert inspect.signature(run_episode).parameters["arm"].default == CRANE_ONLY_ARM == "NO_REALLOC"
    res, _, calls = sf_spt_day
    assert calls["market"] == 0
    assert res.traded_edges == 0 and res.n_space == 0 and res.n_time == 0
    assert res.decisions == 0          # 판매 결정 기록도 비어 있다


def test_rule_crane_day_completes_with_full_admission(sf_spt_day):
    res, captured, _ = sf_spt_day
    assert res.admitted == DAY_LOAD
    assert res.policy_exceptions == 0
    assert res.breakdown["n_trucks"] == DAY_LOAD
    assert res.phi_krw > 0 and res.breakdown["c_vessel"] >= 0


def test_lifecycle_invariants_hold_for_every_truck(sf_spt_day):
    _, captured, _ = sf_spt_day
    records = captured["records"]
    assert len(records) == DAY_LOAD
    bad = []
    for rec in records.values():
        try:
            validate(rec)
        except Exception as ex:          # noqa: BLE001
            bad.append(str(ex))
        if rec.gate_in_s is not None and rec.copino_notice_s > rec.gate_in_s + 1e-6:
            bad.append(f"{rec.doc_key}: 통지 {rec.copino_notice_s} > 게이트 {rec.gate_in_s}")
    assert not bad, bad[:5]
    done = sum(1 for r in records.values() if r.gate_out_s is not None)
    assert done >= DAY_LOAD - 5          # 창 끝 검열 몇 대만 허용


# ─────────────────────────────────────────────────────────── ⑦ 행동 범위
def test_exec_config_turns_off_preemptive_reposition_only():
    from yard_rl.v7.stage.exec_config import DEFAULT_EXEC_CONFIG, V7_CRANE_ONLY
    assert DEFAULT_EXEC_CONFIG is V7_CRANE_ONLY
    assert V7_CRANE_ONLY.safety_only is True
    assert V7_CRANE_ONLY.wait_mode == "WAIT"


def test_no_preemptive_reposition_candidates_but_rehandle_and_wait_remain():
    """같은 하루를 v7 범위와 v6 범위로 — v7 은 선제 위치조정 목표를 아예 안 묻는다."""
    from yard_rl.v7.stage import episode
    from yard_rl.v7.stage.exec_config import V6_LEGACY
    from yard_rl.v7.world.contract.schema import CandidateKind
    from yard_rl.v7.world.integrated.candidates import CandidateGenerator

    def run(cfg):
        seen = {"future": 0, "kinds": {}}
        mp = pytest.MonkeyPatch()
        orig_future = CandidateGenerator._future_target_bays
        orig_gen = CandidateGenerator.generate

        def future(self, *a, **k):
            seen["future"] += 1
            return orig_future(self, *a, **k)

        def gen(self, *a, **k):
            out = orig_gen(self, *a, **k)
            for gc in out.items:
                seen["kinds"][gc.kind] = seen["kinds"].get(gc.kind, 0) + 1
            return out
        mp.setattr(CandidateGenerator, "_future_target_bays", future)
        mp.setattr(CandidateGenerator, "generate", gen)
        try:
            episode.run_episode(load=DAY_LOAD, seed=SEED, exec_config=cfg)
        finally:
            mp.undo()
        return seen

    v7 = run(None)
    v6 = run(V6_LEGACY)
    assert v7["future"] == 0
    assert v6["future"] > 0
    assert v6["kinds"].get(CandidateKind.REPOSITION, 0) > v7["kinds"].get(CandidateKind.REPOSITION, 0)
    assert v7["kinds"].get(CandidateKind.WAIT, 0) > 0
    assert v7["kinds"].get(CandidateKind.SERVE, 0) > 0
    assert v7["kinds"].get(CandidateKind.PRE_REHANDLE, 0) > 0


# ─────────────────────────────────────────────────────────── ⑥ 짧은 연속 무대
def test_three_day_month_admits_everything_and_trades_nothing(fixed_container_input):
    """3일 스모크(작은 부하) — 30일 완주 대신. 고정 화물 고정물로 반출 대상을 맞춘다."""
    from yard_rl.v7.stage import month_run
    from yard_rl.v7.stage.month import plan_days

    captured = {}
    original = month_run.orders_from_schedule

    def capture(built):
        orders, records = original(built)
        captured.update(orders=orders, records=records, n=len(built["schedule"]))
        return orders, records

    mp = pytest.MonkeyPatch()
    mp.setattr(month_run, "orders_from_schedule", capture)
    try:
        days = plan_days(SEED, loads=[120, 120, 120])
        res = month_run.run_month(seed=SEED, days=days)
    finally:
        mp.undo()
    assert res.traded_edges == 0 and res.decisions == 0
    assert res.skipped == 0, res.truck_skips[:5]
    assert res.admitted == captured["n"] == 360
    assert res.policy_exceptions == 0
    for rec in captured["records"].values():
        validate(rec)
        if rec.gate_in_s is not None:
            assert rec.copino_notice_s <= rec.gate_in_s + 1e-6
    assert [d.n_trucks for d in res.days] == [120, 120, 120]
