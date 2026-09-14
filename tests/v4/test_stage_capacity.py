"""하루 무대가 블록 용량을 넘겨 배정하지 않는가 ([[YR-316]]).

■ 무엇이 잘못됐었나
  반출 대상이 떨어진 블록은 반출 트럭이 반입으로 바뀌어(fallback) 반입이 몰리고,
  장치율 0.65 인 하루 무대에서 그 블록이 넘친다. 넘친 트럭은 안내자가 `용량 부족` 으로
  거절해 **조용히 사라졌다** — 판정 대역 9,000,000 에서 28일 중 13일이 전건 투입에
  실패해 실격됐다([[YR-314]]).

■ 무엇을 지켜야 하나
  ① 넘치던 날(시드 9,002,700 · 부하 5,000): 반입이 상한 안이고 **전건 투입**된다
  ② 안 넘치던 날(시드 9,001,700 · 부하 5,000): 돌린 트럭이 **0** — 바이트 동일해야
     기존 진단 날들의 결과가 그대로다
  ③ 돌려도 트럭 수·도착 시각·흐름은 그대로 — 블록만 바뀐다
"""
from __future__ import annotations

import collections

import pytest

from yard_rl.v4.stage.episode import run_episode
from yard_rl.v4.stage.orders import build_stage
from yard_rl.v4.world.integrated.profiles import build_h21_profile
from yard_rl.v4.world.integrated.terminal_stream import OBS_24H
from yard_rl.v4.world.integrated.yard_layout import terminal_layout

LOAD = 5_000
SEED_OVERFLOW = 9_002_700     # [[YR-314]] 판정 1일차 — 126대 거절됐던 날
SEED_CLEAN = 9_001_700        # 판정 0일차 — 전건 투입됐던 날


def _stage(seed: int) -> dict:
    return build_stage(load=LOAD, seed=seed, profile=build_h21_profile(),
                       layout=terminal_layout(), obs=OBS_24H, lead_mode="DIST")


def _inbound_by_block(b: dict) -> dict:
    c = collections.Counter(e["block"] for e in b["schedule"] if e["flow"] == "GATE_IN")
    return dict(c)


def test_overflow_day_stays_within_cap():
    """★넘치던 날 — 블록별 반입이 상한 안이다."""
    b = _stage(SEED_OVERFLOW)
    assert b["capacity_redirects_total"] > 0, "넘치던 날인데 돌린 트럭이 없다"
    for blk, n in _inbound_by_block(b).items():
        assert n <= b["inbound_cap"][blk], f"{blk}: 반입 {n} > 상한 {b['inbound_cap'][blk]}"


def test_overflow_day_is_fully_admitted():
    """★넘치던 날이 이제 **전건 투입**된다 — 판정 하드가드 1번."""
    r = run_episode(load=LOAD, arm="NO_REALLOC", dispatcher="SF_SPT", seed=SEED_OVERFLOW)
    assert r.admitted == LOAD, f"투입 {r.admitted} != {LOAD}"


def test_clean_day_is_untouched():
    """★안 넘치던 날은 돌린 트럭이 0 — 기존 진단 날의 결과가 그대로다."""
    b = _stage(SEED_CLEAN)
    assert b["capacity_redirects_total"] == 0
    for blk, n in _inbound_by_block(b).items():
        assert n <= b["inbound_cap"][blk]


def test_redirect_changes_block_only():
    """돌려도 트럭 수·도착 시각·흐름 구성은 그대로 — 블록만 바뀐다."""
    b = _stage(SEED_OVERFLOW)
    s = b["schedule"]
    assert len(s) == LOAD
    flows = collections.Counter(e["flow"] for e in s)
    assert flows["GATE_IN"] + flows["GATE_OUT"] == LOAD
    # 도착 시각은 단조 — 배정을 바꿔도 명단 순서는 안 바뀐다
    times = [e["arrival_s"] for e in s]
    assert times == sorted(times)
    # job_id 는 블록을 앞에 붙이므로 고유해야 한다
    assert len({e["job_id"] for e in s}) == LOAD


@pytest.mark.parametrize("seed", [SEED_OVERFLOW, SEED_CLEAN])
def test_cap_matches_engine_rule(seed):
    """상한이 엔진의 거절 규칙과 같은 식이다 — 물리 − 적재 − 양하 − 여유."""
    from yard_rl.v4.world.domain.enums import JobFlow
    from yard_rl.v4.world.integrated.multiblock import CAPACITY_MARGIN
    b = _stage(seed)
    g = build_h21_profile().block
    phys = g.bay_count * g.row_count * g.tier_max
    for blk, scn in b["scenarios"].items():
        dis = sum(1 for j in scn.jobs if j.flow == JobFlow.VESSEL_DISCHARGE)
        assert b["inbound_cap"][blk] == max(0, phys - len(scn.containers) - dis - CAPACITY_MARGIN)
