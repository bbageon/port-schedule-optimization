"""본선 선급이 설계대로 무대에 들어갔는가 ([[YR-212]]).

설계 정본: `.claude/docs/architecture/02b-본선.md`

구 무대는 **전 본선이 동등**했다 — 척당 120 moves · 접안 4.8h · STS 1대.
설계는 3선급(50k·100k·150k GT)이고 물량이 최대 26배 차이난다. 그 차이가 실제로
무대에 나타나는지 여기서 잡는다. 안 그러면 본선 비용(Φ 항4)이 허수가 된다.
"""
from __future__ import annotations

import pytest

from yard_rl.v4.stage.vessels import (DAY_FLEET, STREAM_MOVES_PER_H,
                                      plan_streams, sample_day_vessels)
from yard_rl.v4.world.integrated.vessel import VESSEL_CLASSES, port_time_s
from yard_rl.v4.world.integrated.yard_layout import terminal_layout

SEEDS = range(9_900_500, 9_900_540)          # 비판정(진단) 대역


def test_three_classes_exist_with_design_values():
    got = {c.name: (c.gt, c.teu, c.sts) for c in VESSEL_CLASSES}
    assert got == {"SMALL": (50_000, 3_000, 2),
                   "MEDIUM": (100_000, 7_500, 4),
                   "LARGE": (150_000, 14_000, 6)}


def test_call_volume_is_15_to_30_percent_of_capacity():
    """기항 물량 = 선복 × 15~30%. **만선으로 오지 않는다.**"""
    for s in SEEDS:
        for v in sample_day_vessels(s):
            share = v.moves / v.cls.teu
            assert 0.149 <= share <= 0.301, f"{v.cls.name} {share:.3f}"


def test_port_time_comes_from_the_measured_table():
    """접안 시간은 **유도하지 않고 실측표 lookup** 이다(02b §2)."""
    assert port_time_s(400) == pytest.approx(15.8 * 3600)
    assert port_time_s(675) == pytest.approx(20.4 * 3600)
    assert port_time_s(1_687) == pytest.approx(27.7 * 3600)
    assert port_time_s(3_150) == pytest.approx(38.6 * 3600)
    assert port_time_s(9_999) == pytest.approx(62.6 * 3600)
    # 계단식 — 경계에서 안 튄다
    assert port_time_s(500) <= port_time_s(501)


def test_large_vessel_is_rare_not_absent():
    """대형선 희소성은 **출현 확률**이다 — 매일 오면 배울 변동이 없다."""
    n = sum(1 for s in SEEDS
            if any(v.cls.name == "LARGE" for v in sample_day_vessels(s)))
    share = n / len(list(SEEDS))
    target = dict((c[0], c[2]) for c in DAY_FLEET)["LARGE"]
    assert 0.10 < share < 0.55, f"대형선 출현 {share:.0%} (설계 {target:.0%})"


def test_small_and_medium_come_every_day():
    for s in SEEDS:
        names = [v.cls.name for v in sample_day_vessels(s)]
        assert names.count("SMALL") == 2, names
        assert names.count("MEDIUM") == 1, names


def test_streams_fit_the_one_stream_per_block_contract():
    """스트림 8(대형 없는 날) 또는 14(오는 날) — 21블록 안에 들어간다."""
    lay = terminal_layout()
    seen = set()
    for s in SEEDS:
        rows = plan_streams(sample_day_vessels(s), lay, s)
        assert len(rows) <= len(lay.ids), f"스트림 {len(rows)} > 블록 {len(lay.ids)}"
        assert not rows[0].get("dropped_peers"), "배를 뺐다 — 구성이 설계와 다르다"
        blocks = [r["block"] for r in rows]
        assert len(set(blocks)) == len(blocks), "한 블록에 스트림 둘"
        seen.add(len(rows))
    assert seen <= {8, 14}, f"스트림 수가 설계 밖: {sorted(seen)}"


def test_one_ship_supplies_from_several_blocks():
    """★한 배 = 여러 스트림 = 여러 블록 (다:1). 실제 터미널이 그렇다."""
    lay = terminal_layout()
    rows = plan_streams(sample_day_vessels(9_900_503), lay, 9_900_503)
    by_ship: dict[str, set[str]] = {}
    for r in rows:
        by_ship.setdefault(r["vessel_id"], set()).add(r["block"])
    for vid, blocks in by_ship.items():
        sts = next(r["sts"] for r in rows if r["vessel_id"] == vid)
        assert len(blocks) == sts, f"{vid}: 블록 {len(blocks)} ≠ STS {sts}"


def test_work_fits_inside_the_window():
    """작업은 창 안에 들어간다 — 접안이 창을 넘는 것과 별개다."""
    for s in SEEDS:
        for v in sample_day_vessels(s):
            assert v.work_s() < 24 * 3600, f"{v.cls.name} 작업 {v.work_s()/3600:.1f}h"
            assert v.moves_in_window(24 * 3600) == v.moves


def test_stream_productivity_matches_design():
    """스트림 1개 = STS 1대 = 25~30 moves/h."""
    assert 25.0 <= STREAM_MOVES_PER_H <= 30.0
    lay = terminal_layout()
    for r in plan_streams(sample_day_vessels(9_900_501), lay, 9_900_501):
        assert 3600.0 / r["cadence_s"] == pytest.approx(STREAM_MOVES_PER_H)


def test_daily_volume_is_near_the_design_average():
    """하루 본선 물량 평균 ≈ 3,982 moves (02b §3)."""
    tot = [sum(v.moves for v in sample_day_vessels(s)) for s in SEEDS]
    avg = sum(tot) / len(tot)
    assert 3_000 < avg < 5_000, f"평균 {avg:,.0f} moves (설계 3,982)"
