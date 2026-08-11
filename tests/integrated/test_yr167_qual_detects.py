"""YR-167 자격 **검사기가 결함을 실제로 잡는지** 시험 (39차 감사 보정).

1차 자격이 허위 통과를 낸 이유는 검사식이 항등식이라 실패가 불가능했기 때문이다.
여기서는 각 검사에 **고의 결함을 주입**하고 검사가 False 로 뒤집히는지 본다 —
"통과했다"가 아니라 "실패할 수 있다"를 증명하는 시험이다.
"""
from __future__ import annotations

import pytest

from yard_rl.experiments.yr167_observers import (SnapshotCollector,
                                                 vessel_concurrency)
from yard_rl.integrated.terminal_stream import (DIURNAL_DRAIN_S, DIURNAL_DAY_TOTAL,
                                                DIURNAL_VESSEL_CONCURRENCY,
                                                OBS_24H, ScheduledAnnouncer,
                                                TerminalStreamParams,
                                                build_diurnal, vessel_schedule_24h)
from yard_rl.integrated.profiles import build_h21_profile
from yard_rl.integrated.yard_layout import terminal_layout

SEED = 8_100_000


@pytest.fixture(scope="module")
def built():
    return build_diurnal(build_h21_profile(), SEED, obs=OBS_24H,
                         layout=terminal_layout(),
                         params=TerminalStreamParams(load_4h=DIURNAL_DAY_TOTAL),
                         background_seed=SEED)


def test_drain_extends_sim_end_beyond_day(built):
    """배수 구간이 있어야 하루 끝 게이트인 트럭도 투입 가능하다."""
    assert built["sim_end_s"] == OBS_24H.observe_s + DIURNAL_DRAIN_S
    for scn in built["scenarios"].values():
        assert scn.end_time == built["sim_end_s"]
    # 마지막 도착 + 주행이 지평 안에 들어온다 = 엔진 가드에 걸리지 않는다
    last = max(e["arrival_s"] + e["travel_s"] for e in built["schedule"])
    assert last <= built["sim_end_s"]


def test_no_tail_skip_in_announcer(built):
    """SKIP_TAIL 분기가 실제로 한 건도 발생하지 않아야 한다 (구판은 2건)."""
    ann = ScheduledAnnouncer(built["schedule"], lead_s=1800.0,
                             end_s=built["sim_end_s"])
    tail = [e for e in built["schedule"]
            if e["arrival_s"] + e["travel_s"] > ann.end_s]
    assert tail == []


def test_snapshot_collector_detects_missing_times():
    """스냅샷을 덜 찍으면 complete() 가 False — 구판 W5 는 실패가 불가능했다."""
    s = SnapshotCollector([0.0, 300.0, 600.0])
    assert not s.complete()

    class _Fake:
        blocks: dict = {}
    s.observe(_Fake(), 0.0)
    s.observe(_Fake(), 300.0)
    assert not s.complete()          # 600 미수집 → 미완
    s.observe(_Fake(), 600.0)
    assert s.complete()
    s.observe(_Fake(), 999.0)        # 격자 밖은 무시
    assert s.complete()


def test_snapshot_summary_flags_degenerate():
    """재공·대기열이 상수면 degenerate=True 로 표시된다(아무 일도 안 난 런 탐지)."""
    s = SnapshotCollector([0.0])
    s.rows = [{"t": 0.0, "block": f"Y{i:02d}", "wip": 0, "queue": 0,
               "backlog": 0, "stock": 900, "free": 500} for i in range(21)]
    assert s.summary()["degenerate"] is True
    s.rows[0]["wip"] = 3
    s.rows[1]["queue"] = 2
    assert s.summary()["degenerate"] is False


def test_vessel_concurrency_is_measured_not_assumed(built):
    """동시 활성은 실측이며 설계값 c=6 은 **평균**이다 — 순간값은 오르내린다."""
    params = TerminalStreamParams(load_4h=DIURNAL_DAY_TOTAL)
    dur = params.vessel_moves * params.sts_move_interval_s
    c = vessel_concurrency(built["vessel_schedule"], dur, OBS_24H.observe_s)
    assert c["min"] >= 1
    assert c["max"] > DIURNAL_VESSEL_CONCURRENCY      # 설계값보다 큰 순간이 존재
    assert abs(c["mean"] - DIURNAL_VESSEL_CONCURRENCY) <= 0.5
    assert c["vessel_hours"] == pytest.approx(30 * dur / 3600.0)


def test_concurrency_mean_alone_is_not_discriminative():
    """**평균만으로는 못 잡는다** — 30척을 한 시각에 몰아도 평균은 그대로 6.0이다.

    평균 = 총 본선시간 ÷ 창 이라 배치를 어떻게 바꿔도 동결 상수로 정해진다.
    판별력은 min(끊김 없음)·max(평탄 상한)에서 나온다 — W4 는 그 둘을 본다.
    """
    sched = [{"block": "Y01", "start_s": 0.0} for _ in range(30)]
    c = vessel_concurrency(sched, 17_280.0, OBS_24H.observe_s)
    assert abs(c["mean"] - DIURNAL_VESSEL_CONCURRENCY) <= 0.5   # 평균은 통과해버린다
    assert c["min"] == 0 and c["max"] == 30                     # 여기서 잡힌다


def test_vessel_schedule_shape_unchanged_by_drain():
    """배수 구간 추가가 본선 배치(동결 상수)를 바꾸지 않았는지 — 회귀 보호."""
    layout = terminal_layout()
    params = TerminalStreamParams(load_4h=DIURNAL_DAY_TOTAL)
    s = vessel_schedule_24h(layout, SEED, params, OBS_24H)
    assert len(s) == 30
    assert min(v["start_s"] for v in s) == 0.0
    assert max(v["start_s"] for v in s) == pytest.approx(
        OBS_24H.observe_s - params.vessel_moves * params.sts_move_interval_s)
