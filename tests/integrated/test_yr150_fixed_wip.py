"""YR-150 4차 재정의 — 고정 재공량(WIP) 계약 테스트 (사용자 결정 2026-08-08)."""
import pytest

from yard_rl.experiments.yr149_load_cells import _sim_from
from yard_rl.integrated.load_cells import make_a_cell, make_b_cell
from yard_rl.integrated.multiblock import MultiBlockTerminal, TransferError
from yard_rl.integrated.profiles import build_h21_profile
from yard_rl.integrated.terminal_stream import (ObservationContract,
                                                TerminalStreamParams,
                                                WipAdmissionController,
                                                _job_from_entry, admission_epochs,
                                                build_fixed_wip)
from yard_rl.integrated.yard_layout import terminal_layout

SEED = 6_300_000


@pytest.fixture(scope="module")
def built():
    return build_fixed_wip(build_h21_profile(), SEED, wip_target=63)


# ------------------------------------------------------------------ 생성 계약
def test_fill_sums_to_target_and_covers_all_blocks(built):
    """초기 채움 합계 = L, 전 블록 ≥1 (시간 장부 활성화 조건)."""
    assert len(built["fill"]) == 63
    per = {b: 0 for b in terminal_layout().ids}
    for e in built["fill"]:
        per[e["block"]] += 1
    assert min(per.values()) >= 1


def test_pool_entries_are_predrawn_and_deterministic(built):
    """pool 은 속성 전부 사전 추첨·재구성 시 동일 (런타임 무작위 소비 0)."""
    again = build_fixed_wip(build_h21_profile(), SEED, wip_target=63)
    assert again["pool"] == built["pool"] and again["fill"] == built["fill"]
    e = built["pool"][0]
    assert {"job_id", "block", "flow", "travel_s", "travel_base_s",
            "exit_travel_s"} <= set(e)


def test_gate_out_targets_are_unique_per_block(built):
    """반출 대상은 **(블록, 컨테이너) 단위로** 중복 예약이 없어야 한다.

    컨테이너 id 는 블록별 생성이라 전역 유일이 아니다(작업 id 만 namespacing 됨) —
    투입 검사(`tgt in sim.stacks.containers`)도 블록 범위이므로 이것이 정확한 계약이다.
    """
    tgts = [(e["block"], e["target"]) for e in built["pool"]
            if e["flow"] == "GATE_OUT"]
    tgts += [(s.meta["h21_block"], j.target_container)
             for s in built["scenarios"].values()
             for j in s.jobs if j.is_external_truck and j.target_container]
    assert len(tgts) == len(set(tgts))


def test_scenarios_contain_only_fill_not_pool(built):
    """pool 은 시나리오 밖에 있어야 한다 — 안에 넣으면 용량·backlog 지표를 오염시킨다."""
    n_ext = sum(len([j for j in s.jobs if j.is_external_truck])
                for s in built["scenarios"].values())
    assert n_ext == 63


def test_walkin_prediction_uses_expected_travel_only(built):
    """walk-in 예측 도착 = gate-in + 기대 주행 — 실현 잔여편차 미참조 (누출 0)."""
    layout = terminal_layout()
    for b, s in built["scenarios"].items():
        for j in s.jobs:
            if j.is_external_truck:
                expect = j.appointment_gate_time + layout.gate_to_block_s(b)
                assert j.estimated_block_arrival == pytest.approx(expect)


def test_wip_target_below_block_count_rejected():
    with pytest.raises(ValueError):
        build_fixed_wip(build_h21_profile(), SEED, wip_target=20)


def test_admission_epoch_grid():
    obs = ObservationContract()
    eps = admission_epochs(obs)
    assert eps[0] == 0.0 and eps[-1] <= obs.observe_s
    assert all(b - a == 60.0 for a, b in zip(eps, eps[1:]))


# ------------------------------------------------------------------ 엔진 투입 수술
def _small_mbt():
    return MultiBlockTerminal({"A": _sim_from(make_a_cell(6_310_000, 50)),
                               "B": _sim_from(make_b_cell(6_310_001))})


def _entry(bid: str, jid: str) -> dict:
    return {"job_id": jid, "block": bid, "flow": "GATE_IN", "target": None,
            "size_ft40": True, "travel_s": 300.0, "travel_base_s": 300.0,
            "exit_travel_s": 300.0}


def test_admit_registers_everywhere_and_preserves_invariants():
    mbt = _small_mbt()
    n0 = len(mbt.ledger.records)
    tl = mbt.blocks["A"].time_ledger
    n_tl0, n_sorted0 = len(tl.records), len(tl._a_sorted)
    job = _job_from_entry(_entry("A", "A:W-TEST0"), 0.0)
    mbt.admit_external_job("A", job, gate_in_s=0.0, travel_s=300.0)
    assert len(mbt.ledger.records) == n0 + 1
    assert mbt.ledger.records["A:W-TEST0"].owner == "A"
    assert len(tl.records) == n_tl0 + 1 and len(tl._a_sorted) == n_sorted0 + 1
    assert mbt.blocks["A"].jobs["A:W-TEST0"].actual_block_arrival == 300.0
    mbt.check_invariants()


def test_admit_duplicate_and_bad_block_fail_closed():
    mbt = _small_mbt()
    job = _job_from_entry(_entry("A", "A:W-DUP"), 0.0)
    mbt.admit_external_job("A", job, gate_in_s=0.0, travel_s=300.0)
    with pytest.raises(TransferError):
        mbt.admit_external_job("A", _job_from_entry(_entry("A", "A:W-DUP"), 0.0),
                               gate_in_s=0.0, travel_s=300.0)
    with pytest.raises(TransferError):
        mbt.admit_external_job("Z", _job_from_entry(_entry("Z", "Z:W-X"), 0.0),
                               gate_in_s=0.0, travel_s=300.0)


def test_admit_gate_out_requires_existing_target():
    mbt = _small_mbt()
    e = _entry("A", "A:W-OUT")
    e["flow"], e["target"] = "GATE_OUT", "NO-SUCH-CONTAINER"
    with pytest.raises(TransferError):
        mbt.admit_external_job("A", _job_from_entry(e, 0.0),
                               gate_in_s=0.0, travel_s=300.0)


def test_admit_arrival_beyond_end_rejected():
    mbt = _small_mbt()
    end = mbt.blocks["A"].end
    with pytest.raises(TransferError):
        mbt.admit_external_job("A", _job_from_entry(_entry("A", "A:W-LATE"), end),
                               gate_in_s=end, travel_s=300.0)


def test_controller_counts_pipeline_with_lead():
    """lead>0 이면 투입확정·미진입도 목표에 포함 — 과잉 투입 방지."""
    mbt = _small_mbt()
    job = _job_from_entry(_entry("A", "A:W-PIPE"), 900.0)
    mbt.admit_external_job("A", job, gate_in_s=900.0, travel_s=300.0)
    inside, pipeline = WipAdmissionController.wip_now(mbt, 0.0)
    assert pipeline >= 1                       # 아직 gate-in 전 — pipeline 으로 계수
    inside2, pipeline2 = WipAdmissionController.wip_now(mbt, 900.0)
    assert inside2 >= inside + 1               # gate-in 후에는 내부로 이동


def test_extra_review_epochs_default_is_byte_identical():
    """extra_review_epochs 기본값 () 은 기존 epoch 목록을 바꾸지 않는다(골든 보존)."""
    a, b = _small_mbt(), MultiBlockTerminal(
        {"A": _sim_from(make_a_cell(6_310_000, 50)),
         "B": _sim_from(make_b_cell(6_310_001))}, extra_review_epochs=())
    ea = a.blocks["A"].review_epochs
    eb = b.blocks["A"].review_epochs
    assert ea == eb
    c = MultiBlockTerminal({"A": _sim_from(make_a_cell(6_310_000, 50)),
                            "B": _sim_from(make_b_cell(6_310_001))},
                           extra_review_epochs=(60.0, 120.0))
    assert 60.0 in c.blocks["A"].review_epochs and 120.0 in c.blocks["A"].review_epochs


# ------------------------------------------------------------------ 본선 재보정 (2026-08-08)
def test_vessel_workload_within_derived_anchor(built):
    """계획 본선 작업률이 유도 앵커(145~170 moves/h) 안 — 12 process × 120 moves."""
    obs = ObservationContract()
    planned = 0.0
    n_load = n_dis = 0
    for s in built["scenarios"].values():
        for v in s.vessels:
            planned += min(v.plan.total_moves,
                           max(0.0, (obs.observe_s - v.plan.planned_start_s)
                               / v.plan.sts_move_interval_s))
            if v.work_type.value == "LOAD":
                n_load += 1
            else:
                n_dis += 1
    rate = planned / (obs.observe_s / 3600.0)
    assert 145.0 <= rate <= 170.0
    # 블록당 1 process 여도 양하/적하가 모두 존재해야 한다 (offset 정정)
    assert n_load >= 4 and n_dis >= 4 and n_load + n_dis == 12


def test_vessel_type_offset_default_is_byte_identical():
    """vessel_type_offset 기본값 0 은 기존 생성과 바이트 동일해야 한다(골든 보존)."""
    from yard_rl.integrated.load_cells import generate_block
    a = generate_block(6_320_000, 50)
    b = generate_block(6_320_000, 50)
    assert [v.work_type for v in a.vessels] == [v.work_type for v in b.vessels]
    assert a.vessels and a.vessels[0].work_type.value == "DISCHARGE"  # v=0·offset 0
