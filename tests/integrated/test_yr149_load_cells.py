"""YR-149 5부하 셀 계약 고정 — 중첩·균등 솎기·동결 template·B 동일."""
from __future__ import annotations

import pytest

from yard_rl.integrated.load_cells import (B_SIZE, CELL_SIZES, MASTER_SIZE, _thin,
                                           cell_params, config_digest, ext_job_rows,
                                           make_a_cell, make_b_cell, nested_external)

SEED_A, SEED_B = 907_000, 907_100


def test_thin_is_even_and_exact():
    seq = list(range(150))
    out = _thin(seq, 50)
    assert len(out) == 50 and out == sorted(set(out))     # 정확 개수·중복 없음·순서 보존
    gaps = [b - a for a, b in zip(out, out[1:])]
    assert max(gaps) - min(gaps) <= 1                     # 등간격 (편차 ≤1)


def test_nested_external_is_chain_subset():
    ext = list(range(MASTER_SIZE))
    sets = {m: set(nested_external(ext, m)) for m in CELL_SIZES}
    for small, big in zip(CELL_SIZES, CELL_SIZES[1:]):
        assert sets[small] <= sets[big], f"{small} ⊄ {big}"
    assert all(len(sets[m]) == m for m in CELL_SIZES)


def test_a_cells_nested_with_identical_fields():
    rows = {m: {r[0]: r for r in ext_job_rows(make_a_cell(SEED_A, m))}
            for m in CELL_SIZES}
    for small, big in zip(CELL_SIZES, CELL_SIZES[1:]):
        assert set(rows[small]) <= set(rows[big])
        for jid, row in rows[small].items():
            assert rows[big][jid] == row                  # 유지 작업 필드 바이트 동일
    assert all(len(rows[m]) == m for m in CELL_SIZES)


def test_layout_and_vessels_invariant_across_cells():
    digs = [config_digest(make_a_cell(SEED_A, m)) for m in CELL_SIZES]
    assert len({d["layout"] for d in digs}) == 1          # 초기 적재 동일
    assert len({d["vessels"] for d in digs}) == 1         # 본선 일정·마감 동일


def test_b_cell_deterministic_and_sized():
    d1, d2 = config_digest(make_b_cell(SEED_B)), config_digest(make_b_cell(SEED_B))
    assert d1 == d2 and d1["n_external"] == B_SIZE


def test_frozen_template_shared_by_both_blocks():
    pa, pb = cell_params(MASTER_SIZE), cell_params(B_SIZE)
    for f in ("n_vessels", "vessel_moves", "fill_ratio", "gaussian",
              "time_contract_v2", "gate_block_contract", "vessel_deadline_achievable",
              "vessel_deadline_mult", "arrival_peak_amp", "gate_travel_mu_s",
              "horizon_s", "drain_window_s"):
        assert getattr(pa, f) == getattr(pb, f), f
    assert (pa.n_external, pb.n_external) == (MASTER_SIZE, B_SIZE)   # 유일 차이


def test_exact_truck_count_not_gaussian():
    """gaussian=False 라야 트럭 수가 정확히 고정된다 (spec 요구)."""
    assert cell_params(100).gaussian is False
    scn = make_a_cell(SEED_A, 100)
    assert sum(1 for j in scn.jobs if j.is_external_truck) == 100


def test_unknown_size_rejected():
    with pytest.raises(ValueError):
        nested_external(list(range(MASTER_SIZE)), 60)
