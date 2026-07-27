"""YR-103 G0 — 시간계약(180~420s)·혼잡도 관측 계약 테스트 (spec §검증 게이트 G0)."""
import dataclasses
from types import SimpleNamespace

import pytest

from yard_rl.domain.enums import InformationLevel
from yard_rl.integrated import TerminalSimulator
from yard_rl.integrated.block_congestion import W_D, _c01, block_congestion
from yard_rl.integrated.profiles import build_calibrated_profile
from yard_rl.integrated.scenario_gen import (GATE_BLOCK_MAX_S, GATE_BLOCK_MIN_S,
                                             calibrated_load_params,
                                             generate_terminal_scenario)

SEED = 837_000


def _params(gb: bool):
    return dataclasses.replace(calibrated_load_params("mid", vessel_deadline_mult=2.0),
                               time_contract_v2=True, gate_block_contract=gb)


@pytest.fixture(scope="module")
def scen_on():
    return generate_terminal_scenario(build_calibrated_profile(), SEED, _params(True))


# ---------------------------------------------------------------- 시간계약
def test_g0_support_and_order(scen_on):
    n = 0
    for j in scen_on.jobs:
        a = getattr(j, "actual_gate_in", None)
        b = j.actual_block_arrival
        if a is None or b is None or not j.is_external_truck:
            continue
        n += 1
        assert GATE_BLOCK_MIN_S - 1e-9 <= b - a <= GATE_BLOCK_MAX_S + 1e-9   # B−A ∈ [180,420]
        assert a <= b                                                        # A ≤ B
    assert n > 0


def test_g0_deterministic(scen_on):
    s2 = generate_terminal_scenario(build_calibrated_profile(), SEED, _params(True))
    for j1, j2 in zip(scen_on.jobs, s2.jobs):
        assert j1.actual_block_arrival == j2.actual_block_arrival
        assert getattr(j1, "estimated_block_arrival", None) == \
               getattr(j2, "estimated_block_arrival", None)


def test_g0_off_bytes_identical():
    """opt-in OFF = 기존 v2 와 완전 동일 (전용 스트림 미소비 검증)."""
    prof = build_calibrated_profile()
    a = generate_terminal_scenario(prof, SEED, _params(False))
    b = generate_terminal_scenario(prof, SEED,
                                   dataclasses.replace(calibrated_load_params(
                                       "mid", vessel_deadline_mult=2.0), time_contract_v2=True))
    for j1, j2 in zip(a.jobs, b.jobs):
        assert j1.actual_block_arrival == j2.actual_block_arrival
        assert getattr(j1, "actual_gate_in", None) == getattr(j2, "actual_gate_in", None)
    assert "gate_block_contract" not in a.meta


def test_g0_meta_stamp(scen_on):
    assert scen_on.meta.get("gate_block_contract") == "180-420s-v1"


def test_g0_estimated_uses_contract_mean(scen_on):
    for j in scen_on.jobs:
        est = getattr(j, "estimated_block_arrival", None)
        appt = getattr(j, "appointment_gate_time", None)
        if est is None or appt is None:
            continue
        assert est - appt == pytest.approx(300.0)      # 예측 = 예약 + 계약 중심값 (draw 미참조)


# ---------------------------------------------------------------- 혼잡도 (fake sim 단위)
class _FS:
    def __init__(self, now, jobs, avail, n_cont, sla=1800.0):
        self.now = now
        self.jobs = {f"J{i}": j for i, j in enumerate(jobs)}
        blk = SimpleNamespace(bay_count=24, row_count=10, tier_max=6)
        cranes = [SimpleNamespace(crane_id="YC-L"), SimpleNamespace(crane_id="YC-W")]
        self.profile = SimpleNamespace(block=blk, cranes=cranes, long_wait_sla_s=sla)
        self._avail = avail
        self.fleet = SimpleNamespace(get=lambda cid: SimpleNamespace(
            state=SimpleNamespace(available_at=self._avail.get(cid, 0.0))))
        self.stacks = SimpleNamespace(containers={f"C{i}": None for i in range(n_cont)})


def _truck(status, arrival=None, est=None, appt=None):
    return SimpleNamespace(is_external_truck=True, status=SimpleNamespace(name=status),
                           actual_block_arrival=arrival, estimated_block_arrival=est,
                           appointment_gate_time=appt)


def test_congestion_empty_is_zero():
    c = block_congestion(_FS(0.0, [], {}, 0))
    assert c == {"Q": 0.0, "W": 0.0, "L": 0.0, "F": 0.0, "D": 0.0,
                 "C_minus": 0.0, "C_zero": 0.0, "C_plus": 0.0}


def test_congestion_monotone_and_bounded():
    prev = -1.0
    for n_wait in (2, 6, 12, 24):                       # Q·W·L 동반 증가 → C_int 단조↑
        jobs = [_truck("WAITING", arrival=0.0) for _ in range(n_wait)]
        c = block_congestion(_FS(900.0, jobs, {"YC-L": 1500.0, "YC-W": 1200.0}, 400))
        for v in c.values():
            assert 0.0 <= v <= 1.0
        assert c["C_zero"] >= prev
        prev = c["C_zero"]


def test_congestion_sign_variants():
    jobs = ([_truck("WAITING", arrival=0.0) for _ in range(6)]
            + [_truck("PLANNED", est=1000.0 + 420.0, appt=1000.0)])   # D=1 (최대 지연예상)
    c = block_congestion(_FS(600.0, jobs, {}, 100))
    assert c["D"] == pytest.approx(1.0)
    assert c["C_minus"] == pytest.approx(_c01(c["C_zero"] - W_D))
    assert c["C_plus"] == pytest.approx(_c01(c["C_zero"] + W_D))
    assert c["C_minus"] <= c["C_zero"] <= c["C_plus"]


def test_congestion_info_safety_future_actual_not_read():
    """미도착(PLANNED) 작업의 미래 actual 을 변조해도 관측 불변 (G0 정보안전)."""
    base = [_truck("PLANNED", est=1300.0, appt=1000.0)]
    tam = [_truck("PLANNED", arrival=99_999.0, est=1300.0, appt=1000.0)]  # 미래 actual 오염
    assert block_congestion(_FS(0.0, base, {}, 50)) == block_congestion(_FS(0.0, tam, {}, 50))


def test_congestion_runtime_integration():
    """실제 sim 중간 상태에서 [0,1]·비퇴화 (G0 통합)."""
    prof = build_calibrated_profile()
    sim = TerminalSimulator(prof, generate_terminal_scenario(prof, SEED, _params(True)),
                            check_invariants=True)
    sim.info_level = InformationLevel.PRE_ADVICE
    for _ in range(12):
        dp = sim.run_until_decision()
        if dp is None:
            break
        from yard_rl.integrated.baselines import ResolverPolicy, ServiceFirstSPTPreference, _apply
        from yard_rl.integrated.candidates import CandidateGenerator
        gen = CandidateGenerator()
        gen_by = {c: gen.generate(sim, c, InformationLevel.PRE_ADVICE) for c in dp.crane_ids}
        _apply(sim, ResolverPolicy(ServiceFirstSPTPreference(), "SF").decide(sim, dp, gen_by))
    c = block_congestion(sim)
    assert all(0.0 <= v <= 1.0 for v in c.values())
    assert c["F"] > 0.0                                  # runtime 장치율 실반영
