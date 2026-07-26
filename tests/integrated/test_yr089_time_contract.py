"""YR-089 트럭 시간계약 v2 — spec 필수 계약 테스트.

A(게이트진입) ≤ B(블록도착) ≤ S(작업시작) ≤ C(작업완료) ≤ O(게이트진출).
학습비용 = 블록 처리시간(C−B) 적분, 최종 KPI = 터미널 턴타임(O−A), S−B 는 진단.
v1(기본값) 경로 바이트 동일(골든은 전체 스위트가 보증) — 여기서는 계약 등식·지식경계·결정론.
"""
import copy

from yard_rl.domain.enums import InformationLevel
from yard_rl.integrated import TerminalSimulator, build_integrated_profile
from yard_rl.integrated.baselines import (FIFOPreference, ResolverPolicy,
                                          ServiceFirstSPTPreference, run_joint_episode)
from yard_rl.integrated.candidates import CandidateGenerator
from yard_rl.integrated.cost_config import RewardCalculator, neutral_lambda_config
from yard_rl.integrated.scenario_gen import TerminalGenParams, generate_terminal_scenario

PROF = build_integrated_profile()
RC = RewardCalculator(neutral_lambda_config())
SEED = 310000
V2 = TerminalGenParams(time_contract_v2=True)
EPS = 1e-6


def _v2_sim(level=InformationLevel.PRE_ADVICE, seed=SEED, scenario=None):
    sc = scenario or generate_terminal_scenario(PROF, seed, V2)
    return TerminalSimulator(PROF, sc, info_level=level)


def _run_sf(sim):
    return run_joint_episode(sim, ResolverPolicy(ServiceFirstSPTPreference(), "SF"), RC,
                             generator=CandidateGenerator())


# ------------------------------------------------- 호환 (v1 불변)
def test_v1_default_has_no_v2_fields_and_no_ledger():
    sc = generate_terminal_scenario(PROF, SEED)          # 기본 = v1
    trucks = [j for j in sc.jobs if j.is_external_truck]
    assert trucks
    for j in trucks:
        assert j.appointment_gate_time is None and j.estimated_block_arrival is None
        assert j.exit_travel_s is None and j.actual_gate_out is None
    sim = TerminalSimulator(PROF, sc)
    assert sim.time_ledger is None                       # v1 = 구 비용 경로 그대로
    assert "time_contract" not in (sc.meta or {})


def test_v2_rng_reinterprets_same_stream_as_appointment():
    """스트림 격리: v2 예약시각 == v1 실제도착 (같은 공식·같은 draw 열 재해석) — 변화는
    정확히 '예약→실제' 인과 재구성으로 한정된다."""
    v1 = generate_terminal_scenario(PROF, SEED)
    v2 = generate_terminal_scenario(PROF, SEED, V2)
    t1 = {j.job_id: j for j in v1.jobs if j.is_external_truck}
    t2 = {j.job_id: j for j in v2.jobs if j.is_external_truck}
    assert set(t1) == set(t2)                            # 트럭 집합·flow 보존
    for jid, j2 in t2.items():
        assert abs(j2.appointment_gate_time - t1[jid].actual_block_arrival) < EPS
        assert j2.appointment_window_start < j2.appointment_gate_time < j2.appointment_window_end


# ------------------------------------------------- 사건 순서·지식경계
def test_event_order_per_truck():
    sim = _v2_sim()
    _run_sf(sim)
    done = [r for r in sim.time_ledger.records.values() if r.job_done is not None]
    assert done, "완료 트럭이 있어야 함"
    for r in done:
        assert (r.gate_in - EPS <= r.block_arrival <= r.service_start + EPS
                <= r.job_done + 2 * EPS <= r.gate_out + 3 * EPS)


def test_estimate_is_prediction_not_truth():
    """예측 = 예약 + 준수예측0 + 기대주행 (실현 draw 미참조) — 진실 도착과 일반적으로 다름."""
    sc = generate_terminal_scenario(PROF, SEED, V2)
    trucks = [j for j in sc.jobs if j.is_external_truck]
    for j in trucks:
        assert abs(j.estimated_block_arrival
                   - (j.appointment_gate_time + V2.gate_travel_mu_s)) < EPS
        assert abs(j.provided_eta - j.estimated_block_arrival) < EPS   # deprecated alias
    diff = sum(1 for j in trucks
               if abs(j.estimated_block_arrival - j.actual_block_arrival) > 1.0)
    assert diff >= 0.8 * len(trucks)


def test_appointment_shift_does_not_change_realized_cost():
    """실제 사건 고정 + 예약만 이동 → 실현 비용 불변 (비용은 actual 만 읽는다).
    BLOCK_ARRIVAL 정보수준 = 예약·ETA 미소비 경로라 행동 동일성도 보장."""
    sc1 = generate_terminal_scenario(PROF, SEED, V2)
    sc2 = copy.deepcopy(sc1)
    for j in sc2.jobs:
        if j.is_external_truck:
            j.appointment_gate_time += 600.0
            j.appointment_window_start += 600.0
            j.appointment_window_end += 600.0
            j.estimated_block_arrival += 600.0
            j.provided_eta = j.estimated_block_arrival
    r1 = run_joint_episode(_v2_sim(InformationLevel.BLOCK_ARRIVAL, scenario=sc1),
                           ResolverPolicy(FIFOPreference(), "FIFO"), RC,
                           generator=CandidateGenerator())
    r2 = run_joint_episode(_v2_sim(InformationLevel.BLOCK_ARRIVAL, scenario=sc2),
                           ResolverPolicy(FIFOPreference(), "FIFO"), RC,
                           generator=CandidateGenerator())
    assert abs(r1["total_cost"] - r2["total_cost"]) < EPS
    assert abs(r1["terminal_truck_area_h"] - r2["terminal_truck_area_h"]) < EPS


# ------------------------------------------------- 적분 등식
def test_terminal_area_equals_turntime_plus_censored():
    sim = _v2_sim()
    _run_sf(sim)
    tl = sim.time_ledger
    assert abs(tl.terminal_area_s - sum(tl.terminal_turntime_samples_s())) < 1e-3


def test_block_area_equals_block_turntime_samples():
    sim = _v2_sim()
    _run_sf(sim)
    tl = sim.time_ledger
    end = tl.closed_end_s
    expect = 0.0
    for r in tl.records.values():                        # 적분과 같은 정의로 재구성 (독립 산식)
        if r.block_arrival is None or r.block_arrival >= end:
            continue
        expect += min(r.job_done if r.job_done is not None else end, end) - r.block_arrival
    assert abs(tl.block_area_s - expect) < 1e-3


def test_block_turntime_decomposition_per_truck():
    sim = _v2_sim()
    _run_sf(sim)
    for r in sim.time_ledger.records.values():
        if r.job_done is None:
            continue
        yc_wait = r.service_start - r.block_arrival
        service = r.job_done - r.service_start
        assert abs((r.job_done - r.block_arrival) - (yc_wait + service)) < EPS
        # O−A == (B−A)+(S−B)+(C−S)+(O−C)
        total = ((r.block_arrival - r.gate_in) + yc_wait + service
                 + (r.gate_out - r.job_done))
        assert abs((r.gate_out - r.gate_in) - total) < EPS


def test_cost_truck_terms_equal_block_ledger():
    """v2 학습비용 원료 == 블록 점유 적분 (비용 장부 ↔ TimeLedger 등식). 첫 결정 이전
    폐기분까지 포함해 전 구간 cut 합으로 검증."""
    sim = _v2_sim()
    gen = CandidateGenerator()
    pol = ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    raw_tw = raw_lw = 0.0
    dp = sim.run_until_decision()
    raw = sim.cost.cut()
    raw_tw += raw.get("truck_wait", 0.0); raw_lw += raw.get("long_wait", 0.0)
    while dp is not None:
        gen_by = {c: gen.generate(sim, c, sim.info_level) for c in dp.crane_ids}
        from yard_rl.integrated.baselines import _apply
        _apply(sim, pol.decide(sim, dp, gen_by))
        dp = sim.run_until_decision()
        raw = sim.cost.cut()
        raw_tw += raw.get("truck_wait", 0.0); raw_lw += raw.get("long_wait", 0.0)
    tl = sim.time_ledger
    assert abs(raw_tw - tl.block_area_s) < 1e-3
    assert abs(raw_lw - tl.block_tail_area_s) < 1e-3
    assert tl.block_area_s > 0


# ------------------------------------------------- 검열·완주 guard·결정론
def test_censoring_no_reward_for_leaving_trucks_unserved():
    """전량 WAIT(미서비스) → 모든 도착 트럭이 end−B 노출 — 미완료로 비용을 낮출 수 없다."""
    from yard_rl.integrated.baselines import _apply, _wait_of
    sim = _v2_sim(InformationLevel.BLOCK_ARRIVAL)
    gen = CandidateGenerator()
    dp = sim.run_until_decision()
    while dp is not None:
        gb = {c: gen.generate(sim, c, sim.info_level) for c in dp.crane_ids}
        _apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})
        dp = sim.run_until_decision()
    tl = sim.time_ledger
    served = [r for r in tl.records.values() if r.job_done is not None]
    assert not served
    assert tl.censored_exposure_s() > 0
    arrived = [r for r in tl.records.values()
               if r.block_arrival is not None and r.block_arrival < tl.closed_end_s]
    expect = sum(tl.closed_end_s - r.block_arrival for r in arrived)
    assert abs(tl.block_area_s - expect) < 1e-3          # 전원 검열 노출 == 적분


def test_v2_no_new_event_kinds_and_deterministic():
    """gate 사건은 이벤트 큐에 없다(장부 내 경계처리) → event_log 어휘 불변 + 같은 seed 결정론."""
    sim1, sim2 = _v2_sim(), _v2_sim()
    r1, r2 = _run_sf(sim1), _run_sf(sim2)
    kinds = {k for _, k, _ in sim1.event_log}
    assert not any(k.startswith("TRUCK_GATE") for k in kinds)
    assert abs(r1["total_cost"] - r2["total_cost"]) < EPS
    assert r1["n_decisions"] == r2["n_decisions"]
    assert abs(r1["block_turntime_mean_min"] - r2["block_turntime_mean_min"]) < EPS


def test_no_completion_after_evaluation_end():
    """외부감사 결함1 회귀 가드: 평가창(end) 밖 완료는 없다 — end 시점 RUNNING 은
    미완료(검열·backlog). 비용 적분(end 절단)과 완주율이 같은 시간창을 본다."""
    sim = _v2_sim()
    sim.end = 3600.0                                     # 평가창 강제 축소 → 미완료 유발
    r = _run_sf(sim)
    assert sim.clock <= sim.end + 1e-6                   # 시계가 평가창을 넘지 않는다
    done = [j for j in sim.jobs.values() if j.status.name == "DONE"]
    assert all(j.service_end <= sim.end + 1e-6 for j in done)
    assert r["completion_rate"] < 1.0 and r["backlog"] > 0
    tl = sim.time_ledger
    assert all(rec.job_done <= sim.end + 1e-6
               for rec in tl.records.values() if rec.job_done is not None)
    # 적분 등식이 축소창에서도 성립 (표본 C ≤ end ⇒ 표본합 == 적분)
    end = tl.closed_end_s
    expect = 0.0
    for rec in tl.records.values():
        if rec.block_arrival is None or rec.block_arrival >= end:
            continue
        expect += min(rec.job_done if rec.job_done is not None else end, end) - rec.block_arrival
    assert abs(tl.block_area_s - expect) < 1e-3


def test_unfinished_backlog_counts_running():
    """외부감사 2차: 종료시점 RUNNING 도 backlog — unfinished_backlog 가 DONE(·CANCELLED)
    외 전부를 센다 (runner/direct_job_env 소비 경로 정합)."""
    from yard_rl.domain.enums import JobStatus
    sim = _v2_sim()
    sim.end = 3600.0
    r = _run_sf(sim)
    assert sim.unfinished_backlog() == r["backlog"]      # 보고 정의와 일치 (DONE 외 전부)
    done_jobs = [j for j in sim.jobs.values() if j.status == JobStatus.DONE]
    assert done_jobs
    before = sim.unfinished_backlog()
    done_jobs[0].status = JobStatus.RUNNING              # RUNNING 이 실제로 세어지는지
    assert sim.unfinished_backlog() == before + 1
    done_jobs[0].status = JobStatus.DONE


def test_v2_metrics_reported_and_named_separately():
    r = _run_sf(_v2_sim())
    for k in ("block_turntime_mean_min", "block_turntime_p95_min",
              "terminal_turntime_mean_min", "terminal_turntime_p95_min",
              "terminal_truck_area_h", "censored_exposure_h"):
        assert k in r
    # 세 지표 분리: 터미널 턴타임 ≥ 블록 처리시간 ≥ YC 대기 (포함 관계 — 혼용 방지의 산술 근거)
    assert r["terminal_turntime_mean_min"] > r["block_turntime_mean_min"] > r["mean_wait_min"]
