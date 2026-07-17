"""YR-050 — ETA 선제작업 결정 시점·연착 신호 계약.

배경 (YR-048 적대 리뷰 파생): 엔진은 실행 가능한 SERVE 가 있을 때만 결정을 만들어
ETA 가 보이는 한산기·첫 트럭 도착 전에는 선제 재조작(PRE_REHANDLE)을 고를 수 없었다.
또 ETA 가 지났는데 트럭이 안 온 구간에서 predicted_arrival_gap_s 음수가 0 으로 잘려
"지금 도착 예정"과 "연착"이 구분 불가였다. 고정하는 계약:
- 결정 개방 = SERVE ∪ ETA 주도 선제 재조작 기회(PRE_ADVICE 한정, provided_eta·결정
  지평만 사용 — 도착 진실 미열람). wake 1회당 크레인별 1회 질문(armed) — 기회 잔존이
  결정을 증식시키지 않는다 (재질문 무제한이던 1차 구현은 REPO 88% 퇴화 실측으로 폐기).
- 연착 신호 = predicted_arrival_gap_s 부호 보존 (itc-v2, 음수 = ETA 경과·미도착).
- 낮은 정보수준(GATE_IN·BLOCK_ARRIVAL)은 wake 완전 비활성 — 이벤트 스트림 불변.
"""
from yard_rl.contract import build_feature_vector
from yard_rl.contract.schema import CandidateKind
from yard_rl.domain.enums import (ContainerSize, InformationLevel, JobFlow, JobStatus,
                                  LoadStatus)
from yard_rl.domain.models import Container, Job
from yard_rl.integrated import (BaselinePreference, CandidateGenerator, CentralResolver,
                                CraneAssignment, ReferenceDispatcher, TerminalScenario,
                                TerminalSimulator, build_integrated_profile, record_episode)

PROF = build_integrated_profile()          # 결정 지평 1800s · YC 2기(bay 1~40 공유)
PA = InformationLevel.PRE_ADVICE
GEN = CandidateGenerator()


def _c(cid, bay, row, tier):
    return Container(container_id=cid, size=ContainerSize.FT40, load_status=LoadStatus.FULL,
                     block="B1", bay=bay, row=row, tier=tier)


def _out(jid, target, arrival, eta):
    return Job(job_id=jid, flow=JobFlow.GATE_OUT, release_time=0.0,
               actual_gate_in=max(0.0, arrival - 600.0), actual_block_arrival=arrival,
               provided_eta=eta, target_container=target)


def _sc(jobs, containers):
    return TerminalScenario(scenario_id="yr050", seed=0, horizon_s=7200.0,
                            drain_window_s=1800.0, containers=containers, jobs=jobs,
                            vessels=[], injected_events=[])


def _blocked_target_sc(eta, arrival):
    """반출 대상 C-T(bay5) 위에 blocker C-B → ETA 만 보이면 선제 재조작 기회."""
    return _sc([_out("J-OUT-T", "C-T", arrival, eta)],
               {"C-T": _c("C-T", 5, 1, 1), "C-B": _c("C-B", 5, 1, 2)})


def _drive(sim):
    """중앙 resolver 로 완주 — [(결정시각, (행동, ...)), ...] 반환."""
    r = CentralResolver(BaselinePreference())
    log = []
    while True:
        dp = sim.run_until_decision()
        if dp is None:
            return log
        gen_by = {c: GEN.generate(sim, c, sim.info_level) for c in dp.crane_ids}
        resn = r.resolve(sim, dp, gen_by)
        r.apply(sim, resn, gen_by)
        log.append((dp.time, tuple(cr.action.value for cr in resn.resolutions)))


# ------------------------------------------------------ 결정 개방 + 첫 트럭 혜택
def test_pre_rehandle_decided_before_first_arrival_and_first_truck_benefits():
    """첫 실제 도착 전에 선제 재조작이 결정·완료되고, 도착 후 SERVE 는 재조작 0."""
    sim = TerminalSimulator(PROF, _blocked_target_sc(eta=900.0, arrival=1500.0),
                            info_level=PA)
    log = _drive(sim)
    pre_times = [t for (t, acts) in log if "PRE_REHANDLE" in acts]
    assert pre_times and pre_times[0] < 1500.0, \
        "ETA 만 보이는 구간에서 선제 재조작 결정이 열리지 않음 (YR-050 회귀)"
    j = sim.jobs["J-OUT-T"]
    assert j.status == JobStatus.DONE
    assert j.rehandle_count == 0            # 혜택 경로: 도착 후 서비스는 재조작 없음
    assert sim.kpis.rehandle_count == 1     # 재조작은 도착 전 선제분 1건뿐


def test_busy_at_wake_crane_stays_armed_until_idle():
    """wake 순간 바쁜 크레인은 armed 가 유지되어, 유휴화 시점에 선제 결정이 열린다.

    기하: 바쁜 대상은 bay 3 (corridor 1..3 이 선제 대상 bay 5 를 안 덮음). wake(50s)는
    서비스 중에 발생 — 그 시점 idle 인 크레인의 선제는 corridor 간섭으로 막히고(WAIT),
    작업하던 크레인이 유휴화되는 시점에 armed 잔존으로 선제 결정이 열린다.
    """
    containers = {"C-T": _c("C-T", 5, 1, 1), "C-B": _c("C-B", 5, 1, 2),
                  "C-X": _c("C-X", 3, 2, 1)}
    jobs = [_out("J-BUSY-X", "C-X", 0.0, None),
            _out("J-OUT-T", "C-T", 2100.0, 1850.0)]    # wake = 1850-1800 = 50 (서비스 중)
    sim = TerminalSimulator(PROF, _sc(jobs, containers), info_level=PA)
    log = _drive(sim)
    pre_times = [t for (t, acts) in log if "PRE_REHANDLE" in acts]
    assert pre_times and 50.0 < pre_times[0] < 2100.0, \
        f"바쁜 크레인이 유휴화될 때 선제 결정이 안 열림: {log}"
    assert sim.jobs["J-OUT-T"].rehandle_count == 0


def test_without_eta_no_decision_before_arrival():
    """counterfactual — ETA 없으면 첫 도착 전 결정 자체가 없고 재조작이 서비스에 붙는다."""
    sim = TerminalSimulator(PROF, _blocked_target_sc(eta=None, arrival=1500.0),
                            info_level=PA)
    log = _drive(sim)
    assert log and all(t >= 1500.0 for (t, _) in log)
    assert sim.jobs["J-OUT-T"].rehandle_count == 1


# ------------------------------------------------------ 정보 경계 (누출 0)
def test_low_info_levels_bit_identical_with_or_without_eta():
    """GATE_IN·BLOCK_ARRIVAL 에서는 wake 완전 비활성 — ETA 유무가 거동을 못 바꾼다."""
    for level in (InformationLevel.GATE_IN, InformationLevel.BLOCK_ARRIVAL):
        runs = []
        for eta in (900.0, None):
            sim = TerminalSimulator(PROF, _blocked_target_sc(eta=eta, arrival=1500.0),
                                    info_level=level)
            assert sim._next_wake_time() is None       # 낮은 수준은 wake 조회조차 없음
            ReferenceDispatcher().run(sim)
            runs.append((sim.event_stream_hash(), len(sim.event_log)))
        assert runs[0] == runs[1], f"{level}: ETA 가 낮은 정보수준 거동을 바꿈 (누출)"


def test_wake_derives_from_provided_eta_never_actual_arrival():
    """도착 진실이 달라도 provided_eta 가 같으면 도착 전 결정열이 동일 (누출 0)."""
    logs, wakes = [], []
    for arrival in (1500.0, 2200.0):
        sim = TerminalSimulator(PROF, _blocked_target_sc(eta=900.0, arrival=arrival),
                                info_level=PA)
        wakes.append(list(sim._eta_wakes))
        logs.append([(t, a) for (t, a) in _drive(sim) if t < 1500.0])
    assert wakes[0] == wakes[1] == [(0.0, "J-OUT-T")]   # max(0, 900-1800)
    assert logs[0] == logs[1]


def test_wake_schedule_gate_out_with_eta_only():
    """wake 는 GATE_OUT+ETA 만: GATE_IN 은 선제 재조작 대상이 없어 시드하지 않는다."""
    gate_in = Job(job_id="J-IN-X", flow=JobFlow.GATE_IN, release_time=0.0,
                  actual_gate_in=100.0, actual_block_arrival=700.0, provided_eta=650.0,
                  inbound_size=ContainerSize.FT40, inbound_load=LoadStatus.FULL)
    sc = _sc([_out("J-OUT-T", "C-T", 3000.0, 2500.0), gate_in],
             {"C-T": _c("C-T", 5, 1, 1), "C-B": _c("C-B", 5, 1, 2)})
    sim = TerminalSimulator(PROF, sc, info_level=PA)
    assert sim._eta_wakes == [(700.0, "J-OUT-T")]       # 2500-1800; GATE_IN 부재


# ------------------------------------------------------ 재결정 유한성·결정론
def test_all_wait_terminates_without_same_time_redecision_loop():
    """wake·armed 는 1회성 — 전 크레인이 계속 WAIT 해도 유한 결정으로 완주."""
    sim = TerminalSimulator(PROF, _blocked_target_sc(eta=900.0, arrival=1500.0),
                            info_level=PA)
    n, prev = 0, None
    while True:
        dp = sim.run_until_decision()
        if dp is None:
            break
        n += 1
        assert n < 50, "결정 폭주 — 재질문 무한루프 의심"
        assert prev is None or dp.time > prev, \
            f"같은 시각 {dp.time} 재결정 — wake 1회성 위반"
        prev = dp.time
        sim.commit_decisions([CraneAssignment(c, CandidateKind.WAIT)
                              for c in dp.crane_ids])
    assert sim.terminal


def test_deterministic_replay():
    hashes, logs = set(), []
    for _ in range(2):
        sim = TerminalSimulator(PROF, _blocked_target_sc(eta=900.0, arrival=1500.0),
                                info_level=PA)
        logs.append(_drive(sim))
        hashes.add(sim.event_stream_hash())
    assert len(hashes) == 1 and logs[0] == logs[1]


# ------------------------------------------------------ 연착(음수 gap) 신호
def test_negative_gap_reaches_validated_record():
    """ETA 경과·미도착 구간의 선제 재조작 후보가 음수 gap 으로 레코드에 실린다.

    t=0 에 두 크레인이 바쁜 SERVE 로 소진(선제 후보는 +60 으로만 노출)된 뒤, 미끼 트럭
    도착이 여는 SERVE 결정(t≈250 또는 첫 유휴화 시점)에서 J-OUT-T 의 gap = 60 − now < 0.
    record_episode 가 매 결정 validate_all 을 통과시키므로 음수 값이 계약 위반 없이
    저장됨을 함께 증명한다 (v1 은 clip_lo=0 절단으로 이 값이 0 이 됐다).
    """
    containers = {"C-T": _c("C-T", 5, 1, 1), "C-B": _c("C-B", 5, 1, 2),
                  "C-D": _c("C-D", 20, 2, 1),
                  "C-X": _c("C-X", 30, 2, 1), "C-Y": _c("C-Y", 35, 3, 1)}
    jobs = [_out("J-BUSY-X", "C-X", 0.0, None), _out("J-BUSY-Y", "C-Y", 0.0, None),
            _out("J-DECOY", "C-D", 250.0, 250.0),
            _out("J-OUT-T", "C-T", 900.0, 60.0)]
    sim = TerminalSimulator(PROF, _sc(jobs, containers), info_level=PA)
    recs = record_episode(sim, ReferenceDispatcher(), info_level=PA,
                          episode_id="YR050-NEG")
    gaps = [c.features.value_of("predicted_arrival_gap_s")
            for r in recs for ob in r.observations for c in ob.candidates.items
            if c.kind == CandidateKind.PRE_REHANDLE
            and c.features.known_of("predicted_arrival_gap_s")]
    assert gaps, "선제 재조작 후보가 레코드에 없음"
    assert min(gaps) < 0.0, f"음수 연착 gap 이 보존되지 않음: {gaps}"


def test_schema_preserves_negative_gap_and_gates_by_level():
    raw = {"action_kind_idx": 1 / 3, "is_external": 1.0, "is_vessel": 0.0,
           "predicted_arrival_gap_s": -120.0, "reach_s": 10.0,
           "expected_service_time_s": 50.0, "expected_handling_count": 1.0,
           "blocker_count": 1.0, "expected_rehandle_time_s": 40.0, "end_bay": 5.0,
           "lane_congestion_local": 0.0, "interference_penalty_s": 0.0,
           "resequence_count": 0.0}
    pa = build_feature_vector("candidate", raw, now=500.0, info_level=PA)
    assert pa.known_of("predicted_arrival_gap_s") is True
    assert pa.value_of("predicted_arrival_gap_s") == -120.0    # 절단 없음 (itc-v2)
    ba = build_feature_vector("candidate", raw, now=500.0,
                              info_level=InformationLevel.BLOCK_ARRIVAL)
    assert ba.known_of("predicted_arrival_gap_s") is False     # 정보 게이트 유지
