"""YR-056 COORD 협조 feature 계약 테스트 (itc-v3).

torch 불필요 (capture·record 경로) — 양 환경에서 실행.
seed 310000 대역: 실험 판정용으론 소각, 기제 검증용 사용 가능 (yr045 하네스 관례).
"""
from yard_rl.contract.schema import SCHEMA, SCHEMA_VERSION, AblationGroup
from yard_rl.domain.enums import InformationLevel
from yard_rl.integrated import (TerminalSimulator, build_integrated_profile,
                                build_minimal_terminal_scenario)
from yard_rl.integrated.adapter import capture, record_episode
from yard_rl.integrated.scenario_gen import generate_terminal_scenario

PROF = build_integrated_profile()
PA = InformationLevel.PRE_ADVICE
_COORD_YC = ("neighbor_busy_kind", "neighbor_busy_target_bay",
             "neighbor_available_in_s", "recent_yield_count")


def _records(seed=None, ablation_off=()):
    sc = (generate_terminal_scenario(PROF, seed) if seed is not None
          else build_minimal_terminal_scenario())
    sim = TerminalSimulator(PROF, sc)
    return sim, record_episode(sim, info_level=PA, episode_id=f"T-{seed}",
                               ablation_off=ablation_off)


def test_schema_v3_coord_fields():
    # v5 (YR-088) 는 vessel flow_margin 만 추가 — COORD 채널 계약(이 파일의 대상)은 불변.
    assert SCHEMA_VERSION == "itc-v5"
    for name in _COORD_YC:
        assert SCHEMA.spec("yc", name).ablation is AblationGroup.COORD
    assert SCHEMA.spec("candidate", "contention_risk").ablation is AblationGroup.COORD
    # 채널 순서: 신규 필드는 각 그룹 말미 (기존 채널 index 불변)
    assert SCHEMA.names("yc")[-4:] == _COORD_YC
    assert SCHEMA.names("candidate")[-1] == "contention_risk"


def test_neighbor_intent_observed_when_neighbor_busy():
    """상대가 실행 중인 결정 시점: busy kind/target known=1, 값이 계획과 일치.

    minimal 시나리오는 결정 4회가 전부 동시(둘 다 idle)라 busy 상대가 없다 —
    엇갈린 완료가 생기는 생성 시나리오로 검증.
    """
    _sim, recs = _records(seed=310000)
    seen_busy = seen_idle = False
    for rec in recs:
        for ob in rec.observations:
            kind, kn, _ = ob.features.channel("neighbor_busy_kind")
            bay, kn_bay, _ = ob.features.channel("neighbor_busy_target_bay")
            avail, kn_av, _ = ob.features.channel("neighbor_available_in_s")
            assert kn == kn_bay          # busy 의도 두 채널은 동시 관측
            if kn:
                seen_busy = True
                assert 0.0 <= kind <= 1.0 and bay >= 0.0
                assert kn_av and avail >= 0.0   # busy 면 가용시각도 관측
            else:
                seen_idle = True
    assert seen_busy, "에피소드 전체에서 busy 상대 관측 0 — 산출 배선 의심"
    assert seen_idle, "idle 상대의 결측(known=0) 경로 미관측"


def test_contention_risk_flags_shared_jobs():
    """경합 시나리오: 공유 가능 작업 후보에 contention_risk ≥ 0.5 가 발현."""
    _sim, recs = _records(seed=310003)
    top = 0.0
    for rec in recs:
        for ob in rec.observations:
            for c in ob.candidates.items:
                v, kn, _ = c.features.channel("contention_risk")
                if kn:
                    top = max(top, v)
    assert top >= 0.5, f"경합 신호 미발현 (max={top})"


def test_recent_yield_count_matches_resolution_log():
    """counter == resolution_log 의 LOST_CONTENTION 집계 (배관 정합) + 발현 확인."""
    found = False
    for seed in (310000, 310001, 310002, 310003):
        sim, _recs = _records(seed=seed)
        tally: dict[str, int] = {}
        for res in sim.resolution_log:
            for r in res.resolutions:
                if r.yield_reason == "LOST_CONTENTION":
                    tally[r.crane_id] = tally.get(r.crane_id, 0) + 1
        for yc in sim.fleet.all():
            assert yc.recent_yield_count == tally.get(yc.crane_id, 0)
        if any(tally.values()):
            found = True
            break
    assert found, "4개 seed 에서 경합 양보 0건 — 시나리오·배관 재확인 필요"


def test_coord_ablation_restores_v2_information():
    """ablation_off=(COORD,) → 5채널 전부 known=0 (v2 와 동일 정보량)."""
    _sim, recs = _records(seed=310003, ablation_off=(AblationGroup.COORD,))
    for rec in recs:
        assert "COORD" in rec.audit.ablation_off
        for ob in rec.observations:
            for name in _COORD_YC:
                _v, kn, _a = ob.features.channel(name)
                assert not kn, f"{name} 이 ablation off 인데 관측됨"
            for c in ob.candidates.items:
                _v, kn, _a = c.features.channel("contention_risk")
                assert not kn
