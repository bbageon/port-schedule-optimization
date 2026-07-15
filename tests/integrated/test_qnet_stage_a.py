"""YR-039 Stage A 계약 테스트 — 시나리오 생성기·인코딩·Q망·QPreference."""
import pytest

torch = pytest.importorskip("torch")  # optional [rl]

from yard_rl.contract import CandidateKind
from yard_rl.domain.enums import InformationLevel
from yard_rl.integrated import (BaselinePreference, CandidateGenerator,
                                CentralResolver, TerminalSimulator,
                                record_episode)
from yard_rl.integrated.adapter import capture
from yard_rl.integrated.encoding import (DecisionEncoding, encode_observation,
                                         encoding_dims, fv_to_vec)
from yard_rl.integrated.fixtures import build_integrated_profile
from yard_rl.integrated.qnet import (CandidateQNet, QNetConfig, QPreference,
                                     score_decision)
from yard_rl.integrated.scenario_gen import (TerminalGenParams,
                                             generate_terminal_scenario)

PROF = build_integrated_profile()
LEVEL = InformationLevel.PRE_ADVICE
PARAMS = TerminalGenParams(n_external=8, n_vessels=2, vessel_moves=6,
                           horizon_s=7_200.0, drain_window_s=3_600.0)


def _scenario(seed=300_001):
    return generate_terminal_scenario(PROF, seed, PARAMS)


# ------------------------------------------------------------- 시나리오 생성기
def test_scenario_generator_deterministic():
    a, b = _scenario(), _scenario()
    assert a.scenario_id == b.scenario_id
    assert [j.job_id for j in a.jobs] == [j.job_id for j in b.jobs]
    assert {c: (v.bay, v.row, v.tier) for c, v in a.containers.items()} == \
           {c: (v.bay, v.row, v.tier) for c, v in b.containers.items()}
    c = generate_terminal_scenario(PROF, 300_002, PARAMS)
    assert [j.job_id for j in a.jobs] != [j.job_id for j in c.jobs] or \
           a.containers.keys() != c.containers.keys()


def test_generated_scenario_completes_episode_with_contract():
    """record_episode 완주 — 매 결정 validate_all·invariant 통과 (엔진 계약)."""
    sim = TerminalSimulator(PROF, _scenario(), check_invariants=True)
    records = record_episode(sim, info_level=LEVEL, episode_id="gen-e2e")
    assert records and records[-1].terminal
    assert all(r.cost.total_normalized >= 0.0 for r in records)


# ---------------------------------------------------------------- 인코딩
def test_encoding_masks_and_dims():
    sim = TerminalSimulator(PROF, _scenario(), check_invariants=True)
    sim.info_level = LEVEL
    dp = sim.run_until_decision()
    state, obs, _gen_by = capture(sim, dp.crane_ids, LEVEL, "enc", 0)
    enc = encode_observation(state, obs[0])
    fg, fy, fq, fc = encoding_dims(enc)
    assert fg == 2 * len(state.features.names)
    assert fc == 2 * len(obs[0].candidates.items[0].features.names)
    # 결측 중화: value*known 채널에서 known=0 위치는 0, 지시자 채널은 {0,1}
    fv = obs[0].candidates.items[0].features
    vec = fv_to_vec(fv)
    n = len(fv.names)
    for i, k in enumerate(fv.known):
        if not k:
            assert vec[i] == 0.0
        assert vec[n + i] in (0.0, 1.0)
    # 패딩·infeasible 은 selectable=False
    cs = obs[0].candidates
    for i, (p, f) in enumerate(zip(cs.pad_mask, cs.feasible_mask)):
        assert enc.selectable[i] == (p and f)


# ------------------------------------------------------------- Q망·Preference
def _one_encoding() -> DecisionEncoding:
    sim = TerminalSimulator(PROF, _scenario(), check_invariants=True)
    sim.info_level = LEVEL
    dp = sim.run_until_decision()
    state, obs, _ = capture(sim, dp.crane_ids, LEVEL, "q", 0)
    return encode_observation(state, obs[0])


def test_zero_init_scores_all_zero_and_dueling_shape():
    enc = _one_encoding()
    dims = encoding_dims(enc)
    for dueling in (False, True):
        torch.manual_seed(0)
        net = CandidateQNet(dims, QNetConfig(dueling=dueling))
        scores = score_decision(net, enc)
        assert scores and all(v == 0.0 for v in scores.values())  # §1 계약
        assert set(scores) <= set(enc.candidate_ids)


def test_untrained_qpreference_equals_baseline_episode():
    """미학습 QPreference resolver ≡ BaselinePreference resolver (전 결정 일치)."""
    torch.manual_seed(0)
    enc = _one_encoding()
    net = CandidateQNet(encoding_dims(enc))
    gen = CandidateGenerator()

    def drive(preference, with_net: bool):
        sim = TerminalSimulator(PROF, _scenario(), check_invariants=True)
        sim.info_level = LEVEL
        resolver = CentralResolver(preference)
        picks = []
        dp = sim.run_until_decision()
        while dp is not None:
            gen_by = {c: gen.generate(sim, c, LEVEL) for c in dp.crane_ids}
            if with_net:
                state, obs, _ = capture(sim, dp.crane_ids, LEVEL, "cmp", 0)
                scores = {}
                for ob in obs:
                    e = encode_observation(state, ob)
                    scores.update({(ob.crane_id, cid): q
                                   for cid, q in score_decision(net, e).items()})
                preference.set_scores(scores)
            resn = resolver.resolve(sim, dp, gen_by)
            picks.append(tuple((r.crane_id, r.action.value, r.chosen_token)
                               for r in resn.resolutions))
            resolver.apply(sim, resn, gen_by)
            dp = sim.run_until_decision()
        return picks

    base = drive(BaselinePreference(), with_net=False)
    qp = drive(QPreference(), with_net=True)
    assert base == qp


def test_network_learns_direction():
    """단일 SGD 스텝으로 선택 후보의 Q 가 목표 방향으로 이동 (학습 가능성)."""
    enc = _one_encoding()
    net = CandidateQNet(encoding_dims(enc))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    t = lambda x: torch.tensor([x], dtype=torch.float32)  # noqa: E731
    cand = torch.tensor([list(map(list, enc.cand))], dtype=torch.float32)
    sel = torch.tensor([list(enc.selectable)], dtype=torch.bool)
    idx = enc.selectable.index(True)
    for _ in range(60):
        q = net(t(list(enc.g)), t(list(enc.yc)), t(list(enc.queue)), cand, sel)
        loss = (q[0, idx] - 2.5) ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(net(t(list(enc.g)), t(list(enc.yc)), t(list(enc.queue)),
                     cand, sel)[0, idx]) == pytest.approx(2.5, abs=0.3)
