"""Historical label recovery reads state before the selected job is applied."""
from pathlib import Path

from yard_rl.v7.crane.features import crane_features
from yard_rl.v7.world.contract.schema import CandidateKind
from test_candidate_recall import state, generate


def test_capture_before_dispatch_and_validate_choice_after_dispatch(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts/v7'))
    import count_transform_study as study
    sim,dp=state()
    gb=generate(sim,dp,None)
    gc=next(g for g in gb['YC-A'].items if g.kind==CandidateKind.SERVE)
    full=crane_features(sim,'YC-A',gc,feature_mode='feasible',include_flow=True)
    sample=dict(decision_key=[round(sim.clock,6),'Y01','YC-A'],job=gc.job_ref.job_id,
                row=full[:9],is_store=bool(full[9]),action='PICKED')
    probe=study.CountProbe([sample])
    probe.block_names={id(sim):'Y01'}
    probe.before_select(sim,dp,gb,None)
    def prohibit_post_apply_read(*args,**kwargs):
        raise AssertionError('post-dispatch features must not replace historical inputs')
    monkeypatch.setattr(study,'crane_features',prohibit_post_apply_read)
    probe.decision(sim,dp,gb,{'YC-A':gc},0,0)
    assert probe.recovered=={0:gc.feasible_serve_count}
