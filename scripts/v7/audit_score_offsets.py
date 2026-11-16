"""Post-hoc score-contract counterexample, not a terminal performance experiment."""
from pathlib import Path
from types import SimpleNamespace as NS

from context_learning import OUT
from six_step_learning import sha
from yard_rl.v7.world.contract.schema import CandidateKind
from yard_rl.v7.world.integrated.resolver import CentralResolver
from yard_rl.integrated.repro import write_result


def resolve(offset):
    scores={('C1','A'):0.,('C1','B'):1.,('C2','A'):2.,('C2','B'):10.}
    scores={key:value+offset.get(key[0],0.) for key,value in scores.items()}
    class Preference:
        def rank(self,sim,cid,gc):return (0,scores[cid,gc.job_ref.job_id],gc.job_ref.job_id)
    batches={c:NS(items=[NS(candidate_id=i,kind=CandidateKind.SERVE,feasible=True,mandatory=False,
                           job_ref=NS(job_id=j,token=j)) for i,j in enumerate(('A','B'))])
             for c in ('C1','C2')}
    # The oracle admits all distinct-job assignments; only cross-crane job contention remains.
    sim=NS(dry_run_commit=lambda trial:NS(plans={c:NS() for c,ref in trial.items() if ref is not None}))
    result=CentralResolver(Preference()).resolve(sim,NS(crane_ids=('C1','C2'),time=0.),batches)
    chosen={r.crane_id:batches[r.crane_id].items[r.chosen_candidate_id].job_ref.job_id for r in result.resolutions}
    gaps={c:scores[c,'A']-scores[c,'B'] for c in ('C1','C2')}
    return dict(scores={f'{c}/{j}':s for (c,j),s in scores.items()},gaps=gaps,chosen=chosen)


def main():
    base=resolve({});shift=resolve({'C1':100.})
    assert base['gaps']==shift['gaps'] and base['chosen']!=shift['chosen']
    sources=[Path(__file__),Path('src/yard_rl/v7/world/integrated/resolver.py'),Path('scripts/v7/context_learning.py')]
    write_result(OUT/'score-offset-audit.json',dict(base=base,shift=shift,
        identical_within_crane_difference_loss=True,different_joint_assignment=True,
        scope='Post-hoc mathematical counterexample using the real resolver with a toy all-feasible oracle. '
              'Not a measurement of this effect in the trained networks; no policy or experiment settings changed.',
        inference='Within-crane score differences alone do not identify cross-crane offsets consumed by a global greedy resolver.',
        sources={p.as_posix():sha(p) for p in sources}))
    print(base,shift)


if __name__=='__main__':main()
