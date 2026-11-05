"""Sampling changes comparisons, never dispatch; known rare work remains eligible."""
from collections import Counter
from types import SimpleNamespace as NS

from yard_rl.v7.crane.sampling import CoverageSampler
from yard_rl.v7.world.domain.enums import JobFlow


def test_sampling_reproducible_covers_cranes_and_retains_representative_comparisons():
    def gc(jid):
        return NS(job_ref=NS(job_id=jid))
    picked, common, rare = [gc(i) for i in ('picked','common','rare')]
    sim = NS(jobs={k:NS(flow=v) for k,v in (
        ('picked',JobFlow.GATE_OUT),('common',JobFlow.GATE_OUT),('rare',JobFlow.VESSEL_LOAD))})
    choices = [(c,picked,[common,rare]) for c in ('A','B')]
    rank = lambda sim,c,g: (0 if g.job_ref.job_id == 'common' else 1,)
    traces=[]
    for _ in range(2):
        sampler=CoverageSampler(9913600)
        traces.append([(c,a.job_ref.job_id) for c,p,a in
                       [sampler.select(sim,'BLOCK',choices,rank) for _ in range(24)]])
    assert traces[0] == traces[1]
    assert Counter(c for c,a in traces[0]) == {'A':12,'B':12}
    assert all(a=='common' for c,a in traces[0][::2])
    assert any(a=='rare' for c,a in traces[0][1::2])
