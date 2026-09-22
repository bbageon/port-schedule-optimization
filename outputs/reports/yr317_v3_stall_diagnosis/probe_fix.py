"""Does the existing feasible-first generator already break the legacy deadlock? (YR-317-k)

Read-only. Same captured Y17 snapshot, three generators compared:
  1. the legacy default generator (what produced the deadlock),
  2. the feasible-first generator written for the vertical layout,
  3. the legacy generator with a larger candidate budget (to separate "pruned out"
     from "never generated").
"""
import gzip
import json
import pickle
import sys
from pathlib import Path

SOURCE = Path('/home/geonu/yr317-v3-independent-7e2fb14')
sys.path.insert(0, str(SOURCE / 'src'))

from yard_rl.v3.world.domain.enums import InformationLevel
from yard_rl.v3.world.integrated.candidates import CandidateGenerator
from yard_rl.v3.world.integrated.policy_config import LEGACY_DEFAULT


class FeasibleCandidateGenerator(CandidateGenerator):
    """Copy of src/yard_rl/v3/layouts/candidates.py, inlined because the frozen
    7e2fb14 checkout predates that module. Same scores and tie-breaks; the
    nonmandatory budget simply prefers feasible work."""

    def _prune(self, raw):
        budget = self.k_max - 1
        mandatory = [c for c in raw if c.mandatory]
        rest = [c for c in raw if not c.mandatory]
        rest.sort(key=lambda c: (not c.feasible, -c.score) + self._order_key(c))
        kept = mandatory + rest[:max(0, budget - len(mandatory))]
        if not any(c.feasible for c in kept):
            best = next((c for c in rest if c.feasible), None)
            if best is not None:
                kept.append(best)
        return kept

SNAP = Path('/mnt/c/Users/geonu/orca/workspaces/port_reinforcement/강화학습-판매/outputs/reports/'
            'yr317_v3_stall_diagnosis/legacy-probe-21000000-Y17-b/Y17-snapshot.pkl.gz')


def summarise(gen, sim, cid):
    items = gen.generate(sim, cid, InformationLevel.PRE_ADVICE).items
    kinds = {}
    for g in items:
        key = g.kind.value
        kinds.setdefault(key, {'n': 0, 'feasible': 0})
        kinds[key]['n'] += 1
        kinds[key]['feasible'] += int(bool(g.feasible))
    executable = [dict(kind=g.kind.value, job=None if g.job_ref is None else g.job_ref.job_id,
                       target_bay=getattr(g.job_ref, 'reposition_target_bay', None) if g.job_ref else None)
                  for g in items if g.feasible and g.kind.value != 'WAIT']
    return dict(total=len(items), by_kind=kinds, executable=executable)


def main():
    with gzip.open(SNAP, 'rb') as stream:
        sim = pickle.load(stream)
    # The escape hook clears yields before opening the decision, so reproduce that state.
    cleared = pickle.loads(pickle.dumps(sim))
    for c in cleared.fleet.all():
        c.yielded = False
    variants = {
        'legacy_default': CandidateGenerator(config=LEGACY_DEFAULT),
        'feasible_first': FeasibleCandidateGenerator(config=LEGACY_DEFAULT),
        'legacy_budget_200': CandidateGenerator(config=LEGACY_DEFAULT, k_max=200),
    }
    report = {}
    for name, gen in variants.items():
        report[name] = {cid: summarise(gen, sim, cid) for cid in sim.fleet.ids()}
        report[name + '__yields_cleared'] = {cid: summarise(gen, cleared, cid) for cid in cleared.fleet.ids()}
        for cid, row in report[name + '__yields_cleared'].items():
            print(name, '(yields cleared)', cid, 'total', row['total'], 'by_kind', row['by_kind'],
                  'executable', row['executable'][:3], flush=True)
        for cid, row in report[name].items():
            print(name, cid, 'total', row['total'], 'by_kind', row['by_kind'],
                  'executable', row['executable'][:3], flush=True)
    report['reading'] = (
        'If legacy_default offers only WAIT while feasible_first or legacy_budget_200 offers an '
        'executable REPOSITION, the escape candidate was generated and then pruned out by the '
        'candidate budget, not missing from the generator.')
    Path(SNAP.parent / 'fix-probe.json').write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str),
                                                    encoding='utf-8')
    print('written', SNAP.parent / 'fix-probe.json')


if __name__ == '__main__':
    main()
