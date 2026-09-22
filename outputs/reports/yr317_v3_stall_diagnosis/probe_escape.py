"""Why did the existing escape-bay logic emit nothing in the captured stall? (YR-317-k)

Read-only. Loads the Y17 snapshot taken from the frozen 7e2fb14 run and calls the
candidate generator's escape path directly, printing every decision point.
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
from yard_rl.v3.world.integrated.reservation import Corridor

SNAP = Path('/mnt/c/Users/geonu/orca/workspaces/port_reinforcement/강화학습-판매/outputs/reports/'
            'yr317_v3_stall_diagnosis/legacy-probe-21000000-Y17-b/Y17-snapshot.pkl.gz')


def main():
    with gzip.open(SNAP, 'rb') as stream:
        sim = pickle.load(stream)
    gen = CandidateGenerator(config=LEGACY_DEFAULT)
    gap = sim.reservations.safety_gap_bay
    corridors = sim.interference_deadlock_corridors()
    report = dict(safety_gap_bay=gap, corridors=[list(c) for c in corridors], cranes={})
    print('corridors', corridors, 'gap', gap)
    # Corridor.overlaps boundary semantics decide whether an escape target survives.
    probe = [(d, Corridor(4.0, 4.0).overlaps(Corridor(1.0, 2.0), d)) for d in (1.0, 1.9, 2.0, 2.1)]
    report['overlap_boundary_probe'] = probe
    print('overlap probe (distance exactly gap?)', probe)
    for cid in sim.fleet.ids():
        yc, spec = sim.fleet.get(cid), sim.fleet.spec(cid)
        pos = yc.state.position_bay
        entry = dict(position_bay=pos, idle=yc.idle, yielded=yc.yielded,
                     service_bay_min=spec.service_bay_min, service_bay_max=spec.service_bay_max,
                     escape_bays=sorted(gen._escape_bays(sim, cid)), per_corridor=[])
        for lo, hi in corridors:
            blocking = Corridor(pos, pos).overlaps(Corridor(lo, hi), gap)
            row = dict(corridor=[lo, hi], this_crane_blocks=blocking, targets=[])
            for raw in (lo - gap, hi + gap):
                clamped = float(min(max(raw, spec.service_bay_min), spec.service_bay_max))
                row['targets'].append(dict(
                    raw=raw, clamped=clamped, same_as_now=abs(clamped - pos) < 1e-9,
                    still_overlaps=Corridor(clamped, clamped).overlaps(Corridor(lo, hi), gap)))
            entry['per_corridor'].append(row)
        # what the generator finally offers
        items = gen.generate(sim, cid, InformationLevel.PRE_ADVICE).items
        entry['generated'] = [dict(kind=g.kind.value, job=None if g.job_ref is None else g.job_ref.job_id,
                                   feasible=g.feasible, reason=g.mask_reason) for g in items]
        # if an escape target exists, can the engine actually plan and reserve it?
        for target in entry['escape_bays']:
            from yard_rl.v3.world.integrated.jobplan import JobRef
            from yard_rl.v3.world.contract.schema import CandidateKind
            ref = JobRef(job_id=f'PROBE:{cid}:{int(target)}', token=None,
                         kind=CandidateKind.REPOSITION, target_container=None, lane_id=None,
                         eligible_crane_ids=(cid,), is_vessel=False, is_external=False,
                         reposition_target_bay=target)
            plan = sim._plan(cid, ref)
            entry.setdefault('escape_plans', []).append(dict(
                target=target, plan_present=plan is not None,
                corridor=None if plan is None else list(plan.corridor),
                reject=None if plan is None else sim.reservations.reject_reason(sim._reservation(plan))))
        report['cranes'][cid] = entry
        print(cid, json.dumps(entry, ensure_ascii=False, default=str)[:900])
    Path(SNAP.parent / 'escape-probe.json').write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str),
                                                       encoding='utf-8')
    print('written', SNAP.parent / 'escape-probe.json')


if __name__ == '__main__':
    main()
