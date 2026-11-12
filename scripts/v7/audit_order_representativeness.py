"""Reproducible aggregate audit of private order coverage and synthetic demand.

No vehicle/container identifiers or raw source rows are published. Timestamped
snapshots, entered truck visits, and generated jobs remain separate populations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from calibrate_order_information import read, stamp, stats, save

OUT = Path('outputs/reports/yr332_order_representativeness')
INFORMATION = Path('outputs/reports/yr332_real_order_information/profile.json')
PREREG = Path('.claude/docs/dashboard-task-specs/YR-332-aj-observed-order-representativeness.md')
LOADS = (3500, 7500, 15000)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def block_prefix(value):
    """Only documented-looking location prefixes; no guessed block mapping."""
    m = re.match(r'^([A-Z]\d{2})', value or '')
    return m.group(1) if m else None


def distribution(rows):
    """Each row: day, seconds after local midnight, flow, optional block."""
    by_day = defaultdict(Counter)
    by_hour = defaultdict(Counter)
    blocks = Counter()
    day_times = defaultdict(list)
    minute_counts = defaultdict(Counter)
    for r in rows:
        t, day, flow = r['second'], r['day'], r['flow']
        assert 0 <= t < 86400, r
        by_day[day][flow] += 1
        by_hour[flow][int(t // 3600)] += 1
        day_times[(day, flow)].append(t)
        day_times[(day, 'ALL')].append(t)
        minute_counts[(day, int(t // 3600))][int(t % 3600 // 60)] += 1
        if r.get('block'):
            blocks[r['block']] += 1
    flow_counts = Counter(r['flow'] for r in rows)
    all_hours = Counter()
    for v in by_hour.values():
        all_hours.update(v)
    gaps = defaultdict(list)
    for (_, flow), ts in day_times.items():
        ts.sort()
        gaps[flow].extend(b-a for a, b in zip(ts, ts[1:]))
    minute_dispersion = []
    for v in minute_counts.values():
        values = [v[i] for i in range(60)]
        mean = statistics.mean(values)
        if mean:
            minute_dispersion.append(statistics.pvariance(values) / mean)
    known_blocks = sum(blocks.values())
    return dict(n=len(rows), flow_counts=dict(sorted(flow_counts.items())),
        by_day={k: dict(v) for k, v in sorted(by_day.items())},
        hour_counts={k: [v[i] for i in range(24)] for k, v in sorted(by_hour.items())},
        all_hour_counts=[all_hours[i] for i in range(24)],
        night_22_to_06_share=sum(all_hours[i] for i in (22, 23, 0, 1, 2, 3, 4, 5))/len(rows) if rows else None,
        busiest_hour=max(range(24), key=lambda i: all_hours[i]) if rows else None,
        peak_hour_share=max(all_hours.values())/len(rows) if rows else None,
        block_counts=dict(sorted(blocks.items())), known_block_n=known_blocks,
        block_hhi=sum((v/known_blocks)**2 for v in blocks.values()) if known_blocks else None,
        interarrival_s={k: stats(v) for k, v in gaps.items()},
        minute_dispersion_within_hour=stats(minute_dispersion),
        definition='Local clock; interarrival gaps exclude cross-date intervals; minute dispersion is variance/mean of 60 minute bins within each nonempty hour.')


def reservation_cohorts(visits, orders):
    by_doc = defaultdict(list)
    for r in orders:
        notice, reserved = stamp(r['log_time']), stamp(r['in_out_reserve_time'])
        if r['copino_err_type'] == 'NM' and notice and reserved and reserved.year >= 2000:
            by_doc[r['doc_key']].append((notice, reserved, r['in_out']))
    cohorts, reasons, timing = defaultdict(list), Counter(), defaultdict(list)
    for r in visits:
        a = stamp(r['gate_in_time'])
        if a is None:
            reasons['no_gate_entry'] += 1
            continue
        candidates = sorted({x for k in re.split(r'[|;,]', r['doc_keys'])
            for x in by_doc.get(k.strip(), []) if x[0] <= a and x[2] == r['in_out']})
        reason = ('no_matching_pre_entry_valid_reservation' if not candidates else
            'first_notice_not_observed_with_valid_reservation' if
            candidates[0][0] != stamp(r['copino_first_time']) else 'included')
        reasons[reason] += 1
        cohorts['included' if reason == 'included' else 'excluded'].append(truck_row(r))
        if reason == 'included':
            notice, reserved, flow = candidates[0]
            for key in (f'{flow}:all', f'{flow}:hour-{a.hour:02d}', f'{flow}:date-{a.date()}'):
                timing[key].append(((a-reserved).total_seconds(), (a-notice).total_seconds()))
    return dict(counts=dict(reasons), distributions={k: distribution(v) for k,v in cohorts.items()},
        timing={k: dict(error_s=stats([x[0] for x in v]), lead_s=stats([x[1] for x in v])) for k,v in sorted(timing.items())})


def truck_row(r):
    a = stamp(r['gate_in_time'])
    return dict(day=a.date().isoformat(), second=(a-a.replace(hour=0,minute=0,second=0,microsecond=0)).total_seconds(),
        flow='GATE_IN' if r['in_out']=='IN' else 'GATE_OUT', block=block_prefix(r['gate_in_loc']))


def source_audit(root):
    files = sorted(root.rglob('*.csv'))
    manifest = [dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size, sha256=sha(p)) for p in files]
    previous = json.loads(Path('outputs/reports/yr332_real_order_information/source-manifest.json').read_text(encoding='utf-8'))
    assert manifest == previous, 'Source collection changed; preregister the new period before use.'
    visits = read(sorted(root.glob('기사/방문/*.csv')))
    orders = read(sorted(root.glob('기사/오더/*.csv')))
    entries = [r for r in visits if stamp(r['gate_in_time']) is not None]
    assert all(r['in_out'] in ('IN','OUT') for r in entries)
    assert len(visits) == len({r['visit_id'] for r in visits})
    keys = [(r['truck_hash'],r['con_hash'],r['gate_in_time'],r['in_out']) for r in entries]
    assert len(keys) == len(set(keys))
    intervals, chronology = {}, {}
    for start, end in [('gate_in_time','enter_block_time'),('enter_block_time','job_done_time'),
                       ('gate_in_time','job_done_time'),('job_done_time','gate_out_time')]:
        pairs=[(stamp(r[start]),stamp(r[end])) for r in entries]
        values=[(b-a).total_seconds() for a,b in pairs if a is not None and b is not None]
        intervals[start+'->'+end]=dict(observed=stats(values),missing=len(entries)-len(values),
            nonnegative=stats([x for x in values if x>=0]))
        chronology[start+'->'+end]=sum(x<0 for x in values)
    cohorts=reservation_cohorts(visits,orders)
    assert cohorts['counts']['included']==29285
    assert sum(cohorts['counts'].values())==len(visits)
    data=distribution([truck_row(r) for r in entries])
    start=datetime.fromisoformat(min(data['by_day']))
    end=datetime.fromisoformat(max(data['by_day']))+timedelta(days=1)
    vessel_rows=[]; vessel_seen=set(); vessel_total=Counter(); outside=Counter(); missing=Counter()
    for f in sorted(root.glob('선박/양적하목록/*.csv')):
        for r in read([f]):
            key=(f.stem,r['conNo'],r['loadDiscGbn'])
            assert key not in vessel_seen, 'duplicate call/container/flow'
            vessel_seen.add(key)
            flow={'D':'VESSEL_DISCHARGE','L':'VESSEL_LOAD'}.get(r['loadDiscGbn'])
            assert flow is not None, r['loadDiscGbn']
            vessel_total[flow]+=1
            t=stamp(r.get('atw'))
            if t is None:
                missing[flow]+=1
            elif start<=t<end:
                vessel_rows.append(dict(day=t.date().isoformat(),second=(t-t.replace(hour=0,minute=0,second=0,microsecond=0)).total_seconds(),flow=flow,block=block_prefix(r.get('yardLocation'))))
            else:
                outside[flow]+=1
    snapshots=sorted(root.glob('선박/스케줄/*.csv'),key=lambda p:re.findall(r'\d{8}',p.stem)[-1])
    calls_by_key={};snapshot_rows=0
    for path in snapshots:
        for row in read([path]):
            snapshot_rows+=1
            key=tuple(row[k] for k in ('terminalCode','terminalPortArrivalYear','terminalVesselCode','terminalVoyageNo'))
            calls_by_key[key]=row
    calls=list(calls_by_key.values())
    berth_pairs=[(stamp(r['atb']),stamp(r['atd'])) for r in calls]
    berth_overlap=[]
    for i in range((end-start).days*24):
        lo=start+timedelta(hours=i);hi=lo+timedelta(hours=1)
        berth_overlap.append(sum(a is not None and b is not None and a<hi and b>lo and b>=a for a,b in berth_pairs))
    return dict(manifest=manifest,private_root='오더데이터',visits=len(visits),entered=len(entries),
        status_all=dict(Counter(r['status'] for r in visits)),status_entered=dict(Counter(r['status'] for r in entries)),
        cancellations_with_entry=sum(bool(r['cancel_time']) for r in entries),
        source_channels=dict(Counter(r['source'] for r in entries)),
        intervals_s=intervals,chronology_violations=chronology,
        trucks=data,reservation_selection=cohorts,
        vessels=dict(snapshot_rows=sum(vessel_total.values()),snapshot_flow_counts=dict(vessel_total),
            atw_in_window=distribution(vessel_rows),atw_outside_window=dict(outside),atw_missing=dict(missing),
            berth_overlap_hourly=berth_overlap,berth_overlap_stats=stats(berth_overlap),
            schedule_snapshot_rows=snapshot_rows,unique_calls=len(calls),
            schedule_selection='Latest collected snapshot per terminal/year/vessel/voyage; retrospective diagnostics only.',
            known_atb_atd_pairs=sum(a is not None and b is not None for a,b in berth_pairs),
            limitation='atw is a recorded operation timestamp, not a verified yard-ready or crane-completion timestamp. Snapshot coverage and crane mapping are unconfirmed.'),
        limitations=['Whole-terminal truck coverage is not established by these collection channels.',
            'All 13 days were already viewed; this is development data, not a sealed holdout.',
            'Block prefixes are descriptive only; actual-to-H21 geometry mapping is absent.',
            'Gate-to-completion includes waiting and travel; it is not pure crane service.'])


def synthetic_audit():
    from yard_rl.v7.stage.orders import build_stage
    from yard_rl.v7.world.integrated.profiles import build_h21_profile
    profile=json.loads(INFORMATION.read_text(encoding='utf-8'))
    cases=[('evaluation',d,load,9917600+100*d+10*l) for d in range(3) for l,load in enumerate(LOADS)]
    cases += [('training',m*6+i,load,9917200+100*m+10*i) for m in range(3) for i,load in enumerate(LOADS*2)]
    groups=defaultdict(list);vessels=defaultdict(list);metadata=[]
    for split,day,load,seed in cases:
        built=build_stage(load=load,seed=seed,profile=build_h21_profile(),information_profile=profile)
        assert len(built['schedule'])==load
        groups[split].extend(dict(day=str(day)+':'+str(load),second=e['arrival_s'],flow=e['flow'],block=e['block']) for e in built['schedule'])
        vs=Counter()
        for bid,scn in built['scenarios'].items():
            for j in scn.jobs:
                if j.flow.value.startswith('VESSEL_'):
                    vs[j.flow.value]+=1
                    if 0 <= j.release_time < 86400:
                        vessels[split].append(dict(day=str(day)+':'+str(load),second=j.release_time,flow=j.flow.value,block=bid))
        metadata.append(dict(split=split,load=load,seed=seed,flow_counts=dict(Counter(e['flow'] for e in built['schedule'])),
            flow_fallbacks=built['flow_fallbacks_total'],vessel_jobs=dict(vs)))
        print(f'input audit {split} seed={seed} load={load}',flush=True)
    return dict(cases=metadata,trucks={k:distribution(v) for k,v in groups.items()},
        vessels_planned_release={k:distribution(v) for k,v in vessels.items()},
        limitation='Synthetic vessel planned releases and observed atw are different event stages; no direct timing calibration is implied.')


def run(root, output):
    from yard_rl.integrated.repro import repro_stamp
    import subprocess
    assert not subprocess.check_output(['git','status','--short','--untracked-files=no']).strip()
    assert not output.exists(), 'Preserve previous audit evidence; use a new output path.'
    observed=source_audit(root)
    synthetic=synthetic_audit()
    result=dict(observed=observed,synthetic=synthetic,
        repro=repro_stamp(experiment='YR-332-aj-input-audit',seeds={'existing_inputs':[r['seed'] for r in synthetic['cases']]},
            params=dict(raw_dates=['2026-10-28','2026-11-09'],private_rows_exported=False,information_profile_sha256=sha(INFORMATION)),prereg=PREREG.as_posix()))
    save(output,result)
    print(json.dumps(dict(entered=observed['entered'],actual_mix=observed['trucks']['flow_counts'],synthetic_mix=synthetic['trucks']['evaluation']['flow_counts']),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path,default=OUT/'input-audit.json')
    args=parser.parse_args();run(args.root,args.output)
