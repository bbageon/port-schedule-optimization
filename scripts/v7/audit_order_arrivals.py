"""Separate order records, possible vehicle visits, and within-hour clustering."""
from collections import Counter,defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path
import random
import statistics as st

from benchmark_default_eval import save
from six_step_learning import sha

OUT=Path('outputs/reports/yr332_order_representativeness')


def dispersion_control(hour_counts):
    actual=[];matched=[]
    for (day,hour),minutes in sorted(hour_counts.items()):
        n=sum(minutes.values());mean=n/60
        actual.append(st.pvariance([minutes[i] for i in range(60)])/mean)
        rng=random.Random(f'observed-hour-control:{day}:{hour}')
        control=Counter(min(59,int(60*(i+rng.random())/n)) for i in range(n))
        matched.append(st.pvariance([control[i] for i in range(60)])/mean)
    return dict(hours=len(hour_counts),actual_mean=st.mean(actual),stratified_control_mean=st.mean(matched),
        actual_median=st.median(actual),stratified_control_median=st.median(matched),
        definition='Same observed date/hour counts, 60 minute bins, independent within-stratum jitter. Variance divided by mean. Describes clustering, not causal attribution.')


def main():
    manifest=json.loads((OUT/'input-audit-corrected.json').read_text(encoding='utf-8'))
    root=Path(manifest['observed']['private_root'])
    files=sorted(root.glob('기사/방문/*.csv'))
    rows=[]
    for p in files:
        with p.open(encoding='utf-8-sig',newline='') as stream:
            rows.extend(r for r in csv.DictReader(stream) if r['gate_in_time'])
    groups=defaultdict(list);hour_orders=defaultdict(Counter);hour_visits=defaultdict(Counter)
    day_hour_flows=defaultdict(Counter)
    for r in rows:
        t=datetime.fromisoformat(r['gate_in_time'])
        groups[(r['truck_hash'],r['gate_in_time'])].append(r)
        hour_orders[(t.date().isoformat(),t.hour)][t.minute]+=1
        day_hour_flows[(t.date().isoformat(),t.hour)][r['in_out']]+=1
    for _,time in groups:
        t=datetime.fromisoformat(time)
        hour_visits[(t.date().isoformat(),t.hour)][t.minute]+=1
    signatures=Counter()
    for group in groups.values():
        if len(group)>1:
            signature=f"orders={len(group)};containers={len({r['con_hash'] for r in group})};flows={'+'.join(sorted({r['in_out'] for r in group}))}"
            signatures[signature]+=1
    cancelled=[r for r in rows if r['cancel_time']]
    uncancelled=Counter(r['in_out'] for r in rows if not r['cancel_time'])
    report=dict(entered_order_records=len(rows),nonempty_truck_hash=sum(bool(r['truck_hash']) for r in rows),
        distinct_truck_entry_timestamp_keys=len(groups),
        same_truck_timestamp_group_sizes=dict(sorted(Counter(len(v) for v in groups.values()).items())),
        records_in_multi_order_truck_timestamps=sum(len(v) for v in groups.values() if len(v)>1),
        multi_order_group_signatures=dict(signatures),matched_hour_dispersion=dispersion_control(hour_orders),
        grouped_truck_timestamp_dispersion=dispersion_control(hour_visits),
        cancellation=dict(entered_with_cancel_time=len(cancelled),uncancelled_flow_counts=dict(uncancelled),
            cancelled_with_job_done=sum(bool(r['job_done_time']) for r in cancelled)),
        simultaneous_IN_OUT_hours=sum(v['IN']>0 and v['OUT']>0 for v in day_hour_flows.values()),
        observed_nonempty_hours=len(day_hour_flows),
        scope='A truck hash and entry timestamp is a candidate visit key, not a verified physical visit identifier. Do not delete distinct container jobs or equate an order count with vehicle count.',
        source_sha256={p.relative_to(root).as_posix():sha(p) for p in files},script_sha256=sha(Path(__file__)))
    save(OUT/'arrival-clustering-audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','multi_order_group_signatures')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
