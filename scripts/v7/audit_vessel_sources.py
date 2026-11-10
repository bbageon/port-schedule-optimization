"""Check original vessel JSON and raw log field coverage without exposing IDs."""
from collections import Counter
from collections import defaultdict
from datetime import datetime
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re

from calibrate_order_information import stamp, stats, save, OUT


def reconcile_vessel_times(root):
    fields=('loadDiscGbn','conNo','atw','firstYardInTime','lastYardInTime')
    pairs=rows=0;parsed=Counter()
    def key(r):return tuple(str(r.get(k) or '') for k in fields)
    for p in sorted((root/'선박'/'양적하목록').glob('*.csv')):
        q=p.with_suffix('.json');assert q.exists()
        original=json.loads(q.read_text(encoding='utf-8-sig'))['response']['data']['conList']
        with p.open(encoding='utf-8-sig') as h:converted=list(csv.DictReader(h))
        assert Counter(map(key,converted))==Counter(map(key,original)),p.name
        pairs+=1;rows+=len(converted)
        for r in converted:
            if r.get('atw'):
                datetime.fromisoformat(r['atw']);parsed[r['loadDiscGbn']]+=1
    result=dict(status='PASS',csv_json_pairs=pairs,rows=rows,fields=list(fields),
        atw_parsed_by_flow=dict(parsed),
        meaning='CSV and original JSON timing values agree; physical event meaning still requires a field definition.')
    save(OUT/'vessel-time-reconciliation.json',result)
    return result


def audit(root):
    paths=Counter();manifest=[];calls={};move_keys=set();rows=0;durations=[];gap=[]
    work_counts=defaultdict(Counter)
    work_times=defaultdict(lambda: defaultdict(list))
    def walk(value,path=''):
        if isinstance(value,dict):
            for k,v in value.items():paths[path+'.'+k]+=1;walk(v,path+'.'+k)
        elif isinstance(value,list):
            for v in value:walk(v,path+'[]')
    for f in sorted((root/'선박').rglob('*.json')):
        manifest.append(dict(path=f.relative_to(root).as_posix(),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
        d=json.loads(f.read_text(encoding='utf-8-sig'));walk(d)
        data=d.get('response',{}).get('data')
        if not isinstance(data,dict) or 'conList' not in data:continue
        key=tuple(str(data.get(k)) for k in ('terminalInYear','terminalVesselCode','terminalVoyageNo'))
        calls[key]=data
        for r in data['conList']:
            rows+=1;move_keys.add((*key,r.get('loadDiscGbn'),r.get('conNo')))
    for data in calls.values():
        a,b=stamp(data.get('atb')),stamp(data.get('atd'))
        if a and b:durations.append((b-a).total_seconds()/3600)
        for r in data['conList']:
            flow=str(r.get('loadDiscGbn'));c=work_counts[flow];c['rows']+=1
            w=stamp(r.get('atw'))
            if not w:
                c['atw_missing_or_unparsed']+=1
                continue
            c['atw_parsed']+=1
            if a and b:
                c['atw_with_actual_berth']+=1
                c['atw_within_actual_berth']+=a<=w<=b
            for key,t in [('atb_to_atw_hours',a),('first_yard_in_minus_atw_hours',stamp(r.get('firstYardInTime'))),
                          ('last_yard_in_minus_atw_hours',stamp(r.get('lastYardInTime')))]:
                if t:
                    delta=(w-t if key=='atb_to_atw_hours' else t-w).total_seconds()/3600
                    work_times[flow][key].append(delta)
        a,b=stamp(data.get('etb')),stamp(data.get('etd'))
        if a and b:gap.append((b-a).total_seconds()/3600)
    log_fields=Counter();line_count=0
    for f in sorted(root.glob('기사/_raw/*.gz')):
        manifest.append(dict(path=f.relative_to(root).as_posix(),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
        with gzip.open(f,'rt',encoding='utf-8') as h:
            for line in h:
                line_count+=1
                log_fields.update(set(re.findall(r'\b([A-Za-z][A-Za-z0-9_]*)\s*=',line)))
    event_keys=defaultdict(set);visit_matches=Counter()
    queue_fields={k:Counter() for k in ('block_waiting_cars','exp_waiting_min','block_avg_tt')}
    for f in sorted(root.glob('기사/이벤트/*.csv')):
        manifest.append(dict(path=f.relative_to(root).as_posix(),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
        for r in csv.DictReader(f.open(encoding='utf-8-sig')):
            event_keys[r['event']].add((r['con_no'],r['truck_no'],r['event_time'],r['in_out']))
            for k in queue_fields:queue_fields[k][r[k]]+=1
    for f in sorted(root.glob('기사/방문/*.csv')):
        manifest.append(dict(path=f.relative_to(root).as_posix(),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
        for r in csv.DictReader(f.open(encoding='utf-8-sig')):
            for field,event in (('gate_in_time','GateIn'),('job_done_time','JobDone'),('gate_out_time','GateOut')):
                if r[field]:
                    visit_matches[event+'_rows']+=1
                    visit_matches[event+'_matched']+=(r['con_no'],r['truck_no'],r[field],r['in_out']) in event_keys[event]
    assert all(visit_matches[e+'_rows']==visit_matches[e+'_matched'] for e in ('GateIn','JobDone','GateOut'))
    result=dict(vessel_json_key_paths=dict(paths),vessel_calls=len(calls),container_rows=rows,
        unique_call_container_flow=len(move_keys),duplicate_rows=rows-len(move_keys),
        actual_berth_hours=stats(durations),snapshot_planned_berth_hours=stats(gap),
        actual_berths_over_24h=sum(x>24 for x in durations),raw_log_lines=line_count,
        visit_event_reconciliation=dict(visit_matches),queue_field_values={k:dict(v) for k,v in queue_fields.items()},
        raw_log_assignment_fields=dict(sorted(log_fields.items())),
        vessel_atw_timestamps={flow:dict(counts=dict(counts),
            time_differences={k:stats(v) for k,v in work_times[flow].items()}) for flow,counts in work_counts.items()},
        caveats=['Final snapshots are not historical as-of forecasts.',
                 'Berth duration includes multiple ship and yard resources; not an individual crane cycle.',
                 'atw has timestamp values; whether it denotes quay work or yard completion needs a field definition.',
                 'atw differences across containers are not per-crane service durations without crane IDs and start times.',
                 'No measured internal tractor fleet/cycle is inferred from external truck visits.'])
    save(OUT/'vessel-source-audit.json',result);save(OUT/'original-source-manifest.json',manifest)
    reconcile_vessel_times(root)
    print(json.dumps({k:v for k,v in result.items() if k not in ('vessel_json_key_paths','raw_log_assignment_fields')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);audit(parser.parse_args().root)
