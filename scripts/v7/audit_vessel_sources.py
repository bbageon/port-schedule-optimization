"""Check original vessel JSON and raw log field coverage without exposing IDs."""
from collections import Counter
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re

from calibrate_order_information import stamp, stats, save, OUT


def audit(root):
    paths=Counter();manifest=[];calls={};move_keys=set();rows=0;durations=[];gap=[]
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
        caveats=['Final snapshots are not historical as-of forecasts.',
                 'Berth duration includes multiple ship and yard resources; not an individual crane cycle.',
                 'No measured internal tractor fleet/cycle is inferred from external truck visits.'])
    save(OUT/'vessel-source-audit.json',result);save(OUT/'original-source-manifest.json',manifest)
    print(json.dumps({k:v for k,v in result.items() if k not in ('vessel_json_key_paths','raw_log_assignment_fields')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);audit(parser.parse_args().root)
