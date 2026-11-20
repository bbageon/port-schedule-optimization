"""Compare observed real truck intervals, with explicit raw-message provenance."""
from collections import Counter, defaultdict
import csv
import gzip
import re

from gate_time_provenance import GateEvidence, CONFIRMED, parse_time, direction
from recheck_transport_queue import RAW_ASSIGNMENT, inventory
from inbound_realism_common import RAW, save, stats

EVENTS = {'GateIn': ('gateInTime','gate_in_time'), 'EnterBlock': ('blockInTime','enter_block_time'),
          'JobDone': ('jobDoneTime','job_done_time'), 'GateOut': ('gateOutTime','gate_out_time')}


def main():
    manifest, integrity = inventory(RAW)
    assert not integrity['changed'] and not integrity['new'] and not integrity['missing']
    proof=GateEvidence(); explicit={k:set() for k in EVENTS}; messages=Counter(); raw_fields=Counter()
    for path in sorted((RAW/'기사/_raw').glob('*.gz')):
        with gzip.open(path,'rt',encoding='utf-8') as f:
            for line in f:
                proof.consume(line)
                m=re.search(r'(\w+)Result REQUEST TO TSS|/api/vbs/terminal/(\w+)\s*:',line)
                if not m: continue
                event=m[1] or m[2]
                if event not in EVENTS: continue
                fields=dict(RAW_ASSIGNMENT.findall(line));raw_fields.update(fields.keys())
                if fields.get('rehandleCount','').strip() not in ('','null','None'):
                    messages[event+':nonmissing_rehandle_count']+=1
                t=parse_time(fields.get(EVENTS[event][0]))
                if t is None:
                    messages[event+':missing_event_time']+=1;continue
                if fields.get('inOut','').strip() not in ('1','2','FR','TO','IN','OUT'):
                    messages[event+':missing_direction']+=1;continue
                k=(fields.get('conNo','').strip(),fields.get('truckNo','').strip(),t,direction(fields['inOut'].strip()))
                explicit[event].add(k);messages[event+':explicit_time']+=1
    rows=[]
    for path in sorted((RAW/'기사/방문').glob('*.csv')):
        with path.open(encoding='utf-8-sig',newline='') as f: rows.extend(csv.DictReader(f))
    assert len(rows)==len({r['visit_id'] for r in rows})
    evidence=Counter();coverage=defaultdict(Counter);cohorts=defaultdict(list)
    for r in rows:
        flow=r['in_out'];cls=proof.classify(r);evidence[flow+':'+cls]+=1
        t={e:parse_time(r.get(field)) for e,(_,field) in EVENTS.items()}
        confirmed={e:t[e] is not None and (r['con_no'],r['truck_no'],t[e],flow) in explicit[e] for e in EVENTS}
        for e in EVENTS:
            coverage[flow][e+':present']+=t[e] is not None
            coverage[flow][e+':raw_confirmed']+=confirmed[e]
        if cls not in CONFIRMED: continue
        assert confirmed['GateIn']
        row=dict(times=t,confirmed=confirmed,source=r)
        cohorts[flow].append(row)
    result={}
    for flow,items in cohorts.items():
        intervals=defaultdict(list);invalid=Counter();triples=[];rehandles=[];reported=defaultdict(list)
        stage_counts=Counter();days=defaultdict(list);rehandle_missing=0;full_chain=[];derived_errors=defaultdict(list)
        for x in items:
            t=x['times'];ok=x['confirmed'];r=x['source']
            for e in EVENTS: stage_counts[e+':raw_confirmed']+=ok[e]
            for name,a,b in [('gate_to_block','GateIn','EnterBlock'),('block_to_done','EnterBlock','JobDone'),
                             ('gate_to_done','GateIn','JobDone'),('gate_to_out','GateIn','GateOut')]:
                if ok[a] and ok[b]:
                    v=(t[b]-t[a]).total_seconds()/60
                    if v < 0: invalid[name]+=1
                    else: intervals[name].append(v)
            if all(ok[e] for e in ('GateIn','EnterBlock','JobDone')):
                if t['GateIn']<=t['EnterBlock']<=t['JobDone']:
                    values=[(t['EnterBlock']-t['GateIn']).total_seconds()/60,
                            (t['JobDone']-t['EnterBlock']).total_seconds()/60,
                            (t['JobDone']-t['GateIn']).total_seconds()/60]
                    triples.append(values);days[str(t['GateIn'].date())].append(values)
                    if ok['GateOut']:
                        if t['JobDone']<=t['GateOut']: full_chain.append(values)
                        else: invalid['exit_before_done']+=1
                    if r.get('rehandle_count','')!='': rehandles.append(float(r['rehandle_count']))
                    else: rehandle_missing+=1
                    for k in ('wait_min','work_min','turn_min'):
                        if r.get(k): reported[k].append(float(r[k]))
                    for k,v in [('wait_min',values[0]),('work_min',values[1])]:
                        if r.get(k): derived_errors[k].append(abs(float(r[k])-v))
                else: invalid['full_triple']+=1
        result[flow]=dict(confirmed_gate_orders=len(items),source_confirmed_stage_counts=dict(stage_counts),
            observed_intervals={k:stats(v) for k,v in intervals.items()},negative_intervals=dict(invalid),
            matched_triplets=dict(n=len(triples),gate_to_block=stats([v[0] for v in triples]),
                block_to_done=stats([v[1] for v in triples]),gate_to_done=stats([v[2] for v in triples])),
            triplet_coverage=len(triples)/len(items),reported_rehandles=stats(rehandles),rehandle_missing=rehandle_missing,
            full_chain=dict(n=len(full_chain),gate_to_done=stats([v[2] for v in full_chain]),
                            block_to_done=stats([v[1] for v in full_chain])),
            derived_column_error_min={k:dict(n=len(v),max=max(v),within_rounding=all(x<=.0500001 for x in v))
                                      for k,v in derived_errors.items()},
            reported_derived_columns={k:stats(v) for k,v in reported.items()},
            per_date={d:dict(n=len(v),gate_to_done=stats([x[2] for x in v]),block_to_done=stats([x[1] for x in v]))
                      for d,v in sorted(days.items())})
    out=dict(task='YR-332-bd',raw_integrity=integrity,raw_files=manifest,visits=len(rows),
        evidence_classes=dict(evidence),raw_messages=dict(messages),event_time_coverage={k:dict(v) for k,v in coverage.items()},
        groups=result,raw_timing_fields=sorted(k for k in raw_fields if any(s in k.lower() for s in ('time','handl','start'))),
        scope='Observed source-confirmed intervals; selected completion sample, no causal policy comparison or pure service-time inference')
    save('real-records.json',out)
    print('real records: unchanged source files',integrity['files'],flush=True)
    for flow,g in result.items():
        print(flow,'gate',g['confirmed_gate_orders'],'coverage',g['triplet_coverage'],
              'triplets',g['matched_triplets'],'rehandles',g['reported_rehandles'],flush=True)


if __name__=='__main__': main()
