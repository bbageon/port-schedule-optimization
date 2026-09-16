"""Recompute the new daily artifacts without launching a simulator."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def audit(folder):
    result = json.loads((folder/'result.json').read_text(encoding='utf-8'))
    daily = [json.loads(x) for x in (folder/'daily-final.jsonl').read_text(encoding='utf-8').splitlines()]
    with gzip.open(folder/'operating-state.jsonl.gz', 'rt', encoding='utf-8') as f:
        states = [json.loads(x) for x in f]
    with gzip.open(folder/'requests.jsonl.gz', 'rt', encoding='utf-8') as f:
        requests = [json.loads(x) for x in f]
    close = lambda a,b: math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-6)
    checks = dict(daily_rows=len(daily)==len(result['days']),
        final_rows_match=all({k:v for k,v in row.items() if k not in ('seed','arm','policy_label')}==d
                            for row,d in zip(daily,result['days'])),
        file_hashes=all(hashlib.sha256((folder/name).read_bytes()).hexdigest()==key
                        for name,key in result['daily_artifacts'].items()),
        sample_grid=[s['at_s'] for s in states]==[i*result['daily_observation']['sample_s']
                        for i in range(result['daily_observation']['samples'])],
        block_sums=all(close(total,sum(b[k] for b in s['blocks'].values()))
                      for s in states for k,total in s['total'].items()),
        midnight_links=all(a['operational']['end']==b['operational']['start']
                           for a,b in zip(daily,daily[1:])))
    queue_matches = True
    for s in states:
        actual = {bid:0 for bid in s['blocks']}
        for r in requests:
            at, start = r['block_in_s'], r['service_start_s']
            if at is not None and at <= s['at_s'] and (start is None or start > s['at_s']):
                actual[r['final_block']] += 1
        queue_matches &= all(actual[bid]==row['truck_queue'] for bid,row in s['blocks'].items())
    checks['queues_match_request_events'] = queue_matches
    for d in daily:
        op = d['operational']; i=d['index']
        cohort = [r for r in requests if r['requested_day']==i]
        times = sorted(r['accounted_turn_time_s'] for r in cohort if r['accounted_turn_time_s'] is not None)
        mean = sum(times)/len(times) if times else 0
        p90 = times[min(len(times)-1,int(.9*len(times)))] if times else 0
        a = op['actions']; total=a['spatial']+a['temporal']
        checks[f'day_{i}'] = all([
            d['seed']==result['seed'], d['arm']==result['arm'], d['policy_label']==result['label'],
            not d['provisional'], op['cohort']['requested']==len(cohort)==d['load'],
            op['cohort']['entered_by_cutoff']+op['cohort']['not_entered_by_cutoff']==len(cohort),
            close(d['phi_krw'],sum(d[k] for k in ('c_wait','c_move','c_rehandle','c_vessel'))),
            close(d['c_wait'],sum(r['accounted_wait_krw'] for r in cohort)),
            close(d['mean_turn_time_s'],mean), close(d['p90_turn_time_s'],p90),
            d['n_censored']==op['cohort']['censored_by_cutoff'],
            a['spatial_share']==(a['spatial']/total if total else None),
            a['temporal_share']==(a['temporal']/total if total else None),
            close(op['queue']['truck_queue_mean'],
                  (op['end']['total']['truck_queue_area_s']-op['start']['total']['truck_queue_area_s'])/86400)
        ])
    return dict(passed=all(checks.values()), checks=checks, day_rows=len(daily), state_rows=len(states))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    root=Path(args.run)
    results={arm:audit(root/('on_'+arm)) for arm in ('NO_REALLOC','RL','RL_TIME')}
    out=dict(passed=all(r['passed'] for r in results.values()), arms=results)
    Path(args.out).write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2))
    return 0 if out['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
