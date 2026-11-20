"""Decompose every saved holdout truck order, retaining unfinished exposure."""
from collections import defaultdict
import math
import statistics as st

from inbound_realism_common import SOURCE, cases, label, read, save, sha, stats

FLOWS = ('GATE_IN', 'GATE_OUT')
FIELDS = ('gate_in_s', 'block_in_s', 'service_start_s', 'job_done_s')
PARTS = ('gate_to_block', 'block_to_start', 'start_to_done')


def intervals(row, horizon):
    t = row['times']
    points = [min(horizon, t[k]) if t[k] is not None else horizon for k in FIELDS]
    assert all(b >= a-1e-6 for a, b in zip(points, points[1:])), 'nonmonotone clocks'
    parts = [(b-a)/60 for a, b in zip(points, points[1:])]
    total = (points[-1]-points[0])/60
    assert math.isclose(sum(parts), total, abs_tol=1e-8)
    return dict(zip(PARTS, parts), total=total)


def group(rows, horizon):
    values = [intervals(r, horizon) for r in rows]
    complete = [r for r in rows if r['times']['job_done_s'] is not None and r['times']['job_done_s'] <= horizon]
    started = [r for r in complete if r['times']['service_start_s'] is not None]
    return dict(n=len(rows), unfinished=len(rows)-len(complete),
        means={k:st.mean(x[k] for x in values) for k in (*PARTS, 'total')},
        completed_total=stats([intervals(r, horizon)['total'] for r in complete]),
        completed_service=stats([(r['times']['job_done_s']-r['times']['service_start_s'])/60 for r in started]),
        hours={str(h):dict(n=len(g), means={k:st.mean(x[k] for x in g) for k in (*PARTS,'total')})
               for h in range(4) if (g := [v for r,v in zip(rows,values) if int(r['times']['gate_in_s']//21600)==h])})


def main():
    expected = read(SOURCE / 'summary.json')
    summaries, manifests, pairs = [], {}, {}
    for i,c in enumerate(cases()):
        if c['split'] != 'holdout' or c['mode'] != 'argmax':
            continue
        path = SOURCE / 'orders' / f'{i:03d}.json.gz'
        rows = read(path);manifests[str(path.relative_to(SOURCE))] = sha(path)
        assert len(rows)==c['load'] and len({r['job_id'] for r in rows})==len(rows)
        for flow in FLOWS:
            subset = [r for r in rows if r['flow']==flow]
            summaries.append(dict(case=c, label=label(c), flow=flow,
                at24=group(subset,86400),at26=group(subset,93600)))
        if c['arm'] in ('zero','global') and c['stage']=='final':
            pairs.setdefault((c['seed'],c['index'],c['load']),{})[c['arm']] = rows
    aggregates=[]
    for arm in ('initial','zero','global','SF_SPT'):
        for load in (0,3500,7500,15000):
            for flow in FLOWS:
                selected=[r for r in summaries if r['label']==arm and r['flow']==flow
                          and (not load or r['case']['load']==load)]
                result=dict(arm=arm,load=load,flow=flow,cases=len(selected))
                for window in ('at24','at26'):
                    result[window]=dict(means={k:st.mean(r[window]['means'][k] for r in selected)
                        for k in (*PARTS,'total')},unfinished_mean=st.mean(r[window]['unfinished'] for r in selected),
                        completed_service_mean=st.mean(r[window]['completed_service']['mean'] for r in selected),
                        completed_total_mean=st.mean(r[window]['completed_total']['mean'] for r in selected))
                want=(expected['by_load'][str(load)]['groups'][arm] if load else expected['groups'][arm])['flows'][flow]['mean_min']
                assert math.isclose(result['at24']['means']['total'], want, rel_tol=1e-10,abs_tol=1e-8), (result,want)
                aggregates.append(result)
    paired=[]
    for (seed,day,load),arms in sorted(pairs.items()):
        assert len(arms)==2
        left={r['job_id']:r for r in arms['zero']};right={r['job_id']:r for r in arms['global']}
        assert left.keys()==right.keys()
        for key,a in left.items():
            b=right[key]
            assert all(a[k]==b[k] for k in ('flow','block','scheduled_gate_s','notice_s','travel_s'))
            assert a['times']['gate_in_s']==b['times']['gate_in_s']
        for flow in FLOWS:
            keys=[k for k in left if left[k]['flow']==flow]
            for horizon in (86400,93600):
                both=[k for k in keys if all(x[k]['times']['job_done_s'] is not None and
                      x[k]['times']['job_done_s']<=horizon for x in (left,right))]
                delta=[intervals(right[k],horizon)['total']-intervals(left[k],horizon)['total'] for k in keys]
                paired.append(dict(seed=seed,day=day,load=load,flow=flow,horizon=horizon,n=len(keys),
                    paired_delta=stats(delta),improved=sum(v < -1e-7 for v in delta),worsened=sum(v > 1e-7 for v in delta),
                    both_complete_n=len(both),both_complete={a:{k:st.mean(intervals(data[j],horizon)[k] for j in both)
                    for k in (*PARTS,'total')} for a,data in (('zero',left),('global',right))}))
    assert len(pairs)==27
    save('simulation.json',dict(task='YR-332-bd',aggregates=aggregates,per_case=summaries,paired=paired,
        sources=manifests,verified=dict(pairs=27,report_means=32,identical_gate_arrivals=True,
        component_identity=True),scope='Saved synthetic holdout trajectories; completed-only comparison is descriptive and selected'))
    print('simulation: 27 pairs and 32 published means verified',flush=True)
    for r in aggregates:
        if r['arm'] in ('zero','global') and r['load'] in (0,15000): print(r['arm'],r['load'],r['flow'],r['at24'],flush=True)


if __name__=='__main__': main()
