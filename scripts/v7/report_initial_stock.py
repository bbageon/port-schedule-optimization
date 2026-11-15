"""Independent raw-record checks and summaries for the stock/shape campaign."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics as st

from check_initial_stock import OUT, BANK, check, read, sha, write_result
from horizon_metrics import aggregate
from six_step_learning import gz_read
from verify_job_completion import raw_metrics, assert_metrics
from yard_rl.v7.evaluation_probe import digest


def evidence(index, manifest):
    paths = {name: OUT / name / f'{index:03d}.json.gz' for name in ('evaluation', 'orders', 'flow-orders')}
    values = {name: gz_read(path) for name, path in paths.items()}
    data, orders, flows = (values[name] for name in ('evaluation', 'orders', 'flow-orders'))
    assert data['case'] == manifest['configuration']['cases'][index]
    assert data['repro'] == manifest['repro'] and data['campaign_sha256'] == sha(OUT / 'campaign.json')
    assert digest(orders) == data['orders_sha256']
    metrics = raw_metrics(orders)
    assert metrics == data['metrics_by_flow']; assert_metrics(metrics, data['result'])
    assert len({r['job_id'] for r in flows}) == len(flows)
    for h in (24, 26):
        assert aggregate(flows, h * 3600.) == data['completion_by_flow'][f'at{h}']
    result = data['result']
    assert result['admitted'] == data['case']['load']
    assert result['policy_exceptions'] == result['rollout_worlds'] == result['traded_edges'] == 0
    assert data['probe']['ledger_audit']['skipped'] == 0
    assert all(r['review_count'] == 1 for r in orders)
    assert data['visit_stats']['late_releases'] == data['visit_stats']['late_exits'] == 0
    return data, {p.as_posix(): sha(p) for p in paths.values()}


def stock():
    campaign = check(); audit = read(OUT / 'input-audit.json')
    assert not audit['failures'] and audit['repeat_equal']
    assert len(audit['records']) == 183
    records = audit['records']; changed = [r for r in records if r['before_conversions']]
    assert all(r['after_conversions'] == 0 for r in records)
    assert all(r['unchanged'] for r in records if not r['before_conversions'])
    data, hashes = [], {}
    for index in range(4):
        row, h = evidence(index, campaign); data.append(row); hashes.update(h)
    a, b = data[2:]
    for key in ('candidates_sha256', 'choices_sha256', 'events_sha256', 'jobs_sha256',
                'records_sha256', 'result_sha256'):
        assert a['probe'][key] == b['probe'][key], key
    assert a['result'] == b['result'] and a['orders_sha256'] == b['orders_sha256']
    counts = [n for r in changed for n in r['initial_stock']['initial_counts_after'].values()]
    summary = dict(cases=len(records), changed_cases=len(changed), unchanged_cases=len(records)-len(changed),
        old_conversions=sum(r['before_conversions'] for r in records), new_conversions=0,
        known_conversions=sum(r['before_conversions'] for r in changed if r['seed'] < 9935000),
        new_seed_conversions=sum(r['before_conversions'] for r in changed if r['seed'] >= 9935000),
        changed_inputs=[{k: r[k] for k in ('seed', 'load', 'before_conversions')} for r in changed],
        initial_block_count_min=min(counts), initial_block_count_max=max(counts),
        initial_terminal_counts=sorted({r['stock_count'] for r in records}),
        repeat_input_equal=True, normal_runtime_equal=True, runtime_cases=4,
        runtime=[dict(case=r['case'], truck_minutes=r['result']['mean_gate_to_job_done_s']/60,
            phi_krw=r['result']['phi_krw'], by_flow=r['completion_by_flow'], stock=r['stock'],
            seconds=r['seconds']) for r in data], artifact_hashes=hashes,
        repro=campaign['repro'], campaign_sha256=sha(OUT/'campaign.json'),
        scope='finite synthetic initial-stock feasibility, no actual-yard allocation validation')
    write_result(OUT/'stock-summary.json', summary)
    bank=read(BANK)
    assert all(any(r['seed']==s['input_seed'] and r['load']==15000 and not r['after_conversions']
                   for r in records) for split in bank['splits'].values() for s in split)
    bank['initial_stock_mode']='preserve'
    bank['supply_validation']=dict(input_audit_sha256=sha(OUT/'input-audit.json'),
        rejected_backgrounds=0,scope='original 36 seeds restored; finite tested domain')
    write_result(OUT/'bank-stock-preserved.json',bank)
    print({k:summary[k] for k in ('cases','changed_cases','old_conversions','new_conversions','normal_runtime_equal')})


def shapes():
    campaign = check(); rows = {}; hashes = {}
    for index in range(4,58):
        data, h = evidence(index,campaign); hashes.update(h)
        c=data['case']; rows[(c['index'],c['load'],c['rule'],c['shape'])]=data
    pairs=[]
    for i in range(3):
        for load in (3500,7500,15000):
            for rule in ('SF_SPT','FIFO','LWKR'):
                a,b=(rows[(i,load,rule,s)] for s in ('fixed','variable'))
                assert a['stock']==b['stock']
                af,bf=(r['completion_by_flow']['at24'] for r in (a,b))
                assert {f:r['n'] for f,r in af.items()} == {f:r['n'] for f,r in bf.items()}
                flows={f:dict(fixed_minutes=af[f]['mean_s']/60,variable_minutes=bf[f]['mean_s']/60,
                    change_percent=100*(bf[f]['mean_s']/af[f]['mean_s']-1),
                    fixed_unfinished=af[f]['unfinished'],variable_unfinished=bf[f]['unfinished'])
                    for f in af}
                pairs.append(dict(index=i,load=load,rule=rule,by_flow=flows,
                    fixed_cost=a['result']['phi_krw'],variable_cost=b['result']['phi_krw'],
                    cost_change_percent=100*(b['result']['phi_krw']/a['result']['phi_krw']-1)))
    groups=[]
    for load in (3500,7500,15000):
        for rule in ('SF_SPT','FIFO','LWKR'):
            group=[r for r in pairs if r['load']==load and r['rule']==rule]
            by_flow={}
            for flow in ('TRUCK','GATE_IN','GATE_OUT','VESSEL_DISCHARGE','VESSEL_LOAD'):
                # Pool actual order counts; three days can have different flow counts.
                totals={}
                for shape in ('fixed','variable'):
                    cells=[rows[(i,load,rule,shape)]['completion_by_flow']['at24'][flow] for i in range(3)]
                    totals[shape]=dict(n=sum(r['n'] for r in cells), total_s=math.fsum(r['total_s'] for r in cells),
                                       unfinished=sum(r['unfinished'] for r in cells))
                    totals[shape]['mean_min']=totals[shape]['total_s']/totals[shape]['n']/60
                by_flow[flow]=dict(**totals, change_percent=100*(totals['variable']['mean_min']/totals['fixed']['mean_min']-1))
            groups.append(dict(load=load,rule=rule,by_flow=by_flow,
                cost_change_percent=100*(sum(r['variable_cost'] for r in group)/sum(r['fixed_cost'] for r in group)-1),
                pair_min_percent=min(r['by_flow']['TRUCK']['change_percent'] for r in group),
                pair_max_percent=max(r['by_flow']['TRUCK']['change_percent'] for r in group)))
    changes=[r['by_flow']['TRUCK']['change_percent'] for r in pairs]
    summary=dict(cases=54,pairs=pairs,groups=groups, min_truck_change=min(changes),max_truck_change=max(changes),
        median_absolute_truck_change=st.median(abs(x) for x in changes),
        artifact_hashes=hashes,repro=campaign['repro'],campaign_sha256=sha(OUT/'campaign.json'),
        definition='positive change = longer; gate entry to job completion, censored at 24h, all orders retained',
        scope='three development curves, frozen rules; not learning improvement or independent real-period validation')
    write_result(OUT/'shape-summary.json',summary)
    print({'cases':54,'min_truck_change':min(changes),'max_truck_change':max(changes)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['stock','shapes']);args=p.parse_args()
    stock() if args.command=='stock' else shapes()
