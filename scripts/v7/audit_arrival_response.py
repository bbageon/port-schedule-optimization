"""Describe gate orders versus actual block-ready orders from fixed-policy raw ledgers."""
from collections import defaultdict
from pathlib import Path
import statistics as st

from evaluate_visit_links import index
from six_step_learning import gz_read,sha
from yard_rl.integrated.repro import write_result

ROOT=Path('outputs/reports/yr332_parametric_arrivals')


def measure(rows,field):
    groups=defaultdict(list)
    used=0
    for row in rows:
        t=row[field]
        if t is not None and t<86400:
            groups[(row['block'],int(t//3600))].append(t%3600);used+=1
    intervals=[b-a for xs in groups.values() for a,b in zip(sorted(xs),sorted(xs)[1:])]
    return dict(index_1m=index(groups,60),index_10m=index(groups,600),observed_orders=used,
        not_yet_observed=len(rows)-used,groups_with_two=sum(len(x)>=2 for x in groups.values()),
        within_block_hour_gap_s=dict(n=len(intervals),mean=st.mean(intervals),median=st.median(intervals),
            cv=st.pstdev(intervals)/st.mean(intervals)))


def main():
    paths=sorted((ROOT/'evaluation').glob('*.json.gz'));assert len(paths)==36
    rows=[]
    for path in paths:
        d=gz_read(path);fp=ROOT/'flow-orders'/path.name
        trucks=[r for r in gz_read(fp) if r['flow'].startswith('GATE_')]
        rows.append(dict(case=d['case'],gate_by_destination_block=measure(trucks,'origin_s'),
            actual_block_ready=measure(trucks,'ready_s'),raw_sha256=sha(fp)))
    groups=[]
    for load in (3500,7500,15000):
        for rule in ('SF_SPT','FIFO'):
            for arm in ('observed','parametric'):
                selected=[r for r in rows if r['case']['load']==load and r['case']['rule']==rule and r['case']['arm']==arm]
                groups.append(dict(load=load,rule=rule,arm=arm,**{field:dict(
                    index_1m=st.mean(r[field]['index_1m'] for r in selected),
                    index_10m=st.mean(r[field]['index_10m'] for r in selected),
                    not_yet_observed=sum(r[field]['not_yet_observed'] for r in selected))
                    for field in ('gate_by_destination_block','actual_block_ready')}))
    result=dict(rows=rows,groups=groups,scope='post-execution descriptive audit, not a calibration or new policy selection',
        definitions=dict(unit='orders, not unique physical visits',
            block_ready='recorded block_in_s, affected by prior visit tasks and the dispatch policy',
            index='mean within block/hour variance-to-mean ratio, conditioned on at least two observed orders',
            interval='within the same block/hour; excludes across-hour gaps and right-censored future arrivals'))
    write_result(ROOT/'block-arrival-audit.json',result)
    print(groups)


if __name__=='__main__':main()
