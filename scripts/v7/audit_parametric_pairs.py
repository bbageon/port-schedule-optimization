"""Check individual ownership/visit plans, beyond equal aggregate input counts."""
from concurrent.futures import ProcessPoolExecutor,as_completed

from synthetic_arrivals import OUT,INPUTS,INFO,read,profile
from yard_rl.integrated.repro import write_result
from yard_rl.v7.evaluation_probe import digest


def one(case):
    from yard_rl.v7.stage.orders import build_stage
    from yard_rl.v7.world.integrated.profiles import build_h21_profile
    plans=[]
    time_fields={'arrival_s','gate_in_s','lead_s','reserved_s','eta_offset_s'}
    changed=set()
    for arm in ('observed','parametric'):
        c=dict(case,arm=arm,axis='minutes')
        b=build_stage(load=c['load'],seed=c['seed'],profile=build_h21_profile(),
                      information_profile=read(INFO),demand_profile=profile(c))
        plans.append({r['job_id']:r for r in b['schedule']})
    a,b=plans;assert a.keys()==b.keys()
    for job,row in a.items():
        other=b[job];different={k for k in row.keys()|other.keys() if row.get(k)!=other.get(k)}
        assert different<=time_fields,(case,job,different)
        changed|=different
    attributes={j:{k:v for k,v in r.items() if k not in time_fields} for j,r in a.items()}
    return dict(case=case,orders=len(a),changed_fields=sorted(changed),same_attributes_sha256=digest(attributes))


def main():
    cases=[r['case'] for r in read(INPUTS/'input-audit.json')['records'] if r['case']['arm']=='variable']
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(one,c) for c in cases]):rows.append(f.result())
    rows.sort(key=lambda r:(r['case']['split'],r['case']['index'],r['case']['load']))
    write_result(OUT/'paired-attribute-audit.json',dict(cases=len(rows),records=rows,all_passed=True))
    print('same non-time order attributes / visit plans:',len(rows),'pairs',flush=True)


if __name__=='__main__':main()
