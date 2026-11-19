"""Independent source aggregates, fit equations, bundle wiring and exact replay."""
from __future__ import annotations

from collections import Counter,defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path
import re

import numpy as np

from confirmed_demand_support import BASE,OUT,BA,OLD_BANK,read,save,sha
from verify_gate_time_calibration import source_keys,row_key
from yard_rl.v7.input_bundle import InputBundle


def independent_visits(rows):
    grouped = defaultdict(list)
    for row in rows:grouped[(row['truck_hash'],row['gate_in_time'])].append(row)
    trucks = defaultdict(list)
    for (truck,at),items in grouped.items():trucks[truck].append((datetime.fromisoformat(at),items))
    result = []
    for items in trucks.values():
        items.sort(key=lambda r:r[0])
        for i,(at,rs) in enumerate(items):
            followup = any(r[k] for r in rs for k in ('enter_block_time','job_done_time','gate_out_time'))
            if i+1 < len(items) and not followup and (items[i+1][0]-at).total_seconds() <= 1800:
                continue
            result.append((at,rs))
    return result


def independent_patterns(visits):
    counts = Counter()
    for _,rows in visits:
        blocks = defaultdict(lambda:[0,0]);unknown=False
        for row in rows:
            block=None
            for key in ('enter_block_loc','gate_in_loc','last_con_loc'):
                match=re.match(r'^[A-Z][0-9]{2}',row[key] or '')
                if match:block=match.group(0);break
            if block is None:unknown=True;break
            blocks[block][row['in_out']=='OUT'] += 1
        p=tuple(sorted(tuple(v) for v in blocks.values()))
        if not unknown and all(sum(s[k] for s in p) <= 2 for k in (0,1)):counts[p] += 1
    return [dict(stops=[list(s) for s in p],count=n) for p,n in sorted(counts.items())]


def check_curve_equations(fits, days, basis):
    matrix=np.asarray(basis).T
    result=[]
    for i,f in enumerate(fits):
        source=days[f['date']];n=sum(source);target=np.array(source)/n;w=np.asarray(f['weights'])
        predicted=matrix@w;gradient=2*matrix.T@(predicted-target);active=w>1e-7
        level=gradient[active].mean()
        noise=.5*np.abs(np.random.default_rng(9934000+i).multinomial(n,predicted,size=1000)/n-predicted).sum(axis=1)
        expected_threshold=float(np.quantile(noise,.95));residual=float(.5*np.abs(target-predicted).sum())
        ok=(f['orders']==n and f['hour_counts']==source and np.all(w>=0) and abs(w.sum()-1)<1e-12
            and np.allclose(predicted,f['hour_masses'],atol=1e-14,rtol=0)
            and max(abs(gradient[active]-level))<=2e-6 and np.all(gradient[~active]>=level-2e-6)
            and abs(expected_threshold-f['noise_tvd_p95'])<1e-12
            and abs(residual-f['residual_tvd'])<1e-12 and residual <= expected_threshold)
        result.append(dict(date=f['date'],passed=bool(ok),residual_tvd=residual,threshold=expected_threshold))
    return result


def check_minutes(matrix, folder):
    x=np.asarray(matrix,dtype=float);x=x[x.sum(axis=1)>=2];n=x.sum(axis=1)
    cov=[float(np.mean(np.sum(x*(x-1),axis=1)/(n*(n-1))))*60-1]
    for lag in (1,2):cov.append(float(np.mean(np.sum(x[:,:-lag]*x[:,lag:],axis=1)/(n*(n-1))))*3600/(60-lag)-1)
    logcov=np.log1p(cov);r1,r2=logcov[1:]/logcov[0]
    a=r1*(1-r2)/(1-r1*r1);b=(r2-r1*r1)/(1-r1*r1)
    ar=read(folder/'correlated-minute-calibration.json');final=read(folder/'multiscale-minute-calibration.json')
    expected=dict(a1=float(a),a2=float(b),sigma=float(np.sqrt(logcov[0])),innovation=float(np.sqrt(1-a*r1-b*r2)),burn_in=128)
    observed={}
    for width in (1,10):
        bins=x.reshape(len(x),60//width,width).sum(axis=2)
        observed[f'index_{width}m']=float(np.mean(bins.var(axis=1)/bins.mean(axis=1)))
    centred=x-x.mean(axis=1,keepdims=True);den=(centred[:,:-1]**2).sum(axis=1);valid=den>0
    observed['lag1']=float(np.mean((centred[valid,:-1]*centred[valid,1:]).sum(axis=1)/den[valid]))
    expected['coarse_sigma']=float(np.sqrt(max(0,observed['index_10m']-ar['mean']['index_10m'])*36/(5*np.mean(n-1))))
    checks=dict(parameters=all(abs(expected[k]-final['parameters'][k])<1e-12 for k in expected),
                observed=all(abs(observed[k]-final['observed'][k])<1e-12 for k in observed))
    for name in ('correlated-minute-calibration.json','multiscale-minute-calibration.json','minute-implementation-check.json'):
        item=read(folder/name)
        means={k:float(np.mean([r[k] for r in item['repeats']])) for k in observed}
        errors={k:abs(means[k]-observed[k]) for k in observed}
        passed=all(errors[k]<=v for k,v in dict(index_1m=.10,index_10m=.15,lag1=.05).items())
        checks[name]=(len(item['repeats'])==100 and item['mean']==means
            and all(abs(errors[k]-item['errors'][k])<1e-12 for k in errors) and item['passed']==passed)
    return checks


def run(root):
    root=Path(root);explicit,_=source_keys()
    rows=[]
    for path in sorted((BASE/'오더데이터/기사/방문').glob('*.csv')):
        with path.open(encoding='utf-8-sig',newline='') as handle:
            rows.extend(r for r in csv.DictReader(handle) if r['gate_in_time'])
    confirmed=[r for r in rows if row_key(r) in explicit]
    source=read(root/'source-summary.json');build=read(root/'build-record.json');inputs=read(root/'input-check.json')
    bank=read(root/'bank.json');old_bank=read(OLD_BANK)
    checks={};equations={}
    for arm,selected in dict(legacy=rows,confirmed=confirmed).items():
        hours=defaultdict(lambda:[0]*24);outs=Counter()
        for r in selected:
            at=datetime.fromisoformat(r['gate_in_time']);date=at.date().isoformat()
            hours[date][at.hour]+=1;outs[date]+=r['in_out']=='OUT'
        daily=read(root/arm/'daily.json');visits=independent_visits(selected)
        checks[f'{arm}:daily_source']=all(r['orders']==sum(hours[r['date']]) and r['hour_counts']==hours[r['date']]
            and r['gate_out_share']==outs[r['date']]/r['orders'] for r in daily['per_date'])
        checks[f'{arm}:visits']=len(visits)==daily['visits']==source['cohorts'][arm]['visits']
        pats=independent_patterns(visits)
        checks[f'{arm}:patterns']=pats==read(root/arm/'visit-link.json')['patterns']
        dates=source['dates'];minute_counts={str(h):[[0]*60 for _ in dates] for h in range(24)}
        for at,_ in visits:minute_counts[str(at.hour)][dates.index(at.date().isoformat())][at.minute]+=1
        checks[f'{arm}:minute_counts']=minute_counts==read(root/arm/'minute-templates.json')['minute_counts']
        equations[arm]=check_curve_equations(read(root/arm/'curve-fit.json')['fits'],hours,bank['basis_hour_masses'])
        checks[f'{arm}:curve_equations']=all(r['passed'] for r in equations[arm])
        checks.update({f'{arm}:{k}':v for k,v in check_minutes([r for ds in minute_counts.values() for r in ds],root/arm).items()})
    fit={r['date']:r for r in read(root/'confirmed/curve-fit.json')['fits']}
    daily={r['date']:r for r in read(root/'confirmed/daily.json')['per_date']}
    identities=[];interpolation=True;parents=True
    for split,values in bank['splits'].items():
        for i,r in enumerate(values):
            a,b=r['parents'];f=r['fraction'];old=old_bank['splits'][split][i]
            w=[(1-f)*x+f*y for x,y in zip(fit[a]['weights'],fit[b]['weights'])]
            interpolation &= r['weights']==w and r['gate_out_share']==(1-f)*daily[a]['gate_out_share']+f*daily[b]['gate_out_share']
            parents &= r['parents']==old['parents'] and f==old['fraction']
            identities.append(r['shape_sha256'])
    checks.update(interpolation=interpolation,paired_parents=parents,distinct_synthetic_curves=len(set(identities))==19,
        train_only_fixed_curve=bank['fixed_hour_masses']==[sum(r['hour_masses'][h] for r in bank['splits']['train'])/13 for h in range(24)],
        train_only_fixed_mix=bank['fixed_gate_out_share']==sum(r['gate_out_share'] for r in bank['splits']['train'])/13,
        source_exclusions=len(rows)-len(confirmed)==source['excluded_orders']==601)
    names=['source-manifest.json','source-summary.json','bank.json','demand-base.json']
    for arm in ('legacy','confirmed'):
        names += [f'{arm}/{n}' for n in ('daily.json','minute-templates.json','visit-link.json','curve-fit.json',
                  'correlated-minute-calibration.json','multiscale-minute-calibration.json','minute-implementation-check.json')]
    replay={name:sha(root/name)==sha(root/'replay'/name) for name in names}
    bundle=InputBundle(root/'bundle.json');repeat_bundle=InputBundle(root/'replay/bundle.json')
    checks['replay_18_calibration_files']=all(replay.values())
    checks['bundle_replay']=all(bundle.case(s,i,3500)==repeat_bundle.case(s,i,3500) for s,rs in bank['splits'].items() for i in range(len(rs)))
    checks['source_and_code_hashes_unchanged']=all(sha(BASE/p)==h for p,h in build['source_sha256'].items())
    checks['raw_files_unchanged']=all(sha(BASE/'오더데이터'/m['path'])==m['sha256'] for m in read(root/'source-manifest.json'))
    checks['runs_from_clean_commit']=build['repro']['code']['git_dirty'] is False and inputs['repro']['code']['git_dirty'] is False
    checks['complete_input_campaign']=(inputs['status']=='PASS' and len(inputs['records'])==102 and len(inputs['pairs'])==45
        and len(inputs['repeats'])==6 and all(r['exact'] for r in inputs['repeats']))
    checks['order_counts']=inputs['generated_orders']==884000 and inputs['replay_orders']==21000
    checks['axis_guards']=all(all(r['checks'].values()) for r in inputs['pairs'])
    checks['all_bundle_shapes_used']=sum(r['case']['arm']=='all' for r in inputs['records'])==57
    checks['no_policy_execution']=inputs['configuration']['policy_execution'] is False
    result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,replay_matches=replay,curve_equations=equations,
        verification_script_sha256=sha(Path(__file__)),limitations=['Verification of calibration and inputs only; no policy performance or operational realism claim.'])
    save(root/'verification.json',result)
    print(json.dumps(dict(status=result['status'],checks=checks),ensure_ascii=False,indent=2))
    return all(checks.values())


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=OUT)
    raise SystemExit(0 if run(p.parse_args().root) else 1)
