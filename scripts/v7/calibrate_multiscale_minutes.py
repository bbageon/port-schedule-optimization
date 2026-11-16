"""One predeclared residual-based ten-minute variance correction, input statistics only."""
import json
from pathlib import Path

import numpy as np

from calibrate_correlated_minutes import draw_weights
from synthetic_inputs import TEMPLATES, OUT, minute_metrics, read, sha
from yard_rl.integrated.repro import write_result


def main():
    previous=read(OUT/'correlated-minute-calibration.json')
    d=read(TEMPLATES);x=np.array([r for days in d['minute_counts'].values() for r in days if sum(r)>=2],dtype=float)
    n=x.sum(axis=1);params=dict(previous['parameters'])
    residual=max(0,previous['observed']['index_10m']-previous['mean']['index_10m'])
    params['coarse_sigma']=float(np.sqrt(residual*36/(5*np.mean(n-1))))
    observed=minute_metrics(x);repeats=[]
    for seed in range(9936500,9936600):
        rng=np.random.default_rng(seed);p=draw_weights(rng,len(n),params)
        coarse=np.repeat(rng.normal(0,params['coarse_sigma'],size=(len(n),6)),10,axis=1)
        p*=np.exp(coarse);p/=p.sum(axis=1,keepdims=True)
        sample=np.array([rng.multinomial(int(k),prob) for k,prob in zip(n,p)])
        repeats.append(minute_metrics(sample))
    mean={k:float(np.mean([r[k] for r in repeats])) for k in observed}
    errors={k:abs(mean[k]-observed[k]) for k in observed}
    passed=all(errors[k]<=v for k,v in dict(index_1m=.10,index_10m=.15,lag1=.05).items())
    result=dict(method='correlated_minutes',parameters=params,observed=observed,mean=mean,errors=errors,
                passed=passed,repeats=repeats,source_sha256=sha(TEMPLATES),previous_sha256=sha(OUT/'correlated-minute-calibration.json'),
                prereg='docs/research/v7-discharge/minute-arrival-revision-2.md',scope='post-failure input-only multiscale correction')
    write_result(OUT/'multiscale-minute-calibration.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='repeats'},ensure_ascii=True),flush=True)


if __name__=='__main__':main()
