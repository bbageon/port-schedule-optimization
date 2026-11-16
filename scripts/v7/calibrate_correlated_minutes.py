"""Input-only correction for the failed independent-minute model; no policy scores."""
import json
from pathlib import Path

import numpy as np

from synthetic_inputs import TEMPLATES, OUT, minute_metrics, read, sha
from yard_rl.integrated.repro import write_result


def draw_weights(rng,n,parameters):
    a,b,sigma,innovation=[parameters[k] for k in ('a1','a2','sigma','innovation')]
    z0=np.zeros(n);z1=np.zeros(n);values=[]
    for t in range(188):
        z=a*z1+b*z0+innovation*rng.standard_normal(n);z0,z1=z1,z
        if t>=128: values.append(z)
    logw=sigma*np.array(values).T
    weights=np.exp(logw-logw.max(axis=1,keepdims=True))
    return weights/weights.sum(axis=1,keepdims=True)


def main():
    d=read(TEMPLATES);x=np.array([r for days in d['minute_counts'].values() for r in days if sum(r)>=2],dtype=float)
    n=x.sum(axis=1)
    cov=[float(np.mean((x*(x-1)).sum(axis=1)/(n*(n-1))))*60-1]
    for lag in (1,2):
        cov.append(float(np.mean((x[:,:-lag]*x[:,lag:]).sum(axis=1)/(n*(n-1))))*3600/(60-lag)-1)
    logcov=np.log1p(cov);rho1,rho2=logcov[1:]/logcov[0]
    a1=rho1*(1-rho2)/(1-rho1**2);a2=(rho2-rho1**2)/(1-rho1**2)
    assert abs(a2)<1 and a1+a2<1 and a2-a1<1
    innovation=np.sqrt(1-a1*rho1-a2*rho2)
    params=dict(a1=float(a1),a2=float(a2),sigma=float(np.sqrt(logcov[0])),innovation=float(innovation),burn_in=128)
    observed=minute_metrics(x);repeats=[]
    for seed in range(9936300,9936400):
        rng=np.random.default_rng(seed);p=draw_weights(rng,len(n),params)
        sample=np.array([rng.multinomial(int(k),prob) for k,prob in zip(n,p)])
        repeats.append(minute_metrics(sample))
    mean={k:float(np.mean([r[k] for r in repeats])) for k in observed}
    errors={k:abs(mean[k]-observed[k]) for k in observed}
    passed=all(errors[k]<=v for k,v in dict(index_1m=.10,index_10m=.15,lag1=.05).items())
    result=dict(method='lognormal_ar2_minutes',parameters=params,covariance=cov,observed=observed,mean=mean,errors=errors,
                passed=passed,repeats=repeats,source_sha256=sha(TEMPLATES),
                prereg='docs/research/v7-discharge/minute-arrival-revision.md',scope='post-failure input-only model revision')
    write_result(OUT/'correlated-minute-calibration.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='repeats'},ensure_ascii=True),flush=True)


if __name__=='__main__':main()
