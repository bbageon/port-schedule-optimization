"""Check the actual Python arrival generator against the frozen calibration criteria."""
import bisect
import json
import random

import numpy as np

from synthetic_inputs import OUT,TEMPLATES,read,minute_metrics
from yard_rl.v7.correlated_arrivals import weights
from yard_rl.integrated.repro import write_result


def main():
    cal=read(OUT/'multiscale-minute-calibration.json');assert cal['passed']
    d=read(TEMPLATES);matrix=np.array([r for days in d['minute_counts'].values() for r in days if sum(r)>=2])
    counts=matrix.sum(axis=1);observed=minute_metrics(matrix);repeats=[]
    for seed in range(9936700,9936800):
        rng=random.Random(seed);rows=[]
        for n in counts:
            p=weights(rng,cal['parameters']);cdf=np.cumsum(p).tolist();c=[0]*60
            for _ in range(int(n)):c[min(59,bisect.bisect_right(cdf,rng.random()))]+=1
            rows.append(c)
        repeats.append(minute_metrics(rows))
    mean={k:float(np.mean([r[k] for r in repeats])) for k in observed}
    errors={k:abs(mean[k]-observed[k]) for k in observed}
    passed=all(errors[k]<=v for k,v in dict(index_1m=.10,index_10m=.15,lag1=.05).items())
    result=dict(observed=observed,mean=mean,errors=errors,passed=passed,repeats=repeats)
    write_result(OUT/'minute-implementation-check.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='repeats'}));assert passed


if __name__=='__main__':main()
