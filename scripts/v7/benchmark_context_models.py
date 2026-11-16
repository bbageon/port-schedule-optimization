"""Standalone neural latency; full candidate generation/resolution is separately measured by episodes."""
import statistics as st
import time

import numpy as np
import torch

from context_learning import OUT,SEEDS
from six_step_learning import load_net
from yard_rl.integrated.repro import write_result


def main():
    torch.set_num_threads(1);torch.manual_seed(9948001);rows=[]
    for arm in ('point_replay','mean_context','star_attention'):
        net=load_net(OUT/f'model-{SEEDS[0]}'/f'{arm}.pt')
        for n in (12,128,512,1200):
            x=torch.randn(n,10);g=torch.zeros(n,dtype=torch.long)
            def call():return net(x,g) if getattr(net,'requires_candidate_context',False) else net(x)
            with torch.no_grad():
                for _ in range(20):call()
                values=[]
                for _ in range(200):
                    start=time.perf_counter_ns();y=call();values.append((time.perf_counter_ns()-start)/1e6)
            assert y.shape==(n,) and torch.isfinite(y).all()
            rows.append(dict(arm=arm,candidates=n,median_ms=st.median(values),p95_ms=float(np.quantile(values,.95)),
                             max_ms=max(values),parameters=sum(p.numel() for p in net.parameters())))
    write_result(OUT/'network-latency.json',dict(rows=rows,threads=1,torch=torch.__version__,
        scope='CPU neural forward only, no candidate-generation or resolver cost; no candidate truncation'))
    print(rows)


if __name__=='__main__':main()
