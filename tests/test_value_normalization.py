"""Critical contracts: unchanged control, preserved predictions and exact resume."""
import copy

import pytest
import torch

from yard_rl.v7.crane.actor_critic import GraphState, SchedulingActorCritic, pack_states
from yard_rl.v7.crane.on_policy import DecisionSample, action_statistics, ppo_update
from yard_rl.v7.crane.value_normalization import NormalizedValueNet, fit


def samples(net, seed=12):
    g=torch.Generator().manual_seed(seed)
    states=[GraphState(torch.rand(2+i%3,10,generator=g),torch.arange(2+i%3),
                       torch.zeros(2+i%3,dtype=torch.long),i/16) for i in range(16)]
    rows=[DecisionSample(s,[list(range(len(s.features)))],[i%len(s.features)],0.,0.,0.,
                         target=-float(1+i%7)) for i,s in enumerate(states)]
    with torch.no_grad():
        ll,v=net(pack_states(states)); lp,_=action_statistics(ll,rows)
    for r,p,value in zip(rows,lp,v): r.old_log_probability=float(p);r.old_value=float(value)
    return rows


def same(a,b):
    if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): same(x,y)
    else: assert a==b


def test_control_exactly_reproduces_existing_optimizer():
    torch.set_num_threads(1);torch.manual_seed(27)
    old=SchedulingActorCritic().eval()
    torch.manual_seed(27);new=NormalizedValueNet().eval()
    a=torch.optim.Adam(old.parameters(),lr=.0003);b=torch.optim.Adam(new.parameters(),lr=.0003)
    data=samples(old)
    x=ppo_update(old,a,data,seed=49,max_samples=16,epochs=2,batch_size=8)
    y=fit(new,b,data,arm='control',seed=49,max_samples=16,epochs=2,batch_size=8)
    same(x,y);same(a.state_dict(),b.state_dict())
    for k,v in old.state_dict().items(): same(v,new.state_dict()[k])


def test_popart_preserves_all_predictions_and_cumulative_statistics():
    torch.manual_seed(31);net=NormalizedValueNet().eval()
    batch=pack_states([r.state for r in samples(net)])
    original_logits,original_values=net(batch)
    seen=[]
    for values in ([-1.,-9.,-3.],[-100.,-10.],[-2.]*10):
        seen.extend(values);net.update_statistics(values)
        logits,predictions=net(batch)
        assert torch.equal(logits,original_logits)
        torch.testing.assert_close(predictions,original_values,atol=2e-5,rtol=0)
        expected=torch.tensor(seen,dtype=torch.float64)
        assert net.target_mean.item()==pytest.approx(expected.mean().item())
        assert net.target_scale.item()==pytest.approx(expected.std(unbiased=False).item())
    count=net.target_count.clone()
    net.eval();net(batch)
    assert torch.equal(count,net.target_count)


def test_resume_includes_adam_and_normalizer(tmp_path):
    torch.manual_seed(33);net=NormalizedValueNet().eval()
    opt=torch.optim.Adam(net.parameters(),lr=.0003)
    fit(net,opt,samples(net),arm='popart',seed=35,max_samples=16,epochs=2,batch_size=8)
    data=samples(net,20)
    path=tmp_path/'checkpoint.pt'
    torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict()),path)
    saved=torch.load(path,weights_only=False)
    other=NormalizedValueNet().eval();other.load_state_dict(saved['model'])
    other_opt=torch.optim.Adam(other.parameters(),lr=.0003);other_opt.load_state_dict(saved['optimizer'])
    for n,o in ((net,opt),(other,other_opt)):
        fit(n,o,data,arm='popart',seed=41,max_samples=16,epochs=2,batch_size=8)
    same(net.state_dict(),other.state_dict());same(opt.state_dict(),other_opt.state_dict())


def test_invalid_statistics_do_not_mutate_the_model():
    net=NormalizedValueNet();prior=copy.deepcopy(net.state_dict())
    for bad in ([],[float('nan')],[float('inf')]):
        with pytest.raises(ValueError):net.update_statistics(bad)
    same(prior,net.state_dict())
