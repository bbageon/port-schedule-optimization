from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/v7'))
from global_policy_inputs import bind
from global_policy_train import fit_once
from yard_rl.v7.crane.actor_critic import GraphState,pack_states
from yard_rl.v7.crane.global_context import QueueSnapshot,expand_value_net
from yard_rl.v7.crane.on_policy import DecisionSample
from yard_rl.v7.crane.value_normalization import NormalizedValueNet


def test_large_prediction_error_does_not_veto_real_policy_update():
    torch.set_num_threads(1);torch.manual_seed(73)
    net=expand_value_net(NormalizedValueNet(kind='transformer',width=32,heads=4,layers=2))
    obs=QueueSnapshot(43200.,(('A',20),),20).observation(43200.,'A')
    state=GraphState(torch.randn(3,10),torch.arange(3),torch.zeros(3,dtype=torch.long),.5,obs['feature'])
    with torch.no_grad():logits,value=net(pack_states([state]))
    logp=float(torch.log_softmax(logits[0],0)[0])
    sample=DecisionSample(state,[[0,1,2]],[0],logp,float(value[0]),0.,-1000.,obs)
    samples=[replace(sample,target=-1000.-i) for i in range(8)]
    result,diagnostic=fit_once(net,torch.optim.Adam(net.parameters(),lr=.0003),samples,
        [dict(load=3500)]*8,seed=74,epochs=1,batch_size=4)
    assert diagnostic['before']['rmse']>900 and result['optimizer_steps']==2
    assert net.target_count.item()==8


def test_invalid_executed_probability_still_blocks_policy_update():
    torch.manual_seed(75)
    net=expand_value_net(NormalizedValueNet())
    state=GraphState(torch.randn(3,10),torch.arange(3),torch.zeros(3,dtype=torch.long),.5,0.)
    sample=DecisionSample(state,[[0,1,2]],[0],100.,0.,0.,-1.,None)
    with pytest.raises(ValueError,match='Normalization changed'):
        fit_once(net,torch.optim.Adam(net.parameters(),lr=.0003),[sample]*8,
                 [dict(load=3500)]*8,seed=76,epochs=1,batch_size=4)


def test_bc_input_binding_restores_historical_rollout_after_failure():
    import global_value_rollout as old
    original=(old.INFO,old.demand,old.RUNTIME)
    with pytest.raises(RuntimeError):
        with bind(old,INFO=Path('new-confirmed-profile'),demand=lambda *a:None,RUNTIME={'test':True}):
            assert old.INFO!=original[0]
            raise RuntimeError('simulated failed run')
    assert (old.INFO,old.demand,old.RUNTIME)==original
