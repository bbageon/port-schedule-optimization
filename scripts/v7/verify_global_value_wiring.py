"""Exercise the extended model through the actual simulator before any PPO claim."""
from global_value_inputs import OUT, SEEDS, input_case, checkpoint_path, save
from global_value_experiment import check
from global_value_rollout import rollout
from six_step_learning import gz_read, gz_save
from value_stability import restore
from yard_rl.v7.crane.global_context import expand_value_net


def main():
    import torch
    torch.set_num_threads(1);check()
    old,_,_=restore(checkpoint_path(SEEDS[0]));net=expand_value_net(old)
    c=input_case('train',12,3500)
    data,orders,flows,resolver=rollout(c,net,mode='global',stochastic=True,collect=True,action_seed=9969212)
    prior=gz_read(OUT/'smoke-off.json.gz')
    keys=('candidates_sha256','choices_sha256','events_sha256','jobs_sha256','records_sha256','result_sha256')
    assert all(data['probe'][k]==prior['probe'][k] for k in keys)
    assert data['orders_sha256']==prior['orders_sha256']
    assert data['global_audit']['rng_state_sha256']==prior['global_audit']['rng_state_sha256']
    assert all(s.state.global_waiting==s.global_observation['feature'] for s in resolver.samples)
    gz_save(OUT/'smoke-global.json.gz',data)
    save(OUT/'extended-smoke.json',dict(passed=True,case=c,global_branch_used=True,
        source_cases=['smoke-off.json.gz','smoke-global.json.gz'],neural_samples=len(resolver.samples),
        exact_fields=list(keys)+['orders_sha256','rng_state_sha256']))
    print('extended model actual episode: all actions, events, costs and RNG exactly match',flush=True)


if __name__=='__main__':main()
