from pathlib import Path
import sys
import json
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/v3'))
import run_block_only_evaluation as runner
from independent_eval_checks import read, save, sha


def test_primary_config_hash_uses_exact_frozen_source_not_windows_line_endings(tmp_path,monkeypatch):
    frozen=tmp_path/'frozen';workspace=tmp_path/'workspace'
    frozen.mkdir();workspace.mkdir()
    raw='{\n  "prereg": "original.md"\n}\n'
    (frozen/'base.json').write_bytes(raw.encode())
    (workspace/'base.json').write_bytes(raw.replace('\n','\r\n').encode())
    save(workspace/'primary/launch.json',dict(config_sha256=sha(frozen/'base.json')))
    cfg=dict(arm='RL_SPACE',seeds=list(runner.SEEDS),base_config='base.json',
             primary_run='primary',prereg='added.md',files={'base.json':sha(workspace/'base.json')})
    save(tmp_path/'config.json',cfg)
    monkeypatch.setattr(runner,'ROOT',frozen)
    def verify(args):
        assert args.config==frozen/'base.json'
        return read(args.config)
    monkeypatch.setattr(runner.base,'verified_config',verify)
    args=NS(config=tmp_path/'config.json',workspace=workspace)
    _,value=runner.verified_config(args)
    assert value['prereg']=='added.md'
    save(frozen/'base.json',dict(prereg='changed.md'))
    with pytest.raises(ValueError,match='configuration contents differ'):
        runner.verified_config(args)


def test_resource_reservation_counts_primary_future_workers_and_external_run():
    g = runner.GIB
    cpus = runner.quota(total=62.6*g, available=40*g, primary_cpus=list(range(13)),
        primary_rss=[g]*13, extra_rss=[], external_rss=[2*g], occupied=[])
    assert cpus == [13]  # Available-memory headroom is tighter than the total-worker cap.
    # Primary between child launches still owns all thirteen slots.
    assert runner.quota(total=62.6*g, available=55*g, primary_cpus=list(range(13)),
        primary_rss=[g]*3, extra_rss=[], external_rss=[2*g], occupied=[]) == [13, 14]
    assert not runner.quota(total=62.6*g, available=35*g, primary_cpus=list(range(13)),
        primary_rss=[g]*13, extra_rss=[g,g], external_rss=[2*g], occupied=[13,14])


def test_low_memory_pauses_launches_and_primary_completion_releases_capacity():
    g = runner.GIB
    assert not runner.quota(total=62.6*g, available=10*g, primary_cpus=list(range(13)),
        primary_rss=[g]*13, extra_rss=[], external_rss=[2*g], occupied=[])
    cpus = runner.quota(total=62.6*g, available=60*g, primary_cpus=[], primary_rss=[],
        extra_rss=[], external_rss=[2*g], occupied=[])
    assert len(cpus) == 15 and all(0 <= c < 20 for c in cpus)


def test_drained_primary_queue_releases_only_idle_cpus_and_their_memory_reservation():
    configured = list(range(13))
    status = dict(state='running', phase='months', pending=0,
                  active=[dict(cpu=c) for c in (1, 4, 6, 9, 12)])
    reserved = runner.reserved_primary_cpus(status, configured)
    assert reserved == [1, 4, 6, 9, 12]
    g = runner.GIB
    cpus = runner.quota(total=62.6*g, available=40*g, primary_cpus=reserved,
        primary_rss=[3*g]*5, extra_rss=[], external_rss=[2*g], occupied=[0, 23])
    assert len(cpus) == 10 and not set(cpus) & {0, 1, 4, 6, 9, 12, 23}
    status['pending'] = 1
    assert runner.reserved_primary_cpus(status, configured) == configured
    del status['pending']
    assert runner.reserved_primary_cpus(status, configured) == configured


def test_drained_queue_rejects_unexpected_or_shared_primary_cpu():
    for cpus in ([2, 2], [23]):
        status = dict(state='running', phase='months', pending=0,
                      active=[dict(cpu=c) for c in cpus])
        with pytest.raises(ValueError, match='active CPUs'):
            runner.reserved_primary_cpus(status, list(range(13)))


def test_recovered_completed_months_are_counted_but_never_relaunched(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'resources', lambda *a: ([4], dict(primary_failed=False)))
    monkeypatch.setattr(runner.time, 'sleep', lambda _: None)
    retained = [dict(month=dict(seed=20_000_000, arm='RL_SPACE'), state='completed')]
    launched = []
    class Process:
        pid = 123
        def __init__(self, cmd, **kwargs):
            seed = int(cmd[cmd.index('--seed')+1])
            launched.append(seed)
            assert '--recover-from' not in cmd
            save(tmp_path/'months'/str(seed)/'RL_SPACE/completion.json',
                 dict(month=dict(seed=seed, arm='RL_SPACE'), state='completed'))
        def poll(self):
            return 0
    monkeypatch.setattr(runner.subprocess, 'Popen', Process)
    args = NS(workspace=tmp_path, config=tmp_path/'config', out=tmp_path,
              recover_from=tmp_path/'old')
    done = runner.run_jobs(args, {}, [20_100_000], retained=retained)
    assert launched == [20_100_000] and done[0] == retained[0]
    status = read(tmp_path/'progress.json')
    assert (status['planned'], status['completed'], status['retained']) == (2, 2, 1)


def test_block_only_rejects_temporal_actions_even_with_passed_generic_audit(tmp_path):
    result = dict(arm='RL_SPACE', time=0, space=4, days=[dict(n_time=0)])
    save(tmp_path/'result.json',result)
    save(tmp_path/'completion.json',dict(audit=dict(passed=True),
        result_sha256=sha(tmp_path/'result.json')))
    assert runner.validate_block_only(tmp_path,smoke=True)['passed']
    result['time']=1
    save(tmp_path/'result.json',result)
    save(tmp_path/'completion.json',dict(audit=dict(passed=True),
        result_sha256=sha(tmp_path/'result.json')))
    with pytest.raises(ValueError,match='Block-only path check failed'):
        runner.validate_block_only(tmp_path)


def test_diagnostic_must_exercise_spatial_action(tmp_path):
    save(tmp_path/'result.json',dict(arm='RL_SPACE',time=0,space=0,days=[dict(n_time=0)]))
    save(tmp_path/'completion.json',dict(audit=dict(passed=True),
        result_sha256=sha(tmp_path/'result.json')))
    with pytest.raises(ValueError,match='spatial_path_exercised'):
        runner.validate_block_only(tmp_path,smoke=True)
    assert runner.validate_block_only(tmp_path)['passed']  # Quiet real months are not selected out.


def test_failed_addon_worker_drains_other_worker_and_preserves_pending(tmp_path,monkeypatch):
    launched=[]
    def resources(args,cfg,active):
        used={c for _,c in active.values()}
        return [c for c in (13,14) if c not in used],dict(primary_failed=False)
    monkeypatch.setattr(runner,'resources',resources)
    monkeypatch.setattr(runner.time,'sleep',lambda _:None)
    class Process:
        def __init__(self,cmd,**kwargs):
            self.seed=int(cmd[cmd.index('--seed')+1]);self.pid=100+len(launched)
            self.polls=0;self.killed=False;launched.append(self)
        def poll(self):
            self.polls+=1
            if self.seed==1:return 1
            if self.polls==1:return None
            save(tmp_path/'months'/str(self.seed)/'RL_SPACE/completion.json',dict(completed=True))
            return 0
        def terminate(self):self.killed=True
        def kill(self):self.killed=True
        def wait(self,timeout):return 0
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    args=NS(workspace=tmp_path,config=tmp_path/'config',out=tmp_path)
    with pytest.raises(RuntimeError,match='other active runs finished'):
        runner.run_jobs(args,{},[1,2,3])
    assert len(launched)==2 and not any(p.killed for p in launched)
    outcomes=read(tmp_path/'months-outcomes.json')
    assert len(outcomes['completed'])==1 and outcomes['pending']==[3]


def test_existing_block_arm_wires_space_enabled_and_time_disabled():
    from yard_rl.v3.stage.episode import _Ctx
    ctx=_Ctx(seller_net=None,buyer_net=None,layout=None,announcer=None,arm='RL_SPACE',
             grid_s=60,window_s=10800,explore=0,seed=1,episode_end_s=86400,cf_horizon_s=10800)
    market=ctx.make_market(NS())
    bridge=ctx.make_bridge(market,orders={},records={},on_decision=None)
    assert not market.seller.no_space and bridge.no_time
    assert not market.buyer.always_buy  # Acceptance policy remains in place.
