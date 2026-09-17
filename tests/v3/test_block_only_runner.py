from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/v3'))
import run_block_only_evaluation as runner
from independent_eval_checks import read, save, sha


def test_resource_reservation_counts_primary_future_workers_and_external_run():
    g = runner.GIB
    cpus = runner.quota(total=62.6*g, available=38*g, primary_cpus=list(range(13)),
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
