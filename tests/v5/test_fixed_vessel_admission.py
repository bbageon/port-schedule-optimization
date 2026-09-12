"""Fixed manifests must fail atomically, including without a journal callback."""
import copy
from types import SimpleNamespace as NS

import pytest

from yard_rl.v5.stage.container_contract import ContainerContractError
from yard_rl.v5.stage.month_engine import inject_vessel
from yard_rl.v5.world.integrated.multiblock import TransferError


@pytest.mark.parametrize('targets,asked,reserved', [
    (None, 1, False), ('A', 1, False), (['A', 'A'], 2, False),
    (['A', 'UNKNOWN'], 2, False), (['A'], 2, False), ([12], 1, False),
    (['A'], 1, True),
])
def test_invalid_fixed_manifest_changes_no_world_state(targets, asked, reserved):
    sim = NS(clock=0, end=86400, vessels={}, queue=NS(items=[]),
             stacks=NS(containers={'A': 'box A', 'B': 'box B'}),
             jobs={'owner': NS(target_container='A')} if reserved else {})
    mbt = NS(blocks={'B1': sim})
    before = copy.deepcopy(mbt)
    row = dict(work='LOAD', start_s=600, cadence_s=120, moves=asked, targets=targets)
    with pytest.raises(TransferError, match='fixed vessel target'):
        inject_vessel(mbt, 'B1', row, key='ship', size_seed='unit')
    assert mbt == before


@pytest.mark.parametrize('failure', ['truck', 'vessel'])
def test_missing_journal_callback_does_not_allow_failed_work_to_continue(
        monkeypatch, fixed_container_input, failure):
    from yard_rl.v5.ppo.continuous import make_plan
    from yard_rl.v5.ppo.model import BlockPolicy
    from yard_rl.v5.ppo.runtime import PPORuntime
    from yard_rl.v5.stage import month_run

    rt = PPORuntime(BlockPolicy(), stop_s=60)
    if failure == 'vessel':
        def bad_vessel(*args, **kwargs):
            raise TransferError('fixed manifest cannot be admitted')
        monkeypatch.setattr(month_run, 'inject_vessel', bad_vessel)
        expected, message = TransferError, 'fixed manifest'
    else:
        build, bind = month_run.build_month, PPORuntime.bind
        def bad_capacity(self, *args):
            bind(self, *args)
            self.mbt.capacity_margin = 100000
        def first_delivery(*args, **kwargs):
            data = build(*args, **kwargs)
            e = dict(data['schedule'][0], flow='GATE_IN', target=None,
                     arrival_s=60, lead_s=60)
            e['con_no'] = 'IN_' + e['job_id']
            data['schedule'] = [e]
            return data
        monkeypatch.setattr(PPORuntime, 'bind', bad_capacity)
        monkeypatch.setattr(month_run, 'build_month', first_delivery)
        expected, message = ContainerContractError, 'Fixed truck admission failed'
    with pytest.raises(expected, match=message):
        month_run.run_month(seed=9900306, days=make_plan(9900306, 3, 1), ppo=rt)
    assert not rt.updates and not rt.buffer
