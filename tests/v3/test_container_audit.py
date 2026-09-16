from types import SimpleNamespace as NS
from yard_rl.v3.stage.container_audit import ContainerFlowAudit
from yard_rl.v3.world.domain.enums import JobFlow, JobStatus, ContainerSize
from yard_rl.v3.world.domain.models import Job


def world(stock=()):
    return NS(blocks={'Y01': NS(stacks=NS(containers={c: None for c in stock}), jobs={})})


def done(key, flow, target=None, start=10, end=20):
    return Job(key,flow,0,None,None,target_container=target,
               inbound_size=ContainerSize.FT40 if flow == JobFlow.GATE_IN else None,
               status=JobStatus.DONE,service_start=start,service_end=end)


def test_same_day_supply_then_removal_survives_pruning():
    m=world(); audit=ContainerFlowAudit(m)
    m.blocks['Y01'].jobs={'out':done('out',JobFlow.GATE_OUT,'IN_in',start=30,end=40),
                          'in':done('in',JobFlow.GATE_IN)}
    audit.observe(m)
    m.blocks['Y01'].jobs.clear()
    summary,links=audit.finish(m)
    assert summary['passed'] and summary['completed_removals'] == 1
    assert links[0]['source']['source_job'] == 'in'
    assert summary['remaining_containers'] == 0


def test_phantom_container_and_wrong_physical_stock_are_detected():
    m=world(['C1']); audit=ContainerFlowAudit(m)
    m.blocks['Y01'].jobs={'out':done('out',JobFlow.GATE_OUT,'C2')}
    summary,_=audit.finish(m)
    assert not summary['passed'] and any('without source' in s for s in summary['issues'])
    m=world(['C1']); audit=ContainerFlowAudit(m)
    m.blocks['Y01'].stacks.containers.clear()
    assert not audit.finish(m)[0]['passed']


def test_future_supply_and_double_removal_cannot_pass():
    m=world(); audit=ContainerFlowAudit(m)
    m.blocks['Y01'].jobs={'in':done('in',JobFlow.GATE_IN,end=50),
                         'out':done('out',JobFlow.GATE_OUT,'IN_in',start=10,end=20),
                         'out2':done('out2',JobFlow.GATE_OUT,'IN_in',start=60,end=70)}
    summary,_=audit.finish(m)
    assert any('before physical supply' in s for s in summary['issues'])
    assert any('removed twice' in s for s in summary['issues'])
