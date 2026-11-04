"""Restore the saved initial state and workload, without generating new targets."""
from .container_contract import ContainerContractError
from .fixed_seed import fixed_seed_audit
from .month import DAY_S
from ..world.domain.enums import ContainerSize, LoadStatus
from ..world.domain.models import Container
from ..world.integrated.scenario import TerminalScenario


def restore_input(document, *, seed, days, lead_mode):
    audit = fixed_seed_audit(document)
    if not audit['passed']:
        raise ContainerContractError('Invalid fixed seed input', report=audit)
    if (document['seed'] != seed or document['days'] != [d.as_dict() for d in days]
            or document['lead_mode'] != lead_mode):
        raise ContainerContractError('Seed/calendar/notice mode differs from saved data')
    scenarios = {}
    for bid, rows in document['initial_inventory'].items():
        boxes = {}
        for row in rows:
            box = Container(**(row | {'size': ContainerSize(row['size']),
                'load_status': LoadStatus(row['load_status']),
                'special_flags': frozenset(row['special_flags'])}))
            boxes[box.container_id] = box
        scenarios[bid] = TerminalScenario(f'fixed-{seed}-{bid}', seed,
            len(days)*DAY_S, 7200, boxes, [], [])
    # No generated scenario jobs/events are injected into the restored world.
    return dict(day0=dict(scenarios=scenarios), schedule=document['schedule'])
