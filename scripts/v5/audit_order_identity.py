"""Compare public order IDs and physical IDs without running or training a world."""
import argparse
from collections import Counter
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=9900306)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.source_root.resolve()
    sys.path.insert(0, str(root / 'src'))
    from yard_rl.v3.schema import Order as V3Order
    from yard_rl.v5.schema import Order
    from yard_rl.v5.ppo.provenance import code_stamp
    from yard_rl.v5.stage.month import build_month, plan_month_vessels, truck_net_by_block
    from yard_rl.v5.stage.orders import orders_from_schedule
    from yard_rl.v5.world.integrated.yard_layout import terminal_layout

    stamp = code_stamp()
    data = build_month(args.seed)
    orders, _ = orders_from_schedule(data)
    names = [cid for scn in data['day0']['scenarios'].values() for cid in scn.containers]
    pickups = Counter((e['block'], e['target']) for e in data['schedule'] if e['flow'] == 'GATE_OUT')
    mismatches = []
    for e in data['schedule']:
        physical = e['target'] if e['flow'] == 'GATE_OUT' else f"IN_{e['job_id']}"
        if orders[e['job_id']].con_no != physical:
            mismatches.append(dict(order=e['job_id'], public=orders[e['job_id']].con_no,
                                   physical=physical, flow=e['flow']))
    vessels = plan_month_vessels(data['days'], terminal_layout(),
                               truck_net=truck_net_by_block(data['schedule']))
    loads = [row for rows in vessels.values() for row in rows if row['work'] == 'LOAD']
    result = dict(code=stamp, seed=args.seed, days=[d.as_dict() for d in data['days']],
                  mode='input-audit-only; zero simulation and learning steps',
                  cpus=sorted(os.sched_getaffinity(0)),
                  audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  schema_fields_v3=[f.name for f in fields(V3Order)],
                  schema_fields_v5=[f.name for f in fields(Order)],
                  truck_orders=len(orders), identity_mismatches=len(mismatches),
                  mismatch_examples=mismatches[:6], initial_count=len(names),
                  initial_distinct_ids=len(set(names)), pickup_count=sum(pickups.values()),
                  distinct_block_targets=len(pickups),
                  repeated_pickup_assignments=sum(n - 1 for n in pickups.values()),
                  vessel_load_streams=len(loads), vessel_load_moves=sum(r['moves'] for r in loads),
                  vessel_load_streams_with_fixed_targets=sum('targets' in r for r in loads))
    try:
        from yard_rl.v5.stage.container_contract import audit_container_plan
    except ModuleNotFoundError as error:
        if error.name != 'yard_rl.v5.stage.container_contract':
            raise
    else:
        result['container_contract'] = audit_container_plan(data, vessels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('days', 'mismatch_examples')},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
