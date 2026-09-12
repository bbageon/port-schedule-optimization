"""Generate fixed-cargo-v2 data only; refuse dirty source and existing output."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

from yard_rl.v5.ppo.continuous import make_plan
from yard_rl.v5.ppo.provenance import code_stamp
from yard_rl.v5.stage.container_contract import audit_container_plan
from yard_rl.v5.stage.fixed_seed import RULES, build_fixed_seed, canonical_bytes, sha
from yard_rl.v5.stage.month import build_month, plan_month_vessels, truck_net_by_block
from yard_rl.v5.stage.seed_bundle import load_seed_bundle, save_seed_bundle
from yard_rl.v5.world.integrated.profiles import build_h21_profile
from yard_rl.v5.world.integrated.yard_layout import terminal_layout


def generate(output, seed, days, load):
    stamp = code_stamp()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def write(name, value):
        (out / name).write_bytes(canonical_bytes(value) + b'\n')
    profile, layout = build_h21_profile(), terminal_layout()
    rule_path = '.claude/docs/dashboard-task-specs/YR-306-fixed-seed-regeneration.md'
    rule_text = (Path(__file__).resolve().parents[2] / rule_path).read_text(encoding='utf-8')
    write('manifest.json', dict(code=stamp, seed=seed, days=[d.as_dict() for d in days],
        debug_load=load, profile=asdict(profile), rules=RULES, cpus=sorted(os.sched_getaffinity(0)),
        preregistration=rule_path, preregistration_commit='1469b02',
        amended_rules_at_source_commit=stamp.get('git_head'),
        rules_sha256_lf=hashlib.sha256(rule_text.encode()).hexdigest(),
        simulation_steps=0, learning_updates=0))
    try:
        built = build_month(seed, days=days, profile=profile, layout=layout)
        vessels = plan_month_vessels(days, layout, truck_net=truck_net_by_block(built['schedule']))
        before = audit_container_plan(built, vessels)
        write('baseline-audit.json', before)
        document, audit = build_fixed_seed(built, vessels, seed=seed)
        # No clock/workload fields may be hidden inside "target repair".
        normalized = lambda e: canonical_bytes({k: v for k, v in e.items()
            if k not in ('target', 'con_no', 'size_ft40', 'size_class')})
        initial_unchanged, changed_block_labels = True, 0
        for bid, scenario in built['day0']['scenarios'].items():
            exported = document['initial_inventory'][bid]
            initial_unchanged &= len(exported) == len(scenario.containers)
            for row in exported:
                original = asdict(scenario.containers[row['container_id']])
                original['special_flags'] = sorted(original['special_flags'])
                changed_block_labels += original['block'] != bid
                initial_unchanged &= row['block'] == bid and (
                    {k: v for k, v in original.items() if k != 'block'} ==
                    {k: v for k, v in row.items() if k != 'block'})
        counts = dict(
            truck_fields_unchanged=all(normalized(a) == normalized(b)
                for a, b in zip(built['schedule'], document['schedule'], strict=True)),
            incoming_sizes_unchanged=all(a['size_ft40'] == b['size_ft40']
                and canonical_bytes(a.get('size_class')) == canonical_bytes(b.get('size_class'))
                for a, b in zip(built['schedule'], document['schedule'], strict=True) if a['flow']=='GATE_IN'),
            initial_physical_inventory_unchanged=initial_unchanged,
            vessel_fields_unchanged=all(
                canonical_bytes({k:v for k,v in a.items() if k!='targets'}) ==
                canonical_bytes({k:v for k,v in b.items() if k!='targets'})
                for day, rows in vessels.items()
                for a,b in zip(rows, document['vessels_by_day'][str(day)], strict=True)),
            day_plan_unchanged=[d.as_dict() for d in days] == document['days'])
        if not all(counts.values()):
            raise AssertionError(f'Unregistered input change: {counts}')
        changed_sizes = sum(a['size_ft40'] != b['size_ft40'] for a,b in
                            zip(built['schedule'], document['schedule'], strict=True))
        artifact = save_seed_bundle(document, out / 'seed-data.json.gz')
        loaded, read_audit = load_seed_bundle(out / 'seed-data.json.gz', expected_sha256=artifact['sha256'])
        if loaded != document:
            raise AssertionError('Saved input differs from generated input')
        if read_audit != audit:
            raise AssertionError('Read-back audit differs from original audit')
        report = dict(state='generated_and_validated', code=stamp, seed=seed, audit=audit,
            checks=counts, artifact=artifact, raw_schedule_sha256=sha(built['schedule']),
            raw_vessels_sha256=sha(vessels), initial_inventory_sha256=sha(document['initial_inventory']),
            normalized_initial_block_labels=changed_block_labels,
            changed_outbound_size_metadata=changed_sizes, elapsed_s=time.monotonic()-started,
            simulation_steps=0, learning_updates=0, ready_for_training=False)
        write('audit.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    except Exception as error:
        write('failure.json', dict(error=f'{type(error).__name__}: {error}',
            details=getattr(error, 'report', None), elapsed_s=time.monotonic()-started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=9900306)
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--debug-load', type=int)
    args = parser.parse_args()
    generate(args.output, args.seed, make_plan(args.seed, args.days, args.debug_load), args.debug_load)
