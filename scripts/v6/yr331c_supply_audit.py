"""YR-331-c: rebuild per-block source/exit ledgers before and after the vessel stock repair.

Writes outputs/reports/yr331_c_supply/{supply-audit.json, failing-input-9921000.json.gz}.
Input generation only: no simulator, policy or reward is executed.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from yard_rl.v6.stage.container_contract import ContainerContractError
from yard_rl.v6.stage.fixed_seed import build_fixed_seed, canonical_bytes, initial_inventory
from yard_rl.v6.stage.month import (build_month, initial_stock_by_block, plan_days,
                                    plan_month_vessels, truck_net_by_block)
from yard_rl.v6.world.integrated.terminal_stream import OBS_24H
from yard_rl.v6.world.integrated.yard_layout import terminal_layout

OUT = Path('outputs/reports/yr331_c_supply')
CASES = [(9921000, [3500, 5000, 7500]), (9911000, [3500, 5000, 7500]),
         (9931000, [3500, 5000, 7500, 12500, 3500]), (9941000, [3500, 5000, 7500, 12500, 3500]),
         (9951000, [3500, 5000, 7500, 12500, 3500]), (9961000, [3500, 7500, 3500]),
         (9971000, [3500, 7500, 3500]), (9981000, [3500, 7500, 3500])]


def sha(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def ledger(built, ships):
    init = initial_stock_by_block(built)
    gi, go, dis, load = Counter(), Counter(), Counter(), Counter()
    for e in built['schedule']:
        (go if e['flow'] == 'GATE_OUT' else gi)[e['block']] += 1
    for rows in ships.values():
        for r in rows:
            (dis if r['work'] == 'DISCHARGE' else load)[r['block']] += r['moves']
    return {b: dict(initial=init[b], gate_in=gi[b], discharge=dis[b], gate_out=go[b], load=load[b],
                    sources=init[b] + gi[b] + dis[b], exits=go[b] + load[b],
                    spare=init[b] + gi[b] + dis[b] - go[b] - load[b]) for b in sorted(init)}


def generate(built, ships, seed):
    try:
        doc, audit = build_fixed_seed(built, ships, seed=seed)
        return dict(passed=audit['passed'], content_sha256=sha(doc), sources=audit['sources'],
                    exits=audit['exits'], truck_orders=audit['truck_orders'],
                    planned_waiting_exits=audit['planned_waiting_exits'],
                    max_planned_wait_lower_bound_s=audit['max_planned_wait_lower_bound_s'])
    except ContainerContractError as exc:
        return dict(passed=False, error=str(exc), report=getattr(exc, 'report', None))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    layout, runs = terminal_layout(), []
    for seed, loads in CASES:
        days = plan_days(seed, loads=loads)
        built = build_month(seed, days=days)
        tn = truck_net_by_block(built['schedule'])
        legacy = plan_month_vessels(days, layout, obs=OBS_24H, truck_net=tn)
        guarded = plan_month_vessels(days, layout, obs=OBS_24H, truck_net=tn,
                                     initial_stock=initial_stock_by_block(built))
        changed = [dict(day=d, key=o['key'], block=o['block'], moves=o['moves'],
                        legacy=o['work'], repaired=n['work'])
                   for d in legacy for o, n in zip(legacy[d], guarded[d]) if o != n]
        lo, ln = ledger(built, legacy), ledger(built, guarded)
        row = dict(seed=seed, loads=loads, schedule_sha256=sha(built['schedule']),
                   truck_orders=len(built['schedule']), changed_streams=changed,
                   legacy=dict(generation=generate(built, legacy, seed),
                               short_blocks={b: v for b, v in lo.items() if v['spare'] < 0},
                               min_spare=min(v['spare'] for v in lo.values())),
                   repaired=dict(generation=generate(built, guarded, seed),
                                 short_blocks={b: v for b, v in ln.items() if v['spare'] < 0},
                                 min_spare=min(v['spare'] for v in ln.values()),
                                 max_end_stock=max(v['spare'] for v in ln.values())),
                   plan_identical=legacy == guarded)
        if changed:
            blocks = sorted({c['block'] for c in changed})
            row['ledger_changed_blocks'] = {b: dict(legacy=lo[b], repaired=ln[b],
                                                    truck_net_by_day=[tn.get(d.index, {}).get(b, 0) for d in days],
                                                    legacy_streams=[r for rows in legacy.values() for r in rows if r['block'] == b])
                                            for b in blocks}
        if seed == 9921000:
            failing = dict(seed=seed, loads=loads, days=[d.as_dict() for d in days],
                           initial_inventory=initial_inventory(built), schedule=built['schedule'],
                           vessels_by_day={str(d): rows for d, rows in sorted(legacy.items())},
                           note='YR-331-c original failing input (legacy vessel plan); build_fixed_seed refuses it')
            payload = canonical_bytes(json.loads(canonical_bytes(failing)))
            path = OUT / 'failing-input-9921000.json.gz'
            with path.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as gz:
                gz.write(payload)
            row['failing_input'] = dict(path=path.name, content_sha256=hashlib.sha256(payload).hexdigest())
        print(seed, 'changed', [(c['block'], c['moves']) for c in changed],
              'legacy', row['legacy']['generation']['passed'], 'repaired', row['repaired']['generation']['passed'])
        runs.append(row)
    (OUT / 'supply-audit.json').write_text(json.dumps(dict(
        task='YR-331-c', scope='input generation only; no simulation, policy or reward',
        rule='plan_month_vessels(initial_stock=...): if a block ends the month with planned sources < '
             'planned exits, its latest LOAD stream becomes DISCHARGE; repeat until none or no LOAD left',
        runs=runs), ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
