"""YR-331-c: how often does the legacy vessel plan leave a block short, and does the repair fix it?

Planned ledger only (same totals the fixed-seed generator checks). Seeds are arbitrary,
not the reserved diagnostic range; writes outputs/reports/yr331_c_supply/supply-sweep.json.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from yard_rl.v6.stage.month import (build_month, initial_stock_by_block, plan_days,
                                    plan_month_vessels, truck_net_by_block)
from yard_rl.v6.world.integrated.terminal_stream import OBS_24H
from yard_rl.v6.world.integrated.yard_layout import terminal_layout

PATTERNS = {'3d': [3500, 5000, 7500], '5d': [3500, 5000, 7500, 12500, 3500],
            '9d': [3500, 3500, 5000, 5000, 7500, 12500, 15000, 3500, 3500]}


def spare(built, ships):
    s = Counter(initial_stock_by_block(built))
    for e in built['schedule']:
        s[e['block']] += -1 if e['flow'] == 'GATE_OUT' else 1
    for rows in ships.values():
        for r in rows:
            s[r['block']] += r['moves'] if r['work'] == 'DISCHARGE' else -r['moves']
    return s


def main(n):
    layout, rows = terminal_layout(), []
    for name, loads in PATTERNS.items():
        for k in range(n):
            seed = 5_000_000 + 100_000 * k
            days = plan_days(seed, loads=loads)
            built = build_month(seed, days=days)
            tn = truck_net_by_block(built['schedule'])
            old = plan_month_vessels(days, layout, obs=OBS_24H, truck_net=tn)
            new = plan_month_vessels(days, layout, obs=OBS_24H, truck_net=tn,
                                     initial_stock=initial_stock_by_block(built))
            so, sn = spare(built, old), spare(built, new)
            changed = sum(o != n_ for d in old for o, n_ in zip(old[d], new[d]))
            rows.append(dict(pattern=name, seed=seed, legacy_short={b: v for b, v in so.items() if v < 0},
                             repaired_short={b: v for b, v in sn.items() if v < 0},
                             changed_streams=changed, unchanged=old == new,
                             repaired_max_end_stock=max(sn.values())))
            print(name, seed, rows[-1]['legacy_short'], '->', rows[-1]['repaired_short'], changed, flush=True)
    summary = {}
    for name in PATTERNS:
        rs = [r for r in rows if r['pattern'] == name]
        summary[name] = dict(seeds=len(rs), legacy_short_seeds=sum(bool(r['legacy_short']) for r in rs),
                             repaired_short_seeds=sum(bool(r['repaired_short']) for r in rs),
                             changed_when_not_short=sum(not r['legacy_short'] and not r['unchanged'] for r in rs),
                             max_end_stock=max(r['repaired_max_end_stock'] for r in rs))
    out = Path('outputs/reports/yr331_c_supply/supply-sweep.json')
    out.write_text(json.dumps(dict(summary=summary, rows=rows), ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
