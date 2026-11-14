"""YR-332-al read-only definition audit: multi-container truck visits in the development data.

Uses the ak physical-visit definition. Writes the visit-pattern distribution (block stops with
inbound/outbound counts), the chassis evidence (20/40 ft, chassis positions), sequence and exit
facts, and the real order/visit clustering targets. Aggregate counts only, no identifiers.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics as st

from benchmark_default_eval import save
from six_step_learning import sha
import audit_visit_units as audit

OUT = Path('outputs/reports/yr332_visit_links')


def size_class(row):
    t = row['con_type'] or ''
    return '20' if t[:1] == '2' else '40' if t[:1] == '4' else 'unknown'


def pattern_of(rows):
    blocks = [audit.block_of(r) for r in rows]
    if None in blocks:
        return None
    by = {}
    for r, b in zip(rows, blocks):
        k = by.setdefault(b, [0, 0])
        k[0 if r['in_out'] == 'IN' else 1] += 1
    return tuple(sorted(tuple(x) for x in by.values()))


def index(strata, width):
    bins = 3600 // width
    vals = []
    for xs in strata.values():
        if len(xs) < 2:
            continue
        c = Counter(min(bins - 1, int(x // width)) for x in xs)
        vals.append(st.pvariance([c[i] for i in range(bins)]) / (len(xs) / bins))
    return st.mean(vals)


def main():
    rows, manifest = audit.load_rows()
    entered = [r for r in rows if r['gate_in_time']]
    visits = audit.physical_visits(audit.gate_records(entered), audit.MERGE_S)
    patterns, excluded, sizes, chassis = Counter(), Counter(), Counter(), Counter()
    unknown = Counter()
    for v in visits:
        p = pattern_of(v['rows'])
        if p is None:
            unknown[len(v['rows'])] += 1
            continue
        if sum(s[0] for s in p) > 2 or sum(s[1] for s in p) > 2:
            excluded[p] += 1
            continue
        patterns[p] += 1
        n_in = sum(r['in_out'] == 'IN' for r in v['rows'])
        n_out = len(v['rows']) - n_in
        for flow, n in (('IN', n_in), ('OUT', n_out)):
            if n == 2:
                sizes[(flow, '+'.join(sorted(size_class(r) for r in v['rows'] if r['in_out'] == flow)))] += 1
        if len(v['rows']) == 2 and n_in == 1:
            sizes[('IN+OUT', '+'.join(size_class(r) for r in sorted(v['rows'], key=lambda r: r['in_out'])))] += 1
        if n_out == 2 and n_in == 0:
            chassis['+'.join(sorted(r['chassis_loc'] or '-' for r in v['rows']))] += 1
    known = sum(patterns.values())
    singles = patterns[((0, 1),)] + patterns[((1, 0),)]
    multi = {p: c for p, c in patterns.items() if sum(map(sum, p)) > 1}
    orders_known = sum(c * sum(map(sum, p)) for p, c in patterns.items())
    orders_multi = sum(c * sum(map(sum, p)) for p, c in multi.items())
    same = sum(c for p, c in multi.items() if len(p) == 1)
    mixed = sum(c for p, c in multi.items() if sum(s[0] for s in p) and sum(s[1] for s in p))
    # Clustering targets, ak definitions: gate level (all orders / visits) and per block.
    gate_orders, gate_visits = defaultdict(list), defaultdict(list)
    block_orders, block_visits = defaultdict(list), defaultdict(set)
    for n, v in enumerate(visits):
        t = v['arrival'];x = t.minute * 60 + t.second;k = (t.date(), t.hour)
        gate_visits[k].append(x)
        for r in v['rows']:
            gate_orders[k].append(x)
            b = audit.block_of(r)
            if b:
                block_orders[(*k, b)].append(x);block_visits[(*k, b)].add((n, x))
    block_visits = {k: [x for _, x in v] for k, v in block_visits.items()}
    targets = {f'{name}_{w}s': index(strata, w) for name, strata in
               (('gate_orders', gate_orders), ('gate_visits', gate_visits),
                ('block_orders', block_orders), ('block_visits', block_visits)) for w in (60, 600)}
    spec = dict(method='observed_visit_patterns', size_rule='chassis_two_teu', order_rule='inbound_first',
        patterns=[dict(stops=[list(s) for s in p], count=c) for p, c in sorted(patterns.items())],
        source=dict(audit_path=(OUT / 'link-definition.json').as_posix(),
            definition='Physical truck visits (ak); pattern = per-block [inbound, outbound] counts; '
                       'visits with an unknown block or more than two orders of one flow are excluded.',
            scope='Development data 2026-10-28..11-09 already viewed; aggregate counts only.'))
    data = dict(visits=len(visits), known_block_visits=known, unknown_block_visits=dict(unknown),
        excluded_over_capacity={'|'.join(map(str, p)): c for p, c in excluded.items()},
        single_visits=singles, multi_visits=sum(multi.values()),
        multi_visit_share=sum(multi.values()) / known, orders_known=orders_known,
        orders_in_multi_share=orders_multi / orders_known,
        multi_same_block_share=same / sum(multi.values()), multi_mixed_flow_share=mixed / sum(multi.values()),
        same_flow_pair_sizes={f'{a}:{b}': c for (a, b), c in sorted(sizes.items())},
        two_outbound_chassis_positions=dict(chassis),
        sequence=json.loads((Path('outputs/reports/yr332_visit_arrivals') / 'visit-unit-audit.json')
                            .read_text(encoding='utf-8'))['relations']['observed_sequence'],
        clustering_targets=targets, spec=spec,
        rules=dict(capacity='One 40 ft or two 20 ft per chassis: two same-flow orders must both be 20 ft.',
                   order='Inbound drop-offs before outbound pick-ups (same block 1,716 vs 21; '
                         'different blocks outbound-first 0).',
                   exit='One gate exit after the last job (same exit time in 4,944 of 5,050 recorded multi visits).'),
        source_sha256=manifest, script_sha256=sha(Path(__file__)))
    save(OUT / 'link-definition.json', data)
    print(json.dumps({k: data[k] for k in ('visits', 'known_block_visits', 'multi_visits', 'multi_visit_share',
        'orders_in_multi_share', 'multi_same_block_share', 'multi_mixed_flow_share', 'same_flow_pair_sizes',
        'two_outbound_chassis_positions', 'clustering_targets', 'excluded_over_capacity')}, indent=1))


if __name__ == '__main__':
    main()
