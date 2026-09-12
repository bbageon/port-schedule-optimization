"""Explicit, small fixed-cargo fixtures for engine wiring tests.

Not the original 30-day workload and NOT evidence it is repaired. The production
input audit stays enabled: these tests supply genuinely unique pickup targets
and one day's fixed vessel manifests, instead of bypassing the audit.
"""
from collections import defaultdict

import pytest


@pytest.fixture
def fixed_container_input(monkeypatch):
    from yard_rl.v5.stage import month_run
    original_build, original_vessels = month_run.build_month, month_run.plan_month_vessels
    context = {}

    def build(*args, **kwargs):
        data = original_build(*args, **kwargs)
        assert len(data['schedule']) <= 2000, 'Small test fixture, never a production repair'
        available = {b: set(s.containers) for b, s in data['day0']['scenarios'].items()}
        used = defaultdict(set)
        for e in data['schedule']:
            if e['flow'] != 'GATE_OUT':
                continue
            bid = e['block']
            cid = e['target']
            if cid not in available[bid] or cid in used[bid]:
                cid = sorted(available[bid] - used[bid])[0]
            e['target'] = e['con_no'] = cid
            used[bid].add(cid)
        context.update(available=available, used=used)
        return data

    def vessels(*args, **kwargs):
        rows = original_vessels(*args, **kwargs)
        # Keep real vessel work on day one, no unbound replenishment later.
        for day, items in rows.items():
            if day != 0:
                rows[day] = []
                continue
            for row in items:
                if row['work'] == 'LOAD':
                    bid = row['block']
                    pool = sorted(context['available'][bid] - context['used'][bid])
                    assert len(pool) >= row['moves']
                    row['targets'] = pool[:row['moves']]
                    context['used'][bid].update(row['targets'])
        return rows

    monkeypatch.setattr(month_run, 'build_month', build)
    monkeypatch.setattr(month_run, 'plan_month_vessels', vessels)
