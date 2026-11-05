"""Save the causal regression fixture's raw outcomes; no learned policy claim."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    fixture = Path('tests/v7/test_discharge_backpressure.py')
    ns = runpy.run_path(str(fixture))
    rows = []
    for ships in (1, 2):
        for cap in (None, 1, 3, 6):
            for delay in (0, 600):
                sim = ns['run_discharge'](cap=cap, ships=ships, yard_delay=delay)
                rows.append(dict(ships=ships, cap=cap, yard_delay_s=delay, **ns['result'](sim)))
    result = dict(schema='yard.v7.discharge-probe.v1', seed=0,
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        tracked_dirty=bool(subprocess.check_output(
            ['git', 'status', '--short', '--untracked-files=no'], text=True).strip()),
        fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
        parameters=dict(moves_per_ship=8, transfer_units=3, transfer_time_s=30,
                        sts_interval_s=20, horizon_s=8000, policy='SF_SPT'), rows=rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for r in rows:
        print(r)
