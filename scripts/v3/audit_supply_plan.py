"""Read the entire frozen bank and save an explicit alternative vessel plan.

Never overwrite bundles or silently substitute these variants into evaluation.
No policy simulation/training and no selection based on policy performance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from yard_rl.v3.eval.seed_bank import load_bundle, digest
from yard_rl.v3.stage.supply_plan import balance_vessel_supply


def daily_bounds(initial, schedule, vessels):
    from collections import Counter
    daily = {}
    for e in schedule:
        daily.setdefault(e['day'], Counter())[e['block']] += 1 if e['flow']=='GATE_IN' else -1
    for day, rows in vessels.items():
        for r in rows:
            daily.setdefault(int(day),Counter())[r['block']] += r['moves']*(1 if r['work']=='DISCHARGE' else -1)
    stock = dict(initial)
    rows = []
    for day in sorted(daily):
        for b,n in daily[day].items():
            stock[b] += n
        rows.append({'day':day,'min_stock':min(stock.values()),'max_stock':max(stock.values()),
                     'negative_blocks':{b:v for b,v in stock.items() if v<0}})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    summary = json.loads((args.bank/'summary.json').read_text(encoding='utf-8'))
    results = []
    for original in summary['rows']:
        source = args.bank/original['bundle']
        data = load_bundle(source, expected_sha256=original['bundle_sha256'])
        # This frozen profile has 24 bays * 10 rows * 6 tiers; use its stored
        # geometry rather than a new assumed capacity.
        geometry = data['profile']['block']
        capacities = {b: geometry['bay_count'] * geometry['row_count'] * geometry['tier_max']
                      for b in data['initial_scenarios']}
        initial = {b:len(s['containers']) for b,s in data['initial_scenarios'].items()}
        revised, report = balance_vessel_supply(data['vessels'], data['schedule'], initial, capacities)
        invariant = lambda rows: [{k:v for k,v in row.items() if k not in ('work','type_offset')}
                                 for day in sorted(rows,key=int) for row in rows[day]]
        assert invariant(revised) == invariant(data['vessels'])
        assert digest(data['schedule']) == original['schedule_sha256']
        assert digest(data['initial_scenarios']) == original['initial_scenarios_sha256']
        output = {'seed':data['seed'],'source_bundle_sha256':original['bundle_sha256'],
                  'source_schedule_sha256':original['schedule_sha256'],
                  'source_vessels_sha256':original['vessels_sha256'],
                  'revised_vessels_sha256':digest(revised),'vessels':revised,'audit':report}
        output['original_daily_balances'] = daily_bounds(initial,data['schedule'],data['vessels'])
        output['revised_daily_balances'] = daily_bounds(initial,data['schedule'],revised)
        envelope_ok = all(new['max_stock'] <= max(max(capacities.values()), old['max_stock'])
                          and new['min_stock'] >= min(0, old['min_stock'])
                          for old,new in zip(output['original_daily_balances'],output['revised_daily_balances']))
        assert envelope_ok
        (args.out/f'seed-{data["seed"]}.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
        results.append({'seed':data['seed'],'changes':report['changes'],
                        'daily_envelope_passed':envelope_ok,
                        'quantity_feasible':report['quantity_feasible'],
                        'before_shortfall':sum(max(0,-v) for v in report['original_end_balance'].values()),
                        'after_shortfall':sum(max(0,-v) for v in report['corrected_end_balance'].values()),
                        'original_negative_days':sum(bool(d['negative_blocks']) for d in output['original_daily_balances']),
                        'revised_negative_days':sum(bool(d['negative_blocks']) for d in output['revised_daily_balances']),
                        'original_max_planned_stock':max(d['max_stock'] for d in output['original_daily_balances']),
                        'revised_max_planned_stock':max(d['max_stock'] for d in output['revised_daily_balances']),
                        'truck_requests':len(data['schedule'])})
        print(json.dumps(results[-1]),flush=True)
    result={'scope':'input-only direction correction candidate; no simulation',
            'months':results,'original_bundles_modified':False,'new_policy_runs':0,
            'new_training_runs':0,'claim_eligible':False,
            'analyzer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result['solver_sha256']=hashlib.sha256((ROOT/'src/yard_rl/v3/stage/supply_plan.py').read_bytes()).hexdigest()
    result['manifest_sha256']=hashlib.sha256((args.bank/'summary.json').read_bytes()).hexdigest()
    (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
