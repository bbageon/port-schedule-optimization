"""Policy-independent quantity correction for a synthetic vessel manifest.

Keep truck demand, initial stock, vessel stream identities/times/volumes fixed.
Only direction assignments may change, and only in blocks whose requested final
stock is outside [0, capacity]. This is an input variant, not an online policy
guard or proof that the jobs can all be served on time.
"""
from collections import defaultdict
from copy import deepcopy


def balance_vessel_supply(vessels, schedule, initial, capacities):
    rows = deepcopy(vessels)
    by_block = defaultdict(list)
    daily_delta = defaultdict(lambda: defaultdict(int))
    before = dict(initial)
    for entry in schedule:
        if entry['flow'] not in ('GATE_IN', 'GATE_OUT'):
            raise ValueError('Unknown truck flow')
        before[entry['block']] += 1 if entry['flow'] == 'GATE_IN' else -1
        day = int(entry.get('day', entry.get('arrival_s', 0) // 86400))
        daily_delta[day][entry['block']] += 1 if entry['flow'] == 'GATE_IN' else -1
    for streams in rows.values():
        for row in streams:
            if row['work'] not in ('DISCHARGE', 'LOAD') or row['moves'] <= 0:
                raise ValueError('Invalid vessel stream')
            before[row['block']] += row['moves'] * (1 if row['work'] == 'DISCHARGE' else -1)
            day = int(row.get('day', row['start_s'] // 86400))
            daily_delta[day][row['block']] += row['moves'] * (1 if row['work'] == 'DISCHARGE' else -1)
            by_block[row['block']].append(row)
    changes, after = [], dict(before)
    for block in sorted(initial):
        balance, cap = before[block], capacities[block]
        if not 0 <= initial[block] <= cap:
            raise ValueError(f'{block}: invalid initial stock/capacity')
        if 0 <= balance <= cap:
            continue
        candidates = sorted(by_block[block], key=lambda r: (r['start_s'], r['key']))
        # Exact subset sum: first minimize changed streams, then distance from
        # initial inventory; deterministic earlier-stream tie break. No seed,
        # policy outcome, or runtime state enters this input-only calculation.
        states = {0: ()}
        by_day = defaultdict(list)
        for i, row in enumerate(candidates):
            by_day[int(row.get('day', row['start_s'] // 86400))].append((i, row))
        planned = initial[block]
        for day in sorted(daily_delta):
            for i, row in by_day[day]:
                additions = {}
                change = 2 * row['moves'] * (1 if row['work'] == 'LOAD' else -1)
                for delta, selected in states.items():
                    new_delta, pick = delta + change, selected + (i,)
                    old = additions.get(new_delta, states.get(new_delta))
                    if old is None or (len(pick), pick) < (len(old), old):
                        additions[new_delta] = pick
                states.update(additions)
            planned += daily_delta[day][block]
            # Daily planned stock is a necessary count check, not a physical
            # service prediction. Do not repair final supply by creating a new
            # intermediate over-capacity/negative-stock violation.
            states = {delta:pick for delta,pick in states.items()
                      if min(0, planned) <= planned + delta <= max(cap, planned)}
        feasible = [(len(pick), abs(balance + delta - initial[block]), pick, delta)
                    for delta, pick in states.items() if 0 <= balance + delta <= cap]
        if not feasible:
            raise ValueError(f'{block}: no feasible direction-only assignment; retain the seed and report failure')
        _, _, selected, delta = min(feasible)
        for i in selected:
            row = candidates[i]
            old = row['work']
            row['work'] = 'DISCHARGE' if old == 'LOAD' else 'LOAD'
            row['type_offset'] = 0 if row['work'] == 'DISCHARGE' else 1
            changes.append({'key': row['key'], 'block': block, 'moves': row['moves'],
                            'start_s': row['start_s'], 'old_work': old, 'new_work': row['work']})
        after[block] = balance + delta
    return rows, {'mode': 'COUNT_BALANCED', 'original_end_balance': before,
                  'corrected_end_balance': after, 'changes': changes,
                  'quantity_feasible': all(0 <= n <= capacities[b] for b, n in after.items()),
                  'timely_service_proven': False,
                  'scope': 'original assignments only; policy reallocation can still create local shortages'}
