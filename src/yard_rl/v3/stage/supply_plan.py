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
    before = dict(initial)
    for entry in schedule:
        if entry['flow'] not in ('GATE_IN', 'GATE_OUT'):
            raise ValueError('Unknown truck flow')
        before[entry['block']] += 1 if entry['flow'] == 'GATE_IN' else -1
    for streams in rows.values():
        for row in streams:
            if row['work'] not in ('DISCHARGE', 'LOAD') or row['moves'] <= 0:
                raise ValueError('Invalid vessel stream')
            before[row['block']] += row['moves'] * (1 if row['work'] == 'DISCHARGE' else -1)
            by_block[row['block']].append(row)
    changes, after = [], dict(before)
    for block in sorted(initial):
        balance, cap = before[block], capacities[block]
        if not 0 <= initial[block] <= cap:
            raise ValueError(f'{block}: invalid initial stock/capacity')
        if 0 <= balance <= cap:
            continue
        sign = 1 if balance < 0 else -1
        original_work = 'LOAD' if sign == 1 else 'DISCHARGE'
        candidates = sorted((r for r in by_block[block] if r['work'] == original_work),
                            key=lambda r: (r['start_s'], r['key']))
        # Exact subset sum: first minimize changed streams, then distance from
        # initial inventory; deterministic earlier-stream tie break. No seed,
        # policy outcome, or runtime state enters this input-only calculation.
        max_moves = (cap - balance) // 2 if sign == 1 else balance // 2
        states = {0: ()}
        for i, row in enumerate(candidates):
            additions = {}
            for moves, selected in states.items():
                new_moves, pick = moves + row['moves'], selected + (i,)
                if new_moves > max_moves:
                    continue
                old = additions.get(new_moves, states.get(new_moves))
                if old is None or (len(pick), pick) < (len(old), old):
                    additions[new_moves] = pick
            states.update(additions)
        feasible = [(len(pick), abs(balance + 2 * sign * moves - initial[block]), pick, moves)
                    for moves, pick in states.items()
                    if 0 <= balance + 2 * sign * moves <= cap]
        if not feasible:
            raise ValueError(f'{block}: no feasible direction-only assignment; retain the seed and report failure')
        _, _, selected, moves = min(feasible)
        for i in selected:
            row = candidates[i]
            old = row['work']
            row['work'] = 'DISCHARGE' if sign == 1 else 'LOAD'
            row['type_offset'] = 0 if sign == 1 else 1
            changes.append({'key': row['key'], 'block': block, 'moves': row['moves'],
                            'start_s': row['start_s'], 'old_work': old, 'new_work': row['work']})
        after[block] = balance + 2 * sign * moves
    return rows, {'mode': 'COUNT_BALANCED', 'original_end_balance': before,
                  'corrected_end_balance': after, 'changes': changes,
                  'quantity_feasible': all(0 <= n <= capacities[b] for b, n in after.items()),
                  'timely_service_proven': False,
                  'scope': 'original assignments only; policy reallocation can still create local shortages'}
