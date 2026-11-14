"""여러 컨테이너를 처리하는 트럭 방문을 보정 입력의 오더 위에 연결한다 ([[YR-332-al]]).

■ 무엇을 하나
  aj·ak가 만든 하루 오더(시간대별 건수·유형·블록·반출 대상·규격 고정)를 **같은 시각대 안에서**
  실제 방문 유형 분포대로 묶는다. 묶인 오더는 한 트럭이다: 한 번 진입하고, 정해진 순서로
  블록을 돌고, 마지막 작업 뒤 한 번 출차한다. 시각대 안 도착 몰림은 오더가 아니라 **방문**에
  실제 분당 방문 수 형판(ak 조건 ③)을 적용한다.

■ 물리 계약 (원자료 근거 — outputs/reports/yr332_visit_links/link-definition.json)
  · 차대 용량: 40피트 1개 또는 20피트 2개. 같은 유형 2개 방문은 실제로 99% 이상 20피트 둘이다.
    그래서 반입 2개면 둘 다 20피트, 반출 2개면 반출 대상 둘 다 20피트여야 한다.
  · 반입 먼저: 실은 박스를 내려야 새 박스를 싣는다. 같은 블록 1,716 대 21, 다른 블록 반출 먼저 0.
  · 같은 블록·같은 유형 오더는 한 정차에서 함께 도착한다. 정차 안 작업 순서는 정하지 않는다.
  · 정차 사이 이동 = 블록 간 거리÷트럭 속도(같은 블록 0). 출차 주행 = 마지막 정차 오더의 값.

■ 바꾸지 않는 것
  시간대별 오더 수, 오더 순서·유형·블록·반출 대상·규격·주행 값, 예약 보정 규칙, 물리, 정책.
  유형을 맞출 오더가 그 시각대에 없으면 그 방문 유형은 만들지 않고 오더는 단독 방문으로 남긴다.
"""
from __future__ import annotations

from collections import Counter
import random

LINK_SCHEMA = 'observed-visit-links-v1'
LINK_METHODS = ('observed_visit_patterns',)
SIZE_RULE = 'chassis_two_teu'
ORDER_RULE = 'inbound_first'
MAX_PER_FLOW = 2
MATCH_TRIES = 8


def canonical(stops) -> tuple:
    return tuple(sorted(tuple(int(x) for x in s) for s in stops))


def validate_visit_link(spec) -> None:
    if not isinstance(spec, dict) or spec.get('method') not in LINK_METHODS:
        raise ValueError(f'visit_link.method must be one of {LINK_METHODS}')
    if spec.get('size_rule') != SIZE_RULE or spec.get('order_rule') != ORDER_RULE:
        raise ValueError('visit_link requires the chassis_two_teu size rule and inbound_first order rule')
    pats = spec.get('patterns')
    if not isinstance(pats, list) or not pats:
        raise ValueError('visit_link.patterns must be a non-empty list')
    seen = set()
    for p in pats:
        stops, count = p.get('stops'), p.get('count')
        if not isinstance(count, int) or count <= 0:
            raise ValueError('pattern count must be a positive int')
        if not isinstance(stops, list) or not 1 <= len(stops) <= 4:
            raise ValueError('pattern stops must list 1..4 blocks')
        if any(len(s) != 2 or min(s) < 0 or sum(s) < 1 for s in stops):
            raise ValueError('each block stop is [n_in, n_out] with at least one order')
        if sum(s[0] for s in stops) > MAX_PER_FLOW or sum(s[1] for s in stops) > MAX_PER_FLOW:
            raise ValueError('chassis capacity: at most two inbound and two outbound orders')
        key = canonical(stops)
        if key != tuple(tuple(s) for s in stops) or key in seen:
            raise ValueError('pattern stops must be canonical (sorted) and unique')
        seen.add(key)
    if not {((0, 1),), ((1, 0),)} <= seen:
        raise ValueError('single-order patterns [[0,1]] and [[1,0]] are required')


def _target_size20(built: dict) -> dict:
    """반출 대상 박스가 20피트인지 — 반출 오더의 규격은 실제 대상 박스가 정한다."""
    from .world.domain.enums import ContainerSize
    out = {}
    for e in built['schedule']:
        if e['flow'] == 'GATE_OUT':
            c = built['scenarios'][e['block']].containers[e['target']]
            out[e['job_id']] = c.size == ContainerSize.FT20
    return out


def _is20(e, out20) -> bool:
    return out20[e['job_id']] if e['flow'] == 'GATE_OUT' else not e['size_ft40']


def _match(pattern, pool, schedule, out20, rng):
    """한 방문 유형을 남은 오더로 채운다. 실패하면 None (오더는 그대로 남는다)."""
    n_in = sum(s[0] for s in pattern)
    n_out = sum(s[1] for s in pattern)
    need20 = {'GATE_IN': n_in == 2, 'GATE_OUT': n_out == 2}
    by_block: dict[str, dict[str, list[int]]] = {}
    for i in pool:
        e = schedule[i]
        if need20[e['flow']] and not _is20(e, out20):
            continue
        by_block.setdefault(e['block'], {'GATE_IN': [], 'GATE_OUT': []})[e['flow']].append(i)
    blocks = sorted(by_block)
    for _ in range(MATCH_TRIES):
        rng.shuffle(blocks)
        used_blocks, chosen = set(), []
        for n_i, n_o in sorted(pattern, key=lambda s: -sum(s)):
            ok = False
            for b in blocks:
                if b in used_blocks:
                    continue
                av = by_block[b]
                if len(av['GATE_IN']) >= n_i and len(av['GATE_OUT']) >= n_o:
                    chosen.append((b, rng.sample(av['GATE_IN'], n_i), rng.sample(av['GATE_OUT'], n_o)))
                    used_blocks.add(b)
                    ok = True
                    break
            if not ok:
                break
        if len(chosen) == len(pattern):
            return chosen
    return None


def _stops(chosen, rng, layout, schedule):
    """정차 순서: 반입 정차 → 반출 정차. 반입·반출이 함께 있는 블록은 반입 마지막·반출 처음."""
    ins = [(b, ix) for b, ix, _ in chosen if ix]
    outs = [(b, ox) for b, _, ox in chosen if ox]
    rng.shuffle(ins)
    rng.shuffle(outs)
    both = {b for b, _ in ins} & {b for b, _ in outs}
    ins.sort(key=lambda s: s[0] in both)
    outs.sort(key=lambda s: s[0] not in both)
    stops = [(b, 'GATE_IN', sorted(ix)) for b, ix in ins] + [(b, 'GATE_OUT', sorted(ox)) for b, ox in outs]
    travel = [float(schedule[stops[0][2][0]]['travel_s'])]
    for (a, _, _), (b, _, _) in zip(stops, stops[1:]):
        travel.append(0.0 if a == b else round(float(layout.block_to_block_s(a, b)), 3))
    exit_s = float(schedule[stops[-1][2][0]]['exit_travel_s'])
    return stops, travel, exit_s


def link_visits(seed, built: dict, spec: dict, within_spec: dict, layout) -> dict:
    """`built['schedule']`의 오더를 방문으로 묶고 방문 도착 시각을 정한다(제자리 수정).

    반환: 달성 통계(목표 대비 유형별 개수, 시각대별 방문 수, 실패한 유형 수).
    """
    from .demand_profile import within_hour_arrivals
    validate_visit_link(spec)
    if within_spec is None:
        raise ValueError('visit_link needs a within_hour arrival template for visit times')
    schedule = built['schedule']
    out20 = _target_size20(built)
    singles = sum(p['count'] for p in spec['patterns'] if sum(map(sum, p['stops'])) == 1)
    multi = [(canonical(p['stops']), p['count']) for p in spec['patterns'] if sum(map(sum, p['stops'])) > 1]
    keys = [k for k, _ in multi]
    weights = [singles] + [c for _, c in multi]
    hours: dict[int, list[int]] = {}
    for i, e in enumerate(schedule):
        hours.setdefault(int(e['arrival_s'] // 3600), []).append(i)
    visits = []                    # (hour, [stop tuples], travel, exit)
    drawn, made, failed = Counter(), Counter(), Counter()
    for h in sorted(hours):
        rng = random.Random(f'h21d:visit:{seed}:{h}')
        pool = list(hours[h])
        need, wish = len(pool), []
        while need > 0:
            k = rng.choices(range(len(weights)), weights=weights)[0]
            size = 1 if k == 0 else sum(map(sum, keys[k - 1]))
            if size > need:
                k, size = 0, 1
            if k:
                wish.append(keys[k - 1])
                drawn[keys[k - 1]] += 1
            need -= size
        rng.shuffle(wish)
        for pattern in sorted(wish, key=lambda p: -sum(map(sum, p))):
            chosen = _match(pattern, pool, schedule, out20, rng)
            if chosen is None:
                failed[pattern] += 1
                continue
            made[pattern] += 1
            taken = {i for _, a, b in chosen for i in (*a, *b)}
            pool = [i for i in pool if i not in taken]
            visits.append((h, *_stops(chosen, rng, layout, schedule)))
        for i in pool:
            e = schedule[i]
            visits.append((h, [(e['block'], e['flow'], [i])], [float(e['travel_s'])], float(e['exit_travel_s'])))
    per_hour = Counter(v[0] for v in visits)
    base = [3600.0 * h + 1800.0 for h in sorted(per_hour) for _ in range(per_hour[h])]
    # Same random stream as the ak order-level redraw: same date order per hour, fewer visits.
    times, meta = within_hour_arrivals(seed, base, within_spec)
    by_hour: dict[int, list[float]] = {}
    for t in times:
        by_hour.setdefault(int(t // 3600), []).append(t)
    order = sorted(range(len(visits)), key=lambda n: (visits[n][0], n))
    cursor = Counter()
    for h in sorted(per_hour):
        idx = [n for n in order if visits[n][0] == h]
        random.Random(f'h21d:visit-time:{seed}:{h}').shuffle(idx)
        for n in idx:
            visits[n] = (*visits[n], by_hour[h][cursor[h]])
            cursor[h] += 1
    for vid, (h, stops, travel, exit_s, t) in enumerate(visits):
        ids = [[b, flow, [schedule[i]['job_id'].split(':')[-1] for i in ix]] for b, flow, ix in stops]
        size = sum(len(ix) for _, _, ix in stops)
        for k, (_, _, ix) in enumerate(stops):
            for i in ix:
                schedule[i]['arrival_s'] = t
                if size > 1:
                    schedule[i]['visit'] = dict(id=f'V-{vid:05d}', stop=k, stops=ids,
                                                travel_s=travel, exit_travel_s=exit_s)
    sizes = Counter(sum(len(ix) for _, _, ix in v[1]) for v in visits)
    in_multi = sum(n * c for n, c in sizes.items() if n > 1)
    return dict(schema=LINK_SCHEMA, visits=len(visits), orders=len(schedule),
        orders_in_multi_visits=in_multi, multi_order_share=in_multi / len(schedule),
        visit_sizes={str(k): v for k, v in sorted(sizes.items())},
        drawn={'|'.join(map(str, k)): v for k, v in sorted(drawn.items())},
        made={'|'.join(map(str, k)): v for k, v in sorted(made.items())},
        failed={'|'.join(map(str, k)): v for k, v in sorted(failed.items())},
        same_block_multi=sum(1 for v in visits if sum(len(s[2]) for s in v[1]) > 1
                             and len({s[0] for s in v[1]}) == 1),
        within_hour=meta)
