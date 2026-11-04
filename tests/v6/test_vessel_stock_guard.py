"""[[YR-331-c]] 본선 적하가 블록의 월말 재고를 음수로 만들지 않는다.

시드 9921000 · 부하 3,500/5,000/7,500 의 Y08 은 첫날 적하 537을 받아
초기 648 + 반입 323 = 원천 971 < 반출 439 + 적하 537 = 976 이 되어 생성이 거부됐다.
`initial_stock` 을 주면 모자란 블록의 **가장 늦은 적하 스트림만** 양하로 돌린다.
모자람이 없는 계획은 그대로다. 주문·물량·상자는 바꾸지 않는다.
"""
from types import SimpleNamespace as NS

import pytest

from yard_rl.v6.stage import vessels as vessels_mod
from yard_rl.v6.stage.container_contract import ContainerContractError
from yard_rl.v6.stage.month import DayPlan, plan_month_vessels

LAYOUT = NS(ids=('B1', 'B2'))          # B2 의 수지 시작값은 +1 (홀수 순번)


def days(n):
    return [DayPlan(i, 3500, 'unit', 7000 + i, i * 86400.0, n) for i in range(n)]


@pytest.fixture
def streams(monkeypatch):
    """day seed → [(block, moves)] 를 스트림 행으로 돌려주는 가짜 배 계획."""
    table = {}

    def plan(fleet, layout, seed, obs=None):
        return [dict(vessel_id=f'V{seed}-{k}', stream=k, moves=m, start_s=0.0,
                     cadence_s=120.0, block=b) for k, (b, m) in enumerate(table.get(seed, []))]

    monkeypatch.setattr(vessels_mod, 'sample_day_vessels', lambda seed, obs=None: [])
    monkeypatch.setattr(vessels_mod, 'plan_streams', plan)
    return table


def works(plan):
    return [(r['day'], r['block'], r['work'], r['moves']) for d in sorted(plan) for r in plan[d]]


def end_stock(plan, initial, truck_net, block):
    v = sum(r['moves'] if r['work'] == 'DISCHARGE' else -r['moves']
            for rows in plan.values() for r in rows if r['block'] == block)
    return initial - sum(t.get(block, 0) for t in truck_net.values()) + v


Y08 = {0: {'B2': 1}, 1: {'B2': 44}, 2: {'B2': 71}}      # 원 시드의 날별 트럭 순반출


@pytest.mark.parametrize('initial, work', [(648, 'DISCHARGE'), (652, 'DISCHARGE'), (653, 'LOAD')])
def test_y08_shape_boundary(streams, initial, work):
    """월말 재고 −5(원 시드) · −1 은 양하로, 정확히 0 은 적하를 그대로 둔다."""
    streams[7000] = [('B2', 537)]
    legacy = plan_month_vessels(days(3), LAYOUT, truck_net=Y08)
    guarded = plan_month_vessels(days(3), LAYOUT, truck_net=Y08, initial_stock={'B1': 0, 'B2': initial})
    assert works(legacy) == [(0, 'B2', 'LOAD', 537)]               # 옛 규칙: net(+1) = goal(+1)
    assert works(guarded) == [(0, 'B2', work, 537)]
    assert end_stock(guarded, initial, Y08, 'B2') >= 0
    row = guarded[0][0]
    assert row['type_offset'] == (0 if work == 'DISCHARGE' else 1)
    assert row['key'] == legacy[0][0]['key'] and row['moves'] == 537  # 물량·이름 불변


def test_without_initial_stock_is_the_legacy_plan(streams):
    streams[7000] = [('B2', 537), ('B1', 200)]
    a = plan_month_vessels(days(3), LAYOUT, truck_net=Y08)
    b = plan_month_vessels(days(3), LAYOUT, truck_net=Y08, initial_stock={'B1': 10_000, 'B2': 10_000})
    assert a == b                                                   # 모자람이 없으면 한 줄도 안 바뀐다


def test_repairs_latest_load_first_and_repeats_until_no_shortage(streams):
    streams.update({7000: [('B2', 300)], 7001: [('B2', 50)]})
    net = {0: {}, 1: {'B2': -400}, 2: {'B2': 400}}
    legacy = plan_month_vessels(days(3), LAYOUT, truck_net=net)
    assert works(legacy) == [(0, 'B2', 'LOAD', 300), (1, 'B2', 'LOAD', 50)]
    assert end_stock(legacy, 10, net, 'B2') == -340
    guarded = plan_month_vessels(days(3), LAYOUT, truck_net=net, initial_stock={'B1': 0, 'B2': 10})
    # 마지막 적하(50)를 돌려도 −240 이라 그다음 늦은 적하(300)까지 돌린다.
    assert works(guarded) == [(0, 'B2', 'DISCHARGE', 300), (1, 'B2', 'DISCHARGE', 50)]
    assert end_stock(guarded, 10, net, 'B2') == 360


def test_truck_only_shortage_is_left_for_the_generator_to_refuse(streams):
    """적하가 없으면 고칠 수단이 없다 — 상자를 만들지 않고 그대로 돌려준다."""
    net = {0: {'B2': 100}}
    plan = plan_month_vessels(days(1), LAYOUT, truck_net=net, initial_stock={'B1': 0, 'B2': 50})
    assert plan == {0: []}


# 생성기의 거부 자체는 test_fixed_seed.py (missing=1 사례) 가 지킨다.


def test_original_failing_seed_9921000_now_generates():
    """원 실패 입력의 회귀 — Y08 첫날 적하 537 한 스트림만 양하로 바뀐다."""
    from yard_rl.v6.stage.fixed_seed import build_fixed_seed
    from yard_rl.v6.stage.month import build_month, initial_stock_by_block, plan_days, truck_net_by_block
    from yard_rl.v6.world.integrated.terminal_stream import OBS_24H
    from yard_rl.v6.world.integrated.yard_layout import terminal_layout

    seed, layout = 9921000, terminal_layout()
    month = plan_days(seed, loads=[3500, 5000, 7500])
    built = build_month(seed, days=month)
    tn = truck_net_by_block(built['schedule'])
    legacy = plan_month_vessels(month, layout, obs=OBS_24H, truck_net=tn)
    with pytest.raises(ContainerContractError) as err:
        build_fixed_seed(built, legacy, seed=seed)
    assert err.value.report == {'Y08': dict(sources=971, exits=976, missing=5)}

    guarded = plan_month_vessels(month, layout, obs=OBS_24H, truck_net=tn,
                                 initial_stock=initial_stock_by_block(built))
    changed = [(o['key'], o['block'], o['moves'], o['work'], n['work'])
               for d in legacy for o, n in zip(legacy[d], guarded[d]) if o != n]
    assert len(changed) == 1
    key, block, moves, old, new = changed[0]
    assert (block, moves, old, new) == ('Y08', 537, 'LOAD', 'DISCHARGE') and key.startswith('D00-')
    document, audit = build_fixed_seed(built, guarded, seed=seed)
    assert audit['passed'] and audit['truck_orders'] == len(built['schedule'])
    assert sum(s['block'] == 'Y08' for s in document['sources']) == 648 + 323 + 537
