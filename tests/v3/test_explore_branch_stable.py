"""탐색을 켜도 분기 세계가 실제 궤적을 재현하는가.

반사실은 분기 시점에서 세계를 복제해 다시 굴린다. 탐색 난수가 **순서에 의존**하면
복제한 세계의 난수 상태를 실제와 맞출 수 없어 factual 가지가 다른 결정을 낸다 —
그러면 라벨 전체가 다른 정책의 것이 된다. 탐색을 끄면 안 터지지만 그건 문제가
없어서가 아니라 **난수를 안 뽑아서**다.

그래서 탐색을 좌표로 뽑는다(`v3/actors/explore.py`). 여기서 그 성질을 검사한다.
"""
from __future__ import annotations

import statistics as st

import pytest

from yard_rl.v3.actors.explore import draw, pick


def test_same_coordinate_same_value():
    """같은 (시드·오더·시각·태그)면 언제 물어도 같은 값 — 분기 안정성의 근거."""
    a = draw(11, "D-00042", 3600.0, "sell:on")
    b = draw(11, "D-00042", 3600.0, "sell:on")
    assert a == b


@pytest.mark.parametrize("changed", [
    {"seed": 12}, {"doc_key": "D-00043"}, {"t": 3660.0}, {"tag": "buy:on"},
])
def test_any_coordinate_change_changes_value(changed):
    """좌표가 하나라도 다르면 값이 달라야 한다 — 안 그러면 결정끼리 얽힌다."""
    base = {"seed": 11, "doc_key": "D-00042", "t": 3600.0, "tag": "sell:on"}
    assert draw(**base) != draw(**{**base, **changed})


def test_uniform_enough_to_drive_exploration():
    """[0,1) 균등 — 탐색 비율이 설정값과 어긋나면 안 된다."""
    xs = [draw(11, f"D-{i:05d}", 0.0, "sell:on") for i in range(5_000)]
    assert 0.0 <= min(xs) and max(xs) < 1.0
    assert abs(st.mean(xs) - 0.5) < 0.02, f"평균 {st.mean(xs)}"
    for lo in (0.0, 0.25, 0.5, 0.75):
        share = sum(1 for x in xs if lo <= x < lo + 0.25) / len(xs)
        assert abs(share - 0.25) < 0.03, f"[{lo},{lo+0.25}) 비율 {share}"


def test_pick_covers_range_and_stays_inside():
    xs = [pick(11, f"D-{i:05d}", 0.0, "sell:which", 7) for i in range(2_000)]
    assert min(xs) == 0 and max(xs) == 6
    assert all(0 <= x < 7 for x in xs)
    assert pick(11, "D-1", 0.0, "t", 0) == 0        # 후보 0 이어도 안 터진다


def test_actor_explore_is_branch_stable():
    """같은 결정을 **새로 만든 행위자**에게 다시 물어도 같은 답이 나온다.

    분기 세계는 행위자를 새로 만든다(망만 공유). 순차 난수였다면 여기서 갈렸다.
    """
    from yard_rl.v3.actors import Buyer, BuyerNet, Seller, SellerNet

    net_s, net_b = SellerNet(), BuyerNet()
    a = Seller(net_s, layout=None, explore=0.5, seed=99)
    b = Seller(net_s, layout=None, explore=0.5, seed=99)
    assert a.seed == b.seed
    x = Buyer(net_b, explore=0.5, seed=99)
    assert x.seed == 99
    for dk in ("D-1", "D-2", "D-3"):
        for t in (0.0, 1800.0):
            assert (draw(a.seed, dk, t, "sell:on")
                    == draw(b.seed, dk, t, "sell:on"))
