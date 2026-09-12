"""크레인 교사·학습기가 계약을 지키는가 ([[YR-305]]).

■ 무엇을 지켜야 하나
  ① 한 결정 → 표본 **둘**(고른 것·대안), 기준선은 두 세계 평균
  ② 검증분을 **결정 단위**로 가른다 — 짝이 갈리면 검증이 낙관적으로 나온다
  ③ ★**배울 수 있는 문제는 실제로 배운다** — 이게 없으면 학습기가 도는지 알 수 없다
  ④ ★목표 중앙격차를 **기록한다** — [[YR-299]] B 가 이 계측이 없어 눈금 문제를
     검증조차 못 했다
"""
from __future__ import annotations

import random

import torch

from yard_rl.v4.crane.fit import CraneTrainer, scale_health
from yard_rl.v4.crane.policy import CRANE_ADV_SCALE, CraneNet, from_advantage
from yard_rl.v4.crane.teacher import CraneLabelCollector, pick_alternative


def test_one_decision_makes_two_samples():
    """고른 것과 대안 — 둘 다 정답이 붙는다."""
    c = CraneLabelCollector()
    c.add(picked_row=[0.0] * 8, alt_row=[1.0] * 8,
          phi_factual=900_000.0, phi_alt=1_100_000.0, crane="YC-L", job="J1")
    ls = c.result()
    assert len(ls) == 2
    assert {s.action for s in ls.samples} == {"PICKED", "ALT"}


def test_baseline_is_the_mean():
    """기준선이 두 세계 평균 — 양쪽 정답의 합이 0 이어야 한다."""
    c = CraneLabelCollector()
    c.add(picked_row=[0.0] * 8, alt_row=[1.0] * 8,
          phi_factual=900_000.0, phi_alt=1_100_000.0, crane="YC-L", job="J1")
    ls = c.result()
    a, b = ls.samples
    assert abs(a.target + b.target) < 1e-9
    # 원화로 되돌리면 절반 격차. ★되돌릴 때는 **그날 눈금**을 써야 한다
    # ([[YR-309]]) — 기본 상수로 되돌리면 조용히 틀린 값이 나온다.
    assert abs(from_advantage(a.target, ls.scale) + 100_000.0) < 1e-6


def test_zero_gap_counted():
    """차이가 0 인 결정을 센다 — 배울 게 없는 결정이 얼마나 되는지 알아야 한다."""
    c = CraneLabelCollector()
    for _ in range(3):
        c.add(picked_row=[0.0] * 8, alt_row=[1.0] * 8,
              phi_factual=1_000_000.0, phi_alt=1_000_000.0, crane="YC-L", job="J")
    assert c.result().zero == 3


def test_alternative_is_runner_up():
    """대안은 **차점자** — 무작위로 고르면 뻔한 후보와 비교해 배울 게 없다."""
    scored = [(0.5, 0), (0.1, 1), (0.9, 2), (0.3, 3)]
    assert pick_alternative(scored) == 3        # 1등은 자리1(0.1), 2등은 자리3(0.3)
    assert pick_alternative([(0.1, 0)]) is None  # 후보 하나면 비교 대상이 없다


#: 가짜 문제의 격차 폭 — **실측 눈금에 맞춘다** ([[YR-308]]: 중앙 9,470원).
#: 여기가 실제 라벨과 자릿수가 다르면 `test_records_gap_for_scale_check` 가
#: 눈금이 아니라 **가짜 문제의 크기**를 검사하게 된다.
_GAP_SPAN = 20_000


def _learnable(n: int = 300):
    """배울 수 있는 가짜 문제 — *"오래 기다린 트럭을 먼저 하면 싸다"*."""
    rng = random.Random(7)
    c = CraneLabelCollector()
    for _ in range(n):
        a = [0, 1, rng.uniform(0, 2), 0, rng.uniform(0, 1), rng.uniform(0, 1), 0,
             rng.uniform(0, 1)]
        b = [0, 1, rng.uniform(0, 2), 0, rng.uniform(0, 1), rng.uniform(0, 1), 0,
             rng.uniform(0, 1)]
        c.add(picked_row=a, alt_row=b,
              phi_factual=1_000_000 - a[2] * _GAP_SPAN,
              phi_alt=1_000_000 - b[2] * _GAP_SPAN, crane="YC-L", job="J")
    return c.result()


def test_actually_learns():
    """★배울 수 있는 문제는 실제로 배운다.

    이 시험이 없으면 학습기가 도는지, 그냥 손실 숫자만 찍는지 구분 못 한다.
    """
    torch.manual_seed(1)
    net = CraneNet()
    rep = CraneTrainer(net).fit(_learnable(), seed=1)
    assert rep.n > 0 and rep.n_val > 0
    net.eval()
    with torch.no_grad():
        short = float(net(torch.tensor([[0, 1, 0.2, 0, .5, .5, 0, .5]],
                                       dtype=torch.float32))[0])
        long_ = float(net(torch.tensor([[0, 1, 1.8, 0, .5, .5, 0, .5]],
                                       dtype=torch.float32))[0])
    assert long_ < short, (
        f"오래 기다린 쪽에 더 낮은(=좋은) 점수를 줘야 하는데 "
        f"짧음 {short:.3f} · 김 {long_:.3f} — 학습이 안 됐다")


def test_val_split_is_by_decision():
    """★검증분을 결정 단위로 가른다 — 짝이 갈리면 검증이 낙관적이 된다."""
    torch.manual_seed(2)
    rep = CraneTrainer(CraneNet()).fit(_learnable(200), seed=3)
    assert rep.n % 2 == 0 and rep.n_val % 2 == 0, (
        f"표본 수가 홀수다 — 짝이 갈렸다 (학습 {rep.n} · 검증 {rep.n_val})")


def test_records_gap_for_scale_check():
    """★목표 중앙격차를 기록한다 — [[YR-299]] B 가 이게 없어 못 가렸다."""
    rep = CraneTrainer(CraneNet()).fit(_learnable(), seed=1)
    assert rep.median_gap_krw > 0
    assert "적정" in scale_health(rep.median_gap_krw), (
        f"눈금이 안 맞는다: {rep.median_gap_krw:,.0f}원 vs "
        f"CRANE_ADV_SCALE {CRANE_ADV_SCALE:,.0f}원")
    assert 0.0 <= rep.zero_ratio <= 1.0


def test_scale_health_flags_bad_scales():
    """눈금이 어긋나면 말해 준다."""
    assert "적정" in scale_health(CRANE_ADV_SCALE)
    assert "눈금이 크다" in scale_health(CRANE_ADV_SCALE * 0.01)
    assert "눈금이 작다" in scale_health(CRANE_ADV_SCALE * 100)


# ─────────────────────────────────────────────── 날별 눈금 ([[YR-309]])

def test_scale_is_the_days_median_gap():
    """★눈금은 **그날 라벨의 중앙 격차**다 — 전 부하 공통 상수가 아니다.

    상수를 쓰면 부하 12,500 인 날의 목표가 3.99, 3,500 인 날이 0.30 으로 13배
    벌어져 학습률이 고정인데 미는 세기가 13배 달라진다 ([[YR-308]] 실측).
    """
    c = CraneLabelCollector()
    for gap in (10_000.0, 20_000.0, 30_000.0):        # 중앙 20,000
        c.add(picked_row=[0.0] * 8, alt_row=[1.0] * 8,
              phi_factual=1_000_000.0, phi_alt=1_000_000.0 + gap,
              crane="YC-L", job="J")
    ls = c.result()
    assert ls.median_gap_krw == 20_000.0
    assert ls.scale == 20_000.0
    assert ls.floor_bound is False


def test_targets_center_on_one():
    """★그날 목표 중앙이 1 근처가 된다 — 이게 이 변경의 목적이다."""
    import statistics as stx
    c = CraneLabelCollector()
    for gap in (4_000.0, 20_000.0, 60_000.0, 90_000.0):
        c.add(picked_row=[0.0] * 8, alt_row=[1.0] * 8,
              phi_factual=1_000_000.0, phi_alt=1_000_000.0 + gap,
              crane="YC-L", job="J")
    ls = c.result()
    ss = ls.samples
    mid = stx.median([abs(ss[i].target - ss[i + 1].target)
                      for i in range(0, len(ss), 2)])
    assert abs(mid - 1.0) < 1e-9, f"목표 중앙이 1 이 아니다: {mid}"
    assert "적정" in scale_health(ls.median_gap_krw, ls.scale)


def test_scale_floor_stops_chasing_rounding():
    """★하한 — 격차 중앙이 0 에 가까우면 나눗셈이 폭발한다.

    차이가 정확히 0 인 결정이 약 21% 다. 그리고 하루 Φ 가 수천만~수십억인데 격차
    중앙이 1,000원 미만이면 그날 선택은 사실상 **비용 중립**이다 — 목표 1 로
    부풀리면 망이 **반올림 잡음을 쫓는다.**
    """
    from yard_rl.v4.crane.policy import CRANE_SCALE_FLOOR

    c = CraneLabelCollector()
    for _ in range(3):
        c.add(picked_row=[0.0] * 8, alt_row=[1.0] * 8,
              phi_factual=1_000_000.0, phi_alt=1_000_000.0,   # 격차 0
              crane="YC-L", job="J")
    ls = c.result()
    assert ls.median_gap_krw == 0.0
    assert ls.scale == CRANE_SCALE_FLOOR, "하한이 안 걸렸다 — 0 나눗셈이 난다"
    assert ls.floor_bound is True
    assert all(s.target == 0.0 for s in ls.samples)


def test_ordering_survives_rescaling():
    """★눈금이 바뀌어도 **정책의 행동은 그대로**다.

    `rank()` 는 점수를 크기순 비교에만 쓴다. 양수로 나누면 순서가 안 바뀌므로
    날마다 눈금이 달라도 배정이 안 바뀐다 — 바뀌는 것은 각 날이 기울기에
    기여하는 **세기**뿐이고, 그것을 고르게 만드는 것이 이 변경의 목적이다.
    """
    def targets(span):
        c = CraneLabelCollector()
        for g in (1.0, 3.0, 7.0):
            c.add(picked_row=[0.0] * 8, alt_row=[1.0] * 8,
                  phi_factual=1_000_000.0, phi_alt=1_000_000.0 + g * span,
                  crane="YC-L", job="J")
        return [s.target for s in c.result().samples]

    small, big = targets(1_000.0), targets(100_000.0)
    assert [x < y for x, y in zip(small, small[1:])] == \
           [x < y for x, y in zip(big, big[1:])], "눈금이 순서를 바꿨다"


def test_fit_intensity_is_a_knob():
    """★회차당 학습 강도를 조절할 수 있다 ([[YR-310]]).

    기본 4.0 은 학습 표본 102개에 408스텝이라 **한 날에 408번 맞추는** 셈이다.
    라벨은 회차마다 버리므로(버퍼 없음) 망이 매 회차 그날 데이터로 다시 그려진다.
    """
    from yard_rl.v4.crane.fit import FIT_STEPS_PER_SAMPLE

    assert FIT_STEPS_PER_SAMPLE == 4.0, "기본값은 실험 전까지 안 움직인다"
    ls = _learnable(200)
    a = CraneTrainer(CraneNet(), steps_coef=4.0).fit(ls, seed=1)
    b = CraneTrainer(CraneNet(), steps_coef=1.0).fit(ls, seed=1)
    assert b.steps < a.steps, f"계수를 낮췄는데 스텝이 안 줄었다: {a.steps} vs {b.steps}"
    assert b.steps_coef == 1.0 and a.steps_coef == 4.0, "원자료에 계수가 남아야 한다"


def test_gentler_fitting_still_learns():
    """★덜 세게 가르쳐도 **배우기는 한다** — 너무 줄이면 학습 자체가 죽는다."""
    torch.manual_seed(1)
    net = CraneNet()
    CraneTrainer(net, steps_coef=1.0).fit(_learnable(), seed=1)
    net.eval()
    with torch.no_grad():
        short = float(net(torch.tensor([[0, 1, 0.2, 0, .5, .5, 0, .5]],
                                       dtype=torch.float32))[0])
        long_ = float(net(torch.tensor([[0, 1, 1.8, 0, .5, .5, 0, .5]],
                                       dtype=torch.float32))[0])
    assert long_ < short, (
        f"계수 1.0 에서 학습이 죽었다 — 짧음 {short:.3f} · 김 {long_:.3f}")
