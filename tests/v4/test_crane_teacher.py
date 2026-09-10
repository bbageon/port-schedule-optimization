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
    a, b = c.result().samples
    assert abs(a.target + b.target) < 1e-9
    # 원화로 되돌리면 절반 격차
    assert abs(from_advantage(a.target) + 100_000.0) < 1e-6


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
              phi_factual=1_000_000 - a[2] * 200_000,
              phi_alt=1_000_000 - b[2] * 200_000, crane="YC-L", job="J")
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
