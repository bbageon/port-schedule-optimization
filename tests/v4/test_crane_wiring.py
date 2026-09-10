"""크레인 교사가 **실제로 배선되었는가** ([[YR-308]]).

■ 왜 이 파일이 따로 있나
  `test_crane_policy.py` 는 정책이 계약을 지키는지 보고, `test_crane_teacher.py` 는
  교사·학습기가 계약을 지키는지 본다. 둘 다 통과하는데도 **라벨이 한 건도 안 나왔다**
  (2026-09-10). 부품은 맞는데 **연결이 틀렸기** 때문이다:

      ① 분기 세계가 `_Ctx.dispatcher` 기본값 `SF_SPT` 로 굴렀다
         — `run_episode` 가 배차를 `_Ctx` 에 **안 넘기고 있었다.** 그래서 사실 가지가
           본 세계와 다른 정책으로 결정해 전 라벨이 "사실 불일치" 로 버려졌다.
      ② 크레인 망이 시드에 안 묶여 본 세계와 분기 세계가 각자 무작위 망을 만들었다
      ③ 예산이 **라벨이 될 수 없는 결정**에 다 나갔다 (WAIT·후보 하나뿐 = 70%)

  세 개 다 "부품 시험" 으로는 안 잡힌다 — 이 파일이 그 자리를 메운다.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from yard_rl.v4.crane.policy import CraneNet, CranePolicy
from yard_rl.v4.dispatch import RL_CRANE
from yard_rl.v4.stage.episode import run_episode
from yard_rl.v4.stage.rollout import RolloutBudget

LOAD, SEED = 300, 9_100_321
HORIZON = 1800.0            # 30분 — 시험용. 실제 학습은 3시간.


def _run(**kw):
    return run_episode(load=LOAD, arm="NO_REALLOC", dispatcher=RL_CRANE,
                       seed=SEED, horizon_s=HORIZON, workers=1, **kw)


@pytest.fixture
def ctxs(monkeypatch):
    """만들어진 `_Ctx` 를 모은다 — 분기 세계가 무엇을 물려받는지 보려고."""
    import yard_rl.v4.stage.episode as E
    made = []
    orig = E._Ctx.__init__

    def spy(self, **kw):
        orig(self, **kw)
        made.append(self)

    monkeypatch.setattr(E._Ctx, "__init__", spy)
    return made


def test_branch_inherits_the_same_crane_floor(ctxs):
    """★분기 세계가 **같은 크레인 바닥**을 쓴다.

    이게 깨지면 사실 가지가 딴 정책으로 결정해 라벨이 통째로 거짓이 된다.
    실제로 깨져 있었고(2026-09-10) 라벨이 0건이었다.
    """
    _run(crane_budget=RolloutBudget(max_labels=1))
    assert ctxs, "_Ctx 가 안 만들어졌다"
    ctx = ctxs[0]
    assert ctx.dispatcher == RL_CRANE, (
        f"분기 세계 배차가 {ctx.dispatcher!r} — 본 세계는 {RL_CRANE!r} 이다. "
        f"사실 가지가 다른 정책으로 결정하게 된다")
    pref = ctx.make_exec_policy().pref
    assert isinstance(pref, CranePolicy)
    assert pref.net is ctx.crane_net, (
        "분기 세계가 **다른 망**을 만들었다 — 라벨이 학습 중인 정책의 것이 아니다")


def test_crane_net_is_seeded():
    """★망 초기값이 시드에 묶인다 ([[YR-304]] 와 같은 함정).

    안 묶이면 같은 시드로 돌려도 Φ 가 달라 짝비교의 전제가 깨진다.
    """
    a = run_episode(load=LOAD, arm="NO_REALLOC", dispatcher=RL_CRANE, seed=SEED)
    b = run_episode(load=LOAD, arm="NO_REALLOC", dispatcher=RL_CRANE, seed=SEED)
    assert a.phi_krw == b.phi_krw, (
        f"같은 시드인데 Φ 가 다르다: {a.phi_krw:,.0f} vs {b.phi_krw:,.0f}")


def test_labels_actually_come_out():
    """★하루를 굴리면 라벨이 **실제로 나온다** — 이 시험이 없어서 0건을 못 봤다."""
    r = _run(crane_budget=RolloutBudget(max_labels=4))
    cs = r.crane_stats
    assert cs["labeled"] > 0, f"라벨 0건 — 회계: {cs}"
    assert len(r.crane_labels) == 2 * cs["labeled"], "한 결정은 표본 둘이어야 한다"
    assert cs["factual_mismatch"] == 0, (
        f"사실 가지가 실제와 다른 결정을 냈다 {cs['factual_mismatch']}건 — "
        f"분기 재조립이 상태를 복원하지 못한 것이다")


def test_budget_is_not_spent_on_unusable_decisions():
    """★예산은 **라벨이 될 수 있는 결정**에만 쓴다.

    크레인 결정은 셋 중 둘이 라벨이 안 된다(WAIT 33% · 후보 하나뿐 37%).
    순서를 거꾸로 두면 예산이 전부 못 쓰는 결정에 나가 라벨이 0건이 된다.
    """
    r = _run(crane_budget=RolloutBudget(max_labels=4))
    cs = r.crane_stats
    assert cs["seen"] >= cs["usable"] >= cs["sampled"] >= cs["labeled"]
    assert cs["sampled"] == 4, f"예산 4건을 다 못 썼다: {cs}"
    assert cs["no_alt"] > 0, "대안 없는 결정이 한 건도 없다 — 판정이 안 돌고 있다"


def test_hook_errors_are_counted_not_hidden():
    """★훅이 터져도 결정을 안 건드리고, **센다**.

    훅 오류가 크레인 예외와 섞이면 "정책이 이상하다" 와 "교사가 이상하다" 를 못 가린다.
    """
    r = _run(crane_budget=RolloutBudget(max_labels=2))
    assert r.crane_stats["hook_errors"] == 0, (
        f"교사 훅이 {r.crane_stats['hook_errors']}번 터졌다")


def test_force_once_beats_everything_in_its_class():
    """★강제 손잡이가 실제로 이긴다 — 없으면 대안 세계를 만들 수 없다."""
    torch.manual_seed(3)
    p = CranePolicy(CraneNet())

    class _Ref:
        job_id = "J-TARGET"
        is_vessel = False
        is_external = True

    class _GC:
        job_ref = _Ref()
        kind = None

    p.force_once["YC-L"] = "J-TARGET"
    key = p.rank(None, "YC-L", _GC())
    assert key[0] == -1, f"강제가 최상위가 아니다: {key}"
    p.force_once.clear()


def test_teacher_off_by_default():
    """교사를 안 붙이면 반사실 세계가 **한 번도 안 뜬다** — 판정 경로의 계약."""
    r = _run()
    assert r.crane_stats == {}
    assert r.crane_labels == []
