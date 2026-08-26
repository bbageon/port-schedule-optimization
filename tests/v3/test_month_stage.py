"""30일 무대가 계약을 지키는가 ([[YR-239]]).

여기서 막는 사고 셋:
  ① 사본 표류 — `MonthTerminal.run` 이 사본과 **루프 상한 말고** 달라지는 것
  ② 정리 누락 — 네 장부 중 하나만 치워 불변식이 깨지는 것
  ③ 날 경계 — 학습에 쓰는 날이 28일이 아니게 되는 것
"""
from __future__ import annotations

import difflib
import inspect

import pytest

from yard_rl.v3.stage.month import (COOLDOWN_DAYS, DAY_S, LOAD_WEIGHTS, N_DAYS,
                                    WARMUP_DAYS, DayPlan, build_month,
                                    job_count, ledger_load, month_vessel_idle,
                                    plan_month, plan_month_vessels,
                                    RETIRE_LAG_S, prune_completed,
                                    retire_done_vessels, summarize,
                                    truck_net_by_block, vessel_meta)
from yard_rl.v3.stage.month_engine import MonthTerminal, inject_vessel
from yard_rl.v3.world.integrated.multiblock import MultiBlockTerminal

SEED = 9_400_777


# ─────────────────────────────────────────────────────── ① 사본 표류
def _code_lines(src: str) -> list[str]:
    """주석·빈 줄·docstring 을 걷어낸 **코드 줄**만. 설명 차이는 표류가 아니다."""
    out, in_doc = [], False
    for raw in src.splitlines():
        ln = raw.strip()
        if in_doc:
            if ln.endswith('"""'):
                in_doc = False
            continue
        if ln.startswith('"""'):
            if not (len(ln) > 3 and ln.endswith('"""')):
                in_doc = True
            continue
        ln = ln.split("#", 1)[0].strip() if "#" in ln and not ln.startswith('f"') else ln
        if ln.startswith("#") or not ln:
            continue
        out.append(ln)
    return out


def test_month_run_differs_from_clone_only_by_the_guard():
    """★`MonthTerminal.run` 은 사본을 옮긴 것 — **상한 한 줄**만 다르다.

    사본(`multiblock.py`)은 못 고치므로 여기서 덮었다. 그 대가는 표류다 —
    사본이 고쳐지면 이쪽만 옛날 코드로 남는다. 기계로 대조한다.
    """
    a = _code_lines(inspect.getsource(MultiBlockTerminal.run))
    b = _code_lines(inspect.getsource(MonthTerminal.run))
    changed = [ln for ln in difflib.unified_diff(a, b, lineterm="")
               if ln[:1] in "+-" and ln[:3] not in ("---", "+++")]
    assert len(changed) == 2, (
        f"사본과 {len(changed)} 줄 다르다 — 상한 한 줄(치환 = -/+ 2줄)만 허용한다:\n"
        + "\n".join(changed))
    assert any("2_000_000" in ln for ln in changed)
    assert any("LOOP_GUARD" in ln for ln in changed)


def test_guard_is_bigger_than_the_clone():
    assert MonthTerminal.LOOP_GUARD > 2_000_000 * (N_DAYS - 1)


# ─────────────────────────────────────────────────────── ③ 날 경계
def test_twenty_eight_training_days():
    """첫날·마지막날은 **연결용** — 학습은 28일이다 (사용자 지시)."""
    days = plan_month(SEED)
    assert len(days) == N_DAYS
    tr = [d for d in days if d.is_train]
    assert len(tr) == N_DAYS - WARMUP_DAYS - COOLDOWN_DAYS == 28
    assert tr[0].index == 1 and tr[-1].index == N_DAYS - 2
    assert not days[0].is_train and not days[-1].is_train


def test_days_are_contiguous():
    """날이 **빈틈없이** 이어진다 — 30일을 한 시뮬로 굴리는 전제다."""
    days = plan_month(SEED)
    for a, b in zip(days, days[1:]):
        assert a.t1 == b.t0
    assert days[-1].t1 == N_DAYS * DAY_S


def test_short_month_knows_its_own_length():
    """3일짜리로 줄여도 **첫날·마지막날**이 연결용이다 (시험용 짧은 달)."""
    days = plan_month(SEED, n_days=3)
    assert [d.is_train for d in days] == [False, True, False]


def test_month_is_reproducible():
    assert [d.load for d in plan_month(SEED)] == [d.load for d in plan_month(SEED)]
    assert [d.load for d in plan_month(SEED)] != [d.load for d in plan_month(SEED + 1)]


def test_load_weights_sum_to_one():
    assert sum(w for _, w, _ in LOAD_WEIGHTS) == pytest.approx(1.0)


def test_summary_counts_only_training_days():
    days = plan_month(SEED)
    s = summarize(days)
    assert s["n_train"] == 28
    assert sum(s["by_label"].values()) == 28
    assert s["trucks_train"] < s["trucks_total"]


# ─────────────────────────────────────────────────── 명단 이어 붙이기
def _short_days(n=2, load=300):
    return [DayPlan(index=i, load=load, label="시험", seed=SEED + 1000 * (i + 1),
                    t0=i * DAY_S, n_days=n) for i in range(n)]


def test_schedule_is_shifted_and_unique():
    """★날마다 도착이 **자기 날 안에** 있고, `docKey` 가 30일 내내 유일하다."""
    days = _short_days(2)
    built = build_month(SEED, days=days)
    keys = [e["job_id"] for e in built["schedule"]]
    assert len(keys) == len(set(keys)), "docKey 가 겹친다 — 날짜 접두가 빠졌다"
    for e in built["schedule"]:
        d = days[e["day"]]
        assert d.t0 <= e["arrival_s"] <= d.t1, "도착이 자기 날 밖이다"
    assert [e["arrival_s"] for e in built["schedule"]] == sorted(
        e["arrival_s"] for e in built["schedule"])


def test_vessels_are_shifted_into_their_day():
    days = _short_days(2)
    from yard_rl.v3.world.integrated.yard_layout import terminal_layout
    v = plan_month_vessels(days, terminal_layout())
    assert set(v) == {0, 1}
    keys = [r["key"] for rows in v.values() for r in rows]
    assert len(keys) == len(set(keys)), "스트림 이름이 겹친다"
    for i, rows in v.items():
        for r in rows:
            assert days[i].t0 <= r["start_s"] <= days[i].t1
    m = vessel_meta([r for rows in v.values() for r in rows])
    assert len(m) == len(keys)


# ──────────────────────────────────────── ★야드 수지 (2026-08-26 실측 사고)
@pytest.mark.parametrize("sd", [SEED, SEED + 1, SEED + 2, SEED + 3])
def test_vessel_work_balances_the_yard(sd):
    """★배가 **내리기만** 하면 야드가 하루 +5,000상자씩 부푼다.

    2026-08-26 실측 사고: `plan_streams` 의 `work` 칸이
    `(len(rows) + k) % 2` 였는데 두 값이 함께 1씩 늘어 합이 **항상 짝수**였다 —
    전 스트림이 양하. 하루 무대는 이 칸을 안 읽어(대신 `type_offset`) 안 드러났고,
    30일 무대가 이 칸으로 배를 붙이자 야드가 **19,656 → 24,828** 로 부풀었다.

    그래서 여기서 **양하 상자 = 적하 상자**를 못 박는다.
    """
    from yard_rl.v3.stage.vessels import plan_streams, sample_day_vessels
    from yard_rl.v3.world.integrated.yard_layout import terminal_layout

    rows = plan_streams(sample_day_vessels(sd), terminal_layout(), sd)
    dis = sum(r["moves"] for r in rows if r["work"] == "DISCHARGE")
    load = sum(r["moves"] for r in rows if r["work"] == "LOAD")
    assert dis == load, f"본선 수지가 안 맞는다 — 양하 {dis} vs 적하 {load}"


def test_work_field_agrees_with_type_offset():
    """`work` 와 `type_offset` 은 **같은 사실**을 말해야 한다.

    하루 무대는 `type_offset` 으로, 30일 무대는 `work` 로 같은 배를 세운다.
    둘이 어긋나면 두 무대가 **다른 세계**가 된다 — 세대 짝비교가 무효가 된다.
    """
    from yard_rl.v3.stage.vessels import plan_streams, sample_day_vessels
    from yard_rl.v3.world.integrated.yard_layout import terminal_layout

    for sd in (SEED, SEED + 1, SEED + 2):
        for r in plan_streams(sample_day_vessels(sd), terminal_layout(), sd):
            want = "DISCHARGE" if r["type_offset"] == 0 else "LOAD"
            assert r["work"] == want, f"{r['key' if 'key' in r else 'vessel_id']}: 어긋남"


# ─────────────────────────────────────────────────────── 판정 — 부호검정
def test_sign_test_matches_published_numbers():
    """★[[YR-211b]] 에 실린 p 값과 **같은 수**가 나와야 한다.

    30일 무대 판정이 하루 무대 판정과 다른 잣대를 쓰면 세대 비교가 무효가 된다.
    """
    from yard_rl.v3.eval.month_judge import sign_test

    assert sign_test([-1.0] * 16)["p"] == pytest.approx(0.0000305, abs=1e-6)
    assert sign_test([-1.0] * 13 + [1.0] * 3)["p"] == pytest.approx(0.0213, abs=1e-3)
    assert sign_test([-1.0] * 12 + [1.0] * 4)["p"] == pytest.approx(0.0768, abs=1e-3)
    assert sign_test([-1.0] * 8 + [1.0] * 8)["p"] == pytest.approx(1.0)


def test_sign_test_drops_ties_and_says_so():
    """동점은 버리되 **몇 개 버렸는지 남긴다** — 조용히 사라지면 표본이 부풀어 보인다."""
    from yard_rl.v3.eval.month_judge import sign_test

    r = sign_test([-1.0] * 6 + [0.0] * 2)
    assert r["n"] == 6 and r["ties"] == 2
    assert r["p"] == pytest.approx(2 / 64)


def test_empty_sign_test_is_not_significant():
    from yard_rl.v3.eval.month_judge import sign_test

    assert sign_test([])["p"] == 1.0
    assert sign_test([0.0, 0.0])["n"] == 0


# ───────────────────────────── ★배를 언제 치워도 되는가 (2026-08-26 실측 사고)
class _Job:
    def __init__(self, vid): self.vessel_id = vid


class _Truth:
    def __init__(self, at): self.actual_completion_s = at


class _Vessel:
    def __init__(self, done, at, wait=0.0):
        self.done, self.truth, self.sts_wait_accum_s = done, _Truth(at), wait


class _Sim:
    def __init__(self, jobs, vessels):
        self.jobs, self.vessels = jobs, vessels
        self.refreshed = 0

    def _refresh_rates(self): self.refreshed += 1


class _Mbt:
    def __init__(self, sim): self.blocks = {"B1": sim}


def test_a_finished_vessel_with_jobs_left_is_not_retired():
    """★남은 job 이 있으면 **못 치운다.**

    양하 job 은 시각이 아니라 **박스의 물리 도착**이 푼다. STS 가 이송보다 빠르므로
    `done` 시점에 안 풀린 job 이 수백 건 남을 수 있고, 그때 배를 치우면
    `_transfer_arrive` 가 배를 못 찾아 **그 job 들이 영원히 PLANNED 로 남는다.**

    실측(부하 300·4일): 남은 job 131 → 1,566 · Φ 1,568만 → 5,812만원 · 72초 → 325초.
    """
    sim = _Sim({"j1": _Job("V-A")}, {"V-A": _Vessel(True, 0.0, 12.0)})
    arch: dict = {}
    assert retire_done_vessels(_Mbt(sim), arch, t=DAY_S) == 0
    assert "V-A" in sim.vessels and not arch


def test_a_finished_vessel_with_no_jobs_is_retired_and_archived():
    """일이 다 끝났으면 치운다 — **유휴 적립값은 옮겨** 손실이 없다."""
    sim = _Sim({}, {"V-A": _Vessel(True, 0.0, 12.0)})
    arch: dict = {}
    assert retire_done_vessels(_Mbt(sim), arch, t=DAY_S) == 1
    assert not sim.vessels and arch["V-A"] == pytest.approx(12.0)
    assert sim.refreshed == 1


def test_a_just_finished_vessel_waits_for_its_last_transfer():
    """끝난 직후에는 안 치운다 — 마지막 이송이 공중에 떠 있으면 KeyError 가 난다."""
    sim = _Sim({}, {"V-A": _Vessel(True, 1000.0)})
    assert retire_done_vessels(_Mbt(sim), {}, t=1000.0 + 60.0) == 0
    assert retire_done_vessels(_Mbt(sim), {}, t=1000.0 + RETIRE_LAG_S) == 1


def test_an_unfinished_vessel_is_never_retired():
    sim = _Sim({}, {"V-A": _Vessel(False, None)})
    assert retire_done_vessels(_Mbt(sim), {}, t=DAY_S * 10) == 0


# ───────────────────────────────── 작업자에게 보낼 짐 줄이기 (분기 전용 명단)
def test_announcer_window_keeps_only_that_stretch():
    """★30일 명단(20만 건)을 통째로 절이면 작업자 하나당 수십 MB 다.

    반사실 분기는 `[t, t+H]` 밖의 도착을 볼 일이 없으므로 그날 것만 잘라 보낸다.
    자른 사본은 **계수기를 새로** 갖는다 — 분기가 원본 계수기를 올리면 투입 검사가
    거짓으로 부푼다(`clone_fresh` 와 같은 이유).
    """
    import pickle

    from yard_rl.v3.stage.orders import V3Announcer

    days = _short_days(3, load=200)
    built = build_month(SEED, days=days)
    ann = V3Announcer(built["schedule"], end_s=3 * DAY_S)
    w = ann.window(DAY_S, 2 * DAY_S)

    assert 0 < len(w.by_epoch) < len(ann.by_epoch)
    assert all(DAY_S <= k <= 2 * DAY_S for k in w.by_epoch)
    assert w.end_s == ann.end_s and w.period_s == ann.period_s
    assert w.n_admitted == 0 and w.skips == []
    assert len(pickle.dumps(w)) < len(pickle.dumps(ann))


# ─────────────────────── ★블록이 차서 양하가 멈추지 않게 (2026-08-26 실측 사고)
def test_each_block_stays_near_its_starting_level():
    """★30일 동안 **블록별 수지**가 묶여 있어야 한다.

    블록 용량은 1,440슬롯이고 30일 무대는 648상자(45%)로 시작한다. 양하 스트림
    하나가 하루 386상자를 쌓으므로, 한 블록이 사흘 내리 양하를 맡으면 **슬롯이 없어
    양하가 멈춘다.**

    실측(고치기 전 · 부하 300): 1일차 끝에 양하 블록 다섯이 92%(1,323~1,337상자)로
    차고 막힌 양하 job 93 → 393 → 570 건. **빈 블록(528상자)은 막힌 게 0** 이었다.

    그래서 `plan_month_vessels` 가 *"덜 찬 블록은 내리고 찬 블록은 싣는다"* 로
    다시 정한다. 여기서 그 수지가 **용량 안에** 머무는지 못 박는다.
    """
    from yard_rl.v3.stage.month import MONTH_FILL_RATIO
    from yard_rl.v3.world.integrated.profiles import build_h21_profile
    from yard_rl.v3.world.integrated.yard_layout import terminal_layout

    geom = build_h21_profile().block
    cap = geom.bay_count * geom.row_count * geom.tier_max
    start = int(cap * MONTH_FILL_RATIO)

    lay = terminal_layout()
    days = plan_month(SEED)
    built = build_month(SEED, days=days)
    tn = truck_net_by_block(built["schedule"])
    v = plan_month_vessels(days, lay, truck_net=tn)

    occ = {b: start for b in lay.ids}       # 블록별 상자 수 (계획 기준)
    hi, totals = 0, []
    for i in sorted(v):
        for r in v[i]:
            occ[r["block"]] += (r["moves"] if r["work"] == "DISCHARGE"
                                else -r["moves"])
        for b, t in tn.get(i, {}).items():  # 트럭이 빼 간 몫 (반출 − 반입)
            occ[b] = occ.get(b, start) - t
        hi = max(hi, max(occ.values()))
        totals.append(sum(occ.values()))

    # ① **블록이 안 찬다** — 92% 에서 양하가 멈추는 것을 실측했다
    assert hi < cap * 0.90, f"블록이 {hi}/{cap}({hi/cap:.0%}) 까지 찬다 — 양하가 멈춘다"
    # ② ★**야드가 안 마른다** — 트럭은 반출 60% 라 하루 1,000상자를 빼 간다.
    #    본선 양하가 그만큼 채우지 않으면 30일이면 텅 빈다(되돌릴 조건 1).
    assert min(totals) > 0.5 * totals[0], (
        f"야드가 마른다 — {totals[0]:,} → 최저 {min(totals):,}")
    assert totals[-1] > 0.5 * totals[0], (
        f"야드가 단조 감소한다 — {totals[0]:,} → {totals[-1]:,}")


def test_month_vessel_work_is_deterministic_and_policy_free():
    """수지 배정은 **계획에서만** 나온다 — 정책·런타임을 안 본다."""
    from yard_rl.v3.world.integrated.yard_layout import terminal_layout

    lay, days = terminal_layout(), plan_month(SEED, n_days=6)
    a = plan_month_vessels(days, lay)
    b = plan_month_vessels(days, lay)
    assert [(r["key"], r["work"]) for i in sorted(a) for r in a[i]] ==            [(r["key"], r["work"]) for i in sorted(b) for r in b[i]]
    for i in sorted(a):                       # 두 칸이 같은 말을 하는가
        for r in a[i]:
            want = "DISCHARGE" if r["type_offset"] == 0 else "LOAD"
            assert r["work"] == want


def test_judge_flags_a_silent_rl_arm():
    """★RL 이 거래를 안 했으면 **무승부라고 적으면 안 된다.**

    실측(2026-08-26): 학습 전 무작위 망은 시드에 따라 **5,179 결정에 거래 3건**만
    내기도 한다(같은 코드가 다른 시드에서는 1,255건). 그러면 표에
    `p=1.0000 무승부` 가 찍히는데, 그건 *비겼다* 가 아니라 *아무것도 안 했다* 는 뜻이다.
    조용히 넘기면 **안 한 것과 못 이긴 것이 안 갈린다.**
    """
    from yard_rl.v3.eval import month_judge as MJ

    days = plan_month(SEED, n_days=4)
    calls = {}

    def fake(kw):
        arm = kw["arm"]
        calls[arm] = True
        return MJ.ArmMonth(arm=arm, traded=(3 if arm == "RL" else 300),
                           phi_by_day={d.index: 1.0e8 for d in days})

    said = []
    old = MJ._run_arm
    MJ._run_arm = fake
    try:
        out = MJ.judge_month(seed=SEED, days=days, arms=("NO_REALLOC", "FCFS"),
                             workers=1, log=said.append)
    finally:
        MJ._run_arm = old

    assert out["rl_silent"] is True and out["rl_traded"] == 3
    assert any("거래를 거의 안 했다" in x for x in said), "경고를 안 냈다"
    assert calls == {"RL": True, "NO_REALLOC": True, "FCFS": True}


def test_judge_does_not_flag_a_busy_rl_arm():
    """거래를 제대로 하면 경고가 **안** 떠야 한다 — 늑대 소년이 되면 못 쓴다."""
    from yard_rl.v3.eval import month_judge as MJ

    days = plan_month(SEED, n_days=4)

    def fake(kw):
        arm = kw["arm"]
        return MJ.ArmMonth(arm=arm, traded=(280 if arm == "RL" else 300),
                           phi_by_day={d.index: 1.0e8 for d in days})

    said = []
    old = MJ._run_arm
    MJ._run_arm = fake
    try:
        out = MJ.judge_month(seed=SEED, days=days, arms=("NO_REALLOC", "FCFS"),
                             workers=1, log=said.append)
    finally:
        MJ._run_arm = old
    assert out["rl_silent"] is False
    assert not any("거의 안 했다" in x for x in said)
