"""사본 엔진이 원본 엔진과 **같은 하루**를 굴리는가.

바이트 동일 검사(`test_world_clone.py`)는 "파일이 같다"만 본다. 여기서는 실제로
하루를 굴려 **결과 수치가 같은지** 본다 — 사본이 다른 모듈을 잘못 끌어오면 파일은
같아도 동작이 갈릴 수 있다.

같아야 하는 이유: 세대 간 비교(v2 vs v3)가 **짝비교**라 두 세대가 물리적으로 같은
무대를 받아야 한다. 어긋나면 그 비교로는 아무것도 못 말한다.

빠르게 돌리려고 하루 물량을 300대로 줄였다 — 등식 검사라 규모가 크지 않아도 된다.
"""
from __future__ import annotations

DAY_TOTAL = 300
SEED = 9_900_777


def _run(pkg):
    """`pkg` 가 가리키는 엔진 트리로 하루를 굴리고 결과를 돌려준다."""
    import importlib

    ts = importlib.import_module(f"{pkg}.terminal_stream")
    mb = importlib.import_module(f"{pkg}.multiblock")
    bl = importlib.import_module(f"{pkg}.baselines")
    cd = importlib.import_module(f"{pkg}.candidates")
    pc = importlib.import_module(f"{pkg}.policy_config")
    pr = importlib.import_module(f"{pkg}.profiles")
    yl = importlib.import_module(f"{pkg}.yard_layout")
    eng = importlib.import_module(f"{pkg}.engine")
    lvl = importlib.import_module(
        f"{pkg.rsplit('.', 1)[0]}.domain.enums").InformationLevel.PRE_ADVICE

    prof, layout = pr.build_h21_profile(), yl.terminal_layout()
    built = ts.build_diurnal(prof, SEED, obs=ts.OBS_24H, layout=layout,
                             params=ts.TerminalStreamParams(load_4h=DAY_TOTAL),
                             day_total=DAY_TOTAL, background_seed=SEED)

    def sim_from(scn):
        s = eng.TerminalSimulator(prof, scn, check_invariants=True)
        s.info_level = lvl
        return s

    mbt = mb.MultiBlockTerminal(
        {b: ts.ensure_time_ledger(sim_from(s)) for b, s in built["scenarios"].items()},
        extra_review_epochs=ts.admission_epochs(ts.OBS_24H))
    ann = ts.ScheduledAnnouncer(built["schedule"], lead_s=1800.0,
                                end_s=built["sim_end_s"])
    gens: dict[int, object] = {}
    pol = bl.ResolverPolicy(bl.ServiceFirstSPTPreference(), "SF")

    def exec_policy(sim, dp):
        g = gens.setdefault(id(sim), cd.CandidateGenerator(config=pc.LEGACY_DEFAULT))
        gb = {c: g.generate(sim, c, lvl) for c in dp.crane_ids}
        bl._apply(sim, pol.decide(sim, dp, gb))

    out = mbt.run(exec_policy, review_fn=ann.review)
    turns = sorted(mbt.ledger.a_to_o_samples_s(ts.OBS_24H.observe_s))
    return {"terminal_total": out["terminal_total"],
            "route_cost_s": out["route_cost_s"],
            "end": out["end"], "admitted": ann.n_admitted,
            "n_turns": len(turns), "turn_sum": round(sum(turns), 6),
            "rehandles": sum(s.kpis.rehandle_count for s in mbt.blocks.values()),
            "empty_gantry_m": round(
                sum(s.kpis.empty_gantry_m for s in mbt.blocks.values()), 6)}


import contextlib


@contextlib.contextmanager
def _same_arrival_curve():
    """★두 트리의 **도착 곡선 상수만** 잠깐 맞춘다.

    v3 는 2026-08-26 에 **6차 계약**(야간 38% · 봉우리 셋)으로 갈아탔고 원본은
    5차(야간 15% · 봉우리 둘) 그대로다 — v1·v2 의 과거 수치를 지키려고 일부러
    안 건드렸다(`test_world_clone.OVERRIDDEN` 에 선언).

    그런데 **이 시험이 보려는 것은 엔진이 같은가**이지 무대 상수가 같은가가 아니다.
    상수를 맞춰 놓고 굴려야, 남는 차이가 **진짜 엔진 차이**다.

    ⚠️ 모듈 전역을 바꿔도 소용없다 — `diurnal_arrivals` 의 기본값은 **def 시점에
    묶인다**. 그래서 `__kwdefaults__` 를 직접 갈아 끼우고 끝나면 되돌린다.
    """
    import yard_rl.integrated.terminal_stream as o
    import yard_rl.v4.world.integrated.terminal_stream as n
    saved = {}
    for fn in ("diurnal_rate", "diurnal_arrivals"):
        f = getattr(o, fn)
        saved[fn] = dict(f.__kwdefaults__)
        f.__kwdefaults__.update(night_frac=n.DIURNAL_NIGHT_FRAC,
                                peaks=n.DIURNAL_PEAKS)
    try:
        yield
    finally:
        for fn, kd in saved.items():
            getattr(o, fn).__kwdefaults__.update(kd)


def test_clone_engine_reproduces_original():
    """원본 트리와 v3 사본 트리가 **같은 수치**를 낸다 (도착 곡선을 맞춘 뒤)."""
    with _same_arrival_curve():
        a = _run("yard_rl.integrated")
        b = _run("yard_rl.v4.world.integrated")
    assert a == b, (
        f"사본 엔진이 원본과 다른 하루를 굴렸다\n  원본 {a}\n  사본 {b}\n"
        f"→ 사본이 원본 밖 모듈을 끌어오고 있거나, 사본에 빠진 모듈이 있다.")
    assert a["admitted"] == DAY_TOTAL, f"투입 미완 {a['admitted']}"
