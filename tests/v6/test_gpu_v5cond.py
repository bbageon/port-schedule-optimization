"""조건부 규약(gpu/v5cond.py) 이 v5 와 **같은 답**을 내는가 ([[YR-327]] 조각 7 · key=cond).

■ 무엇을 지키나 — v5 를 실제로 굴려 결정마다 같은 것을 묻는다 (기대값 손기입 없음)
  ① **순차 조건부 마스크** — v5 `ppo/crane.py:20-36` `joint_mask` 를 그 시점 상태에서 직접 부르고,
     같은 상태를 옮긴 배열 세계의 `joint_mask_items` 와 items 순서대로 `==` (bool 전건).
     크레인을 번호순으로 돌며 앞 크레인의 조건부 약속이 뒤 크레인 마스크를 바꾸는 것까지.
  ② **선호 정렬키** — v5 `CentralResolver._pair_key` 로 정렬한 쌍 목록 (크레인, candidate_id) 과
     배열 `pair_order` 의 순서가 전건 `==`. 선호 셋: baseline · sf_spt · fifo.
  ③ **전원 WAIT 예외 대체** — v5 `stage/episode.py:209-216` 과 같은 모양의 try/except 로 굴리며
     **예외를 일부러 유발**하고(첫 크레인에게 실린 infeasible 후보를 고르게 한다), 배열
     `decision_would_raise` + `substitute_all_wait` 가 같은 판정·같은 결정열을 내는지 본다.
     판정은 결정마다 `deepcopy(sim)` 로 후보 하나하나 실제로 `_apply` 해 보고 대조한다 (참·거짓 양쪽).
  ④ **엔진 통합** — `make_seq_policy` 를 공동 규약 `policy_fn` 으로 끼워 배열 엔진을 완주시키고,
     같은 규칙으로 굴린 v5 (`CandidateGenerator` + `joint_mask` + `baselines._apply`) 와 결정열 `==`.
  ⑤ **겉옷이 옳은 결정을 안 건드린다** — 같은 무대를 `guard_all_wait` 유무로 두 번 완주시켜 세계·흔적의
     잎 전부가 비트 동일. 순차 조건부 판과, v5 `_rule_policy` 조합(ResolverPolicy + 대체) 판 둘 다.
     그 v5 구동에서 예외는 **한 번도 안 난다** — `CentralResolver` 가 `dry_run_commit` 으로 수용한 것만
     배정하기 때문(D-ORACLE)이고, 그래서 `policy_exceptions == 0` 하드가드가 성립한다.
  ⑥ `feasible_joint` == v5 `baselines._feasible_joint` (무작위 공동 후보 조합) · `prior_open` ==
     v5 `next(reversed(selected.values()))` · WAIT 후보가 항상 feasible 이라 v5 35행 RuntimeError 는
     구조상 도달 불가 (상시 단언) · items 칸 넘침은 `V_RESOLVER_TRUNC` · jit·eager·vmap 일치 ·
     `lru_cache` 규약.

■ 무대 — `test_gpu_cands3.STAGES` 12종을 그대로 쓴다 (PRE_ADVICE·BLOCK_ARRIVAL · K=2 · 교착 · prune ·
  mandatory PLAN_FAILED · 무작위 야드 8종). 발판(`world_from_sim`·`all3_jit`)도 거기서 빌린다.

실행: Windows `.venv-jax` · x64 CPU (`scripts/v6/run_tests_windows.sh`).
"""
from __future__ import annotations

import copy
import importlib.util
import os
import random

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)   # ★float64 — engine_step.py 머리말
jnp = jax.numpy

from yard_rl.v6.gpu import dispatch as DP                                            # noqa: E402
from yard_rl.v6.gpu import v5cond as VC                                              # noqa: E402
from yard_rl.v6.gpu.geom import Geom                                                 # noqa: E402
from yard_rl.v6.gpu.host_convert import to_block_world                               # noqa: E402
from yard_rl.v6.gpu.state import EMPTY_ID, PK_WAIT, V_RESOLVER_TRUNC                 # noqa: E402
# v5 정본
from yard_rl.v6.ppo.crane import joint_mask as v5_joint_mask                         # noqa: E402
from yard_rl.v6.world.contract.schema import CandidateKind                           # noqa: E402
from yard_rl.v6.world.integrated.baselines import (FIFOPreference,                   # noqa: E402
                                                   ServiceFirstSPTPreference, _apply, _feasible_joint,
                                                   _wait_of)
from yard_rl.v6.world.integrated.candidates import CandidateGenerator                # noqa: E402
from yard_rl.v6.world.integrated.engine import TerminalSimulator                     # noqa: E402
from yard_rl.v6.world.integrated.policy_config import LEGACY_DEFAULT                 # noqa: E402
from yard_rl.v6.world.integrated.resolver import BaselinePreference, CentralResolver  # noqa: E402


def _load(name: str, fname: str):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_C3 = _load("_c3_for_cond", "test_gpu_cands3.py")
_EQ = _C3._EQ                                       # test_gpu_engine_equiv (run_array · _array_joint_decisions)
STAGES, world_from_sim, all3_jit, _np = _C3.STAGES, _C3.world_from_sim, _C3.all3_jit, _C3._np
_arr_key, _key, PA = _C3._arr_key, _C3._key, _C3.PA
_caps = _C3._caps
#: v5 선호 — `pref_cols` 의 세 이름과 1:1
PREF_CLS = {"baseline": BaselinePreference, "sf_spt": ServiceFirstSPTPreference, "fifo": FIFOPreference}
#: 시험 전체 집계 (마지막 시험이 보고)
REPORT: dict[str, dict] = {}


# ───────────────────────────────────────────────── jit 발판
_JIT: dict = {}


def _j(key, fn):
    if key not in _JIT:
        _JIT[key] = jax.jit(fn)
    return _JIT[key]


def mask_jit(g):
    return _j(("mask", g), lambda w, fl, pr, sel, k: VC.joint_mask_items(w, fl, pr, sel, k, g))


def seq_jit(g, pick):
    return _j(("seq", g, pick), lambda w, fl, pr, op: VC.sequential_conditional(w, fl, pr, op, pick, g))


def raise_jit(g):
    return _j(("raise", g), lambda w, fl, pr, ch, op: (VC.decision_would_raise(w, fl, pr, ch, op, g),
                                                      VC.first_failing_crane(w, fl, pr, ch, op, g)))


def feas_jit(g):
    return _j(("feas", g), lambda w, fl, t: VC.feasible_joint(w, fl, t, g))


def order_jit(g, pref):
    return _j(("order", g, pref),
              lambda p, w, fl, pr, v: VC.pair_order(pref, p, w, fl, pr, v, g))


# ───────────────────────────────────────────────── v5 구동 발판
class Rec:
    """결정 하나의 v5 기록 + 그 시점 배열 세계."""

    __slots__ = ("pre", "t", "open_ids", "items", "masks", "chosen", "proposed", "pairs", "failed",
                 "sim_copy")

    def __init__(self, **kw):
        for s in self.__slots__:
            setattr(self, s, kw.get(s))


def _armed_hook(sim):
    """`_decision_cranes` 호출 시점(armed 소진 **전**)의 eta_armed 를 가로챈다 — test_gpu_cands3 와 같은 발판."""
    snap = {"armed": set()}
    orig = sim._decision_cranes

    def hooked():
        snap["armed"] = set(sim._eta_armed)
        return orig()

    sim._decision_cranes = hooked
    return snap


def _pre_world(sim, snap, w0, tb, g):
    pre = world_from_sim(sim, w0, tb, g)
    armed = jnp.asarray([cid in snap["armed"] for cid in tb.crane_ids])
    return pre._replace(wake=pre.wake._replace(eta_armed=armed))


def drive_v5(prof, scn, level, decide, w0, tb, g, *, guard=False, want_pairs=None,
             want_copy=False, max_dec=10_000):
    """v5 완주 — 결정마다 배열 세계를 찍고 `decide(sim, dp, gb)` 로 결정한다.

    guard=True 면 `stage/episode.py:209-216` 과 **같은 모양**으로 감싼다:
        try: _apply(sim, decide(...))
        except Exception: exc += 1; _apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})
    want_pairs 가 선호 이름이면 `CentralResolver._pair_key` 로 정렬한 쌍 목록도 기록한다.
    """
    sim = TerminalSimulator(prof, scn, check_invariants=True, info_level=level)
    gen = CandidateGenerator(config=LEGACY_DEFAULT)
    snap = _armed_hook(sim)
    resolver = CentralResolver(PREF_CLS[want_pairs]()) if want_pairs else None
    recs: list[Rec] = []
    exc = {"n": 0}
    while (dp := sim.run_until_decision()) is not None:
        assert len(recs) < max_dec, "결정 폭주"
        pre = _pre_world(sim, snap, w0, tb, g)
        gb = {c: gen.generate(sim, c, level) for c in dp.crane_ids}
        items = {c: [_key(gc) for gc in gb[c].items] for c in dp.crane_ids}
        for c in dp.crane_ids:                                         # WAIT 은 항상 마지막·항상 feasible
            last = gb[c].items[-1]
            assert last.kind == CandidateKind.WAIT and last.feasible is True, (c, items[c])
        pairs = None
        if resolver is not None:
            ps = [(c, gc) for c in dp.crane_ids for gc in gb[c].items if gc.feasible]
            ps.sort(key=lambda cg: resolver._pair_key(sim, cg[0], cg[1]))
            pairs = [(tb.crane_ids.index(c), gc.candidate_id) for (c, gc) in ps]
        sim_copy = copy.deepcopy(sim) if want_copy else None
        box: dict = {}

        def _run():                                                    # ★decide 도 try 안 — v5 213행
            out = decide(sim, dp, gb)
            box["assign"], box["masks"] = (out if isinstance(out, tuple) else (out, None))
            _apply(sim, box["assign"])

        failed = False
        if guard:
            try:
                _run()
            except Exception:
                exc["n"] += 1
                failed = True
                _apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})
        else:
            _run()
        assert "assign" in box, "decide 자체가 터졌다 — 이 시험 무대에서는 안 나야 한다"
        proposed = {c: _key(box["assign"][c]) for c in dp.crane_ids}
        chosen = {c: ("WAIT",) for c in dp.crane_ids} if failed else dict(proposed)
        recs.append(Rec(pre=pre, t=float(dp.time), open_ids=tuple(dp.crane_ids),
                        items=items, masks=box["masks"], chosen=chosen, proposed=proposed,
                        pairs=pairs, failed=failed, sim_copy=sim_copy))
    assert sim.terminal
    return sim, recs, exc["n"]


def arrays_at(rec, tb, g, horizon, pre_advice):
    """그 결정 시점의 배열 후보 — (out, fl, pr) jnp 과 numpy 사본, 그리고 items 열표·키표."""
    out_j, fl_j, pr_j, _ = all3_jit(g, pre_advice)(rec.pre, horizon)
    fl, pr = _np(fl_j), _np(pr_j)
    K, C = fl.raw.shape
    N = rec.pre.n
    icol = np.stack([np.asarray(VC.item_cols(pr_j, k)) for k in range(K)])
    keymap = [{_arr_key(fl, k, c, N, tb): c for c in range(C) if pr.keep[k, c]} for k in range(K)]
    return (out_j, fl_j, pr_j), (fl, pr), icol, keymap


def _item_keys(fl, icol, k, tb, N):
    return [_arr_key(fl, k, int(c), N, tb) for c in icol[k] if int(c) >= 0]


# ═══════════════════════════════════════════════ ① 순차 조건부 마스크
def _py_pick_first(mask):
    return int(np.argmax(np.asarray(mask)))


def _py_pick_last(mask):
    m = np.asarray(mask)
    return int(len(m) - 1 - np.argmax(m[::-1]))


PICKS = {"first": (_py_pick_first, VC.pick_first_masked), "last": (_py_pick_last, VC.pick_last_masked)}


def _seq_decide(gen, level, py_pick):
    """v5 `ppo/crane.py:72-78` 과 같은 순차 조건부 — joint_mask 로 마스크를 받아 py_pick 으로 고른다."""
    def decide(sim, dp, gb):
        selected: dict = {}
        masks: dict = {}
        for cid in sorted(dp.crane_ids):
            cands = gb[cid].items
            m = v5_joint_mask(sim, [(cid, gc) for gc in cands], selected)
            masks[cid] = [bool(b) for b in m.tolist()]
            assert any(masks[cid]), "WAIT 조차 불가 — v5 35행 RuntimeError 자리"
            selected[cid] = cands[py_pick(masks[cid])]
        return selected, masks
    return decide


@pytest.mark.parametrize("pick", sorted(PICKS))
@pytest.mark.parametrize("label", sorted(STAGES))
def test_joint_mask_and_sequential_conditional(label, pick):
    """v5 `joint_mask` 를 결정마다 직접 불러 배열판과 items 순서대로 `==`, 그리고 순차 조건부의 답도 `==`."""
    prof, scn, level, _pref = STAGES[label]()
    caps, _ = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    horizon, pre_advice = float(prof.decision_horizon_s), level == PA
    py_pick, arr_pick = PICKS[pick]
    gen = CandidateGenerator(config=LEGACY_DEFAULT)
    sim, recs, exc = drive_v5(prof, scn, level, _seq_decide(gen, level, py_pick), w0, tb, g)
    assert recs, f"[{label}] 결정이 한 번도 안 열렸다"
    assert exc == 0
    K = len(tb.crane_ids)
    n_mask, n_diff_rows, n_prior = 0, 0, 0
    for i, rec in enumerate(recs):
        tag = f"[{label}/{pick} #{i} t={rec.t:.3f}]"
        (out_j, fl_j, pr_j), (fl, pr), icol, keymap = arrays_at(rec, tb, g, horizon, pre_advice)
        N = rec.pre.n
        open_ = np.asarray([cid in rec.open_ids for cid in tb.crane_ids])
        # items 순서가 v5 와 같은지 먼저 (조각 3 의 ④ 를 이 시험에서도 못박는다)
        for cid in rec.open_ids:
            k = tb.crane_ids.index(cid)
            assert _item_keys(fl, icol, k, tb, N) == rec.items[cid], f"{tag} items 순서 {cid}"
        # ★마스크 — 앞 크레인의 조건부 약속을 배열 sel 로 다시 쌓으며 크레인마다 대조
        sel = np.full((K,), EMPTY_ID, np.int32)
        for cid in sorted(rec.open_ids):
            k = tb.crane_ids.index(cid)
            m_j, ic_j = mask_jit(g)(rec.pre, fl_j, pr_j, jnp.asarray(sel), jnp.int32(k))
            m = np.asarray(m_j)[:len(rec.items[cid])]
            v5m = rec.masks[cid]
            assert m.tolist() == v5m, f"{tag} ① joint_mask {cid}\n  v5 ={v5m}\n  arr={m.tolist()}"
            n_mask += len(v5m)
            n_diff_rows += int(any(v5m[:-1]) and not all(v5m[:-1]))    # 부분 마스크가 실제로 났는가
            sel[k] = keymap[k][rec.chosen[cid]]
        # ★순차 조건부 전체 — 한 번에 굴린 답이 같은가 (열 번호·items 색인)
        so = _np(seq_jit(g, arr_pick)(rec.pre, fl_j, pr_j, jnp.asarray(open_)))
        assert int(so.flags) == 0, f"{tag} items 칸 넘침"
        got = {}
        for k, cid in enumerate(tb.crane_ids):
            if not open_[k]:
                assert int(so.choice[k]) == EMPTY_ID, f"{tag} 안 물은 크레인에 답이 있다 {cid}"
                continue
            got[cid] = _arr_key(fl, k, int(so.choice[k]), N, tb)
            assert int(so.item[k]) == py_pick(rec.masks[cid]), f"{tag} items 색인 {cid}"
        assert got == rec.chosen, f"{tag} ① 순차 조건부 답\n  v5 ={rec.chosen}\n  arr={got}"
        # prior — v5 `next(reversed(selected.values()))`
        ks = [tb.crane_ids.index(c) for c in sorted(rec.open_ids)]
        want = {k: (ks[j - 1] if j > 0 else EMPTY_ID) for j, k in enumerate(ks)}
        for k in ks:
            assert int(so.prior[k]) == want[k], f"{tag} prior {k}"
            n_prior += int(want[k] >= 0)
    REPORT[f"mask/{label}/{pick}"] = dict(decisions=len(recs), mask_cells=n_mask,
                                          partial_mask=n_diff_rows, prior_nonempty=n_prior,
                                          backlog=sim.unfinished_backlog(), hash=sim.event_stream_hash())


def test_joint_mask_partial_and_prior_actually_exercised():
    """집계 — 마스크가 **일부만** 살아 있는 행(= 조건부 제약이 실제로 물린 행) ≥ 5 · prior 있는 단계 ≥ 5.

    이게 0 이면 위 시험은 "전부 True 인 마스크" 만 비교한 셈이라 아무 것도 못 지킨다.
    """
    rows = [v for k, v in REPORT.items() if k.startswith("mask/")]
    if not rows:
        for label in sorted(STAGES):
            for pick in sorted(PICKS):
                test_joint_mask_and_sequential_conditional(label, pick)
        rows = [v for k, v in REPORT.items() if k.startswith("mask/")]
    assert sum(r["partial_mask"] for r in rows) >= 5, [r["partial_mask"] for r in rows]
    assert sum(r["prior_nonempty"] for r in rows) >= 5, [r["prior_nonempty"] for r in rows]
    assert sum(r["mask_cells"] for r in rows) >= 500


# ═══════════════════════════════════════════════ ② 선호 정렬키
@pytest.mark.parametrize("pref", sorted(PREF_CLS))
@pytest.mark.parametrize("label", ["eta-basic", "crowded-eta", "dead-first-eta", "eta-random-s1",
                                   "eta-random-s4", "block-arrival-s9"])
def test_pair_order_matches_v5_pair_key(label, pref):
    """v5 `_pair_key` 로 정렬한 쌍 (크레인, candidate_id) 목록 == 배열 `pair_order` 의 순서 (전건)."""
    prof, scn, level, _ = STAGES[label]()
    caps, _ = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    horizon, pre_advice = float(prof.decision_horizon_s), level == PA
    params = DP.resolver_params(tb, g)
    gen = CandidateGenerator(config=LEGACY_DEFAULT)
    resolver = CentralResolver(PREF_CLS[pref]())

    def decide(sim, dp, gb):
        """v5 CentralResolver 로 진행 — 정렬키 시험의 무대를 그 선호의 궤적으로 만든다."""
        resn = resolver.resolve(sim, dp, gb)
        return {r.crane_id: (gb[r.crane_id].items[r.chosen_candidate_id] if r.chosen_candidate_id is not None
                             else _wait_of(gb[r.crane_id])) for r in resn.resolutions}

    sim, recs, exc = drive_v5(prof, scn, level, decide, w0, tb, g, want_pairs=pref)
    assert recs and exc == 0
    n_pairs = 0
    for i, rec in enumerate(recs):
        tag = f"[{label}/{pref} #{i} t={rec.t:.3f}]"
        (out_j, fl_j, pr_j), (fl, pr), icol, _ = arrays_at(rec, tb, g, horizon, pre_advice)
        K, C = fl.raw.shape
        open_ = jnp.asarray([cid in rec.open_ids for cid in tb.crane_ids])
        valid = pr_j.keep & fl_j.feasible & open_[:, None]
        order = np.asarray(order_jit(g, pref)(params, rec.pre, fl_j, pr_j, valid))
        nv = int(np.asarray(valid).sum())
        assert nv == len(rec.pairs), f"{tag} 유효 쌍 수 arr={nv} v5={len(rec.pairs)}"
        got = [(int(p) // C, int(pr.candidate_id[int(p) // C, int(p) % C])) for p in order[:nv]]
        assert got == rec.pairs, f"{tag} ② 정렬 순\n  v5 ={rec.pairs}\n  arr={got}"
        n_pairs += nv
    REPORT[f"pref/{label}/{pref}"] = dict(decisions=len(recs), pairs=n_pairs,
                                          backlog=sim.unfinished_backlog())


def test_pref_cols_shape_contract():
    """세 선호가 `(tier int32, value f64, name int32)` 삼조로 정리된다 — sf_spt 만 앞에 두 열이 더 붙는다."""
    prof, scn, level, _ = STAGES["eta-basic"]()
    caps, _ = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    params = DP.resolver_params(tb, g)
    from yard_rl.v6.gpu import cands3 as C3
    out = C3.candidates3(w0, g, horizon_s=float(prof.decision_horizon_s), pre_advice=True)
    fl = C3.flat_view(out)
    want = {"baseline": 3, "sf_spt": 5, "fifo": 3}
    for pref, n in want.items():
        cols = VC.pref_cols(pref, params, w0, fl, g)
        assert len(cols) == n, (pref, len(cols))
        assert all(c.shape == fl.kind.shape for c in cols)
        assert cols[-1].dtype == jnp.int32 and cols[-2].dtype == jnp.float64
        assert cols[0].dtype == jnp.int32
    with pytest.raises(ValueError):
        VC.pref_cols("nope", params, w0, fl, g)
    assert set(VC.PREF_NAMES) == set(want)


# ═══════════════════════════════════════════════ ③ 전원 WAIT 예외 대체
def _mixed_decide(gen, level, *, every: int = 2):
    """번갈아 굴린다 — 짝수 결정은 **정상 순차 조건부**, 홀수 결정은 **첫 크레인**에게 실린
    infeasible 후보를 물려 v5 가 터지게 한다.

    ★왜 첫 크레인인가: v5 `_apply` 는 `sorted(assign)` 순이라 첫 크레인이 실패하면 **아무도 배정되기
      전에** 터져 `stage/episode.py:216` 의 전원 WAIT 대체가 그대로 관측된다. 뒤 크레인에서 터지면
      대체 `_apply` 가 앞 크레인을 다시 배정하려다 DECISION_COVERAGE 로 **잡히지 않고** 죽는다
      (v5cond 머리말 '한계 둘'). 번갈아 하는 것은 에피소드가 진행되게 해서 '안 터지는' 결정도 함께
      대조하기 위한 것이다 (술어가 상수여도 통과하는 시험이 되지 않게).
    """
    state = {"i": 0}
    seq = _seq_decide(gen, level, _py_pick_first)

    def decide(sim, dp, gb):
        i = state["i"]
        state["i"] = i + 1
        if i % every == 0:
            return seq(sim, dp, gb)[0]
        first = sorted(dp.crane_ids)[0]
        out = {c: _wait_of(gb[c]) for c in dp.crane_ids}
        bad = next((gc for gc in gb[first].items if not gc.feasible), None)
        if bad is not None:
            out[first] = bad
        return out
    return decide


@pytest.mark.parametrize("label", sorted(STAGES))
def test_all_wait_substitution_matches_v5(label):
    """예외를 일부러 유발해 굴린 v5 (episode.py:209-216 과 같은 try/except) 와 배열 판정·결정열이 `==`.

    v5 가 **대체 전에 고른 것**(`rec.proposed`)을 그대로 배열 열 번호로 옮겨 `decision_would_raise` 를
    묻고, 참이면 `substitute_all_wait` 이 전원 WAIT 을 내는지 본다.
    """
    prof, scn, level, _ = STAGES[label]()
    caps, _ = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    horizon, pre_advice = float(prof.decision_horizon_s), level == PA
    gen = CandidateGenerator(config=LEGACY_DEFAULT)
    sim, recs, exc = drive_v5(prof, scn, level, _mixed_decide(gen, level), w0, tb, g, guard=True)
    assert recs
    K = len(tb.crane_ids)
    n_fail = 0
    for i, rec in enumerate(recs):
        tag = f"[{label} #{i} t={rec.t:.3f}]"
        (out_j, fl_j, pr_j), (fl, pr), icol, keymap = arrays_at(rec, tb, g, horizon, pre_advice)
        N = rec.pre.n
        open_ = np.asarray([cid in rec.open_ids for cid in tb.crane_ids])
        choice = np.full((K,), EMPTY_ID, np.int32)
        for c in rec.open_ids:
            k = tb.crane_ids.index(c)
            choice[k] = keymap[k][rec.proposed[c]]
        failed_j, first_bad = raise_jit(g)(rec.pre, fl_j, pr_j, jnp.asarray(choice), jnp.asarray(open_))
        failed = bool(failed_j)
        assert failed == rec.failed, f"{tag} ③ 실패 판정 arr={failed} v5={rec.failed} 선택={rec.proposed}"
        if failed:
            n_fail += 1
            assert int(first_bad) == tb.crane_ids.index(sorted(rec.open_ids)[0]), f"{tag} 처음 갈리는 크레인"
        else:
            assert int(first_bad) == EMPTY_ID, tag
        ch2, ls2 = VC.substitute_all_wait(fl_j, jnp.asarray(choice), jnp.zeros((K,), bool),
                                          jnp.asarray(open_), failed_j)
        got = {}
        for k, cid in enumerate(tb.crane_ids):
            if not open_[k]:
                continue
            got[cid] = _arr_key(fl, k, int(np.asarray(ch2)[k]), N, tb)
        assert got == rec.chosen, f"{tag} ③ 대체 뒤 결정 v5={rec.chosen} arr={got}"
        assert not bool(np.asarray(ls2).any()), f"{tag} 대체 경로는 yield_count 를 올리지 않는다"
    REPORT[f"wait/{label}"] = dict(decisions=len(recs), v5_exceptions=exc, arr_failed=n_fail,
                                   ok_decisions=len(recs) - n_fail, backlog=sim.unfinished_backlog())
    assert n_fail == exc, (n_fail, exc)


def test_v5_exception_path_is_actually_reached():
    """집계 — 전원 WAIT 대체가 무대 전체에서 **실제로** 발화한다 (≥ 1). 안 나면 위 시험은 빈 대조다."""
    rows = [v for k, v in REPORT.items() if k.startswith("wait/")]
    if not rows:
        for label in sorted(STAGES):
            test_all_wait_substitution_matches_v5(label)
        rows = [v for k, v in REPORT.items() if k.startswith("wait/")]
    assert sum(r["v5_exceptions"] for r in rows) >= 1, [r["v5_exceptions"] for r in rows]
    assert sum(r["ok_decisions"] for r in rows) >= 1, "안 터지는 결정도 있어야 술어가 상수가 아니다"
    assert sum(r["arr_failed"] for r in rows) == sum(r["v5_exceptions"] for r in rows)


@pytest.mark.parametrize("label", ["eta-basic", "crowded-eta", "plan-failed-mandatory", "eta-random-s2"])
def test_decision_would_raise_matches_real_apply(label):
    """참·거짓 양쪽 — 결정마다 `deepcopy(sim)` 로 후보 하나하나 **실제로** `_apply` 해 보고 배열 판정과 대조.

    첫 크레인에게 items 의 후보를 차례로 물리고 나머지는 WAIT 로 둔 뒤, v5 가 예외를 내는지(참) 안 내는지
    (거짓)를 `decision_would_raise` 와 전건 `==`. 이것이 전원 WAIT 규칙의 **술어** 자체를 지키는 시험이다.
    """
    prof, scn, level, _ = STAGES[label]()
    caps, _ = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    horizon, pre_advice = float(prof.decision_horizon_s), level == PA
    gen = CandidateGenerator(config=LEGACY_DEFAULT)
    sim, recs, exc = drive_v5(prof, scn, level, _seq_decide(gen, level, _py_pick_first), w0, tb, g,
                              want_copy=True)
    K = len(tb.crane_ids)
    n_try, n_raise = 0, 0
    for i, rec in enumerate(recs[:8]):
        (out_j, fl_j, pr_j), (fl, pr), icol, keymap = arrays_at(rec, tb, g, horizon, pre_advice)
        N = rec.pre.n
        open_ = np.asarray([cid in rec.open_ids for cid in tb.crane_ids])
        first = sorted(rec.open_ids)[0]
        kf = tb.crane_ids.index(first)
        for j, key in enumerate(rec.items[first]):
            s2 = copy.deepcopy(rec.sim_copy)        # 결정이 **열린 채**로 찍은 사본 (pending 그대로)
            gb2 = {c: gen.generate(s2, c, level) for c in rec.open_ids}
            assert [_key(gc) for gc in gb2[first].items] == rec.items[first]
            assign = {c: _wait_of(gb2[c]) for c in rec.open_ids}
            assign[first] = gb2[first].items[j]
            raised = False
            try:
                _apply(s2, assign)
            except Exception:
                raised = True
            choice = np.full((K,), EMPTY_ID, np.int32)
            for c in rec.open_ids:
                k = tb.crane_ids.index(c)
                choice[k] = keymap[k][("WAIT",)]
            choice[kf] = keymap[kf][key]
            got, fb = raise_jit(g)(rec.pre, fl_j, pr_j, jnp.asarray(choice), jnp.asarray(open_))
            assert bool(got) == raised, (f"[{label} #{i} t={rec.t:.3f}] {first} {key}: "
                                         f"v5 raise={raised} arr={bool(got)}")
            assert int(fb) == (kf if raised else EMPTY_ID)
            n_try += 1
            n_raise += int(raised)
    REPORT[f"pred/{label}"] = dict(trials=n_try, raises=n_raise)
    assert n_try >= 3


def test_apply_predicate_saw_both_answers():
    """집계 — 위 술어 시험이 '터진다' 와 '안 터진다' 를 **둘 다** 봤다 (한쪽만 보면 술어가 상수여도 통과한다)."""
    rows = [v for k, v in REPORT.items() if k.startswith("pred/")]
    if not rows:
        for label in ["eta-basic", "crowded-eta", "plan-failed-mandatory", "eta-random-s2"]:
            test_decision_would_raise_matches_real_apply(label)
        rows = [v for k, v in REPORT.items() if k.startswith("pred/")]
    tot, raised = sum(r["trials"] for r in rows), sum(r["raises"] for r in rows)
    assert raised >= 1 and raised < tot, (raised, tot)


# ═══════════════════════════════════════════════ ④ 엔진 통합 (배열 엔진 완주)
@pytest.mark.parametrize("pick", sorted(PICKS))
@pytest.mark.parametrize("label", ["eta-basic", "crowded-eta", "dead-first-eta", "eta-random-s1",
                                   "spt-s7", "block-arrival-s9"])
def test_engine_end_to_end_with_seq_policy(label, pick):
    """`make_seq_policy` 를 공동 규약 정책으로 끼운 배열 엔진 완주 == 같은 규칙 v5 의 결정열 (크레인·종류·오더)."""
    prof, scn, level, _ = STAGES[label]()
    caps, _ = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    py_pick, arr_pick = PICKS[pick]
    gen = CandidateGenerator(config=LEGACY_DEFAULT)
    sim, recs, exc = drive_v5(prof, scn, level, _seq_decide(gen, level, py_pick), w0, tb, g)
    assert exc == 0
    v5_dec = [(r.t, r.open_ids, [(c, _kind_name(r.chosen[c]), _job_name(r.chosen[c], c))
                                 for c in r.open_ids]) for r in recs]
    policy_fn = VC.make_seq_policy(arr_pick, g)
    w, trace, tb2 = _EQ.run_array(prof, scn, policy_fn=policy_fn, params_fn=lambda w0_, tb_, g_: None,
                                  level=level, joint=True)
    assert int(w.violation) == 0, f"위반 비트 {int(w.violation)}"
    _EQ.compare_joint_decisions(v5_dec, w, trace, tb2, f"{label}/{pick}")
    REPORT[f"engine/{label}/{pick}"] = dict(decisions=len(recs), steps=int(w.steps),
                                            backlog=sim.unfinished_backlog(), hash=sim.event_stream_hash())


def test_guard_is_transparent_when_nothing_fails(label="crowded-eta"):
    """순차 조건부는 마스크가 막으므로 실패가 안 난다 — `guard_all_wait` 을 씌워도 **답이 그대로**다.

    (씌운 판과 안 씌운 판을 같은 무대에서 완주시켜 세계의 잎 전부를 비교한다. 겉옷이 옳은 결정을
    건드리면 여기서 큰 소리로 걸린다.)
    """
    prof, scn, level, _ = STAGES[label]()
    g = Geom.from_profile(prof)
    a = VC.make_seq_policy(VC.pick_first_masked, g, guard=True)
    b = VC.make_seq_policy(VC.pick_first_masked, g, guard=False)
    outs = []
    for fn in (a, b):
        w, trace, tb = _EQ.run_array(prof, scn, policy_fn=fn, params_fn=lambda w0_, tb_, g_: None,
                                     level=level, joint=True)
        assert int(w.violation) == 0
        outs.append((w, trace))
    for x, y in zip(jax.tree_util.tree_leaves(outs[0]), jax.tree_util.tree_leaves(outs[1])):
        assert bool(jnp.all(_same(x, y))), "전원 WAIT 겉옷이 옳은 결정을 바꿨다"


@pytest.mark.parametrize("label", ["eta-basic", "crowded-eta", "spt-s7"])
def test_guard_on_resolver_is_transparent_and_v5_never_raises(label):
    """v5 `_rule_policy` 조합(ResolverPolicy + 전원 WAIT 대체) 그대로 — **예외가 한 번도 안 난다**.

    `CentralResolver` 는 `dry_run_commit` 으로 수용한 것만 배정하므로 `_apply` 가 터질 수 없다
    (불변식 D-ORACLE, resolver.py 머리말 5행). 그래서 v5 의 `policy_exceptions` 는 0 이고, 배열
    `guard_all_wait` 을 `dispatch.make_resolver` 위에 씌워도 **세계가 비트 그대로**여야 한다 —
    그것이 "겉옷이 옳은 결정을 안 건드린다" 의 규칙 정책 판 증거다.
    """
    prof, scn, level, pref = STAGES[label]()
    pref_name = {"baseline": "baseline", "spt": "sf_spt"}[pref]
    caps, _ = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)

    # ── v5: episode.py:209-216 과 같은 try/except + ResolverPolicy
    from yard_rl.v6.world.integrated.baselines import ResolverPolicy
    pol = ResolverPolicy(PREF_CLS[pref_name](), pref_name)
    sim, recs, exc = drive_v5(prof, scn, level, (lambda s_, dp_, gb_: pol.decide(s_, dp_, gb_)),
                              w0, tb, g, guard=True)
    assert exc == 0, f"규칙 정책에서 예외가 났다 ({exc}) — D-ORACLE 위반"
    assert all(not r.failed for r in recs)

    # ── 배열: 같은 resolver 를 겉옷 유무로 두 번 (count_lost=False = baselines._apply 규약)
    base = DP.make_resolver(pref_name, g, count_lost=False)
    outs = []
    for fn in (base, VC.guard_all_wait(base, g)):
        w, trace, tb2 = _EQ.run_array(prof, scn, policy_fn=fn,
                                      params_fn=lambda w0_, tb_, g_: DP.resolver_params(tb_, g_),
                                      level=level, joint=True)
        assert int(w.violation) == 0
        outs.append((w, trace))
    for x, y in zip(jax.tree_util.tree_leaves(outs[0]), jax.tree_util.tree_leaves(outs[1])):
        assert bool(jnp.all(_same(x, y))), "겉옷이 규칙 resolver 의 답을 바꿨다"
    REPORT[f"resolver-guard/{label}"] = dict(decisions=len(recs), v5_exceptions=exc, pref=pref_name)


def _same(x, y):
    """비트 동일 — NaN 은 NaN 과 같다고 본다 (빈 칸 표시가 NaN 인 열이 많다)."""
    eq = x == y
    if jnp.issubdtype(x.dtype, jnp.floating):
        eq = eq | (jnp.isnan(x) & jnp.isnan(y))
    return eq


def _kind_name(key):
    return "WAIT" if key == ("WAIT",) else key[0]


def _job_name(key, cid):
    if key == ("WAIT",):
        return None
    if key[0] == "REPOSITION":
        from yard_rl.v6.gpu.host_convert import repo_job_id
        return repo_job_id(cid, key[1])
    return key[1]


# ═══════════════════════════════════════════════ ⑤ 단위·규약
@pytest.mark.parametrize("label", ["eta-basic", "crowded-eta", "eta-random-s3"])
def test_feasible_joint_matches_v5(label):
    """`feasible_joint` == v5 `baselines._feasible_joint` — 결정마다 공동 후보 조합을 무작위로 40조."""
    prof, scn, level, _ = STAGES[label]()
    caps, _ = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    horizon, pre_advice = float(prof.decision_horizon_s), level == PA
    gen = CandidateGenerator(config=LEGACY_DEFAULT)
    sim, recs, _ = drive_v5(prof, scn, level, _seq_decide(gen, level, _py_pick_first), w0, tb, g,
                            want_copy=True)
    rng = random.Random(0)
    K = len(tb.crane_ids)
    n, n_true = 0, 0
    for i, rec in enumerate(recs[:6]):
        (out_j, fl_j, pr_j), (fl, pr), icol, keymap = arrays_at(rec, tb, g, horizon, pre_advice)
        N = rec.pre.n
        s2 = copy.deepcopy(rec.sim_copy)
        gb2 = {c: gen.generate(s2, c, level) for c in rec.open_ids}
        for _ in range(40):
            assign, trial = {}, np.full((K,), EMPTY_ID, np.int32)
            for c in rec.open_ids:
                k = tb.crane_ids.index(c)
                gc = rng.choice(gb2[c].items)
                assign[c] = gc
                trial[k] = keymap[k][_key(gc)]
            want = bool(_feasible_joint(s2, assign))
            got = bool(feas_jit(g)(rec.pre, fl_j, jnp.asarray(trial)))
            assert got == want, (f"[{label} #{i}] feasible_joint v5={want} arr={got} "
                                 f"{[(c, _key(g_)) for c, g_ in assign.items()]}")
            n += 1
            n_true += int(want)
    REPORT[f"fj/{label}"] = dict(trials=n, feasible=n_true)
    assert n >= 40 and 0 < n_true < n, (n, n_true)


def test_prior_open_matches_python():
    """`prior_open` == v5 `next(reversed(selected.values()))` 의 뜻 (열린 크레인 중 직전 번호)."""
    rng = random.Random(7)
    for K in (1, 2, 3, 4, 5):
        for _ in range(40):
            op = [rng.random() < 0.6 for _ in range(K)]
            ks = [k for k in range(K) if op[k]]
            want = [EMPTY_ID] * K
            for j, k in enumerate(ks):
                want[k] = ks[j - 1] if j > 0 else EMPTY_ID
            got = np.asarray(VC.prior_open(jnp.asarray(op))).tolist()
            for k in ks:
                assert got[k] == want[k], (K, op, got, want)


def test_item_cols_and_wait_contract():
    """items 열표 규약 — WAIT 은 항상 마지막 items 이고 항상 feasible (v5 35행 RuntimeError 도달 불가 근거)."""
    prof, scn, level, _ = STAGES["crowded-eta"]()
    caps, _ = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    from yard_rl.v6.gpu import cands3 as C3
    out = C3.candidates3(w0, g, horizon_s=float(prof.decision_horizon_s), pre_advice=True)
    fl, pr = C3.flat_view(out), C3.prune(C3.flat_view(out), g)
    C = fl.raw.shape[1]
    assert VC.wait_col(fl) == C - 1
    assert bool(jnp.all(fl.feasible[:, C - 1])) and bool(jnp.all(fl.kind[:, C - 1] == PK_WAIT))
    for k in range(w0.k):
        icol = np.asarray(VC.item_cols(pr, k))
        used = [int(c) for c in icol if int(c) >= 0]
        assert len(set(used)) == len(used)
        assert used[int(np.asarray(pr.n_kept)[k])] == C - 1, "WAIT 은 n_kept 번째 items"
        assert len(used) == int(np.asarray(VC.n_items(pr))[k])
    assert VC.item_max(12) == 13


def test_truncation_flag_when_items_overflow():
    """items 칸을 억지로 줄이면 조용히 자르지 않고 `V_RESOLVER_TRUNC` 를 켠다."""
    prof, scn, level, _ = STAGES["crowded-eta"]()
    caps, _ = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    from yard_rl.v6.gpu import cands3 as C3
    out = C3.candidates3(w0, g, horizon_s=float(prof.decision_horizon_s), pre_advice=True)
    fl = C3.flat_view(out)
    pr = C3.prune(fl, g)
    open_ = jnp.ones((w0.k,), bool)
    lo = VC.sequential_conditional(w0, fl, pr, open_, VC.pick_first_masked, g, k_max=1)
    assert int(lo.flags) == V_RESOLVER_TRUNC
    hi = VC.sequential_conditional(w0, fl, pr, open_, VC.pick_first_masked, g)
    assert int(hi.flags) == 0


def test_guard_and_policy_are_cached_objects():
    """`lru_cache` 규약 — 같은 인자면 **같은 함수 객체** (jit static 키에 id 가 들어간다)."""
    prof, scn, _, _ = STAGES["eta-basic"]()
    g = Geom.from_profile(prof)
    a = VC.make_seq_policy(VC.pick_first_masked, g)
    b = VC.make_seq_policy(VC.pick_first_masked, g)
    assert a is b
    c = VC.make_seq_policy(VC.pick_last_masked, g)
    assert c is not a
    d = VC.make_seq_policy(VC.pick_first_masked, g, guard=False)
    assert d is not a
    assert VC.guard_all_wait(d, g) is VC.guard_all_wait(d, g)


def test_jit_eager_vmap_agree():
    """jit·eager·vmap(세계 2개) 의 잎이 전부 비트 동일 — 학습 경로가 배치로 돌 자리."""
    prof, scn, level, _ = STAGES["eta-basic"]()
    caps, _ = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    from yard_rl.v6.gpu import cands3 as C3
    hz = float(prof.decision_horizon_s)
    out = C3.candidates3(w0, g, horizon_s=hz, pre_advice=True)
    fl, pr = C3.flat_view(out), C3.prune(C3.flat_view(out), g)
    op = jnp.ones((w0.k,), bool)
    f = lambda w, fl_, pr_, op_: VC.sequential_conditional(w, fl_, pr_, op_, VC.pick_first_masked, g)
    eager = f(w0, fl, pr, op)
    jitted = jax.jit(f)(w0, fl, pr, op)
    for a, b in zip(jax.tree_util.tree_leaves(eager), jax.tree_util.tree_leaves(jitted)):
        assert bool(jnp.all(a == b)), "jit 과 eager 가 갈린다"
    stack = lambda t: jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), t)
    vm = jax.vmap(f, in_axes=(0, 0, 0, 0))(stack(w0), stack(fl), stack(pr), stack(op))
    for a, b in zip(jax.tree_util.tree_leaves(eager), jax.tree_util.tree_leaves(vm)):
        assert bool(jnp.all(b[0] == a)) and bool(jnp.all(b[1] == a)), "vmap 이 갈린다"


def test_zz_report(capsys):
    """집계 보고 — 무엇을 얼마나 대조했나."""
    with capsys.disabled():
        print("\n=== 조각 7 조건부 규약 (key=cond) 집계 ===")
        for k in sorted(REPORT):
            print(f"  {k}: {REPORT[k]}")
