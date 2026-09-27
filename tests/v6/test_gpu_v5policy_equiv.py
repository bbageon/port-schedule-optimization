"""★**학습 정책망**이 v5 와 같은 결정을 내는가 ([[YR-327]] 조각 7 통합 · key=integrate).

조각 1~6 은 "세계"를 옮겼고 규칙 정책(SF_SPT resolver)까지 v5 와 비트 일치했다. 여기서 옮기는 것은
v5 가 **학습에 쓰는 망** (`ppo/model.BlockPolicy` — tanh · 37특징 · 행동점수 + 상태가치) 과 그 망이
결정을 내는 절차 (`ppo/crane.CraneActor` — 크레인마다 마스크를 주는 순차 조건부) 다.
같은 가중치를 주면 같은 답이 나와야 조각 8(학습 루프)에서 v5 체크포인트를 v6 에서 굴려 대조할 수 있다.

■ 무엇을 지키나 — 기대값을 손으로 적지 않는다. v5 를 **실제로 굴려** 받은 값과 맞댄다.
  (a) **한 결정 단위** (`test_one_decision_matches_v5`) — 크레인 2 무대에서 v5 를 완주시키며
      결정마다·크레인마다 v5 가 망에 넣은 것과 받은 것을 그 자리에서 가로챈다:
        · 블록 요약 8칸 (`block_features` 대 `v5feat.block_row`)            — `==`
        · 후보 행렬 37칸 (float32, 망이 실제로 보는 값)                      — `==`
        · 마스크 (`ppo/crane.joint_mask` 대 `v5cond.joint_mask_items`)        — `==`
        · 행동 점수·상태 가치·로그확률·확률                                   — |Δ| 허용오차 (아래 ★)
        · **최종 선택**(`probs.argmax`) 과 한 벌 전체(`sequential_conditional`) — `==`
        · 직전 물은 크레인 (`ppo/crane.py:53` 대 `v5cond.prior_open`)          — `==`
  (b) **하루 전체** (`test_y01_net_ground_truth_replay`) — `scripts/v6/dump_ground_truth.py --policy v5net`
      이 남긴 정답 궤적(블록 Y01 · 본선 240건 · 크레인 2)을 배열 엔진 `run`(jit) 이 **사건 하나하나·해시·
      비용 13항·KPI·결정 수**까지 그대로 내는가. 망 셋으로: 고정 시드 무작위 둘 + **실제 학습 체크포인트**.
  (c) **경합이 실제로 일어났나** (`test_contention_actually_happens`) — 크레인 2 에서 DUP_JOB(같은 일감을
      두 크레인이) · CRANE_INTERFERENCE(안전거리) · 공동 비성립(dry_run) · LOST_CONTENTION(규칙 resolver 쪽)
      이 **0 이 아닌지** 센다. 0 이면 (a)(b) 의 경합 경로는 대조되지 않은 것이다.
  (d) **터미널 21블록** (`test_terminal_net_truth_is_reproducible`) — 배열 대조가 **아니다**. 터미널 정답
      궤적이 저장된 가중치로 다시 굴려도 같은지(재현성)와, 블록 요약의 **기록 기반 네 칸**이 거기서
      실제로 밟히는지만 본다. 배열 쪽 터미널 학습 정책망은 조각 8 몫이다 (그 시험 본문 ★ 참조).
  (e) ★**외부트럭이 있는 하루** (`test_net_day_array_matches_v5`, 2026-09-26 검증 반박으로 추가) —
      (b) 의 Y01 은 외부트럭이 **0** 이어서 블록 요약의 기록 기반 네 칸(블록 안·오는 중·곧 올 통지·
      줄 선 대수)과 후보 칸 13·15(외부트럭·누적대기), 필수(mandatory) 후보, PRE_REHANDLE 선택,
      교착 탈출이 **하나도** 안 밟혔다. 그래서 트럭 22대 + 본선 1척인 하루 무대를 하나 더 만들어
      배열 엔진이 **스스로** 그 열들을 채우며 v5 와 같은 궤적을 내는지 본다. 대조 항목:
        · 사건 로그 전열·해시·비용 13항·KPI·결정 수                                   — `==`
        · 오더별 (상태·크레인·재처리·S·C) **와 원장 세 칸 (게이트인 A·블록도착 B·게이트아웃 O)** — `==`
          ← (a)(b) 어디서도 대조되지 않던 열이다 ((a) 는 v5 장부를 옮겨 심고, (b) 는 트럭이 0 이었다)
        · **결정마다** 배열 세계가 스스로 만든 블록 요약 8칸 대 v5 `block_features`             — `==`
          ← 배열 엔진을 결정마다 세워(`jax.jit(step)` + 파이썬 루프) 그 자리에서 잰다
      기대값은 저장하지 않는다 — v5 를 그 자리에서 함께 굴려 받는다 (하루 0.3초).

■ ★이 층만은 v5 와 **비트 일치가 불가능**하다 (`gpu/v5net.py` 머리말의 측정)
  v5 는 float32 · torch(MKL GEMM · libm tanh), 배열판은 float64 · XLA 다. 원인이 둘 다 우리 코드 밖이라
  막을 수단이 없다. 그래서 이 층의 동등성 기준은 **"결정(argmax)이 같다 + |Δ| 가 허용오차 안
  + 1·2위 최소 격차가 |Δ| 보다 충분히 크다"** 이다. 시험은 격차를 찍기만 하지 않고 **단언한다**
  (`test_zz_report`: 최소격차 ≥ 20 × |Δ|).
  ⚠️ **갈릴 확률은 0 이 아니다** — 망 2,600벌 × 85만 2,800 결정 재생 대조에서 뒤집힘 **4건
     (4.7e-06/결정)**, 최악 여유 **0.17배**였다 (`scripts/v6/probe_net_flip_rate.py --nets 2600` ·
     `outputs/v6/net_flip_rate.json` · v5net.py 머리말 ★★). 여기 통과는 "이 표본에서 안 갈렸다" 는 뜻이고
     "갈릴 여지가 없다" 가 아니다. 조각 8 은 '해시 일치' 대신 **'첫 갈린 결정까지의 접두사 일치'** 로
     판정해야 한다.

실행 (Git Bash · Windows 파이썬 · CPU x64):
    PYTHONPATH="src;tests/v6" PYTHONIOENCODING=utf-8 .venv-jax/Scripts/python.exe \
        -m pytest tests/v6/test_gpu_v5policy_equiv.py -q -p no:cacheprovider -s
"""
from __future__ import annotations

import importlib.util
import json
import os

import numpy as np                                  # ★numpy → jax → torch 순서 (OMP #15 방어)
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)           # ★float64 — 동등성은 x64 에서만
jnp = jax.numpy
torch = pytest.importorskip("torch")

from yard_rl.v6.gpu import cands3 as C3                                              # noqa: E402
from yard_rl.v6.gpu import dispatch as DP                                            # noqa: E402
from yard_rl.v6.gpu import engine_step as ES                                         # noqa: E402
from yard_rl.v6.gpu import v5cond as VC                                              # noqa: E402
from yard_rl.v6.gpu import v5feat as VF                                              # noqa: E402
from yard_rl.v6.gpu import v5net as VN                                               # noqa: E402
from yard_rl.v6.gpu.geom import Geom                                                 # noqa: E402
from yard_rl.v6.gpu.host_convert import event_stream_hash, from_block_world, to_block_world   # noqa: E402
from yard_rl.v6.gpu.reserve import CRANE_INTERFERENCE, DUP_JOB                       # noqa: E402
from yard_rl.v6.gpu.state import (COST_TERMS, PK_PRE_REHANDLE, PK_REPOSITION, PK_SERVE, PK_WAIT,   # noqa: E402
                                  V_NET_NONFINITE)                                    # noqa: E402
# ── v5 정본 (읽기만 한다) ──────────────────────────────────────────────────────
from yard_rl.v6.world.domain.enums import InformationLevel                           # noqa: E402
from yard_rl.v6.world.integrated.engine import TerminalSimulator                     # noqa: E402

PA = InformationLevel.PRE_ADVICE
#: 결정마다 items 칸 폭 (v5 `_prune` 의 k_max + WAIT 1)
I_MAX = VC.item_max()
#: 점수·가치의 허용오차 — float32(v5) 대 float64(배열) 의 실측 상한 1.21e-07 보다 **한 자리** 여유.
#: ★2026-09-26 검증 반박으로 1e-5 → 1e-6 으로 조였다. 1e-5 는 실측 |Δ|(5.3e-08) 의 약 200배라
#:   5e-06 급 회귀(예: 가중치를 float32 로 잘못 깎아 싣는 실수)도 통과했다.
TOL = 1e-6
#: ★1·2위 최소 격차가 점수 |Δ| 보다 이만큼은 커야 한다 — 결정이 갈릴 여지의 하한 (test_zz_report 가 단언).
#:   층 자체의 갈림 확률은 6e-06/결정이므로(v5net.py 머리말) 이 배율이 줄면 회귀로 본다.
GAP_MARGIN = 20.0
#: 시험 전체 집계 (마지막 시험이 찍는다)
REPORT: dict[str, dict] = {}
_TRUTH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                          "outputs", "reports", "yr327_v6_port", "ground_truth")


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_HERE = os.path.dirname(os.path.abspath(__file__))
#: 특징 담당 시험의 하네스를 그대로 쓴다 — 무대·`world_from_sim`·명단(Order)·`gate_out_s` 보정
_FEAT = _load("_tgv5feat_for_policy", os.path.join(_HERE, "test_gpu_v5feat.py"))
#: 정답 궤적 덤프 스크립트 (v5 쪽 구동기·가중치 적재)
D = _load("_dgt_for_v5policy", os.path.join(_HERE, "..", "..", "scripts", "v6", "dump_ground_truth.py"))

STAGES = _FEAT.STAGES
world_from_sim, _caps = _FEAT.world_from_sim, _FEAT._caps
_with_gate_out, _orders_of, _reserve_arr = _FEAT._with_gate_out, _FEAT._orders_of, _FEAT._reserve_arr
BID = _FEAT.BID
KIND_NAME = {PK_SERVE: "SERVE", PK_PRE_REHANDLE: "PRE_REHANDLE", PK_REPOSITION: "REPOSITION",
             PK_WAIT: "WAIT", -1: "-"}


# ───────────────────────────────────────────────── v5 쪽 — 결정마다 가로채기
def run_v5_net(prof, scn, w0, tb, g, end_s: float, policy, *, max_decisions=None, record: bool = True,
               reserve_of=None):
    """v5 를 **학습 정책망**으로 완주시키며 결정마다 기록을 남긴다.

    결정 절차는 v5 원본 `ppo/crane.CraneActor` 그대로다 (`dump_ground_truth._net_policy` 가 그것을 부른다) —
    `joint_mask`·`candidate_row`·`encode`·`BlockPolicy.distribution`·`_apply` 를 여기서 다시 쓰지 않는다.

    `max_decisions` 는 **None(끝까지)** 이 기본이다 — 중간에 끊으면 세계가 덜 굴러 해시가 달라진다.
    `record=False` 면 결정별 배열 세계(`world_from_sim`)·가로챈 행렬을 안 모은다 (하루 전체 대조용).
    `reserve_of` 는 `jid -> 통지된 도착예정 초` — 없으면 하네스가 실현 게이트인을 쓴다
    (`dump_ground_truth.net_terminal_inputs` 머리말 ★ 정보 경계). 트럭이 있는 무대는 반드시 준다.

    돌려주는 `rt` 에 `block_trace` 를 달아 둔다 — 결정마다 `(시각, 블록 요약 8칸)` 이라 (e) 하루 시험이
    배열 엔진이 **스스로 만든** 8칸과 결정 하나하나 맞댈 수 있다.
    """
    sim = TerminalSimulator(prof, scn, check_invariants=True, info_level=PA)
    picks: list[dict] = []
    pol, rt, exc = D._net_policy(policy, sim, BID, end_s, PA, on_select=picks.append,
                                 reserve_of=reserve_of)
    #: 블록 요약은 `CraneActor` 안에서 `_sync` **뒤에** 계산된다 — 그 값을 그 자리에서 받는다
    blocks: list[list[float]] = []
    block_trace: list[tuple] = []
    rt.block_trace = block_trace
    orig_bs = rt.block_state

    def hooked_block_state(bid, t):
        v = orig_bs(bid, t)
        blocks.append([float(x) for x in v])
        block_trace.append((float(t), np.asarray([float(x) for x in v], np.float64)))
        return v

    if not record:
        rt.on_select = None                      # 순전파 한 번씩 덜 — 하루 전체는 최종 상태만 본다
    rt.block_state = hooked_block_state
    recs, n_dec = [], 0
    while (dp := sim.run_until_decision()) is not None:
        now = sim.now
        n_dec += 1
        if not record:
            pol(sim, dp)
            continue
        escape = bool(sim.event_log) and sim.event_log[-1][1] == "DEADLOCK_ESCAPE" and sim.event_log[-1][0] == now
        pre = _with_gate_out(world_from_sim(sim, w0, tb, g), sim, tb)
        base = len(picks)
        pol(sim, dp)
        recs.append(dict(t=now, pre=pre, escape=escape, crane_ids=sorted(dp.crane_ids),
                         block=blocks[-1], picks=picks[base:]))
        if max_decisions is not None and len(recs) >= max_decisions:
            break
    return sim, recs, exc["n"], rt, n_dec


# ───────────────────────────────────────────────── 배열 쪽
_JIT: dict = {}


def _cands_jit(g: Geom, pre_advice: bool):
    key = (g, pre_advice)
    if key not in _JIT:
        def f(w, hz):
            c3 = C3.candidates3(w, g, horizon_s=hz, pre_advice=pre_advice)
            fl = C3.flat_view(c3)
            return c3, fl, C3.prune(fl, g)
        _JIT[key] = jax.jit(f)
    return _JIT[key]


def _np(tree):
    return jax.tree_util.tree_map(np.asarray, tree)


def _gap(scores: np.ndarray, mask: np.ndarray) -> float:
    """마스크 안 1·2위 점수 격차 (후보가 하나면 +inf) — 결정이 갈릴 여지의 눈금."""
    v = np.sort(scores[mask])[::-1]
    return float("inf") if v.size < 2 else float(v[0] - v[1])


def check_stage(label: str, policy, params_of) -> dict:
    """무대 하나 — 결정마다·크레인마다 v5 와 배열판을 맞댄다. 돌려주는 값은 보고용 집계."""
    prof, scn = STAGES[label]()
    caps, _ = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    hz = float(prof.decision_horizon_s)
    end_s = float(scn.horizon_s) + float(scn.drain_window_s)
    crane_order = tuple(tb.crane_index[c.crane_id] for c in prof.cranes)
    sim, recs, exc_n, rt, _n = run_v5_net(prof, scn, w0, tb, g, end_s, policy,
                                          max_decisions=_FEAT.MAX_DECISIONS)
    assert recs, f"{label}: 결정이 하나도 없다 — 무대가 공허하다"
    assert exc_n == 0, (f"{label}: v5 학습 경로에서 예외 {exc_n} 건 — `CraneActor` 는 마스크가 곧 예약 가능 "
                        f"집합이라 구조상 실패가 안 나야 한다 (ppo/crane.py:81)")
    res_arr = _reserve_arr(_orders_of(sim), tb, w0.n)
    params = params_of(w0, end_s, res_arr)
    bind = DP.make_v5net_pick(g, crane_order=crane_order)
    cidx = tb.crane_index
    fn = _cands_jit(g, True)

    n_dec = n_crane = n_rows = 0
    n_skip_escape = 0
    worst = dict(score=0.0, value=0.0, logp=0.0, prob=0.0)
    min_gap = float("inf")
    chosen_kinds: dict[str, int] = {}
    prior_kinds: dict[str, int] = {}
    n_mask_off = 0
    for i, rec in enumerate(recs):
        if rec["escape"]:
            # 탈출 결정은 v5 가 yielded **해제 뒤** 후보를 세므로 `pre`(해제 전) 에서 재현할 수 없다.
            # 엔진 경로(`decide_joint` → `recandidates`)가 그 자리를 덮는다.
            # ★2026-09-26 정정 — 전에 여기 "하루 전체 시험 (b) 가 판정한다" 고 적었는데 **사실과 달랐다**:
            #   (b) 의 Y01 정답 궤적 세 벌 모두 `deadlock_escapes` 가 0 이라 망 경로의 탈출 결정은
            #   어디서도 대조되지 않았다. 지금은 (e) `test_net_day_array_matches_v5` 의 하루 무대가
            #   망마다 탈출을 1~2회 내고 그 궤적을 배열 엔진이 사건 하나까지 재현한다 — 그쪽이 판정한다.
            n_skip_escape += 1
            continue
        pre, t = rec["pre"], float(rec["t"])
        K = pre.k
        c3, fl, pr = fn(pre, hz)
        fln, prn = _np(fl), _np(pr)
        blk = VF.block_row(pre, g, end_s=end_s, reserve_s=res_arr, crane_order=crane_order)
        want_blk = np.asarray(rec["block"], np.float64)
        bad = np.argwhere(np.asarray(blk) != want_blk)
        assert not len(bad), (f"{label} #{i} t={t:.3f}: 블록 요약이 갈린다 — 칸 "
                              f"{[(int(c), VF.FEATURE_NAMES[int(c)], float(want_blk[c]), float(np.asarray(blk)[c])) for (c,) in bad]}")

        open_ = np.zeros((K,), bool)
        for cid in rec["crane_ids"]:
            open_[cidx[cid]] = True
        open_j = jnp.asarray(open_)
        # ── 크레인 순 scan 을 손으로 되풀어 단계마다 대조 (배열 `sequential_conditional` 과 같은 순서) ──
        prior_arr = np.asarray(VC.prior_open(open_j))
        sel = np.full((K,), -1, np.int32)
        want_items, prev_open = [], -1
        for pos, cid in enumerate(rec["crane_ids"]):
            k = cidx[cid]
            p = rec["picks"][pos]
            assert p["role"] == "crane" and abs(p["t"] - t) < 1e-9, f"{label} #{i}: 가로챈 결정 시각이 어긋난다"
            assert int(prior_arr[k]) == prev_open, (
                f"{label} #{i} {cid}: 직전 물은 크레인 arr={int(prior_arr[k])} v5={prev_open} "
                f"(ppo/crane.py:53 `next(reversed(selected.values()))`)")
            # 마스크 (joint_mask) — items 칸
            m_j, icol_j = VC.joint_mask_items(pre, fl, pr, jnp.asarray(sel), k, g)
            m, icol = np.asarray(m_j), np.asarray(icol_j)
            want_mask = np.asarray(p["mask"], bool)
            n = int(want_mask.shape[0])
            assert np.array_equal(m[:n], want_mask), (
                f"{label} #{i} {cid}: joint_mask 가 갈린다 arr={m[:n].tolist()} v5={want_mask.tolist()}")
            assert not m[n:].any(), f"{label} #{i} {cid}: 없는 후보 칸의 마스크가 살아 있다"
            n_mask_off += int((~want_mask).sum())
            # 특징 37칸 (float32 — 망이 실제로 보는 값)
            pk_i = int(max(prior_arr[k], 0))
            pcol = int(sel[pk_i]) if prior_arr[k] >= 0 else -1
            pkind, pbay = VF.prior_from_choice(fl, pk_i, pcol)
            prior_kinds[KIND_NAME[int(pkind)]] = prior_kinds.get(KIND_NAME[int(pkind)], 0) + 1
            fo = VF.features(pre, g, fl, pr, block=blk, prior_kind=jnp.full((K,), pkind, jnp.int32),
                             prior_end_bay=jnp.full((K,), pbay, VF.F), c_max=I_MAX)
            assert int(fo.overflow) == 0, f"{label} #{i}: candidate_id 가 items 칸 {I_MAX} 를 넘었다"
            assert int(fo.n_items[k]) == n, (
                f"{label} #{i} {cid}: 후보 수 arr={int(fo.n_items[k])} v5={n}")
            x32 = np.asarray(VF.as_net_input(fo.x[k]))
            bad = np.argwhere(x32[:n] != np.asarray(p["x"]))
            assert not len(bad), (
                f"{label} #{i} {cid}: 망 입력 {len(bad)} 칸 불일치 — 첫 자리 후보 {bad[0][0]} 칸 "
                f"{VF.FEATURE_NAMES[bad[0][1]]} v5={p['x'][bad[0][0], bad[0][1]]!r} arr={x32[bad[0][0], bad[0][1]]!r}")
            # ★float64 로도 `==` (2026-09-26 검증 반박) — 위 대조는 float32 로 깎은 뒤라 마지막 비트 차이가
            #   캐스팅에 묻힌다. v5 `on_select` 가 캐스팅 **전** 24칸(`rows`)을 그대로 남겨 두므로 그것과 댄다.
            x64 = np.asarray(fo.x[k], np.float64)
            want64 = np.asarray(p["rows"], np.float64)
            bad64 = np.argwhere(x64[:n, :VF.CRANE_ROW_DIM] != want64)
            assert not len(bad64), (
                f"{label} #{i} {cid}: 망 입력이 **float64** 에서 {len(bad64)} 칸 불일치 — 첫 자리 후보 "
                f"{bad64[0][0]} 칸 {VF.FEATURE_NAMES[bad64[0][1]]} "
                f"v5={want64[bad64[0][0], bad64[0][1]]!r} arr={x64[bad64[0][0], bad64[0][1]]!r}")
            # 망 — 행동 점수·상태 가치·분포·최종 선택
            xn = jnp.asarray(x32, VF.F)
            sc = VN.actor_scores(params.net, xn)
            va = VN.state_values(params.net, xn)
            lp = VN.log_probs(sc, m_j)
            pb = VN.probs(sc, m_j)
            act = int(VN.greedy_action(sc, m_j))
            sc_n, va_n, lp_n, pb_n = np.asarray(sc), np.asarray(va), np.asarray(lp), np.asarray(pb)
            worst["score"] = max(worst["score"], float(np.abs(sc_n[:n] - p["raw_logits"]).max()))
            worst["value"] = max(worst["value"], float(np.abs(va_n[:n] - p["value"]).max()))
            fin = np.isfinite(p["logits"])
            worst["logp"] = max(worst["logp"], float(np.abs(lp_n[:n][fin] - p["logits"][fin]).max()))
            worst["prob"] = max(worst["prob"], float(np.abs(pb_n[:n] - p["probs"]).max()))
            assert act == p["action"], (
                f"{label} #{i} {cid}: **선택이 갈린다** arr={act} v5={p['action']} "
                f"(1·2위 격차 {_gap(sc_n, m):.3e} · 점수 |Δ| {np.abs(sc_n[:n] - p['raw_logits']).max():.3e})")
            min_gap = min(min_gap, _gap(sc_n, m))
            col = int(icol[act])
            chosen_kinds[KIND_NAME[int(fln.kind[k, col])]] = chosen_kinds.get(KIND_NAME[int(fln.kind[k, col])], 0) + 1
            want_items.append(act)
            sel[k] = col
            prev_open = k
            n_crane += 1
            n_rows += n
        for nm, v in worst.items():
            assert v <= TOL, f"{label} #{i}: {nm} 의 |Δ| {v:.3e} 가 허용오차 {TOL:.0e} 를 넘었다"
        # ── 한 벌 전체 — 배열 `sequential_conditional`(= policy_fn 이 부르는 것) 이 v5 와 같은 답인가 ──
        seq = VC.sequential_conditional(pre, fl, pr, open_j, bind(params, pre, fl, pr, blk), g)
        got_item, got_choice = np.asarray(seq.item), np.asarray(seq.choice)
        for pos, cid in enumerate(rec["crane_ids"]):
            k = cidx[cid]
            assert int(got_item[k]) == want_items[pos], (
                f"{label} #{i} {cid}: 순차 조건부 items 색인 arr={int(got_item[k])} v5={want_items[pos]}")
            assert int(got_choice[k]) == int(sel[k]), (
                f"{label} #{i} {cid}: 순차 조건부 열 번호 arr={int(got_choice[k])} 손계산={int(sel[k])}")
        assert (got_choice[~open_] == -1).all(), f"{label} #{i}: 안 물은 크레인에 답이 들어갔다"
        assert int(seq.flags) == 0, f"{label} #{i}: items 칸 넘침 (V_RESOLVER_TRUNC)"
        n_dec += 1
    assert n_dec, f"{label}: 대조한 결정이 0 (탈출 {n_skip_escape} 건뿐)"
    return dict(decisions=n_dec, escape_skipped=n_skip_escape, crane_decisions=n_crane, rows=n_rows,
                worst=dict(worst), min_gap=min_gap, chosen=chosen_kinds, priors=prior_kinds,
                masked_off=n_mask_off, net_calls=rt.n_select)


# ───────────────────────────────────────────────── 망 셋
def _fixed_net(seed: int, hidden: int = 64):
    return D.build_net(seed=seed, hidden=hidden)


def _params_of(policy):
    sd = {k: v.detach().cpu().numpy() for k, v in policy.state_dict().items()}

    def make(w0, end_s, reserve_s):
        return DP.v5net_params(sd, w0, end_s=end_s, reserve_s=reserve_s)
    return make


# ───────────────────────────────────────────────── (a) 한 결정 단위
#: ★망을 둘 쓴다 — 무작위 초기화 망은 **종류 칸의 가중치**가 어느 종류를 고를지 사실상 정해 버린다.
#:   한 망만 쓰면 고른 행동이 한 종류뿐이고, 그러면 직전 크레인 칸(19~23)의 네 종류가 안 밟힌다.
#:   seed 101·1234 는 SERVE·PRE_REHANDLE·REPOSITION·WAIT 를 **모두** 고르는 망이다 (실측으로 골랐다).
STAGE_NETS = [101, 1234]


@pytest.mark.parametrize("net_seed", STAGE_NETS)
@pytest.mark.parametrize("label", list(STAGES))
def test_one_decision_matches_v5(label, net_seed):
    """결정마다·크레인마다 — 마스크·37칸·점수·가치·선택·한 벌 전체가 v5 와 같은가 (크레인 2 무대)."""
    pol = _fixed_net(seed=net_seed)
    REPORT[f"(a) {label}/s{net_seed}"] = r = check_stage(label, pol, _params_of(pol))
    assert r["crane_decisions"] > 0
    assert r["chosen"], f"{label}: 고른 행동이 하나도 없다"


def test_two_cranes_see_each_others_commitment():
    """★순차 조건부가 **실제로** 작동하는가 — 둘째 크레인의 직전칸(19~23)이 첫째의 선택을 담았나.

    담기지 않으면 `prior_*` 다섯 칸이 언제나 0 이고, 순차 조건부를 옮긴 것이 공허하다.
    네 종류가 **모두** 직전칸에 실려야 한다 — 한 종류만 실리면 원핫 네 칸 중 셋은 대조되지 않는다.
    """
    seen: dict[str, int] = {}
    chosen: dict[str, int] = {}
    for k, r in REPORT.items():
        if not k.startswith("(a)"):
            continue
        for kk, v in r["priors"].items():
            seen[kk] = seen.get(kk, 0) + v
        for kk, v in r["chosen"].items():
            chosen[kk] = chosen.get(kk, 0) + v
    assert seen, "앞 시험 (a) 가 하나도 안 돌았다"
    assert seen.get("-", 0) > 0, "첫 크레인(직전 없음)이 한 번도 안 나왔다"
    assert sum(v for k, v in seen.items() if k != "-") > 0, (
        f"둘째 크레인이 한 번도 직전 선택을 못 봤다 — 순차 조건부가 공허하다 {seen}")
    for kind in ("SERVE", "PRE_REHANDLE", "REPOSITION", "WAIT"):
        assert chosen.get(kind, 0) > 0, f"망이 {kind} 를 한 번도 안 골랐다 — 그 경로가 대조되지 않았다 {chosen}"
        assert seen.get(kind, 0) > 0, f"직전칸에 {kind} 가 한 번도 안 실렸다 (칸 19~22 중 하나가 늘 0) {seen}"
    REPORT["(a) 합계"] = dict(priors=seen, chosen=chosen)


# ───────────────────────────────────────────────── (c) 경합이 실제로 일어났나
def test_contention_actually_happens():
    """크레인 2 에서 DUP_JOB · CRANE_INTERFERENCE · 공동 비성립 · LOST_CONTENTION 이 **0 이 아닌가**.

    (a)(b) 가 통과해도 경합이 한 번도 없었다면 그 경로는 대조되지 않은 것이다. 여기서는 v5 학습 경로를
    굴리며 결정 시작 시점의 거절 코드와 `joint_mask` 가 실제로 무엇을 막았는지 센다.
    LOST_CONTENTION 은 **규칙 resolver 만의** 사유(`resolver.py:96`)라 같은 세계에서 `resolve_central`
    (count_lost=True) 을 따로 불러 센다 — 학습 경로(`_apply`)는 yield_reason 을 넘기지 않는다.
    """
    pol = _fixed_net(seed=STAGE_NETS[0])
    params_of = _params_of(pol)
    tot = dict(dup_code=0, interf_code=0, dup_masked=0, joint_blocked=0, lost=0, decisions=0,
               k2_decisions=0)
    for label in STAGES:
        prof, scn = STAGES[label]()
        caps, _ = _caps(scn, prof)
        w0, tb = to_block_world(prof, scn, **caps)
        g = Geom.from_profile(prof)
        hz = float(prof.decision_horizon_s)
        end_s = float(scn.horizon_s) + float(scn.drain_window_s)
        sim, recs, exc_n, _rt, _n = run_v5_net(prof, scn, w0, tb, g, end_s, pol,
                                               max_decisions=_FEAT.MAX_DECISIONS)
        res_arr = _reserve_arr(_orders_of(sim), tb, w0.n)
        params = params_of(w0, end_s, res_arr)
        rp = DP.resolver_params(tb, g)
        cidx = tb.crane_index
        fn = _cands_jit(g, True)
        for rec in recs:
            if rec["escape"]:
                continue
            pre = rec["pre"]
            K = pre.k
            c3, fl, pr = fn(pre, hz)
            c3n, fln, prn = _np(c3), _np(fl), _np(pr)
            tot["decisions"] += 1
            open_ = np.zeros((K,), bool)
            for cid in rec["crane_ids"]:
                open_[cidx[cid]] = True
            if open_.sum() >= 2:
                tot["k2_decisions"] += 1
            # 결정 시작 시점 거절 코드 (SERVE·PRE·REPO 셋)
            for code, mask in ((c3n.m.code, c3n.m.disp & ~c3n.m.taken[None, :]),
                               (c3n.pre_code, c3n.pre_gate), (c3n.repo_code, c3n.repo_valid)):
                tot["dup_code"] += int(((code == DUP_JOB) & mask).sum())
                tot["interf_code"] += int(((code == CRANE_INTERFERENCE) & mask).sum())
            # joint_mask 가 막은 것 — 토큰(DUP_JOB 뜻) 과 공동 비성립(dry_run) 을 갈라 센다
            sel = np.full((K,), -1, np.int32)
            for cid in rec["crane_ids"]:
                k = cidx[cid]
                m = np.asarray(VC.joint_mask_items(pre, fl, pr, jnp.asarray(sel), k, g)[0])
                icol = np.asarray(VC.item_cols(pr, k))
                toks = np.asarray(VC.sel_tokens(fl, jnp.asarray(sel).at[k].set(-1)))
                taken = np.asarray(VC.taken_tokens(jnp.asarray(toks), pre.n))
                for i_it in range(I_MAX):
                    col = int(icol[i_it])
                    if col < 0 or not bool(fln.feasible[k, col]):
                        continue
                    kind = int(fln.kind[k, col])
                    tk = int(fln.job[k, col]) if kind in (PK_SERVE, PK_PRE_REHANDLE) else -1
                    if tk >= 0 and bool(taken[tk]):
                        tot["dup_masked"] += 1
                    elif not bool(m[i_it]) and kind != PK_WAIT:
                        tot["joint_blocked"] += 1
                # 정책이 아니라 결정적 대역으로 채운다 (여기서는 '무엇이 막혔나' 만 센다)
                idx = int(np.argmax(m)) if m.any() else -1
                sel[k] = int(icol[idx]) if idx >= 0 else -1
            # LOST_CONTENTION — 규칙 resolver 쪽
            _ch, lost, _fl = DP.resolve_central(rp, pre, c3, fl, pr, jnp.asarray(open_), g,
                                               pref="sf_spt", count_lost=True)
            tot["lost"] += int(np.asarray(lost).sum())
        del params
    REPORT["(c) 경합"] = tot
    assert tot["k2_decisions"] > 0, "두 크레인이 한꺼번에 물은 결정이 한 번도 없다"
    assert tot["interf_code"] > 0, f"CRANE_INTERFERENCE 가 한 번도 안 났다 {tot}"
    assert tot["dup_code"] + tot["dup_masked"] > 0, f"DUP_JOB(같은 일감 경합) 이 한 번도 안 났다 {tot}"
    assert tot["joint_blocked"] + tot["lost"] > 0, f"공동 비성립·LOST_CONTENTION 이 한 번도 안 났다 {tot}"


# ───────────────────────────────────────────────── (b) 하루 전체 — 정답 궤적 Y01
#: (표식, 사람이 읽을 이름) — `dump_ground_truth.py --policy v5net` 이 만든 궤적
NETS = [("seed1h64", "무작위 시드 1"), ("seed20260927h64", "무작위 기본 시드"),
        ("ckpt-policy", "실제 학습 체크포인트 yr302-final-train")]


def _truth(tag: str):
    p = os.path.join(_TRUTH_DIR, D.block_truth_name("Y01", 30, 9900777, D.NET, PA, tag))
    if not os.path.exists(p):
        return None, None
    truth = json.load(open(p, encoding="utf-8"))
    return truth, os.path.join(_TRUTH_DIR, truth["net_weights"])


def _y01_caps(scn):
    n0 = len(scn.jobs)
    n_max = max(8, 1 << (n0 - 1).bit_length())
    s_max = 8 * n_max + 256
    return dict(n_max=n_max, q_cap=max(32, 4 * n_max), log_cap=s_max + n_max), s_max


def _first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    if len(a) != len(b):
        return min(len(a), len(b)), (a[len(b)] if len(a) > len(b) else None), (b[len(a)] if len(b) > len(a) else None)
    return None


@pytest.mark.parametrize("tag,name", NETS, ids=[t for t, _ in NETS])
def test_y01_net_ground_truth_replay(tag, name):
    """하루 전체 — 배열 엔진(jit)이 학습 정책망으로 v5 정답 궤적을 **그대로** 낸다."""
    truth, npz = _truth(tag)
    if truth is None:
        pytest.skip(f"정답 궤적 없음 ({tag}) — scripts/v6/dump_ground_truth.py --mode block --block Y01 "
                    f"--policy v5net 로 만든다")
    label = f"(b) Y01-{tag}"
    prof, built = D._build(truth["load"], truth["seed"])
    scn = built["scenarios"][truth["block"]]
    caps, s_max = _y01_caps(scn)
    w0, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    hz = float(prof.decision_horizon_s)
    crane_order = tuple(tb.crane_index[c.crane_id] for c in prof.cranes)
    params = DP.v5net_params(D.load_net_npz(npz), w0, end_s=truth["bridge_end_s"])
    pol = DP.make_v5net_policy(g, crane_order=crane_order)
    w, trace = ES.run_jit(w0, params, g, pol, s_max, True, True, hz, True)
    d = from_block_world(w, tb)
    # ① 사건 로그 전열 · 해시
    arlog = [(round(t, 6), k, p) for (t, k, p) in d["event_log"]]
    tlog = [(round(t, 6), k, p) for (t, k, p) in truth["events"]]
    diff = _first_diff(arlog, tlog)
    if diff is not None:
        i, x, y = diff
        ctx = "\n".join(f"    #{j}: arr={arlog[j] if j < len(arlog) else '-'}  |  v5={tlog[j] if j < len(tlog) else '-'}"
                        for j in range(max(0, i - 4), min(max(len(arlog), len(tlog)), i + 4)))
        pytest.fail(f"[{label}] ① 사건 로그가 {i}번째에서 갈린다: arr={x} v5={y}\n{ctx}\n"
                    f"  (arr {len(arlog)}건 · v5 {len(tlog)}건 · violation={d['violation_names']} "
                    f"overflow={d['overflow']} terminal={d['terminal']})")
    h = event_stream_hash(w, tb)
    assert h == truth["event_hash"], f"[{label}] ① 해시 arr={h} v5={truth['event_hash']}"
    # ② 비용 13항 · ③ KPI · ④ 결정 수
    cr = truth["cost_raw"]
    bad = [(t, d["cost_episode"][t], cr[t]) for t in COST_TERMS if d["cost_episode"][t] != cr[t]]
    assert not bad, f"[{label}] ② 비용 항목이 갈린다 (항, arr, v5): {bad}"
    got_k = {k: d["kpi"][k] for k in truth["kpis"]}
    assert got_k == truth["kpis"], f"[{label}] ③ KPI arr={got_k} v5={truth['kpis']}"
    n_steps = int(w.steps)
    n_dec = int(np.asarray(trace.decided)[:n_steps].sum())
    assert n_dec == truth["n_decisions"], f"[{label}] ④ 결정 수 arr={n_dec} v5={truth['n_decisions']}"
    assert d["violation"] == 0 and d["overflow"] == 0 and d["terminal"], \
        f"[{label}] violation={d['violation_names']} overflow={d['overflow']} terminal={d['terminal']}"
    # ⑤ 지금 굴린 v5 (같은 망) 와 오더·크레인·격자
    sim, _recs, exc_n, rt, n_dec5 = run_v5_net(prof, scn, w0, tb, g, float(truth["bridge_end_s"]),
                                               D.net_from_npz(npz), record=False)
    assert sim.event_stream_hash() == truth["event_hash"], (
        f"[{label}] v5 재구동이 정답과 다르다 (같은 망을 다시 실었는데 궤적이 갈렸다) "
        f"arr={sim.event_stream_hash()} truth={truth['event_hash']}")
    assert n_dec5 == truth["n_decisions"]
    assert exc_n == 0 and rt.n_select == truth["n_net_selects"], (
        f"[{label}] 망 호출 arr={rt.n_select} v5={truth['n_net_selects']} 예외={exc_n}")
    for jid, j in sim.jobs.items():
        a = d["jobs"][jid]
        got = (a["status"], a["assigned_crane"], a["rehandle_count"], a["service_start"], a["service_end"])
        exp = (j.status.name, j.assigned_crane, j.rehandle_count, j.service_start, j.service_end)
        assert got == exp, f"[{label}] ⑤ 오더 {jid}: arr={got} v5={exp}"
    for cid in sim.fleet.ids():
        yc, a = sim.fleet.get(cid), d["cranes"][cid]
        got = (a["position_bay"], a["trolley_row"], a["served_count"], a["assigned_job"],
               a["loaded_travel_m"], a["empty_travel_m"], a["available_at"], a["yielded"])
        exp = (yc.state.position_bay, yc.state.trolley_row, yc.served_count, yc.state.assigned_job,
               yc.state.loaded_travel_m, yc.state.empty_travel_m, yc.state.available_at, yc.yielded)
        assert got == exp, f"[{label}] ⑤ 크레인 {cid}: arr={got} v5={exp}"
    assert d["piles"] == {k: v for k, v in sim.stacks._stacks.items() if v}, f"[{label}] ⑤ piles"
    assert d["containers"] == {c: (x.bay, x.row, x.tier) for c, x in sim.stacks.containers.items()}
    # + 학습 경로 run_while == run (잎 전부 비트)
    ww = ES.run_while_jit(w0, params, g, pol, s_max, True, True, hz, True)
    names = [str(p) for p, _ in jax.tree_util.tree_leaves_with_path(w)]
    la, lb = jax.tree_util.tree_leaves(w), jax.tree_util.tree_leaves(ww)
    badl = [names[i] for i, (x, y) in enumerate(zip(la, lb))
            if not np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True)]
    assert not badl, f"[{label}] run_while 와 run 이 다른 잎 {badl}"
    kinds: dict[str, int] = {}
    for (_, k, _) in d["event_log"]:
        kinds[k] = kinds.get(k, 0) + 1
    REPORT[label] = dict(net=name, hash=h, events=len(arlog), decisions=n_dec, steps=n_steps,
                         net_calls=rt.n_select, actions=truth["crane_actions"],
                         repo=sum(1 for (_, k, p) in d["event_log"] if k == "DISPATCH" and ":REPO:" in p),
                         rehandles=truth["kpis"]["rehandle_count"], kinds=kinds)


# ──────────────────────────── 터미널 21블록 학습 정책망 — v5 재구동 재현 가드 (배열 대조 아님)
#: 터미널 정답 궤적 두 벌 — 고정 시드 무작위 망과 **실제 학습 체크포인트**
TERMINAL_NETS = [("seed20260927h64", "무작위 기본 시드"),
                 ("ckpt-policy", "실제 학습 체크포인트 yr302-final-train")]


@pytest.mark.parametrize("tag,name", TERMINAL_NETS, ids=[t for t, _ in TERMINAL_NETS])
def test_terminal_net_truth_is_reproducible(tag, name):
    """터미널 정답 궤적(`--mode terminal --policy v5net`)이 **저장된 가중치로 다시 굴려도 같은가**.

    ★이것은 **배열 동등성 대조가 아니다.** 배열 쪽 터미널 학습 정책망은 조각 8 몫이다 — 블록 요약 8칸
      중 넷(블록 안·오는 중·곧 올 통지·줄 선 대수)이 **터미널 기록**(`ExecutionRecord`, `MarketBridge._sync`
      가 `값 ≤ t` 로 걸러 찍는다)에서 오는데, 배열 쪽에는 ① 블록축으로 쌓은 `reserve_s`·`end_s` 와
      ② 트럭이 승인될 때 그 칸을 채우는 자리가 아직 없다 (open_issues). 블록 Y01 단독 무대는 외부트럭이
      0 이라 그 넷이 전부 0 이어서, 이 네 칸은 **터미널 무대에서만** 밟힌다.
    여기서 못박는 것은 정답의 **재현성**이다 — `.npz` 로 실은 같은 망이 같은 궤적을 내야 조각 8 이 쓴다.
    그리고 그 네 칸이 실제로 0 이 아니었는지 세어, 이 무대가 블록 무대가 못 덮는 곳을 덮는다는 것을 보인다.
    """
    p = os.path.join(_TRUTH_DIR, D.terminal_truth_name(30, 9900777, D.NET, PA, tag))
    if not os.path.exists(p):
        pytest.skip(f"터미널 학습 정책망 정답 궤적 없음 ({tag}) — "
                    f"dump_ground_truth.py --mode terminal --policy v5net")
    truth = json.load(open(p, encoding="utf-8"))
    npz = os.path.join(_TRUTH_DIR, truth["net_weights"])
    # 블록 요약의 **기록 기반 네 칸**이 0 이 아닌 횟수를 센다 (칸 번호는 features/block.py:147-156 순서)
    orig = D.NetRuntime.block_state
    seen = dict(inside=0, pipeline=0, announced_soon=0, waiting=0, calls=0)

    def hooked(self, bid, t):
        v = orig(self, bid, t)
        seen["calls"] += 1
        for nm, i in (("inside", 0), ("pipeline", 1), ("announced_soon", 5), ("waiting", 7)):
            if v[i] > 0:
                seen[nm] += 1
        return v

    D.NetRuntime.block_state = hooked
    try:
        got = D.run_terminal(truth["load"], truth["seed"], D.NET, net=D.net_from_npz(npz))
    finally:
        D.NetRuntime.block_state = orig
    for b in sorted(truth["blocks"]):
        a, e = got["blocks"][b], truth["blocks"][b]
        for key in ("event_hash", "n_events", "kpis", "unfinished", "clock_s", "cost_raw", "jobs", "ledger"):
            assert a[key] == e[key], f"터미널 {b}: {key} 가 정답과 다르다 (같은 망을 다시 실었다)"
    for key in ("n_net_selects", "crane_actions", "policy_exceptions", "admitted", "turn_sum_s",
                "n_turns", "locked", "ann_ledger", "totals", "route_cost_s"):
        assert got[key] == truth[key], f"터미널 {key} 가 정답과 다르다 arr={got[key]} v5={truth[key]}"
    assert truth["policy_exceptions"] == 0
    for nm in ("inside", "announced_soon", "waiting"):
        assert seen[nm] > 0, (f"블록 요약의 기록 기반 칸 {nm} 이 한 번도 0 이 아니지 않았다 — 이 무대도 "
                              f"그 칸을 안 밟았다는 뜻이라 조각 8 대조 대상이 못 된다 {seen}")
    REPORT[f"(d) 터미널-{tag}"] = dict(net=name, blocks=len(truth["blocks"]),
                                       events=sum(v["n_events"] for v in truth["blocks"].values()),
                                       unfinished=sum(v["unfinished"] for v in truth["blocks"].values()),
                                       net_calls=truth["n_net_selects"], actions=truth["crane_actions"],
                                       block_row_nonzero={k: v for k, v in seen.items() if k != "calls"},
                                       block_row_calls=seen["calls"])


# ─────────────────── (e) ★외부트럭이 있는 **하루** — 배열 엔진이 스스로 채운 열까지 v5 와 대조
#: ★왜 이 무대가 필요한가 (2026-09-26 검증 반박)
#:   (b) 의 블록 Y01 은 작업 240건이 **전부 본선**이고 외부트럭이 0 이다. 그래서
#:     · 블록 요약 8칸 중 기록 기반 넷(0 블록 안 · 1 오는 중 · 5 곧 올 통지 · 7 줄 선 대수) 이 전 행에서 0
#:     · 후보 칸 13(외부트럭) · 15(누적대기) 도 0, 필수(mandatory) 후보 0건
#:     · PRE_REHANDLE 은 후보 자체가 없다 (`candidates.iter_pre_rehandle_jobs` 가 flow==GATE_OUT 을 요구)
#:     · 교착 탈출(DEADLOCK_ESCAPE) 도 3벌 모두 0건
#:   즉 "학습 경로 전체 동등성" 의 절반이 **빈 채로** 통과했다. (a) 무대들은 그 칸을 밟지만 v5 시간장부를
#:   배열 세계로 옮겨 심고(`_with_gate_out`) `reserve_s` 도 v5 작업 목록으로 만들어 넣는 재생 대조라,
#:   **배열 원장이 스스로 채운 값**은 어디서도 v5 와 대조되지 않았다.
#:   여기서는 배열 엔진이 하루를 **혼자 끝까지** 굴리며 그 열들을 자기 힘으로 채운다.
def truck_day_scenario():
    """트럭 22대 + 본선 1척 하루 (블록 하나 · 크레인 2 · 지평 14,400 + 배수 1,800).

    · 행 1 bay 1~10 tier 1 = 반출 대상, bay 4~9 는 위에 blocker 2단 → 재조작·**선재조작**(PRE_REHANDLE)
    · 행 2 bay 1~6 tier 1 = blocker 없는 반출 대상 (계획이 가벼운 쪽)
    · 행 4 bay 1~6 tier 1 = 본선 적하 대상 6 (본선 여유 칸이 움직인다)
    · ① 몰려오는 반출 8 (게이트 0 · 블록도착 300) → 줄이 생기고 뒤쪽은 누적대기 1,440 초를 넘겨 **필수** 후보
    · ② 흩어진 반출 8 (게이트 t · 블록도착 t+600) → '오는 중'·'곧 올 통지분' 칸이 살아난다
    · ③ 반입 6 (같은 간격) → 적재 계획·자리 찾기
    돌려주는 값 `(시나리오, {작업 → 통지된 도착예정})` — 그 사전이 `Order.in_out_reserve_s` 의 출처다
    (**통지된 예정**이지 실현 게이트인이 아니다 — 정보 경계).
    """
    _c, _out, _in_, _vl = _FEAT._c, _FEAT._out, _FEAT._in_, _FEAT._vl
    containers, jobs, reserve = {}, [], {}
    for b in range(1, 11):
        containers[f"T{b:02d}"] = _c(f"T{b:02d}", b, 1, 1)
        if 4 <= b <= 9:                                    # blocker 2단 → 재조작 · 선재조작
            containers[f"T{b:02d}B2"] = _c(f"T{b:02d}B2", b, 1, 2)
            containers[f"T{b:02d}B3"] = _c(f"T{b:02d}B3", b, 1, 3)
    for b in range(1, 7):
        containers[f"U{b:02d}"] = _c(f"U{b:02d}", b, 2, 1)
        containers[f"V{b}"] = _c(f"V{b}", b, 4, 1)          # 본선 적하 대상
    for i, b in enumerate(range(1, 9)):                     # ① 몰려오는 반출 8
        jid = f"J-OUT-A{i}"
        jobs.append(_out(jid, f"T{b:02d}", 0.0, 300.0, 300.0, exit_s=300.0))
        reserve[jid] = 300.0
    tail = [f"T{b:02d}" for b in (9, 10)] + [f"U{b:02d}" for b in range(1, 7)]
    for i, tgt in enumerate(tail):                          # ② 흩어진 반출 8
        gi = 1800.0 + 900.0 * i
        jid = f"J-OUT-B{i}"
        jobs.append(_out(jid, tgt, gi, gi + 600.0, gi + 600.0, exit_s=300.0))
        reserve[jid] = gi + 600.0
    for i in range(6):                                      # ③ 반입 6
        gi = 900.0 + 1200.0 * i
        jid = f"J-IN-{i}"
        jobs.append(_in_(jid, gi, gi + 600.0, gi + 600.0, _FEAT.FT40, exit_s=300.0))
        reserve[jid] = gi + 600.0
    for m in range(6):                                      # ④ 본선 적하 6 moves
        jobs.append(_vl(f"J-VL-{m}", f"V{m + 1}", 600.0 + 400.0 * m, 9000.0, "V-LOAD"))
    ves = [_FEAT.VesselProcess("V-LOAD", _FEAT.VesselWorkType.LOAD,
                               _FEAT.VesselPlan(planned_start_s=600.0, planned_completion_s=9000.0,
                                                completion_basis=_FEAT.CompletionBasis.PLAN_COMPUTED,
                                                etd_s=11000.0, total_moves=6, sts_move_interval_s=300.0,
                                                quay_buffer_cap=3))]
    scn = _FEAT.TerminalScenario(scenario_id="net-day-trucks", seed=0, horizon_s=14400.0,
                                 drain_window_s=1800.0, containers=containers, jobs=jobs,
                                 vessels=ves, injected_events=[])
    return scn, reserve


#: 하루 대조에 쓰는 망 셋 — 101·1234 는 PRE_REHANDLE 을 실제로 고르고, 20260927 은 SERVE/WAIT 만 고른다
DAY_NETS = [101, 1234, 20260927]
_DAY_JIT: dict = {}


def _day_step(g: Geom, policy_fn, hz: float):
    """`engine_step.step` 한 번을 jit — 파이썬 루프로 불러 **결정마다** 세계를 꺼낸다.

    결정이 열리는 스텝에서는 세계가 그 스텝 **시작 상태 그대로**다 (`step` 의 `try_decide` 가
    `~due_now & ~woke` 를 요구하므로 사건 처리도 wake 도 없었고, 결정은 시계를 앞으로 돌리지 않는다).
    그래서 '스텝 전 세계' 가 곧 'v5 가 `block_features` 를 계산한 그 순간의 세계' 다.
    """
    key = ("step", g, policy_fn, hz)
    if key not in _DAY_JIT:
        _DAY_JIT[key] = jax.jit(lambda w, p: ES.step(w, None, params=p, g=g, policy_fn=policy_fn,
                                                     check=True, pre_advice=True, horizon_s=hz,
                                                     joint=True))
    return _DAY_JIT[key]


def _day_block_row(g: Geom, crane_order):
    key = ("blk", g, crane_order)
    if key not in _DAY_JIT:
        _DAY_JIT[key] = jax.jit(lambda w, e, r: VF.block_row(w, g, end_s=e, reserve_s=r,
                                                             crane_order=crane_order))
    return _DAY_JIT[key]


@pytest.mark.parametrize("net_seed", DAY_NETS)
def test_net_day_array_matches_v5(net_seed):
    """★외부트럭이 도는 하루 — 배열 엔진이 v5 와 사건 하나까지 같고, **원장 세 칸**·**결정마다 블록 요약**도 같다."""
    prof = _FEAT.prof_k2(2.0)
    scn, reserve = truck_day_scenario()
    end_s = float(scn.horizon_s) + float(scn.drain_window_s)
    caps, s_max = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    hz = float(prof.decision_horizon_s)
    crane_order = tuple(tb.crane_index[c.crane_id] for c in prof.cranes)
    res = np.full((w0.n,), np.inf)
    for jid, v in reserve.items():
        res[tb.job_index[jid]] = round(float(v), 3)          # stage/orders.py:88 규약
    res_j = jnp.asarray(res)

    pol_torch = _fixed_net(seed=net_seed)
    params = _params_of(pol_torch)(w0, end_s, res_j)
    policy_fn = DP.make_v5net_policy(g, crane_order=crane_order)

    # ── ① 배열 엔진: 결정마다 세워 블록 요약을 그 자리에서 잰다 (jit 한 step + 파이썬 루프)
    stepf, blkf = _day_step(g, policy_fn, hz), _day_block_row(g, crane_order)
    w = w0
    arr_blocks: list[tuple] = []
    for _ in range(s_max):
        if bool(w.terminal):
            break
        w_pre = w
        w, tr = stepf(w, params)
        if bool(tr.decided):
            arr_blocks.append((float(w_pre.clock),
                               np.asarray(blkf(w_pre, params.end_s, params.reserve_s), np.float64)))
    w = ES.finish(w)
    d = from_block_world(w, tb)

    # ── ② v5 를 **그 자리에서** 같은 망으로 굴린다 (기대값 저장 없음 · 하루 0.3초)
    sim, _recs, exc_n, rt, n_dec5 = run_v5_net(prof, scn, w0, tb, g, end_s, pol_torch,
                                               record=False, reserve_of=reserve.get)
    assert exc_n == 0, f"v5 학습 경로 예외 {exc_n} 건 (ppo/crane.py:81 은 구조상 실패가 없다)"

    # ── ③ 사건 로그 전열 · 해시
    arlog = [(round(t, 6), k, p) for (t, k, p) in d["event_log"]]
    v5log = [(round(t, 6), k, p) for (t, k, p) in sim.event_log]
    diff = _first_diff(arlog, v5log)
    if diff is not None:
        i, x, y = diff
        ctx = "\n".join(f"    #{j}: arr={arlog[j] if j < len(arlog) else '-'}  |  "
                        f"v5={v5log[j] if j < len(v5log) else '-'}"
                        for j in range(max(0, i - 4), min(max(len(arlog), len(v5log)), i + 4)))
        pytest.fail(f"[(e) 하루/s{net_seed}] 사건 로그가 {i}번째에서 갈린다: arr={x} v5={y}\n{ctx}\n"
                    f"  (arr {len(arlog)}건 · v5 {len(v5log)}건 · violation={d['violation_names']} "
                    f"overflow={d['overflow']} terminal={d['terminal']})")
    assert event_stream_hash(w, tb) == sim.event_stream_hash()
    assert d["violation"] == 0 and d["overflow"] == 0 and d["terminal"], (
        f"violation={d['violation_names']} overflow={d['overflow']} terminal={d['terminal']}")

    # ── ④ 비용 13항 · KPI · 결정 수
    cr = sim.cost.episode_raw()
    bad = [(t_, d["cost_episode"][t_], cr[t_]) for t_ in COST_TERMS if d["cost_episode"][t_] != cr[t_]]
    assert not bad, f"비용 항목이 갈린다 (항, arr, v5): {bad}"
    k5 = sim.kpis
    want_kpi = {"queue_area_s": k5.queue_area_s, "tail_area_s": k5.tail_area_s,
                "loaded_gantry_m": k5.loaded_gantry_m, "empty_gantry_m": k5.empty_gantry_m,
                "rehandle_count": k5.rehandle_count, "pre_rehandle_count": k5.pre_rehandle_count,
                "completed_external": k5.completed_external, "completed_vessel": k5.completed_vessel,
                "vessel_delay_s": k5.vessel_delay_s, "positioning_count": k5.positioning_count}
    assert {k: d["kpi"][k] for k in want_kpi} == want_kpi, f"KPI arr={d['kpi']} v5={want_kpi}"
    assert len(arr_blocks) == n_dec5, f"결정 수 arr={len(arr_blocks)} v5={n_dec5}"

    # ── ⑤ ★오더별 상태 + **원장 세 칸**(게이트인 A · 블록도착 B · 게이트아웃 O) — (a)(b) 가 못 덮던 열
    tl = getattr(sim, "time_ledger", None)
    assert tl is not None and len(tl.records) == len(reserve), (
        f"v5 시간장부에 트럭이 안 등록됐다 — 이 무대의 목적이 사라진다 ({tl and len(tl.records)})")
    for jid, j in sim.jobs.items():
        a, r = d["jobs"][jid], (tl.records.get(jid) if tl else None)
        got = (a["status"], a["assigned_crane"], a["rehandle_count"], a["service_start"], a["service_end"],
               a["gate_in"], a["block_arrival"], a["actual_gate_out"])
        exp = (j.status.name, j.assigned_crane, j.rehandle_count, j.service_start, j.service_end,
               (r.gate_in if r else None), (r.block_arrival if r else None), (r.gate_out if r else None))
        assert got == exp, (f"오더 {jid}: arr={got} v5={exp} "
                            f"(꼬리 셋이 배열 원장 gate_in_s·block_in_s·gate_out_s 다)")
    for cid in sim.fleet.ids():
        yc, a = sim.fleet.get(cid), d["cranes"][cid]
        got = (a["position_bay"], a["trolley_row"], a["served_count"], a["assigned_job"],
               a["loaded_travel_m"], a["empty_travel_m"], a["available_at"], a["yielded"])
        exp = (yc.state.position_bay, yc.state.trolley_row, yc.served_count, yc.state.assigned_job,
               yc.state.loaded_travel_m, yc.state.empty_travel_m, yc.state.available_at, yc.yielded)
        assert got == exp, f"크레인 {cid}: arr={got} v5={exp}"
    assert d["piles"] == {k: v for k, v in sim.stacks._stacks.items() if v}
    assert d["containers"] == {c: (x.bay, x.row, x.tier) for c, x in sim.stacks.containers.items()}

    # ── ⑥ ★**결정마다** 블록 요약 8칸 — 배열 세계가 자기 원장에서 만든 값 대 v5 `block_features`
    v5_blocks = rt.block_trace
    assert len(v5_blocks) == len(arr_blocks), (
        f"블록 요약 호출 수 arr={len(arr_blocks)} v5={len(v5_blocks)}")
    for i, ((ta, va), (tb_, vb)) in enumerate(zip(arr_blocks, v5_blocks)):
        assert ta == tb_, f"결정 #{i} 시각이 어긋난다 arr={ta} v5={tb_}"
        dif = np.argwhere(va != vb).reshape(-1)
        assert not dif.size, (
            f"결정 #{i} t={ta:.3f}: 블록 요약이 갈린다 — 칸 "
            f"{[(int(c), VF.FEATURE_NAMES[int(c)], float(vb[c]), float(va[c])) for c in dif]}")

    # ── ⑦ 공허 방지 — 이 무대가 Y01 이 못 덮는 것을 실제로 덮었나
    nz = {VF.FEATURE_NAMES[c]: sum(1 for _, v in arr_blocks if v[c] != 0.0) for c in range(8)}
    for nm in ("block_inside_10", "block_pipeline_10", "block_announced_soon_10", "block_waiting_10"):
        assert nz[nm] > 0, f"기록 기반 칸 {nm} 이 한 번도 0 이 아니지 않았다 — 무대가 공허하다 {nz}"
    REPORT[f"(e) 하루-s{net_seed}"] = dict(
        hash=event_stream_hash(w, tb), events=len(arlog), decisions=n_dec5, steps=int(w.steps),
        net_calls=rt.n_select, actions=dict(sorted(rt.crane_actions.items())),
        escapes=sim.deadlock_escape_count, rehandles=k5.rehandle_count,
        trucks=len(tl.records), unfinished=sim.unfinished_backlog(), block_row_nonzero=nz)


def test_net_day_covers_what_y01_cannot():
    """★(e) 하루 무대가 (b) Y01 의 **빈 곳**을 실제로 밟았나 — 안 밟았으면 (e) 를 만든 뜻이 없다.

    Y01 3벌은 PRE_REHANDLE 0 · 교착 탈출 0 · 외부트럭 0 이었다 (정답 JSON 의 `crane_actions`·
    `deadlock_escapes`·`jobs`). 여기서 그 셋이 0 이 아니어야 한다.
    """
    days = {k: r for k, r in REPORT.items() if k.startswith("(e)")}
    assert days, "앞 시험 (e) 가 하나도 안 돌았다"
    assert all(r["trucks"] > 0 for r in days.values()), f"외부트럭이 없다 {days}"
    assert sum(r["actions"].get("PRE_REHANDLE", 0) for r in days.values()) > 0, (
        f"PRE_REHANDLE 를 한 번도 안 골랐다 — Y01 의 빈 곳이 그대로다 {days}")
    assert sum(r["escapes"] for r in days.values()) > 0, (
        f"교착 탈출이 한 번도 안 났다 — 망 경로의 탈출 결정이 여전히 미검증이다 {days}")
    y01 = {k: r for k, r in REPORT.items() if k.startswith("(b)")}
    if y01:      # 정답 궤적이 있을 때만 (그쪽이 0 이라는 사실을 못박아 둔다)
        assert all(r["actions"].get("PRE_REHANDLE", 0) == 0 for r in y01.values()), (
            "Y01 이 PRE_REHANDLE 를 골랐다 — (e) 무대의 근거 설명을 고쳐야 한다")


# ─────────────────── ★vmap — 세계를 쌓아도 각자 자기 답을 내는가 (조각 8 이 전부 vmap 아래서 돈다)
def test_vmap_stacked_worlds_keep_their_own_answers():
    """★가중치가 **서로 다른** 망 세 벌을 B=3 으로 쌓아 `vmap(run)`·`vmap(run_while)` 을 돌린다.

    ★왜 (2026-09-26 검증 반박): 조각 8(다중블록 조정자·배치 학습)은 전부 vmap 아래서 도는데 이 파일에
      vmap 대조가 **한 건도 없었다**. 게다가 배치 행렬곱은 낱개와 critic 값이 1 ulp 갈린다(v5net.py 머리말) —
      1·2위 격차가 좁아지면 vmap 판이 정답 궤적과 다른 결정을 낼 수 있는데 그것을 잡을 가드가 없었다.
    무엇을 보나: 세계마다 (ㄱ) 낱개 `run_jit` 과 **잎 전부 비트 동일** (ㄴ) 자기 정답 해시 (ㄷ) violation 0.
      세 궤적의 길이가 다르므로(래기드) `run_while` 의 배치 술어 마스킹까지 한 번에 덮는다.
    """
    have = [(tag, t, npz) for tag, t, npz in ((tag, *_truth(tag)) for tag, _ in NETS) if t is not None]
    if len(have) < 2:
        pytest.skip("Y01 학습 정책망 정답 궤적이 2벌 미만 — dump_ground_truth.py --policy v5net")
    t0 = have[0][1]
    prof, built = D._build(t0["load"], t0["seed"])
    scn = built["scenarios"][t0["block"]]
    caps, s_max = _y01_caps(scn)
    w0, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    hz = float(prof.decision_horizon_s)
    crane_order = tuple(tb.crane_index[c.crane_id] for c in prof.cranes)
    pol = DP.make_v5net_policy(g, crane_order=crane_order)
    plist = [DP.v5net_params(D.load_net_npz(npz), w0, end_s=t["bridge_end_s"]) for _, t, npz in have]
    B = len(plist)
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *plist)
    wb = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (B,) + jnp.shape(x)), w0)

    run_v = jax.jit(jax.vmap(lambda w, p: ES.run(w, p, g, pol, s_max, True, True, hz, True)[0],
                             in_axes=(0, 0)))
    while_v = jax.jit(jax.vmap(lambda w, p: ES.run_while(w, p, g, pol, s_max, True, True, hz, True),
                               in_axes=(0, 0)))
    got_run, got_while = run_v(wb, stacked), while_v(wb, stacked)
    names = [str(q) for q, _ in jax.tree_util.tree_leaves_with_path(w0)]
    for b, (tag, t, _npz) in enumerate(have):
        one, _tr = ES.run_jit(w0, plist[b], g, pol, s_max, True, True, hz, True)
        exp = jax.tree_util.tree_leaves(one)
        for nm, batched in (("run(scan)", got_run), ("run_while", got_while)):
            leaves = jax.tree_util.tree_leaves(batched)
            bad = [names[i] for i, (x, y) in enumerate(zip(exp, leaves))
                   if not np.array_equal(np.asarray(x), np.asarray(y)[b], equal_nan=True)]
            assert not bad, f"[{tag}] vmap {nm} 이 낱개와 다른 잎 {bad}"
        w_b = jax.tree_util.tree_map(lambda x: x[b], got_while)
        assert event_stream_hash(w_b, tb) == t["event_hash"], (
            f"[{tag}] vmap 세계 {b} 의 해시 arr={event_stream_hash(w_b, tb)} v5={t['event_hash']}")
        assert int(w_b.violation) == 0
    REPORT["(f) vmap"] = dict(B=B, tags=[tag for tag, _, _ in have],
                              hashes=[t["event_hash"] for _, t, _ in have])


# ─────────────────── ★guard=True — "한 크레인 실패 = 전원 WAIT" 대체를 인위적으로 밟는다
def test_guard_all_wait_substitutes_when_decision_would_raise():
    """`v5cond.guard_all_wait` 의 대체 거동 — **발동 사례가 0 건**이라 합성으로 밟는다.

    ★정직한 문장 (2026-09-26 검증 반박): 이 규칙은 실제 무대에서 **한 번도 발동하지 않았다**
      (무대 8종·Y01 3벌·터미널 2벌 모두 v5 예외 0). 그리고 v5 학습 드라이버는 `stage/month_run.py:541-544`
      에서 `exec_policy = ppo.execute` 로 갈아끼우므로 `episode.py:212-216` 의 try/except 를 **거치지 않는다**
      — 그래서 `make_v5net_policy` 의 기본 `guard=False` 가 옳다. 여기서는 그 코드가 **죽어 있지 않다**는
      것만 못박는다: 실패를 내는 답을 감싸면 전 크레인 WAIT 열로 바뀌고 `lost` 가 꺼진다.
    """
    prof = _FEAT.prof_k2(2.0)
    scn, reserve = truck_day_scenario()
    caps, s_max = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    hz = float(prof.decision_horizon_s)
    crane_order = tuple(tb.crane_index[c.crane_id] for c in prof.cranes)
    res = np.full((w0.n,), np.inf)
    for jid, v in reserve.items():
        res[tb.job_index[jid]] = round(float(v), 3)
    pol_torch = _fixed_net(seed=101)
    params = _params_of(pol_torch)(w0, float(scn.horizon_s) + float(scn.drain_window_s), jnp.asarray(res))
    policy_fn = DP.make_v5net_policy(g, crane_order=crane_order)
    stepf, fn = _day_step(g, policy_fn, hz), _cands_jit(g, True)
    # ★두 크레인이 **같은 오더**를 고를 수 있는 실제 결정 시점을 찾는다 (t=0 에는 트럭이 아직 없다)
    w, found = w0, None
    for _ in range(s_max):
        if bool(w.terminal) or found is not None:
            break
        w_pre = w
        w, tr = stepf(w, params)
        if not bool(tr.decided):
            continue
        c3, fl, pr = fn(w_pre, hz)
        fln, prn = _np(fl), _np(pr)
        per_job: dict[int, dict] = {}
        for k in range(w_pre.k):
            for c in range(fln.kind.shape[1]):
                if bool(prn.keep[k, c]) and int(fln.kind[k, c]) == PK_SERVE and bool(fln.feasible[k, c]):
                    per_job.setdefault(int(fln.job[k, c]), {})[k] = c
        shared = next((cols for cols in per_job.values() if len(cols) >= 2), None)
        if shared is None:
            continue
        clash = jnp.asarray([shared.get(k, -1) for k in range(w_pre.k)], jnp.int32)
        if bool(VC.decision_would_raise(w_pre, fl, pr, clash, jnp.ones((w_pre.k,), bool), g)):
            found = (w_pre, c3, fl, pr, clash)
    assert found is not None, "두 크레인이 같은 오더를 고르는 결정을 못 찾았다 — 무대가 바뀌었다"
    wd, c3, fl, pr, clash = found
    open_ = jnp.ones((wd.k,), bool)
    wait_c = VC.wait_col(fl)

    def clash_policy(params_, world, c3_, fl_, pr_, open__):
        return clash, jnp.ones((world.k,), bool), jnp.zeros((), jnp.int32)

    ch, lost, _flags = VC.guard_all_wait(clash_policy, g)(None, wd, c3, fl, pr, open_)
    assert np.array_equal(np.asarray(ch), np.full((wd.k,), wait_c, np.int32)), \
        f"대체가 전 크레인 WAIT 열이 아니다 {np.asarray(ch)} (WAIT 열 {wait_c})"
    assert not np.asarray(lost).any(), "대체 경로는 yield_reason 을 안 넘기므로 lost 가 전부 거짓이어야 한다"

    # ② 실패가 없는 답은 **그대로 둔다** (대체가 무조건 발동하지 않는다)
    def wait_policy(params_, world, c3_, fl_, pr_, open__):
        return VC.all_wait_choice(fl_, open__), jnp.zeros((world.k,), bool), jnp.zeros((), jnp.int32)

    ch2, _l2, _f2 = VC.guard_all_wait(wait_policy, g)(None, wd, c3, fl, pr, open_)
    assert np.array_equal(np.asarray(ch2), np.asarray(VC.all_wait_choice(fl, open_)))
    assert DP.make_v5net_policy(g, guard=True) is not DP.make_v5net_policy(g)


def test_nonfinite_features_raise_a_violation_bit():
    """★비유한 특징(NaN·inf)이 섞이면 **조용히** 이상한 결정을 내지 않는가 (`V_NET_NONFINITE`).

    v5 `ppo/model.encode`(17-18행)는 그때 ValueError 를 던지고 `ppo/crane.py:81` 이 그 예외를 일부러
    전파한다. jit 안에서는 던질 수 없으므로 배열판은 위반 비트로 알린다 — 전에는 `all_finite` 가 통합
    경로에 **연결돼 있지 않아** NaN 이 그대로 망에 들어갔다 (2026-09-26 검증 반박).
    실현 가능성은 낮지만(안 실린 행은 0 으로 지우고 vessel_slack 은 [−2,2] 로 자른다) 조각 8 이 특징을
    늘리거나 학습 중 값이 발산하면 첫 증상이 '조용히 이상한 결정' 이 된다.
    """
    prof, scn = STAGES["feat-basic"]()
    caps, _ = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    hz = float(prof.decision_horizon_s)
    c3, fl, pr = _cands_jit(g, True)(w0, hz)
    open_ = jnp.ones((w0.k,), bool)
    sd = {k: v.detach().cpu().numpy() for k, v in _fixed_net(101).state_dict().items()}
    pol = DP.make_v5net_policy(g)
    # 성한 입력 — 비트가 꺼져 있다
    ok_p = DP.v5net_params(sd, w0, end_s=9000.0, reserve_s=jnp.full((w0.n,), jnp.inf))
    _ch, _lo, flags = pol(ok_p, w0, c3, fl, pr, open_)
    assert int(flags) & V_NET_NONFINITE == 0, "성한 특징인데 비유한 비트가 켜졌다"
    # ★NaN 을 특징에 실어 넣는다 — `reserve_s` 는 블록 요약 칸 5 로 바로 들어간다
    nan_res = jnp.full((w0.n,), jnp.inf).at[0].set(jnp.nan)
    bad_p = ok_p._replace(reserve_s=nan_res)
    _ch2, _lo2, flags2 = pol(bad_p, w0, c3, fl, pr, open_)
    if int(flags2) & V_NET_NONFINITE == 0:
        # reserve_s 의 NaN 은 비교식에서 걸러질 수 있다 — 그때는 end_s 로 NaN 을 넣어 칸 6 을 오염시킨다
        _ch3, _lo3, flags2 = pol(ok_p._replace(end_s=jnp.asarray(jnp.nan)), w0, c3, fl, pr, open_)
    assert int(flags2) & V_NET_NONFINITE, (
        f"비유한 특징인데 위반 비트가 안 켜졌다 flags={int(flags2)} — v5 는 여기서 예외를 던진다")


# ───────────────────────────────────────────────── 규약·회귀 가드
def test_policy_fn_signature_is_the_joint_contract():
    """`make_v5net_policy` 가 엔진 공동 규약과 같은 서명·같은 함수 객체(lru_cache)인가."""
    g = Geom.from_profile(_FEAT.prof_k2(2.0))
    a = DP.make_v5net_policy(g)
    b = DP.make_v5net_policy(g)
    assert a is b, "같은 인자인데 다른 함수 객체 — jit static 키가 매번 달라져 전체 재추적이 난다"
    assert DP.make_v5net_policy(g, guard=True) is not a
    import inspect
    assert list(inspect.signature(a).parameters) == ["params", "world", "c3", "fl", "pr", "open_"]


def test_rule_policies_still_work_after_integration():
    """조각 1~6 의 규칙 경로(SF_SPT resolver)가 그대로 남아 있는가 — 정답 궤적 Y01 원본 해시."""
    p = os.path.join(_TRUTH_DIR, D.block_truth_name("Y01", 30, 9900777, "sf_spt", PA))
    if not os.path.exists(p):
        pytest.skip("원본 정답 궤적 없음")
    truth = json.load(open(p, encoding="utf-8"))
    prof, built = D._build(truth["load"], truth["seed"])
    scn = built["scenarios"][truth["block"]]
    caps, s_max = _y01_caps(scn)
    w0, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    pol = DP.make_resolver("sf_spt", g, count_lost=False)
    w, _tr = ES.run_jit(w0, DP.resolver_params(tb, g), g, pol, s_max, True, True,
                        float(prof.decision_horizon_s), True)
    assert event_stream_hash(w, tb) == truth["event_hash"] == "6668fa4902c4efe4"
    assert int(w.violation) == 0


def test_v5net_params_rejects_wrong_shapes():
    """터미널 층 입력을 조용히 추측하지 않는다 — 모양이 다르면 크게 실패한다."""
    prof, scn = STAGES[list(STAGES)[0]]()
    caps, _ = _caps(scn, prof)
    w0, _tb = to_block_world(prof, scn, **caps)
    sd = {k: v.detach().cpu().numpy() for k, v in _fixed_net(1).state_dict().items()}
    with pytest.raises(ValueError):
        DP.v5net_params(sd, w0, end_s=1.0, reserve_s=jnp.zeros((w0.n + 1,)))
    with pytest.raises(KeyError):
        DP.v5net_params({"nope": np.zeros((2, 2))}, w0, end_s=1.0)
    p = DP.v5net_params(sd, w0, end_s=9000.0)
    assert p.reserve_s.shape == (w0.n,) and bool(jnp.all(jnp.isinf(p.reserve_s)))
    assert p.net.input_dim == 37 and p.net.hidden == 64


def test_zz_report(capsys):
    """마지막 — 무대별 대조 규모·|Δ|·1·2위 격차·행동 분포. ★격차에 **단언**을 붙인다.

    ★2026-09-26 검증 반박: 전에는 1·2위 최소 격차를 **찍기만** 했다. 그러면 여유가 줄어드는 회귀
      (특징이 뭉치게 바뀌거나 |Δ| 가 커지는 변경)를 아무도 못 잡는다. 이제 `최소격차 ≥ 20 × |Δ|` 를
      요구한다 — 이 층의 동등성이 '증명' 이 아니라 '여유' 에 기대고 있음을 시험이 직접 지킨다.
    """
    assert REPORT, "앞 시험이 하나도 안 돌았다"
    with capsys.disabled():
        print("\n[학습 정책망 동등성 보고]")
        for k, r in sorted(REPORT.items()):
            if k.startswith("(a)") and "worst" in r:
                w = r["worst"]
                print(f"  {k:28s} 결정 {r['decisions']:3d}(탈출건너뜀 {r['escape_skipped']}) "
                      f"크레인결정 {r['crane_decisions']:3d} 행 {r['rows']:4d}  "
                      f"|Δ| 점수 {w['score']:.2e} 가치 {w['value']:.2e} 로그확률 {w['logp']:.2e} "
                      f"확률 {w['prob']:.2e}  1·2위 최소격차 {r['min_gap']:.2e}")
                print(f"  {'':28s} 고른 것 {r['chosen']} · 직전칸 {r['priors']} · 마스크 off {r['masked_off']}")
            elif k.startswith("(b)"):
                print(f"  {k:28s} {r['net']:32s} hash={r['hash']} 사건 {r['events']} 결정 {r['decisions']} "
                      f"망호출 {r['net_calls']} REPO {r['repo']} 재조작 {r['rehandles']} 행동 {r['actions']}")
            else:
                print(f"  {k:28s} {r}")
        gaps = [r["min_gap"] for k, r in REPORT.items() if "min_gap" in r]
        worst = max((r["worst"]["score"] for r in REPORT.values() if "worst" in r), default=0.0)
        if gaps:
            print(f"  ★결정이 갈릴 여지: 1·2위 최소격차 {min(gaps):.3e} vs 점수 |Δ| 최댓값 {worst:.3e} "
                  f"→ {min(gaps) / max(worst, 1e-300):.1f} 배 여유 (요구 {GAP_MARGIN:.0f} 배 이상)")
            print("  ⚠️ 이 여유는 **이 표본의 성질**이다 — 층 자체의 갈림 확률은 4.7e-06/결정이고 "
                  "(망 2,600벌 × 85.28만 결정에서 뒤집힘 4건) 최악 여유는 0.17배까지 내려간다. "
                  "scripts/v6/probe_net_flip_rate.py --nets 2600 으로 다시 잴 수 있다.")
    # ★단언 — 찍기만 하면 여유가 줄어드는 회귀를 못 잡는다
    assert gaps, "1·2위 격차를 잰 시험이 하나도 안 돌았다 (a) 무대가 전부 건너뛰어졌다"
    assert min(gaps) >= GAP_MARGIN * worst, (
        f"결정이 갈릴 여지가 좁아졌다 — 1·2위 최소격차 {min(gaps):.3e} < {GAP_MARGIN:.0f} × 점수 |Δ| "
        f"{worst:.3e} = {GAP_MARGIN * worst:.3e}. 이 층의 동등성은 '증명' 이 아니라 '여유' 에 기대므로, "
        f"여유가 줄면 조각 8 의 한 달 대조가 해시 불일치로 끝난다 (v5net.py 머리말 ★★).")
