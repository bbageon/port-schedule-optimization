"""YR-147 2단계 — A/B/C 학습·계측 (사전등록 동결: spec YR-147 §2단계 구현 계약).

A = 현재 무기한 WAIT (YR-145 B2 체크포인트 재사용 — 학습 없음)
B = DEFER_ALL (후보 삭제 없이 전 대기 유한화, 만료 now+600s)
C = DEFER_TRIGGER (관측 trigger 있을 때만 전략적 DEFER — 부재 시 구조 fallback 전용)
유일 변경 = 대기 행동 의미. 결속+one-shot(B2 계약)·보상·상태·PPO 불변.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from statistics import fmean

import torch
from torch import nn
from torch.distributions import Categorical

from ..integrated import candidates as cand_mod
from ..integrated.baselines import JointRolloutGreedy, _apply, _wait_of
from . import yr139_blockq_v4_ppo as y139
from ..integrated.candidates import CandidateGenerator
from ..integrated.encoding import StateNorm
from ..integrated.joint_distill import JointPairNet
from . import yr088_joint_rl as y88
from .yr090_dense_vessel import BASE, CELLS
from .yr100_candidate_eval import RC_EVAL
from .yr136_softplus_contract import _sim_contract
from .yr138_episode_pilot import _v2_hard_total
from .yr139_blockq_v4_ppo import train_one
from .yr145_prepo_status_gate import OUT as OUT145
from .yr147_wait_baseline import PROG, _record_prepo

OUT = Path("outputs/reports/yr147_defer")
ARM_WAIT_MODE = {"a": "WAIT", "b": "DEFER_ALL", "c": "DEFER_TRIGGER"}
ARM_ROOT = {"a": OUT145 / "b2", "b": OUT / "b", "c": OUT / "c"}
DEV_EPS = [(cell, BASE[cell] + i) for cell in CELLS for i in range(16)]
PAIR_CELL_QUOTA, PAIR_EP_CAP, TOPK = 8, 2, 2


def train(ts: int, arm: str):
    prev = cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT, cand_mod.WAIT_MODE
    cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT = True, True     # B2 계약 유지
    cand_mod.WAIT_MODE = ARM_WAIT_MODE[arm]
    try:
        return train_one(ts, out_root=OUT / arm)
    finally:
        (cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT, cand_mod.WAIT_MODE) = prev


# ------------------------------------------------------------------ 3단계 라벨 (22차 계약)
def select_progress_combos(assigns, prog_idx, rng, k_max=4):
    """정책독립 후보선정 (22차 — PPO 점수 top-K 금지): 전부 ≤k_max 면 전부.
    초과 시 ①총 계획시간 최소(SF-SPT 근사·결정론) ②행동유형 서명 층화 ③무작위 잔여."""
    if len(prog_idx) <= k_max:
        return list(prog_idx)
    def plan_dur(i):
        a = assigns[i]
        return sum((c.plan.duration_s if getattr(c, "plan", None) is not None
                    else 0.0) for c in a.values())
    chosen = [min(prog_idx, key=plan_dur)]
    sigs = {tuple(sorted(assigns[chosen[0]][c].kind.name for c in assigns[chosen[0]]))}
    for i in sorted(prog_idx):
        if len(chosen) >= k_max:
            break
        sig = tuple(sorted(assigns[i][c].kind.name for c in assigns[i]))
        if sig not in sigs and i not in chosen:
            chosen.append(i); sigs.add(sig)
    rest = [i for i in prog_idx if i not in chosen]
    while len(chosen) < k_max and rest:
        chosen.append(rest.pop(rng.randrange(len(rest))))
    return chosen


def lex_label(sim, dp, assigns, idx_list, policy):
    """공동행동별 (미완 비율, backlog, v2 비용) — 사전식 비교용 (완주→backlog→비용 우선,
    22차 ④: 비용 최솟값 선행 금지). 반사실 미래는 라벨 전용 — 정책 관측 진입 금지."""
    out = {}
    for i in idx_list:
        s2 = copy.deepcopy(sim)
        _record_prepo(s2, dp, assigns[i])
        _apply(s2, assigns[i])
        r = _continue_to_end(s2, policy, CandidateGenerator())
        out[i] = (round(1.0 - r["compl"], 6), r["backlog"], r["phi"])
    return out


# ------------------------------------------------------------------ 3단계 학습 (B vs R)
# 23차 계약 (spec §3단계 실행 전 계약, 동결): 유일 차이 = 보조 순위손실.
TRAIN_OFFSET, SPC3 = 16, 16      # 신규 훈련 대역 BASE+16..31 (B·R 동일 — 22차 "같은 신규 대역")
EPS_COST = 0.1                   # 라벨 동점폭 (v2 비용 단위 — 신경망 점수에는 미적용)
LAMBDA_RANK = 0.1                # 보조손실 계수 (상태별 평균→배치 평균 뒤 적용·튜닝 금지)
LABEL_CAP_ITER = 4               # 반복당 라벨 상태 상한 (계산 예산)
MIN_EXPOSURE = 30                # 런당 최소 라벨 상태 (미달 = "조작 노출 부족" — 효과 없음 아님)


def _rank_rng(ts, it, ep, dec):
    """라벨 후보 전용 난수 스트림 — (초기화, 반복, 에피소드, 결정) 유도, PPO 난수 비소비."""
    h = hashlib.sha256(f"rank:{ts}:{it}:{ep}:{dec}".encode()).digest()
    import random as _random
    return _random.Random(int.from_bytes(h[:8], "big"))


def _lex_pref(lab_ww, lab_prog):
    """사전식 선호 — 'PROG' | 'WAIT' | None(동점: 미완·backlog 동률 ∧ |Δv2| ≤ EPS_COST)."""
    if lab_ww[0] != lab_prog[0]:
        return "PROG" if lab_prog[0] < lab_ww[0] else "WAIT"
    if lab_ww[1] != lab_prog[1]:
        return "PROG" if lab_prog[1] < lab_ww[1] else "WAIT"
    if abs(lab_ww[2] - lab_prog[2]) <= EPS_COST:
        return None
    return "PROG" if lab_prog[2] < lab_ww[2] else "WAIT"


def rank_pair_loss(cost, ww_idx, pairs):
    """margin 없는 pairwise logistic (RankNet) — 상태 내 쌍 평균. cost 는 작을수록 좋음."""
    terms = []
    for i, pref in pairs:
        d = cost[i] - cost[ww_idx]           # 진행 − 전원연기
        terms.append(torch.nn.functional.softplus(d if pref == "PROG" else -d))
    return torch.stack(terms).mean()


def _new_counters():
    return {"labeled": 0, "iter_labels": 0, "rank_updates": 0, "prog_pref": 0,
            "wait_pref": 0, "ties": 0, "exhaustive_states": 0, "by_cell": {}}


def run_episode_rank(actor, critic, norm, cell, seed, rng, *, ts, it, ep,
                     collect, counters):
    """yr139.run_episode 복제 + (R 전용) 라벨 수집 — collect=None 이면 PPO 경로 동일.

    라벨 반사실은 deepcopy 분기에서만 진행 — 본 궤적·PPO 난수·관측에 영향 없음."""
    sim = _sim_contract(cell, seed)
    gen = CandidateGenerator()
    jr = JointRolloutGreedy(y139.RC_TRAIN, horizon_s=1800.0, generator=gen,
                            forbid_strategic_wait=True)
    trans = []
    phi_prev = y139.phi_v2(sim)
    phi0 = phi_prev
    pend = None
    dp = sim.run_until_decision()
    k = 0
    while dp is not None:
        phi_now = y139.phi_v2(sim)
        if pend is not None:
            pend[4] = -(phi_now - phi_prev)
            trans.append(tuple(pend))
            pend = None
        phi_prev = phi_now
        gen_by = {c: gen.generate(sim, c, y88.LEVEL) for c in dp.crane_ids}
        rows, assigns = y88.build_rows(sim, dp, gen_by, norm, jr, k)
        if not assigns:
            _apply(sim, {c: _wait_of(gen_by[c]) for c in dp.crane_ids})
        else:
            x = torch.tensor(rows, dtype=torch.float32)
            with torch.no_grad():
                logits, _ = actor(x)
                logits = -logits
                dist = Categorical(logits=logits)
                act = (int(dist.sample()) if len(assigns) > 1
                       else int(torch.argmax(logits)))
                logp = float(dist.log_prob(torch.tensor(act)))
                val = float(critic(torch.tensor([y139._state_vec(rows[0])],
                                                dtype=torch.float32))[0])
            pend = [rows, act, logp, val, 0.0]
            if (collect is not None and counters["iter_labels"] < LABEL_CAP_ITER
                    and len(dp.crane_ids) >= 2
                    and all(g.kind.name == "WAIT" for g in assigns[act].values())):
                prog = [i for i, a2 in enumerate(assigns)
                        if any(a2[c].kind.name in PROG for c in a2)]
                if prog:
                    rrng = _rank_rng(ts, it, ep, k)
                    sel = select_progress_combos(assigns, prog, rrng, 4)
                    label_policy = y88.RLPolicy(actor, norm, name="label")
                    labs = lex_label(sim, dp, assigns, [act] + sel, label_policy)
                    pairs = [(i, p) for i in sel
                             if (p := _lex_pref(labs[act], labs[i])) is not None]
                    counters["labeled"] += 1
                    counters["iter_labels"] += 1
                    counters["by_cell"][cell] = counters["by_cell"].get(cell, 0) + 1
                    counters["rank_updates"] += len(pairs)
                    counters["prog_pref"] += sum(1 for _, p in pairs if p == "PROG")
                    counters["wait_pref"] += sum(1 for _, p in pairs if p == "WAIT")
                    counters["ties"] += len(sel) - len(pairs)
                    counters["exhaustive_states"] += 1 if len(prog) <= 4 else 0
                    if pairs:
                        collect.append((rows, act, pairs))
            if cand_mod.PREPO_ONE_SHOT:
                for c in dp.crane_ids:
                    ref = getattr(assigns[act][c], "job_ref", None)
                    jid = cand_mod.prepo_bound_jid(getattr(ref, "job_id", "") or "")
                    if jid is not None:
                        if not hasattr(sim, "_prepo_history"):
                            sim._prepo_history = set()
                        sim._prepo_history.add(jid)
            _apply(sim, assigns[act])
        dp = sim.run_until_decision()
        k += 1
    phi_end = y139.phi_v2(sim, sim.end)
    if pend is not None:
        pend[4] = -(phi_end - phi_prev)
        trans.append(tuple(pend))
    elif trans:
        r0, a0, l0, v0, rw = trans[-1]
        trans[-1] = (r0, a0, l0, v0, rw - (phi_end - phi_prev))
    return trans, {"total": phi_end - phi0}


def ppo_update_rank(actor, critic, opt, batch, rng, rank_batch, log):
    """yr139.ppo_update 복제 + 순위 보조손실(라운드로빈 분할·λ=0.1) + 손실/기울기 기록."""
    advs = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    advs = (advs - advs.mean()) / (advs.std() + 1e-6)
    rets = torch.tensor([b[4] for b in batch], dtype=torch.float32)
    idx_all = list(range(len(batch)))
    for _ in range(4):
        rng.shuffle(idx_all)
        n_mb = (len(idx_all) + 63) // 64
        for mi, s0 in enumerate(range(0, len(idx_all), 64)):
            mb = idx_all[s0:s0 + 64]
            loss_pi, loss_v, ent = 0.0, 0.0, 0.0
            for i in mb:
                rows, act, logp_old, _a, _r = batch[i]
                x = torch.tensor(rows, dtype=torch.float32)
                logits, _ = actor(x)
                dist = Categorical(logits=-logits)
                logp = dist.log_prob(torch.tensor(act))
                ratio = torch.exp(logp - logp_old)
                a_i = advs[i]
                loss_pi = loss_pi - torch.min(
                    ratio * a_i, torch.clamp(ratio, 1 - y139.CLIP, 1 + y139.CLIP) * a_i)
                v = critic(torch.tensor([y139._state_vec(rows[0])],
                                        dtype=torch.float32))[0]
                loss_v = loss_v + (v - rets[i]) ** 2
                ent = ent + dist.entropy()
            nmb = len(mb)
            loss = loss_pi / nmb + 0.5 * loss_v / nmb - y139.ENT * ent / nmb
            r_chunk = rank_batch[mi::n_mb] if rank_batch else []
            if r_chunk:
                r_terms = []
                for rows_r, ww_idx, pairs in r_chunk:
                    cost, _ = actor(torch.tensor(rows_r, dtype=torch.float32))
                    r_terms.append(rank_pair_loss(cost, ww_idx, pairs))
                loss_rank = torch.stack(r_terms).mean()
                loss = loss + LAMBDA_RANK * loss_rank
                log["rank_loss"].append(float(loss_rank))
            log["ppo_loss"].append(float(loss_pi / nmb))
            opt.zero_grad(); loss.backward()
            gnorm = nn.utils.clip_grad_norm_(list(actor.parameters())
                                             + list(critic.parameters()), 1.0)
            log["grad_norm"].append(float(gnorm))
            opt.step()


def train_one_v3(ts: int, mode: str, *, out_base: Path | None = None,
                 norm_ts: int | None = None, safety_only: bool = False,
                 bound: bool = True) -> Path:
    """3단계 학습 — mode 'b'(보조손실 없음) | 'r'(보조 순위손실). 그 외 전부 동일.

    YR-143 재사용 파라미터(기본값 = YR-147 동작 불변): out_base(출력 루트),
    norm_ts(정규화 참조 초기화 — norm_refs 는 3초기화 동일 검증 d3082288),
    safety_only(C0 — 능동 위치조정 미발행), bound(결속 발행)."""
    assert mode in ("b", "r")
    out = (out_base or (OUT / f"train_{mode}")) / f"ppo_s{ts}"
    out.mkdir(parents=True, exist_ok=True)
    prev = (cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT, cand_mod.WAIT_MODE,
            cand_mod.SAFETY_ONLY)
    cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT = bound, True
    cand_mod.WAIT_MODE = "DEFER_ALL"
    cand_mod.SAFETY_ONLY = safety_only
    try:
        ck0 = torch.load(Path("outputs/reports/yr125_diff_credit")
                         / f"diff1_s{norm_ts if norm_ts else ts}"
                         / "rl_net.pt", map_location="cpu")
        norm = StateNorm(refs=ck0["norm_refs"])
        torch.manual_seed(ts)
        actor, critic = JointPairNet(250), y139.Critic()
        opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()),
                               lr=y139.LR)
        import random as _random
        rng = _random.Random(ts)
        y88.FORBID_WAIT = True
        cells = list(CELLS)
        curve, log = [], {"ppo_loss": [], "rank_loss": [], "grad_norm": []}
        counters = _new_counters()
        for it in range(1, y139.N_ITER + 1):
            counters["iter_labels"] = 0
            batch, rank_batch, totals = [], [], []
            for e in range(y139.EPS_PER_ITER):
                cell = cells[(it * y139.EPS_PER_ITER + e) % len(cells)]
                seed = BASE[cell] + TRAIN_OFFSET + rng.randrange(SPC3)
                trans, st = run_episode_rank(
                    actor, critic, norm, cell, seed, rng, ts=ts, it=it, ep=e,
                    collect=(rank_batch if mode == "r" else None), counters=counters)
                totals.append(st["total"])
                if trans:
                    adv, ret = y139._gae(trans)
                    for (rows, act, logp, _v, _r), a_, r_ in zip(trans, adv, ret):
                        batch.append((rows, act, logp, a_, r_))
            if not batch:
                continue
            ppo_update_rank(actor, critic, opt, batch, rng, rank_batch, log)
            curve.append({"iter": it, "mean_total": round(fmean(totals), 3)})
            if it % 5 == 0 or it == 1:
                print(f"[{mode} s{ts} it{it}] mean {fmean(totals):.2f} "
                      f"labels {counters['labeled']}", flush=True)
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                    "in_dim": 250, "train_seed": ts}, out / "net.pt")
        (out / "train_meta.json").write_text(json.dumps(
            {"mode": mode, "train_band": f"BASE+{TRAIN_OFFSET}..{TRAIN_OFFSET+SPC3-1}",
             "eps_cost": EPS_COST, "lambda_rank": LAMBDA_RANK,
             "label_cap_iter": LABEL_CAP_ITER, "min_exposure": MIN_EXPOSURE,
             "counters": counters,
             "loss_log_tail": {k: v[-20:] for k, v in log.items()},
             "loss_log_mean": {k: (fmean(v) if v else None) for k, v in log.items()},
             "exposure_ok": counters["labeled"] >= MIN_EXPOSURE if mode == "r" else None,
             "curve": curve}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{mode} s{ts}] 완료 labels={counters['labeled']}", flush=True)
    finally:
        (cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT, cand_mod.WAIT_MODE,
         cand_mod.SAFETY_ONLY) = prev
    return out / "net.pt"


# ------------------------------------------------------------------ 3단계 판정
BAND3_PATH = OUT / "band3.json"
BAND3_START, BAND3_CEIL, BAND3_N = 910_024, 920_000, 8   # 셀별 8 = 32 시나리오 (MDE 동결)


def _collect_hashes_excluding_out() -> set[str]:
    import re
    pat = re.compile(r"rz1:[0-9a-f]{16}")
    got: set[str] = set()
    for p in Path("outputs/reports").rglob("*.json"):
        if OUT in p.parents:
            continue
        try:
            got |= set(pat.findall(p.read_text(encoding="utf-8", errors="ignore")))
        except OSError:
            continue
    return got


def make_band3():
    from ..integrated.seedbank import assign_band, independence_report
    import subprocess
    exclude = _collect_hashes_excluding_out()
    band = assign_band(family="yr147-judgment", cells={c: None for c in CELLS},
                       n=BAND3_N,
                       generate=lambda key, _p, seed: _sim_contract(key, seed).scenario,
                       exclude=exclude, start_seed=BAND3_START)
    for ss in band.seeds.values():
        for s in ss:
            assert BAND3_START <= s < BAND3_CEIL, f"대역 정수 이탈: {s}"
    rep = independence_report(band, forbidden={"past-recorded": exclude})
    assert rep["ok"], rep
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    OUT.mkdir(parents=True, exist_ok=True)
    BAND3_PATH.write_text(json.dumps(
        {**band.freeze_json(), "independence": rep, "n_excluded_hashes": len(exclude),
         "created_commit": head}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[band3] {sum(len(v) for v in band.seeds.values())} seeds frozen")
    return band


def _load_band3():
    from ..integrated.seedbank import BandSpec, independence_report, realization_hash
    d = json.loads(BAND3_PATH.read_text(encoding="utf-8"))
    band = BandSpec(family=d["family"], seeds=d["seeds"], hashes=d["realization_hashes"])
    for cell, ss in band.seeds.items():
        for s, h in zip(ss, band.hashes[cell]):
            assert BAND3_START <= s < BAND3_CEIL
            assert realization_hash(_sim_contract(cell, s).scenario) == h, f"{cell}:{s}"
    rep = independence_report(band, forbidden={"past-recorded":
                                               _collect_hashes_excluding_out()})
    assert rep["ok"], rep
    return band


def _episode3(cell, seed, mk_policy, *, bound, one_shot, wait_mode):
    from .yr141_bound_prepo import _episode
    prev = cand_mod.WAIT_MODE
    cand_mod.WAIT_MODE = wait_mode
    try:
        return _episode(cell, seed, mk_policy, bound=bound, one_shot=one_shot)
    finally:
        cand_mod.WAIT_MODE = prev


def _load3(root, ts):
    ck = torch.load(root / f"ppo_s{ts}" / "net.pt", map_location="cpu")
    actor = JointPairNet(250); actor.load_state_dict(ck["actor"]); actor.eval()
    ck0 = torch.load(Path("outputs/reports/yr125_diff_credit") / f"diff1_s{ts}"
                     / "rl_net.pt", map_location="cpu")
    norm = StateNorm(refs=ck0["norm_refs"])

    def mk():
        y88.FORBID_WAIT = True
        return y88.RLPolicy(actor, norm, name=f"{root.name}:{ts}")
    return mk


def evaluate3() -> dict:
    from statistics import stdev
    from ..integrated.baselines import ResolverPolicy, ServiceFirstSPTPreference
    from ..integrated.repro import repro_stamp
    band = _load_band3()
    eval_eps = [(c, s) for c in CELLS for s in band.seeds[c]]
    ts_list = (88_000, 99_000, 123_000)
    print(f"[eval3] SF {len(eval_eps)}", flush=True)
    sf = [_episode3(c, s, lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF"),
                    bound=False, one_shot=False, wait_mode="WAIT") for c, s in eval_eps]
    cfg = {"A": (OUT145 / "b2", "WAIT"), "B": (OUT / "train_b", "DEFER_ALL"),
           "R": (OUT / "train_r", "DEFER_ALL")}
    rows, ckpt_sha = {}, {}
    for arm, (root, wm) in cfg.items():
        for ts in ts_list:
            print(f"[eval3] {arm}:{ts}", flush=True)
            ckpt_sha[f"{arm}:{ts}"] = hashlib.sha256(
                (root / f"ppo_s{ts}" / "net.pt").read_bytes()).hexdigest()
            mk = _load3(root, ts)
            rows[f"{arm}:{ts}"] = [_episode3(c, s, mk, bound=True, one_shot=True,
                                             wait_mode=wm) for c, s in eval_eps]
    # 시나리오(3초기화 평균) 독립 단위 (23차) — R−B v2
    scen_diff = []
    for j, (c, s) in enumerate(eval_eps):
        r_m = fmean(rows[f"R:{ts}"][j]["v2_total"] for ts in ts_list)
        b_m = fmean(rows[f"B:{ts}"][j]["v2_total"] for ts in ts_list)
        scen_diff.append({"cell": c, "seed": s, "diff": r_m - b_m})
    diffs = [d["diff"] for d in scen_diff]
    n = len(diffs)
    mean_d = fmean(diffs)
    sd_d = stdev(diffs)
    ci_hw = 1.696 * sd_d / (n ** 0.5)          # t(0.95, df=31) 단측
    per_init_dir = {ts: fmean(rows[f"R:{ts}"][j]["v2_total"]
                              - rows[f"B:{ts}"][j]["v2_total"]
                              for j in range(n)) for ts in ts_list}
    guards = {}
    for arm in ("B", "R"):
        eps = [e for ts in ts_list for e in rows[f"{arm}:{ts}"]]
        guards[arm] = {"compl_min": min(e["compl"] for e in eps),
                       "backlog_max": max(e["backlog"] for e in eps),
                       "healthy_all": all(e["healthy"] for e in eps),
                       "repo_dom": sum(1 for e in eps
                                       if e["shares"].get("REPOSITION", 0) > 0.60)}
    exposure = {}
    for ts in ts_list:
        m = json.loads((OUT / "train_r" / f"ppo_s{ts}" / "train_meta.json")
                       .read_text(encoding="utf-8"))
        exposure[ts] = {"labeled": m["counters"]["labeled"],
                        "ok": m["exposure_ok"],
                        "prog_pref": m["counters"]["prog_pref"],
                        "wait_pref": m["counters"]["wait_pref"],
                        "ties": m["counters"]["ties"]}
    j = {"guards_B": all([guards["B"]["compl_min"] >= 1.0,
                          guards["B"]["backlog_max"] == 0,
                          guards["B"]["healthy_all"], guards["B"]["repo_dom"] == 0]),
         "guards_R": all([guards["R"]["compl_min"] >= 1.0,
                          guards["R"]["backlog_max"] == 0,
                          guards["R"]["healthy_all"], guards["R"]["repo_dom"] == 0]),
         "dir_2of3": sum(1 for v in per_init_dir.values() if v < 0) >= 2,
         "mean_neg": mean_d < 0,
         "ci_upper_neg": (mean_d + ci_hw) < 0,
         "exposure_ok_all": all(e["ok"] for e in exposure.values())}
    j["success"] = all([j["guards_B"], j["guards_R"], j["dir_2of3"],
                        j["mean_neg"], j["ci_upper_neg"]])
    if not j["exposure_ok_all"]:
        j["tag"] = "조작 노출 부족"
    res = {"repro": repro_stamp(
               experiment="YR-147 3단계 — B(유한 DEFER) vs R(B+반사실 순위손실)",
               seeds={"train": list(ts_list), **{c: band.seeds[c] for c in CELLS}},
               profile_id="calibrated",
               prereg="23차 계약 — 시나리오(3초기화 평균) 32단위·통과 = 방향 ≥2/3 ∧ "
                      "평균<0 ∧ CI 상한<0 ∧ 양군 하드가드. λ=0.1·동점폭 0.1·훈련 대역 "
                      "BASE+16..31·최소 노출 30.",
               extra={"band_digest": json.loads(BAND3_PATH.read_text(
                          encoding="utf-8"))["digest"],
                      "ckpt_sha256": ckpt_sha}),
           "judgment": {**j, "mean_R_minus_B": mean_d, "ci_halfwidth": ci_hw,
                        "sd_scen": sd_d, "per_init_dir": per_init_dir,
                        "n_scenarios": n},
           "guards": guards, "exposure": exposure, "scen_diff": scen_diff,
           "sf": sf, "arms": rows}
    (OUT / "results3.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print(json.dumps({k: v for k, v in j.items()}, ensure_ascii=False))
    print(f"mean R−B = {mean_d:.4f} ± {ci_hw:.4f} (단측 CI 상한 {mean_d + ci_hw:.4f})")
    return res


# ------------------------------------------------------------------ 2단계 계측 (파일럿)
def _load(arm: str, ts: int):
    ck = torch.load(ARM_ROOT[arm] / f"ppo_s{ts}" / "net.pt", map_location="cpu")
    actor = JointPairNet(250); actor.load_state_dict(ck["actor"]); actor.eval()
    ck0 = torch.load(Path("outputs/reports/yr125_diff_credit") / f"diff1_s{ts}"
                     / "rl_net.pt", map_location="cpu")
    norm = StateNorm(refs=ck0["norm_refs"])
    y88.FORBID_WAIT = True
    return actor, norm, y88.RLPolicy(actor, norm, name=f"{arm}:{ts}")


def _continue_to_end(sim, policy, gen):
    while True:
        dp = sim.run_until_decision()
        if dp is None:
            break
        gen_by = {c: gen.generate(sim, c, y88.LEVEL) for c in dp.crane_ids}
        assign = policy.decide(sim, dp, gen_by)
        _record_prepo(sim, dp, assign)
        _apply(sim, assign)
    jobs = list(sim.jobs.values())
    compl = (sum(1 for j in jobs if j.status.name == "DONE") / len(jobs)) if jobs else 1.0
    return {"phi": _v2_hard_total(sim), "compl": compl, "backlog": sim.unfinished_backlog()}


def _measure_ep(cell, seed, actor, norm, policy, jr, do_pairs, pair_budget):
    sim = _sim_contract(cell, seed)
    gen = CandidateGenerator()
    st = {"dec": 0, "multi": 0, "ww_avoidable": 0, "kind_counts": {}, "pairs": [],
          "defer_exec": 0, "defer_triggered": 0, "defer_untriggered": 0}
    ep_pairs = 0
    while True:
        dp = sim.run_until_decision()
        if dp is None:
            break
        gen_by = {c: gen.generate(sim, c, y88.LEVEL) for c in dp.crane_ids}
        assign = policy.decide(sim, dp, gen_by)
        st["dec"] += 1
        for c in dp.crane_ids:
            k = assign[c].kind.name
            st["kind_counts"][k] = st["kind_counts"].get(k, 0) + 1
            if k == "WAIT" and getattr(assign[c], "defer_until", None) is not None:
                st["defer_exec"] += 1
                if getattr(assign[c], "defer_trigger", None) is not None:
                    st["defer_triggered"] += 1
                else:
                    st["defer_untriggered"] += 1
        if len(dp.crane_ids) >= 2:
            st["multi"] += 1
        if len(dp.crane_ids) >= 2 and all(g.kind.name == "WAIT" for g in assign.values()):
            rows, assigns = y88.build_rows(sim, dp, gen_by, norm, jr, 0)
            prog = [i for i, a in enumerate(assigns)
                    if any(a[c].kind.name in PROG for c in a)]
            if prog:
                st["ww_avoidable"] += 1
                if do_pairs and ep_pairs < PAIR_EP_CAP and len(st["pairs"]) < pair_budget:
                    with torch.no_grad():
                        cost, _ = actor(torch.tensor(rows, dtype=torch.float32))
                    top = sorted(prog, key=lambda i: float(cost[i]))[:TOPK]
                    phis = []
                    for i in top:
                        s2 = copy.deepcopy(sim)
                        _record_prepo(s2, dp, assigns[i])
                        _apply(s2, assigns[i])
                        phis.append(_continue_to_end(s2, policy, CandidateGenerator()))
                    sW = copy.deepcopy(sim)
                    _apply(sW, assign)
                    rW = _continue_to_end(sW, policy, CandidateGenerator())
                    st["pairs"].append({
                        "cell": cell, "seed": seed, "t": float(sim.now),
                        "pred_ww": min(float(cost[i]) for i, a in enumerate(assigns)
                                       if all(a[c].kind.name == "WAIT" for c in a)),
                        "pred_prog_topk": [float(cost[i]) for i in top],
                        "phi_ww": rW["phi"], "phi_prog_topk": [p["phi"] for p in phis],
                        "d_wait_realized": rW["phi"] - min(p["phi"] for p in phis),
                        "compl_ww": rW["compl"],
                        "compl_prog_min": min(p["compl"] for p in phis)})
                    ep_pairs += 1
        _record_prepo(sim, dp, assign)
        _apply(sim, assign)
    jobs = list(sim.jobs.values())
    compl = (sum(1 for j in jobs if j.status.name == "DONE") / len(jobs)) if jobs else 1.0
    st.update({"cell": cell, "seed": seed, "v2_total": _v2_hard_total(sim),
               "compl": compl, "backlog": sim.unfinished_backlog(),
               "defer_wakes": sum(1 for e in sim.event_log if e[1] == "DEFER_WAKE")})
    return st


def measure(arm: str, ts: int) -> dict:
    prev = cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT, cand_mod.WAIT_MODE
    cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT = True, True
    cand_mod.WAIT_MODE = ARM_WAIT_MODE[arm]
    try:
        actor, norm, policy = _load(arm, ts)
        jr = JointRolloutGreedy(RC_EVAL, horizon_s=1800.0, generator=CandidateGenerator(),
                                forbid_strategic_wait=True)
        do_pairs = arm in ("b", "c")                 # A 짝은 1단계(비층화) 참조
        eps, pairs = [], []
        for cell in CELLS:                           # 층화: 셀별 짝 할당 (21차 ①)
            budget = PAIR_CELL_QUOTA
            for i in range(16):
                st = _measure_ep(cell, BASE[cell] + i, actor, norm, policy, jr,
                                 do_pairs, budget)
                budget -= len(st["pairs"])
                pairs.extend(st.pop("pairs"))
                eps.append(st)
    finally:
        (cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT, cand_mod.WAIT_MODE) = prev
    multi = sum(e["multi"] for e in eps)
    ww = sum(e["ww_avoidable"] for e in eps)
    kc = {}
    for e in eps:
        for k, v in e["kind_counts"].items():
            kc[k] = kc.get(k, 0) + v
    tot_k = sum(kc.values()) or 1
    d = [p["d_wait_realized"] for p in pairs]
    summary = {
        "arm": arm, "ts": ts, "n_eps": len(eps),
        "compl_min": min(e["compl"] for e in eps),
        "n_incomplete_eps": sum(1 for e in eps if e["compl"] < 1.0),
        "backlog_total": sum(e["backlog"] for e in eps),
        "ww_avoidable": ww, "multi": multi,
        "ww_rate": (ww / multi if multi else 0.0),
        "v2_mean": fmean(e["v2_total"] for e in eps),
        "share": {k: round(v / tot_k, 4) for k, v in sorted(kc.items())},
        "defer_exec": sum(e["defer_exec"] for e in eps),
        "defer_triggered": sum(e["defer_triggered"] for e in eps),
        "defer_untriggered": sum(e["defer_untriggered"] for e in eps),
        "defer_wakes": sum(e["defer_wakes"] for e in eps),
        "n_pairs": len(pairs),
        "pair_cells": {c: sum(1 for p in pairs if p["cell"] == c) for c in CELLS},
        "d_wait_neg": (sum(1 for x in d if x < 0) / len(d) if d else None),
        "d_wait_zero": (sum(1 for x in d if x == 0) / len(d) if d else None),
        "d_wait_pos": (sum(1 for x in d if x > 0) / len(d) if d else None)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"pilot_{arm}_s{ts}.json").write_text(
        json.dumps({"summary": summary, "eps": eps, "pairs": pairs,
                    "prereg_commit": "2369af8"}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=0)
    ap.add_argument("--train3", type=int, default=0, help="3단계 학습 (arm b|r)")
    ap.add_argument("--measure", type=int, default=0)
    ap.add_argument("--make-band3", action="store_true")
    ap.add_argument("--eval3", action="store_true")
    ap.add_argument("--arm", choices=("a", "b", "c", "r"), default=None)
    a = ap.parse_args()
    if a.make_band3:
        make_band3()
    if a.eval3:
        evaluate3()
    if a.train:
        assert a.arm in ("b", "c"), "A 는 YR-145 체크포인트 재사용 — 학습 없음"
        train(a.train, a.arm)
    if a.train3:
        assert a.arm in ("b", "r")
        train_one_v3(a.train3, a.arm)
    if a.measure:
        measure(a.arm, a.measure)
    print("DONE")
