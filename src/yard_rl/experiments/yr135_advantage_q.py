"""YR-135 — 공동 Advantage-Q 1단계: V(상태)/A(후보) 분리 구조 (BlockQ-v3, 오프라인).

■ 세대 전환 근거 (외부 7차 피드백 — v2 마감)
갱신×20 무효(YR-127)·절대예측↑에도 순위 불가(YR-128/130)·WAIT 기준 차분→REPO 풍선
(YR-125)·순위손실만→비용 크기 상실(YR-131)·미완 개수→시드별 방향 반전(YR-132).
결론: 결함은 "절대 Q" 자체가 아니라 **하나의 총비용 회귀값에 비용 크기 예측과 후보
순위 판단을 동시에 맡긴 계약**. 표준 수정 = 절대 Q 유지 + V/A 분리(dueling):
  Q(s,a) = V(s) + A(s,a) − mean_{a'∈s} A(s,a')
같은 상태에서 V 는 공통이므로 선택은 A 순서가 결정하고, Q 전체는 비용 크기를 보존한다.

■ 1단계 계약 (결과 미열람 동결 — 검증 순서의 첫 단만)
- 유지: 공동후보 생성·중앙 공동배정·물리 안전 resolver·비용계약·상태 인코딩(250차원)·
  같은 상태 K 후보 비교. **유일 변경 = 점수망 구조** (단일 점수 → V/A 분리).
- 중단 목록 준수: WAIT 기준 차분 없음(표적 = 후보별 **절대** C600 rollout 비용) ·
  미완 개수 보정 없음 · 게이트/예약/추가 상태정보 없음 · λ/시드 원인 미세조정 없음.
- SF-SPT 지위: 학습표적 기준이 아니라 **후보집합 안의 동등한 경쟁자**(그 공동선택은
  열거 조합에 자연 포함) + rollout 후속 기준정책 + 성능 기준선.
- 표적: y(a) = C600(a)/SCALE — 후보별 절대 창비용(기준행동 rollout 불필요·잔여항 없음).
  순위 진실은 결정 내 C600 순서(= 이전 D 순서와 동일 — 기준값이 결정 내 상수였으므로
  YR-131/132 수치와 직접 비교 가능).
- 학습: 결정 단위 배치(그룹별 V 1회 + A 후보별) Huber(Q, y)·fresh init·Adam 5e-4·
  clip 10·선택지표 = 선택 대역 Pearson r(Q̂, y)·patience 100. 순위 보조손실은 **없음**
  (2단계 — 1단계에서 순위 부족할 때만 별도 단일축).
- 판정 대역 = **새 대역 BASE+1100/1101 (미열람)**, DIFF1 궤적 라벨(프로토콜 연속).
  J1 순위: 3/3 결정 내 ρ ≥ 0.30 ∧ top-1 ≥ 0.35.
  J2 절대 적합: 3/3 Pearson r(Q̂, C600) ≥ 0.5.
  J3 선택 후 손실: 3/3 mean regret ≤ 같은 대역의 YR-131-b 참조망.
  성공 = J1∧J2∧J3 → 4단계(에피소드 판정) 제안. J2 만 통과·J1 미달 → 2단계(A 순위
  보조손실) 단일축. J2 미달 → 구조 재검. 진단·개발 — 판정·채택 아님.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from statistics import fmean

import torch
from torch import nn

from ..contract.schema import CandidateKind
from ..integrated.baselines import (JointRolloutGreedy, ResolverPolicy,
                                    ServiceFirstSPTPreference, _apply, _rollout_cost,
                                    _wait_of)
from ..integrated.candidates import CandidateGenerator
from ..integrated.encoding import StateNorm
from ..integrated.joint_distill import JointPairNet
from ..integrated.repro import repro_stamp
from .yr088_joint_rl import LEVEL, RC as RC_TRAIN, build_rows
from .yr090_dense_vessel import BASE, CELLS, SCALE, _sim
from .yr119_wait_retrain import _set_forbid_wait
from .yr128_rank_diagnosis import _spearman

OUT = Path("outputs/reports/yr135_advantage_q")
DIFF1 = Path("outputs/reports/yr125_diff_credit")
B131 = Path("outputs/reports/yr131_k_candidates")
TRAIN_SEEDS = (88_000, 99_000, 123_000)
HORIZON_S = 600.0
BANDS = {"train": [(c, BASE[c] + i) for c in CELLS for i in range(2)],
         "select": [(c, BASE[c] + 700 + i) for c in CELLS for i in range(2)],
         "judge": [(c, BASE[c] + 1100 + i) for c in CELLS for i in range(2)],
         # 2단계 판정 대역 — 1100 대역은 1단계 판정으로 열람됨(재사용 금지)
         "judge2": [(c, BASE[c] + 1300 + i) for c in CELLS for i in range(2)]}
AUX_W, MARGIN, PAIR_EPS = 1.0, 0.01, 0.01   # 2단계 순위 보조손실 (동등 가중 앵커, 동결)
# 250차원 행의 구획 (build_rows 계약): ctx_a 0:116 · blk_a 116:154 · ctx_b 154:212 · blk_b 212:250
CTX_A, BLK_A, CTX_B, BLK_B = (0, 116), (116, 154), (154, 212), (212, 250)


class JointDuelingNet(nn.Module):
    """V(상태 문맥만) + A(전체 행) − mean A (결정 내). Q 절대값과 순위 능력을 분리 보존."""

    def __init__(self, in_dim: int = 250):
        super().__init__()
        self.in_dim = in_dim
        v_dim = (CTX_A[1] - CTX_A[0]) + (CTX_B[1] - CTX_B[0])
        self.v_net = nn.Sequential(nn.Linear(v_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.a_net = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(),
                                   nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def _state_vec(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x[:, CTX_A[0]:CTX_A[1]], x[:, CTX_B[0]:CTX_B[1]]], dim=1)

    def q_groups(self, x: torch.Tensor, groups: list[list[int]]) -> torch.Tensor:
        """결정 그룹별 Q = V + (A − mean_그룹 A). x 전체에 대해 그룹 단위로 계산."""
        a = self.a_net(x).squeeze(-1)
        v = self.v_net(self._state_vec(x)).squeeze(-1)
        q = torch.empty_like(a)
        for ii in groups:
            idx = torch.tensor(ii, dtype=torch.long)
            q[idx] = v[idx] + a[idx] - a[idx].mean()
        return q


def _episode_labels_abs(net, norm, cell, seed, band, ep_id):
    """궤적 = DIFF1 argmin (프로토콜 연속). 라벨 = 후보별 **절대** C600 (기준 rollout 없음)."""
    sim = _sim(cell, seed)
    gen = CandidateGenerator()
    jr = JointRolloutGreedy(RC_TRAIN, horizon_s=1800.0, generator=gen,
                            forbid_strategic_wait=False)
    base_pol = ResolverPolicy(ServiceFirstSPTPreference(), "BASE")
    out = []
    dp = sim.run_until_decision()
    k = 0
    while dp is not None:
        gen_by = {c: gen.generate(sim, c, LEVEL) for c in dp.crane_ids}
        rows, assigns = build_rows(sim, dp, gen_by, norm, jr, k)
        if not assigns:
            _apply(sim, {c: _wait_of(gen_by[c]) for c in dp.crane_ids})
        else:
            with torch.no_grad():
                sc, _ = net(torch.tensor(rows, dtype=torch.float32))
            pick = int(torch.argmin(sc))
            if len(assigns) >= 2:
                for i, a in enumerate(assigns):
                    ca, _ = _rollout_cost(sim, a, RC_TRAIN, horizon_s=HORIZON_S,
                                          base_policy=base_pol, generator=gen)
                    out.append((rows[i], {
                        "band": band, "dec": f"{ep_id}:{k}", "exec": i == pick,
                        "c": ca, "has_wait": any(a[c].kind == CandidateKind.WAIT
                                                 for c in dp.crane_ids)}))
            _apply(sim, assigns[pick])
        dp = sim.run_until_decision()
        k += 1
    return out


def _label_band(ts: int, band: str) -> Path:
    dst = OUT / f"dataset_{band}_s{ts}.pt"
    if dst.exists():
        return dst
    ck = torch.load(DIFF1 / f"diff1_s{ts}" / "rl_net.pt", map_location="cpu")
    net = JointPairNet(ck["in_dim"]); net.load_state_dict(ck["state"]); net.eval()
    norm = StateNorm(refs=ck["norm_refs"])
    rows_x, metas = [], []
    _set_forbid_wait(False)
    try:
        for cell, seed in BANDS[band]:
            lab = _episode_labels_abs(net, norm, cell, seed, band, f"{cell}:{seed}")
            rows_x += [r for r, _ in lab]
            metas += [m for _, m in lab]
            print(f"[label {band} s{ts} {cell}:{seed}] {len(lab)} 누적 {len(metas)}",
                  flush=True)
    finally:
        _set_forbid_wait(True)
    torch.save({"X": torch.tensor(rows_x, dtype=torch.float32), "meta": metas,
                "scale": SCALE}, dst)
    return dst


def _load(ts: int, band: str):
    d = torch.load(OUT / f"dataset_{band}_s{ts}.pt", map_location="cpu")
    X, meta = d["X"], d["meta"]
    y = torch.tensor([m["c"] / SCALE for m in meta], dtype=torch.float32)
    by: dict[str, list[int]] = {}
    for i, m in enumerate(meta):
        by.setdefault(m["dec"], []).append(i)
    groups = [ii for ii in by.values() if len(ii) >= 2]
    return {"X": X, "y": y, "meta": meta, "groups": groups}


def _pearson_t(a, b):
    if a.numel() < 3 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _rank_eval(pred, data):
    top1, rhos, regret = [], [], []
    for ii in data["groups"]:
        dh = [float(pred[i]) for i in ii]
        dd = [float(data["y"][i]) for i in ii]
        top1.append(1.0 if dh.index(min(dh)) == dd.index(min(dd)) else 0.0)
        regret.append((dd[dh.index(min(dh))] - min(dd)) * SCALE)
        if len(ii) >= 3:
            r = _spearman(dh, dd)
            if r is not None:
                rhos.append(r)
    return {"top1": round(fmean(top1), 4), "rho": round(fmean(rhos), 4) if rhos else None,
            "regret_mean": round(fmean(regret), 4), "n_decisions": len(top1)}


def train135(ts: int) -> Path:
    tr, va = _load(ts, "train"), _load(ts, "select")
    torch.manual_seed(ts)
    net = JointDuelingNet(tr["X"].shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    g = torch.Generator().manual_seed(ts)
    best = {"sel": -1.0, "epoch": 0, "state": None}
    n_groups = len(tr["groups"])
    for ep in range(1, 2001):
        net.train()
        perm = torch.randperm(n_groups, generator=g).tolist()
        for s0 in range(0, n_groups, 16):
            gs = [tr["groups"][i] for i in perm[s0:s0 + 16]]
            flat = [i for ii in gs for i in ii]
            local = []
            off = 0
            for ii in gs:
                local.append(list(range(off, off + len(ii))))
                off += len(ii)
            x = tr["X"][flat]
            q = net.q_groups(x, local)
            loss = nn.functional.smooth_l1_loss(q, tr["y"][flat])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step()
        net.eval()
        with torch.no_grad():
            pv = net.q_groups(va["X"], va["groups"])
        sel = _pearson_t(pv, va["y"])
        if sel > best["sel"] + 1e-3:
            best = {"sel": sel, "epoch": ep, "state": copy.deepcopy(net.state_dict())}
        if ep - best["epoch"] >= 100:
            break
    net.load_state_dict(best["state"]); net.eval()
    out = OUT / f"vaq_s{ts}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "in_dim": net.in_dim, "arm": "VAQ",
                "train_seed": ts, "best_epoch": best["epoch"]}, out / "net.pt")
    print(f"[135 s{ts}] best_ep {best['epoch']} select r {best['sel']:.4f}", flush=True)
    return out / "net.pt"


def judge() -> dict:
    ev = {}
    for ts in TRAIN_SEEDS:
        data = _load(ts, "judge")
        ck = torch.load(OUT / f"vaq_s{ts}" / "net.pt", map_location="cpu")
        net = JointDuelingNet(ck["in_dim"]); net.load_state_dict(ck["state"]); net.eval()
        with torch.no_grad():
            p = net.q_groups(data["X"], data["groups"])
        r_abs = _pearson_t(p, data["y"])
        m = _rank_eval(p, data)
        ck_b = torch.load(B131 / f"b_s{ts}" / "net.pt", map_location="cpu")
        ref = JointPairNet(ck_b["in_dim"]); ref.load_state_dict(ck_b["state"]); ref.eval()
        with torch.no_grad():
            pr, _ = ref(data["X"])
        mr = _rank_eval(pr, data)
        ev[ts] = {"VAQ": {"r_abs": round(r_abs, 4), **m},
                  "ref_131b": {"r_abs": round(_pearson_t(pr, data["y"]), 4), **mr}}
        print(f"[judge s{ts}] {json.dumps(ev[ts], ensure_ascii=False)}", flush=True)
    j1 = all((e["VAQ"]["rho"] or 0) >= 0.30 and e["VAQ"]["top1"] >= 0.35
             for e in ev.values())
    j2 = all(e["VAQ"]["r_abs"] >= 0.5 for e in ev.values())
    j3 = all(e["VAQ"]["regret_mean"] <= e["ref_131b"]["regret_mean"] for e in ev.values())
    judgment = {"J1_rank": j1, "J2_abs_fit": j2, "J3_regret_vs_131b": j3,
                "success": bool(j1 and j2 and j3), "per_seed": ev}
    res = {"repro": repro_stamp(
               experiment="YR-135 공동 Advantage-Q 1단계 (V/A 분리, 오프라인)",
               seeds={"train_ckpt": list(TRAIN_SEEDS),
                      **{b: sorted({s for _, s in e}) for b, e in BANDS.items()}},
               profile_id="calibrated",
               prereg="유일 변경 = 점수망 구조(단일→V/A 분리). 표적 = 후보별 절대 C600 "
                      "(차분·잔여항 없음). 순위 보조손실 없음(2단계 몫). 판정 = 새 대역 "
                      "BASE+1100/1101: J1 3/3 ρ≥0.30∧top1≥0.35 · J2 3/3 r(Q̂,C600)≥0.5 · "
                      "J3 3/3 regret ≤ 131-b 참조. 진단 — 판정·채택 아님.",
               extra={"horizon_s": HORIZON_S}),
           "judgment": judgment}
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(json.dumps({"J1": j1, "J2": j2, "J3": j3,
                      "success": judgment["success"]}, ensure_ascii=False))
    return res


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    for band in ("train", "select", "judge"):
        for ts in TRAIN_SEEDS:
            _label_band(ts, band)
    for ts in TRAIN_SEEDS:
        train135(ts)
    return judge()


# ---------------------------------------------------------------- 2단계 (순위 보조손실)
def train_stage2(ts: int) -> Path:
    """1단계와 동일 구조·데이터. 유일 추가 = 결정 내 pairwise margin 순위 보조손실.

    loss = Huber(Q, y) + AUX_W × margin_rank (131-b 규약: margin 0.01·잡음쌍 제외 0.01).
    선택지표 = select 대역 결정 내 ρ (순위가 결핍 능력) — 절대 r 은 병행 기록.
    """
    tr, va = _load(ts, "train"), _load(ts, "select")
    torch.manual_seed(ts)
    net = JointDuelingNet(tr["X"].shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    g = torch.Generator().manual_seed(ts)
    pair_by_group = []
    for ii in tr["groups"]:
        ps = [(a, b) for a in ii for b in ii
              if float(tr["y"][a]) + PAIR_EPS < float(tr["y"][b])]
        pair_by_group.append(ps)
    best = {"sel": -1.0, "epoch": 0, "state": None, "r": 0.0}
    n_groups = len(tr["groups"])
    for ep in range(1, 2001):
        net.train()
        perm = torch.randperm(n_groups, generator=g).tolist()
        for s0 in range(0, n_groups, 16):
            gi = perm[s0:s0 + 16]
            gs = [tr["groups"][i] for i in gi]
            flat = [i for ii in gs for i in ii]
            pos = {gidx: k for k, gidx in enumerate(flat)}
            local, pa, pb = [], [], []
            off = 0
            for i, ii in zip(gi, gs):
                local.append(list(range(off, off + len(ii))))
                off += len(ii)
                for a, b in pair_by_group[i]:
                    pa.append(pos[a]); pb.append(pos[b])
            x = tr["X"][flat]
            q = net.q_groups(x, local)
            loss = nn.functional.smooth_l1_loss(q, tr["y"][flat])
            if pa:
                loss = loss + AUX_W * nn.functional.margin_ranking_loss(
                    q[pb], q[pa], torch.ones(len(pa)), margin=MARGIN)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step()
        net.eval()
        with torch.no_grad():
            pv = net.q_groups(va["X"], va["groups"])
        m = _rank_eval(pv, va)
        sel = m["rho"] or -1.0
        if sel > best["sel"] + 1e-3:
            best = {"sel": sel, "epoch": ep, "state": copy.deepcopy(net.state_dict()),
                    "r": _pearson_t(pv, va["y"])}
        if ep - best["epoch"] >= 100:
            break
    net.load_state_dict(best["state"]); net.eval()
    out = OUT / f"vaq2_s{ts}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "in_dim": net.in_dim, "arm": "VAQ2",
                "train_seed": ts, "best_epoch": best["epoch"]}, out / "net.pt")
    print(f"[135-2 s{ts}] best_ep {best['epoch']} select ρ {best['sel']:.4f} "
          f"(r {best['r']:.4f})", flush=True)
    return out / "net.pt"


def judge_stage2() -> dict:
    ev = {}
    for ts in TRAIN_SEEDS:
        data = _load(ts, "judge2")
        out_ts = {}
        for name, ctor, ckp in (
                ("VAQ2", JointDuelingNet, OUT / f"vaq2_s{ts}" / "net.pt"),
                ("VAQ1", JointDuelingNet, OUT / f"vaq_s{ts}" / "net.pt"),
                ("ref_131b", JointPairNet, B131 / f"b_s{ts}" / "net.pt")):
            ck = torch.load(ckp, map_location="cpu")
            net = ctor(ck["in_dim"]); net.load_state_dict(ck["state"]); net.eval()
            with torch.no_grad():
                if isinstance(net, JointDuelingNet):
                    p = net.q_groups(data["X"], data["groups"])
                else:
                    p, _ = net(data["X"])
            out_ts[name] = {"r_abs": round(_pearson_t(p, data["y"]), 4),
                            **_rank_eval(p, data)}
        ev[ts] = out_ts
        print(f"[judge2 s{ts}] {json.dumps(ev[ts], ensure_ascii=False)}", flush=True)
    j1 = all((e["VAQ2"]["rho"] or 0) >= 0.30 and e["VAQ2"]["top1"] >= 0.35
             for e in ev.values())
    j2 = all(e["VAQ2"]["r_abs"] >= 0.5 for e in ev.values())
    j3 = all(e["VAQ2"]["regret_mean"] <= e["ref_131b"]["regret_mean"]
             for e in ev.values())
    judgment = {"J1_rank": j1, "J2_abs_fit_kept": j2, "J3_regret_vs_131b": j3,
                "success": bool(j1 and j2 and j3), "per_seed": ev}
    res = {"repro": repro_stamp(
               experiment="YR-135 2단계 — V/A + 결정 내 순위 보조손실 (오프라인)",
               seeds={"train_ckpt": list(TRAIN_SEEDS),
                      "judge2": sorted({s for _, s in BANDS['judge2']})},
               profile_id="calibrated",
               prereg="유일 추가 = 순위 보조손실(α=1.0 동등 앵커·margin 0.01·eps 0.01). "
                      "판정 대역 = BASE+1300/1301 신규(1100 열람됨). J1 3/3 ρ≥0.30∧"
                      "top1≥0.35 · J2 3/3 r≥0.5 유지 · J3 regret ≤ 131-b. 라벨은 v1 "
                      "비용계약 — 구조 진단(채택 승계 금지, YR-136 재라벨이 채택 경로). "
                      "진단 — 판정·채택 아님.",
               extra={"aux_w": AUX_W, "margin": MARGIN, "pair_eps": PAIR_EPS}),
           "judgment": judgment}
    (OUT / "results_stage2.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"J1": j1, "J2": j2, "J3": j3,
                      "success": judgment["success"]}, ensure_ascii=False))
    return res


def run_stage2() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    for ts in TRAIN_SEEDS:
        _label_band(ts, "judge2")
    for ts in TRAIN_SEEDS:
        train_stage2(ts)
    return judge_stage2()


# ---------------------------------------------------------------- 3단계 (YR-136 v2 재라벨)
# 라벨 = 창(600s) 전후 v2 총비용(J) 변화 — 정렬 조항 그대로: 창 안 실현 + 창끝 잔여
# 기대비용(예측 Ô/F̂ 에 내재). 진입 전(PLANNED) 트럭은 양쪽 세계 공통 근사로 제외(고지).
BANDS2 = {"train": [(c, BASE[c] + i) for c in CELLS for i in range(2)],
          "select": [(c, BASE[c] + 700 + i) for c in CELLS for i in range(2)],
          "judge3": [(c, BASE[c] + 1500 + i) for c in CELLS for i in range(2)]}


def _j_total_v2(sim, kf) -> float:
    from ..integrated.cost_curve_v2 import (j_truck, j_vessel, predict_gate_out,
                                            predict_vessel_completion, truck_target_s)
    tot = 0.0
    for jid, j in sim.jobs.items():
        if not getattr(j, "is_external_truck", False):
            continue
        a = getattr(j, "actual_gate_in", None)
        if a is None or j.status.name == "PLANNED":
            continue
        o = getattr(j, "actual_gate_out", None)
        if o is None:
            o_hat = predict_gate_out(sim, jid)
            if o_hat is None:
                continue
            o = o_hat + kf.bias_t_s
        tot += j_truck(o, a, truck_target_s(sim, a), kf.kappa_t_s)
    for v in sim.vessels.values():
        p = v.plan.planned_completion_s
        if p is None:
            continue
        f = getattr(getattr(v, "truth", None), "actual_completion_s", None)
        if f is None:
            f_hat = predict_vessel_completion(sim, v)
            if f_hat is None:
                continue
            f = f_hat + kf.bias_v_s
        tot += j_vessel(f, p, kf.kappa_v_s)
    return tot


def _episode_labels_v2(net, norm, cell, seed, band, ep_id, kf):
    sim = _sim(cell, seed)
    gen = CandidateGenerator()
    jr = JointRolloutGreedy(RC_TRAIN, horizon_s=1800.0, generator=gen,
                            forbid_strategic_wait=False)
    base_pol = ResolverPolicy(ServiceFirstSPTPreference(), "BASE")
    out = []
    dp = sim.run_until_decision()
    k = 0
    while dp is not None:
        gen_by = {c: gen.generate(sim, c, LEVEL) for c in dp.crane_ids}
        rows, assigns = build_rows(sim, dp, gen_by, norm, jr, k)
        if not assigns:
            _apply(sim, {c: _wait_of(gen_by[c]) for c in dp.crane_ids})
        else:
            with torch.no_grad():
                sc, _ = net(torch.tensor(rows, dtype=torch.float32))
            pick = int(torch.argmin(sc))
            if len(assigns) >= 2:
                j0 = _j_total_v2(sim, kf)
                for i, a in enumerate(assigns):
                    _, scratch = _rollout_cost(sim, a, RC_TRAIN, horizon_s=HORIZON_S,
                                               base_policy=base_pol, generator=gen)
                    out.append((rows[i], {
                        "band": band, "dec": f"{ep_id}:{k}", "exec": i == pick,
                        "c": _j_total_v2(scratch, kf) - j0,
                        "has_wait": any(a[c].kind == CandidateKind.WAIT
                                        for c in dp.crane_ids)}))
            _apply(sim, assigns[pick])
        dp = sim.run_until_decision()
        k += 1
    return out


def _label_band_v2(ts: int, band: str) -> Path:
    from ..integrated.cost_curve_v2 import KappaFit
    dst = OUT / f"dataset2_{band}_s{ts}.pt"
    if dst.exists():
        return dst
    kf = KappaFit.load()
    ck = torch.load(DIFF1 / f"diff1_s{ts}" / "rl_net.pt", map_location="cpu")
    net = JointPairNet(ck["in_dim"]); net.load_state_dict(ck["state"]); net.eval()
    norm = StateNorm(refs=ck["norm_refs"])
    rows_x, metas = [], []
    _set_forbid_wait(False)
    try:
        for cell, seed in BANDS2[band]:
            lab = _episode_labels_v2(net, norm, cell, seed, band, f"{cell}:{seed}", kf)
            rows_x += [r for r, _ in lab]
            metas += [m for _, m in lab]
            print(f"[label2 {band} s{ts} {cell}:{seed}] {len(lab)} 누적 {len(metas)}",
                  flush=True)
    finally:
        _set_forbid_wait(True)
    torch.save({"X": torch.tensor(rows_x, dtype=torch.float32), "meta": metas,
                "scale": SCALE}, dst)
    return dst


def _load2(ts: int, band: str):
    d = torch.load(OUT / f"dataset2_{band}_s{ts}.pt", map_location="cpu")
    X, meta = d["X"], d["meta"]
    y = torch.tensor([m["c"] / SCALE for m in meta], dtype=torch.float32)
    by: dict[str, list[int]] = {}
    for i, m in enumerate(meta):
        by.setdefault(m["dec"], []).append(i)
    return {"X": X, "y": y, "meta": meta,
            "groups": [ii for ii in by.values() if len(ii) >= 2]}


def train_stage3(ts: int) -> Path:
    """2단계 프로토콜 그대로(V/A + 순위 보조손실 α=1.0) — 유일 차이 = v2 라벨."""
    tr, va = _load2(ts, "train"), _load2(ts, "select")
    torch.manual_seed(ts)
    net = JointDuelingNet(tr["X"].shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    g = torch.Generator().manual_seed(ts)
    pair_by_group = [[(a, b) for a in ii for b in ii
                      if float(tr["y"][a]) + PAIR_EPS < float(tr["y"][b])]
                     for ii in tr["groups"]]
    best = {"sel": -1.0, "epoch": 0, "state": None, "r": 0.0}
    n_groups = len(tr["groups"])
    for ep in range(1, 2001):
        net.train()
        perm = torch.randperm(n_groups, generator=g).tolist()
        for s0 in range(0, n_groups, 16):
            gi = perm[s0:s0 + 16]
            gs = [tr["groups"][i] for i in gi]
            flat = [i for ii in gs for i in ii]
            pos = {gidx: kk for kk, gidx in enumerate(flat)}
            local, pa, pb = [], [], []
            off = 0
            for i, ii in zip(gi, gs):
                local.append(list(range(off, off + len(ii))))
                off += len(ii)
                for a, b in pair_by_group[i]:
                    pa.append(pos[a]); pb.append(pos[b])
            q = net.q_groups(tr["X"][flat], local)
            loss = nn.functional.smooth_l1_loss(q, tr["y"][flat])
            if pa:
                loss = loss + AUX_W * nn.functional.margin_ranking_loss(
                    q[pb], q[pa], torch.ones(len(pa)), margin=MARGIN)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step()
        net.eval()
        with torch.no_grad():
            pv = net.q_groups(va["X"], va["groups"])
        sel = _rank_eval(pv, va)["rho"] or -1.0
        if sel > best["sel"] + 1e-3:
            best = {"sel": sel, "epoch": ep, "state": copy.deepcopy(net.state_dict()),
                    "r": _pearson_t(pv, va["y"])}
        if ep - best["epoch"] >= 100:
            break
    net.load_state_dict(best["state"]); net.eval()
    out = OUT / f"vaq3_s{ts}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "in_dim": net.in_dim, "arm": "VAQ3",
                "train_seed": ts, "best_epoch": best["epoch"]}, out / "net.pt")
    print(f"[135-3 s{ts}] best_ep {best['epoch']} select ρ {best['sel']:.4f} "
          f"(r {best['r']:.4f})", flush=True)
    return out / "net.pt"


def judge_stage3() -> dict:
    ev = {}
    for ts in TRAIN_SEEDS:
        data = _load2(ts, "judge3")
        out_ts = {}
        for name, ctor, ckp in (
                ("VAQ3", JointDuelingNet, OUT / f"vaq3_s{ts}" / "net.pt"),
                ("VAQ2_v1labels", JointDuelingNet, OUT / f"vaq2_s{ts}" / "net.pt"),
                ("ref_131b", JointPairNet, B131 / f"b_s{ts}" / "net.pt")):
            ck = torch.load(ckp, map_location="cpu")
            net = ctor(ck["in_dim"]); net.load_state_dict(ck["state"]); net.eval()
            with torch.no_grad():
                if isinstance(net, JointDuelingNet):
                    p = net.q_groups(data["X"], data["groups"])
                else:
                    p, _ = net(data["X"])
            out_ts[name] = {"r_abs": round(_pearson_t(p, data["y"]), 4),
                            **_rank_eval(p, data)}
        ev[ts] = out_ts
        print(f"[judge3 s{ts}] {json.dumps(ev[ts], ensure_ascii=False)}", flush=True)
    j1 = all((e["VAQ3"]["rho"] or 0) >= 0.30 and e["VAQ3"]["top1"] >= 0.35
             for e in ev.values())
    j2 = all(e["VAQ3"]["r_abs"] >= 0.5 for e in ev.values())
    j3 = all(e["VAQ3"]["regret_mean"] <= e["ref_131b"]["regret_mean"]
             for e in ev.values())
    judgment = {"J1_rank": j1, "J2_abs_fit": j2, "J3_regret_vs_131b": j3,
                "success": bool(j1 and j2 and j3), "per_seed": ev}
    res = {"repro": repro_stamp(
               experiment="YR-135 3단계 — YR-136 v2 라벨(ΔJ 포텐셜) 재평가 (채택 경로)",
               seeds={"train_ckpt": list(TRAIN_SEEDS),
                      "judge3": sorted({s for _, s in BANDS2['judge3']})},
               profile_id="calibrated",
               prereg="라벨 = 창 전후 v2 총비용 J 변화(실현+창끝 예측 잔여 내재·PLANNED "
                      "제외 고지). 학습 = 2단계 프로토콜 동일(유일 차이 = 라벨). 판정 대역 "
                      "BASE+1500/1501 신규. J1 3/3 ρ≥0.30∧top1≥0.35 · J2 3/3 r≥0.5 · "
                      "J3 regret ≤ 131-b(같은 v2 진실). κ 동결값 사용·조정 금지.",
               extra={"aux_w": AUX_W, "margin": MARGIN}),
           "judgment": judgment}
    (OUT / "results_stage3.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"J1": j1, "J2": j2, "J3": j3,
                      "success": judgment["success"]}, ensure_ascii=False))
    return res


def run_stage3() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    for band in ("train", "select", "judge3"):
        for ts in TRAIN_SEEDS:
            _label_band_v2(ts, band)
    for ts in TRAIN_SEEDS:
        train_stage3(ts)
    return judge_stage3()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=(1, 2, 3))
    a = ap.parse_args()
    {1: run, 2: run_stage2, 3: run_stage3}[a.stage]()
    print("DONE")
