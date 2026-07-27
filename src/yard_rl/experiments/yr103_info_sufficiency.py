"""YR-103 G1 — 공개정보 기대상금 + 혼잡점수 예측력 (YR-102 GO/NO-GO 관문).

■ 사전등록 (결과 미열람 동결 — spec §G1·결정문서 §5)
Part A — 공개정보 기대상금 (선택/평가 표본 분리로 winner's curse 차단):
- 환경: yr099 블록쌍(A=high-tight·B=mid-loose)·SF 고정·t=0 review·전수 반입 후보,
  gate_block_contract=True (180~420s 계약). 시드 6 (신규 대역 835000/836000+i).
- 같은 공개 snapshot(예약·계획) 고정, 숨은 실현(준수오차·이동 draw)을 전용 난수열
  "g1real:{seed}:{r}" 로 R=10개 생성 (공개 오차분포 = 생성 분포族 그대로).
- d_{j,r} = terminal(transfer j | 실현 r) − terminal(KEEP | 실현 r)  [SF 실측, numeraire]
- 선택 r∈0..4 (mean_sel), 평가 r∈5..9 (mean_eval — 선택과 독립 → 무편향):
  j* = argmin_j mean_sel[d_j]. **주판정 KEEP 규칙 = mean_sel[d_{j*}] < −MARGIN(0.5)**
  (yr099 배포 규칙과 동일 — 검증 정정 2026-07-27: 이전 문구 "≥0 이면 KEEP" 와 코드가
  불일치했음. margin 규칙을 주판정으로 선언, 0-임계 변형은 참고 보고로 병기).
- **주판정: 시드 pooled mean_eval[d_{j*}] CI 상한 < 0 →
  "공개정보에 회수 가능한 신호 존재 = YR-102(분해 quote) GO" /
  CI 0 포함·평균 ≥ 0 → "공개정보/결정시점 한계 — 동적 갱신정보 필요" (spec §G2 조항).**
- 참고 보고(판정 아님): 실현별 오라클 mean_r[min_j d_{j,r}](상한)·비분리 낙관치
  min_j mean_eval[d_j](curse 크기 진단)·transfer/KEEP 수.
Part B — 혼잡점수 예측력 (관측 검증 1차 — 부호 확정은 신규 seed 재현 후):
- SF 단독 실행 중 900s 간격 스냅샷에서 C_minus/zero/plus 기록, 각 스냅샷의
  **후속 1800s 실현 numeraire 비용**과 Spearman ρ (표본 = 시드×셀×스냅샷).
- 보고: 세 부호 각각의 ρ — C_zero ρ>0 여부(예측력 존재)·최고 부호. D 변별력 한계
  (예상 g2b 상수) 명시.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import random
from pathlib import Path
from statistics import fmean

from ..integrated.block_congestion import block_congestion
from ..integrated.scenario_gen import trunc_normal
from .yr090_dense_vessel import _ci
from .yr099_transfer_mvp import (CELL_A, CELL_B, GAIN_MARGIN, ROUTE_S, _params, _scen,
                                 _sim_of, eligible_inbound, move_job, sf_run)

OUT = Path("outputs/reports/yr103_info_sufficiency")
BASE_A, BASE_B = 835_000, 836_000
N_SEEDS, R_TOTAL, R_SEL = 6, 10, 5

# 검증 정정(critical): 공용 _ci 는 df=19 외 fallback 2.1 — n=6(df=5)엔 2.571 이 옳다
# (주판정 CI 가 GO 쪽으로 18% 좁아지는 산술 오류). df별 양측 95% t.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 19: 2.093, 23: 2.069}


def _ci_t(d):
    from statistics import stdev
    m, n = fmean(d), len(d)
    se = stdev(d) / n ** 0.5
    t = _T95.get(n - 1, 2.0 if n > 30 else 2.571)
    return round(m, 3), round(m - t * se, 3), round(m + t * se, 3)


def _params103(cell):
    return dataclasses.replace(_params(cell), gate_block_contract=True)


def _scen103(seed: int, cell):
    from ..integrated.profiles import build_calibrated_profile
    from ..integrated.scenario_gen import generate_terminal_scenario
    return generate_terminal_scenario(build_calibrated_profile(), seed, _params103(cell))


def realize(scen, params, key: str):
    """공개 snapshot 고정 + 숨은 실현 재추출 — 전용 난수열(생성 실현 열과 분리).

    준수오차: clip(N(0,σ),±2σ) · 이동: 계약 분포(180~420) — 생성기와 같은 분포族.
    appointment/estimated(공개)는 불변 — '같은 공개정보, 다른 숨은 미래'.
    """
    from ..integrated.scenario_gen import (GATE_BLOCK_MAX_S, GATE_BLOCK_MEAN_S,
                                           GATE_BLOCK_MIN_S, GATE_BLOCK_SIGMA_S)
    s = copy.deepcopy(scen)
    adh = random.Random(f"g1real-adh:{key}")
    g2b = random.Random(f"g1real-g2b:{key}")
    sig = params.appt_adherence_sigma_s
    for j in s.jobs:
        appt = getattr(j, "appointment_gate_time", None)
        if appt is None:
            continue
        delta = max(-2.0 * sig, min(2.0 * sig, adh.gauss(0.0, sig))) if sig > 0 else 0.0
        a_in = max(0.0, appt + delta)
        j.actual_gate_in = a_in
        j.actual_block_arrival = a_in + trunc_normal(
            g2b, GATE_BLOCK_MEAN_S, GATE_BLOCK_SIGMA_S / GATE_BLOCK_MEAN_S,
            lo=GATE_BLOCK_MIN_S, hi=GATE_BLOCK_MAX_S)
    return s


# ---------------------------------------------------------------- Part A
def part_a_seed(i: int) -> dict:
    sa0, sb0 = _scen103(BASE_A + i, CELL_A), _scen103(BASE_B + i, CELL_B)
    pa, pb = _params103(CELL_A), _params103(CELL_B)
    cands = ([("A->B", jid) for jid in eligible_inbound(sa0)]
             + [("B->A", jid) for jid in eligible_inbound(sb0)])
    d = {c: [] for c in cands}                       # d[(dir,jid)][r]
    oracle_r = []
    for r in range(R_TOTAL):
        ra = realize(sa0, pa, f"s{i}:A:{r}")
        rb = realize(sb0, pb, f"s{i}:B:{r}")
        keep = sf_run(ra)["total"] + sf_run(rb)["total"]
        best_r = 0.0
        for (direction, jid) in cands:
            src, dst = (ra, rb) if direction == "A->B" else (rb, ra)
            try:
                s_wo, d_w = move_job(src, dst, jid)
            except ValueError:
                d[(direction, jid)].append(0.0)
                continue
            tot = sf_run(s_wo)["total"] + sf_run(d_w)["total"] + ROUTE_S / 3600.0
            dj = tot - keep
            d[(direction, jid)].append(dj)
            best_r = min(best_r, dj)
        oracle_r.append(best_r)
    # 선택(0..R_SEL-1) / 평가(R_SEL..) 분리
    sel = {c: fmean(v[:R_SEL]) for c, v in d.items()}
    ev = {c: fmean(v[R_SEL:]) for c, v in d.items()}
    j_star = min(sel, key=lambda c: sel[c])
    picked = sel[j_star] < -GAIN_MARGIN              # 선택표본 기대이득이 margin 넘게 음수일 때만
    d_pub = ev[j_star] if picked else 0.0            # 공개정보 결정의 무편향 기대이득
    return {"seed": i, "d_public": round(d_pub, 3), "picked": bool(picked),
            "j_star": list(j_star), "sel_gain": round(sel[j_star], 3),
            "eval_gain": round(ev[j_star], 3),
            "oracle_mean": round(fmean(oracle_r), 3),
            "optimistic_min_eval": round(min(ev.values()), 3),
            "n_cands": len(cands)}


# ---------------------------------------------------------------- Part B (v2 — 검증 정정)
def part_b_seed(i: int, every_s=900.0, window_s=1800.0) -> list[dict]:
    """SF 실행 중 스냅샷 C± ↔ **스냅샷 시각 기준** 후속 window 실현비용 (검증 정정 v2).

    정정: ①창 = [t, t+window) 구간비용 합(버킷 근사 아님 — 시작시각 귀속, 경계 걸침
    오차는 구간 1개 이내 한계 명시) ②catch-up 중복 제거(next_snap = t+every)
    ③말미 창 잘림 검열 — t+window > 에피소드 끝이면 표본 제외 ④seed 태그 저장.
    """
    from ..domain.enums import InformationLevel
    from ..integrated.baselines import ResolverPolicy, ServiceFirstSPTPreference, _apply, _wait_of
    from ..integrated.candidates import CandidateGenerator
    from ..integrated.cost_config import RewardCalculator
    rc = RewardCalculator.numeraire_v1()
    rows = []
    for cell, base in ((CELL_A, BASE_A), (CELL_B, BASE_B)):
        sim = _sim_of(_scen103(base + i, cell))
        pol = ResolverPolicy(ServiceFirstSPTPreference(), "SF")
        gen = CandidateGenerator()
        snaps, intervals = [], []                 # intervals = (start, cost)
        dp = sim.run_until_decision()
        sim.cost.cut()
        last_b, next_snap = sim.now, every_s
        while dp is not None:
            gen_by = {c: gen.generate(sim, c, InformationLevel.PRE_ADVICE)
                      for c in dp.crane_ids}
            raw = sim.cost.cut()
            intervals.append((last_b, rc.cost_for(interval_start_s=last_b,
                                                  interval_end_s=sim.now, raw=raw,
                                                  risk_max=0.0).total_normalized))
            last_b = sim.now
            if sim.now >= next_snap:
                snaps.append((sim.now, block_congestion(sim)))
                next_snap = sim.now + every_s     # catch-up 중복 제거
            try:
                _apply(sim, pol.decide(sim, dp, gen_by))
            except Exception:
                _apply(sim, {c: _wait_of(gen_by[c]) for c in dp.crane_ids})
            dp = sim.run_until_decision()
        raw = sim.cost.cut()
        intervals.append((last_b, rc.cost_for(interval_start_s=last_b, interval_end_s=sim.now,
                                              raw=raw, risk_max=0.0).total_normalized))
        end = sim.now
        for t, c in snaps:
            if t + window_s > end:                # 창 잘림 검열 — 표본 제외
                continue
            fut = sum(cost for s, cost in intervals if t <= s < t + window_s)
            rows.append({"seed": i, "cell": cell[0], "t": round(t, 1),
                         "future_cost": round(fut, 3), **c})
    return rows


def _spearman(xs, ys) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda k: v[k])
        rk = [0.0] * len(v)
        for pos, k in enumerate(order):
            rk[k] = pos
        return rk
    rx, ry = rank(xs), rank(ys)
    mx, my = fmean(rx), fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def run(shard: int, n_shards: int):
    OUT.mkdir(parents=True, exist_ok=True)
    rows_a, rows_b = [], []
    for i in range(N_SEEDS):
        if i % n_shards != shard:
            continue
        ra = part_a_seed(i)
        rows_a.append(ra)
        print(f"[A seed {i}] d_public={ra['d_public']:+.2f} picked={ra['picked']} "
              f"sel={ra['sel_gain']:+.2f} eval={ra['eval_gain']:+.2f} "
              f"oracle={ra['oracle_mean']:+.2f}", flush=True)
        rb = part_b_seed(i)
        rows_b.extend(rb)
        print(f"[B seed {i}] snapshots={len(rb)}", flush=True)
    p = OUT / f"shard{shard}.json"
    p.write_text(json.dumps({"a": rows_a, "b": rows_b}, ensure_ascii=False, indent=1),
                 encoding="utf-8")
    print(f"saved {p}")


def _rho_by(rows_b, key_filter=None) -> dict:
    sub = [r for r in rows_b if key_filter is None or r["cell"] == key_filter]
    if len(sub) < 3:
        return {}
    fut = [r["future_cost"] for r in sub]
    return {k: round(_spearman([r[k] for r in sub], fut), 3)
            for k in ("C_minus", "C_zero", "C_plus")}


def merge(n_shards: int) -> dict:
    rows_a, rows_b = [], []
    for s in range(n_shards):
        d = json.loads((OUT / f"shard{s}.json").read_text(encoding="utf-8"))
        rows_a.extend(d["a"])
        rows_b.extend(d["b"])
    # Part B v2 재수집분이 있으면 그것을 사용 (검증 정정 — shard 의 구버전 b 대체)
    pb2 = OUT / "partb_v2.json"
    if pb2.exists():
        rows_b = json.loads(pb2.read_text(encoding="utf-8"))
    rows_a.sort(key=lambda r: r["seed"])
    assert [r["seed"] for r in rows_a] == list(range(N_SEEDS)), "시드 커버리지 불완전"
    dp = [r["d_public"] for r in rows_a]
    # 참고 변형(0-임계 KEEP — 검증 정정: 주판정은 margin 규칙, 이건 병기 보고)
    dp0 = [(r["eval_gain"] if r["sel_gain"] < 0 else 0.0) for r in rows_a]
    res = {"part_a": {"rows": rows_a, "d_public_ci": _ci_t(dp),
                      "d_public_ci_zero_threshold_ref": _ci_t(dp0),
                      "n_transfer": sum(1 for r in rows_a if r["picked"]),
                      "oracle_mean": round(fmean(r["oracle_mean"] for r in rows_a), 3),
                      "optimistic_min_eval_mean":
                          round(fmean(r["optimistic_min_eval"] for r in rows_a), 3)},
           "part_b": {"n_snapshots": len(rows_b), "source": "v2" if pb2.exists() else "shard",
                      "spearman_pooled": _rho_by(rows_b),
                      "spearman_high": _rho_by(rows_b, "high"),
                      "spearman_mid": _rho_by(rows_b, "mid")},
           "prereg": "A: pooled d_public CI(t df=5) 상한<0 → YR-102 GO / 아니면 '이 MC 규칙"
                     "(R_SEL=5) 한계' — 공개정보 상금 부재의 증명 아님(비대칭 명시). "
                     "B: 셀별 ρ 병기·C_zero ρ>0 여부·최고 부호 보고(확정은 신규 seed 재현 후)."}
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"G1 d_public CI={res['part_a']['d_public_ci']} (0-thr ref {res['part_a']['d_public_ci_zero_threshold_ref']}) "
          f"transfers={res['part_a']['n_transfer']}/{N_SEEDS} "
          f"oracle={res['part_a']['oracle_mean']}\n"
          f"spearman pooled={res['part_b']['spearman_pooled']} "
          f"high={res['part_b']['spearman_high']} mid={res['part_b']['spearman_mid']}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--partb-v2", action="store_true", help="검증 정정판 Part B 만 재수집")
    a = ap.parse_args()
    if a.partb_v2:
        OUT.mkdir(parents=True, exist_ok=True)
        rows = []
        for i in range(N_SEEDS):
            rows.extend(part_b_seed(i))
            print(f"[B2 seed {i}] cum={len(rows)}", flush=True)
        (OUT / "partb_v2.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    elif a.merge:
        merge(a.n_shards)
    else:
        run(a.shard, a.n_shards)
    print("DONE")
