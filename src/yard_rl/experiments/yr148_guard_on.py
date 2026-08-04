"""YR-148 — 안전장치 ON 재판정: C0+guard vs C1+guard "배포 구성 전체 효과" (동결).

■ 주장 명칭(26차): PREPOSITION 순수 효과가 아니라 **배포 구성 전체 효과** — guard
  개입이 정책마다 다를 수 있으므로. Δ = (C0+guard) − (C1+guard), 방향 YR-143 승계.
■ 주분석(26차 동결): **다중방향 부트스트랩** — 학습초기화 8·평가 시나리오 44 를 각각
  복원 추출(B=10,000·rng 148), 통계량 = 추출 격자 평균. 단측 95% 상한 U(95백분위)·
  하한 L(5백분위). **POWER_FAIL**: 지표별 (U−L)/2 > δ_m 이면 그 지표 판정 불가
  ("효과 없음" 아님). δ 승계(v2 1.0/본선 3.0분/v1 2.0·가정값)·민감도 ×0.5/×1.5.
■ 4분류 + BOTH_GUARD_FAIL + 개입률(상한 1% — 초과 시 "정책 미완" 태그) 병행.
■ 산출물에 **허가증 원장 전량 저장**(개입별 원래/대체 행동·wake 원인 포함 — 26차).
■ 재학습 없음(YR-143 확증 8쌍 재사용)·신규 대역 44판(910500+ — 910400대 열람 편입).
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path
from statistics import fmean

from ..integrated.seedbank import (BandSpec, assign_band, independence_report,
                                   realization_hash)
from .yr090_dense_vessel import CELLS
from .yr136_softplus_contract import _sim_contract
from .yr143_no_repo import CONFIRM_TS, DELTA
from .yr146_deploy_guard import CAP_RATE, IV_KEYS, _episode_guard

OUT = Path("outputs/reports/yr148_guard_on")
BAND_PATH = OUT / "band.json"
BAND_START, BAND_N = 910_500, 11            # 44 시나리오 (YR-143 표본 산정 승계)
BOOT_B, BOOT_SEED = 10_000, 148


def _collect_hashes() -> set[str]:
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


def make_band():
    exclude = _collect_hashes()
    band = assign_band(family="yr148-judgment", cells={c: None for c in CELLS},
                       n=BAND_N,
                       generate=lambda key, _p, seed: _sim_contract(key, seed).scenario,
                       exclude=exclude, start_seed=BAND_START)
    rep = independence_report(band, forbidden={"past-recorded": exclude})
    assert rep["ok"], rep
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    OUT.mkdir(parents=True, exist_ok=True)
    BAND_PATH.write_text(json.dumps(
        {**band.freeze_json(), "independence": rep, "n_excluded_hashes": len(exclude),
         "created_commit": head}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[band] {sum(len(v) for v in band.seeds.values())} seeds frozen")


def _boot_ci(diff, rng):
    """다중방향 부트스트랩 — diff[init][scen], 초기화·시나리오 복원 추출 격자 평균."""
    n_i, n_s = len(diff), len(diff[0])
    stats = []
    for _ in range(BOOT_B):
        ii = [rng.randrange(n_i) for _ in range(n_i)]
        ss = [rng.randrange(n_s) for _ in range(n_s)]
        stats.append(fmean(diff[i][s] for i in ii for s in ss))
    stats.sort()
    return {"mean": fmean(fmean(r) for r in diff),
            "L": stats[int(0.05 * BOOT_B)], "U": stats[int(0.95 * BOOT_B)]}


def evaluate() -> dict:
    from ..integrated.repro import repro_stamp
    d = json.loads(BAND_PATH.read_text(encoding="utf-8"))
    band = BandSpec(family=d["family"], seeds=d["seeds"], hashes=d["realization_hashes"])
    for cell, ss in band.seeds.items():
        for s, h in zip(ss, band.hashes[cell]):
            assert realization_hash(_sim_contract(cell, s).scenario) == h, f"{cell}:{s}"
    eval_eps = [(c, s) for c in CELLS for s in band.seeds[c]]
    rows = {}
    for arm in ("c0", "c1"):
        for ts in CONFIRM_TS:
            print(f"[eval] {arm}+guard:{ts}", flush=True)
            rows[f"{arm}:{ts}"] = [_episode_guard(c, s, arm, ts, guard_on=True)
                                   for c, s in eval_eps]
    guards, iv = {}, {}
    for arm in ("c0", "c1"):
        eps = [e for ts in CONFIRM_TS for e in rows[f"{arm}:{ts}"]]
        guards[arm] = {"compl_min": min(e["compl"] for e in eps),
                       "backlog_max": max(e["backlog"] for e in eps),
                       "healthy_all": all(e["healthy"] for e in eps),
                       "ok": (min(e["compl"] for e in eps) >= 1.0
                              and max(e["backlog"] for e in eps) == 0
                              and all(e["healthy"] for e in eps))}
        joint = sum(e["guard"]["joint"] for e in eps)
        n_iv = sum(sum(e["guard"][k] for k in IV_KEYS) for e in eps)
        iv[arm] = {"joint": joint, "interventions": n_iv,
                   "rate": n_iv / joint if joint else 0.0,
                   "rate_ok": (n_iv / joint if joint else 0.0) <= CAP_RATE}
    metrics = {"v2": "v2_total", "berth": "berth_over_min", "v1": "v1_total"}
    rng = random.Random(BOOT_SEED)
    stats, power_fail = {}, {}
    for m, key in metrics.items():
        diff = [[rows[f"c0:{ts}"][j][key] - rows[f"c1:{ts}"][j][key]
                 for j in range(len(eval_eps))] for ts in CONFIRM_TS]
        st = _boot_ci(diff, rng)
        stats[m] = {**st, "delta": DELTA[m], "half_width": (st["U"] - st["L"]) / 2}
        power_fail[m] = stats[m]["half_width"] > DELTA[m]

    def classify(scale):
        dl = {m: DELTA[m] * scale for m in DELTA}
        if not guards["c0"]["ok"] and not guards["c1"]["ok"]:
            return "BOTH_GUARD_FAIL"
        if not guards["c0"]["ok"]:
            return "C0_GUARD_FAIL(C1 유지)"
        if not guards["c1"]["ok"]:
            return "C1_GUARD_FAIL"
        usable = [m for m in DELTA if not power_fail[m]]
        if any(power_fail.values()):
            return f"POWER_FAIL({[m for m in DELTA if power_fail[m]]})"
        if (stats["v2"]["U"] < 0 and stats["berth"]["U"] < dl["berth"]
                and stats["v1"]["U"] < dl["v1"]):
            return "REMOVE_SUPERIOR"
        if all(stats[m]["U"] < dl[m] for m in DELTA):
            return "REMOVE_NONINFERIOR"
        if any(stats[m]["L"] > dl[m] for m in DELTA):
            return "KEEP_C1_MEANINGFUL"
        return "INCONCLUSIVE"

    verdict = {"판정": classify(1.0), "민감도_δx0.5": classify(0.5),
               "민감도_δx1.5": classify(1.5),
               "정책미완_태그": {a: not iv[a]["rate_ok"] for a in iv}}
    res = {"repro": repro_stamp(
               experiment="YR-148 — C0+guard vs C1+guard 배포 구성 전체 효과 재판정",
               seeds={"train": list(CONFIRM_TS), **{c: band.seeds[c] for c in CELLS}},
               profile_id="calibrated",
               prereg="배포 구성 전체 효과(순수 PREPO 효과 주장 금지)·다중방향 부트스트랩 "
                      "B=10000 rng148·POWER_FAIL=(U−L)/2>δ·4분류+BOTH_GUARD_FAIL·"
                      "개입률 ≤1% 태그·δ 승계·허가증 원장 전량 저장·재학습 없음",
               extra={"band_digest": d["digest"]}),
           "guards": guards, "iv": iv, "stats": stats, "power_fail": power_fail,
           "verdict": verdict, "arms": rows}
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(json.dumps({**verdict, "iv": iv,
                      **{m: {k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in s.items()} for m, s in stats.items()}},
                     ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-band", action="store_true")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.make_band:
        make_band()
    if a.eval:
        evaluate()
    print("DONE")
