"""YR-145 — 결속 PREPOSITION 상태 게이트 완결 (PLANNED 한정) + 재판정 (동결).

■ 유일 변경 (YR-142 v2 대비 단일축): iter_eta_reposition_jobs 상태 필터를
  status == PLANNED 한정으로 교체 — YR-142 기각 원인(예측 ETA 6209.7s vs 실제 도착
  5136.9s, 17.9분 조기 도착 후 RUNNING 작업에 발행) 봉쇄. 도착·배정·진행·완료 전부 소멸.

■ 계약 (YR-142 v2 `24c7c7c` 승계 + 19차 감사 추가):
  비교군 4 SF/A/B1/B2 · 시드뱅크 신규 대역(커서 910012+, 열람한 910000~910011 지문은
  수집 제외 집합에 자동 편입 — 자기 OUT 제외 규칙) · 판정값 완전 원정밀도(비용·행동비중
  포함, 반올림은 보고 전용) · 차단 계수 = (결정시점·크레인·작업) 고유 삼족.

■ 판정 7항 + 강화 3 (J7 에 19차 "one-shot 차단 발생 >0" 추가):
  J1 B2 전판 완주 100% ∧ backlog 0 ∧ 건전성     J2 위치조정 장악(>60%) 0
  J3 vs SF v2 <0 ≥2/3     J4 (B2−A) v2 ≤0 ≥2/3     J5 (B2−A) 본선 ≤0 ≥2/3
  J6 (B2−A) v1 ≤0 ≥2/3
  J7 반복 0 ∧ 만료 후 실행 0 ∧ PREPO 실행 >0 ∧ **one-shot 차단 발생 >0**
  (B2−B1) 분리 대조는 보고 전용 — YR-142 실측: 반복 제거 달성·비용 순효과 불일치 1/3.

■ 허용 주장 한계 (17차 동결 승계): "작업 ID one-shot 마스크 규칙의 효과"까지.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from statistics import fmean

from ..integrated import candidates as cand_mod
from ..integrated.baselines import ResolverPolicy, ServiceFirstSPTPreference
from ..integrated.repro import repro_stamp
from ..integrated.seedbank import BandSpec, assign_band, independence_report, realization_hash
from .yr090_dense_vessel import CELLS
from .yr136_softplus_contract import _sim_contract
from .yr139_blockq_v4_ppo import train_one
from .yr141_bound_prepo import OUT140, TRAIN_SEEDS, _episode, _mk_ppo
from .yr142_prepo_enforce import ARM_FLAGS, _collect_past_hashes as _collect_h142

OUT = Path("outputs/reports/yr145_status_gate")
BAND_PATH = OUT / "band.json"
BAND_START, BAND_CEIL = 910_012, 920_000


def _collect_past_hashes() -> set[str]:
    """yr142 수집기 재사용하되 제외 대상을 자기(OUT)로 교체 — yr142 의 열람 지문
    (910000~910011)은 이제 수집 대상이 되어 자동으로 금지 집합에 들어간다."""
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


def make_band() -> BandSpec:
    exclude = _collect_past_hashes()
    band = assign_band(family="yr145-judgment",
                       cells={c: None for c in CELLS}, n=3,
                       generate=lambda key, _p, seed: _sim_contract(key, seed).scenario,
                       exclude=exclude, start_seed=BAND_START)
    for ss in band.seeds.values():
        for s in ss:
            assert BAND_START <= s < BAND_CEIL, f"대역 정수 이탈: {s}"
    rep = independence_report(band, forbidden={"past-recorded": exclude})
    assert rep["ok"], f"대역 독립성 실패: {rep}"
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    OUT.mkdir(parents=True, exist_ok=True)
    BAND_PATH.write_text(json.dumps(
        {**band.freeze_json(), "independence": rep, "n_excluded_hashes": len(exclude),
         "created_commit": head}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[band] {sum(len(v) for v in band.seeds.values())} seeds frozen "
          f"(exclude {len(exclude)} hashes) → {BAND_PATH}")
    return band


def load_band_checked() -> BandSpec:
    d = json.loads(BAND_PATH.read_text(encoding="utf-8"))
    band = BandSpec(family=d["family"], seeds=d["seeds"], hashes=d["realization_hashes"])
    for cell, ss in band.seeds.items():
        for s, h_frozen in zip(ss, band.hashes[cell]):
            assert BAND_START <= s < BAND_CEIL, f"대역 정수 이탈: {cell}:{s}"
            h_now = realization_hash(_sim_contract(cell, s).scenario)
            assert h_now == h_frozen, f"지문 불일치 {cell}:{s} — {h_now} != {h_frozen}"
    rep = independence_report(band, forbidden={"past-recorded": _collect_past_hashes()})
    assert rep["ok"], f"대역 독립성 실패: {rep}"
    return band


def train(ts: int, arm: str):
    b, o = ARM_FLAGS[arm]
    prev = cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT
    cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT = b, o
    try:
        return train_one(ts, out_root=OUT / arm)
    finally:
        cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT = prev


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def evaluate() -> dict:
    band = load_band_checked()
    eval_eps = [(c, s) for c in CELLS for s in band.seeds[c]]
    print(f"[eval] SF {len(eval_eps)}", flush=True)
    sf = [_episode(c, s, lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF"),
                   bound=False) for c, s in eval_eps]
    arms_cfg = {"A": (OUT140, False, False), "B1": (OUT / "b1", True, False),
                "B2": (OUT / "b2", True, True)}
    rows: dict[str, list[dict]] = {}
    ckpt_sha = {}
    for arm, (root, bound, one_shot) in arms_cfg.items():
        for ts in TRAIN_SEEDS:
            print(f"[eval] {arm}:{ts}", flush=True)
            ckpt_sha[f"{arm}:{ts}"] = _sha(root / f"ppo_s{ts}" / "net.pt")
            mk = _mk_ppo(root, ts)
            rows[f"{arm}:{ts}"] = [_episode(c, s, mk, bound=bound, one_shot=one_shot)
                                   for c, s in eval_eps]
    per_seed = {}
    for ts in TRAIN_SEEDS:
        A, B1, B2 = rows[f"A:{ts}"], rows[f"B1:{ts}"], rows[f"B2:{ts}"]
        per_seed[ts] = {  # 전부 원정밀도 — 반올림은 report.md 전용
            "B2_vs_SF_v2": fmean(b["v2_total"] - s["v2_total"] for b, s in zip(B2, sf)),
            "B2_minus_A_v2": fmean(b["v2_total"] - a["v2_total"] for b, a in zip(B2, A)),
            "B2_minus_A_berth": fmean(b["berth_over_min"] - a["berth_over_min"]
                                      for b, a in zip(B2, A)),
            "B2_minus_A_v1": fmean(b["v1_total"] - a["v1_total"] for b, a in zip(B2, A)),
            "B2_minus_B1_v2": fmean(b["v2_total"] - x["v2_total"] for b, x in zip(B2, B1)),
            "B2_minus_B1_berth": fmean(b["berth_over_min"] - x["berth_over_min"]
                                       for b, x in zip(B2, B1)),
            "B2_minus_B1_v1": fmean(b["v1_total"] - x["v1_total"] for b, x in zip(B2, B1)),
            "compl_min": min(r["compl"] for r in B2),
            "backlog_max": max(r["backlog"] for r in B2),
            "healthy_all": all(r["healthy"] for r in B2),
            "repo_dom": sum(1 for r in B2 if r["shares"].get("REPOSITION", 0) > 0.60),
            "repo_share_mean": fmean(r["shares"].get("REPOSITION", 0) for r in B2),
            "prepo_repeat": sum(r["prepo_repeat"] for r in B2),
            "prepo_expired": sum(r["prepo_expired"] for r in B2),
            "prepo_exec_total": sum(r["prepo_exec_total"] for r in B2),
            "prepo_offered_total": sum(r["prepo_offered"] for r in B2),
            "prepo_blocked_total": sum(r["prepo_blocked"] for r in B2),
            "prepo_blocked_jobs_total": sum(r["prepo_blocked_jobs"] for r in B2),
            "B1_prepo_repeat": sum(r["prepo_repeat"] for r in B1),
            "B1_prepo_exec_total": sum(r["prepo_exec_total"] for r in B1)}
    v = list(per_seed.values())
    j = {"J1": all(x["compl_min"] >= 1.0 and x["backlog_max"] == 0 and x["healthy_all"]
                   for x in v),
         "J2": all(x["repo_dom"] == 0 for x in v),
         "J3": sum(1 for x in v if x["B2_vs_SF_v2"] < 0) >= 2,
         "J4": sum(1 for x in v if x["B2_minus_A_v2"] <= 0) >= 2,
         "J5": sum(1 for x in v if x["B2_minus_A_berth"] <= 0) >= 2,
         "J6": sum(1 for x in v if x["B2_minus_A_v1"] <= 0) >= 2,
         "J7": all(x["prepo_repeat"] == 0 and x["prepo_expired"] == 0 for x in v)
               and sum(x["prepo_exec_total"] for x in v) > 0
               and sum(x["prepo_blocked_total"] for x in v) > 0}   # 19차: 차단 실발생
    j["success"] = all(j.values())
    isolation = {ts: {k: per_seed[ts][k] for k in
                      ("B2_minus_B1_v2", "B2_minus_B1_berth", "B2_minus_B1_v1",
                       "B1_prepo_repeat", "B1_prepo_exec_total")} for ts in TRAIN_SEEDS}
    res = {"repro": repro_stamp(
               experiment="YR-145 상태 게이트 PLANNED 한정 — 4군 재판정 (시드뱅크 신규 대역)",
               seeds={"train": list(TRAIN_SEEDS), **{c: band.seeds[c] for c in CELLS}},
               profile_id="calibrated",
               prereg="유일 변경 = iter_eta_reposition_jobs 상태 필터 PLANNED 한정. "
                      "그 외 YR-142 v2(24c7c7c) 승계 + 19차: 원정밀도·고유 차단 삼족·"
                      "J7 차단>0. 허용 주장 = one-shot 마스크 규칙까지.",
               extra={"n_eval": len(eval_eps), "band_digest": json.loads(
                          BAND_PATH.read_text(encoding="utf-8"))["digest"],
                      "arm_flags": {k: list(vv) for k, vv in ARM_FLAGS.items()},
                      "ckpt_sha256_16": ckpt_sha}),
           "band_fingerprints": {f"{c}:{s}": h for c in CELLS
                                 for s, h in zip(band.seeds[c], band.hashes[c])},
           "sf": sf, "arms": rows,
           "judgment": {**j, "per_seed": {str(k): vv for k, vv in per_seed.items()}},
           "isolation_B2_minus_B1": {str(k): vv for k, vv in isolation.items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(json.dumps(j, ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-band", action="store_true")
    ap.add_argument("--train", type=int, default=0)
    ap.add_argument("--arm", choices=("b1", "b2"))
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.make_band:
        make_band()
    if a.train:
        assert a.arm, "--train 은 --arm b1|b2 필수"
        train(a.train, a.arm)
    if a.eval:
        evaluate()
    print("DONE")
