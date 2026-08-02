"""YR-142 — v4-B-2: one-shot PREPOSITION 재발행 금지 (17·18차 감사 반영 정정판 v2, 동결).

■ 축 분리 (18차 감사): BOUND_REPO(결속 발행) ⊥ PREPO_ONE_SHOT(재발행 금지).
  비교군 4 (모두 같은 신규 대역):
    SF  수동 규칙 (bound=False)
    A   자유 위치조정 PPO — YR-140 체크포인트 재사용 (bound=False)
    B1  결속 PREPOSITION·반복 허용 — 신규 학습 (bound=True, one_shot=False)
    B2  결속 PREPOSITION·one-shot — 신규 학습 (bound=True, one_shot=True)
  one-shot 효과는 (B2−B1) 짝 대조로 **분리 보고**(성공 조건 아님 — 주장 근거 전용).

■ 대역 (18차: BASE+3500·+4000 산술 대역 폐기): seedbank.assign_band 시작 910000,
  과거 기록 지문 전수 제외, band.json 동결. 평가 시 이중 fail-fast(지문 재계산 일치 +
  정수 범위 + 과거 지문 교집합 0). 산술 offset 인자는 제거 — 기본값 실수 원천 봉쇄.

■ 판정 (YR-141 동결 7항 유지 + 강화 2 — 원정밀도, 반올림은 보고 출력 전용):
  J1 B2 전판 완주 100% ∧ backlog 0 ∧ **건전성(healthy) 전판** (강화)
  J2 B2 위치조정 계열 장악(>60%) 0
  J3 vs SF v2 비용 방향 <0 이 ≥2/3     J4 (B2−A) v2 짝 평균 ≤0 이 ≥2/3
  J5 (B2−A) 본선 초과분 ≤0 이 ≥2/3     J6 (B2−A) v1 전체비용 ≤0 이 ≥2/3
  J7 반복 실행 0 ∧ 만료 후 실행 0 ∧ **대역 전체 PREPO 실행 >0** (공허 통과 방지, 강화)

■ 허용 주장 한계 (17차): 정책 관측은 결속 작업 식별·ETA 를 보지 못한다 — 결과 주장은
  "작업 ID one-shot 마스크 규칙의 효과"까지. ETA 관측 배선은 별도 단일축.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from statistics import fmean

from ..integrated import candidates as cand_mod
from ..integrated.baselines import ResolverPolicy, ServiceFirstSPTPreference
from ..integrated.repro import repro_stamp
from ..integrated.seedbank import BandSpec, assign_band, independence_report, realization_hash
from .yr090_dense_vessel import BASE, CELLS
from .yr136_softplus_contract import _sim_contract
from .yr139_blockq_v4_ppo import train_one
from .yr141_bound_prepo import OUT140, TRAIN_SEEDS, _episode, _mk_ppo

OUT = Path("outputs/reports/yr142_prepo_enforce")
BAND_PATH = OUT / "band.json"
BAND_START, BAND_CEIL = 910_000, 920_000   # 미사용 정수 구간 (901k~905k·920k+ 는 이송 계열)
ARM_FLAGS = {"b1": (True, False), "b2": (True, True)}   # (BOUND_REPO, PREPO_ONE_SHOT)


# ------------------------------------------------------------------ 대역 (시드뱅크)
def _collect_past_hashes() -> set[str]:
    """outputs/reports 의 모든 JSON 에서 기록된 실현 지문(rz1:...)을 전수 수집.
    자기 자신(yr142_prepo_enforce/)은 제외 — 대역 검사가 자기 지문에 걸리지 않게."""
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
    band = assign_band(family="yr142-judgment",
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
    """평가 직전 fail-fast — 지문 재계산 일치·정수 범위·과거 지문 교집합 0."""
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


# ------------------------------------------------------------------ 학습 / probe
def train(ts: int, arm: str):
    b, o = ARM_FLAGS[arm]
    prev = cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT
    cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT = b, o
    try:
        return train_one(ts, out_root=OUT / arm)
    finally:
        cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT = prev


def probe_dup_removed() -> dict:
    """18차 감사 지시 — 중복 결속 조합 제외 규칙이 실제 공동후보 목록을 바꿨는지 실측.
    훈련 대역 전 시드(4셀×16)·argmax 로 B1/B2 구식 체크포인트 순회, 규칙 발화 계수."""
    rows = {}
    for arm, root, one_shot in (("b1", Path("outputs/reports/yr141_bound_prepo"), False),
                                ("b2", OUT, True)):
        fired = 0
        for ts in TRAIN_SEEDS[:1]:                       # 발화 여부가 목적 — 1 초기화 충분
            mk = _mk_ppo(root, ts)
            for cell in CELLS:
                for i in range(16):
                    r = _episode(cell, BASE[cell] + i, mk, bound=True, one_shot=one_shot)
                    fired += r["prepo_dup_removed"]
        rows[arm] = fired
    print(json.dumps({"probe_dup_removed": rows}, ensure_ascii=False))
    return rows


# ------------------------------------------------------------------ 평가 / 판정
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
        per_seed[ts] = {  # 전부 원정밀도 — 반올림은 report.md 전용 (17차 감사)
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
               and sum(x["prepo_exec_total"] for x in v) > 0}
    j["success"] = all(j.values())
    isolation = {ts: {k: per_seed[ts][k] for k in
                      ("B2_minus_B1_v2", "B2_minus_B1_berth", "B2_minus_B1_v1",
                       "B1_prepo_repeat", "B1_prepo_exec_total")} for ts in TRAIN_SEEDS}
    res = {"repro": repro_stamp(
               experiment="YR-142 v4-B-2 one-shot PREPOSITION — 4군 판정 (시드뱅크 대역)",
               seeds={"train": list(TRAIN_SEEDS), **{c: band.seeds[c] for c in CELLS}},
               profile_id="calibrated",
               prereg="축 분리 BOUND⊥ONE_SHOT·4군 SF/A/B1/B2·시드뱅크 대역(fail-fast)·"
                      "원정밀도 판정·J1 건전성 강화·J7 노출 유효 강화·(B2−B1) 분리 보고. "
                      "허용 주장 = one-shot 마스크 규칙 효과까지 (관측은 결속 작업·ETA "
                      "미노출 — 17차).",
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
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--train", type=int, default=0)
    ap.add_argument("--arm", choices=("b1", "b2"))
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.make_band:
        make_band()
    if a.probe:
        probe_dup_removed()
    if a.train:
        assert a.arm, "--train 은 --arm b1|b2 필수"
        train(a.train, a.arm)
    if a.eval:
        evaluate()
    print("DONE")
