"""YR-168 ② — **눈가림 파일럿**: 짝차이의 잡음 크기만 잰다 (사전등록 §E 동결본).

목적은 단 하나 — 본 판정에 필요한 **표본 수를 산출**하기 위한 표준편차 σ 다.
평균·부호·중앙값·개별값·순위상관은 **개봉하지 않는다**(봉인 파일에 넣고 해시만 공개).

짝 구성 (사전등록 §D):
  · 기저 세계 = 전건 KEEP (판매 0건)
  · 처치 세계 = **피크대 [11,16)시에서 계산 기준선이 처음 발행하는 공간 판매 1건**만
    집행, 그 외 전건 KEEP. 동순위는 (t, job_id, dst) 사전순.
  · 실현 이득 G = Φ(기저) − Φ(처치)  — Φ 는 전역(수신 블록 부담·주행 포함)
  · 하드 가드: 두 세계 모두 3,600 완주·누락 0·취소 0, 처치 직전 상태 digest 동일

집행 정책은 자격을 통과한 무대와 같은 **SF(ServiceFirstSPT) 결정론**이다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import pstdev, stdev

from ..integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference,
                                    _apply, _wait_of)
from ..integrated.candidates import CandidateGenerator
from ..integrated.multiblock import MultiBlockTerminal
from ..integrated.policy_config import LEGACY_DEFAULT
from ..integrated.profiles import build_h21_profile
from ..integrated.repro import code_dirty
from ..integrated.sell_review import (ANNOUNCE_LEAD_S,
                                      UnifiedSellOrchestrator)
from ..integrated.terminal_stream import (DIURNAL_DAY_TOTAL, OBS_24H,
                                          ScheduledAnnouncer, TerminalStreamParams,
                                          admission_epochs, build_diurnal,
                                          ensure_time_ledger)
from ..integrated.yard_layout import terminal_layout
from .yr088_joint_rl import LEVEL
from .yr149_load_cells import _sim_from
from .yr150_h21_pilot import _git, _sha256
from .yr151_transfer_ppo import load_kf, phi_terminal
from .yr167_observers import cancelled_external

OUT = Path("outputs/reports/yr168_pilot")
PREREG = Path(".claude/docs/strategy-history/"
              "2026-08-11-YR-168-공간판매-재판정-사전등록.md")
GATE = Path("outputs/reports/yr153_research_gates/current_gate.json")
SEED_BASE = 8_300_000            # 동결 — 자격·관찰 대역과 교집합 0
N_PILOT = 12                     # 동결 — 파일럿 쌍 수
PEAK_S = (11 * 3600.0, 16 * 3600.0)   # 동결 — 피크대 [11,16)시
DELTA_DESIGN = 0.5               # 동결 — 표본설계용 가정 효과(합격선 아님)
N_MIN, N_CAP = 100, 300          # 동결 — 기본 표본 하한·예산 상한
Z_A, Z_B = 1.645, 0.842          # 단측 α=0.05 · 검정력 80%


class EstimatorBaseline:
    """**동결 계산 기준선** — 판매 여부·수신 블록은 전적으로 견적(NetGain)이 정한다.

    정책의 역할은 블록마다 후보 1건을 **지명**하는 것뿐이고, 지명 규칙은 결정론이다
    (공개 ETA 최소, 동순위는 job_id 사전순). 압력 임계 같은 사전 필터를 두지 않는
    이유는 Q2 가 "견적이 이로운 판매를 골라내는가"를 묻기 때문이다 — 필터를 앞에
    두면 무엇이 골라낸 것인지 구분되지 않는다.

    `CalcGreedy`(sell_review.py)는 `decide_space`/`decide_time` 인터페이스라
    UnifiedSellOrchestrator(`policy.decide(mbt, src, cands, t)`)와 맞지 않는다.
    """

    mode = "live"

    def decide(self, mbt, src: str, cands: list, t: float) -> str | None:
        return min(cands, key=lambda c: (c[1], c[0]))[0] if cands else None


def _stamp() -> dict:
    return {"code_commit": _git("rev-parse", "HEAD"),
            "code_dirty": bool(code_dirty()),
            "prereg_sha256": _sha256(PREREG) if PREREG.exists() else None,
            "gate_sha256": _sha256(GATE) if GATE.exists() else None}


def _state_digest(mbt) -> str:
    """처치 직전 상태 해시 — 두 세계가 같은 지점에 있었음을 증명한다."""
    import hashlib
    h = hashlib.sha256()
    for bid, sim in sorted(mbt.blocks.items()):
        h.update(f"|B{bid}|{round(sim.clock, 6)}".encode())
        for jid, j in sorted(sim.jobs.items()):
            h.update(f"|{jid}|{j.status.name}|{j.actual_gate_in}|"
                     f"{j.actual_block_arrival}|{j.flow.name}".encode())
        tl = sim.time_ledger
        for jid in sorted(tl.records):
            r = tl.records[jid]
            h.update(f"|L{jid}|{r.gate_in}|{r.block_arrival}|"
                     f"{r.service_start}|{r.job_done}|{r.gate_out}".encode())
    return h.hexdigest()


def _env(seed: int):
    prof, layout = build_h21_profile(), terminal_layout()
    params = TerminalStreamParams(load_4h=DIURNAL_DAY_TOTAL)
    built = build_diurnal(prof, seed, obs=OBS_24H, layout=layout, params=params,
                          background_seed=seed)
    mbt = MultiBlockTerminal(
        {b: ensure_time_ledger(_sim_from(s, prof))
         for b, s in built["scenarios"].items()},
        extra_review_epochs=admission_epochs(OBS_24H))
    ann = ScheduledAnnouncer(built["schedule"], lead_s=ANNOUNCE_LEAD_S,
                             end_s=built["sim_end_s"])
    return built, mbt, ann, layout


def _exec_policy():
    pol = ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    gens: dict[int, CandidateGenerator] = {}

    def policy(sim, dp):
        g = gens.setdefault(id(sim), CandidateGenerator(config=LEGACY_DEFAULT))
        gb = {c: g.generate(sim, c, LEVEL) for c in dp.crane_ids}
        try:
            _apply(sim, pol.decide(sim, dp, gb))
        except Exception:
            _apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})
    return policy


def propose(seed: int) -> dict | None:
    """dry-run 1회 — 피크대 첫 공간 판매 제안 1건 (결정론 선택)."""
    built, mbt, ann, layout = _env(seed)
    orch = UnifiedSellOrchestrator(EstimatorBaseline(), layout, load_kf(),
                                   dry_run=True)

    def review(m, t):
        ann.review(m, t)
        orch.review(m, t)

    mbt.run(_exec_policy(), review_fn=review)
    rows = [e for e in orch.ledger
            if e.get("decision") == "DRY_WOULD_COMMIT" and e.get("axis") == "SPACE"
            and PEAK_S[0] <= e["t"] < PEAK_S[1]]
    if not rows:
        return None
    rows.sort(key=lambda e: (e["t"], e["job_id"], str(e.get("dst"))))
    e = rows[0]
    return {"t": e["t"], "job_id": e["job_id"], "src": e["src"],
            "dst": e.get("dst"), "delta_j": e.get("delta_j"),
            "n_peak_space_proposals": len(rows),
            "n_night_proposals": sum(
                1 for x in orch.ledger
                if x.get("decision") == "DRY_WOULD_COMMIT"
                and 2 * 3600.0 <= x["t"] < 8 * 3600.0)}


def _world(seed: int, treat: dict, execute: bool) -> dict:
    """한 세계를 끝까지 돌린다. execute=True 면 처치 시각에 판매 1건만 집행.

    두 세계 모두 **같은 지점(투입 직후)** 에서 digest 를 뜬다 — 그 뒤 처치 세계만
    이송을 부른다. 투입은 개방 루프라 두 세계가 동일하다.
    """
    built, mbt, ann, layout = _env(seed)
    st: dict = {"digest": None, "ok": None}

    def review(m, t):
        ann.review(m, t)
        if st["digest"] is None and abs(t - treat["t"]) < 1e-6:
            st["digest"] = _state_digest(m)
            if execute:
                st["ok"] = bool(m.try_pre_gate_transfer(
                    treat["job_id"], treat["dst"],
                    travel_s=layout.gate_to_block_s(treat["dst"])))

    mbt.run(_exec_policy(), review_fn=review)
    rows = [r for sim in mbt.blocks.values() for r in sim.time_ledger.records.values()]
    return {"phi": phi_terminal(mbt, OBS_24H.observe_s),
            "n_registered": len(rows),
            "n_completed": sum(1 for r in rows if r.gate_out is not None),
            "n_plan": len(built["schedule"]),
            "n_skips": sum(1 for e in ann.ledger if e["event"] in ("SKIP", "SKIP_TAIL")),
            "n_cancelled": cancelled_external(mbt),
            "digest": st["digest"], "transfer_ok": st["ok"]}


def pair(idx: int) -> dict:
    seed = SEED_BASE + idx
    prop = propose(seed)
    if prop is None:
        return {"idx": idx, "seed": seed, "valid": False,
                "reason": "피크대 공간 판매 제안 없음"}
    base = _world(seed, prop, execute=False)
    treat = _world(seed, prop, execute=True)
    guards = {
        "base_complete": base["n_completed"] == base["n_plan"] == base["n_registered"],
        "treat_complete": treat["n_completed"] == treat["n_plan"] == treat["n_registered"],
        "no_skip": base["n_skips"] == 0 and treat["n_skips"] == 0,
        "no_cancel": base["n_cancelled"] == 0 and treat["n_cancelled"] == 0,
        "digest_equal": (base["digest"] is not None
                         and base["digest"] == treat["digest"]),
        "transfer_executed": bool(treat["transfer_ok"]),
    }
    return {"idx": idx, "seed": seed, "valid": all(guards.values()),
            "guards": guards, "proposal": prop,
            "G": base["phi"] - treat["phi"],
            "base": {k: v for k, v in base.items() if k != "digest"},
            "treat": {k: v for k, v in treat.items() if k != "digest"},
            "stamp": _stamp()}


def summarize() -> dict:
    """**눈가림 합산** — σ·유효 쌍 수·산출 N 만 공개. 평균·부호·개별값은 봉인."""
    cells = [json.loads((OUT / f"pair{i:03d}.json").read_text(encoding="utf-8"))
             for i in range(N_PILOT) if (OUT / f"pair{i:03d}.json").exists()]
    ok = [c for c in cells if c.get("valid")]
    gs = [c["G"] for c in ok]
    if len(gs) < 3:
        raise RuntimeError(f"유효 쌍 {len(gs)} < 3 — σ 추정 불가")
    sigma = stdev(gs)
    n_req = int(-(-((Z_A + Z_B) ** 2 * sigma ** 2) // (DELTA_DESIGN ** 2)))
    n_final = max(N_MIN, n_req)
    sealed = OUT / "sealed_raw.json"
    sealed.write_text(json.dumps(
        {"note": "봉인 — 본 판정 사전등록 동결 전 열람 금지", "G": gs,
         "pairs": ok}, ensure_ascii=False, indent=1), encoding="utf-8")
    res = {
        "task": "YR-168-pilot", "blinded": True,
        "stamp": _stamp(),
        "n_pairs_run": len(cells), "n_pairs_valid": len(ok),
        "invalid": [{"idx": c["idx"], "reason": c.get("reason") or c.get("guards")}
                    for c in cells if not c.get("valid")],
        "sigma_paired_difference": round(sigma, 4),
        "sigma_population": round(pstdev(gs), 4),
        "delta_design": DELTA_DESIGN,
        "n_required_raw": n_req,
        "n_final": n_final,
        "budget_cap": N_CAP,
        "verdict": ("POWER_FAIL — 산출 표본이 예산 상한 초과" if n_final > N_CAP
                    else f"진행 가능 — 본 판정 표본 {n_final}쌍"),
        "sealed_raw_sha256": _sha256(sealed),
        "disclosure_note": "평균·부호·중앙값·개별 G·순위상관은 개봉하지 않았다 "
                           "(사전등록 §E). 이 파일에는 잡음 크기만 있다.",
        "peak_proposals_per_seed": [c["proposal"]["n_peak_space_proposals"] for c in ok],
        "night_proposals_per_seed": [c["proposal"]["n_night_proposals"] for c in ok],
    }
    p = OUT / "pilot_blinded.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (p.with_suffix(".json.sha256")).write_text(_sha256(p) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("stamp", "invalid")}, ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, default=None)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.summarize:
        summarize()
    elif a.pair is not None:
        r = pair(a.pair)
        (OUT / f"pair{a.pair:03d}.json").write_text(
            json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        # 눈가림 — G 는 화면에 찍지 않는다
        print(json.dumps({"idx": r["idx"], "valid": r["valid"],
                          "guards": r.get("guards"),
                          "reason": r.get("reason")}, ensure_ascii=False))
    print("DONE")
