"""YR-151 — shadow dry-run **실측 검증** (0B 직전 관문 — spec 순서 1).

계약: shadow(구경만 하되 resolver 검증은 통과)는 **본 실행을 단 1바이트도 바꾸지
않아야** 한다. 검증 방식 = 같은 시드에서 두 에피소드를 돌려 시간 장부를 전수 대조:
  기준 런: 판매 계층 없음(투입 컨트롤러만)
  shadow 런: 투입 + PpoSellPolicy(mode="shadow") + dry_run resolver
  S1 본 실행 불변: 두 런의 (작업 → 블록·게이트인·도착·게이트아웃) 전수 일치
  S2 배선 흐름: shadow trail ≥1 · would-commit ≥1 · critic 입력 비영(수집기 배선)
  S3 결정론: shadow 재실행 시 trail 행동열·resolver 원장 완전 동일
  S4 실행 동결: 채택 ExecutionHead 구성 해시가 런 전후 불변
실행 정책 = 채택 PPO(YR-143 C0 + StateNorm + 대기 허가증 — 판정 경로와 동일 조립).
관측창 = 단축(워밍업 30분+측정 2시간) — 배선 계약 검사라 창 길이와 무관(정직 고지).
성능 주장 없음. TransferHead 는 미학습 초기화(배선 검증 목적 — 가중치 값 무관).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..integrated.candidates import CandidateGenerator
from ..integrated.multiblock import MultiBlockTerminal
from ..integrated.policy_config import ADOPTED_C0_GUARD
from ..integrated.profiles import build_h21_profile
from ..integrated.repro import code_dirty
from ..integrated.sell_review import (ANNOUNCE_LEAD_S, UnifiedSellOrchestrator)
from ..integrated.terminal_stream import (ObservationContract,
                                          TerminalStreamParams,
                                          WipAdmissionController,
                                          admission_epochs, build_fixed_wip,
                                          hotspot_rotation)
from ..v1.ppo_policy import PpoSellPolicy, TransferActor, TransferCritic
from ..integrated.yard_layout import terminal_layout
from ..integrated.baselines import _apply
from .yr088_joint_rl import LEVEL
from .yr149_load_cells import _sim_from
from .yr150_h21_pilot import _git, _sha256
from .yr151_transfer_ppo import (AdoptedExecFleet, exec_config_hash,
                                 load_adopted_execution_head, load_kf)
from .yr157_band_qual import (N_HOTSPOT, SEED as BAND_SEED, hotspot_seed,
                              pair_seed)

OUT = Path("outputs/reports/yr151_shadow_verify")
PREREG = Path(".claude/docs/dashboard-task-specs/YR-151-block-ppo-sell-head.md")
W, LOAD = 3.0, 100          # ★최종 확정 주 무대(2026-08-10·짝 체계·장치율 0.65)
MASTER = 150                # 중첩 명단 계약
OBS = ObservationContract(warmup_s=1_800.0, measure_s=7_200.0, snapshot_s=300.0)


def _episode(with_shadow: bool, exec_actor, exec_norm):
    layout = terminal_layout()
    seed = pair_seed(W, 0)
    hs = hotspot_rotation(layout, hotspot_seed(W, 0), N_HOTSPOT)
    params = TerminalStreamParams(load_4h=LOAD, hotspot_blocks=hs, hotspot_weight=W)
    built = build_fixed_wip(build_h21_profile(), seed, wip_target=LOAD, obs=OBS,
                            layout=layout, params=params,
                            background_seed=BAND_SEED, master_load=MASTER)
    mbt = MultiBlockTerminal({b: _sim_from(s) for b, s in built["scenarios"].items()},
                             extra_review_epochs=admission_epochs(OBS))
    ctrl = WipAdmissionController(built["pool"], wip_target=LOAD,
                                  lead_s=ANNOUNCE_LEAD_S, end_s=OBS.observe_s)
    policy = orch = None
    if with_shadow:
        torch.manual_seed(7_000_000)             # 미학습 초기화 — 시드 선고정(재현)
        policy = PpoSellPolicy(TransferActor(), TransferCritic(), mode="shadow",
                               sample=True, seed=7_000_000, layout=layout)
        orch = UnifiedSellOrchestrator(policy, layout, load_kf(), dry_run=True)

    fleet = AdoptedExecFleet(exec_actor, exec_norm, config=ADOPTED_C0_GUARD)
    gens: dict[int, CandidateGenerator] = {}
    exc = {"n": 0}

    def exec_policy(sim, dp):
        g = gens.setdefault(
            id(sim), CandidateGenerator(config=ADOPTED_C0_GUARD))
        gb = {c: g.generate(sim, c, LEVEL) for c in dp.crane_ids}
        _apply(sim, fleet.get(sim).decide(sim, dp, gb))

    def review(mbt_, t):
        ctrl.review(mbt_, t)
        if orch is not None:
            orch.review(mbt_, t)

    mbt.run(exec_policy, review_fn=review)

    rows = {}
    for b, sim in mbt.blocks.items():
        tl = getattr(sim, "time_ledger", None)
        if tl:
            for jid, r in tl.records.items():
                rows[jid] = (b, round(r.gate_in, 6),
                             None if r.block_arrival is None else round(r.block_arrival, 6),
                             None if r.gate_out is None else round(r.gate_out, 6))
    return {"rows": rows, "policy": policy, "orch": orch, "exc": exc["n"]}


def run() -> dict:
    exec_actor, exec_norm = load_adopted_execution_head()
    h0 = exec_config_hash(exec_actor, 221_000, ADOPTED_C0_GUARD)

    base = _episode(False, exec_actor, exec_norm)
    sh1 = _episode(True, exec_actor, exec_norm)
    sh2 = _episode(True, exec_actor, exec_norm)

    s1 = base["rows"] == sh1["rows"]
    trail = sh1["policy"].trail
    dry = [e for e in sh1["orch"].ledger if e["decision"] == "DRY_WOULD_COMMIT"]
    s2 = (len(trail) >= 1 and len(dry) >= 1
          and all(float(tr["critic_in"].abs().sum()) > 0 for tr in trail[:50]))
    acts1 = [(tr["t"], tr["src"], tr["action"]) for tr in trail]
    acts2 = [(tr["t"], tr["src"], tr["action"]) for tr in sh2["policy"].trail]
    s3 = acts1 == acts2 and sh1["orch"].ledger == sh2["orch"].ledger
    s4 = exec_config_hash(exec_actor, 221_000, ADOPTED_C0_GUARD) == h0

    checks = {"S1_env_invariant_rows_identical": s1,
              "S2_wiring_flows": s2,
              "S3_shadow_deterministic": s3,
              "S4_exec_config_frozen": s4,
              "no_exec_exceptions": base["exc"] == 0 and sh1["exc"] == 0}
    verdict = {"shadow_verify_all_pass": all(checks.values()), "checks": checks,
               "n_rows": len(base["rows"]), "n_trail": len(trail),
               "n_would_commit": len(dry),
               "n_resolver_entries": len(sh1["orch"].ledger),
               "note": "배선 계약 검증 — 성능 주장 없음. TransferHead 미학습 초기화. "
                       "관측창 단축(계약 검사라 창 길이 무관)."}
    dirty = bool(code_dirty())
    res = {"task": "YR-151-shadow-verify", "cell": f"w{W}-L{LOAD}",
           "runtime": {"commit": _git("rev-parse", "HEAD"), "git_dirty": dirty,
                       "prereg_file": str(PREREG),
                       "prereg_sha256": _sha256(PREREG) if PREREG.exists() else None,
                       "exec_head_hash": h0,
                       "exec_policy_config": ADOPTED_C0_GUARD.as_dict(),
                       "seeds": {"cell": pair_seed(W, 0),
                                 "background": BAND_SEED, "net_init": 7_000_000},
                       "observation": OBS.as_dict()},
           "verdict": verdict}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "shadow_verify.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "shadow_verify.json.sha256").write_text(_sha256(p) + "\n",
                                                   encoding="utf-8")
    print(json.dumps({"verdict": verdict, "dirty": dirty}, ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.run:
        run()
    print("DONE")
