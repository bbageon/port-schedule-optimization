"""턴타임을 네 토막으로 갈라 본다 — **어디서 길어졌나** (사용자 지시 2026-09-22).

    게이트인 ─진입─► 블록도착 ─대기─► 작업시작 ─작업─► 작업완료 ─반출─► 게이트아웃

같은 날을 학습 크레인과 규칙 크레인으로 각각 굴려 토막별로 견준다. 합계(턴타임)만
보면 *"붐비는 날에 길다"* 가 **크레인을 기다려서인지 작업이 느려서인지** 모른다.

    PYTHONPATH=src python scripts/v4/probe_turn_parts.py \\
        --ckpt outputs/v4/crane-train-v2/crane_071.pt --loads 3500,15000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PARTS = ("진입", "대기", "작업", "반출")


def run_one(load: int, seed: int, dispatcher: str, ckpt: str | None) -> dict:
    """하루를 굴리고 그 날의 토막 평균을 돌려준다."""
    import yard_rl.v4.stage.episode as E

    grabbed: dict = {}
    orig = E.MultiBlockTerminal

    class Spy(orig):                       # 원장을 꺼내려고 한 겹 씌운다
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            grabbed["mbt"] = self

    net = None
    if dispatcher == "RL_CRANE":
        import torch
        from yard_rl.v4.crane.policy import CraneNet
        net = CraneNet()
        net.load_state_dict(torch.load(ckpt)["crane"])
        net.eval()

    E.MultiBlockTerminal = Spy
    try:
        r = E.run_episode(load=load, arm="NO_REALLOC", dispatcher=dispatcher,
                          seed=seed, crane_net=net)
    finally:
        E.MultiBlockTerminal = orig
    p = grabbed["mbt"].ledger.turn_time_parts_s()
    p["phi"] = r.phi_krw
    return p


def main(argv=None) -> int:
    from yard_rl.v4.crane.train import EVAL_LOADS, EVAL_SEED_BASE

    ap = argparse.ArgumentParser(description="턴타임 4토막 분해 — 학습 vs 규칙")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--loads", default="3500,7500,15000")
    ap.add_argument("--seed-base", type=int, default=EVAL_SEED_BASE)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    loads = [int(x) for x in a.loads.split(",") if x]

    rows = []
    print("  부하   정책        턴타임 =   진입  +   대기  +  작업  +  반출   (완주)")
    for load in loads:
        i = EVAL_LOADS.index(load) if load in EVAL_LOADS else 0
        seed = a.seed_base + i * 10 + load // 1000
        pair = {}
        for disp in ("SF_SPT", "RL_CRANE"):
            p = run_one(load, seed, disp, a.ckpt)
            pair[disp] = p
            print(f"{load:>7,}  {'규칙' if disp == 'SF_SPT' else '학습':<8} "
                  f"{p['턴타임']/60:>7.1f}분 = "
                  + " + ".join(f"{p[k]/60:>6.1f}" for k in PARTS)
                  + f"   ({p['n']:,})")
        d = {k: (pair["RL_CRANE"][k] - pair["SF_SPT"][k]) / 60 for k in PARTS}
        print(f"{'':>7}  {'차이':<8} "
              f"{(pair['RL_CRANE']['턴타임']-pair['SF_SPT']['턴타임'])/60:>+7.1f}분 = "
              + " + ".join(f"{d[k]:>+6.1f}" for k in PARTS))
        rows.append({"load": load, "seed": seed,
                     "rule": pair["SF_SPT"], "rl": pair["RL_CRANE"],
                     "diff_min": d})
    out = Path(a.out or "outputs/v4/turn_parts.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
