"""블록 재배치가 혼잡에서 왜 해로운가 ([[YR-248]]).

    PYTHONPATH=src python scripts/v3/space_harm.py

28일 평가에서 **블록 재배치만** 쓰면 초혼잡(15,000)에서 −2.41억으로 손해였다
(`03b` §8-3). 가설 둘을 Φ 항별로 가른다:

    ① 크레인이 더 움직인다        → `c_move` 가 오른다
    ② 옮겨 간 블록이 곧 막힌다     → `c_wait` 가 오른다 (이동 비용은 그대로)

★한계: 30일 안의 하루가 아니라 **독립 하루**다. 야드 초기상태가 이어지지 않으므로
  크기는 다를 수 있고, 여기서 보는 것은 **어느 항이 움직이나** 라는 방향이다.
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor

LOADS = (5_000, 15_000)
CKPT = "outputs/v3/month-02/ckpt_029.pt"
ARMS = ("NO_REALLOC", "RL_SPACE", "RL_TIME", "RL")


def one(kw):
    import torch
    from yard_rl.v3.actors import BuyerNet, SellerNet
    from yard_rl.v3.stage.episode import run_episode
    s, b = SellerNet(), BuyerNet()
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    s.load_state_dict(ck["seller"]); b.load_state_dict(ck["buyer"])
    r = run_episode(load=kw["load"], arm=kw["arm"], seed=9_900_777,
                    seller_net=s, buyer_net=b)
    d = r.breakdown
    return {"arm": kw["arm"], "load": kw["load"], "phi": r.phi_krw,
            "c_wait": d["c_wait"], "c_move": d["c_move"],
            "c_rehandle": d["c_rehandle"], "n_censored": d["n_censored"],
            "p90": d["p90_turn_time_s"], "n_space": r.n_space, "n_time": r.n_time}


def main() -> int:
    jobs = [{"arm": a, "load": v} for v in LOADS for a in ARMS]
    with ProcessPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(one, jobs))
    by = {(r["arm"], r["load"]): r for r in rows}
    for v in LOADS:
        base = by[("NO_REALLOC", v)]
        print(f"\n{'='*74}\n■ 부하 {v:,} — 안 팔기 대비 항목별 증감(원)\n")
        print(f"{'팔':<11} {'Φ':>14} {'대기':>14} {'이동':>12} {'재조작':>11} "
              f"{'미완료':>7} {'P90분':>7} {'공간':>6} {'시간':>6}")
        for a in ARMS:
            r = by[(a, v)]
            f = (lambda k: r[k] - base[k]) if a != "NO_REALLOC" else (lambda k: r[k])
            tag = "(절대값)" if a == "NO_REALLOC" else ""
            print(f"{a:<11} {f('phi'):>14,.0f} {f('c_wait'):>14,.0f} "
                  f"{f('c_move'):>12,.0f} {f('c_rehandle'):>11,.0f} "
                  f"{r['n_censored']:>7,} {r['p90']/60:>7.0f} "
                  f"{r['n_space']:>6,} {r['n_time']:>6,} {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
