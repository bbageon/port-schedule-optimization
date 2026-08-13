"""YR-170 속도 진단 — 에피소드 1회의 시간이 **어디로 가는가** (사용자 질문 2026-08-13).

배경: 회차당 약 25분(에피소드 4개 병렬 + PPO 갱신)이라 실험 한 번에 반나절이 든다.
GPU(RTX 5090)가 있는데 이 경로는 전부 CPU 다. 다만 신경망이 작고(은닉 64) 시뮬레이션이
순수 파이썬이라, **GPU 를 켜기 전에 병목이 어디인지부터 재야 한다.**

세 갈래를 분리해서 잰다:
  ① 시뮬레이션 본체 (이벤트 큐·크레인 일정·후보 생성)
  ② 집행 정책 (채택 PPO 헤드 = 결정마다 21블록 rollout / vs SF 단순 규칙)
  ③ 판매 검토 (resolver — 60초 격자마다 전 블록 제안 matching)
  ④ PPO 갱신 (표본 1개씩 순회 — 배칭 가능성)
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import time
from pathlib import Path

from ..integrated.policy_config import ADOPTED_C0_GUARD
from .yr151_transfer_ppo import load_kf
from .yr170_sell_ppo_diurnal import KeepAllTrail, run_episode_diurnal

OUT = Path("outputs/reports/yr170_speed_probe")


def _episode(exec_head: str, sell: bool) -> dict:
    """exec_head: 'adopted'|'sf' · sell: 판매 정책 사용 여부"""
    if sell:
        import torch
        from ..integrated.transfer_head import (PpoSellPolicy, TransferActor,
                                                TransferCritic)
        from ..integrated.yard_layout import terminal_layout
        torch.manual_seed(0)
        pol = PpoSellPolicy(TransferActor(), TransferCritic(), mode="live",
                            sample=True, seed=0, layout=terminal_layout())
    else:
        pol = KeepAllTrail()
    t0 = time.perf_counter()
    ep = run_episode_diurnal(
        8_400_000, pol, load_kf(), exec_head=exec_head,
        exec_config=(ADOPTED_C0_GUARD if exec_head == "adopted" else None))
    return {"exec_head": exec_head, "sell": sell,
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "phi_final": round(ep["phi_final"], 1),
            "n_space": ep["n_space"], "n_time": ep["n_time"]}


def matrix() -> dict:
    """2×2 — 집행 정책 × 판매 유무. 차이가 곧 각 축의 비용이다."""
    rows = [_episode("sf", False), _episode("sf", True),
            _episode("adopted", False), _episode("adopted", True)]
    base = rows[0]["elapsed_s"]
    for r in rows:
        r["x_vs_sf_keep"] = round(r["elapsed_s"] / base, 2)
    res = {"probe": "episode_matrix", "rows": rows,
           "note": "sf+keep 을 1 로 둔 배수. 차이가 집행 정책·판매 검토의 비용."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "episode_matrix.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False))
    return res


def profile_episode(exec_head: str = "adopted", top: int = 30) -> dict:
    """cProfile — 함수별 누적 시간 상위 N."""
    pr = cProfile.Profile()
    pr.enable()
    _episode(exec_head, sell=True)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(top)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"profile_{exec_head}.txt"
    p.write_text(s.getvalue(), encoding="utf-8")
    print(s.getvalue()[:4000])
    return {"path": str(p)}


def update_cost(n_samples: int = 60_879) -> dict:
    """PPO 갱신 — 현재(표본 1개씩) vs 묶음 처리의 비용 비교(합성 표본)."""
    import torch
    from ..integrated.transfer_head import TransferActor, TransferCritic
    torch.manual_seed(0)
    actor, critic = TransferActor(), TransferCritic()
    n = min(n_samples, 4096)                       # 측정용 축소 후 선형 환산
    rows = [torch.rand(3, 14) for _ in range(n)]   # 후보 2 + KEEP 1 가정
    cin = torch.rand(n, 15)
    t0 = time.perf_counter()
    for i in range(n):
        actor(rows[i])
        critic(cin[i:i + 1])
    loop_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    big = torch.stack(rows)                        # (n, 3, 14) — 후보 수 같을 때
    actor(big)
    critic(cin)
    batch_s = time.perf_counter() - t0
    scale = n_samples / n
    res = {"probe": "ppo_update_cost", "measured_n": n,
           "loop_s_per_pass": round(loop_s, 3),
           "batched_s_per_pass": round(batch_s, 4),
           "speedup": round(loop_s / max(batch_s, 1e-9), 1),
           "projected_loop_s_per_iteration": round(loop_s * scale * 4, 1),
           "projected_batched_s_per_iteration": round(batch_s * scale * 4, 1),
           "note": "1 iteration = 4 epoch. 후보 수가 달라 실제 배칭엔 패딩·마스킹 필요."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "update_cost.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["matrix", "profile", "update"],
                    default="update")
    ap.add_argument("--exec-head", default="adopted")
    a = ap.parse_args()
    if a.what == "matrix":
        matrix()
    elif a.what == "profile":
        profile_episode(a.exec_head)
    else:
        update_cost()
    print("DONE")
