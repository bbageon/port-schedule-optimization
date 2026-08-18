"""YR-183 — 교정 설정으로 동결 예산 40회차 (5시드 × 20코어).

■ 답할 질문 하나
  YR-185 교정으로 규칙과의 격차가 96.6 → 40.9 로 줄었다. 회차는 여전히 25%
  (10/40)였고 곡선은 마지막까지 개선 중이었다. **남은 40.9 를 학습량으로
  메우는가.**

■ 사전등록 (실행 전 동결)
  `.claude/docs/strategy-history/2026-08-18-YR-183-교정설정-40회차-사전등록.md`
  · 주판정 = **규칙 대비**, 모드 = **argmax**(추첨도 병행 보고)
    — YR-185 가 실패한 두 지점을 명시적으로 고쳤다.
  · 바뀌는 축은 **회차(10→40)와 시드 수(1→5)** 뿐. 나머지는 YR-185 와 동일.

■ 병렬 구조 (20코어)
  시드 5개를 스레드로 동시에 굴리고, 각 시드가 자기 4워커 프로세스풀로
  에피소드를 돌린다(5 × 4 = 20). 시드당 벽시계는 1시드일 때와 같으므로
  **같은 시간에 5배의 증거**를 얻는다. 남은 4코어는 다른 세션 몫이다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path("outputs/reports/yr183_budget40")
PREREG = (".claude/docs/strategy-history/"
          "2026-08-18-YR-183-교정설정-40회차-사전등록.md")
ARM = "fixed15"
TRAIN_SEEDS = (8_400_000, 8_500_000, 8_600_000, 8_700_000, 8_800_000)
N_ITER, EPS_PER_ITER = 40, 4          # ★동결 예산
WORKERS_PER_SEED = EPS_PER_ITER       # 5 × 4 = 20 코어


def _train_one(actor, critic, ts: int, out: Path) -> list[dict]:
    """시드 하나의 40회차. 자기 프로세스풀(4워커)을 쓴다."""
    import torch
    from concurrent.futures import ProcessPoolExecutor
    from .yr151_transfer_ppo import LR, ppo_update
    from .yr170_sell_ppo_diurnal import GRAD_CLIP, MINIBATCH
    from .yr171c_train import _worker
    from .yr174_txn_reward import scale_report
    from .yr185_retrain import ZERO_KEEP, resample_zeros

    opt_a = torch.optim.Adam(actor.parameters(), lr=LR)
    opt_c = torch.optim.Adam(critic.parameters(), lr=LR)
    out.mkdir(parents=True, exist_ok=True)
    hist: list[dict] = []
    for it in range(N_ITER):
        sd_a = {k: v.detach().cpu() for k, v in actor.state_dict().items()}
        sd_c = {k: v.detach().cpu() for k, v in critic.state_dict().items()}
        jobs = [(ts + it * EPS_PER_ITER + e, sd_a, sd_c, ts + it * 100 + e, ARM)
                for e in range(EPS_PER_ITER)]
        with ProcessPoolExecutor(max_workers=WORKERS_PER_SEED) as pool:
            eps = list(pool.map(_worker, jobs))
        raw = [b for ep in eps for b in ep["batch"]]
        batch, rep = resample_zeros(raw, keep=ZERO_KEEP, seed=ts + it)
        stats = ppo_update(actor, critic, opt_a, opt_c, batch,
                           minibatch=MINIBATCH, seed=ts + it,
                           grad_clip=GRAD_CLIP)
        n = len(eps)
        hist.append({"iter": it, **stats,
                     "phi_final_mean": sum(e["phi_final"] for e in eps) / n,
                     "n_space_mean": sum(e["n_space"] for e in eps) / n,
                     "n_time_mean": sum(e["n_time"] for e in eps) / n,
                     "credit_sum_mean": sum(e["credit_sum"] for e in eps) / n,
                     "resample": rep, "scale": scale_report(batch)})
        (out / "train.json").write_text(json.dumps(
            {"train_seed": ts, "arm": ARM, "zero_keep": ZERO_KEEP,
             "n_iter_target": N_ITER, "history": hist,
             "in_progress": it < N_ITER - 1},
            ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()},
                   out / "net.pt")
    return hist


def train() -> Path:
    import torch
    import torch.multiprocessing as _mp
    from concurrent.futures import ThreadPoolExecutor
    from ..integrated.repro import code_dirty, repro_stamp, write_result
    from ..integrated.terminal_stream import OBS_24H
    from ..integrated.transfer_head import TransferActor, TransferCritic
    from .yr170_sell_ppo_diurnal import GRAD_CLIP, MINIBATCH
    from .yr185_retrain import ZERO_KEEP
    _mp.set_sharing_strategy("file_system")
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)

    # 신경망 초기화는 **주 스레드에서 순차로** 한다 — torch.manual_seed 가 전역이라
    # 스레드에서 부르면 시드가 서로 섞인다.
    nets = {}
    for ts in TRAIN_SEEDS:
        torch.manual_seed(ts)
        nets[ts] = (TransferActor(), TransferCritic())

    with ThreadPoolExecutor(max_workers=len(TRAIN_SEEDS)) as tp:
        futs = {ts: tp.submit(_train_one, *nets[ts], ts, OUT / f"s{ts}")
                for ts in TRAIN_SEEDS}
        hists = {ts: f.result() for ts, f in futs.items()}

    payload = {"experiment": "YR-183 교정 설정 동결 예산 40회차",
               "prereg": PREREG, "kind": "training",
               "note": "성능 판정은 별도 평가(yr183_eval)에서. 여기는 학습만.",
               "arm": ARM, "train_seeds": list(TRAIN_SEEDS),
               "n_iter": N_ITER, "eps_per_iter": EPS_PER_ITER,
               "zero_keep": ZERO_KEEP,
               "changed_axis": "회차 10→40 · 시드 1→5 (그 외 YR-185 와 동일)",
               "history_by_seed": {str(k): v for k, v in hists.items()},
               "code_dirty": bool(code_dirty()),
               "stamp": repro_stamp(
                   experiment="YR-183 40회차 5시드",
                   seeds={"train": list(TRAIN_SEEDS)},
                   params={"arm": ARM, "N_ITER": N_ITER,
                           "EPS_PER_ITER": EPS_PER_ITER,
                           "ZERO_KEEP": ZERO_KEEP, "MINIBATCH": MINIBATCH,
                           "GRAD_CLIP": GRAD_CLIP,
                           "observation": OBS_24H.as_dict()},
                   prereg=PREREG)}
    p = OUT / "train_all.json"
    write_result(p, payload)
    return p


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    p = train()
    d = json.loads(p.read_text(encoding="utf-8"))
    print(f"{'iter':>4}", end="")
    for ts in TRAIN_SEEDS:
        print(f" {('s'+str(ts)[:3]):>9}", end="")
    print("   0비율   critic")
    for i in (0, 9, 19, 29, N_ITER - 1):
        print(f"{i:>4}", end="")
        for ts in TRAIN_SEEDS:
            h = d["history_by_seed"][str(ts)][i]
            print(f" {h['phi_final_mean']:>9.1f}", end="")
        h0 = d["history_by_seed"][str(TRAIN_SEEDS[0])][i]
        print(f"  {1-h0['scale']['nonzero_share']:>6.1%} "
              f"{h0['scale']['value_rms']:>8.4f}")
    print("DONE", p)
