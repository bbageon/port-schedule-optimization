"""자격시험 한 번 = 수집 → 학습 → 관문 A~F ([[YR-223]] 1~3단계).

■ 순서와 값
      1  수집    날 × 창 → 궤적            (프로세스 병렬 · 가장 비싸다)
      2  학습    소형 LSTM · 날 단위 held-out
      3  A~E     모델만으로 판정            (몇 초)
      4  ★F      반사실 개입과 대조         (다시 시뮬레이터가 필요 · 비싸다)

■ ★F 를 따로 돌리는 이유
  A·B·D 가 미달이면 F 를 돌릴 이유가 없다 — 모델이 아무것도 안 배웠는데 그 순위로
  개입해 봐야 잡음을 재는 것이다. **A·B·D 를 먼저 보고, 통과했을 때만 F 로 간다.**
"""
from __future__ import annotations

import json
import pathlib
import time

from ..v3 import CF_HORIZON_S
from . import collect as C
from . import contrib as K
from . import gates as G
from . import train as T
from .runner import intervene, snapshot_at

#: 사전등록 기본 설정 — 착수 시 동결한다.
PRESET = {
    "loads": (3_500, 5_000),
    "seeds": tuple(9_900_900 + i for i in range(8)),   # 비판정 대역 · 날 8개
    "n_windows": 13,                                    # 날마다 창 13개 → 208 궤적
    "horizon_s": CF_HORIZON_S,                          # 3시간
    "explore": 0.0,
    "val_frac": 0.25,                                   # 날 8개 중 2일이 검증
    "epochs": 400,
    "hidden": 32,
    "init_seeds": (0, 1, 2),                            # 관문 C
    "top_k": 20,                                        # 관문 C·F 가 보는 상위 epoch
    "f_windows": 3,                                     # 관문 F 로 개입할 창 수
}


def _log(msg, on_log=None):
    if on_log is not None:
        on_log(msg)


# ------------------------------------------------------------------ 1단계 수집
def step_collect(*, out_dir, workers, ckpt_path=None, preset=None, on_log=None):
    cfg = {**PRESET, **(preset or {})}
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_done = [0]

    def on_day(r):
        n_done[0] += 1
        _log(f"  날 {n_done[0]}/{len(cfg['loads']) * len(cfg['seeds'])} "
             f"부하{r['load']} 시드{r['seed']} 창{len(r['windows'])} "
             f"{r['secs']:.0f}초 (누적 {time.time() - t0:.0f}초)", on_log)

    days = C.collect(loads=cfg["loads"], seeds=cfg["seeds"],
                     n_windows=cfg["n_windows"], ckpt_path=ckpt_path,
                     horizon_s=cfg["horizon_s"], explore=cfg["explore"],
                     workers=workers, on_day=on_day)
    p = C.save(days, out / "windows.json")
    wins = C.load(p)
    summ = C.summarize(wins)
    _log(f"수집 끝 {time.time() - t0:.0f}초 · {summ}", on_log)
    (out / "collect_summary.json").write_text(
        json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    return wins, summ


# ------------------------------------------------------- 2~3단계 학습 + A~E
def step_train_and_gates(wins, *, out_dir, preset=None, on_log=None):
    cfg = {**PRESET, **(preset or {})}
    out = pathlib.Path(out_dir)
    kw = dict(val_frac=cfg["val_frac"], epochs=cfg["epochs"],
              hidden=cfg["hidden"])
    t0 = time.time()
    model, norm, rep = T.fit(wins, seed=cfg["init_seeds"][0], **kw)
    _log(f"기본 학습 {time.time() - t0:.0f}초 · 파라미터 {rep.n_params} · "
         f"학습 {rep.train_loss:.4f} 검증 {rep.val_loss:.4f} "
         f"상관 {rep.val_corr:+.3f} 평균오차 {rep.val_mae_krw:,.0f}원", on_log)

    _m, _n, rep_b = T.fit(wins, seed=cfg["init_seeds"][0], ablate_actions=True, **kw)
    _m, _n, rep_d = T.fit(wins, seed=cfg["init_seeds"][0], shuffle_order=True, **kw)

    # 관문 C — 초기값을 바꿔 다시 학습하고 상위 epoch 순위가 닮았는지 본다.
    _tr, va = T.split_by_seed(wins, cfg["val_frac"])
    probe = (va or wins)[: min(5, len(va or wins))]
    base_c = [K.contributions(model, norm, w) for w in probe]
    rank_corrs = []
    for sd in cfg["init_seeds"][1:]:
        m2, n2, _r2 = T.fit(wins, seed=sd, **kw)
        for w, b in zip(probe, base_c):
            rank_corrs.append(K.rank_agreement(b, K.contributions(m2, n2, w),
                                               k=cfg["top_k"]))

    cons = [K.conservation(K.contributions(model, norm, w), w.y_krw) for w in wins]

    rpt = G.GateReport()
    rpt.add(G.gate_a(rep))
    rpt.add(G.gate_b(rep, rep_b))
    rpt.add(G.gate_c(rank_corrs))
    rpt.add(G.gate_d(rep, rep_d))
    rpt.add(G.gate_e(cons))
    for g in rpt.gates:
        _log(f"  관문 {g.name} {'통과' if g.passed else '미달'} "
             f"{g.value:+.3f} (문턱 {g.threshold}) — {g.note}", on_log)

    import torch
    torch.save({"model": model.state_dict(), "norm": norm.as_dict(),
                "hidden": cfg["hidden"]}, out / "rudder.pt")
    (out / "gates_ABCDE.json").write_text(
        json.dumps({"fit": rep.as_dict(), "ablated": rep_b.as_dict(),
                    "shuffled": rep_d.as_dict(), **rpt.as_dict()},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return model, norm, rep, rpt


# ------------------------------------------------------------------ 4단계 ★F
def step_gate_f(model, norm, wins, *, out_dir, ckpt_path=None, preset=None,
                on_log=None):
    """★상위 epoch 에 **실제로 개입**해 부호가 맞는지 본다.

    창 하나당: 하루 굴리기(169초) + 정책창 1 + 개입창 k. k=20 이면 약 12분.
    """
    cfg = {**PRESET, **(preset or {})}
    out = pathlib.Path(out_dir)
    nets = C.load_nets(ckpt_path)
    # ★Y 가 큰 창부터 고른다 — 0원짜리 창에 개입해 봐야 부호가 잡음이다.
    picks = sorted(wins, key=lambda w: -abs(w.y_krw))[: cfg["f_windows"]]
    pairs, rows = [], []
    for w in picks:
        t0 = time.time()
        cs = K.contributions(model, norm, w)
        tops = K.top_epochs(cs, cfg["top_k"])
        box = snapshot_at(load=w.load, seed=w.seed, t0=w.t0,
                          seller_net=nets[0], buyer_net=nets[1],
                          explore=cfg["explore"])
        for c in tops:
            r = intervene(box["ctx"], mbt=box["mbt"], orders=box["orders"],
                          records=box["records"], decided=box["decided"],
                          t0=box["t0"], horizon_s=w.horizon_s, epochs=[c.t])
            pairs.append((c.krw, r["d_krw"]))
            rows.append({"seed": w.seed, "load": w.load, "t0": w.t0, "t": c.t,
                         "contrib_krw": c.krw, "actual_krw": r["d_krw"]})
        _log(f"  창 시드{w.seed} t0={w.t0/3600:.1f}h · 개입 {len(tops)}건 "
             f"{time.time() - t0:.0f}초", on_log)
    g = G.gate_f(pairs)
    _log(f"  관문 F {'통과' if g.passed else '미달'} {g.value:+.3f} "
         f"(문턱 {g.threshold}) — {g.note}", on_log)
    (out / "gate_F.json").write_text(
        json.dumps({**g.as_dict(), "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return g, rows
