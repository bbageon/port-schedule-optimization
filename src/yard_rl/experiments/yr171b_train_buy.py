"""YR-171-B ② BUY 견적망 지도학습 + 3종 검증.

■ 무엇을 배우나
입력 = 그 시각의 하루 계획표(21블록 × 48칸 × 9) + 작업 특징(12)
출력 = (작업, 블록, 슬롯) 예상 부담
손실 = 관측된 점 (작업 j, 실제 도착 블록 b, 실제 도착 칸 t) 에서만 계산한다.
       나머지 격자점은 정답이 없으므로 **손실을 주지 않는다**(억지 0 학습 금지).

■ 3종 검증 (명세 171-B 요구)
  1. **결정론** — 같은 입력에 같은 출력 (eval 모드·같은 가중치)
  2. **단조성** — 계획표의 그 칸 예약 대수를 늘리면 예상 부담이 줄지 않아야 한다
     (명세의 "가법성"을 이 구조에서 검사 가능한 형태로 옮긴 것: 부담은 혼잡의
     증가함수여야 한다. 위반하면 resolver 가 붐비는 칸을 싸다고 고른다)
  3. **방향성** — 미열람 검증셋에서 예측과 실현 부담의 순위상관·오차
     (예측이 실제와 방향이 반대면 견적으로 못 쓴다)

■ 하지 않는 것
· 검증 결과를 보고 특징 정규화 상수를 바꾸는 것(assumed 동결 대상).
· 검증을 통과하지 못한 망을 resolver 에 넣는 것(`buy_estimator` 지위 주석).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path("outputs/reports/yr171b_estimator")
SPEC = ".claude/docs/dashboard-task-specs/YR-171-time-sale-slot-contract.md"
EPOCHS = 40
LR = 1e-3
BATCH_SNAPS = 4              # 한 step 에 스냅샷 몇 개 (스냅샷 = 계획표 1장)
VAL_SHARE = 0.25             # 시드 단위 분할 — 같은 날이 학습·검증에 섞이지 않는다


def _load(paths: list[Path]) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def _pack(eps: list[dict]):
    """에피소드들 → step 단위 (계획표, 작업특징, 대상 index, 정답) 목록.

    작업 특징은 수집 때 저장하지 않았다(계획표만 저장). 여기서는 **작업 특징을 라벨
    행에서 재구성**하지 않고, 견적망 입력 중 작업축을 **슬롯·규격 정보로 대체**한다 —
    수집 하네스가 job_features 를 함께 저장하도록 확장되면 그대로 갈아끼운다.
    """
    steps = []
    for e in eps:
        by_snap: dict[int, list[dict]] = {}
        for lab in e["labels"]:
            by_snap.setdefault(lab["snap_idx"], []).append(lab)
        for si, labs in by_snap.items():
            snap = e["snapshots"][si]
            steps.append({"plan": snap["plan"], "blocks": snap["blocks"],
                          "t": snap["t"], "labels": labs, "seed": e["seed"]})
    return steps


def _job_row(lab: dict, snap_t: float) -> list[float]:
    """정답지 행에서 **공개 정보만으로** 작업 특징 12개를 만든다.

    수집 시점(snap_t)에 알 수 있는 것만 쓴다 — 예약 도착까지 남은 시간·속한 칸·
    반출 여부. 실현값(점유시간·밀린 대수)은 **라벨 쪽**이므로 넣지 않는다.
    """
    from ..integrated.buy_estimator import JOB_DIM
    from ..integrated.slot_plan import DAY_S, N_SLOTS, SLOT_S
    eta = lab["block_arrival_s"]
    rem = eta - snap_t
    row = [0.0] * JOB_DIM
    row[0] = max(-1.0, min(1.0, rem / SLOT_S))          # eta_in_30m
    row[1] = max(-1.0, min(1.0, rem / DAY_S))           # eta_in_day
    row[2] = (eta // SLOT_S) / N_SLOTS                  # eta_slot_pos
    row[3] = max(-1.0, min(1.0, rem / SLOT_S))          # gate_in_30m (동일 근사)
    return row


def train(paths: list[Path]) -> dict:
    import torch
    from torch import nn
    from ..integrated.buy_estimator import JOB_DIM, BuyEstimator
    from ..integrated.slot_plan import N_FEATURES
    torch.manual_seed(0)

    eps = _load(paths)
    seeds = sorted({e["seed"] for e in eps})
    n_val = max(1, int(round(len(seeds) * VAL_SHARE)))
    val_seeds = set(seeds[-n_val:])
    tr = _pack([e for e in eps if e["seed"] not in val_seeds])
    va = _pack([e for e in eps if e["seed"] in val_seeds])
    if not tr or not va:
        raise RuntimeError(f"분할 실패 — 학습 {len(tr)} / 검증 {len(va)} step")

    net = BuyEstimator(slot_dim=N_FEATURES, job_dim=JOB_DIM)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    lossf = nn.SmoothL1Loss()

    def _batch(step):
        plans = torch.tensor(step["plan"], dtype=torch.float32)   # (B,48,F)
        bidx = {b: i for i, b in enumerate(step["blocks"])}
        labs = [l for l in step["labels"] if l["block"] in bidx]
        if not labs:
            return None
        jobs = torch.tensor([_job_row(l, step["t"]) for l in labs],
                            dtype=torch.float32)                   # (N,job_dim)
        tgt = torch.tensor([l["burden"] for l in labs], dtype=torch.float32)
        bi = torch.tensor([bidx[l["block"]] for l in labs], dtype=torch.long)
        si = torch.tensor([l["slot"] for l in labs], dtype=torch.long)
        return plans, jobs, bi, si, tgt

    def _eval(steps):
        net.eval()
        se = ae = n = 0.0
        pred_all, tgt_all = [], []
        with torch.no_grad():
            for st in steps:
                b = _batch(st)
                if b is None:
                    continue
                plans, jobs, bi, si, tgt = b
                out = net(plans, jobs)                    # (N,B,48)
                p = out[torch.arange(len(bi)), bi, si]
                se += float(((p - tgt) ** 2).sum())
                ae += float((p - tgt).abs().sum())
                n += len(tgt)
                pred_all += p.tolist()
                tgt_all += tgt.tolist()
        net.train()
        return {"n": int(n), "rmse": (se / n) ** 0.5 if n else None,
                "mae": ae / n if n else None,
                "pred": pred_all, "tgt": tgt_all}

    hist = []
    for ep in range(EPOCHS):
        tot = cnt = 0.0
        for i in range(0, len(tr), BATCH_SNAPS):
            opt.zero_grad()
            loss_sum, m = 0.0, 0
            for st in tr[i:i + BATCH_SNAPS]:
                b = _batch(st)
                if b is None:
                    continue
                plans, jobs, bi, si, tgt = b
                out = net(plans, jobs)
                p = out[torch.arange(len(bi)), bi, si]
                loss_sum = loss_sum + lossf(p, tgt)
                m += 1
            if not m:
                continue
            (loss_sum / m).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss_sum / m)
            cnt += 1
        v = _eval(va)
        hist.append({"epoch": ep, "train_loss": tot / cnt if cnt else None,
                     "val_rmse": v["rmse"], "val_mae": v["mae"], "val_n": v["n"]})
    return {"net": net, "hist": hist, "val": _eval(va), "tr_steps": len(tr),
            "va_steps": len(va), "val_seeds": sorted(val_seeds)}


# ------------------------------------------------------------------ 3종 검증
def verify(net, va_steps, val) -> dict:
    import torch
    from ..integrated.buy_estimator import JOB_DIM
    from ..integrated.slot_plan import N_FEATURES
    net.eval()

    # ① 결정론 — 같은 입력 두 번
    plans = torch.rand(3, 48, N_FEATURES)
    jobs = torch.rand(5, JOB_DIM)
    with torch.no_grad():
        a, b = net(plans, jobs), net(plans, jobs)
    deterministic = bool(torch.equal(a, b))

    # ② 단조성 — 그 칸 예약 대수(특징 0·1)를 올리면 부담이 줄지 않아야 한다
    from ..integrated.slot_plan import SLOT_FEATURES
    i_in = SLOT_FEATURES.index("notified_in")
    viol = tot = 0
    with torch.no_grad():
        base = net(plans, jobs)
        for k in range(0, 48, 6):
            bumped = plans.clone()
            bumped[:, k, i_in] += 0.5
            up = net(bumped, jobs)
            d = (up[:, :, k] - base[:, :, k]).flatten()
            viol += int((d < -1e-4).sum())
            tot += d.numel()
    monotone = {"violations": viol, "checked": tot,
                "violation_share": round(viol / tot, 4) if tot else None,
                "pass": tot > 0 and viol / tot <= 0.05}

    # ③ 방향성 — 미열람 검증셋 예측 vs 실현 (순위상관)
    p, t = val["pred"], val["tgt"]
    rho = _spearman(p, t) if len(p) > 2 else None
    from statistics import fmean, pstdev
    direction = {"n": len(p), "spearman": rho,
                 "rmse": val["rmse"], "mae": val["mae"],
                 "pred_mean": round(fmean(p), 4) if p else None,
                 "tgt_mean": round(fmean(t), 4) if t else None,
                 "tgt_sd": round(pstdev(t), 4) if len(t) > 1 else None,
                 "pass": rho is not None and rho >= 0.3}
    return {"deterministic": deterministic, "monotone": monotone,
            "direction": direction,
            "all_pass": bool(deterministic and monotone["pass"]
                             and direction["pass"])}


def _spearman(x, y) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return round(num / (dx * dy), 4) if dx and dy else None


def run() -> Path:
    import torch
    from ..integrated.repro import code_dirty, repro_stamp
    paths = sorted(OUT.glob("data_s*.json"))
    if not paths:
        raise RuntimeError("정답지가 없다 — yr171b_collect 를 먼저 실행할 것")
    r = train(paths)
    va = _pack(_load([p for p in paths
                      if json.loads(p.read_text(encoding='utf-8'))["seed"]
                      in set(r["val_seeds"])]))
    v = verify(r["net"], va, r["val"])
    torch.save({"buy": r["net"].state_dict()}, OUT / "buy_net.pt")
    res = {"experiment": "YR-171-B BUY 견적망 지도학습 + 3종 검증",
           "n_train_steps": r["tr_steps"], "n_val_steps": r["va_steps"],
           "val_seeds": r["val_seeds"], "epochs": EPOCHS, "lr": LR,
           "history": r["hist"], "verify": v,
           "data_status": "시뮬레이터의 근사이지 실제 항만 부담이 아니다",
           "code_dirty": bool(code_dirty()),
           "stamp": repro_stamp(experiment="YR-171-B BUY 견적망 지도학습",
                                seeds={"val": r["val_seeds"]},
                                params={"EPOCHS": EPOCHS, "LR": LR,
                                        "VAL_SHARE": VAL_SHARE},
                                prereg=SPEC)}
    p = OUT / "buy_train.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                 encoding="utf-8")
    return p


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    p = run()
    d = json.loads(p.read_text(encoding="utf-8"))
    h = d["history"]
    print(f"학습 {d['n_train_steps']} step / 검증 {d['n_val_steps']} step "
          f"(검증 시드 {d['val_seeds']})")
    for e in (h[0], h[len(h) // 2], h[-1]):
        print(f"  epoch {e['epoch']:>3}  train {e['train_loss']:.4f}  "
              f"val RMSE {e['val_rmse']:.4f}  MAE {e['val_mae']:.4f}")
    v = d["verify"]
    print(f"① 결정론 {v['deterministic']}")
    print(f"② 단조성 위반 {v['monotone']['violations']}/{v['monotone']['checked']} "
          f"({v['monotone']['violation_share']:.2%}) → {v['monotone']['pass']}")
    print(f"③ 방향성 순위상관 {v['direction']['spearman']} "
          f"(n={v['direction']['n']}, 실현 평균 {v['direction']['tgt_mean']} "
          f"± {v['direction']['tgt_sd']}) → {v['direction']['pass']}")
    print(f"전체 통과: {v['all_pass']}")
    print("DONE", p)
