"""30일 무대 학습 — **하루가 곧 한 회차**다 ([[YR-239]]).

사용자 지시 2026-08-26: *"나머지 28일을 학습해봐. 단 학습 중간보고는 28일중에서
하루가 끝날때마다 평가를 하는거지"*

■ 하루 무대 학습(`loop.run_training`)과 무엇이 다른가

| | 하루 회차 | **30일 무대** |
|---|---|---|
| 한 회차 | 무대를 새로 세워 하루 | **이어지는 세계의 하루** |
| 야드 출발점 | 날마다 인공 초기 적재 | 어제가 남긴 야드 |
| 부하 | 회차마다 돌려 가며 | **가중 추첨** (현실 분포) |
| 기준선 | 같은 시드 안 팔기를 **따로 굴린다** | 같은 세계를 두 번 못 굴린다 → 안 함 |
| 판정 | 회차 Φ | **부하별로 묶어서** (사용자 지시) |

■ ★기준선을 왜 못 붙이나
  하루 무대에서는 같은 시드로 `NO_REALLOC` 을 한 번 더 굴려 짝 격차를 봤다.
  30일은 **세계가 하나뿐**이라 그 짝이 없다 — 안 팔기로 30일을 따로 굴리면 야드가
  갈라져 5일째부터는 다른 세계다. 그래서 여기서는 **날별 Φ 만** 남기고, 팔 대조는
  달이 끝난 뒤 **같은 시드로 팔을 바꿔 통째로 다시 굴려** 짝비교한다.

■ ★라벨은 하루가 끝나면 버린다
  하루 무대와 같다 — 반사실 라벨에는 유효기간이 있다(04b §3). 어제 라벨은 어제
  정책 기준이라 오늘 학생에게는 낡았다.

■ 이 수치는 **진단**이다
  학습은 비판정 대역에서만 돈다. 판정은 [[YR-210]] 이 새 대역에서 한 번만 한다.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from ..eval.guards import DIAGNOSTIC_BAND
from ..stage.month import N_DAYS, plan_month, summarize
from ..stage.month_run import run_month
from .fit import StudentTrainer
from .loop import (DIAGNOSTIC_BASE, EXPLORE_END, EXPLORE_START, IterReport,
                   LABELS_PER_ITER, TrainState, _collect, _split_by_decision)
from ..actors import BuyerNet, SellerNet


def explore_of(day, *, n_days: int) -> float:
    """ε 선형 감쇠 — **날 번호**로 정한다. 연결용 첫날도 탐색은 한다."""
    if n_days <= 1:
        return EXPLORE_END
    f = min(1.0, max(0.0, day.index / (n_days - 1)))
    return EXPLORE_START + (EXPLORE_END - EXPLORE_START) * f


def run_month_training(*, seed: int = DIAGNOSTIC_BASE + 700,
                       n_days: int = N_DAYS,
                       labels_per_day: int = LABELS_PER_ITER,
                       out_dir: str | Path = "outputs/v3/month",
                       workers: int = 1, val_frac: float = 0.2,
                       days=None, log=print) -> tuple[TrainState, object]:
    """30일을 한 번에 굴리며 **날마다** 학생을 갱신한다.

    돌려주는 것: (학습 상태, `MonthResult`). 중간보고는 `log` 로 나간다.
    """
    # ★학습은 **비판정(진단) 대역**에서만 돈다 — 판정 대역을 학습에 쓰면
    #   [[YR-210]] 의 "새 대역·재사용 금지" 계약이 깨진다.
    if (int(seed) // 100_000) * 100_000 != DIAGNOSTIC_BAND:
        raise ValueError(
            f"학습 시드 {seed:,} 가 진단 대역({DIAGNOSTIC_BAND:,}~)이 아니다 — "
            f"판정 대역을 학습에 쓰면 그 대역이 오염된다")
    out = Path(out_dir)
    s_net, b_net = SellerNet(), BuyerNet()
    st = TrainState(s_net, b_net, StudentTrainer(s_net, b_net))
    days = list(days) if days else plan_month(seed, n_days=n_days)
    n = len(days)
    log(f"■ 30일 무대 · 시드 {seed:,} · 학습 {sum(d.is_train for d in days)}일")
    log(f"  {summarize(days)}")
    t_start = time.time()
    marks = {"t": t_start}

    def on_fit(day, rows) -> dict:
        """그날 라벨로 학생을 갱신한다. **다음 날은 새 정책으로 간다.**"""
        col, zero = _collect(rows)
        ls = col.result()
        if not ls.seller:
            if day.is_train:
                raise RuntimeError(
                    f"{day.index}일차 판매 표본 0 — 학습 신호가 없다 "
                    f"(즉시 중단 · 06 하드가드). 라벨 {len(rows)}건")
            return {"n_seller": 0, "note": "연결용 날 — 표본 없음"}
        tr, va = _split_by_decision(ls, val_frac)
        st.trainer.steps_per_iter = max(50, min(600, 4 * len(tr.seller)))
        m = st.trainer.fit(tr, seed=day.seed)
        vs, vb = st.trainer.evaluate(va) if va.seller else (0.0, 0.0)
        return {"n_seller": len(ls.seller), "n_buyer": len(ls.buyer),
                "zero_ratio": (zero / len(rows) if rows else 0.0),
                "steps": st.trainer.steps_per_iter,
                "seller_loss": m.seller_loss, "buyer_loss": m.buyer_loss,
                "val_seller_loss": float(vs), "val_buyer_loss": float(vb),
                "n_val": len(va.seller)}

    def on_day(rep) -> None:
        now = time.time()
        f = rep.fit or {}
        log(f"[{rep.index:>2}일] {rep.label} 부하 {rep.load:>6,} "
            f"ε{rep.explore:.2f} {(now - marks['t'])/60:>5.1f}분 "
            f"· Φ {rep.phi_krw:>14,.0f}원 "
            f"· 트럭 {rep.n_trucks:>5} (미완 {rep.n_censored:>4}) "
            f"· 거래 {rep.traded:>4}(공간 {rep.n_space}·시간 {rep.n_time}) "
            f"· 라벨 {rep.n_labels:>3} "
            f"· 손실 {f.get('seller_loss', 0):.5f}/{f.get('buyer_loss', 0):.5f} "
            f"· 검증 {f.get('val_seller_loss', 0):.5f}/{f.get('val_buyer_loss', 0):.5f}"
            f"{'' if rep.train else '  (연결용)'}")
        # ★`st.history` 를 채운다 — 안 채우면 `TrainState.save` 가 **빈 history.json**
        #   을 쓴다. 파일이 있는데 비어 있으면 다음 사람이 "학습이 안 됐나" 로 읽는다.
        #   하루 무대의 `IterReport` 와 **같은 칸**을 쓴다(세대 비교가 되게).
        st.history.append(IterReport(
            it=rep.index, load=rep.load, seed=seed, explore=rep.explore,
            secs=(now - marks["t"]), phi_rl=rep.phi_krw,
            n_labels=rep.n_labels, n_seller=int(f.get("n_seller", 0)),
            n_buyer=int(f.get("n_buyer", 0)),
            zero_label_ratio=float(f.get("zero_ratio", 0.0)),
            traded_edges=rep.traded, n_space=rep.n_space, n_time=rep.n_time,
            seller_loss=float(f.get("seller_loss", 0.0)),
            buyer_loss=float(f.get("buyer_loss", 0.0)),
            val_seller_loss=float(f.get("val_seller_loss", 0.0)),
            val_buyer_loss=float(f.get("val_buyer_loss", 0.0)),
            n_val=int(f.get("n_val", 0)), worlds=rep.worlds))
        marks["t"] = now
        st.save(out, rep.index)
        (out / "days.json").write_text(
            json.dumps([r.as_dict() for r in _live_so_far(rep)],
                       ensure_ascii=False, indent=1), encoding="utf-8")

    seen: list = []

    def _live_so_far(rep):
        seen.append(rep)
        return seen

    res = run_month(seed=seed, arm="RL", seller_net=s_net, buyer_net=b_net,
                    days=days, labels_per_day=labels_per_day, workers=workers,
                    explore_of_day=lambda d: explore_of(d, n_days=n),
                    on_fit=on_fit, on_day=on_day)

    log(f"■ 끝 — {(time.time() - t_start)/3600:.2f}시간")
    _report_by_load(res, log)
    (out / "month.json").write_text(
        json.dumps({"seed": seed, "plan": res.plan,
                    "days": [d.as_dict() for d in res.days],
                    "live": [d.as_dict() for d in res.live]},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    return st, res


def _report_by_load(res, log) -> None:
    """★부하별 판정 표 (사용자 지시 2026-08-26) — **표본이 얇으면 그 사실도 적는다**."""
    log("■ 부하별 (학습 28일 · 확정 Φ)")
    for load, ds in res.by_load().items():
        phis = sorted(d.phi_krw for d in ds)
        mid = phis[len(phis) // 2]
        thin = " ⚠️표본 얇음" if len(ds) < 4 else ""
        log(f"  부하 {load:>6,} · {ds[0].label} · {len(ds):>2}일 "
            f"· 중앙 Φ {mid:>15,.0f}원 · 평균 트럭 "
            f"{sum(d.n_trucks for d in ds) / len(ds):>7,.0f}대{thin}")
