"""학습 루프 — 에피소드 → 라벨 → 망 갱신 → 반복 ([[YR-218]]).

설계 정본: `.claude/docs/architecture/06-학습과-판정.md` §1 · `04b-학습-잣대.md` §3

■ 이게 없어서 정책을 한 번도 학습시킨 적이 없었다
  교사([[YR-204]])·수집기·학습기는 다 있었는데 **돌리는 코드가 없었다.**
  그때까지의 모든 수치는 무작위 초기화 망이었다.

■ 한 회차가 하는 일
      ① 학습 부하 하나를 골라 하루를 굴린다 (교사 붙임 · 라벨 K건 표본)
      ② 같은 시드로 **안 팔기**도 굴린다 — 짝 격차를 진단으로 본다
      ③ 라벨을 학생 두 망에 먹인다 (같은 눈금·한 옵티마이저 스텝)
      ④ 체크포인트와 진단을 남긴다

■ ★라벨은 회차마다 버린다
  반사실 라벨에는 **유효기간**이 있다 — 상대가 학습하면 내 라벨이 낡는다(04b §3).
  v2 의 150,000 FIFO 버퍼는 라벨이 **직접 관측**이라 안 상했기에 가능했다.
  여기서는 회차가 끝나면 그 라벨을 버린다. 버퍼가 없다.

■ ★탐색은 ε 로 어닐링한다
  [[YR-193]] 의 교훈은 *"목표 눈금이 변하는데 잡음을 절대값으로 고정하지 마라"* 였다
  (v2 는 σ 를 절대값 0.20 으로 박아 눈금이 줄자 상대 강도가 8.7배가 됐다).
  우리 정책은 이산 argmin 이라 탐색이 **ε(무작위로 고를 확률)** 이고, ε 는 애초에
  눈금과 무관하다 — 그 실패 모드가 구조적으로 없다. 회차에 따라 선형 감쇠만 한다.

■ ★판정 대역을 쓰지 않는다
  학습은 **비판정 대역(9,900,0xx)** 에서만 돈다. 여기서 나온 Φ 수치는 **진단**이고
  논문 주장이 아니다 — 판정은 [[YR-210]] 이 새 대역에서 한 번만 한다.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch

from ..actors import BuyerNet, SellerNet
from ..eval import TRAIN_LOADS
from ..stage import RolloutBudget, run_episode
from .fit import StudentTrainer
from .labels import LabelCollector

#: 비판정(진단) 시드 대역 — 판정 대역은 여기서 절대 쓰지 않는다.
DIAGNOSTIC_BASE = 9_900_000

#: 탐색 ε — 시작·끝. 선형 감쇠.
EXPLORE_START = 0.50
EXPLORE_END = 0.05

#: 회차당 라벨 표본 수. 전수는 불가능하다(회차당 14~30시간 · [[YR-215]]).
LABELS_PER_ITER = 64


@dataclass
class IterReport:
    """회차 하나의 진단. **전부 비판정 대역 수치다.**"""

    it: int = 0
    load: int = 0
    seed: int = 0
    explore: float = 0.0
    secs: float = 0.0
    phi_rl: float = 0.0
    phi_keep: float = 0.0            # 같은 시드 안 팔기 — 짝 격차의 기준
    gap: float = 0.0                 # phi_rl − phi_keep (음수 = 재배치가 이득)
    gap_ratio: float = 0.0
    n_labels: int = 0
    n_seller: int = 0
    n_buyer: int = 0
    zero_label_ratio: float = 0.0    # 라벨 0 비율 — 무너지면 [[YR-217]] 재발
    traded_edges: int = 0
    n_space: int = 0
    n_time: int = 0
    seller_loss: float = 0.0
    buyer_loss: float = 0.0
    #: ★검증 손실 — **학습에 안 쓴 표본**에서의 오차.
    #: 학습 손실만 보면 과적합을 못 본다: 망 5,761 파라미터에 표본 80개면
    #: 본 것은 거의 완벽히 맞히면서 처음 보는 상황에서는 크게 틀릴 수 있다.
    #: argmin 이 실전에서 쓰는 것은 **처음 보는 상황**이므로 이쪽이 진짜 오차다.
    val_seller_loss: float = 0.0
    val_buyer_loss: float = 0.0
    n_val: int = 0
    worlds: int = 0

    def line(self) -> str:
        return (f"[{self.it:>3}] 부하 {self.load:,} ε{self.explore:.2f} "
                f"{self.secs/60:>5.1f}분 · 라벨 {self.n_labels:>3}"
                f"(판매 {self.n_seller:>3}·구매 {self.n_buyer:>3}·0비율 "
                f"{self.zero_label_ratio:>5.1%}) · 거래 {self.traded_edges:>4} "
                f"· 격차 {self.gap:>+14,.0f} ({self.gap_ratio:>+6.2%}) "
                f"· 손실 {self.seller_loss:.5f}/{self.buyer_loss:.5f}"
                f" · 검증 {self.val_seller_loss:.5f}/{self.val_buyer_loss:.5f}")


@dataclass
class TrainState:
    """루프 전체 상태. 체크포인트로 저장·복원한다."""

    seller_net: SellerNet
    buyer_net: BuyerNet
    trainer: StudentTrainer
    history: list[IterReport] = field(default_factory=list)

    def save(self, path: Path, it: int) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"seller": self.seller_net.state_dict(),
                    "buyer": self.buyer_net.state_dict(), "it": it},
                   path / f"ckpt_{it:03d}.pt")
        (path / "history.json").write_text(
            json.dumps([asdict(h) for h in self.history], ensure_ascii=False,
                       indent=1), encoding="utf-8")


def explore_at(it: int, total: int) -> float:
    """ε 선형 감쇠 — 모듈 머리의 [[YR-193]] 주석 참조."""
    if total <= 1:
        return EXPLORE_END
    f = min(1.0, max(0.0, it / (total - 1)))
    return EXPLORE_START + (EXPLORE_END - EXPLORE_START) * f


def _collect(labels: list[dict]) -> tuple[LabelCollector, int]:
    """에피소드가 낸 라벨 행 → 학생이 먹을 표본.

    ★교사가 **굴린 두 행에만** 라벨이 붙는다 — 나머지 후보는 굴리지 않았으니
    목표가 없다(굴리면 rollout 이 그만큼 는다).
    """
    col = LabelCollector()
    zero = 0
    for row in labels:
        se = row["seller"]
        gap = abs(row["phi_seller_alt"] - row["phi_factual"])
        zero += gap < 1e-6
        keys = se.get("coord_keys") or []
        alt_coord = row.get("seller_alt_coord")
        alt_idx = keys.index(alt_coord) if alt_coord in keys else (
            0 if row["seller_alt"] == "KEEP" else None)
        col.add_seller(se, phi_factual=row["phi_factual"],
                       phi_alt=row["phi_seller_alt"], alt_index=alt_idx)
        col.note_worlds(row["worlds"])
        if "phi_buyer_alt" in row:
            col.add_buyer(row["buyer"], phi_factual=row["phi_factual"],
                          phi_alt=row["phi_buyer_alt"])
    return col, zero


def _split_by_decision(ls, val_frac: float):
    """라벨을 학습/검증으로 가른다 — **결정 단위**로.

    같은 결정의 두 행(사실·대안)은 서로의 답을 담고 있다(중심화라 부호만 반대).
    행 단위로 자르면 검증 답을 학습에서 본 셈이 되어 과적합이 안 잡힌다.
    """
    from .labels import LabelSet

    keys = sorted({s.doc_key for s in ls.seller})
    n_val = max(1, int(round(len(keys) * val_frac))) if keys else 0
    stride = max(1, len(keys) // max(1, n_val))
    val_keys = set(keys[::stride][:n_val])
    tr, va = LabelSet(), LabelSet()
    for s in ls.seller:
        (va if s.doc_key in val_keys else tr).seller.append(s)
    for s in ls.buyer:
        (va if s.doc_key in val_keys else tr).buyer.append(s)
    tr.worlds = va.worlds = ls.worlds
    return tr, va


def run_training(*, iters: int = 20, out_dir: str | Path = "outputs/v4/train",
                 labels_per_iter: int = LABELS_PER_ITER,
                 time_budget_s: float | None = None,
                 seed_base: int = DIAGNOSTIC_BASE + 600,
                 loads: tuple[int, ...] = TRAIN_LOADS,
                 workers: int = 1, val_frac: float = 0.2,
                 log=print) -> TrainState:
    """회차를 돌린다. **표본 0 이면 즉시 멈춘다**(06 하드가드).

    `time_budget_s` 를 주면 시간으로도 끊는다 — 회차 수를 **결과 보고 늘리면
    사전등록이 무너지므로**, 예산을 먼저 정해 두고 그 안에서 돈다.
    """
    out = Path(out_dir)
    s_net, b_net = SellerNet(), BuyerNet()
    st = TrainState(s_net, b_net, StudentTrainer(s_net, b_net))
    t_start = time.time()

    for it in range(iters):
        if time_budget_s is not None and time.time() - t_start > time_budget_s:
            log(f"시간 예산 {time_budget_s/3600:.1f}시간 소진 — {it} 회차에서 종료")
            break
        load = loads[it % len(loads)]
        seed = seed_base + it * 10 + (load // 1000)
        eps = explore_at(it, iters)
        t0 = time.time()

        budget = RolloutBudget(max_labels=labels_per_iter, identity_checks=0)
        ep = run_episode(load=load, arm="RL", seed=seed, budget=budget,
                         explore=eps, seller_net=s_net, buyer_net=b_net,
                         workers=workers)
        # 같은 시드 안 팔기 — 짝 격차의 기준선(교사 없이 돌아 싸다)
        base = run_episode(load=load, arm="NO_REALLOC", seed=seed)

        col, zero = _collect(ep.labels)
        ls = col.result()
        rep = IterReport(
            it=it, load=load, seed=seed, explore=eps,
            phi_rl=ep.phi_krw, phi_keep=base.phi_krw,
            gap=ep.phi_krw - base.phi_krw,
            gap_ratio=(ep.phi_krw - base.phi_krw) / max(1e-9, base.phi_krw),
            n_labels=len(ep.labels), n_seller=len(ls.seller), n_buyer=len(ls.buyer),
            zero_label_ratio=(zero / len(ep.labels) if ep.labels else 0.0),
            traded_edges=ep.traded_edges, n_space=ep.n_space, n_time=ep.n_time,
            worlds=ep.rollout_worlds)

        if not ls.seller:
            rep.secs = time.time() - t0
            st.history.append(rep)
            st.save(out, it)
            raise RuntimeError(
                f"{it} 회차 판매 표본 0 — 학습 신호가 없다(즉시 중단·06 하드가드). "
                f"거래 {ep.traded_edges} · 결정 {ep.decisions}")

        # ★스텝 수를 표본에 맞춘다. v2 는 150,000 버퍼에 600 스텝이었는데, 여기 표본은
        #   회차당 수십 건이다(라벨에 유효기간이 있어 버퍼가 없다 — 04b §3).
        #   600 스텝을 그대로 쓰면 같은 몇 줄을 수백 번 되먹여 그 회차 라벨에 과적합한다.
        # ★표본을 학습/검증으로 가른다. 회차 안에서 **결정 단위**로 잘라야
        #   같은 결정의 두 행(사실·대안)이 양쪽에 흩어지지 않는다 — 흩어지면
        #   검증 표본의 답을 학습에서 이미 본 셈이 되어 과적합을 못 잡는다.
        tr_ls, va_ls = _split_by_decision(ls, val_frac)
        st.trainer.steps_per_iter = max(50, min(600, 4 * len(tr_ls.seller)))
        fit = st.trainer.fit(tr_ls, seed=seed)
        rep.seller_loss, rep.buyer_loss = fit.seller_loss, fit.buyer_loss
        rep.val_seller_loss, rep.val_buyer_loss = st.trainer.evaluate(va_ls)
        rep.n_val = len(va_ls.seller)
        rep.secs = time.time() - t0
        st.history.append(rep)
        st.save(out, it)
        log(rep.line())

    return st
