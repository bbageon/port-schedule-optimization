"""크레인 학습 루프 — 하루 → 라벨 → 망 갱신 → 반복 ([[YR-308]]).

재배정층 `train/loop.py` 와 **같은 골격·같은 설정**이다. 다른 것은 셋뿐:

    ① 라벨이 크레인 결정에서 나온다 (`crane_budget`)
    ② 재배정을 **끈다** (`arm="NO_REALLOC"`) — 한 번에 한 축
    ③ 기준선이 `SF_SPT` 다 — *"규칙 크레인보다 나은가"* 가 이 축의 질문이다

■ ★판정 대역을 쓰지 않는다
  학습은 **비판정 대역(9,900,0xx)** 에서만 돈다. 여기 Φ 는 진단이고 주장이 아니다.

■ ★라벨은 회차마다 버린다
  반사실 라벨에는 유효기간이 있다 — 망이 바뀌면 *"차점자"* 가 딴 후보가 되므로
  지난 회차 라벨은 이미 없는 비교의 답이다. 버퍼를 두지 않는다.

■ ⚠️ 탐색이 없다
  재배정층은 ε 로 무작위 행동을 섞는데, 크레인층은 **아직 안 섞는다**. 이유는
  차점자 규약이다 — 대안이 이미 *"망이 2등으로 본 것"* 이라 탐색이 그 안에 있다.
  이게 모자란지는 라벨 0비율(`zero`)로 본다: 높아지면 비교가 뻔해진 것이다.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from ..eval import TRAIN_LOADS
from ..stage import RolloutBudget, run_episode
from .fit import CraneFitReport, CraneTrainer, scale_health
from .policy import CraneNet

#: 비판정(진단) 시드 대역 — 판정 대역은 여기서 절대 쓰지 않는다.
DIAGNOSTIC_BASE = 9_900_000

#: 회차당 라벨 표본 수. 재배정층과 **같은 규모**로 맞췄다 — 축이 달라서 예산이
#: 다르면 "어느 축이 나은가" 가 예산 차이인지 축 차이인지 못 가린다.
LABELS_PER_ITER = 64

#: ★고정 평가일 — 학습에 **안 쓰는** 날. 매번 **같은 날**을 굴린다.
#:
#: 왜 필요한가 (2026-09-10 첫 10회차의 교훈):
#:   회차마다 시드가 달라 *"부하 12,500 이 +69.22% → +1.81% 로 좋아졌다"* 를 봐도
#:   **학습 덕인지 그날이 쉬웠는지 못 가린다.** 짝이 안 맞는 비교였다.
#:   같은 날을 두 정책으로 굴려야 그날의 난이도가 상쇄된다.
EVAL_SEED_BASE = DIAGNOSTIC_BASE + 900
EVAL_LOADS = (3_500, 7_500, 15_000)


@dataclass
class CraneIterReport:
    """회차 하나의 진단. **전부 비판정 대역 수치다.**"""

    it: int = 0
    load: int = 0
    seed: int = 0
    secs: float = 0.0
    phi_rl: float = 0.0              # 학습 크레인
    phi_rule: float = 0.0            # 같은 시드 SF_SPT — 이 축의 기준선
    gap: float = 0.0                 # phi_rl − phi_rule (음수 = 학습이 이득)
    gap_ratio: float = 0.0
    #: 표본 회계 — **버린 것까지** 남긴다. 라벨 수만 보면 "64건 뽑았다" 가
    #: 실제로 12건이어도 안 보인다.
    sampled: int = 0
    no_alt: int = 0
    no_pick: int = 0
    factual_mismatch: int = 0
    force_failed: int = 0
    n_labels: int = 0                # 학습에 실제로 쓴 표본 수
    zero_label_ratio: float = 0.0
    worlds: int = 0
    loss: float = 0.0
    val_loss: float = 0.0            # ★진짜 지표 — 안 본 표본에서의 오차
    n_val: int = 0
    median_gap_krw: float = 0.0      # 목표 중앙 격차 — 그날 라벨이 얼마짜리였나
    #: ★그날 실제로 쓴 눈금(원) ([[YR-309]]). 회차마다 다르다 — 그게 이 변경의 요점이다.
    scale_krw: float = 0.0
    floor_bound: bool = False        # 하한이 걸렸나 = 그날은 비용 중립에 가까웠다
    scale: str = ""
    #: ★날별로 나눈 검증 손실 — **회차끼리 비교하려면 이 값이어야 한다.**
    #:  날마다 목표 눈금이 3,000~42,000원(14배)까지 벌어져서, 날것 손실은
    #:  "학습이 나빠졌다" 와 "그날 격차가 컸다" 를 못 가린다 (2026-09-10 실측).
    val_loss_norm: float = 0.0

    def line(self) -> str:
        return (f"[{self.it:>3}] 부하 {self.load:,} {self.secs/60:>5.1f}분 "
                f"· 라벨 {self.n_labels:>3}/{self.sampled:<3}"
                f"(대안없음 {self.no_alt:>3}·불일치 {self.factual_mismatch:>2}"
                f"·강제실패 {self.force_failed:>2}·0비율 {self.zero_label_ratio:>5.1%}) "
                f"· 격차 {self.gap:>+14,.0f} ({self.gap_ratio:>+6.2%}) "
                f"· 손실 {self.loss:.5f} · 검증 {self.val_loss:.5f} "
                f"· 눈금 {self.scale_krw:>8,.0f}원"
                f"{'(하한)' if self.floor_bound else '      '} {self.scale}")


@dataclass
class CraneTrainState:
    net: CraneNet
    trainer: CraneTrainer
    history: list = field(default_factory=list)
    #: 고정 평가일 결과 — 회차별로 쌓인다
    evals: list = field(default_factory=list)

    def save(self, path: Path, it: int) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"crane": self.net.state_dict(), "it": it},
                   path / f"crane_{it:03d}.pt")
        (path / "history.json").write_text(
            json.dumps([asdict(h) for h in self.history], ensure_ascii=False,
                       indent=1), encoding="utf-8")
        (path / "evals.json").write_text(
            json.dumps(self.evals, ensure_ascii=False, indent=1),
            encoding="utf-8")


def evaluate(net, *, loads=EVAL_LOADS, seed_base: int = EVAL_SEED_BASE,
             workers: int = 1) -> dict:
    """고정 평가일을 **같은 시드로** RL·규칙 각각 굴려 짝비교한다.

    ★짝비교여야 하는 이유: 날마다 난이도가 다르다. 다른 날끼리 견주면 정책 차이와
      날 차이가 섞여 *"학습이 됐나"* 를 물을 수 없다.

    교사를 안 붙이므로 반사실 세계가 안 뜬다 — 하루 굴리는 값만 든다.
    """
    rows = []
    for i, load in enumerate(loads):
        seed = int(seed_base) + i * 10 + load // 1000
        rl = run_episode(load=load, arm="NO_REALLOC", dispatcher="RL_CRANE",
                         seed=seed, crane_net=net)
        rule = run_episode(load=load, arm="NO_REALLOC", dispatcher="SF_SPT",
                           seed=seed)
        rows.append({"load": load, "seed": seed,
                     "phi_rl": rl.phi_krw, "phi_rule": rule.phi_krw,
                     "gap": rl.phi_krw - rule.phi_krw,
                     "gap_ratio": (rl.phi_krw - rule.phi_krw)
                     / max(1e-9, rule.phi_krw)})
    ratios = sorted(r["gap_ratio"] for r in rows)
    return {"rows": rows, "median_gap_ratio": ratios[len(ratios) // 2],
            "n_win": sum(1 for r in rows if r["gap_ratio"] < 0)}


def _eval_line(it: int, ev: dict) -> str:
    per = " ".join(f"{r['load']//1000}k {r['gap_ratio']:+.2%}" for r in ev["rows"])
    return (f"    ▸ 고정 평가일 [{it:>3}] 중앙 {ev['median_gap_ratio']:+.2%} · "
            f"이긴 날 {ev['n_win']}/{len(ev['rows'])} · {per}")


def run_crane_training(*, iters: int = 20,
                       out_dir: str | Path = "outputs/v4/crane-train",
                       labels_per_iter: int = LABELS_PER_ITER,
                       time_budget_s: float | None = None,
                       seed_base: int = DIAGNOSTIC_BASE + 700,
                       loads: tuple[int, ...] = TRAIN_LOADS,
                       horizon_s: float = 10_800.0,
                       workers: int = 1, val_frac: float = 0.2,
                       eval_every: int = 5, log=print) -> CraneTrainState:
    """회차를 돌린다. **표본 0 이면 즉시 멈춘다** (06 하드가드).

    `time_budget_s` 를 주면 시간으로도 끊는다 — 회차 수를 **결과 보고 늘리면
    사전등록이 무너지므로** 예산을 먼저 정해 두고 그 안에서 돈다.
    """
    out = Path(out_dir)
    #: ★망 초기값을 시드에 묶는다 ([[YR-304]] 의 교훈). 안 묶으면 학습 전 팔이
    #:  실행마다 다른 정책이라 같은 시드로 돌려도 Φ 가 재현되지 않는다.
    torch.manual_seed(int(seed_base) + 3)
    net = CraneNet()
    st = CraneTrainState(net, CraneTrainer(net))
    t_start = time.time()

    def do_eval(it: int) -> None:
        """고정 평가일로 **짝비교**한다. 이게 없으면 학습 여부를 못 묻는다."""
        if eval_every <= 0:
            return
        ev = evaluate(net, workers=workers)
        ev["it"] = it
        st.evals.append(ev)
        log(_eval_line(it, ev))

    do_eval(-1)          # ★학습 **전** 기준점 — 없으면 나중 값을 견줄 데가 없다

    for it in range(iters):
        if time_budget_s is not None and time.time() - t_start > time_budget_s:
            log(f"시간 예산 {time_budget_s/3600:.1f}시간 소진 — {it} 회차에서 종료")
            break
        load = loads[it % len(loads)]
        seed = seed_base + it * 10 + (load // 1000)
        t0 = time.time()

        budget = RolloutBudget(max_labels=labels_per_iter, identity_checks=0)
        ep = run_episode(load=load, arm="NO_REALLOC", dispatcher="RL_CRANE",
                         seed=seed, crane_net=net, crane_budget=budget,
                         horizon_s=horizon_s, workers=workers)
        # 같은 시드 규칙 크레인 — 이 축의 기준선(교사 없이 돌아 싸다)
        base = run_episode(load=load, arm="NO_REALLOC", dispatcher="SF_SPT",
                           seed=seed)

        cs = ep.crane_stats
        rep = CraneIterReport(
            it=it, load=load, seed=seed,
            phi_rl=ep.phi_krw, phi_rule=base.phi_krw,
            gap=ep.phi_krw - base.phi_krw,
            gap_ratio=(ep.phi_krw - base.phi_krw) / max(1e-9, base.phi_krw),
            sampled=cs.get("sampled", 0), no_alt=cs.get("no_alt", 0),
            no_pick=cs.get("no_pick", 0),
            factual_mismatch=cs.get("factual_mismatch", 0),
            force_failed=cs.get("force_failed", 0),
            worlds=cs.get("worlds", 0),
            n_labels=cs.get("labeled", 0),
            zero_label_ratio=(cs.get("zero", 0) / max(1, cs.get("labeled", 0))))

        if not ep.crane_labels:
            rep.secs = time.time() - t0
            st.history.append(rep)
            st.save(out, it)
            raise RuntimeError(
                f"{it} 회차 크레인 표본 0 — 학습 신호가 없다(즉시 중단·06 하드가드). "
                f"표본 {rep.sampled} · 대안없음 {rep.no_alt} · "
                f"사실불일치 {rep.factual_mismatch} · 강제실패 {rep.force_failed}")

        ls = _labelset(ep.crane_labels, cs)
        fit: CraneFitReport = st.trainer.fit(ls, seed=seed, val_frac=val_frac)
        rep.loss, rep.val_loss, rep.n_val = fit.loss, fit.val_loss, fit.n_val
        rep.median_gap_krw = fit.median_gap_krw
        rep.scale_krw, rep.floor_bound = fit.scale_krw, fit.floor_bound
        #: ★그날 눈금 기준으로 본다 — [[YR-309]] 이후 이 값은 **항상 1.00 근처**여야
        #:  한다(하한이 걸린 날만 예외). 사실상 불변식 검사다.
        rep.scale = scale_health(fit.median_gap_krw, fit.scale_krw)
        #: 검증손실은 눈금이 이미 날별로 맞춰져 있어 그대로 비교 가능하다.
        #: 그래도 하한이 걸린 날은 어긋나므로 한 번 더 나눠 남긴다.
        rep.val_loss_norm = rep.val_loss / max(
            1e-9, fit.median_gap_krw / max(1e-9, fit.scale_krw))
        rep.secs = time.time() - t0
        st.history.append(rep)
        st.save(out, it)
        log(rep.line())
        if eval_every > 0 and (it + 1) % eval_every == 0:
            do_eval(it)

    return st


def _labelset(samples, stats):
    """에피소드가 낸 표본 → 학습기가 먹는 그릇 (얇은 어댑터)."""
    from .teacher import CraneLabelSet
    from .policy import CRANE_ADV_SCALE as _DEFAULT   # 회계에 눈금이 없을 때만
    ls = CraneLabelSet()
    ls.samples = list(samples)
    ls.worlds = int(stats.get("worlds", 0))
    ls.zero = int(stats.get("zero", 0))
    #: ★에피소드가 실은 그날 눈금을 그대로 받는다 ([[YR-309]]) — 여기서 다시 계산하면
    #:  하한 처리가 두 곳에 생겨 언젠가 갈라진다.
    ls.scale = float(stats.get("scale", _DEFAULT))
    ls.median_gap_krw = float(stats.get("median_gap_krw", 0.0))
    ls.floor_bound = bool(stats.get("floor_bound", False))
    return ls
