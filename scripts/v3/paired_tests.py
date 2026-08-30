"""날짜별 짝비교 검정 — 부호검정과 윌콕슨 부호순위검정을 **한 자리에서** 만든다.

    PYTHONPATH=src python scripts/v3/paired_tests.py outputs/v3/judge-30d
    PYTHONPATH=src python scripts/v3/paired_tests.py outputs/v3/judge-locked

■ 왜 이 스크립트가 필요한가
  논문 §5.3 은 *"윌콕슨 부호순위검정도 같은 결론을 준다"* 고 적어 놓고 **p 값을 하나도
  싣지 않았다**. 심사자가 확인할 수 없고, 그 문장이 참인지도 아무도 계산해 본 적이 없다.
  여기서 두 검정을 같은 자료로 함께 계산해 표에 그대로 옮길 수 있는 형태로 낸다.

■ 두 검정은 서로 다른 질문에 답한다 — 그래서 둘 다 싣는다
  · **부호검정**: "어느 쪽이 싼 날이 며칠인가" 만 센다. 하루의 차이가 1만원이든
    3억원이든 한 표다. 크기에 완전히 눈을 감는다.
  · **윌콕슨 부호순위검정**: 차이를 **크기 순으로 매긴 뒤** 부호별로 합한다. 이기는
    날이 적어도 크게 이기면 점수가 올라간다.
  이 논문의 주장 자체가 *"자주 이기는 게 아니라 혼잡한 날에 크게 줄인다"* 이므로,
  부호검정 하나만으로는 주장한 성질을 원리적으로 볼 수 없다.

■ 층화 부트스트랩 (§5.6)
  하루 평균 차이의 95% 구간을 **수요 수준별로 나눠** 되뽑아 만든다. 설계가 고정한
  수요 구성(12·8·4·2·2일)을 재표본이 흐트러뜨리지 않게 하려는 것이다. 되뽑기 횟수와
  난수 시드를 여기서 못박아 재현 가능하게 한다.

■ 부호 규약
  차이 = (비교 정책 비용) − (제안 정책 비용). **양수면 제안 정책이 싸다.**
  두 검정 다 양측이므로 부호 규약이 p 값을 바꾸지는 않는다.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # 콘솔이 cp949 여도 깨지지 않게

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scipy import stats

from figdata import month

MN = 1e6                       # 백만원
N_BOOT = 20_000                # 되뽑기 횟수 — 논문에 적는 수와 같아야 한다
BOOT_SEED = 20260830           # 부트스트랩 난수 시드 (재현용)

#: 표에 싣는 순서와 이름 — 논문 `tab:paired` 와 같은 순서로 낸다.
ROWS = (
    ("RL_EARLY", "Untrained model", "Learning effect"),
    ("SPACE_TIME_LL", "Least-loaded block + slot rule", "Same action range"),
    ("LEAST_SLACK", "Least slack", "Block rule"),
    ("FCFS", "First-come-first-served", "Block rule"),
    ("SPT", "Shortest processing time", "Block rule"),
    ("NETGAIN", "Net gain", "Block rule"),
    ("NO_REALLOC", "No reallocation", "Baseline"),
)


def load(judge_dir: pathlib.Path):
    """arm 별 날짜→비용과 판정 시드를 읽는다."""
    arms = {}
    for f in sorted((judge_dir / "arms").glob("arm_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        arms[d["arm"]] = {int(k): v for k, v in d["phi_by_day"].items()}
    seed = json.loads(next(judge_dir.glob("judge_*.json")).read_text(encoding="utf-8"))["seed"]
    days = [i for i in sorted(arms["RL"]) if 1 <= i <= 28]
    return arms, days, seed


def sign_test(diff) -> tuple[int, int, float]:
    """양측 부호검정 — 무승부(차이 0)는 표본에서 뺀다 (정확 이항)."""
    pos = int(sum(1 for d in diff if d > 0))
    neg = int(sum(1 for d in diff if d < 0))
    n = pos + neg
    p = stats.binomtest(pos, n, 0.5, alternative="two-sided").pvalue
    return pos, n, p


def signed_rank(diff) -> float:
    """양측 윌콕슨 부호순위검정 — 차이의 크기까지 쓴다.

    ★`method="exact"` 를 **못박는다**. 28일에 동점(차이 0)이 없으므로 정확검정이
    가능하고, 근사식을 쓰면 값이 달라진다 (예: 판정 대역 순이득 비교에서 정확
    0.0017 vs 정규근사+연속성보정 0.0026). 결론은 같지만 논문에 싣는 수는 하나여야
    하므로 재현 가능한 정확검정으로 고정한다.
    """
    return float(stats.wilcoxon(diff, alternative="two-sided", method="exact").pvalue)


def boot_ci(diff, strata=None, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """하루 평균 차이의 95% 백분위 부트스트랩 구간.

    `strata` 를 주면 **층화** 부트스트랩 — 수요 수준별로 그 층의 날짜 안에서만
    되뽑아 설계가 정한 수요 구성을 보존한다.
    """
    rng = np.random.default_rng(seed)
    d = np.asarray(diff, dtype=float)
    if strata is None:
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        means = d[idx].mean(axis=1)
    else:
        groups = {}
        for i, s in enumerate(strata):
            groups.setdefault(s, []).append(i)
        means = np.zeros(n_boot)
        for g in groups.values():
            gi = np.asarray(g)
            pick = rng.integers(0, len(gi), size=(n_boot, len(gi)))
            means += d[gi[pick]].sum(axis=1)
        means /= len(d)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi), float((means < 0).mean())


def report(judge_dir: pathlib.Path):
    arms, days, seed = load(judge_dir)
    load_of = {p.index: p.load for p in month().plan_month(seed)}
    print(f"■ {judge_dir}  (시드 {seed:,} · {len(days)}일)")
    print(f"{'비교 정책':30s} {'제안이 싼 날':>12s} {'부호검정 p':>12s} "
          f"{'부호순위 p':>12s} {'중앙값(백만원)':>14s}")
    latex = []
    for arm, name, purpose in ROWS:
        if arm not in arms:
            print(f"{name:30s} {'— 자료 없음 —':>12s}")
            continue
        diff = [(arms[arm][i] - arms["RL"][i]) / MN for i in days]
        pos, n, p_sign = sign_test(diff)
        p_rank = signed_rank(diff)
        med = float(np.median(diff))
        print(f"{name:30s} {f'{pos}/{n}':>12s} {p_sign:12.4f} {p_rank:12.4f} {med:14.1f}")
        latex.append(f"{purpose} & {name} & {pos}/{n} & {fmt_p(p_sign)} & {fmt_p(p_rank)} \\\\")

    print("\n[LaTeX 행 — tab:paired 에 그대로]")
    for row in latex:
        print("  " + row)

    # ── §5.6 학습 기여의 부트스트랩 구간 ─────────────────────────
    if "RL_EARLY" in arms:
        diff = [(arms["RL_EARLY"][i] - arms["RL"][i]) / MN for i in days]
        strata = [load_of[i] for i in days]
        s_lo, s_hi, _ = boot_ci(diff, strata)
        u_lo, u_hi, u_neg = boot_ci(diff)
        print(f"\n[학습 기여 · 하루 평균 차이 {np.mean(diff):.1f} 백만원 "
              f"(되뽑기 {N_BOOT:,}회 · 시드 {BOOT_SEED})]")
        print(f"  층화 95% 구간   [{s_lo:.1f}, {s_hi:.1f}] 백만원/일")
        print(f"  비층화 95% 구간 [{u_lo:.1f}, {u_hi:.1f}] 백만원/일 "
              f"· 재표본의 {1 - u_neg:.1%} 가 양수(제안이 싼 쪽)")


def fmt_p(p: float) -> str:
    """표에 넣을 p — 0.05 미만은 굵게, 아주 작으면 부등호로."""
    if p < 0.001:
        return r"\textbf{$<$0.001}"
    return (r"\textbf{%.3f}" % p) if p < 0.05 else ("%.3f" % p)


if __name__ == "__main__":
    targets = sys.argv[1:] or ["outputs/v3/judge-30d"]
    for t in targets:
        report(ROOT / t if not pathlib.Path(t).is_absolute() else pathlib.Path(t))
        print()
