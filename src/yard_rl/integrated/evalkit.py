"""YR-106 — 판정 지표·검정력 공용 유틸 (설계감사 critical 대응) + YR-106-b 게이트 C 정정.

설계감사(2026-07-27) 결론: 주지표 `terminal_total` 의 **70%가 본선비용**이고 짝지은 분산의
대부분이 거기서 온다(sd 18.75 vs 트럭측 5.49). 그래서 총비용으로 유의판정하려면 시드가
약 97개 필요한데 전 판정이 n=8 로 수행됐다 — **"효과 없음"과 "이 지표로는 못 본다"가
구분되지 않았다.**

본 모듈이 강제하는 3가지:
  ① **채널 분해** — 비용을 truck/vessel/move/other 로 쪼개, 이송·배치 실험의 **1차 판정을
     트럭 채널**로 둔다(본선 잡음에 익사시키지 않는다). 총비용·본선은 병기 보고.
  ② **MDE(최소검출가능효과) 필수** + **동등성은 TOST** — CI 가 0 을 포함해도 "효과 없음"이
     아니다. 동등성은 사전 δ 에 대한 **두 개의 단측검정**으로만 주장한다.
  ③ **하드 guard 기계화** — 완주율·backlog·healthy 를 사람 눈이 아니라 함수가 막는다.

■ YR-106-b 정정 (게이트 C, 2026-07-28) — 세 가지 결함을 고쳤다
  (a) **t 값 하드코딩 표** → 정확한 분포 함수(`statfuncs`). 표에 없는 df 가 조용히
      대표값으로 대체되던 경로 제거 (df 21·22·24~28 이 0.920 으로 떨어져 MDE 7% 과대).
      **n=8(df=7)은 표에 있었으므로 과거 n=8 판정 수치는 불변**이다.
  (b) **필요표본수** — 상수 (1.96+0.84) 로 1회 계산하던 것을 **목표 n 의 자유도로 반복
      수렴**시키고, 파일럿 sd 의 불확실성을 χ² 상측한계로 반영한다(선택). 기존식은
      작은 파일럿에서 필요 n 을 **체계적으로 과소추정**했다.
  (c) **동등성 판정** — `MDE ≤ δ` 라는 임시 규칙은 동등성 검정이 아니다(평균이 δ 근처면
      CI 가 δ 를 넘어도 "효과 없음"이 나올 수 있었다). 표준 **TOST**(90% CI ⊂ (−δ,+δ))로 교체.

사용 규약: 사전등록에 `mde_note` 를 반드시 적고, 확증 대역 n 은 선택 대역의 **2배 이상**.
1차 판정 채널을 **사전 고정**하면 다중비교 보정이 불필요하다 — 나머지 채널은 탐색적 표기.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean, stdev

from ..contract import COST_TERMS
from .statfuncs import sd_upper_conf, t_ppf

# --- 채널 정의 (COST_TERMS 13항의 완전분할 — 합 = total_normalized) ---
CHANNELS: dict[str, tuple[str, ...]] = {
    "truck": ("truck_wait", "long_wait"),
    "vessel": ("vessel_delay", "depart_delay", "sts_wait", "transfer_wait"),
    "move": ("crane_travel", "empty_travel", "rehandle"),
    "other": ("lane_cong", "interference", "resequence", "imbalance"),
}
_COVERED = tuple(t for ts in CHANNELS.values() for t in ts)
assert set(_COVERED) == set(COST_TERMS) and len(_COVERED) == len(COST_TERMS), "채널 분할 불완전"

ALPHA = 0.05          # 양측 유의수준 (CI 95%)
POWER = 0.80          # MDE·표본수 산출 검정력
MAX_N = 100_000       # 필요표본수 탐색 상한 (수렴 실패 감지용)


def channel_split(contribs: dict[str, float]) -> dict[str, float]:
    """항목별 기여 → 채널별 합 (완전분할이므로 Σ채널 == Σ항목)."""
    return {ch: sum(contribs.get(t, 0.0) for t in terms) for ch, terms in CHANNELS.items()}


def required_n(sd: float, delta: float, *, power: float = POWER, alpha: float = ALPHA,
               sd_conf: float | None = None, sd_df: int | None = None) -> int | None:
    """δ 를 검정력 `power` 로 잡는 데 필요한 **짝지은 표본수** — 목표 df 로 반복 수렴.

    조건: (t_{1−α/2, n−1} + t_{power, n−1}) · sd/√n ≤ |δ|.
    t 가 n 에 의존하므로 상수 z 로 1회 계산하면 작은 n 에서 과소추정된다 → n 을 1씩 올리며
    실제로 조건을 만족하는 최소 n 을 찾는다.

    sd_conf: 주면 sd 를 그 신뢰도의 **상측한계**로 한 번 부풀린 뒤 푼다(예 0.80).
      불확실성은 **파일럿 표본의 자유도**(`sd_df`)에서 오므로 목표 n 이 아니라 sd_df 로
      계산해야 한다 — 목표 df 를 쓰면 큰 n 일수록 보정이 작아져 **자기충족적으로 과소추정**된다.
      (n=8 파일럿이면 배수 ≈1.353 → 트럭 채널 δ=3 필요 n 이 14 → 24 로 는다.)
    """
    d = abs(delta)
    if d <= 0 or sd <= 0:
        return None
    if sd_conf is not None:
        if sd_df is None or sd_df < 1:
            raise ValueError("sd_conf 를 쓰려면 파일럿 자유도 sd_df 가 필요하다")
        sd = sd_upper_conf(sd, sd_df, sd_conf)
    n = 2
    while n <= MAX_N:
        df = n - 1
        if (t_ppf(1.0 - alpha / 2.0, df) + t_ppf(power, df)) * sd / n ** 0.5 <= d:
            return n
        n += 1
    return None


@dataclass(frozen=True)
class PairedResult:
    """짝지은 비교 1건 — CI·MDE·동등성(TOST)을 **항상 함께** 낸다."""

    n: int
    mean: float
    sd: float
    se: float
    ci_lo: float
    ci_hi: float
    mde80: float                      # 이 표본으로 80% 검정력에서 잡을 수 있는 최소 효과
    label: str
    required_n_for_mean: int | None   # 진단용(사후) — 관측 평균만한 효과를 잡으려면
    required_n_for_delta: int | None  # **사전 δ** 기준 필요 표본수 (사전등록용)
    tost_lo: float | None = None      # 동등성 판정에 쓴 (1−2α) CI
    tost_hi: float | None = None
    equivalent: bool | None = None    # None = δ 미지정이라 판정 불가

    def as_dict(self) -> dict:
        d = {"n": self.n, "mean": round(self.mean, 3), "sd": round(self.sd, 3),
             "ci": [round(self.ci_lo, 3), round(self.ci_hi, 3)],
             "mde80": round(self.mde80, 3), "label": self.label,
             "required_n_for_mean": self.required_n_for_mean,
             "required_n_for_delta": self.required_n_for_delta,
             "equivalent": self.equivalent}
        if self.tost_lo is not None:
            d["tost_ci90"] = [round(self.tost_lo, 3), round(self.tost_hi, 3)]
        return d


def _label(ci_lo: float, ci_hi: float, mde: float, delta: float | None,
           equivalent: bool | None) -> str:
    """라벨 규칙표 — NHST(유의) × TOST(동등)의 2×2 를 빠짐없이 덮는다.

        유의 O · 동등 O → "유의하나 δ 미만" (진짜 효과지만 실질적으로 무의미)
        유의 O · 동등 X → "유의(개선/악화)"
        유의 X · 동등 O → "효과 없음(δ 배제)"        ← 여기서만 '효과 없음' 주장 가능
        유의 X · 동등 X → "검출 실패(표본 부족)"
        δ 미지정        → "미검출" 까지만 (동등성 주장 불가)
    """
    if mde == 0.0 and ci_lo == 0.0 and ci_hi == 0.0:
        # 퇴화 채널 — 전 시드에서 차이가 정확히 0 (예: numeraire_v1 의 other 채널은
        # 4개 항 가중치가 전부 0). 여기에 '효과 없음' 증명서를 내주면 **구조적 거짓
        # 주장**이 된다(변동이 없어서 0 인 것이지 처리가 무효라서 0 인 게 아니다).
        return "판정 대상 아님(전 시드 차이 0 — 가중치 0 또는 미측정)"
    sig = ci_hi < 0 or ci_lo > 0
    if delta is None:
        if sig:
            return "유의(개선)" if ci_hi < 0 else "유의(악화)"
        return f"미검출(CI 0 포함) — MDE {mde:.2f}. 관심효과 δ 미지정이라 '효과 없음' 주장 불가"
    d = abs(delta)
    if sig and equivalent:
        return f"유의하나 δ 미만(실질 무의미 — |효과| < δ={d:.2f})"
    if sig:
        return "유의(개선)" if ci_hi < 0 else "유의(악화)"
    if equivalent:
        return f"효과 없음(TOST 통과 — 90% CI ⊂ ±{d:.2f})"
    return f"검출 실패(표본 부족 — MDE {mde:.2f}, δ={d:.2f}, TOST 미통과)"


def paired(diffs: list[float], *, power_t: float | None = None,
           delta_interest: float | None = None, sd_conf: float | None = None) -> PairedResult:
    """짝지은 차이 표본 → CI + MDE + TOST + 라벨. diffs 는 (처리 − 대조) 부호 그대로.

    delta_interest: **사전등록된 관심효과 크기**. 없으면 '효과 없음' 대신 '미검출'만 낸다.
    sd_conf: 필요표본수 산출 시 sd 상측한계 신뢰도(예 0.80). None 이면 점추정 sd 사용.
    """
    n = len(diffs)
    if n < 2:
        raise ValueError("짝지은 표본이 2개 미만")
    m, sd = fmean(diffs), stdev(diffs)
    se = sd / n ** 0.5
    df = n - 1
    t95 = t_ppf(1.0 - ALPHA / 2.0, df)
    t80 = power_t if power_t is not None else t_ppf(POWER, df)
    lo, hi = m - t95 * se, m + t95 * se
    mde = (t95 + t80) * se
    # TOST — 동등성은 (1−2α) CI 가 ±δ 안에 완전히 들어갈 때만 (표준 규약: α=0.05 → 90% CI)
    t90 = t_ppf(1.0 - ALPHA, df)
    tlo, thi = m - t90 * se, m + t90 * se
    eq = None if delta_interest is None else bool(
        tlo > -abs(delta_interest) and thi < abs(delta_interest))
    return PairedResult(
        n=n, mean=m, sd=sd, se=se, ci_lo=lo, ci_hi=hi, mde80=mde,
        label=_label(lo, hi, mde, delta_interest, eq),
        required_n_for_mean=(required_n(sd, m, sd_conf=sd_conf, sd_df=df)
                             if abs(m) > 1e-12 else None),
        required_n_for_delta=(None if delta_interest is None else
                              required_n(sd, delta_interest, sd_conf=sd_conf, sd_df=df)),
        tost_lo=tlo, tost_hi=thi, equivalent=eq)


def paired_by_channel(rows_treat: list[dict], rows_ctrl: list[dict], *,
                      delta_interest: dict[str, float] | None = None,
                      primary: str | None = None,
                      sd_conf: float | None = None) -> dict[str, dict]:
    """arm 별 채널 스냅샷 리스트 → 채널별 짝지은 판정.

    각 row 는 channel_split 결과(dict) + 선택적 'total'. 같은 인덱스가 같은 시드여야 한다
    (호출부가 보장 — evalkit 은 길이만 검사).

    primary: **사전 고정된 1차 판정 채널**. 지정하면 그 채널만 확증적(confirmatory)이고
      나머지는 `role="탐색적"` 로 표기된다 — 1차를 사전 고정했으므로 다중비교 보정이
      필요 없다는 논리를 **결과 파일에 박제**하기 위함(구두 규약으로 두지 않는다).
    """
    if len(rows_treat) != len(rows_ctrl):
        raise ValueError("짝짓기 길이 불일치")
    if primary is not None and primary not in list(CHANNELS) + ["total"]:
        raise ValueError(f"1차 판정 채널 부적격: {primary}")
    out = {}
    di = delta_interest or {}
    for ch in list(CHANNELS) + ["total"]:
        d = [t.get(ch, 0.0) - c.get(ch, 0.0) for t, c in zip(rows_treat, rows_ctrl)]
        if any(abs(x) > 0 for x in d) or ch in CHANNELS:
            r = paired(d, delta_interest=di.get(ch), sd_conf=sd_conf).as_dict()
            if primary is not None:
                r["role"] = "1차(확증)" if ch == primary else "탐색적(보정 없음)"
            out[ch] = r
    return out


@dataclass
class GuardReport:
    ok: bool
    failures: list[str] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise AssertionError("하드 guard 실패: " + "; ".join(self.failures))


def check_guards(rows: list[dict], *, require_completion: float = 1.0,
                 require_backlog_zero: bool = True,
                 require_healthy: bool = False) -> GuardReport:
    """감사 조치 ③ — 완주·backlog·healthy 를 기계적으로 검사.

    rows 의 키: 'compl'|'completion'(비율) · 'backlog'(수) · 'healthy'(bool). 없는 키는
    **검사 불가로 실패 처리**한다(수집조차 안 하던 하네스를 침묵 통과시키지 않기 위함).
    """
    fails: list[str] = []
    for i, r in enumerate(rows):
        c = r.get("compl", r.get("completion"))
        if c is None:
            fails.append(f"row{i}: 완주율 미수집")
        elif c < require_completion - 1e-9:
            fails.append(f"row{i}: 완주율 {c:.4f} < {require_completion}")
        if require_backlog_zero:
            b = r.get("backlog")
            if b is None:
                fails.append(f"row{i}: backlog 미수집")
            elif b != 0:
                fails.append(f"row{i}: backlog {b} != 0")
        if require_healthy:
            h = r.get("healthy")
            if h is None:
                fails.append(f"row{i}: healthy 미수집")
            elif not h:
                fails.append(f"row{i}: healthy False")
    return GuardReport(ok=not fails, failures=fails)


def prereg_power_note(pilot_diffs: list[float], *, target_effect: float,
                      confirm_multiplier: int = 2, sd_conf: float | None = 0.80) -> dict:
    """사전등록용 검정력 메모 — 파일럿 표본으로 필요 n 과 확증 대역 n 을 산출.

    감사 조치: 확증 대역 n 은 선택 대역의 `confirm_multiplier` 배 이상(기본 2배).
    **sd_conf 기본 0.80**: 파일럿 sd 의 불확실성을 반영한 보수적 표본수를 기본으로 낸다
    (n=8 파일럿이면 sd 상한 배수 ≈1.35 → 필요 n 이 약 1.8배). 점추정 값도 함께 보고한다.
    """
    p = paired(pilot_diffs)
    df = p.n - 1
    need = (required_n(p.sd, target_effect, sd_conf=sd_conf, sd_df=df)
            if target_effect else None)
    need_pt = required_n(p.sd, target_effect) if target_effect else None
    return {"pilot_n": p.n, "pilot_sd": round(p.sd, 3), "pilot_mde80": round(p.mde80, 3),
            "target_effect": target_effect,
            "required_n_select": need, "required_n_select_point": need_pt,
            "sd_conf": sd_conf,
            "sd_inflation": (None if sd_conf is None
                             else round(sd_upper_conf(1.0, p.n - 1, sd_conf), 4)),
            "required_n_confirm": None if need is None else need * confirm_multiplier,
            "rule": f"확증 대역 n ≥ 선택 대역 n × {confirm_multiplier} (감사 조치) · "
                    f"필요 n 은 목표 df 반복수렴 + sd 상측한계(conf={sd_conf})"}
