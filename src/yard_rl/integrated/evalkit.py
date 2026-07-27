"""YR-106 — 판정 지표·검정력 공용 유틸 (설계감사 critical 대응).

설계감사(2026-07-27) 결론: 주지표 `terminal_total` 의 **70%가 본선비용**이고 짝지은 분산의
대부분이 거기서 온다(sd 18.75 vs 트럭측 5.49). 그래서 총비용으로 유의판정하려면 시드가
약 97개 필요한데 전 판정이 n=8 로 수행됐다 — **"효과 없음"과 "이 지표로는 못 본다"가
구분되지 않았다.** 그 뿌리는 `vessel_deadline_mult<1` 이 만든 **정책 무관 구조적 선석초과**
(YR-109 가 따로 다룬다).

본 모듈이 강제하는 3가지:
  ① **채널 분해** — 비용을 truck/vessel/move/other 로 쪼개, 이송·배치 실험의 **1차 판정을
     트럭 채널**로 둔다(본선 잡음에 익사시키지 않는다). 총비용·본선은 병기 보고.
  ② **MDE(최소검출가능효과) 필수** — 모든 판정에 "이 표본으로 잡을 수 있는 최소 효과"를
     함께 낸다. CI 가 0 을 포함해도 |효과| < MDE 면 **"효과 없음"이 아니라 "검출 실패"** 다.
  ③ **하드 guard 기계화** — 완주율·backlog·healthy 를 사람 눈이 아니라 함수가 막는다.

사용 규약: 사전등록에 `mde_note` 를 반드시 적고, 확증 대역 n 은 선택 대역의 **2배 이상**.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean, stdev

from ..contract import COST_TERMS

# --- 채널 정의 (COST_TERMS 13항의 완전분할 — 합 = total_normalized) ---
CHANNELS: dict[str, tuple[str, ...]] = {
    "truck": ("truck_wait", "long_wait"),
    "vessel": ("vessel_delay", "depart_delay", "sts_wait", "transfer_wait"),
    "move": ("crane_travel", "empty_travel", "rehandle"),
    "other": ("lane_cong", "interference", "resequence", "imbalance"),
}
_COVERED = tuple(t for ts in CHANNELS.values() for t in ts)
assert set(_COVERED) == set(COST_TERMS) and len(_COVERED) == len(COST_TERMS), "채널 분할 불완전"

# 양측 95% t (YR-104 정정판과 동일 소스 — df 미지 시 보수값)
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
        23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045}
# 단측 80% 검정력용 t (MDE = (t_{.975,df} + t_{.80,df})·se)
_T80 = {1: 1.376, 2: 1.061, 3: 0.978, 4: 0.941, 5: 0.920, 6: 0.906, 7: 0.896, 8: 0.889,
        9: 0.883, 10: 0.879, 11: 0.876, 12: 0.873, 13: 0.870, 14: 0.868, 15: 0.866,
        16: 0.865, 17: 0.863, 18: 0.862, 19: 0.861, 20: 0.860, 23: 0.858, 29: 0.854}


def _t(table: dict[int, float], df: int, fallback: float) -> float:
    if df in table:
        return table[df]
    return fallback if df < 30 else (2.042 if table is _T95 else 0.854)


def channel_split(contribs: dict[str, float]) -> dict[str, float]:
    """항목별 기여 → 채널별 합 (완전분할이므로 Σ채널 == Σ항목)."""
    return {ch: sum(contribs.get(t, 0.0) for t in terms) for ch, terms in CHANNELS.items()}


@dataclass(frozen=True)
class PairedResult:
    """짝지은 비교 1건 — CI 와 **MDE 를 항상 함께** 낸다 (감사 조치 ②)."""

    n: int
    mean: float
    sd: float
    se: float
    ci_lo: float
    ci_hi: float
    mde80: float                 # 이 표본으로 80% 검정력에서 잡을 수 있는 최소 효과
    label: str                   # 아래 _label() 규약
    required_n_for_mean: int | None   # 관측 평균만한 효과를 잡으려면 필요한 n

    def as_dict(self) -> dict:
        return {"n": self.n, "mean": round(self.mean, 3), "sd": round(self.sd, 3),
                "ci": [round(self.ci_lo, 3), round(self.ci_hi, 3)],
                "mde80": round(self.mde80, 3), "label": self.label,
                "required_n_for_mean": self.required_n_for_mean}


def _label(ci_lo: float, ci_hi: float, mde: float,
           delta_interest: float | None) -> str:
    """감사 조치 ②: '유의'·'효과 없음'·'검출 실패'를 라벨 단계에서 분리.

    **주의(자체 정정)**: 판단 기준은 관측 평균이 아니라 **사전 지정 관심효과 δ** 다.
    관측 평균을 쓰면 ①`|mean| ≥ MDE ⟹ CI 가 0 배제`라 '검출 실패' 라벨이 원리적으로
    발화하지 않고 ②사후검정력(post-hoc power) 오류가 된다.
    δ 없이는 "효과 없음"을 주장할 수 없다 — 그 경우 '미검출'까지만 말한다.
    """
    if ci_hi < 0:
        return "유의(개선)"
    if ci_lo > 0:
        return "유의(악화)"
    if delta_interest is None:
        return (f"미검출(CI 0 포함) — MDE {mde:.2f}. 관심효과 δ 미지정이라 "
                f"'효과 없음' 주장 불가")
    d = abs(delta_interest)
    if mde <= d:
        return f"효과 없음(δ={d:.2f} 이상은 배제 — MDE {mde:.2f} ≤ δ)"
    return f"검출 실패(표본 부족 — MDE {mde:.2f} > δ={d:.2f})"


def paired(diffs: list[float], *, power_t: float | None = None,
           delta_interest: float | None = None) -> PairedResult:
    """짝지은 차이 표본 → CI + MDE + 라벨. diffs 는 (처리 − 대조) 부호 그대로.

    delta_interest: **사전등록된 관심효과 크기**. 없으면 '효과 없음' 대신 '미검출'만 낸다.
    """
    n = len(diffs)
    if n < 2:
        raise ValueError("짝지은 표본이 2개 미만")
    m, sd = fmean(diffs), stdev(diffs)
    se = sd / n ** 0.5
    df = n - 1
    t95 = _t(_T95, df, 2.571)
    t80 = power_t if power_t is not None else _t(_T80, df, 0.920)
    lo, hi = m - t95 * se, m + t95 * se
    mde = (t95 + t80) * se
    req = None
    if abs(m) > 1e-12:
        # n' 은 se∝1/√n 이라 (t95+t80)·sd/√n' ≤ |m| 를 만족하는 최소 n' (t 는 대표값 고정 근사)
        req = int(((2.0 + 0.85) * sd / abs(m)) ** 2) + 1
    return PairedResult(n=n, mean=m, sd=sd, se=se, ci_lo=lo, ci_hi=hi, mde80=mde,
                        label=_label(lo, hi, mde, delta_interest),
                        required_n_for_mean=req)


def paired_by_channel(rows_treat: list[dict], rows_ctrl: list[dict], *,
                      delta_interest: dict[str, float] | None = None) -> dict[str, dict]:
    """arm 별 채널 스냅샷 리스트 → 채널별 짝지은 판정.

    각 row 는 channel_split 결과(dict) + 선택적 'total'. 같은 인덱스가 같은 시드여야 한다
    (호출부가 보장 — evalkit 은 길이만 검사).
    """
    if len(rows_treat) != len(rows_ctrl):
        raise ValueError("짝짓기 길이 불일치")
    out = {}
    di = delta_interest or {}
    for ch in list(CHANNELS) + ["total"]:
        d = [t.get(ch, 0.0) - c.get(ch, 0.0) for t, c in zip(rows_treat, rows_ctrl)]
        if any(abs(x) > 0 for x in d) or ch in CHANNELS:
            out[ch] = paired(d, delta_interest=di.get(ch)).as_dict()
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
                      confirm_multiplier: int = 2) -> dict:
    """사전등록용 검정력 메모 — 파일럿 표본으로 필요 n 과 확증 대역 n 을 산출.

    감사 조치: 확증 대역 n 은 선택 대역의 `confirm_multiplier` 배 이상(기본 2배).
    """
    p = paired(pilot_diffs)
    need = int(((2.0 + 0.85) * p.sd / abs(target_effect)) ** 2) + 1 if target_effect else None
    return {"pilot_n": p.n, "pilot_sd": round(p.sd, 3), "pilot_mde80": round(p.mde80, 3),
            "target_effect": target_effect, "required_n_select": need,
            "required_n_confirm": None if need is None else need * confirm_multiplier,
            "rule": f"확증 대역 n ≥ 선택 대역 n × {confirm_multiplier} (감사 조치)"}
