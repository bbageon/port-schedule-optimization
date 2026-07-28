"""YR-113 — 창중 이송 **자체**의 순효과: 기준 규칙(BASE) vs 이송 0(NOTRANSFER).

■ 왜 이 실험이 필요한가
YR-105 정본 판정의 **부수 관찰**: 이송을 12.7→5.2 건으로 줄였더니 트럭 비용이 유의하게
올랐다. 뒤집으면 "기준 규칙의 이송이 트럭에 이로웠다"로 읽히지만, 그 실험의 설계는
**필터 유무**이지 **이송 유무**가 아니다. 게다가 비교 구간이 12.7 vs 5.2 라 "많이 vs 적게"
일 뿐 "있음 vs 없음"이 아니다. 사전등록 밖의 관찰이므로 **가설이지 결과가 아니다.**
여기서 직접 붙인다.

■ arm (유일한 차이 = 이송 허용 여부)
  · BASE        : 혼잡 격차 `C_zero(src) − C_zero(dst) ≥ 0.10` 이면 이송 (YR-105 와 동일)
  · NOTRANSFER  : 이송 0 (review 는 그대로 돌지만 아무것도 옮기지 않는다)
같은 시드·같은 정책·같은 마감(물리 달성가능)·같은 시계. review epoch 자체는 두 arm 모두
동일하게 예약되므로 "epoch 유무" 교락이 없다.

■ 판정 (사전등록 — 결과 미열람 동결)
  1차 판정 채널 = **truck** (사전 고정 → 다중비교 보정 불요).
  Δ = NOTRANSFER − BASE 로 정의한다. **Δ > 0 이면 이송이 이로웠다**(막으니 비싸졌다).
  두 층으로 나눠 보고한다 (사용자 지시):
    (a) **통계적으로 0보다 좋은가** — 95% 신뢰구간이 0을 배제하는가
    (b) **운영적으로 의미 있는가** — 사전 실질기준 δ=3 을 넘는가
        (넘지 못하면 "실재하나 δ 미만"으로, 동등성(TOST) 통과 시 "효과 없음"으로 보고)
  부지표: 본선·이동(주행비 포함)·총비용 채널, 이송 건수, A→O·B→C, guard.
  **주의**: 이송에는 추가 주행비가 붙고 그것은 move 채널에 잡힌다. 트럭 이득이 주행비를
  넘는지는 **총비용 채널로 따로** 본다 — 트럭만 보고 "이롭다"고 말하지 않는다.

■ 표본 (실행 전 산출 — `python -m ...yr113_transfer_net_effect --power`)
  파일럿으로 필요 표본수를 먼저 계산하고, 그 결과가 계획 n 을 넘으면 n 을 올린다.
  대역은 실현 지문 배정(YR-108) — 선택·확증·기존 열람분과 교집합 0.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from ..integrated.evalkit import (CHANNELS, DIAGNOSTIC_NOTE_TRUCK, check_guards,
                                  judge_primary, paired, paired_by_channel,
                                  prereg_power_note, required_n)
from ..integrated.profiles import build_calibrated_profile
from ..integrated.repro import repro_stamp, vessel_physics_rows
from ..integrated.scenario_gen import generate_terminal_scenario
from ..integrated.seedbank import assign_band, independence_report
from . import yr105_conditional_transfer as y5

OUT = Path("outputs/reports/yr113_transfer_net_effect")
PRIMARY = "truck"
DELTA = {"truck": 3.0, "vessel": 10.0, "move": 1.0, "other": 1.0, "total": 10.0}
ORIGINAL_PREREG = (
    "Δ = NOTRANSFER − BASE · 1차 채널 truck · "
    "(a) CI 가 0 배제 (b) δ=3.0 초과 여부를 **분리 보고**. "
    "트럭 이득이 주행비를 넘는지는 total 채널로 따로 본다."
)
ORIGINAL_PREREG_COMMIT = "73ef07bc8f49d877b89e7602cd39537d763c5e5e"

# YR-117 은 YR-113 결과를 본 뒤 추가한 **사후 재분석**이다. 아래 계약은 향후 실험에서
# 결과 열람 전에 호출자가 명시할 때만 확증 계약이 될 수 있다.
RETROSPECTIVE_METRICS = ("total", "a2o_min")
RETROSPECTIVE_DELTA = {"total": 10.0, "a2o_min": 1.0}
RETROSPECTIVE_KEYS = {
    "total": ("total_raw", "total"),
    "a2o_min": ("a2o_mean_min_raw", "a2o_mean_min"),
}
SD_CONF = 0.80
BAND_START = {"pilot": 903_000, "select": 904_000, "confirm": 905_000}


def _band(band: str, n: int, exclude: set[str] | None = None):
    spec = assign_band(family=f"y113-{band}", cells={"A": y5.CELL_A, "B": y5.CELL_B}, n=n,
                       generate=lambda k, c, s: generate_terminal_scenario(
                           build_calibrated_profile(), s, y5._params(c)),
                       exclude=exclude or set(), start_seed=BAND_START[band])
    rep = independence_report(spec)
    if not rep["ok"]:
        raise AssertionError(f"실현 독립성 위반: {rep}")
    return spec


def run_pair(i: int, band: str, seeds: dict[str, int]) -> dict:
    """같은 시드에서 BASE(이송 허용) 와 NOTRANSFER(이송 0) 를 각각 실행."""
    base = y5.run_arm(i, band, vessel_guard=False, seeds=seeds,
                      gap_threshold=y5.THRESH)
    # 이송 0 = 격차 임계를 넘길 수 없게 만든다 (review 는 그대로 돈다 → epoch 교락 없음)
    none = y5.run_arm(i, band, vessel_guard=False, seeds=seeds,
                      gap_threshold=float("inf"))
    assert base["n_jobs"] == none["n_jobs"]
    assert none["n_moved"] == 0, f"NOTRANSFER arm 이 {none['n_moved']}건 이송했다"
    return {"seed": i, "seed_A": seeds["A"], "seed_B": seeds["B"],
            "base": base, "notransfer": none,
            "d_total": round(none["total"] - base["total"], 3)}


def power_note(n_pilot: int = 8) -> dict:
    """**실행 전** 예상 분산·최소검출가능효과 산출 (사용자 지시: 새 비교이므로 재계산)."""
    OUT.mkdir(parents=True, exist_ok=True)
    spec = _band("pilot", n_pilot)
    rows = [run_pair(i, "pilot", {"A": spec.seeds["A"][i], "B": spec.seeds["B"][i]})
            for i in range(n_pilot)]
    out: dict = {"n_pilot": n_pilot, "band": spec.freeze_json(), "channels": {}}
    for ch in list(CHANNELS) + ["total"]:
        d = [r["notransfer"]["chan"][ch] - r["base"]["chan"][ch] for r in rows]
        p = paired(d, delta_interest=DELTA[ch], sd_conf=SD_CONF)
        out["channels"][ch] = {
            **p.as_dict(),
            "required_n_point": required_n(p.sd, DELTA[ch]),
            "required_n_conf80": required_n(p.sd, DELTA[ch], sd_conf=SD_CONF, sd_df=p.n - 1)}
    out["primary_power"] = prereg_power_note(
        [r["notransfer"]["chan"][PRIMARY] - r["base"]["chan"][PRIMARY] for r in rows],
        target_effect=DELTA[PRIMARY], sd_conf=SD_CONF)
    out["moved_base"] = round(fmean(r["base"]["n_moved"] for r in rows), 1)
    out["deadlock_escapes"] = sum(r[a]["deadlock_escapes"] for r in rows
                                  for a in ("base", "notransfer"))
    (OUT / "power_note.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    print(f"\n=== YR-113 사전 검정력 (파일럿 n={n_pilot}) ===")
    print(f"이송 건수(BASE 평균) {out['moved_base']} · 교착 탈출 {out['deadlock_escapes']}회")
    for ch, v in out["channels"].items():
        print(f"  {ch:7s} Δ {v['mean']:+8.3f} sd {v['sd']:7.3f} MDE {v['mde80']:7.3f} "
              f"필요n δ={DELTA[ch]:g} → 점추정 {v['required_n_point']} / 보수 {v['required_n_conf80']}")
    return out


def retrospective_metric_reanalysis(rows: list[dict]) -> dict:
    """YR-113 원자료를 YR-117 total+A→O 계약으로 **사후** 읽는다.

    이 함수의 결과는 새 데이터로 사전등록한 확증이 아니다. 원래 사전등록의 1차 지표는
    `truck` 이었음을 결과 객체에도 함께 박제한다.
    """
    joint = judge_primary(
        [r["notransfer"] for r in rows],
        [r["base"] for r in rows],
        metrics=RETROSPECTIVE_METRICS,
        metric_keys=RETROSPECTIVE_KEYS,
        delta=RETROSPECTIVE_DELTA,
        sd_conf=SD_CONF,
    )
    for metric in RETROSPECTIVE_METRICS:
        joint[metric]["role"] = "YR-117 사후 재분석(확증 아님)"
    stored = {
        "analysis_status": "retrospective_post_hoc_not_confirmatory",
        "confirmatory": False,
        "decision_rule": "AND",
        "joint_and_pass": joint["joint_and_pass"],
        "contract": joint["contract"],
        **{metric: joint[metric] for metric in RETROSPECTIVE_METRICS},
    }
    return {
        "analysis_status": "retrospective_post_hoc_not_confirmatory",
        "confirmatory": False,
        "original_preregistered_primary": PRIMARY,
        "original_prereg": ORIGINAL_PREREG,
        "original_prereg_commit": ORIGINAL_PREREG_COMMIT,
        "joint_contract_examined": joint,
        "stored_view": stored,
    }


def run(band: str, n: int, exclude: set[str] | None = None) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    spec = _band(band, n, exclude)
    rows = []
    for i in range(n):
        sd = {"A": spec.seeds["A"][i], "B": spec.seeds["B"][i]}
        r = run_pair(i, band, sd)
        prof = build_calibrated_profile()
        r["physics_A"] = vessel_physics_rows(
            generate_terminal_scenario(prof, sd["A"], y5._params(y5.CELL_A)))
        rows.append(r)
        print(f"[{band} s{i}] BASE={r['base']['total']:.2f}(mv {r['base']['n_moved']}) "
              f"NONE={r['notransfer']['total']:.2f} d={r['d_total']:+.2f}", flush=True)
    guards = check_guards([{"compl": r[a]["compl"], "backlog": r[a]["backlog"]}
                           for r in rows for a in ("base", "notransfer")])
    n_exc = sum(r[a]["policy_exceptions"] for r in rows for a in ("base", "notransfer"))
    if n_exc:
        guards.ok = False
        guards.failures.append(f"정책 예외 {n_exc}건 — 판정 무효")
    ch = paired_by_channel([r["notransfer"]["chan"] for r in rows],
                           [r["base"]["chan"] for r in rows],
                           delta_interest=DELTA, primary=PRIMARY, sd_conf=SD_CONF)
    posthoc = retrospective_metric_reanalysis(rows)
    # 진단: 트럭 채널이 시계 시작점 때문에 부풀 수 있는 상한 (이송건수 × route_s / 3600)
    truck_bias_cap = round(fmean(r["base"]["n_moved"] for r in rows) * 180.0 / 3600.0, 3)
    res = {"repro": repro_stamp(
               experiment="YR-113 창중 이송 자체의 순효과 (BASE vs 이송 0)",
               seeds=spec.seeds, params={"cell_A": y5._params(y5.CELL_A),
                                         "cell_B": y5._params(y5.CELL_B)},
               profile_id=build_calibrated_profile().terminal_id,
               prereg=ORIGINAL_PREREG,
               extra={"delta_interest": DELTA, "sd_conf": SD_CONF, "band": band,
                      "primary_channel": PRIMARY,
                      "original_prereg_commit": ORIGINAL_PREREG_COMMIT,
                      "yr117_analysis_status": "retrospective_post_hoc_not_confirmatory",
                      "truck_channel_caveat": DIAGNOSTIC_NOTE_TRUCK,
                      "band_spec": spec.freeze_json(),
                      "achievable_deadline": y5.ACHIEVABLE_DEADLINE}),
           "verdict_valid": guards.ok, "band": band, "rows": rows,
           # 과거 evidence reader의 `primary` 키를 보존하되, 사후 분석임을 객체 안에 박제한다.
           "primary": posthoc["stored_view"],
           "yr117_posthoc": posthoc,
           "truck_channel_bias_cap": truck_bias_cap,
           "channels_notransfer_minus_base": ch,
           "guards": {"ok": guards.ok, "failures": guards.failures[:5]},
           "deadlock_escapes_total": sum(r[a]["deadlock_escapes"] for r in rows
                                         for a in ("base", "notransfer")),
           "moved_base": round(fmean(r["base"]["n_moved"] for r in rows), 1)}
    (OUT / f"results_{band}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    print(f"\n=== YR-113 [{band}] n={n} valid={guards.ok} "
          f"이송(BASE) {res['moved_base']} 탈출 {res['deadlock_escapes_total']}회 ===")
    print(f"  --- 원 사전등록 1차 채널: {PRIMARY} ---")
    v = ch[PRIMARY]
    print(f"  {PRIMARY:8s} {v['mean']:+8.3f} CI[{v['ci'][0]:+7.2f},{v['ci'][1]:+7.2f}] "
          f"{v['label']}")
    print("  --- YR-117 사후 재분석(확증 아님): total AND 평균 A→O ---")
    joint = posthoc["joint_contract_examined"]
    for k in RETROSPECTIVE_METRICS:
        v = joint[k]
        print(f"  {k:8s} {v['mean']:+8.3f} CI[{v['ci'][0]:+7.2f},{v['ci'][1]:+7.2f}] "
              f"{v['unit']} · {v['label']}")
    print(f"  공동 AND 통계 판정: {joint['joint_and_pass']} (사후 진단값)")
    print(f"  --- 진단 채널 (트럭 부풀림 상한 {truck_bias_cap}) ---")
    for c, v in ch.items():
        print(f"  {c:7s} {v['mean']:+8.3f} CI[{v['ci'][0]:+7.2f},{v['ci'][1]:+7.2f}] "
              f"{v['label']}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", action="store_true", help="실행 전 검정력 산출만")
    ap.add_argument("--band", default="select", choices=["select", "confirm"])
    ap.add_argument("--seeds", type=int, default=53)   # 사전 검정력 산출값(δ=3·보수)
    a = ap.parse_args()
    y5.ACHIEVABLE_DEADLINE = True          # 신규 판정은 물리 달성가능 마감이 기본
    if a.power:
        power_note()
    else:
        # 실현 독립: 파일럿(검정력 산출에 소모) 과 선택 대역을 확증에서 배제한다
        excl = set(_band("pilot", 8).all_hashes)
        if a.band == "confirm":
            excl |= set(_band("select", 53, excl).all_hashes)
        r = run(a.band, a.seeds, excl)
        raise SystemExit(0 if r["verdict_valid"] else 2)
