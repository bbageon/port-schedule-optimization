"""Rebuild descriptive cost tables from completed runs; never launch experiments."""
import hashlib
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
REPORT = ROOT / "outputs/reports/yr317_v3_request_audit"
ARMS = ("NO_REALLOC", "RL", "RL_NOVETO")
LABELS = {"NO_REALLOC": "배정 변경 없음", "RL": "전체 모형",
          "RL_NOVETO": "수락 거절 기능 제거", "RL_TIME": "시간 조정만",
          "RL_SPACE": "블록 조정만"}
COMPONENTS = {"c_wait": "트럭 체류·장시간 벌점", "c_move": "크레인 추가 이동",
              "c_rehandle": "컨테이너 재조작", "c_vessel": "본선 유휴"}


def main():
    sources = {}

    def read(path):
        raw = path.read_bytes()
        sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def close(a, b):
        assert math.isfinite(a) and math.isfinite(b)
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-4), (a, b)

    reconciliation = read(REPORT / "auto-b0fe052/reconciliation-4.json")
    assert reconciliation["completed_artifacts_valid"]
    assert reconciliation["all_main_arms_complete"]
    assert reconciliation["same_requested_work_across_all_main_arms"]
    diagnostic, identities = {}, set()
    for arm in ARMS:
        path = REPORT / f"run-3279b7f/{arm}/result.json"
        result = read(path)
        ref = reconciliation["completed_arms"][arm]
        assert sources[path.relative_to(ROOT).as_posix()] == ref["artifact_sha256"]["result.json"]
        assert result["seed"] == 9_500_000
        days, requests = result["days"], result["request_summary"]
        assert [d["index"] for d in days] == list(range(10))
        assert all(d["provisional"] is False for d in days)
        assert requests["requested"] == 69_000 and requests["unprocessed"] == 0
        assert requests["states"] == ref["states"]
        assert sum(requests["states"].values()) == requests["requested"]
        assert sum(d["n_trucks"] for d in days) == requests["states"]["COMPLETED"]
        costs = {key: math.fsum(d[key] for d in days) for key in COMPONENTS}
        for day in days:
            close(day["phi_krw"], math.fsum(day[key] for key in COMPONENTS))
        for key, cost in costs.items():
            close(cost, ref["four_cost_totals_krw"][key])
        close(costs["c_wait"], requests["accounted_wait_krw"])
        total = math.fsum(d["phi_krw"] for d in days)
        close(total, math.fsum(costs.values()))
        vessels = result["vessel_admissions"]
        asked, admitted = (sum(v[key] for v in vessels) for key in ("asked", "moves"))
        assert asked == 44_322
        assert admitted == ref["vessel_admissions"]["admitted_moves"]
        identities.add(result["requested_identity_sha256"])
        diagnostic[arm] = {"total_cost_krw": total, "components_krw": costs,
            "truck_requested": requests["requested"], "truck_completed": requests["states"]["COMPLETED"],
            "truck_skipped": requests["skipped"], "vessel_requested": asked,
            "vessel_admitted": admitted, "vessel_not_admitted": asked - admitted,
            "tracking_end_s": requests["end_s"]}
    assert len(identities) == 1
    legacy = {}
    for arm in ("NO_REALLOC", "RL", "RL_TIME", "RL_SPACE"):
        result = read(ROOT / f"outputs/v3/judge-30d/arms/arm_{arm}.json")
        legacy[arm] = {"total_cost_krw": math.fsum(result["phi_by_day"][str(i)] for i in range(1, 29))}
    for group in (diagnostic, legacy):
        baseline = group["NO_REALLOC"]["total_cost_krw"]
        for row in group.values():
            row["saving_krw"] = baseline - row["total_cost_krw"]
            row["saving_percent"] = 100 * row["saving_krw"] / baseline
    extra = diagnostic["RL_NOVETO"]["total_cost_krw"] - diagnostic["RL"]["total_cost_krw"]
    extra_pct = 100 * extra / diagnostic["RL_NOVETO"]["total_cost_krw"]
    extra_pp = diagnostic["RL"]["saving_percent"] - diagnostic["RL_NOVETO"]["saving_percent"]
    payload = {"schema": "yr317.descriptive-performance.v1", "passed": True,
        "scope": {"diagnostic": "one reused seed 9500000, all 10 days (indices 0..9)",
                  "legacy": "original judge-30d, middle 28 days (indices 1..28)",
                  "cost": "existing terminal cost; skipped requests and external rescheduling costs excluded"},
        "diagnostic": diagnostic, "legacy": legacy,
        "full_vs_no_veto": {"saving_krw": extra, "saving_percent_of_no_veto": extra_pct,
                            "saving_percentage_points_of_baseline": extra_pp},
        "input_sha256": sources, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "new_simulations": 0, "new_training_runs": 0, "performance_confirmed": False}
    lines = ["# 성능 비교표: 기존 논문과 이번 10일 진단", "",
        "2026-09-15. **앞선 진행 보고에 빠진 비용 비교표를 원자료에서 다시 계산해 추가했다.**",
        "비교표를 제시하는 것과 일반적인 성능 우위를 확정하는 것은 별개다. 아래 수치는 기존 실행 결과이며 추가학습·새 시뮬레이션 결과가 아니다.", "",
        "## 1. 이번 10일 진단의 비용과 처리량", "",
        "세 정책 모두 시드(상황을 재현하는 난수 시작값) 9,500,000의 같은 기본 트럭 요청 69,000건·본선 작업 44,322회로 시작했다.",
        "10일 전체의 확정 일별 비용을 합산했다. 트럭은 종료 뒤 2시간까지 추적한다. 아래 본선 수는 투입 수이며 완료 수가 아니다.", "",
        "| 정책 | 총비용(억 원) | 기준 대비 절감률 | 트럭 완료(대) | 트럭 미투입(대) | 본선 투입(회) |",
        "|---|---:|---:|---:|---:|---:|"]
    for arm, row in diagnostic.items():
        saving = "기준" if arm == "NO_REALLOC" else f'{row["saving_percent"]:.2f}%'
        lines.append(f'| {LABELS[arm]} | {row["total_cost_krw"] / 1e8:.3f} | {saving} | '
                     f'{row["truck_completed"]:,} | {row["truck_skipped"]:,} | {row["vessel_admitted"]:,} |')
    lines += ["", "절감률 = (배정 변경 없음의 총비용 − 해당 정책 총비용) / 배정 변경 없음의 총비용 × 100. 원 단위 원값으로 계산한 뒤 표시만 반올림했다.",
        f"이 실행에서 전체 모형은 거절 기능 제거보다 {extra / 1e8:.3f}억 원({extra_pct:.2f}%) 낮고, 기준 대비 절감률 차이는 {extra_pp:.2f}%포인트다.",
        "다만 트럭 완료 수는 전체 모형이 44대 적고 본선 투입 수는 28회 많다. 이 비용 차이 전부를 수락 기능의 효과로 분리할 수 없다.",
        "수락 거절 기능 제거는 수락망의 거절 권한만 끈 비교다. 후보 정렬에는 수락망 점수가 남아 있어 수락망 전체 제거 실험으로 부르지 않는다.", "",
        "### 비용 항목별 비교", "", "단위: 백만 원. 크레인 추가 이동과 컨테이너 재조작 비용도 총비용에 포함한다.", "",
        "| 비용 항목 | 배정 변경 없음 | 전체 모형 | 수락 거절 기능 제거 |",
        "|---|---:|---:|---:|"]
    for key, label in COMPONENTS.items():
        cells = " | ".join(f'{diagnostic[a]["components_krw"][key] / 1e6:,.3f}' for a in ARMS)
        lines.append(f"| {label} | {cells} |")
    wait_share = 100 * (diagnostic["NO_REALLOC"]["components_krw"]["c_wait"] -
                       diagnostic["RL"]["components_krw"]["c_wait"]) / diagnostic["RL"]["saving_krw"]
    lines += ["", f"전체 모형의 기준 대비 순절감액 중 {wait_share:.2f}%가 트럭 체류 비용 차이에서 나온다. 이는 비용 항목의 분해이며 시간 조정 행동의 기여율과 같지 않다.",
        "전체 모형은 기준보다 이동·재조작 비용이 높다. 거절 기능 제거와 비교하면 본선 유휴 비용도 더 높아 모든 항목이 함께 개선된 결과는 아니다.", "",
        "## 2. 기존 논문에 사용된 28일 결과", "",
        "원래 30일 실행의 가운데 28일(날짜 번호 1–28)이며 위 10일 진단과 평가 상황·기간이 다르다. 두 표의 절감률을 학습 전후 성능 변화로 비교하지 않는다.", "",
        "| 정책 | 총비용(억 원) | 기준 대비 절감률 |", "|---|---:|---:|"]
    for arm, row in legacy.items():
        saving = "기준" if arm == "NO_REALLOC" else f'{row["saving_percent"]:.2f}%'
        lines.append(f'| {LABELS[arm]} | {row["total_cost_krw"] / 1e8:.3f} | {saving} |')
    lines += ["", "음수 절감률은 비용 증가다. 전체 모형과 시간 조정만의 차이는 약 1.28%포인트이며, 블록 조정만은 전체 기간 비용이 증가했다.",
        "한가할 때의 블록 효과와 혼잡할 때의 시간 효과는 [기존 부하별 표](01-독립반복-부하별효과.md)에서 별도로 확인한다. 이번 10일에는 시간만·블록만 정책을 실행하지 않았다.", "",
        "## 3. 해석 범위와 리뷰 대응", "",
        "- 리뷰 1-4: 수락 거절 기능 비교의 관측 비용을 공개했다. 요청 처리 차이를 해결하고 수락망 전체 제거와 비교해야 별도 기여를 검증할 수 있다.",
        "- 리뷰 1-1·1-5·리뷰 2: 이번 표는 재사용한 시드 한 번의 진단이다. 독립 반복·통계적 유의성·다른 환경 검증을 대신하지 않는다.",
        "- 리뷰 1-2: 기존 비용식은 미투입 요청의 부담과 게이트 밖 대기·일정 변경 비용을 포함하지 않는다. 17.28%를 현실 제약 적용 뒤의 절감률로 쓰지 않는다.",
        "- 기존 10일 기록은 본선 투입까지만 검증했다. 이후 300대 비교는 기록 기능을 켜도 결과가 같음을 검증한 별도 실행으로, 성능 개선 실험이 아니다.", "",
        "다음은 YR-317-g(요청 처리 기록 검증)에서 미투입 시점의 재고·예약과 본선 완료를 확인하는 작업이다. 비용 차이에 처리 누락의 영향이 섞였는지 판단하기 위해 필요하다.", "",
        "## 4. 재계산과 근거", "",
        "저장소 루트에서 `python docs/paper/v3/paper-revision/audit_performance.py`를 실행하면 이 표와 [원 단위 수치·원자료 지문](performance-table.json)을 다시 만든다.",
        "원자료 세 개의 지문을 [기존 최종 대조](../../../../outputs/reports/yr317_v3_request_audit/auto-b0fe052/reconciliation-4.json)와 확인하고, 일별 총비용·네 비용 항목·처리 수를 교차 검산한다.",
        "[진단 상세](../../../../outputs/reports/yr317_v3_request_audit/completed-diagnostic.md) · [후속 작업과 이유](07-요청정합-진행기록.md).", ""]
    assert len(lines) <= 200
    (OUT / "performance-table.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "08-성능비교표.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"passed": True, "diagnostic": diagnostic, "full_vs_no_veto": payload["full_vs_no_veto"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
