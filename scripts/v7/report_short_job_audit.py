"""Check input pairing and reconstruct rule contrasts and reports from the run files."""
from collections import defaultdict
import json
import math
from statistics import mean

from audit_short_job_rule import ARMS, LOADS, OUT, REFERENCE, RULES, write


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def metrics(raw):
    row = raw["result"]
    count = raw["competing_choices"]
    assert count.get("opportunities", 0) == sum(count.get(k, 0) for k in ("urgent_truck","vessel","other"))
    assert math.isclose(row["phi_krw"],sum(row[k] for k in ("c_wait","c_vessel","c_move","c_rehandle")))
    calls = row["vessel_calls"]
    assert row["vessel_complete"] == all(x["finish_s"] is not None for x in calls)
    m = dict(truck=row["mean_gate_to_job_done_s"]/60, cost_lower=row["phi_krw"],
             ship_hours_lower=mean(v["hours"] for v in calls), pending=row["n_job_censored"],
             complete=int(row["vessel_complete"]), opportunities=count.get("opportunities",0),
             chosen_vessel=count.get("vessel",0), chosen_urgent=count.get("urgent_truck",0))
    for flow in ("GATE_IN","GATE_OUT"):
        for field in ("average","p90","p99","maximum","pending"):
            m[flow+"_"+field] = raw["trucks"][flow][field]
    return m


def name(arm):
    return "/".join(arm)


def summarize():
    cells = {}
    inputs = defaultdict(set)
    for arm in ARMS:
        for load in LOADS:
            days = []
            for d in range(3):
                path = OUT/"runs"/("-".join(map(str,(*arm,d,load)))+".json")
                raw = read(path)
                assert raw["case"] == [*arm,d,load] and raw["seed"] == 10084300+d
                r = raw["result"]
                ref = read(REFERENCE/f"rule-0-{d}-{load}.json")
                inputs[d,load].add((r["demand_digest"],r["quay_manifest_digest"]))
                assert (r["demand_digest"],r["quay_manifest_digest"]) == (ref["demand_digest"],ref["quay_manifest_digest"])
                assert r["n_trucks"] == r["admitted"] == load and r["policy_exceptions"] == r["txn_failed"] == 0
                if arm == ("SF_SPT","current","exact"):
                    assert all(r[k] == ref[k] for k in ("c_move","c_rehandle","c_wait","c_vessel","vessel_calls","n_job_censored","mean_gate_to_job_done_s"))
                days.append(metrics(raw))
            avg = {k:mean(x[k] for x in days) for k in days[0]}
            cells[name(arm),load] = dict(days=days,average=avg)
    assert len(inputs)==9 and all(len(v)==1 for v in inputs.values())
    out = {}
    for load in LOADS:
        def avg(rule, guard="current", signal="exact"):
            return cells[name((rule,guard,signal)),load]["average"]
        spt, fifo = avg("SF_SPT"), avg("FCFS_ARRIVAL")
        lead_on = fifo["truck"]-spt["truck"]
        lead_off = avg("FCFS_ARRIVAL","off")["truck"]-avg("SF_SPT","off")["truck"]
        noisy = [avg("SF_SPT",signal=s)["truck"] for s in ("noise_plus","noise_minus")]
        frozen = []
        for d in range(3):
            for seed in (9951101,9951102,9951103):
                r = read(REFERENCE/f"trained-{seed}-{d}-{load}.json")
                assert (r["demand_digest"],r["quay_manifest_digest"]) in inputs[d,load]
                frozen.append(r["mean_gate_to_job_done_s"]/60)
        out[str(load)] = dict(lead_on=lead_on,lead_off=lead_off,guard_contribution_to_lead=lead_on-lead_off,
            frozen_model_truck=mean(frozen),model_minus_fcfs=mean(frozen)-fifo["truck"],
            spt_off_minus_on=avg("SF_SPT","off")["truck"]-spt["truck"],estimated_minus_exact=mean(noisy)-spt["truck"],
            estimated_minus_exact_replicates=[v-spt["truck"] for v in noisy],
            hidden_minus_exact=avg("SF_SPT",signal="hidden")["truck"]-spt["truck"],
            hidden_minus_exact_off=avg("SF_SPT","off","hidden")["truck"]-avg("SF_SPT","off")["truck"],
            rules={k:avg(k) for k in RULES},vessel_exception={k:avg(k,"vessel") for k in RULES})
    decisions={str(n):read(OUT/"decisions"/f"{n}.json") for n in LOADS}
    assert all(v["same_outcome"] and v["day"]==0 and v["seed"]==10084300 for v in decisions.values())
    result=dict(trials=len(cells)*3,matched_inputs=len(inputs),baseline_replications=9,decisions=decisions,
        complete_trials=sum(sum(d["complete"] for d in v["days"]) for v in cells.values()),
        loads=out,cells={f"{k}/{n}":v for (k,n),v in cells.items()})
    write(OUT/"summary.json",result)
    return result


def report(s):
    rows = s["loads"]
    labels = ("짧은 실작업", "실제 도착순", "가까운 작업", "본선 우선·도착순", "가장 오래 기다린 작업")
    first = {n:min(val["rules"],key=lambda k:val["rules"][k]["truck"]) for n,val in rows.items()}
    winners = ", ".join(f"{int(n):,}건은 {labels[RULES.index(k)]}({rows[n]['rules'][k]['truck']:.2f}분)"
                        for n,k in first.items())
    normal = rows["3500"]
    normal_rule, normal_ship = normal["rules"]["SF_SPT"], normal["vessel_exception"]["SF_SPT"]
    lines = ["# 짧은 작업 우선 규칙이 트럭 시간을 줄인 조건", "",
        "현행 다섯 규칙 중 평균 트럭 시간이 가장 짧은 방식은 "+winners+"이었다.",
        f"관측 도착률 범위인 3,500건에서 짧은 작업 규칙의 24분 보호 해제 효과는 {normal['spt_off_minus_on']:+.2f}분, "
        f"순위용 소요시간에 ±30% 오차를 넣은 효과는 {normal['estimated_minus_exact']:+.2f}분이었다.", "",
        f"같은 트럭·기항 명단 9개를 16개 조건에서 재생했다. {s['trials']}회 중 모든 배를 끝낸 실행은 {s['complete_trials']}회다. "
        "직전 규칙의 비용·트럭 시간·본선 원장은 9개 입력 모두 정확히 재현했다.",
        "", "## 실제 블록 도착순과 비교한 평균", "",
        "수치는 게이트 진입부터 야드 작업 완료까지의 분/트럭 오더다. 각 물량에서 세 날짜의 평균을 냈다.",
        "", "| 하루 오더 | " + " | ".join(labels) + " |", "|---:|"+"---:|"*len(labels)]
    for n,row in rows.items():
        lines.append(f"| {int(n):,} | "+" | ".join(f"{row['rules'][k]['truck']:.2f}" for k in RULES)+" |")
    lines += ["", "직전 재학습 모델과 실제 블록 도착순을 같은 명단에서 다시 대조했다. "
        "아래 두 차이를 더하면 재학습 모델과 최단 규칙 사이의 차이다. 이 분해는 저장 성적의 산술 비교다.", "",
        "| 하루 오더 | 재학습 모델(분) | 모델−실제 도착순(분) | 실제 도착순−최단 규칙(분) |", "|---:|---:|---:|---:|"]
    for n,row in rows.items():
        lines.append(f"| {int(n):,} | {row['frozen_model_truck']:.2f} | {row['model_minus_fcfs']:+.2f} | {row['lead_on']:+.2f} |")
    lines += ["", "독립 작업이 모두 준비돼 있고 한 장비의 소요시간이 고정돼 있다면, 짧은 작업부터 끝내는 순서가 평균 완료시간을 최소로 한다. "
        "실제 항만은 장비 위치에 따라 다음 작업 소요시간이 달라지고, 특정 화물이 늦으면 본선 작업도 기다린다. "
        "[Smith의 단일 장비 순서 연구](https://doi.org/10.1002/nav.3800030106)",
        "", "3,500건은 시간당 방문 최대 216~237회로 관측 최대 245회 안에 든다. 7,500건(461~503회)과 15,000건(917~997회)은 혼잡 부담 시험이다. "
        "[물량 근거](../yr332_realism_followup/2026-11-21/README.md)",
        "", "## 두 조건을 분리한 결과", "",
        "| 하루 오더 | 도착순−최단, 24분 보호 켬(분) | 도착순−최단, 보호 끔(분) | 보호로 생긴 두 규칙 격차 변화(분) | 최단 규칙만 보호 해제(분) | 시간 추정에 오차 추가(분) | 시간 순위 제거(분) |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for n,v in rows.items():
        cols=("lead_on","lead_off","guard_contribution_to_lead","spt_off_minus_on","estimated_minus_exact","hidden_minus_exact")
        lines.append(f"| {int(n):,} | "+" | ".join(f"{v[k]:+.2f}" for k in cols)+" |")
    lines += ["", "앞 두 열에서 양수면 짧은 작업의 평균이 더 짧다. 세 번째는 보호를 켠 격차에서 끈 격차를 뺀 값이다. "
        "마지막 세 열에서 양수면 그 변경 때문에 최단 규칙의 트럭 시간이 길어졌다. 각 열은 별도의 짝비교여서 서로 합산할 수 없다.",
        "", "24분은 실제 야드 블록 도착 이후부터 센다. 현행 규칙은 지연 트럭을 먼저 올리고, 실행 가능한 실작업, 계획 소요시간, "
        "본선 여부, 누적 트럭 대기, 작업 식별자 순으로 순위를 가른다. 장비 충돌·같은 일감 중복은 해결기가 다시 검사한다. "
        "학습 모델도 장기대기 작업 보호를 공유한다. 이번 일정에는 새 학습이나 모델의 보호 해제 실험이 없다.",
        "", "시간 추정 오차는 작업·장비별로 고정한 −30~+30%를 순위에만 주고 부호를 뒤집어 두 번 측정했다. "
        "실제 작업의 소요시간·장비 예약을 고정했다. 시간 순위를 완전히 빼면 기존 동률 순서(본선·대기 등)가 작동한다. "
        "두 비교는 정보 정확도에 대한 민감도를 보여준다. 부산 현장의 오차 분포는 미측정이다.",
        "", "## 평균 뒤에 남는 반출 지연과 본선 영향", "",
        "| 하루 오더 | 규칙 | 반입 평균(분) | 반출 평균(분) | 반출 상위 10% 경계(분) | 자정 미완료(건) | 평균 본선 작업(시간/척) | 하루 총비용(억원) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for n,row in rows.items():
        for label,m in (("짧은 실작업",row["rules"]["SF_SPT"]),("실제 도착순",row["rules"]["FCFS_ARRIVAL"]),
                        ("경합 시 본선 먼저·짧은 실작업",row["vessel_exception"]["SF_SPT"])):
            finite=m["complete"]==1
            lines.append(f"| {int(n):,} | {label} | {m['GATE_IN_average']:.2f} | {m['GATE_OUT_average']:.2f} | "
                f"{m['GATE_OUT_p90']:.2f} | {m['pending']:.1f} | "+(f"{m['ship_hours_lower']:.2f} | {m['cost_lower']/1e8:.2f} |" if finite else "미완료 포함 | 미확정 |"))
    if normal_rule["complete"] == normal_ship["complete"] == 1:
        lines += ["",f"3,500건에서 장기대기 트럭과 본선이 경합할 때 본선을 먼저 주면 트럭 평균 "
            f"{normal_ship['truck']-normal_rule['truck']:+.2f}분, 본선 평균 {normal_ship['ship_hours_lower']-normal_rule['ship_hours_lower']:+.2f}시간, "
            f"하루 총비용 {100*(normal_ship['cost_lower']/normal_rule['cost_lower']-1):+.2f}%였다. 단위와 목적에 따라 순위가 달라질 수 있다."]
    lines += ["", "트럭이 자정까지 끝나지 않으면 자정까지의 경과시간을 쓴다. 긴 대기 경계는 일별 90분위의 평균이다. "
        "선박은 당일 하역 가능부터 마지막 안벽 작업까지 집계한다. 자정 이후 신규 도착은 없다.",
        "", "### 현행 다섯 규칙의 총비용", "",
        "| 하루 오더 | "+" | ".join(labels)+" |", "|---:|"+"---:|"*len(labels)]
    for n,r in rows.items():
        values=[r["rules"][k] for k in RULES]
        lines.append(f"| {int(n):,} | "+" | ".join(f"{v['cost_lower']/1e8:.2f}" if v["complete"]==1 else "미확정" for v in values)+" |")
    lines += ["", "단위는 하루 억원이다. 세 날짜 중 한 번이라도 배가 끝나지 않은 규칙은 총액을 확정하지 않았다.",
        "", "### 실제 경합에서 고른 작업", "",
        "| 하루 오더 | 현행 경합 기회(건/일) | 현행 본선 선택 | 현행 장기대기 트럭 선택 | 본선 우선 예외의 경합 기회 | 예외 적용 본선 선택 |",
        "|---:|---:|---:|---:|---:|---:|"]
    for n,r in rows.items():
        a,b=r["rules"]["SF_SPT"],r["vessel_exception"]["SF_SPT"]
        values=[a["opportunities"],a["chosen_vessel"],a["chosen_urgent"],b["opportunities"],b["chosen_vessel"]]
        lines.append(f"| {int(n):,} | "+" | ".join(f"{v:.1f}" for v in values)+" |")
    lines += ["", "한 기회는 자정 전 한 크레인에 24분 이상 트럭과 본선 실작업이 동시에 실행 가능 후보로 있었던 한 번이다. "
        "동시 장비 충돌 검사를 통과해야 실제 선택된다. 정책이 달라지면 다음 대기열도 달라지므로 두 조건의 경합 횟수 자체가 달라질 수 있다.",
        "", "### 짧은 작업 순서가 실제 선택을 바꿀 수 있었던 횟수", "",
        "| 첫 시험일 오더 | 단독 크레인의 실작업 선택 | 같은 등급에 2개 이상 | 시간을 빼면 다른 작업 선택 | 반입·반출 동시 후보 | 그중 반입 선택 |",
        "|---:|---:|---:|---:|---:|---:|"]
    for n,obs in s["decisions"].items():
        cols=("single_crane_service","at_least_two_in_tier","different_choice_without_duration","in_and_out_present","chosen_GATE_IN")
        lines.append(f"| {int(n):,} | "+" | ".join(f"{obs['counts'].get(k,0):,}" for k in cols)+" |")
    lines += ["", "첫 날짜의 기존 배정을 그대로 재생했다. 동시에 한 크레인만 결정하는 시점에 후보에서 소요시간 키만 지우고, "
        "실행 가능한 첫 선택이 달라지는지 세었다. 다음 대기열은 원래 선택대로 흘렀다. 이 표는 누적 정책 효과 표와 분모가 다르다.",
        "", "## 현장에 적용할 수 있는 범위", "",
        "- 부산 DGT는 블록마다 자동 야드크레인 2기를 두고 육측과 해측 작업을 분리한다고 설명한다. "
        "한 크레인 앞에서 트럭·본선이 경합하는 빈도를 해당 터미널에 옮기려면 장비별 작업 이력이 필요하다. [DGT 장비 소개](https://info.dgtbusan.com/DGT/terminal/equipment)",
        "- 부산 PNC의 2023년 선사 공지에는 건화물 수출 컨테이너에 본선 접안에 연동한 반입 마감을 둔 사례가 있다. "
        "출항할 배의 선적 가능성과 작업 순서가 연결되는 실제 사례다. [머스크의 PNC 안내](https://www.maersk.com/~/media_sc9/maersk/local-information/files/asia-pacific/korea/export/busan-pnc-export-full-container-late-gate-in-container-cut-off-rule-implementation.pdf)",
        "- 안벽·야드·차량을 함께 다룬 연구에서 운반차량의 도착이 다음 크레인의 작업 가능 시점을 바꾼다. "
        "베트남 터미널 조사에서 안정적 작업 중 고무차륜 야드크레인의 평균 사이클은 2.4분이었다. 장비·재취급·운반을 포함하는 범위가 달라 "
        "부산의 개별 작업시간이나 ±30% 오차로 직접 환산할 수 없다. [통합 배정 연구](https://www.sciencedirect.com/science/article/pii/S0377221712006832) · "
        "[실측 장비 연구](https://doi.org/10.3390/logistics6030044)",
        "- 부산권 기존 13일 표본에서 게이트 진입→완료 중앙값은 반입 16.08분, 반출 25.53분이었다. "
        "블록 도착→완료 중앙값(10.23분·18.63분)에도 줄에서 기다린 시간이 들어간다. "
        "이 기록에는 실제 장비 작업 시작·재취급·예상 대비 실현 소요시간이 함께 들어 있지 않다. "
        "장비별 대기열과 본선 작업 지시, 예상·실제 시작/종료의 짝이 있어야 현장 우선순위를 검증할 수 있다. [기존 실측 검토](../yr332_inbound_realism/README.md)",
        "", "기록과 계산의 일치는 원자료와 작업보드로 재검증한다. 모델 성능 판정은 직전 비교를 유지한다. "
        "현실성은 장비 역할·물량·시간오차의 현장 근거가 부족해 보류한다. 등록된 조사에는 대기구간·시작시각 확인, "
        "크레인 시간 근거 재감사와 고물량 관측 확보가 남아 있다. [등록 조사 명세](../../../.claude/docs/dashboard-task-specs/YR-332-bi-short-job-rule-advantage-audit.md)",
        "", "[전체 결과·일별 지표](summary.json) · [실행 원장](runs/) · [조건과 시드](run-info.json)", ""]
    (OUT/"README.md").write_text("\n".join(lines),encoding="utf8",newline="\n")


if __name__ == "__main__":
    report(summarize())
