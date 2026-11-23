"""Independently reconcile the continuous fixed-month outcomes against the raw ledgers."""
from collections import Counter
import gzip
import json
import math

from month_crane_input import OUT
from vessel_turnaround_experiment import save
from yard_rl.v7.reward.krw import truck_wait_krw, vessel_krw_per_hour


def read(path):
    return json.loads(path.read_text("utf8"))


def lines(path):
    with gzip.open(path, "rt", encoding="utf8") as stream:
        yield from map(json.loads, stream)


def audit(arm):
    folder = OUT/"runs"/arm
    result = read(folder/"result.json")
    progress = read(folder/"progress.json")["daily"]
    ships = read(folder/"ships.json")
    trucks = list(lines(folder/"trucks.jsonl.gz"))
    waits = list(lines(folder/"cargo-waits.jsonl.gz"))
    future = {r["container"]: r for r in waits}
    jobs = {r["job"]:r for r in trucks}
    limits = read(OUT.parent/"yr332_month_supply"/"deficit-boundaries.json")
    boundaries = []
    for row in limits:
        t = row["cutoff_s"]
        tracked = [future[c] for c in row["waiting_containers"]]
        waiting = [r for r in tracked if r["block"] is not None and r["block"] <= t
                   and (r["out"] is None or r["out"] > t)]
        pending = sum(r["ready"] is None or r["ready"] > t for r in waiting)
        boundaries.append(dict(time_s=t,block=row["block"],planned=len(tracked),
            physically_waiting=len(waiting),awaiting_cargo=pending,
            yard_inventory=progress[int(t//86400)]["blocks"][row["block"]]))
    assert len(boundaries) == 14
    end = 35*86400
    assert len(waits) == len({r["container"] for r in waits}) == 2034
    assert all(r["out"] is None or r["ready"] is not None and r["ready"] <= r["out"] for r in waits)
    assert all(r["out"] == jobs[r["job"]]["finish"] for r in waits)
    assert all(jobs[r["job"]]["start"] is None or r["ready"] is not None and
               r["ready"] <= jobs[r["job"]]["start"] for r in waits)
    assert len(trucks) == len({r["job"] for r in trucks}) == result["admitted"] == 176500
    assert len(progress) == 36 and [r["day"] for r in progress] == list(range(36))
    assert result["skipped"] == 0 and len({r["ship"] for r in ships}) == len(ships)
    for r in trucks:
        dt = (end if r["finish"] is None else r["finish"])-r["gate"]
        assert math.isclose(r["seconds"],dt) and math.isclose(truck_wait_krw(dt),r["cost"])
    for r in ships:
        finishes = r["stream_finishes"]
        assert r["finish"] == (None if None in finishes else max(finishes))
        dt = (end if r["finish"] is None else r["finish"])-r["start"]
        assert math.isclose(r["seconds"],dt) and math.isclose(vessel_krw_per_hour(r["gt"])*dt/3600,r["cost"])
    for window,lo,hi in (("primary",1,29),("all",0,30)):
        a = [r for r in trucks if lo <= r["gate"]/86400 < hi]
        b = [r for r in ships if lo <= r["start"]/86400 < hi]
        w = result["windows"][window]
        assert (len(a),len(b)) == (w["trucks"],w["ships"])
        assert math.isclose(w["costs"]["truck"],sum(r["cost"] for r in a))
        assert math.isclose(w["costs"]["ship"],sum(r["cost"] for r in b))
        assert math.isclose(w["total"],sum(w["costs"].values()))
        assert math.isclose(w["truck_minutes"],sum(r["seconds"] for r in a)/len(a)/60)
        assert math.isclose(w["ship_hours"],sum(r["seconds"] for r in b)/len(b)/3600)
        assert w["unfinished_trucks"] == sum(r["finish"] is None for r in a)
        assert w["unfinished_ships"] == sum(r["finish"] is None for r in b)
    arrivals = Counter(int(r["gate"]//86400) for r in trucks)
    for i,row in enumerate(progress):
        assert row["entered"] == sum(n for d,n in arrivals.items() if d < i) + sum(
            r["gate"] == i*86400 for r in trucks)
        assert row["done"] == sum(r["finish"] is not None and r["finish"] <= i*86400 for r in trucks)
        assert row["cargo"]["physical_inventory"] == sum(row["blocks"].values()) == row["cargo"]["expected_inventory"]
    inputs = ({r["job"]:(r["gate"],r["flow"]) for r in trucks},
              {r["ship"]:(r["start"],r["gt"],len(r["stream_finishes"])) for r in ships})
    return result, inputs, dict(
        cross_midnight=sum(r["finish"] is not None and int(r["gate"]//86400)<int(r["finish"]//86400) for r in trucks),
        future_supply_exits=sum(r["out"] is not None for r in waits),
        future_supply_ready=sum(r["ready"] is not None for r in waits),deficit_midnights=boundaries,
        day30_backlog=progress[30]["unfinished"], peak_backlog=max(r["unfinished"] for r in progress),
        peak_block_inventory=max(max(r["blocks"].values()) for r in progress))


def main():
    results, audits, common = {}, {}, None
    for arm in ("SF_SPT","LWKR","trained"):
        results[arm], inputs, audits[arm] = audit(arm)
        if common is None:
            common = inputs
        else:
            assert inputs == common
    assert len({r["manifest_sha256"] for r in results.values()}) == 1
    summary = dict(runs=results,audits=audits,complete=all(
        r["windows"]["primary"]["unfinished_ships"] == r["windows"]["primary"]["unfinished_trucks"] == 0
        for r in results.values()))
    save(OUT/"summary.json",summary)
    text = ["# 30일 연속 운영에서의 동결 크레인 정책 비교", "",
        "세 방식에 같은 초기 야드와 고정 트럭 주문 176,500건을 투입했다. 30일간 자정마다 상태를 이어가고 5일간 잔여 작업을 추적했다.",
        "저장한 학습 모델 9951101은 최고점을 선택했다. 표에서 잔여가 있으면 비용과 시간은 동일 종료시각까지의 하한이다.", ""]
    names = {"SF_SPT":"짧은 작업 우선","LWKR":"오래 기다린 작업 우선","trained":"학습 모델"}
    for window, title in (("primary","2~29일 진입 작업"),("all","1~30일 전체 진입 작업")):
        text.extend([f"## {title}", "",
            "| 방식 | 총비용(억원) | 트럭(분/건) | 본선(시간/척) | 잔여 트럭 | 잔여 본선 |",
            "|---|---:|---:|---:|---:|---:|"])
        for arm, result in results.items():
            w = result["windows"][window]
            text.append(f"| {names[arm]} | {w['total']/1e8:.2f} | {w['truck_minutes']:.2f} | {w['ship_hours']:.2f} | {w['unfinished_trucks']:,} | {w['unfinished_ships']} |")
        text.append("")
    text.extend(["## 2~29일 비용 구성", "",
        "| 방식 | 트럭 시간(억원) | 본선 작업시간(억원) | 크레인 빈 이동(억원) | 재조작(억원) | 30일째 자정 잔여 트럭 |",
        "|---|---:|---:|---:|---:|---:|"])
    for arm,result in results.items():
        costs = result["windows"]["primary"]["costs"]
        text.append(f"| {names[arm]} · 비용 | {costs['truck']/1e8:.2f} | {costs['ship']/1e8:.2f} | {costs['empty']/1e8:.2f} | {costs['rehandles']/1e8:.2f} | {audits[arm]['day30_backlog']:,} |")
    text.append("")
    text.extend(["트럭은 게이트 진입부터 야드 작업 완료, 본선은 하역 가능부터 마지막 안벽 작업 완료까지 측정했다. 장비의 빈 이동과 재조작은 각 달력 구간에 발생한 비용이다.",
        "한 개의 고정 합성 월과 모델 한 개를 비교했다. 이미 공급을 조사한 달이며 섞인 고물량 날짜는 혼잡 시험이다.",
        "", "[집계와 이월 검산](summary.json) · [입력 명단 지문](input.json) · [일별 기록과 원자료](runs/)"])
    (OUT/"README.md").write_text("\n".join(text)+"\n",encoding="utf8")


if __name__ == "__main__":
    main()
