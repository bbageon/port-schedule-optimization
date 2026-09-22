"""교착 수정 대조 보고서 (YR-317-k).

세 자료를 나란히 놓는다.

1. **등록 기록** — 80건 캠페인이 고정 소스에서 실제로 남긴 시드 21,000,000 재배정없음
   실행의 날별 확정값. 정지가 실제로 일어난 그 실행이다.
2. **수정 전 재생** — 지금 작업 트리에서 같은 입력·같은 정책으로 다시 굴린 것. 1과
   맞으면 기본 경로가 **바뀌지 않았다**는 증거다.
3. **수정 후 재생** — 같은 조건에서 후보 가지치기만 고친 것.

⚠️ **미완료 트럭 수는 서로 비교하지 않는다.** 등록 기록은 배수 구간까지 흡수한 뒤의
확정값이고, 재생의 중간보고는 그 흡수 전 잠정값이라 정의가 다르다. 비용은 같은
정의라 비교할 수 있다.

진단 증거이지 등록된 결과가 아니다 — 시드 하나·블록 하나다.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RECORDED = (ROOT / 'outputs/reports/yr317_v3_independent_eval/run-7e2fb14'
            / 'months/21000000/NO_REALLOC/daily-final.jsonl')
SOURCES = (
    ('fixed', '수정 후', HERE / 'fix-validation-9d/validate-feasible_first.json'),
    ('legacy', '수정 전 재생', HERE / 'fix-validation/validate-legacy.json'),
)


def read_replay(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def read_recorded():
    if not RECORDED.exists():
        return None
    return [json.loads(line) for line in RECORDED.read_text(encoding='utf-8').splitlines() if line]


def span(run):
    first, last = run.get('first_stall'), run.get('last_stall')
    return None if not first else (first['at_day'], last['at_day'])


def main():
    replays = {key: read_replay(path) for key, _, path in SOURCES}
    missing = [str(path) for key, _, path in SOURCES if replays[key] is None]
    if missing:
        raise SystemExit('아직 없음: ' + ', '.join(missing))
    recorded = read_recorded()
    names = {key: name for key, name, _ in SOURCES}

    lines = ['# 교착 수정 대조 — 시드 21,000,000 · 블록 Y17 · 재배정 없음', '',
        __doc__.split('진단 증거')[0].strip(), '',
        '## 1. 정지 신호', '',
        '| 재생 | 정지 신호 구간 | 신호가 잡힌 5분 표본 | 마지막까지 지속 | 실시간 |',
        '|---|---|---:|---|---:|']
    for key, name, _ in SOURCES:
        run = replays[key]
        where = span(run)
        shown = '없음' if where is None else f'{where[0]:.2f}일 → {where[1]:.2f}일'
        end = run['until_day'] - 0.5
        holds = '—' if where is None else ('**예 (안 풀림)**' if where[1] >= end else '아니오 (풀림)')
        lines.append(f"| {name} | {shown} | {run['stall_windows']} | {holds} | "
                     f"{run['elapsed_s'] / 3600:.2f}시간 |")

    lines += ['', '## 2. 날별 비용 (억원)', '',
              '정지가 나면 그 블록의 대기비용이 평가창 끝까지 쌓여 하루 비용이 백 배로 뛴다.',
              '', '| 날 | 부하 | 등록 기록 (수정 전) | 수정 전 재생 | 수정 후 재생 |',
              '|---:|---:|---:|---:|---:|']
    by_day = {key: {row['index']: row for row in replays[key]['days']} for key in replays}
    saved = {row['index']: row for row in (recorded or [])}
    for index in sorted(set(saved) | set().union(*(set(v) for v in by_day.values()))):
        if index > max(max(v, default=-1) for v in by_day.values()):
            break
        load = next((r['load'] for r in (saved.get(index), by_day['legacy'].get(index),
                                         by_day['fixed'].get(index)) if r), '')
        cells = []
        for row in (saved.get(index), by_day['legacy'].get(index), by_day['fixed'].get(index)):
            cells.append('' if row is None else f"{row['phi_krw'] / 1e8:,.2f}")
        lines.append(f"| {index} | {load:,} | " + ' | '.join(cells) + ' |')

    lines += ['', '## 3. 기본 경로가 바뀌지 않았는가', '']
    if recorded:
        agree = [index for index in sorted(saved)
                 if index in by_day['legacy']
                 and abs(saved[index]['phi_krw'] - by_day['legacy'][index]['phi_krw']) < 1.0]
        checked = [index for index in sorted(saved) if index in by_day['legacy']]
        lines.append(f'- 등록 기록과 수정 전 재생이 비용까지 일치한 날: **{len(agree)}/{len(checked)}**')
        lines.append('- 일치하면 옵트인 기본값(`legacy`)이 기존 80건을 그대로 재현한다는 뜻이다.')
    else:
        lines.append('- 등록 기록 파일을 찾지 못해 대조하지 못했다.')

    lines += ['', '## 4. 하루당 실시간 — 캠페인 추정 갱신용', '']
    for key, name, _ in SOURCES:
        run = replays[key]
        last = run['days'][-1]
        lines.append(f"- {name}: 하루 평균 **{last['elapsed_s'] / (last['index'] + 1) / 60:.1f}분** "
                     f"({last['index'] + 1}일 기준)")
    pace = {key: replays[key]['days'][-1]['elapsed_s'] / (replays[key]['days'][-1]['index'] + 1)
            for key in replays}
    if pace['legacy']:
        lines.append(f"- **수정 후 ÷ 수정 전 = {pace['fixed'] / pace['legacy']:.2f}배** — "
                     '[[YR-318]] 실행시간 추정을 이 값으로 갱신한다.')
    lines += ['', '재생기: `validate_fix.py` · 원자료: `fix-validation*/validate-*.json` · '
              f'등록 기록: `{RECORDED.relative_to(ROOT).as_posix()}`', '']

    out = HERE / 'fix-comparison.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({names[k]: dict(stall_windows=replays[k]['stall_windows'], span=span(replays[k]),
        hours=round(replays[k]['elapsed_s'] / 3600, 2)) for k in replays}, ensure_ascii=False))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
