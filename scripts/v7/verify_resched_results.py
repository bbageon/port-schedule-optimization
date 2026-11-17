"""Check the completed report's evidence without rerunning the experiment."""
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from resched_ppo import OUT, check, save
from six_step_learning import gz_read, sha


def main():
    check()
    campaign = json.loads((OUT / 'campaign.json').read_text('utf-8'))
    summary = json.loads((OUT / 'summary.json').read_text('utf-8'))
    for path, expected in summary['artifact_hashes'].items():
        assert sha(Path(path)) == expected, path
    assert all(summary[k] for k in ('source_checks_passed', 'same_input_and_cohort',
                                    'weight_and_action_replay_passed'))
    tests = 0
    for name in ('on-policy-tests-final.xml', 'regression-tests.xml'):
        for suite in ET.parse(OUT / name).getroot().iter('testsuite'):
            assert int(suite.attrib['failures']) == int(suite.attrib['errors']) == 0
            tests += int(suite.attrib['tests'])
    assert tests == 44
    counts = {'main': 0, 'repeat': 0}
    cases = {'main': 0, 'repeat': 0}
    for path in sorted((OUT / 'orders').glob('*.json.gz')):
        kind = 'repeat' if 'repeat' in path.name else 'main'
        counts[kind] += len(gz_read(path))
        cases[kind] += 1
    assert counts == dict(main=468000, repeat=7000), counts
    assert cases == dict(main=54, repeat=2), cases
    documents = [OUT / 'README.md',
        Path('docs/research/v7-discharge/resched-ppo-clarifications.md'),
        Path('.claude/docs/strategy-history/2026-11-17-YR-332-av-resched-ppo.md'),
        *Path('.claude/docs/dashboard-task-specs').glob('YR-332-a[vwx]-*.md')]
    for path in documents:
        body = path.read_text('utf-8')
        assert len(body.splitlines()) <= 200, path
        for target in re.findall(r'\]\(([^)]+)\)', body):
            if '://' not in target:
                resolved = (path.parent / target.split('#')[0]).resolve()
                if resolved != (OUT / 'verification.json').resolve():
                    assert resolved.exists(), (path, target)
    source = json.loads((OUT / 'source-publication-audit.json').read_text('utf-8'))
    replay = [json.loads((OUT / f'{kind}-9951101/replay.json').read_text('utf-8'))
              for kind in ('mlp', 'transformer')]
    assert all(row['passed'] and row['updates'] == 6 for row in replay)
    result = dict(passed=True, tests=tests, evaluation_cases=cases, truck_order_rows=counts,
        checked_artifact_hashes=len(summary['artifact_hashes']), source_publication=source,
        weight_replay_updates=sum(row['updates'] for row in replay),
        manual_figure_review=dict(passed=True, file=(OUT/'schedule-window.png').as_posix(),
                                 sha256=sha(OUT/'schedule-window.png'),
                                 scope='Korean labels, axis limits, fixed selection and layout inspected'),
        report_sha256=sha(OUT/'README.md'), summary_sha256=sha(OUT/'summary.json'),
        protocol_unchanged=True, source_commit=campaign['repro']['code']['git_head'],
        limitations=['No full training-environment replay', 'Three shared synthetic evaluation days',
                     'Real-terminal validity and adoption conditions remain unmet'])
    save(OUT / 'verification.json', result)
    print({k: v for k, v in result.items() if k not in ('source_publication', 'manual_figure_review')})


if __name__ == '__main__':
    main()
