# YR-080 2b — 기준재 교사 증류 (학생층 판정)

> 교사=numeraire {'crane_travel': 0.0, 'empty_travel': 0.0, 'sts_wait': 5.0} @1800s · 학생 재현성 우선(사용자 결정).
> top1 재현 전체 0.4918·분기한정 0.1601(n=1355) · best ep5. **문헌 보정 시뮬 조건.**

| 셀 | arm | berth(분) | P95(분) | 평균대기 | 건전 | 완주 |
|---|---|---|---|---|---|---|
| mid-loose | 학생 | 116.65 | 14.75 | 3.45 | OK | FAIL |
| mid-loose | 교사 | 31.64 | 21.33 | 4.035 | — | OK |
| high-loose | 학생 | 163.86 | 33.83 | 9.789 | OK | FAIL |
| high-loose | 교사 | 79.49 | 33.87 | 8.832 | — | OK |
| mid-tight | 학생 | 252.0 | 12.79 | 3.589 | OK | FAIL |
| mid-tight | 교사 | 116.17 | 17.58 | 4.087 | — | OK |
| high-tight | 학생 | 252.0 | 37.97 | 15.57 | OK | FAIL |
| high-tight | 교사 | 152.36 | 34.55 | 9.9 | — | OK |

## 판정: 학생 **붕괴/부분붕괴** (healthy_all=True·complete_all=False)

yr073(구목적) 붕괴 대조: healthy=false·확보율 57%. 여기서 학생이 건전·완주·교사 berth 재현하면 기준재 교사가 증류 가능(2a 교사층 실효가 배포로 전이).
미달 시 = 증류 자체가 병목(목적과 별개) → 학생 구조/DAgger/FT 후속.
