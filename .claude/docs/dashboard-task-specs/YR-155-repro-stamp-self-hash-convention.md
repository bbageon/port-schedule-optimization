# YR-155 — 재현 스탬프의 자기 해시 규약 정정 (sidecar 분리)

- **Epic**: Infra / **Priority**: 🟡 / **등록일**: 2026-08-06 / **상태**: done
- **3대 게이트 보정 대상**: `reliability` 하나 — 착수 시 YR-153 `authorize-next` 통과 필수
- **발견 경위**: YR-151 0A 를 게이트 하네스에 보고하는 과정에서 확인 —
  [gate report](../../../outputs/reports/yr153_research_gates/report.md)

## ★결과 (2026-08-17) — 완료

`repro.py` 에 공유 헬퍼 3종 신설: `write_result()`·`sha256_file()`·`verify_result()`.
개별 실험이 각자 구현하지 않게 **공유 헬퍼로** 둬 재발을 막는다.

| | |
|---|---|
| 새 규약 | `<result>.json` 확정 후 `<result>.json.sha256` sidecar |
| 자기검증 | **성립함** — sidecar 값 == 실제 파일 해시 |
| 변조 탐지 | 한 글자 추가 → `False` |
| 구 산출물 | sidecar 없음 → `None`(미상), 위반 아님 |

과거 `self_sha256` 필드는 **지우지 않고** 뜻풀이(`convention_since`·
`sidecar_verified`)만 덧붙였다 — 그 값이 무엇이었는지 남아야 과거 판정을
읽을 수 있다. 시험 36/36(신규 5 + 기존 31).

## 무엇이 문제인가

결과 JSON 안에 **자기 자신의 sha256** 을 적는 방식(`self_sha256`)은 원리상 자기검증이 되지 않는다.
파일을 쓰고 → 해시를 재고 → 그 해시를 파일 안에 덧쓰는 순간 파일 내용이 바뀌므로,
기록된 값은 **덧쓰기 전 파일**의 해시가 된다. 실제로 YR-151 0A 에서 기록값 `4862ae71…` 과
현재 파일 해시 `c287a9d5…` 가 다르다. 검증하는 쪽은 이 값을 쓸 수 없다.

0A 게이트 판정은 이 값 대신 **커밋된 실제 파일의 해시**로 고정했으므로 판정 자체는 유효하다.
문제는 규약이며, 그대로 두면 다음 실험도 쓸 수 없는 값을 계속 남긴다.

## 무엇을 하는가

1. 결과 JSON 안의 `self_sha256` 을 **없애고**, 같은 이름의 sidecar 파일
   `<result>.json.sha256` 에 해시를 쓴다(결과 파일은 한 번만 쓰고 더 건드리지 않는다).
2. 이미 남은 `self_sha256` 필드는 **지우지 않고**, "덧쓰기 전 해시 — 검증용 아님" 이라는
   뜻풀이를 하네스 문서에 남긴다(과거 산출물 소급 재작성 금지).
3. `judge_runtime_evidence` 에 넘기는 `artifact_hashes` 는 sidecar 값을 읽어 쓰고,
   sidecar 가 없으면 실제 파일 해시를 계산해 쓰되 그 사실을 evidence 에 표시한다.

## 완료 조건

- 새 실험 1건이 sidecar 방식으로 산출물을 남기고, 하네스가 그 값으로 재현 사슬 검사를 통과.
- `.claude/skills/research-gates.md` §1-3 의 "sha256 을 기록한다" 문장이 sidecar 규약으로 갱신.
