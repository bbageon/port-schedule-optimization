"""영어판 그림 ([[YR-256]]) — outputs/v3/figures/en/*.svg

    PYTHONPATH=src python scripts/v3/figures.py      # 먼저 한국어판을 만들고
    PYTHONPATH=src python scripts/v3/figures_en.py   # 그 라벨만 영어로 바꾼다

■ 왜 다시 그리지 않고 치환하나
  두 판의 **수치·좌표가 어긋나면** 논문 두 판이 다른 그림을 싣게 된다. 같은 SVG 를
  만들고 `<text>` 만 바꾸면 두 판이 픽셀 단위로 같은 그림을 쓴다.

  긴 영어 라벨이 짧은 한국어 자리에 들어가면 넘칠 수 있으므로, 길이가 크게 늘어나는
  항목은 줄여 적었다(예: "제안 정책 (도착시각만)" → "Proposed (time)").
"""
from __future__ import annotations

import pathlib
import re
import sys

SRC = pathlib.Path("outputs/v3/figures")
OUT = SRC / "en"

#: 정확히 일치하는 라벨. 긴 것부터 바꿔야 짧은 것이 먼저 먹지 않는다.
EXACT = {
    # 제목·부제
    "그림 1. 재배치 결정 흐름과 학습·운영의 분리":
        "Fig. 1. Reallocation decision flow and the learning/operation split",
    "그림 2. 시간대별 트럭 도착밀도": "Fig. 2. Time-of-day truck arrival density",
    "24시간 균등 바닥 38% + 정규분포 봉우리 3개":
        "Uniform base of 38% over 24 h plus three normal peaks",
    "그림 3. 회차별 학습·검증 손실과 탐색 확률":
        "Fig. 3. Training and validation loss with exploration probability",
    "하루가 한 회차 · 28일 학습": "One day is one iteration; 28 days trained",
    "그림 4. 정책별 28일 총비용 감소율": "Fig. 4. 28-day cost reduction rate by policy",
    "재배치 없음 대비 · 양수가 비용 감소":
        "Relative to no reallocation; positive is a reduction",
    "그림 5. 날짜별 비용 차이 (제안 정책 − 재배치 없음)":
        "Fig. 5. Daily cost difference (proposed − no reallocation)",
    "28일을 차이 크기로 정렬 · 음수가 제안 정책의 비용이 낮은 날":
        "28 days sorted by magnitude; negative = proposed policy is cheaper",
    "그림 6. 수요수준별·행동유형별 비용 감소":
        "Fig. 6. Cost reduction by demand level and action type",
    "재배치 없음 대비 · 양수가 감소 · 같은 28일":
        "Relative to no reallocation; same 28 days",
    "그림 7. 크레인 작업순서 규칙의 수요 민감도":
        "Fig. 7. Demand sensitivity of crane dispatching rules",
    "재배치 없음 · 각 수요에서 가장 낮은 총비용을 1.0 으로 둔 상대값":
        "No reallocation; lowest total cost at each demand level set to 1.0",
    # 그림 1 상자
    "관측": "Observe", "제안 정책": "Proposal", "수락 정책": "Acceptance",
    "중앙 확정": "Commit", "실행": "Execute",
    "블록 혼잡·공개 도착예정": "block congestion, public ETA",
    "행동별 예상비용 argmin": "argmin of predicted cost",
    "수락·거절 비교": "accept vs. reject",
    "합의·용량 확인 후 일괄 반영": "consent + capacity, applied at once",
    "야드·크레인 상태 갱신": "yard and crane state update",
    "운영 경로 — 학습된 신경망만 사용, 반사실 호출 0회":
        "Operating path — trained networks only, zero counterfactual calls",
    "학습 경로 — 한 결정에서 상태·난수를 복제해 최대 세 세계를 3시간 실행":
        "Learning path — clone state and random stream, run up to three worlds for 3 h",
    "관측 행동 · 제안만 반대 · 수락만 반대  →  Φ 차이를 중심화해 회귀 목표로 사용":
        "observed / proposal reversed / acceptance reversed  →  centred Φ difference as target",
    "학습에서 만든 비용 차이만 정책에 들어가고, 운영에서는 분기 시뮬레이션을 호출하지 않는다.":
        "Only the cost difference produced in training enters the policy; operation invokes no branch simulation.",
    # 축·범례
    "밀도": "density", "억원": "KRW bn", "회차 (일)": "iteration (day)",
    "탐색 확률": "exploration", "사용 행동": "action range",
    "제안 학습": "proposal (train)", "제안 검증": "proposal (val)",
    "수락 학습": "acceptance (train)", "수락 검증": "acceptance (val)",
    "혼잡·초혼잡 수요일": "congested demand days", "그 외 감소일": "other reduction days",
    "증가일": "increase days", "일일 트럭 수요": "daily truck demand",
    "두 행동 모두": "both actions", "도착시각만": "arrival time only",
    "블록만": "block only", "학습 전 모형": "untrained model",
    # 정책 이름
    "제안 정책 (시각만)": "Proposed (time only)",
    "제안 정책 (블록만)": "Proposed (block only)",
    "재배치 없음": "No reallocation",
    "규칙: 한산한 시간대": "Rule: least-loaded slot",
    "규칙: 한산한 블록·시간대": "Rule: least-loaded block+slot",
    "규칙: 선착순": "Rule: FCFS",
    "규칙: 최단 처리시간": "Rule: SPT",
    "규칙: 최소 여유시간": "Rule: least slack",
    "규칙: 순이득 기준": "Rule: net gain",
    "블록+시각": "block+time", "시각": "time", "블록": "block",
    "균등 바닥 (야간 포함)": "uniform base (incl. night)",
    "블록만 쓰는 정책은 학습·규칙 모두 0 부근이고, 감소는 도착시각 조정에서 나온다.":
        "Block-only policies sit near zero for both learned and rule-based cases; the reduction comes from arrival-time adjustment.",
    # 크레인 규칙
    "최단 처리시간 (현행)": "SPT (current)", "선착순": "FCFS", "후착순": "LIFO",
    "누적 대기 우선": "longest wait", "최근접": "nearest", "무작위": "random",
}

#: 숫자가 섞인 라벨 — 정규식으로 바꾼다.
PATTERNS = [
    (re.compile(r"제안 정책이 낮은 날 (\d+)/(\d+) · 양측 부호검정 p(&lt;|<)0\.001"),
     r"Proposed policy lower on \1/\2 days · two-sided sign test p&lt;0.001"),
    (re.compile(r"(\d[\d,]*)대"), r"\1"),
    (re.compile(r"(\d+)일(?![가-힣])"), r"\1 d"),
    (re.compile(r"(\d+)시(?![가-힣])"), r"\1:00"),
    (re.compile(r"(\d+)천(?![가-힣])"), r"\1k"),
]


def translate(svg: str) -> str:
    def one(m):
        s = m.group(1)
        if s in EXACT:
            return f">{EXACT[s]}<"
        for pat, rep in PATTERNS:
            new = pat.sub(rep, s)
            if new != s:
                s = new
        return f">{s}<"

    return re.sub(r">([^<>]+)<", one, svg)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob("*.svg"))
    if not files:
        print("먼저 scripts/v3/figures.py 로 한국어판을 만든다"), sys.exit(1)
    left = set()
    for f in files:
        s = f.read_text(encoding="utf-8")
        out = translate(s)
        for m in re.findall(r">([^<>]*[가-힣][^<>]*)<", out):
            left.add(m)
        (OUT / f.name).write_text(out, encoding="utf-8")
        print(" ·", OUT / f.name)
    if left:
        print("\n⚠️ 아직 한글인 라벨 — EXACT 에 추가한다:")
        for s in sorted(left):
            print("   ", s)
    else:
        print("\n■ 한글 라벨 남은 것 없음")
    return 0


if __name__ == "__main__":
    sys.exit(main())
