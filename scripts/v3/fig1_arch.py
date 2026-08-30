"""그림 1 재도안 — 재배치 아키텍처 (검토용 시안).

    PYTHONPATH=src python scripts/v3/fig1_arch.py

■ 왜 별도 파일인가
  사용자가 **먼저 확인**한 뒤 채택하기로 했다. 승인되면 `figures.py:fig1()` 과
  `figures_en.py` 의 라벨 사전으로 접어 넣고 이 파일은 지운다.

■ ★좌표계를 바꿨다 — 1 단위 = 인쇄 1pt
  옛 도안은 viewBox 760 을 `width=0.68\\textwidth` 로 넣어서, 본문 12.5pt 옆에서
  그림 글씨가 **2.7~3.7pt** 로 앉았다(main.log: 572.14pt 원본 → 236.04pt 요청,
  배율 0.41 · svglib 는 1px=0.75pt 로 낸다). 읽을 수 없는 크기다.

  LNCS 본문 폭이 12.2cm = 347.1pt 이므로 viewBox 폭을 **347** 로 두고
  `width=\\textwidth` 로 넣으면 배율이 1.0 이 되어 **글꼴 숫자가 곧 인쇄 pt** 다.
  아래 6pt·7pt·8pt 는 전부 실제 인쇄 크기다.

■ 빼기 기호는 ASCII 하이픈이다
  Malgun Gothic 에 U+2212(MINUS SIGN) 글리프가 없어 PDF 변환에서 빈칸이 된다.
  실측: `fontTools` cmap 조회 False. 6pt 에서는 하이픈으로 충분히 읽힌다.

■ 두 판을 한 곳에서 낸다
  영문판을 `figures_en.py` 처럼 사후 치환하지 않고 같은 좌표에 라벨만 갈아 낸다 —
  도안이 조밀해서 치환하면 폭 초과를 못 잡는다.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from svg import FAINT, Fig, GRAY, INK, MUTED, NAVY, RULE  # noqa: E402

OUT = pathlib.Path("outputs/v3/figures")
W, H = 347, 250
ON_NAVY = "#c8d4e4"
BAND = "#f2f3f1"

# ─────────────────────────────────────────────────────────── 좌표 (인쇄 pt)
BOX_Y, BOX_H = 22.0, 54.0
BOX = {                      # (x, w)
    "elig": (0.5, 57.0), "prop": (65.5, 82.0),
    "acc": (197.5, 82.0), "commit": (287.5, 59.0),
}
BOUND_X = 172.5              # 정보 경계 (제안·수락 사이 50pt 통로의 중앙)
LANE_Y = 84.0                # 유지로 되돌아가는 가로선
KEEP = (150.0, 90.0, 100.0, 16.0)    # 원배정 유지 상자 (x, y, w, h)
DROP_X = 308.0               # 확정에서 탈락이 떨어지는 자리
WORLD_Y, WORLD_H = 118.0, 44.0
LEARN_Y, LEARN_H = 178.0, 69.0
CELL_W, CELL_GAP = 104.0, 8.0
CELL_X = (9.5, 121.5, 233.5)
CELL_Y, CELL_H = 186.0, 31.0


def cx(key: str) -> float:
    x, w = BOX[key]
    return x + w / 2


# ─────────────────────────────────────────────────────────── 화살촉
def head(f: Fig, x: float, y: float, d: str, c: str = NAVY, s: float = 3.6):
    """(x, y) 에 꼭짓점을 둔 삼각형. d = R|L|D|U."""
    seg = {"R": f"l-{s} -{s*.72} v{s*1.44} z", "L": f"l{s} -{s*.72} v{s*1.44} z",
           "D": f"l-{s*.72} -{s} h{s*1.44} z", "U": f"l-{s*.72} {s} h{s*1.44} z"}[d]
    f.path(f"M{x:.1f} {y:.1f} {seg}", None, c)


def arrow(f: Fig, x1, y1, x2, y2, d, c=NAVY, w=1.0):
    f.path(f"M{x1:.1f} {y1:.1f} L{x2:.1f} {y2:.1f}", c, w=w)
    head(f, x2, y2, d, c)


# ─────────────────────────────────────────────────────────── 본체
def build(L: dict, path: pathlib.Path) -> pathlib.Path:
    f = Fig(W, H, pad=(0, 0, 0, 0))
    f.text(0, 10, L["title"], 8.5, INK, weight=600)

    # ── ① 운영 경로 ────────────────────────────────────────────────────
    f.text(0, 18.5, L["op_band"], 7, MUTED, weight=500)
    f.text(BOUND_X, 18.5, L["boundary"], 6, FAINT, "middle")

    for key, dark, title, subs, ts in (
        ("elig", False, L["elig"], L["elig_sub"], 8),
        ("prop", True, L["prop"], L["prop_sub"], 8),
        ("acc", True, L["acc"], L["acc_sub"], 8),
        ("commit", False, L["commit"], L["commit_sub"], 7.5),
    ):
        x, w = BOX[key]
        f.rect(x, BOX_Y, w, BOX_H, NAVY if dark else "#ffffff",
               NAVY if dark else RULE, rx=3)
        f.text(x + w / 2, 34, title, ts, "#ffffff" if dark else INK, "middle", 600)
        ys = (48.0, 57.0, 66.0) if len(subs) == 3 else (45.5, 54.0, 62.5, 71.0)
        for y, s in zip(ys, subs):
            f.text(x + w / 2, y, s, 6 if dark or key == "elig" else 5.8,
                   ON_NAVY if dark else MUTED, "middle")

    # 자격 → 제안, 수락 → 확정
    arrow(f, BOX["elig"][0] + BOX["elig"][1] + 1, 49, BOX["prop"][0] - 1.5, 49, "R")
    arrow(f, BOX["acc"][0] + BOX["acc"][1] + 1, 49, BOX["commit"][0] - 1.5, 49, "R")

    # 정보 경계 — 끊어 그려서 라벨·화살표 자리를 비운다
    for y1, y2 in ((21, 40), (64, 78)):
        f.line(BOUND_X, y1, BOUND_X, y2, GRAY, 0.8, dash="2 3")
    # 경계를 건너는 유일한 통로
    arrow(f, BOX["prop"][0] + BOX["prop"][1] + 1.5, 49, BOX["acc"][0] - 1.5, 49, "R")
    f.text(BOUND_X, 45, L["offer"], 6, INK, "middle", 600)
    f.text(BOUND_X, 59, L["offer_sub"], 5.5, FAINT, "middle")

    # ── ② 원배정 유지로 되돌아가는 세 갈래 ────────────────────────────
    for x, lab in ((cx("prop"), L["keep_p"]), (cx("acc"), L["keep_a"]),
                   (DROP_X, L["keep_c"])):
        f.line(x, BOX_Y + BOX_H, x, LANE_Y, GRAY, 0.9)
        f.text(x + 2.5, LANE_Y - 2.5, lab, 5.5, MUTED)
    f.line(cx("prop"), LANE_Y, DROP_X, LANE_Y, GRAY, 0.9)
    kx, ky, kw, kh = KEEP
    arrow(f, kx + kw / 2, LANE_Y, kx + kw / 2, ky - 0.5, "D", GRAY, 0.9)
    f.rect(kx, ky, kw, kh, "#ffffff", RULE, rx=3)
    f.text(kx + kw / 2, ky + 10.5, L["keep_box"], 6.2, INK, "middle", 500)

    # ── ③ 세계 ────────────────────────────────────────────────────────
    f.text(0.5, WORLD_Y - 4.5, L["world_band"], 7, MUTED, weight=500)
    f.rect(0.5, WORLD_Y, W - 1, WORLD_H, BAND, RULE, rx=3)

    arrow(f, 334, BOX_Y + BOX_H, 334, WORLD_Y - 0.5, "D")
    f.text(330, 91, L["commit_a"], 5.5, MUTED, "end")
    f.text(330, 99, L["commit_b"], 5.5, MUTED, "end")
    arrow(f, kx + kw / 2, ky + kh, kx + kw / 2, WORLD_Y - 0.5, "D", GRAY, 0.9)
    arrow(f, cx("elig"), WORLD_Y, cx("elig"), BOX_Y + BOX_H + 0.5, "U", GRAY, 0.9)
    f.text(cx("elig") + 3.5, 100, L["observe"], 5.5, MUTED)

    n, cw, cg = 21, 4.5, 1.5
    span = n * cw + (n - 1) * cg
    x0 = 173.5 - span / 2
    for i in range(n):
        f.rect(x0 + i * (cw + cg), 127.5, cw, 8, "#ffffff", GRAY, rx=1)
    f.text(x0 - 16, 133.5, L["gate"], 6, MUTED, "end")
    arrow(f, x0 - 13, 131.5, x0 - 3.5, 131.5, "R", GRAY, 0.9)
    arrow(f, x0 + span + 2.5, 131.5, x0 + span + 12, 131.5, "R", GRAY, 0.9)
    f.text(x0 + span + 15.5, 133.5, L["quay"], 6, MUTED)
    f.text(173.5, 147, L["world_1"], 6, INK, "middle")
    f.text(173.5, 156.5, L["world_2"], 6, MUTED, "middle")

    # ── ④ 학습 경로 ───────────────────────────────────────────────────
    f.text(0.5, LEARN_Y - 5.5, L["learn_band"], 7, MUTED, weight=500)
    arrow(f, 300, WORLD_Y + WORLD_H, 300, LEARN_Y - 0.5, "D", GRAY, 0.9)
    f.text(296, 171, L["snapshot"], 5.5, MUTED, "end")
    f.rect(0.5, LEARN_Y, W - 1, LEARN_H, BAND, RULE, rx=3)

    for x, (t, a, b) in zip(CELL_X, L["worlds"]):
        f.rect(x, CELL_Y, CELL_W, CELL_H, "#ffffff", RULE, rx=3)
        f.text(x + CELL_W / 2, 196, t, 7, INK, "middle", 600)
        f.text(x + CELL_W / 2, 205.5, a, 5.8, MUTED, "middle")
        f.text(x + CELL_W / 2, 214, b, 5.8, MUTED, "middle")
    arrow(f, 173.5, CELL_Y + CELL_H, 173.5, 223.5, "D", GRAY, 0.9)
    f.text(173.5, 232, L["target"], 6, INK, "middle")
    f.text(173.5, 241.5, L["fit"], 6, MUTED, "middle")
    return f.save(path)


# ─────────────────────────────────────────────────────────── 라벨
KO = {
    "title": "그림 1. 재배치 아키텍처와 학습·운영의 분리",
    "op_band": "운영 경로 — 60초마다 결정",
    "boundary": "정보 경계",
    "elig": "자격", "elig_sub": ("게이트 전", "통지창 첫 진입", "작업당 한 번"),
    "prop": "제안 정책",
    "prop_sub": ("자기 블록만 관측", "유지 · 다른 블록 · 이연 칸",
                 "21 → 64 → 64 → 1", "예상비용 argmin"),
    "acc": "수락 정책",
    "acc_sub": ("받는 쪽만 관측", "용량 0 → 망 없이 거절",
                "16 → 64 → 64 → 1", "수락 · 거절 argmin"),
    "commit": "중앙 확정",
    "commit_sub": ("동의한 것만", "작업 한 번", "블록 잔여슬롯", "칸 정원"),
    "offer": "제안 5칸", "offer_sub": "공개 정보만",
    "keep_p": "유지", "keep_a": "거절", "keep_c": "탈락",
    "keep_box": "원배정 유지 · 즉시 잠금",
    "commit_a": "원자 트랜잭션", "commit_b": "실패하면 유지",
    "observe": "상태 관측",
    "world_band": "세계 — 이산사건 시뮬레이터 · 30일 연속",
    "gate": "게이트", "quay": "안벽 · 선박",
    "world_1": "야드 21블록 · 블록당 크레인 2기 · 작업순서는 고정 규칙",
    "world_2": "Φ = 트럭 체류 + 추가 이동 + 재취급 + 선박 유휴  [원]",
    "learn_band": "학습 경로 — 학습 회차에서만 · 운영에서는 호출 0회",
    "snapshot": "결정 직전 복제",
    "worlds": (("세계 1", "실제 그대로", "3시간 굴려 Φ"),
               ("세계 2", "제안만 반대", "수락 정책은 그대로 굴린다"),
               ("세계 3", "수락만 반대", "제안이 있었을 때만")),
    "target": "목표 = (Φ - 그 결정의 기준선) ÷ 10만원 · 라벨은 그날만 쓴다",
    "fit": "두 망을 한 스텝에서 동시 회귀 · 갱신된 가중치가 다음 날 정책이 된다",
}

EN = {
    "title": "Fig. 1. Reallocation architecture and the learning/operation split",
    "op_band": "Operating path — every 60 s",
    "boundary": "information boundary",
    "elig": "Eligibility", "elig_sub": ("before gate-in", "window entry", "once per job"),
    "prop": "Proposal policy",
    "prop_sub": ("sees own block only", "keep · block · slot",
                 "21 → 64 → 64 → 1", "argmin predicted cost"),
    "acc": "Acceptance policy",
    "acc_sub": ("sees receiving side", "capacity 0 → reject",
                "16 → 64 → 64 → 1", "argmin accept/reject"),
    "commit": "Commit",
    "commit_sub": ("consented only", "one job once", "block slots", "slot quota"),
    "offer": "offer · 5 values", "offer_sub": "public only",
    "keep_p": "keep", "keep_a": "reject", "keep_c": "dropped",
    "keep_box": "original assignment, locked",
    "commit_a": "atomic transaction", "commit_b": "fail-closed → keep",
    "observe": "state",
    "world_band": "World — discrete-event simulation, 30 days",
    "gate": "gate", "quay": "quay / vessels",
    "world_1": "21 yard blocks · 2 cranes each · fixed dispatching rule",
    "world_2": "Φ = truck dwell + extra moves + rehandles + vessel idle  [KRW]",
    "learn_band": "Learning path — training only; zero calls at operating time",
    "snapshot": "cloned before the decision",
    "worlds": (("World 1", "as observed", "run 3 h → Φ"),
               ("World 2", "proposal flipped", "acceptance policy runs as-is"),
               ("World 3", "acceptance flipped", "only if an offer was made")),
    "target": "target = (Φ - decision baseline) ÷ 100,000 · labels used that day only",
    "fit": "both networks regress in one step; updated weights drive the next day",
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "en").mkdir(parents=True, exist_ok=True)
    for lab, p in ((KO, OUT / "fig1-architecture-v2.svg"),
                   (EN, OUT / "en" / "fig1-architecture-v2.svg")):
        print(" ·", build(lab, p))
    print(f"■ viewBox {W}×{H} · LNCS 본문폭 12.2cm 에 width=\\textwidth 로 넣으면 "
          f"1 단위 = 1pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
