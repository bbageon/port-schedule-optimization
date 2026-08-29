"""논문 그림용 최소 SVG 헬퍼 ([[YR-254]]).

matplotlib 를 쓰지 않는다 — 이 환경에 numpy 가 없고, 손으로 그리면 논문 조판에
바로 들어가는 **벡터**를 얻는다. 색은 흑백 인쇄에서도 구분되도록 명도를 벌린다.
"""
from __future__ import annotations

FONT = "IBM Plex Sans KR, Malgun Gothic, sans-serif"
MONO = "IBM Plex Mono, Consolas, monospace"

#: 인쇄 안전 팔레트 — 명도 차이로 구분된다.
INK, MUTED, FAINT, RULE = "#1a1a18", "#57575a", "#8a8d93", "#d5d6d2"
NAVY, BLUE, GRAY, PALE = "#1c3a63", "#5b7fa8", "#9aa0a8", "#d3d6da"
WARM, LOSS = "#b07d2b", "#a04a2f"


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


class Fig:
    def __init__(self, w: int, h: int, pad=(52, 16, 40, 64)):
        """pad = (왼, 위, 아래, 오른) 여백."""
        self.w, self.h = w, h
        self.l, self.t, self.b, self.r = pad
        self.parts: list[str] = []

    # ---------------------------------------------------------- 좌표
    @property
    def x0(self): return self.l
    @property
    def x1(self): return self.w - self.r
    @property
    def y0(self): return self.t
    @property
    def y1(self): return self.h - self.b

    # ---------------------------------------------------------- 원시
    def raw(self, s): self.parts.append(s); return self

    def line(self, x1, y1, x2, y2, c=RULE, w=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return self.raw(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
                        f'y2="{y2:.1f}" stroke="{c}" stroke-width="{w}"{d}/>')

    def rect(self, x, y, w, h, fill, stroke=None, rx=0, op=1.0):
        s = f' stroke="{stroke}"' if stroke else ""
        o = f' opacity="{op}"' if op != 1.0 else ""
        return self.raw(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(0,w):.2f}" '
                        f'height="{max(0,h):.2f}" fill="{fill}"{s} rx="{rx}"{o}/>')

    def text(self, x, y, s, size=11, c=INK, anchor="start", weight=400,
             mono=False, dy=0):
        f = MONO if mono else FONT
        return self.raw(
            f'<text x="{x:.1f}" y="{y + dy:.1f}" font-family="{f}" '
            f'font-size="{size}" fill="{c}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{esc(s)}</text>')

    def path(self, d, stroke=None, fill="none", w=1.6, op=1.0):
        s = f' stroke="{stroke}" stroke-width="{w}"' if stroke else ""
        return self.raw(f'<path d="{d}" fill="{fill}"{s} opacity="{op}" '
                        f'stroke-linejoin="round" stroke-linecap="round"/>')

    def title(self, s, sub=None):
        self.text(0, 14, s, size=12.5, weight=600)
        if sub:
            self.text(0, 30, sub, size=10.5, c=MUTED)
        return self

    def render(self) -> str:
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} '
                f'{self.h}" width="100%" role="img">'
                + "".join(self.parts) + "</svg>")

    def save(self, path):
        import pathlib
        pathlib.Path(path).write_text(self.render(), encoding="utf-8")
        return path


def nice_ticks(lo, hi, n=5):
    """읽기 좋은 눈금 — 1·2·2.5·5·10 배수."""
    import math
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            step = m * mag
            break
    start = math.floor(lo / step) * step
    out = []
    v = start
    while v <= hi + step * .5:
        out.append(round(v, 10))
        v += step
    return out
