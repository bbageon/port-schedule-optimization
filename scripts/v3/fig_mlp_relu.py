"""제공받은 MLP 초안을 **구현과 맞게** 고쳐 논문용 그림을 만든다.

    python scripts/v3/fig_mlp_relu.py

입력  docs/paper/v3/figures/final_figure/MLP-draft.png   (사용자 제공 · 손대지 않는다)
출력  docs/paper/v3/figures/final_figure/MLP-relu.png    (논문에 실리는 것)

■ 왜 고쳐야 하나
  초안은 활성함수를 **GELU** 로 적었는데 구현은 ReLU 다:

      src/yard_rl/v3/actors/nets.py
          nn.Linear(in_dim, hid), nn.ReLU(),
          nn.Linear(hid, hid),    nn.ReLU(),
          nn.Linear(hid, 1)

  그림과 코드가 어긋난 채 실리면 재현하려는 독자가 먼저 걸린다.

■ 왜 LaTeX 오버레이가 아니라 그림을 고치나
  앞서 원고에서 `\\begin{picture}` + `\\put` 으로 "ReLU" 상자를 덧그렸다. 세 가지가
  나빴다: ① 회색 상자 안에 **밝은 사각형 자국**이 보인다 ② 덧글씨가 원본의 남색이
  아니라 검정이라 `Layer 1/2/3` 과 색이 어긋난다 ③ 좌표를 픽셀로 박아 둬서 그림을
  다시 내보내면 **아무도 모르게 어긋난다.**
  그림 파일이 스스로 옳은 말을 하게 두면 셋 다 사라지고, 슬라이드 등 다른 곳에
  같은 파일을 써도 안전하다.

■ 수식은 그림에서 떼어 LaTeX 로 조판한다
  초안 아래쪽 수식줄에도 GELU 가 두 번 나온다. 래스터에서 글자를 갈아끼우면
  글자 폭이 달라져 식 전체가 밀린다. 그래서 **수식 띠는 잘라내고** 원고에서 진짜
  수식으로 조판한다 (04-policy.tex). 벡터라 확대해도 깨지지 않는다.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIG = ROOT / "docs/paper/v3/figures/final_figure"
SRC = FIG / "MLP-draft.png"
DST = FIG / "MLP-relu.png"

NAVY = (0, 31, 82)         # 초안의 글자색 (실측 최빈값)
OLD, NEW = "GELU", "ReLU"

#: 후보 글꼴 — 초안의 굵은 산세리프에 가장 가까운 것부터.
FONTS = ("arialbd.ttf", "segoeuib.ttf", "calibrib.ttf", "DejaVuSans-Bold.ttf")


def _load_font(px: int) -> ImageFont.FreeTypeFont:
    for name in FONTS:
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    sys.exit("굵은 산세리프 글꼴을 못 찾았다: " + ", ".join(FONTS))


def _blank_band(ink_rows: np.ndarray, min_h: int = 8):
    """잉크가 하나도 없는 가로 띠를 모두 돌려준다 — 도식과 수식의 경계를 찾는다."""
    out, start = [], None
    for y, n in enumerate(ink_rows):
        if n == 0:
            start = y if start is None else start
        else:
            if start is not None and y - start >= min_h:
                out.append((start, y))
            start = None
    return out


def _fit_cap_height(text: str, target_px: int) -> ImageFont.FreeTypeFont:
    """대문자 높이가 원본과 같아지도록 글꼴 크기를 맞춘다."""
    size = target_px
    for _ in range(40):
        font = _load_font(size)
        top, bottom = font.getbbox("H")[1], font.getbbox("H")[3]
        cap = bottom - top
        if cap == target_px:
            return font
        size += 1 if cap < target_px else -1
    return _load_font(size)


def main() -> None:
    im = Image.open(SRC).convert("RGB")
    a = np.array(im)
    h, w, _ = a.shape

    # ── ① 수식 띠를 잘라낸다 (도식 아래 가장 큰 빈 띠에서 자른다)
    ink = (a.astype(int).sum(2) < 720).sum(1)
    bands = [b for b in _blank_band(ink) if b[0] > h * 0.5]
    if not bands:
        sys.exit("도식과 수식 사이의 빈 띠를 못 찾았다 - 초안이 바뀌었는지 보라")
    # 도식 안에도 작은 빈 띠가 있다(상자 아래 여백). **가장 넓은** 띠가 경계다.
    lo, hi = max(bands, key=lambda t: t[1] - t[0])
    cut = (lo + hi) // 2
    im = im.crop((0, 0, w, cut))
    a = np.array(im)

    # ── ② GELU 두 곳을 찾아 ReLU 로 갈아끼운다
    r, g, b = a[:, :, 0].astype(int), a[:, :, 1].astype(int), a[:, :, 2].astype(int)
    navy = (b > r + 40) & (b > 90) & (b < 200) & (r < 100)
    grey = (abs(r - g) < 5) & (abs(g - b) < 5) & (r > 225) & (r < 250)

    draw = ImageDraw.Draw(im)
    done = 0
    xs_all = np.where(navy.any(0))[0]
    # 남색 덩어리를 가로로 끊어 후보를 만든다
    groups, start = [], xs_all[0]
    for i in range(1, len(xs_all)):
        if xs_all[i] - xs_all[i - 1] > 12:
            groups.append((start, xs_all[i - 1]))
            start = xs_all[i]
    groups.append((start, xs_all[-1]))

    for x0, x1 in groups:
        col = navy[:, x0:x1 + 1]
        ys = np.where(col.any(1))[0]
        if len(ys) == 0:
            continue
        y0, y1 = ys.min(), ys.max()
        wpx, hpx = x1 - x0 + 1, y1 - y0 + 1
        # GELU 는 회색 상자 안에 있는 **대문자 한 줄**이다 (가로 60~90 · 세로 20~30)
        if not (55 <= wpx <= 95 and 18 <= hpx <= 32):
            continue
        pad = 6
        box = grey[max(0, y0 - 40):y1 + 40, max(0, x0 - 40):x1 + 40]
        if box.mean() < 0.3:                      # 회색 상자 안이 아니면 건너뛴다
            continue
        fill = tuple(int(v) for v in np.median(
            a[y0 - 30:y0 - 12, x0:x1 + 1].reshape(-1, 3), axis=0))
        draw.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], fill=fill)
        font = _fit_cap_height(NEW, hpx)
        bb = draw.textbbox((0, 0), NEW, font=font)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        draw.text((cx - (bb[2] - bb[0]) / 2 - bb[0],
                   cy - (bb[3] - bb[1]) / 2 - bb[1]), NEW, font=font, fill=NAVY)
        done += 1
        print(f"  {OLD} -> {NEW}  x {x0}..{x1} · y {y0}..{y1} · 바탕 {fill}")

    if done != 2:
        sys.exit(f"GELU 를 2곳 찾아야 하는데 {done}곳을 고쳤다 — 초안이 바뀌었는지 보라")

    DST.parent.mkdir(parents=True, exist_ok=True)
    im.save(DST)
    print(f"저장 {DST.relative_to(ROOT)}  {im.size[0]}x{im.size[1]}")
    print("formula band cropped; the manuscript typesets it as real math")


if __name__ == "__main__":
    main()
