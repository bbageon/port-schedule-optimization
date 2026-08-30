"""쪽마다 아래 여백이 얼마나 남는지 잰다 — 비부동체 블록의 낭비를 실측한다.

    python whitespace_scan.py <pdf> <라벨>

본문 상자(가장 왼쪽/오른쪽 글자와 가장 위/아래 내용)를 기준으로, 마지막 내용이
끝난 뒤 남은 세로 공간의 비율을 쪽마다 계산한다. 그림·표를 그리는 벡터 요소도
내용으로 친다(get_drawings + get_images + 글자).
"""
import pathlib
import sys

import pymupdf


def page_content_bottom(pg):
    bottoms = []
    for b in pg.get_text("dict")["blocks"]:
        bottoms.append(b["bbox"][3])
    for d in pg.get_drawings():
        bottoms.append(d["rect"].y1)
    for img in pg.get_images(full=True):
        for r in pg.get_image_rects(img[0]):
            bottoms.append(r.y1)
    return max(bottoms) if bottoms else None


def scan(pdf, label):
    doc = pymupdf.open(pdf)
    tops, bots = [], []
    for pg in doc:                       # 본문 상자의 위·아래를 추정
        for b in pg.get_text("dict")["blocks"]:
            tops.append(b["bbox"][1])
            bots.append(b["bbox"][3])
    top, bot = min(tops), max(bots)
    height = bot - top
    waste = []
    for pg in doc:
        cb = page_content_bottom(pg)
        if cb is None:
            continue
        frac = max(0.0, (bot - cb) / height)
        waste.append((pg.number + 1, frac))
    over = [(n, f) for n, f in waste if f > 0.16]
    total_excess = sum(f - 0.16 for _, f in over)
    print(f"== {label}: {doc.page_count}쪽 · 본문높이 {height:.0f}pt")
    print(f"   아래 여백 16% 초과 쪽 {len(over)}개 · 초과분 합계 {total_excess:.2f}쪽분")
    for n, f in sorted(over, key=lambda x: -x[1])[:6]:
        print(f"      {n}쪽 여백 {f:.0%}")
    return len(over), total_excess


if __name__ == "__main__":
    scan(sys.argv[1], sys.argv[2])
