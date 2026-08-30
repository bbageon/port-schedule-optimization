"""부동체가 본문 참조에서 얼마나 떨어졌는지 잰다.

    python float_placement.py <pdf> <en|ko>

캡션 쪽(`Fig. 3:` / `그림 3:`)과 그 번호를 **처음 가리킨 본문 쪽**을 비교한다.
음수면 참조보다 앞에 나온 것이고(그러면 안 된다), 0이면 같은 쪽, 1~2 면 뒤쪽이다.
"""
import pathlib
import re
import sys

import pymupdf


def pages_of(doc, pattern):
    out = []
    for pg in doc:
        if re.search(pattern, pg.get_text()):
            out.append(pg.number + 1)
    return out


def scan(pdf, lang):
    doc = pymupdf.open(pdf)
    if lang == "en":
        kinds = (("Fig.", r"Fig\.\s*{n}:"), ("Table", r"Table\s*{n}:"))
        refpat = {"Fig.": r"Fig\.\s*{n}(?!:)", "Table": r"Table\s*{n}(?!:)"}
    else:
        kinds = (("그림", r"그림\s*{n}:"), ("표", r"표\s*{n}:"))
        refpat = {"그림": r"그림\s*~?{n}(?!:)", "표": r"표\s*~?{n}(?!:)"}
    worst = 0
    for name, cap in kinds:
        for n in range(1, 9):
            cpages = pages_of(doc, cap.format(n=n))
            if not cpages:
                continue
            rpages = pages_of(doc, refpat[name].format(n=n))
            first_ref = min(rpages) if rpages else None
            gap = (cpages[0] - first_ref) if first_ref else None
            flag = "" if gap is None or 0 <= gap <= 2 else "   <-- 확인"
            if gap is not None:
                worst = max(worst, abs(gap))
            print(f"   {name} {n}: 캡션 {cpages[0]}쪽 · 첫 참조 {first_ref}쪽 · 차이 {gap}{flag}")
    print(f"   최대 이동 {worst}쪽")


if __name__ == "__main__":
    scan(sys.argv[1], sys.argv[2])
