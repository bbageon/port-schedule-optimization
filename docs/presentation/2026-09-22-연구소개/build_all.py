"""슬라이드 전체를 다시 그리고 한눈에 보는 모아보기 장을 만든다.

    python build_all.py            # 전부 다시 그리고 contact-sheet.png 까지
    python build_all.py --sheet    # 이미 있는 PNG 로 모아보기만

슬라이드 한 장은 `slide_NN_name.py` 하나에 대응한다. 여기서는 파일을 찾아
순서대로 돌릴 뿐이고, 내용은 각 파일이 갖는다.
"""
from __future__ import annotations

import argparse
import re
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLIDES = HERE / "slides"


def slide_modules() -> list[Path]:
    return sorted(HERE.glob("slide_[0-9][0-9]_*.py"))


def render_all() -> list[Path]:
    made = []
    for module in slide_modules():
        sys.path.insert(0, str(HERE))
        namespace = runpy.run_path(str(module))
        number = int(re.match(r"slide_(\d+)_", module.name).group(1))
        name = re.match(r"slide_\d+_(.+)\.py", module.name).group(1)
        made.append(namespace["save"](namespace["build"](), number, name))
        print(f"  {made[-1].name}")
    return made


def contact_sheet(paths: list[Path], columns: int = 4) -> Path:
    """열두 장을 격자로 붙여 한 장으로 — 발표 흐름을 통째로 볼 때 쓴다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    rows = -(-len(paths) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 4.2, rows * 2.55),
                             dpi=130, facecolor="#FFFFFF")
    for ax, path in zip(axes.flat, paths):
        ax.imshow(mpimg.imread(path))
        ax.set_title(path.stem, fontsize=8, color="#6B7280", pad=3)
        ax.axis("off")
    for ax in list(axes.flat)[len(paths):]:
        ax.axis("off")
    fig.tight_layout(pad=0.6)
    out = SLIDES / "contact-sheet.png"
    fig.savefig(out, dpi=130, facecolor="#FFFFFF")
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", action="store_true",
                        help="다시 그리지 않고 모아보기만 만든다")
    args = parser.parse_args()
    if args.sheet:
        paths = sorted(p for p in SLIDES.glob("[0-9][0-9]-*.png"))
    else:
        print("슬라이드를 그린다:")
        paths = render_all()
    if not paths:
        raise SystemExit("그려진 슬라이드가 없다")
    print(f"모아보기: {contact_sheet(paths)}")
    print(f"슬라이드 {len(paths)}장 · {SLIDES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
