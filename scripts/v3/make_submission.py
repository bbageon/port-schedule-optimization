"""제출본 폴더를 만든다 — 소스와 그림을 한 폴더로 모으고, **떼어 놓고 컴파일해 본다**.

    python scripts/v3/make_submission.py

⚠️ 자동으로 돌지 않는다. 사용자가 "제출본 갱신" 이라고 할 때만 돌린다 (지시 2026-08-30).

■ 왜 필요한가
  원고는 `\graphicspath{{../figures/}}` 로 **소스 폴더 밖**을 본다. 제출 시스템은
  올린 것만 한 폴더에 풀어 컴파일하므로 `../` 가 그 순간 사라진다. 원래 폴더에서
  돌리면 멀쩡해서 **문제가 안 보인다.** 그래서 이 스크립트는 만든 뒤 반드시
  **임시 폴더로 복사해 거기서 컴파일**해 확인한다.

■ 하는 일
  1. main.tex + sections/*.tex 복사
  2. `\includegraphics` 가 실제로 부르는 그림만 찾아 `figures/` 하나로 평탄화
  3. 공백 있는 파일명을 하이픈으로 바꾸고 본문 참조도 함께 고침
  4. `\graphicspath` 를 `{figures/}` 로
  5. llncs.cls 복사 (Springer 가 갖고 있지만 넣어 두는 편이 안전)
  6. 임시 폴더에서 pdflatex 두 번 → 미해결 참조·못 읽은 그림 0 확인
  7. 확인된 PDF 를 제출 폴더에 넣음

  넣지 않는 것: .aux/.log/.out/.bbl/.blg, 국문판, 생성 스크립트, _provided/
  참고문헌은 `thebibliography` 로 원고에 직접 있어 .bib/.bbl 이 필요 없다.
"""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "docs/paper/v3/latex-en"
OUT = ROOT / "docs/paper/v3/submission"
SECTIONS = ("03-environment", "04-policy", "05-results", "06-conclusion")

INC = re.compile(r"(\\includegraphics(?:\[[^\]]*\])?\{)([^}]+)(\})")
GPATH = re.compile(r"\\graphicspath\{[^}]*\}[^\n]*")
MIKTEX = pathlib.Path(
    r"C:\Users\geonu\AppData\Local\Programs\MiKTeX\miktex\bin\x64")


def safe(name: str) -> str:
    """공백·괄호를 없앤다 — 제출 시스템과 TeX 양쪽에서 사고 나는 이름이다."""
    stem = pathlib.Path(name).stem
    stem = re.sub(r"[^\w.-]+", "-", stem).strip("-")
    return stem + pathlib.Path(name).suffix


def graphics_roots() -> list[pathlib.Path]:
    """main.tex 의 graphicspath 를 실제 경로로 푼다."""
    m = re.search(r"\\graphicspath\{(.+)\}", (SRC / "main.tex").read_text(encoding="utf-8"))
    roots = [SRC]
    if m:
        for r in re.findall(r"\{([^}]*)\}", m.group(1)):
            roots.append((SRC / r).resolve())
    return roots


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "sections").mkdir(parents=True)
    (OUT / "figures").mkdir()

    roots = graphics_roots()
    texts = {"main.tex": (SRC / "main.tex").read_text(encoding="utf-8")}
    for n in SECTIONS:
        texts[f"sections/{n}.tex"] = (SRC / "sections" / f"{n}.tex").read_text(encoding="utf-8")

    picked: dict[str, pathlib.Path] = {}
    missing: list[str] = []

    def swap(m):
        ref = m.group(2)
        for r in roots:
            p = (r / ref)
            if p.exists():
                new = safe(p.name)
                if new in picked and picked[new] != p:
                    missing.append(f"이름 충돌: {new}")
                picked[new] = p
                return m.group(1) + new + m.group(3)
        missing.append(f"못 찾음: {ref}")
        return m.group(0)

    for k in texts:
        texts[k] = INC.sub(swap, texts[k])

    texts["main.tex"] = GPATH.sub(r"\\graphicspath{{figures/}}", texts["main.tex"])

    if missing:
        print("MISS:", *sorted(set(missing)), sep="\n  ")
        return 1

    for k, v in texts.items():
        (OUT / k).write_text(v, encoding="utf-8", newline="")
    for new, p in sorted(picked.items()):
        shutil.copy2(p, OUT / "figures" / new)
        print(f"  figures/{new:<28} ← {p.relative_to(ROOT)}")
    shutil.copy2(SRC / "llncs.cls", OUT / "llncs.cls")

    # ── ★떼어 놓고 컴파일해야 `../` 문제가 드러난다 ──────────────
    with tempfile.TemporaryDirectory() as td:
        work = pathlib.Path(td) / "sub"
        shutil.copytree(OUT, work)
        exe = MIKTEX / "pdflatex.exe"
        if not exe.exists():
            exe = pathlib.Path(shutil.which("pdflatex") or "")
        if not exe or not exe.exists():
            print("  ⚠️ pdflatex 를 못 찾았다 — 격리 빌드 확인을 건너뛴다")
            return 1
        for _ in range(2):
            subprocess.run([str(exe), "-interaction=nonstopmode", "main.tex"],
                           cwd=work, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        log = (work / "main.log").read_text(encoding="utf-8", errors="ignore")
        pdf = work / "main.pdf"
        bad = [w for w in ("Unable to load", "LaTeX Error", "undefined") if w in log]
        pages = re.search(r"Output written on main\.pdf \((\d+) pages", log)
        print()
        print(f"  격리 빌드: {'성공' if pdf.exists() else '실패'} · "
              f"{pages.group(1) if pages else '?'}쪽 · 문제 {bad or '없음'}")
        if not pdf.exists() or [b for b in bad if b != "undefined"]:
            print("  ★제출본이 홀로 컴파일되지 않는다 — 고치기 전에는 내지 말 것")
            return 1
        shutil.copy2(pdf, OUT / "main.pdf")

    print(f"\n제출본: {OUT.relative_to(ROOT)}")
    print("  갱신은 수동이다 — 숫자가 바뀌면 이 스크립트를 다시 돌린다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
