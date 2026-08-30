"""역슬래시가 먹힌 흔적을 원고 전체에서 찾는다.

heredoc 를 지나며 `\r` `\t` 같은 escape 가 제어문자로 바뀌면, LaTeX 소스에는
`\ref{...}` 대신 개행 + `ef{...}` 가 남는다. 빌드는 통과하고(그냥 글자가 된다)
PDF 에만 `efsec:env` 처럼 찍히므로 눈으로 놓치기 쉽다.
"""
import pathlib
import re

CTRL = {"\t": "TAB", "\r": "CR", "\f": "FF", "\a": "BEL", "\v": "VT", "\b": "BS"}
FRAG = re.compile(
    r"(?<![\\A-Za-z])"
    r"(ef|extbf|extit|ext|mph|ite|abel|ubsection|ection|race|hspace|nskip"
    r"|ewcommand|extsuperscript|egin|nd|aption|ncludegraphics)\{")


def scan(root="docs/paper/v3"):
    bad = 0
    for f in sorted(pathlib.Path(root).rglob("*.tex")):
        s = f.read_text(encoding="utf-8")
        for ch, name in CTRL.items():
            if ch in s:
                print(f"  [제어문자] {f}: {name} x {s.count(ch)}")
                bad += 1
        for i, line in enumerate(s.splitlines(), 1):
            for m in FRAG.finditer(line):
                print(f"  [명령 조각] {f}:{i}: ...{line[max(0, m.start()-25):m.end()+18]}")
                bad += 1
    print("이상 없음" if not bad else f"의심 {bad}건")
    return bad


if __name__ == "__main__":
    raise SystemExit(1 if scan() else 0)
