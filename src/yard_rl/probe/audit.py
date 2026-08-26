"""v3 에 **빈 곳**이 있는가 — 죽은 배선을 기계로 찾는다.

■ 왜 필요한가
  2026-08-26 에 `Offer.slot_load` 가 **한 번도 채워지지 않는 것**을 발견했다.
  필드도 있고 특징 칸도 있고 주석도 있었는데 **아무도 안 채웠다.** 눈으로는
  못 찾는다 — 코드가 문법적으로 멀쩡하기 때문이다.

  같은 종류의 구멍을 **기계로** 훑는다.

■ 두 갈래로 본다

  ① 정적 — 소스만 읽어서
     · 받기만 하고 **안 쓰는 인자** (`slot_load` 가 정확히 이 경우였다)
     · **빈 컨테이너**로 초기화되고 아무도 안 채우는 상수
     · `TODO`·`FIXME`·`미구현` 표시

  ② 동적 — 실제로 굴려서
     · 특징 열 중 **끝까지 상수인 칸** (죽은 특징)
     · 한쪽 값만 나오는 칸 (사실상 상수)

  ★②가 핵심이다. 정적으로 멀쩡해도 **실행하면 늘 같은 값**이면 그 칸은 없는 것과
  같다. `slot_load` 도 여기서 잡혔을 것이다.

■ 계약
  `rudder`·`slot_load` 와 같다 — **v3 를 한 줄도 안 고친다.** 읽고 굴리기만 한다.
"""
from __future__ import annotations

import ast
import pathlib

V3 = pathlib.Path(__file__).resolve().parents[1] / "v3"

#: 이 이름들은 안 써도 정상이다 (규약상 존재하는 자리).
IGNORED_ARGS = {"self", "cls", "kw", "kwargs", "args", "_"}
#: 인터페이스를 맞추려고 받는 인자 — 안 써도 설계다.
INTERFACE_FILES = {"classical.py"}
MARKERS = ("TODO", "FIXME", "XXX", "미구현", "아직 구현", "임시로")


def unused_arguments(root: pathlib.Path = V3) -> list:
    """받기만 하고 **본문에서 한 번도 안 쓰는** 인자를 찾는다.

    `slot_load` 가 정확히 이 모양이었다 — `seller_action_features` 가 받아 놓고
    반환 목록에 안 넣었다.
    """
    out = []
    for f in sorted(root.rglob("*.py")):
        if "world" in f.parts:
            continue                       # 사본은 원본 소관
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            a = fn.args
            names = [x.arg for x in a.args + a.posonlyargs + a.kwonlyargs]
            used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
            used |= {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
            for nm in names:
                if nm in IGNORED_ARGS or nm.startswith("_"):
                    continue
                if nm not in used:
                    out.append((f.relative_to(root).as_posix(), fn.lineno,
                                fn.name, nm))
    return out


def empty_containers(root: pathlib.Path = V3) -> list:
    """`= {}` · `= []` · `= ()` 로 두고 **아무도 안 채우는** 모듈 상수.

    `slot_capacity = {}` ([[YR-176]] 미구현)이 이 모양이다.
    """
    out = []
    for f in sorted(root.rglob("*.py")):
        if "world" in f.parts:
            continue
        src = f.read_text(encoding="utf-8")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            tgt = node.targets[0]
            if not isinstance(tgt, ast.Name):
                continue
            v = node.value
            empty = ((isinstance(v, ast.Dict) and not v.keys)
                     or (isinstance(v, (ast.List, ast.Tuple, ast.Set))
                         and not v.elts))
            if empty:
                out.append((f.relative_to(root).as_posix(), node.lineno, tgt.id))
    return out


def markers(root: pathlib.Path = V3) -> list:
    """`TODO` 류 표시 — 남긴 사람이 알고 있는 구멍."""
    out = []
    for f in sorted(root.rglob("*.py")):
        if "world" in f.parts:
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            for m in MARKERS:
                if m in line:
                    out.append((f.relative_to(root).as_posix(), i, line.strip()[:88]))
                    break
    return out


def dead_columns(rows, names) -> list:
    """★특징 행에서 **끝까지 상수인 칸** — 실행해 봐야 잡힌다.

    `rows` 는 특징 행 목록, `names` 는 칸 이름. 값의 종류가 1 가지면 죽은 칸이다.
    """
    if not rows:
        return []
    out = []
    for j, nm in enumerate(names):
        v = {round(float(r[j]), 8) for r in rows}
        if len(v) <= 1:
            out.append((j, nm, next(iter(v)) if v else None, "★상수"))
        elif len(v) == 2 and 0.0 in v:
            out.append((j, nm, sorted(v), "두 값뿐"))
    return out
