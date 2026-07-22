"""Extract MuJoCo's C-side enum constants into a JSON snapshot (binding ground truth).

The ProtoSpec schema's enum/member binding annotations name MuJoCo C constants
(``mjGAIN_FIXED``, ``mjLIMITED_FALSE``) and their owning ``mjt*`` enum
(``ctype=mjtGain``). Those facts were validated only implicitly by the C compiler
consuming the generated tables; this snapshot makes them checkable at schema
parse/emit time (a ``(c=)``/``(mjs_c=)`` naming a constant that does not exist, or
that does not belong to its declared ``ctype``, fails loudly with a schema
file:line -- the maintenance contract that replaces the deleted Python dicts).

Every ``mjt*`` (and the two ``user_*`` composite/flexcomp) enum from the vendored
headers is parsed into ``{name, header, enumerators:[{name, value}]}``, with each
enumerator's integer value resolved (explicit ``= expr`` -- int literals, prior
enumerators, ``#define`` macros, and ``+ - * << >> |`` -- else the running ordinal).
Sentinels (``mjNGEOMTYPES`` and friends) are kept verbatim: they are legitimate
enumerators and some maps size against them.

Run from the ``protospec/`` directory:

    uv run python -m tools.bootstrap.extract_mjmodel_enums --mujoco-src ..
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Headers whose enums the schema's binding annotations may reference. mjtype.h
# carries the bulk (mjtGain/mjtGeom/...); mjspec.h the authoring-level enums
# (mjtLimited/mjtBuiltin/...); the two user_* headers the composite/flexcomp
# enums (mjtCompType/mjtFcompType/mjtFcompDof) that the reader's comp/fcomp maps
# name. Order fixes tie-breaking for a constant defined in more than one header
# (none today); later files do not override earlier ones -- a redefinition is an
# error surfaced by the sanity check.
_HEADERS = [
    ("include/mujoco/mjtype.h", "mjtype.h"),
    ("include/mujoco/mjspec.h", "mjspec.h"),
    ("src/user/user_composite.h", "user_composite.h"),
    ("src/user/user_flexcomp.h", "user_flexcomp.h"),
]

_ENUM_OPEN = re.compile(r"^typedef enum\s+(?P<tag>\w+)\s*\{(?:\s*//.*)?$")
_CLOSE = re.compile(r"^\}\s*(?P<name>\w+)\s*;")
_COMMENT_ONLY = re.compile(r"^//")
_ENUMERATOR = re.compile(
    r"^(?P<name>\w+)\s*(?:=\s*(?P<value>.+?))?\s*,?\s*(?://\s*(?P<comment>.*))?$"
)


class MjEnumParseError(Exception):
    """Raised when a vendored header's enum format drifts from what we can parse."""


@dataclass
class Enumerator:
    name: str
    value: int | None
    raw: str | None
    comment: str

    def to_dict(self) -> dict:
        return {"name": self.name, "value": self.value, "comment": self.comment}


@dataclass
class Enum:
    name: str
    tag: str
    header: str
    doc: str
    enumerators: list[Enumerator] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "tag": self.tag,
            "header": self.header,
            "doc": self.doc,
            "enumerators": [e.to_dict() for e in self.enumerators],
        }


def _eval_value(expr: str, syms: dict[str, int], line: int) -> int | None:
    """Resolve an enumerator value expression, or None if it is not integer-computable."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def ev(node: ast.AST):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            return syms.get(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            if v is None:
                return None
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            left, right = ev(node.left), ev(node.right)
            if left is None or right is None:
                return None
            op = node.op
            if isinstance(op, ast.Add):
                return left + right
            if isinstance(op, ast.Sub):
                return left - right
            if isinstance(op, ast.Mult):
                return left * right
            if isinstance(op, ast.LShift):
                return left << right
            if isinstance(op, ast.RShift):
                return left >> right
            if isinstance(op, ast.BitOr):
                return left | right
            if isinstance(op, ast.BitAnd):
                return left & right
        return None

    return ev(tree)


def parse_enums(text: str, header: str, macros: dict[str, int]) -> list[Enum]:
    lines = text.splitlines()
    enums: list[Enum] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        m = _ENUM_OPEN.match(stripped)
        if not m:
            i += 1
            continue
        tag = m.group("tag")
        # Only mjt*/user_* enum tags carry constants the schema binds against.
        if not (tag.startswith("mjt") or tag.startswith("_mjt")):
            i += 1
            continue
        doc = ""
        cm = re.search(r"//\s*(.*)$", lines[i])
        if cm:
            doc = cm.group(1).strip()
        enum = Enum(name="", tag=tag, header=header, doc=doc)
        syms = dict(macros)  # prior enumerators shadow macros within the enum
        counter = 0
        i += 1
        # Buffer physical lines into one logical enumerator: a value expression may
        # span lines (mjtState's bitmask unions end a line with a trailing '|'). A
        # logical enumerator is complete once its comment-stripped code carries the
        # separating ',' (these value exprs never contain a top-level comma).
        buf_code, buf_comment, start_line = "", "", 0

        def flush(code_text: str, at: int) -> None:
            nonlocal counter
            em = _ENUMERATOR.match(code_text)
            if not em:
                raise MjEnumParseError(
                    f"{header}:{at}: unparseable enumerator in {tag}: {code_text!r}"
                )
            raw = em.group("value")
            if raw is not None:
                raw = raw.strip()
                value = _eval_value(raw, syms, at)
            else:
                value = counter
            enum.enumerators.append(
                Enumerator(name=em.group("name"), value=value, raw=raw,
                           comment=buf_comment)
            )
            if value is not None:
                syms[em.group("name")] = value
                counter = value + 1
            else:
                counter += 1

        while i < len(lines):
            raw_line = lines[i]
            code, sep, comment = raw_line.partition("//")
            code = code.strip()
            comment = comment.strip() if sep else ""
            i += 1
            close = _CLOSE.match(code)
            if close:
                if buf_code:  # last enumerator had no trailing comma
                    flush(buf_code.rstrip(","), start_line)
                enum.name = close.group("name")
                enums.append(enum)
                break
            if not code:
                continue
            if not buf_code:
                start_line = i
            buf_code = (buf_code + " " + code).strip() if buf_code else code
            if comment:
                buf_comment = comment
            if "," not in buf_code:
                continue  # value continues on the next physical line
            flush(buf_code.rstrip(","), start_line)
            buf_code, buf_comment = "", ""
        else:
            raise MjEnumParseError(f"{header}: unterminated enum {tag}")
        i += 1
    return enums


def collect_macros(*headers: Path) -> dict[str, int]:
    macros: dict[str, int] = {}
    pattern = re.compile(r"^#define\s+(?P<name>\w+)\s+(?P<value>-?\d+)\b")
    for header in headers:
        if not header.exists():
            continue
        for raw in header.read_text(encoding="utf-8").splitlines():
            m = pattern.match(raw)
            if m:
                macros[m.group("name")] = int(m.group("value"))
    return macros


def read_mujoco_version(mujoco_src: Path) -> str:
    header = mujoco_src / "include" / "mujoco" / "mujoco.h"
    text = header.read_text(encoding="utf-8")
    m = re.search(r"#define\s+mjVERSION_HEADER\s+(\d+)", text)
    if m is None:
        raise MjEnumParseError(f"mjVERSION_HEADER not found in {header}")
    n = int(m.group(1))
    return f"{n // 1_000_000}.{(n // 1000) % 1000}.{n % 1000}"


def run_sanity_checks(enums: list[Enum]) -> None:
    by_const: dict[str, str] = {}
    for e in enums:
        if not e.enumerators:
            raise MjEnumParseError(f"enum {e.name!r} has no enumerators")
        for en in e.enumerators:
            if en.name in by_const and by_const[en.name] != e.name:
                raise MjEnumParseError(
                    f"constant {en.name!r} defined in both {by_const[en.name]!r} "
                    f"and {e.name!r}"
                )
            by_const[en.name] = e.name
    # Spot-check a representative constant from each source header resolves.
    for const, enum, val in (
        ("mjGAIN_FIXED", "mjtGain", 0),
        ("mjGEOM_PLANE", "mjtGeom", 0),
        ("mjLIMITED_FALSE", "mjtLimited", None),
        ("mjCOMPTYPE_PARTICLE", "mjtCompType", 0),
        ("mjFCOMPTYPE_GRID", "mjtFcompType", 0),
    ):
        if by_const.get(const) != enum:
            raise MjEnumParseError(
                f"expected {const!r} in enum {enum!r}, got {by_const.get(const)!r}"
            )


def build_snapshot(mujoco_src: Path) -> dict:
    include_dir = mujoco_src / "include" / "mujoco"
    macros = collect_macros(include_dir / "mjmodel.h", include_dir / "mjtype.h")
    enums: list[Enum] = []
    for rel, tag in _HEADERS:
        path = mujoco_src / rel
        enums.extend(parse_enums(path.read_text(encoding="utf-8"), tag, macros))
    run_sanity_checks(enums)
    enums.sort(key=lambda e: e.name)
    return {
        "source": {
            "headers": [rel for rel, _ in _HEADERS],
            "mujoco_version": read_mujoco_version(mujoco_src),
        },
        "enums": [e.to_dict() for e in enums],
    }


def write_snapshot(snapshot: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mujoco-src", type=Path, required=True,
                        help="Root of the vendored MuJoCo checkout (has src/ and include/).")
    parser.add_argument("--out", type=Path,
                        default=_repo_root() / "snapshots" / "mjmodel_enums.json",
                        help="Destination JSON path (default: snapshots/mjmodel_enums.json).")
    args = parser.parse_args(argv)

    try:
        snapshot = build_snapshot(args.mujoco_src)
    except MjEnumParseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    write_snapshot(snapshot, args.out)
    n_consts = sum(len(e["enumerators"]) for e in snapshot["enums"])
    print(
        f"wrote {args.out} ({len(snapshot['enums'])} enums, {n_consts} constants, "
        f"mujoco {snapshot['source']['mujoco_version']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
