"""Lifted-code registry tooling (CDR-3 / impl-plan Section 4).

Every algorithm we copy out of the vendored MuJoCo tree with provenance is
recorded in ``snapshots/lifted_code.json``. This tool maintains that registry
and drift-gates it against the vendored source, so upstream semantic drift is
caught at bump time instead of silently accumulating.

Subcommands::

    add --id ID --upstream REL --symbol S --local P [--notes N]
        Extract symbol S from the vendored tree (by signature + brace matching,
        or the whole file when S == "*"), write the snapshot, hash it, and
        append/update the entry.

    check [--mujoco-root R]
        Re-extract every registered symbol from the vendored tree and diff it
        against the stored snapshot + hash. Nonzero exit lists divergences.

    verify-headers
        Every file under cpp/compile/lifted/ carries a provenance header and
        appears in the registry, and every registry local_path exists.

    list
        Print the registry as a table.

Extraction anchors on the symbol signature + brace matching, never raw line
numbers (which shift every bump); ``line_span`` is a human reference only.
Hashes and snapshots are LF-normalized (the snapshots dir is ``-text`` in
.gitattributes) so Windows autocrlf does not produce false drift.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

# REPO is the repository root. This tool now lives at attic/tools/, and the
# lifted code it tracks is spread across the repo (attic/compile/*, the library
# protospec/lib/*, and studio/*), so every registry path is stored relative to the
# repo root. (attic/tools/lift_registry.py -> parents[2] == repo root.)
REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "attic" / "snapshots" / "lifted_code.json"
SNAPSHOT_DIR = REPO / "attic" / "snapshots" / "lifted_upstream"
LIFTED_DIR = REPO / "attic" / "compile" / "lifted"

DEFAULT_MUJOCO_ROOT = Path(
    r"C:\Users\jonat\Documents\Unreal Projects\url_proj\Plugins"
    r"\UnrealRoboticsLab\third_party\MuJoCo\src"
)


# --------------------------------------------------------------------------- #
# Vendored-tree discovery                                                     #
# --------------------------------------------------------------------------- #
def find_mujoco_root(override: str | None = None) -> Path | None:
    candidates = []
    if override:
        candidates.append(Path(override))
    for env in ("PROTOSPEC_MUJOCO_ROOT", "PROTOSPEC_CORPUS"):
        v = os.environ.get(env)
        if v:
            candidates.append(Path(v))
    candidates.append(DEFAULT_MUJOCO_ROOT)
    for c in candidates:
        if (c / "src").is_dir() and (c / "include").is_dir():
            return c
    return None


# --------------------------------------------------------------------------- #
# Extraction                                                                  #
# --------------------------------------------------------------------------- #
def lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _match_delims(text: str, start: int, open_ch: str, close_ch: str) -> int:
    """Index just past the delimiter opened at text[start] (== open_ch)."""
    depth = 0
    i = start
    while i < len(text):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("unbalanced delimiters")


def extract(root: Path, upstream_rel: str, symbol: str) -> tuple[str, str]:
    """Return (LF-normalized text, line_span) for a symbol or whole file.

    symbol == "*" returns the entire file. Otherwise the function DEFINITION of
    ``symbol`` is located (the first occurrence of ``symbol(`` whose matching
    ``)`` is followed by ``{``) and returned from the start of its signature
    line through the matching closing brace. Brace/paren matching does not skip
    string/comment contents -- adequate for the free functions we lift, which
    contain no braces inside string literals.
    """
    path = root / upstream_rel
    text = lf(path.read_text(encoding="utf-8"))
    if symbol == "*":
        n = text.count("\n") + 1
        return text, f"1-{n}"

    pattern = re.compile(r"(?<![\w:>.])" + re.escape(symbol) + r"\s*\(")
    for m in pattern.finditer(text):
        open_paren = text.index("(", m.end() - 1)
        close_paren = _match_delims(text, open_paren, "(", ")")
        k = close_paren + 1
        # Skip whitespace and trailing member-function qualifiers (e.g. a
        # `const` method like `mjCGeom::GetVolume() const {`) before the brace.
        while True:
            while k < len(text) and text[k] in " \t\r\n":
                k += 1
            q = re.match(r"(?:const|noexcept|override|final|volatile)\b", text[k:])
            if not q:
                break
            k += q.end()
        if k >= len(text) or text[k] != "{":
            continue  # a declaration or call, not a definition
        line_start = text.rfind("\n", 0, m.start()) + 1
        body_end = _match_delims(text, k, "{", "}") + 1
        block = text[line_start:body_end]
        l1 = text.count("\n", 0, line_start) + 1
        l2 = text.count("\n", 0, body_end) + 1
        return block, f"{l1}-{l2}"
    raise ValueError(f"definition of '{symbol}' not found in {upstream_rel}")


# --------------------------------------------------------------------------- #
# Registry IO                                                                 #
# --------------------------------------------------------------------------- #
def load_registry() -> dict:
    if REGISTRY.is_file():
        return json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {"pin": "3010000", "entries": []}


def save_registry(reg: dict) -> None:
    reg["entries"].sort(key=lambda e: e["id"])
    REGISTRY.write_bytes(
        (json.dumps(reg, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )


# --------------------------------------------------------------------------- #
# Subcommands                                                                 #
# --------------------------------------------------------------------------- #
def cmd_add(args) -> int:
    root = find_mujoco_root(args.mujoco_root)
    if root is None:
        print("error: vendored MuJoCo tree not found", file=sys.stderr)
        return 1
    text, line_span = extract(root, args.upstream, args.symbol)

    snap_name = args.snapshot or _default_snapshot_name(args.upstream, args.symbol)
    snap_path = SNAPSHOT_DIR / snap_name
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snap_path.write_bytes(text.encode("utf-8"))

    locals_ = [s for s in (args.local.split(",")) if s]
    reg = load_registry()
    entry = {
        "id": args.id,
        "upstream_rel_path": args.upstream,
        "symbol": args.symbol,
        "line_span": line_span,
        "content_hash": sha256(text),
        "snapshot": snap_path.relative_to(REPO).as_posix(),
        "local_path": locals_ if len(locals_) > 1 else locals_[0],
        "lifted": args.date,
        "notes": args.notes or "",
    }
    reg["entries"] = [e for e in reg["entries"] if e["id"] != args.id]
    reg["entries"].append(entry)
    save_registry(reg)
    print(f"added/updated '{args.id}' ({line_span}, {entry['content_hash'][:12]})")
    return 0


def _default_snapshot_name(upstream: str, symbol: str) -> str:
    if symbol == "*":
        return Path(upstream).name
    return f"{symbol.replace('::', '__')}.txt"


def cmd_check(args) -> int:
    root = find_mujoco_root(args.mujoco_root)
    if root is None:
        print("error: vendored MuJoCo tree not found (set PROTOSPEC_MUJOCO_ROOT "
              "or --mujoco-root)", file=sys.stderr)
        return 1
    reg = load_registry()
    diverged = []
    for e in reg["entries"]:
        try:
            text, _ = extract(root, e["upstream_rel_path"], e["symbol"])
        except (OSError, ValueError) as ex:
            diverged.append((e["id"], f"re-extraction failed: {ex}"))
            continue
        cur = sha256(text)
        if cur != e["content_hash"]:
            diverged.append(
                (e["id"], f"hash drift: registry {e['content_hash'][:12]} "
                          f"vs upstream {cur[:12]}"))
            continue
        snap = (REPO / e["snapshot"])
        if not snap.is_file():
            diverged.append((e["id"], f"snapshot missing: {e['snapshot']}"))
        elif lf(snap.read_text(encoding="utf-8")) != text:
            diverged.append((e["id"], "snapshot content != upstream extraction"))

    if diverged:
        print("LIFTED-CODE DRIFT:", file=sys.stderr)
        for eid, msg in diverged:
            print(f"  {eid}: {msg}", file=sys.stderr)
        print("Re-lift (or document intentional lag in notes) and refresh the "
              "snapshot with `lift_registry.py add`.", file=sys.stderr)
        return 1
    print(f"check ok: {len(reg['entries'])} entries match the vendored tree")
    return 0


PROVENANCE_MARKERS = ("Lifted from MuJoCo", "Upstream:", "Lifted mjModel",
                      "Lifted mjuu", "for provenance")


def _entry_locals(e) -> list[str]:
    lp = e["local_path"]
    return lp if isinstance(lp, list) else [lp]


def cmd_verify_headers(args) -> int:
    reg = load_registry()
    local_paths = {lp for e in reg["entries"] for lp in _entry_locals(e)}
    problems = []

    # every registered local_path exists and carries a provenance header.
    for lp in sorted(local_paths):
        p = REPO / lp
        if not p.is_file():
            problems.append(f"registry local_path missing on disk: {lp}")
            continue
        head = p.read_text(encoding="utf-8")[:1200]
        if not any(mark in head for mark in PROVENANCE_MARKERS):
            problems.append(f"no provenance header: {lp}")

    # every file under cpp/compile/lifted/ is registered.
    if LIFTED_DIR.is_dir():
        for p in sorted(LIFTED_DIR.rglob("*")):
            if p.suffix not in (".h", ".cc", ".cpp", ".hpp"):
                continue
            rel = p.relative_to(REPO).as_posix()
            if rel not in local_paths:
                problems.append(f"lifted file not in registry: {rel}")

    if problems:
        print("VERIFY-HEADERS failures:", file=sys.stderr)
        for msg in problems:
            print(f"  {msg}", file=sys.stderr)
        return 1
    print(f"verify-headers ok: {len(local_paths)} lifted files, all registered "
          "and attributed")
    return 0


def cmd_list(args) -> int:
    reg = load_registry()
    print(f"pin {reg.get('pin', '?')}  |  {len(reg['entries'])} entries")
    for e in reg["entries"]:
        print(f"  {e['id']:<16} {e['symbol']:<24} {e['upstream_rel_path']}")
        print(f"  {'':<16} -> {', '.join(_entry_locals(e))}  [{e['line_span']}] "
              f"{e['content_hash'][:12]}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add")
    a.add_argument("--id", required=True)
    a.add_argument("--upstream", required=True, help="path relative to the MuJoCo root")
    a.add_argument("--symbol", required=True, help='function name, or "*" for whole file')
    a.add_argument("--local", required=True, help="our copy, repo-relative")
    a.add_argument("--snapshot", default=None)
    a.add_argument("--notes", default="")
    a.add_argument("--date", default="2026-07-08")
    a.add_argument("--mujoco-root", default=None)
    a.set_defaults(func=cmd_add)

    c = sub.add_parser("check")
    c.add_argument("--mujoco-root", default=None)
    c.set_defaults(func=cmd_check)

    v = sub.add_parser("verify-headers")
    v.set_defaults(func=cmd_verify_headers)

    ls = sub.add_parser("list")
    ls.set_defaults(func=cmd_list)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
