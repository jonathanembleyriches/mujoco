#!/usr/bin/env python3
"""ProtoSpec layering boundary gate (post-reorg).

Fast, build-free include scanner. Run from anywhere; exits non-zero on any
un-waivered boundary violation. Lives at <repo>/protospec/tests/ and doubles as a
pytest test via `test_boundaries()` at the bottom.

Because the C++ sources include library headers by *bare basename* (resolved by
`-I` include dirs, e.g. `#include "types.h"`, `#include "binding.h"`), the gate
resolves each studio/library include to the library area that owns that basename,
rather than matching path-qualified spellings. A basename index over
`protospec/lib/**` and `attic/**` drives the classification.

Three rules enforced:
  R1  studio/**  may include the library ONLY through the ALLOWED editor surface
      (generated/, sdk/protospec/, the <protospec/...> umbrella, io/mjcf.h, and the
      compile bridge public headers {binding,pose,report,compile}.h). Never core/,
      validate internals, io internals, python bindings, third_party, or the attic.
  R2  library (protospec/lib/**) and attic/** must NOT include studio/**.
  R3  Tier-0 (MuJoCo-free core: generated/core/io/validate/sdk/include) must NOT
      include the attic native compiler. The compile bridge -> attic native
      dispatch is the single sanctioned Tier-1 waiver (W2).
"""
import os
import re
import sys

REPO_ROOT = os.environ.get(
    "PS_REPO_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

LIB = "protospec/lib"
STUDIO = "studio"
ATTIC = "attic"

INC_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"')

# --- Editor's ALLOWED library surface, expressed over library areas ---------- #
# Wholesale-allowed areas: the object model, the SDK authoring surface, and the
# curated umbrella. (reflect.h / compile.h / validate.h also live under the
# umbrella include/protospec/, so bare spellings of them resolve as allowed.)
ALLOWED_AREAS = {"generated", "sdk", "include"}
# Area-restricted allowances (area -> permitted basenames).
ALLOWED_AREA_BASENAMES = {
    "io": {"mjcf.h"},
    # the compile bridge public surface (dir was `bridge/`, renamed to `compile/`)
    "compile": {"binding.h", "pose.h", "report.h", "bridge.h"},
}

# Explicit, dated waivers: (including_file_suffix, included_spelling, reason).
WAIVERS = [
    ("studio/editor/layers.cc", "tinyxml2.h",
     "W1 GRANDFATHERED 2026-07: editor reaches vendored tinyxml2 directly for raw "
     "MJCF text munging. Follow-up: route through io/ helper, then delete waiver."),
    ("protospec/lib/compile/compile.cc", "native.h",
     "W2 SANCTIONED Tier-1 seam: compile bridge dispatch to attic/compile (now "
     "#ifdef PROTOSPEC_NATIVE-gated, default off). Bridge is MuJoCo-gated (never in "
     "the portable Tier-0 build)."),
]

# generated/mjs_binding.{h,cc} is the one sanctioned exception to "generated/ is
# MuJoCo-free": it is the ProtoSpec->mjSpec field-mapping layer, compiled ONLY by
# the MuJoCo-linked compile target (lib/CMakeLists.txt lists generated sources
# explicitly and excludes it from the Tier-0 `protospec` library). Its MuJoCo
# dependency is a `<mujoco/mujoco.h>` angle include, invisible to this gate's
# quoted-include scanner; its only quoted includes are types.h (generated) and
# compile/mjs_convert.h, neither of which is an attic edge -- so no rule fires.
TIER0 = ("generated/", "core/", "io/", "validate/", "sdk/", "include/")
ATTIC_NATIVE_BASENAMES = {"native.h", "build.h", "context.h", "native_supported.h"}


def _headers(root):
    out = []
    for base, _, names in os.walk(os.path.join(REPO_ROOT, root)):
        # Skip build trees, VCS, and provenance snapshots (frozen upstream copies
        # under */snapshots/ are reference data, not buildable source). Also skip
        # the parked standalone host (attic/studio_host/): it is a self-contained
        # application whose sources legitimately include their own editor/ host/
        # platform/ relative paths -- it depends on the editor by design, which is
        # exactly what R2 forbids for the *reusable* library/native-compiler, not
        # for a parked host app. It is off the default build.
        if ("__pycache__" in base or f"{os.sep}build" in base
                or f"{os.sep}.git" in base or f"{os.sep}snapshots" in base
                or f"{os.sep}studio_host" in base):
            continue
        for n in names:
            if n.endswith((".cc", ".cpp", ".h", ".hpp")):
                out.append(os.path.relpath(os.path.join(base, n), REPO_ROOT))
    return out


def _lib_area(relpath):
    # relpath like protospec/lib/<area>/...  -> return <area>
    parts = relpath.split("/")
    return parts[2] if len(parts) > 2 else ""


# basename -> set(areas) index over the library, and the attic basename set.
_LIB_INDEX = {}
_ATTIC_BASENAMES = set()


def _build_index():
    for f in _headers(LIB):
        if f.endswith((".h", ".hpp")):
            _LIB_INDEX.setdefault(os.path.basename(f), set()).add(_lib_area(f))
    for f in _headers(ATTIC):
        if f.endswith((".h", ".hpp")):
            _ATTIC_BASENAMES.add(os.path.basename(f))


def _waived(relpath, inc):
    for suf, spell, _ in WAIVERS:
        if relpath.replace(os.sep, "/").endswith(suf) and (
            inc == spell or inc.endswith("/" + spell)
        ):
            return True
    return False


def _includes(relpath):
    with open(os.path.join(REPO_ROOT, relpath), errors="replace") as fh:
        for line in fh:
            m = INC_RE.match(line)
            if m:
                yield m.group(1)


def _editor_allowed(inc):
    """True if a studio include targets the sanctioned library surface (or is not
    a library include at all: studio-local / third-party-not-in-lib / std)."""
    if inc.startswith("protospec/"):        # umbrella <protospec/*> + sdk/protospec/*
        return True
    b = os.path.basename(inc)
    areas = _LIB_INDEX.get(b)
    if not areas:                            # not a library header -> not our concern
        return True
    if areas & ALLOWED_AREAS:
        return True
    for area in areas:
        if b in ALLOWED_AREA_BASENAMES.get(area, ()):
            return True
    return False


def scan():
    if not _LIB_INDEX:
        _build_index()
    viol = []

    # R1: studio -> library surface.
    for f in _headers(STUDIO):
        for inc in _includes(f):
            if _editor_allowed(inc):
                continue
            if _waived(f, inc):
                continue
            viol.append(("R1 studio->non-public-lib", f, inc))

    # R2: library / attic -> studio.
    for f in _headers(LIB) + _headers(ATTIC):
        for inc in _includes(f):
            if any(k in inc for k in ("studio/", "editor/", "host/", "platform/")):
                if not _waived(f, inc):
                    viol.append(("R2 lib/attic->studio", f, inc))

    # R3: Tier-0 (MuJoCo-free core) must not include the attic native compiler.
    for f in _headers(LIB):
        area = f[len(LIB) + 1:]
        if not area.startswith(TIER0):
            continue
        for inc in _includes(f):
            if os.path.basename(inc) in ATTIC_NATIVE_BASENAMES and not _waived(f, inc):
                viol.append(("R3 tier0->attic", f, inc))

    # R3b: the compile-bridge dispatch must stay the ONLY library -> attic edge.
    for f in _headers(LIB):
        for inc in _includes(f):
            if os.path.basename(inc) in _ATTIC_BASENAMES and not _waived(f, inc):
                viol.append(("R3b lib->attic (unwaivered)", f, inc))
    return viol


def main():
    v = scan()
    if not v:
        print("boundaries OK: no violations; %d waiver(s) active" % len(WAIVERS))
        return 0
    print("BOUNDARY VIOLATIONS:")
    for rule, f, inc in v:
        print(f"  [{rule}] {f}  includes  {inc}")
    return 1


def test_boundaries():   # pytest entry point
    assert scan() == [], "layering boundary violated; run test_boundaries.py"


if __name__ == "__main__":
    sys.exit(main())
