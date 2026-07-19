"""Three-way native-compiler differential + self-arming ratchet (impl-plan T0.2).

For every corpus model M the driver runs ``ps_native_diff M`` which compiles it
two ways in-process -- leg B (ProtoSpec -> XML -> mj_loadXML, the permanent
oracle that test_differential.py independently keeps == to MuJoCo's own reader)
and leg C (the native compiler) -- and emits a JSON verdict
``{native_supported, path_taken, identical, first_divergence, fallback_reasons}``.

Ratchet (``tests/native_ratchet.json``, checked in):
  * every file listed in ``native_files`` must still compile natively AND be
    bit-identical to leg B -- a regression to fallback or a divergence fails;
  * any file the native path claims (native_supported) that is NOT identical
    fails, listed or not;
  * the total count of natively-compiled files may not drop below
    ``native_count``.

Growing ``cpp/compile/native_supported.h`` (NC1+) is what moves the ratchet up.
Today the native path claims nothing, so the whole corpus runs green as
all-fallback with the ratchet at 0.

Authored (like test_differential.py) to drive ps_native_diff only through its
stable process contract, never its internals; it reuses that harness's corpus
discovery, supported-tag prefilter, and size cap.
"""
from __future__ import annotations

import json
import os
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

MAX_XML_BYTES = 5 * 1024 * 1024
_BODY_CONTEXT = {"worldbody", "frame", "replicate"}
_IGNORE_TAGS = {"mujoco"}


# --------------------------------------------------------------------------- #
# Discovery (mirrors test_differential.py)                                    #
# --------------------------------------------------------------------------- #
def _corpus_root() -> Path | None:
    env = os.environ.get("PROTOSPEC_CORPUS")
    candidates = [Path(env)] if env else []
    candidates.append(
        Path(
            r"C:\Users\jonat\Documents\Unreal Projects\url_proj\Plugins"
            r"\UnrealRoboticsLab\third_party\MuJoCo\src"
        )
    )
    for c in candidates:
        if c.is_dir():
            return c
    return None


def _find_binary(name: str) -> Path | None:
    matches = sorted(
        (ROOT / "cpp").rglob(name),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _load_supported() -> set[str] | None:
    path = ROOT / "cpp" / "io" / "supported.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return {t.lower() for t in data.get("elements", [])}


def _load_ratchet() -> dict:
    path = ROOT / "tests" / "native_ratchet.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("native_count", 0)
    data.setdefault("native_files", [])
    return data


CORPUS_ROOT = _corpus_root()
PS_NATIVE_DIFF = _find_binary("ps_native_diff.exe")
SUPPORTED = _load_supported()
RATCHET = _load_ratchet()


def _mujoco_available() -> bool:
    if PS_NATIVE_DIFF is None:
        return False
    return (PS_NATIVE_DIFF.parent / "mujoco.dll").is_file()


_READY = (
    CORPUS_ROOT is not None
    and _mujoco_available()
    and SUPPORTED is not None
)


def _skip_reason() -> str:
    if CORPUS_ROOT is None:
        return "MuJoCo corpus not found (set PROTOSPEC_CORPUS)"
    if not _mujoco_available():
        return "ps_native_diff.exe / mujoco.dll not built"
    if SUPPORTED is None:
        return "cpp/io/supported.json missing (pathfinder pending)"
    return ""


def _corpus_files() -> list[Path]:
    if CORPUS_ROOT is None:
        return []
    files = []
    for p in CORPUS_ROOT.rglob("*.xml"):
        parts = {s.lower() for s in p.parts}
        if "build" in parts:
            continue
        files.append(p)
    return sorted(files)


def _rel_id(p: Path) -> str:
    try:
        return p.relative_to(CORPUS_ROOT).as_posix()
    except ValueError:
        return p.name


_CORPUS = _corpus_files()
_CORPUS_IDS = [_rel_id(p) for p in _CORPUS]


def _scan_tags(path: Path) -> set[str] | None:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return None
    tags = set()
    for el in root.iter():
        tag = el.tag
        if not isinstance(tag, str):
            continue
        tag = tag.lower()
        if tag in _IGNORE_TAGS:
            continue
        tags.add("body" if tag in _BODY_CONTEXT else tag)
    return tags


# --------------------------------------------------------------------------- #
# Session accounting                                                          #
# --------------------------------------------------------------------------- #
_STATS: Counter = Counter()
_NATIVE_SEEN: set[str] = set()


@pytest.fixture(scope="session", autouse=True)
def _summary():
    yield
    total = sum(_STATS.values())
    if not total:
        return
    print("\n\n=== native differential summary ===")
    for key in ("fallback", "native-identical", "native-diverged",
                "skip-unsupported", "skip-size", "skip-unloadable", "error"):
        if _STATS[key]:
            print(f"  {key:20s} {_STATS[key]}")
    print(f"  {'TOTAL':20s} {total}")
    print(f"  native files seen: {len(_NATIVE_SEEN)} (ratchet floor "
          f"{RATCHET['native_count']})")


# --------------------------------------------------------------------------- #
# Ratchet self-tests (no corpus needed)                                       #
# --------------------------------------------------------------------------- #
def test_ratchet_file_wellformed():
    assert isinstance(RATCHET["native_count"], int)
    assert isinstance(RATCHET["native_files"], list)
    assert RATCHET["native_count"] == len(RATCHET["native_files"]) or (
        RATCHET["native_count"] <= len(RATCHET["native_files"])
    )


# --------------------------------------------------------------------------- #
# The per-model differential + ratchet                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _READY, reason=_skip_reason())
@pytest.mark.parametrize("model", _CORPUS, ids=_CORPUS_IDS or ["<none>"])
def test_native_ratchet(model: Path):
    rel = _rel_id(model)
    listed = rel in RATCHET["native_files"]

    tags = _scan_tags(model)
    if tags is None or (tags - SUPPORTED):
        if listed:
            pytest.fail(f"ratchet lists {rel} but it is no longer IO-supported")
        _STATS["skip-unsupported"] += 1
        pytest.skip("unsupported/unparseable tags")

    if model.stat().st_size > MAX_XML_BYTES:
        if listed:
            pytest.fail(f"ratchet lists {rel} but it exceeds the size cap")
        _STATS["skip-size"] += 1
        pytest.skip(f"XML exceeds {MAX_XML_BYTES} bytes")

    r = subprocess.run(
        [str(PS_NATIVE_DIFF), str(model), "--base-dir", str(model.parent)],
        capture_output=True,
        text=True,
    )
    if r.returncode == 3:
        if listed:
            pytest.fail(f"ratchet lists {rel} but ps_native_diff skipped it")
        _STATS["skip-unsupported"] += 1
        pytest.skip("ps_native_diff: unsupported elements (exit 3)")

    try:
        verdict = json.loads(r.stdout)
    except json.JSONDecodeError:
        _STATS["error"] += 1
        pytest.fail(f"ps_native_diff produced no JSON (exit {r.returncode})\n"
                    f"{r.stdout}\n{r.stderr}")

    # A leg-B (XML oracle) failure here is not a native-compiler bug -- some
    # corpus fixtures are not standalone-loadable; test_differential.py arbitrates
    # leg A==B separately. Treat it as a skip unless the file is on the ratchet.
    if not verdict.get("xml_ok", False):
        if listed:
            pytest.fail(f"ratchet lists {rel} but leg B (XML) failed:\n{r.stderr}")
        _STATS["skip-unloadable"] += 1
        pytest.skip("leg B (XML path) did not compile this fixture")

    native = bool(verdict.get("native_supported"))
    identical = bool(verdict.get("identical"))

    # A claimed-native model must be bit-identical to the XML oracle.
    if native and not identical:
        _STATS["native-diverged"] += 1
        pytest.fail(f"native leg diverged from XML leg on {rel}: "
                    f"{verdict.get('first_divergence')}")

    # Ratchet: a previously-native file may not regress.
    if listed:
        assert native, f"ratchet regression: {rel} fell back to XML"
        assert identical, f"ratchet regression: {rel} diverged"

    if native:
        _NATIVE_SEEN.add(rel)
        _STATS["native-identical"] += 1
    else:
        _STATS["fallback"] += 1


@pytest.mark.skipif(not _READY, reason=_skip_reason())
def test_zzz_ratchet_floor():
    # Runs after the parametrized sweep (name sorts last): the corpus produced
    # at least as many native files as the ratchet floor, and every listed file
    # was seen native.
    assert len(_NATIVE_SEEN) >= RATCHET["native_count"], (
        f"native count {len(_NATIVE_SEEN)} dropped below ratchet floor "
        f"{RATCHET['native_count']}")
    missing = set(RATCHET["native_files"]) - _NATIVE_SEEN
    assert not missing, f"ratchet files never seen native: {sorted(missing)}"
