"""Differential compile harness -- the backbone test of ProtoSpec (plan 10.1).

For every corpus model M the pipeline runs::

    A = mj_loadXML(M)                    # MuJoCo's own reader
    B = mj_loadXML(ps_roundtrip(M))      # ProtoSpec reader -> writer -> MuJoCo

and field-diffs the two ``mjModel`` structs with ``mj_model_diff`` (sizes,
name tables, every mjxmacro array field, plus forward-kinematics invariants).
Any divergence is a bug in the reader or writer, and the test fails naming the
model. This file is deliberately authored by a different agent than the IO code
it exercises (plan Section 12): it drives the tools only through their stable
process contract, never their internals.

Two tools cooperate, located under ``cpp/**`` in any config directory:

* ``ps_roundtrip in.xml``  (owned by the IO pathfinder) -- prints the
  ProtoSpec-roundtripped MJCF to stdout. Exit 0 = ok, 3 = file uses elements
  outside the currently supported set (skip), 1 = real error (fail).
* ``mj_model_diff a.xml b.xml`` (this harness) -- exit 0 identical, 2 differ,
  1 load error.

``cpp/io/supported.json`` (owned by the pathfinder) lists the lowercase MJCF
tags fully supported today; the harness only runs the round trip on files whose
every tag is supported, so the differential scope grows automatically as the
reader does. When ``ps_roundtrip`` or ``supported.json`` is absent (the
pathfinder may land after this harness), the pipeline is skipped wholesale but
the ``mj_model_diff`` self-tests still run.

Asset-path strategy
-------------------
MJCF resolves ``meshdir`` / ``texturedir`` / ``assetdir`` relative to the
*directory of the XML file handed to* ``mj_loadXML``. For the roundtripped model
to resolve the same mesh/texture files as the original, it must load from the
original's directory. The harness therefore writes the round-trip output into
the original file's own directory under a unique dotfile name
(``._ps_rt_<pid>_<n>.xml``) and deletes it in a ``finally``. This is the only
strategy that keeps relative asset resolution byte-for-byte identical without
copying (potentially large) asset trees; a sibling temp dir would break every
relative ``meshdir``.

Runtime cap
-----------
``mj_model_diff`` always runs ``mj_forward`` on both models. To keep the suite
bounded, models whose source XML exceeds ``MAX_XML_BYTES`` are skipped for size
rather than silently truncated; the skipped-for-size list is printed in the
session summary.
"""

from __future__ import annotations

import os
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import json
import pytest

ROOT = Path(__file__).resolve().parent.parent

# XML larger than this (bytes) is skipped for size; see module docstring.
MAX_XML_BYTES = 5 * 1024 * 1024

# The document root tag and read-time constructs that are not schema elements
# themselves. worldbody/frame/replicate validate against the body row, so they
# require "body" to be supported rather than their own tag.
_BODY_CONTEXT = {"worldbody", "frame", "replicate"}
_IGNORE_TAGS = {"mujoco"}


# --------------------------------------------------------------------------- #
# Discovery                                                                    #
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
    """Locate a built exe under cpp/** in any config dir, newest first."""
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


CORPUS_ROOT = _corpus_root()
MJ_MODEL_DIFF = _find_binary("mj_model_diff.exe")
PS_ROUNDTRIP = _find_binary("ps_roundtrip.exe")
SUPPORTED = _load_supported()


def _mujoco_available() -> bool:
    if MJ_MODEL_DIFF is None:
        return False
    # The runtime DLL is copied next to the exe by the harness CMake build.
    return (MJ_MODEL_DIFF.parent / "mujoco.dll").is_file()


# --------------------------------------------------------------------------- #
# Session summary                                                             #
# --------------------------------------------------------------------------- #
_STATS: Counter = Counter()
_SIZE_SKIPS: list[str] = []


@pytest.fixture(scope="session", autouse=True)
def _summary():
    yield
    total = sum(_STATS.values())
    if not total:
        return
    print("\n\n=== differential harness summary ===")
    for key in (
        "identical",
        "differ",
        "skip-unsupported",
        "skip-roundtrip",
        "skip-size",
        "skip-unloadable-original",
        "load-error",
    ):
        if _STATS[key]:
            print(f"  {key:28s} {_STATS[key]}")
    print(f"  {'TOTAL':28s} {total}")
    if _SIZE_SKIPS:
        print("  skipped-for-size:")
        for name in _SIZE_SKIPS:
            print(f"    {name}")


# --------------------------------------------------------------------------- #
# Corpus enumeration                                                          #
# --------------------------------------------------------------------------- #
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
    """Return the set of element tags in the doc, or None if not parseable XML."""
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
# Self-tests (run today, no pathfinder required)                              #
# --------------------------------------------------------------------------- #
_SELF_MJCF = """<mujoco>
  <worldbody>
    <body name="b1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 0 1"/>
      <geom name="g1" type="box" pos="0.1 0.2 0.3" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.mark.skipif(
    not _mujoco_available(),
    reason="mj_model_diff.exe / mujoco.dll not built (cmake -S cpp/harness)",
)
def test_self_identical(tmp_path):
    a = tmp_path / "a.xml"
    a.write_text(_SELF_MJCF, encoding="utf-8")
    r = subprocess.run(
        [str(MJ_MODEL_DIFF), str(a), str(a)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"expected identical (0), got {r.returncode}\n{r.stdout}\n{r.stderr}"
    assert "IDENTICAL" in r.stdout


@pytest.mark.skipif(
    not _mujoco_available(),
    reason="mj_model_diff.exe / mujoco.dll not built (cmake -S cpp/harness)",
)
def test_self_geom_pos_diff(tmp_path):
    a = tmp_path / "a.xml"
    b = tmp_path / "b.xml"
    a.write_text(_SELF_MJCF, encoding="utf-8")
    b.write_text(
        _SELF_MJCF.replace('pos="0.1 0.2 0.3"', 'pos="0.1 0.2 0.9"'),
        encoding="utf-8",
    )
    r = subprocess.run(
        [str(MJ_MODEL_DIFF), str(a), str(b)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 2, f"expected differ (2), got {r.returncode}\n{r.stdout}\n{r.stderr}"
    assert "geom_pos" in r.stdout, f"diff report did not name geom_pos:\n{r.stdout}"


@pytest.mark.skipif(
    not _mujoco_available(),
    reason="mj_model_diff.exe / mujoco.dll not built (cmake -S cpp/harness)",
)
def test_self_load_error(tmp_path):
    a = tmp_path / "a.xml"
    a.write_text(_SELF_MJCF, encoding="utf-8")
    missing = tmp_path / "does_not_exist.xml"
    r = subprocess.run(
        [str(MJ_MODEL_DIFF), str(a), str(missing)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 1, f"expected load error (1), got {r.returncode}"
    assert "load error" in r.stderr


# --------------------------------------------------------------------------- #
# The differential pipeline                                                   #
# --------------------------------------------------------------------------- #
_PIPELINE_READY = (
    CORPUS_ROOT is not None
    and _mujoco_available()
    and PS_ROUNDTRIP is not None
    and SUPPORTED is not None
)


def _pipeline_skip_reason() -> str:
    if CORPUS_ROOT is None:
        return "MuJoCo corpus not found (set PROTOSPEC_CORPUS)"
    if not _mujoco_available():
        return "mj_model_diff.exe / mujoco.dll not built"
    if PS_ROUNDTRIP is None:
        return "ps_roundtrip.exe not built yet (pathfinder pending)"
    if SUPPORTED is None:
        return "cpp/io/supported.json missing (pathfinder pending)"
    return ""


@pytest.mark.skipif(not _PIPELINE_READY, reason=_pipeline_skip_reason())
@pytest.mark.parametrize("model", _CORPUS, ids=_CORPUS_IDS or ["<none>"])
def test_roundtrip_matches_mujoco(model: Path):
    tags = _scan_tags(model)
    if tags is None:
        _STATS["skip-unsupported"] += 1
        pytest.skip("not parseable XML")

    unsupported = tags - SUPPORTED
    if unsupported:
        _STATS["skip-unsupported"] += 1
        pytest.skip(f"unsupported tags: {sorted(unsupported)}")

    if model.stat().st_size > MAX_XML_BYTES:
        _STATS["skip-size"] += 1
        _SIZE_SKIPS.append(_rel_id(model))
        pytest.skip(f"XML exceeds {MAX_XML_BYTES} bytes")

    # Run the ProtoSpec round trip.
    rt = subprocess.run(
        [str(PS_ROUNDTRIP), str(model)],
        capture_output=True,
        text=True,
    )
    if rt.returncode == 3:
        _STATS["skip-unsupported"] += 1
        pytest.skip("ps_roundtrip: unsupported elements (exit 3)")
    if rt.returncode != 0:
        _STATS["skip-roundtrip"] += 1
        pytest.fail(
            f"ps_roundtrip failed (exit {rt.returncode}) on {model}\n{rt.stderr}"
        )

    # Write the round trip into the ORIGINAL's directory so relative assets
    # (mesh/texture dirs) resolve identically; delete it afterward.
    sibling = model.parent / f"._ps_rt_{os.getpid()}_{abs(hash(str(model))) % 100000}.xml"
    try:
        try:
            sibling.write_text(rt.stdout, encoding="utf-8")
        except OSError as e:
            _STATS["skip-roundtrip"] += 1
            pytest.skip(f"cannot write round trip beside original: {e}")

        diff = subprocess.run(
            [str(MJ_MODEL_DIFF), str(model), str(sibling)],
            capture_output=True,
            text=True,
        )
    finally:
        try:
            sibling.unlink()
        except OSError:
            pass

    if diff.returncode == 1:
        # Distinguish "the original itself is not a standalone-loadable model"
        # (a corpus fixture, not our bug) from "our round trip won't load".
        if "load error in a" in diff.stderr:
            _STATS["skip-unloadable-original"] += 1
            pytest.skip(f"original not standalone-loadable: {diff.stderr.strip()}")
        _STATS["load-error"] += 1
        pytest.fail(f"mj_model_diff load error on round trip:\n{diff.stderr}")

    if diff.returncode == 2:
        _STATS["differ"] += 1
        pytest.fail(f"round trip differs from original:\n{diff.stdout}")

    assert diff.returncode == 0, f"unexpected exit {diff.returncode}\n{diff.stdout}\n{diff.stderr}"
    _STATS["identical"] += 1
