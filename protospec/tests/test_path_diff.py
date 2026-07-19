"""Compile-path differential gate for the ProtoSpec bridge (ps_path_diff).

This test proves the *harness* that will judge the upcoming mjSpec compile path,
built and exercised NOW against the shipping XML path so that when
``CompilePath::MjsPath`` lands it is a one-flag flip:

    ps_path_diff --path-a XmlPath --path-b MjsPath <corpus>

Today the tool runs two gates over the checked-in fixture corpus
(``fixtures/pathdiff/``):

* **identity** -- ``XmlPath`` vs ``XmlPath``: the same deterministic compile path
  twice must yield a bit-identical ``mjModel`` (every MJMODEL_SIZES int, every
  name table in id order, every MJMODEL_POINTERS array, and the qpos0 forward-
  kinematics invariants). Any divergence is a determinism bug in the harness.
* **against-stock** -- ProtoSpec's ``XmlPath`` result vs stock ``mj_loadXML`` of
  the *original* file. This catches ProtoSpec reader/writer drift (the round trip
  changing the model), which a path-vs-path diff cannot see because both legs
  share the same front end. Auto-naming is disabled on this leg so the compare is
  against stock's pristine name tables.

Build strategy
--------------
The tool is a MuJoCo-linked C++ program. Rather than drive the CMake tree (which
targets the author's Windows MuJoCo), a session fixture compiles it directly with
``g++``: the three harness ``.cc`` files (ps_path_diff + model_diff_lib +
plugin_registry) against the ProtoSpec include dirs, linking the prebuilt
``libprotospec_core.a`` + ``libmujoco.so`` from the studio ``build_ps`` tree.

    Speed vs. staleness: linking the prebuilt ``libprotospec_core.a`` is far
    faster than recompiling every ProtoSpec ``.cc`` from source, but the archive
    is a *snapshot*. If the compile bridge / reader / writer sources change
    without rebuilding ``build_ps``, this test links stale object code. Rebuild
    the studio ``build_ps`` (or point ``PROTOSPEC_BUILD_PS_LIB`` at a fresh
    build) after touching ``protospec/lib`` compile/io sources.

The build_ps lib dir and MuJoCo include dir default to the studio checkout beside
this repo and are overridable:

    PROTOSPEC_BUILD_PS_LIB   dir holding libprotospec_core.a + libmujoco.so
    PROTOSPEC_MJ_INCLUDE     dir holding mujoco/mujoco.h + mujoco/mjxmacro.h

The whole module skips cleanly when those are absent (e.g. CI without the studio
build), so a plain ``uv run pytest`` stays green everywhere.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

PROTOSPEC = Path(__file__).resolve().parents[1]
LIB = PROTOSPEC / "lib"
HARNESS = LIB / "harness"
FIXTURES = PROTOSPEC / "tests" / "fixtures" / "pathdiff"

# Overridable locations of the prebuilt studio libraries and MuJoCo headers.
DEFAULT_BUILD_PS_LIB = Path(
    "/home/buzz/Documents/proto/mujoco-studio/build_ps/lib"
)
DEFAULT_MJ_INCLUDE = Path("/home/buzz/Documents/proto/mujoco-studio/include")


def _build_ps_lib() -> Path | None:
    p = Path(os.environ.get("PROTOSPEC_BUILD_PS_LIB", str(DEFAULT_BUILD_PS_LIB)))
    if (p / "libprotospec_core.a").is_file() and (
        (p / "libmujoco.so").is_file() or (p / "libmujoco.so.3.10.0").is_file()
    ):
        return p
    return None


def _mj_include() -> Path | None:
    p = Path(os.environ.get("PROTOSPEC_MJ_INCLUDE", str(DEFAULT_MJ_INCLUDE)))
    if (p / "mujoco" / "mujoco.h").is_file() and (
        p / "mujoco" / "mjxmacro.h"
    ).is_file():
        return p
    return None


BUILD_PS_LIB = _build_ps_lib()
MJ_INCLUDE = _mj_include()
CXX = os.environ.get("CXX", "g++")


def _skip_reason() -> str:
    if shutil.which(CXX) is None:
        return f"no C++ compiler ({CXX}) on PATH"
    if BUILD_PS_LIB is None:
        return (
            "studio build_ps libs absent (set PROTOSPEC_BUILD_PS_LIB to a dir "
            "with libprotospec_core.a + libmujoco.so)"
        )
    if MJ_INCLUDE is None:
        return (
            "MuJoCo headers absent (set PROTOSPEC_MJ_INCLUDE to a dir with "
            "mujoco/mujoco.h + mjxmacro.h)"
        )
    if not (HARNESS / "ps_path_diff.cc").is_file():
        return "ps_path_diff.cc source missing"
    return ""


SKIP_REASON = _skip_reason()
pytestmark = pytest.mark.skipif(bool(SKIP_REASON), reason=SKIP_REASON)


@pytest.fixture(scope="session")
def ps_path_diff(tmp_path_factory) -> Path:
    """Compile ps_path_diff once per session; return the exe path."""
    out_dir = tmp_path_factory.mktemp("ps_path_diff_build")
    exe = out_dir / "ps_path_diff"
    cmd = [
        CXX,
        "-std=c++20",
        "-O2",
        str(HARNESS / "ps_path_diff.cc"),
        str(HARNESS / "model_diff_lib.cc"),
        str(HARNESS / "plugin_registry.cc"),
        f"-I{LIB / 'include'}",
        f"-I{LIB / 'generated'}",
        f"-I{LIB / 'compile'}",
        f"-I{LIB / 'io'}",
        f"-I{HARNESS}",
        f"-I{MJ_INCLUDE}",
        str(BUILD_PS_LIB / "libprotospec_core.a"),
        str(BUILD_PS_LIB / "libmujoco.so"),
        f"-Wl,-rpath,{BUILD_PS_LIB}",
        "-o",
        str(exe),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(
            "ps_path_diff failed to build (build_ps libs may be stale vs the "
            f"current protospec sources):\n{proc.stderr[-4000:]}"
        )
    assert exe.is_file()
    return exe


def _fixture_files() -> list[Path]:
    return sorted(FIXTURES.glob("*.xml"))


FIXTURE_FILES = _fixture_files()
FIXTURE_IDS = [p.stem for p in FIXTURE_FILES]


def _run(exe: Path, *args: str) -> subprocess.CompletedProcess:
    # Plugin-bearing fixtures need the first-party engine plugins (built beside
    # libmujoco.so in the studio build_ps/lib). Point the harness's registry at
    # that dir so <extension> models load on both legs; harmless for the rest.
    env = dict(os.environ)
    if BUILD_PS_LIB is not None:
        env.setdefault("PROTOSPEC_PLUGIN_DIR", str(BUILD_PS_LIB))
    return subprocess.run(
        [str(exe), *args], capture_output=True, text=True, timeout=300, env=env
    )


# --------------------------------------------------------------------------- #
# Fixture-corpus gates                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not FIXTURE_FILES, reason="no pathdiff fixtures found")
@pytest.mark.parametrize("model", FIXTURE_FILES, ids=FIXTURE_IDS or ["<none>"])
def test_identity_fixture(ps_path_diff: Path, model: Path):
    """XmlPath vs XmlPath must be bit-identical for every fixture."""
    r = _run(ps_path_diff, "--path-a", "XmlPath", "--path-b", "XmlPath", str(model))
    assert r.returncode == 0, f"identity diff FAILED:\n{r.stdout}\n{r.stderr}"
    # Per-model verdict lines start with PASS/FAIL; the trailing summary line
    # ("N PASS, 0 FAIL") also contains "FAIL", so match on line prefixes.
    assert any(l.startswith("PASS") for l in r.stdout.splitlines()), r.stdout
    assert not any(l.startswith("FAIL") for l in r.stdout.splitlines()), r.stdout


@pytest.mark.skipif(not FIXTURE_FILES, reason="no pathdiff fixtures found")
@pytest.mark.parametrize("model", FIXTURE_FILES, ids=FIXTURE_IDS or ["<none>"])
def test_mjs_parity_fixture(ps_path_diff: Path, model: Path):
    """XmlPath vs MjsPath must be bit-identical for every fixture: the mjSpec
    shim (CompilePath::MjsPath) compiles the same model to the same mjModel the
    XML oracle does. Any divergence is a builder bug (mjs_builder.cc)."""
    r = _run(ps_path_diff, "--path-a", "XmlPath", "--path-b", "MjsPath", str(model))
    assert r.returncode == 0, f"mjs parity diff FAILED:\n{r.stdout}\n{r.stderr}"
    assert any(l.startswith("PASS") for l in r.stdout.splitlines()), r.stdout
    assert not any(l.startswith("FAIL") for l in r.stdout.splitlines()), r.stdout


def test_whole_corpus_mjs_parity_one_shot(ps_path_diff: Path):
    """One invocation over the whole fixture dir: XmlPath vs MjsPath, all PASS."""
    if not FIXTURE_FILES:
        pytest.skip("no pathdiff fixtures found")
    r = _run(ps_path_diff, "--path-a", "XmlPath", "--path-b", "MjsPath", str(FIXTURES))
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"{len(FIXTURE_FILES)} PASS, 0 FAIL" in r.stdout, r.stdout


@pytest.mark.skipif(not FIXTURE_FILES, reason="no pathdiff fixtures found")
@pytest.mark.parametrize("model", FIXTURE_FILES, ids=FIXTURE_IDS or ["<none>"])
def test_against_stock_fixture(ps_path_diff: Path, model: Path):
    """ProtoSpec XmlPath must match stock mj_loadXML of the original file."""
    r = _run(ps_path_diff, "--against-stock", str(model))
    assert r.returncode == 0, f"against-stock diff FAILED:\n{r.stdout}\n{r.stderr}"
    assert any(l.startswith("PASS") for l in r.stdout.splitlines()), r.stdout
    assert not any(l.startswith("FAIL") for l in r.stdout.splitlines()), r.stdout


def test_whole_corpus_identity_one_shot(ps_path_diff: Path):
    """One invocation over the whole fixture dir: every model PASSes, exit 0."""
    if not FIXTURE_FILES:
        pytest.skip("no pathdiff fixtures found")
    r = _run(ps_path_diff, str(FIXTURES))
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"{len(FIXTURE_FILES)} PASS, 0 FAIL" in r.stdout, r.stdout


def test_whole_corpus_against_stock_one_shot(ps_path_diff: Path):
    if not FIXTURE_FILES:
        pytest.skip("no pathdiff fixtures found")
    r = _run(ps_path_diff, "--against-stock", str(FIXTURES))
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"{len(FIXTURE_FILES)} PASS, 0 FAIL" in r.stdout, r.stdout


# --------------------------------------------------------------------------- #
# Contract guards (exit codes + benchmark line)                                #
# --------------------------------------------------------------------------- #
def test_reader_reject_is_fail(ps_path_diff: Path, tmp_path: Path):
    """A malformed model must FAIL with a nonzero exit (exit-code contract)."""
    bad = tmp_path / "bad.xml"
    bad.write_text("<mujoco><worldbody><body></mujoco", encoding="utf-8")
    r = _run(ps_path_diff, str(bad))
    assert r.returncode != 0, r.stdout
    assert "FAIL" in r.stdout, r.stdout


def test_bench_line_emitted(ps_path_diff: Path):
    """Benchmark mode prints a machine-readable per-stage line and exits 0."""
    bench_model = FIXTURES / "benchmark_many.xml"
    if not bench_model.is_file():
        pytest.skip("benchmark_many.xml not generated")
    r = _run(ps_path_diff, "--bench", "2", str(bench_model))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "BENCH model=" in r.stdout, r.stdout
    for tok in ("write_ms=", "parse_ms=", "compile_ms=", "total_ms="):
        assert tok in r.stdout, f"missing {tok} in bench line:\n{r.stdout}"


# --------------------------------------------------------------------------- #
# Big-corpus run (env-gated)                                                   #
# --------------------------------------------------------------------------- #
# Point PROTOSPEC_CORPUS at a directory of models (e.g. the MuJoCo model corpus)
# to run identity diffs across all of them. Models the ProtoSpec reader/bridge
# cannot handle are out of scope for THIS harness (test_differential.py /
# test_bridge_corpus.py own that accounting) and are counted as skips, not
# failures. A genuine identity DIVERGENCE (deterministic path disagreeing with
# itself) is always a failure. Known-bad models can be excluded by name via
# PROTOSPEC_PATHDIFF_SKIP (comma-separated filename substrings).
_CORPUS_ENV = os.environ.get("PROTOSPEC_CORPUS")


@pytest.mark.skipif(
    not _CORPUS_ENV, reason="set PROTOSPEC_CORPUS to run the big-corpus identity gate"
)
def test_corpus_identity(ps_path_diff: Path):
    corpus = Path(_CORPUS_ENV)
    if not corpus.is_dir():
        pytest.skip(f"PROTOSPEC_CORPUS is not a directory: {corpus}")
    skiplist = [
        s.strip()
        for s in os.environ.get("PROTOSPEC_PATHDIFF_SKIP", "").split(",")
        if s.strip()
    ]
    models = [
        p
        for p in sorted(corpus.rglob("*.xml"))
        if "build" not in {s.lower() for s in p.parts}
        and not p.name.startswith("._ps_")
        and not any(s in p.name for s in skiplist)
    ]
    if not models:
        pytest.skip(f"no models under {corpus}")

    divergences: list[str] = []
    out_of_scope = 0
    passed = 0
    for m in models:
        r = _run(ps_path_diff, str(m))
        if r.returncode == 0:
            passed += 1
            continue
        # Distinguish a genuine identity divergence (real failure) from a model
        # this harness simply cannot compile (reader reject / compile error /
        # missing assets) -- out of scope here, owned by the other harnesses.
        if "first divergence:" in r.stdout or "SIZE INTS" in r.stdout or (
            "NAME TABLES" in r.stdout and "compile failed" not in r.stdout
        ):
            divergences.append(f"{m}: {r.stdout.strip().splitlines()[0]}")
        else:
            out_of_scope += 1

    print(
        f"\n[corpus identity] {passed} pass, {out_of_scope} out-of-scope, "
        f"{len(divergences)} divergence(s) of {len(models)} models"
    )
    assert not divergences, "identity divergences (path disagreed with itself):\n" + "\n".join(
        divergences[:20]
    )
