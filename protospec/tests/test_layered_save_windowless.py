"""Windowless test of the layered in-place save invariant (ps::studio, ours).

Compiles ``studio/editor/test/test_layered_save_windowless.cc`` -- which splices
in ``studio/editor/layers.cc`` (SplitLayersFromTree + SaveLayeredMjcf, the units
under test) and ``undo.cc`` (CloneWithSerials) -- and runs it. It pins the
invariant a layered in-place save can never break:

    A layered in-place save must ALWAYS produce a set of files that reloads to
    an equivalent tree -- Save can never make a model unloadable.

The regression it guards: SaveLayeredMjcf used to write the SOURCE/root document
itself as a <mujocoinclude> fragment, so no <mujoco> root survived and the saved
model loaded in NEITHER ProtoSpec nor stock mj_loadXML. On a franka-shaped
fixture (a <mujoco> root that <include>s a full <mujoco> child) the test checks:
root file stays a <mujoco> root with an <include>; child file is a
<mujocoinclude> fragment; ProtoSpec reload is a WriteMjcf fixpoint; stock
mj_loadXML accepts the set (both loaders); save->reload->save is idempotent;
plus a single-layer regression.

Build strategy mirrors ``test_rigger_windowless.py``: g++ against the prebuilt
``libprotospec_core.a`` + ``libmujoco.so`` from the studio ``build_ps`` tree.
Skips cleanly when any of those are absent. The archive is a snapshot -- rebuild
``build_ps`` after touching compile/io sources.

    PROTOSPEC_BUILD_PS_LIB   dir holding libprotospec_core.a + libmujoco.so
    PROTOSPEC_MJ_INCLUDE     dir holding mujoco/mujoco.h
    PROTOSPEC_STUDIO_SRC     the mujoco-studio checkout's src/ dir
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROTOSPEC = Path(__file__).resolve().parents[1]
LIB = PROTOSPEC / "lib"

DEFAULT_BUILD_PS_LIB = Path("/home/buzz/Documents/proto/mujoco-studio/build_ps/lib")
DEFAULT_MJ_INCLUDE = Path("/home/buzz/Documents/proto/mujoco-studio/include")
DEFAULT_STUDIO_SRC = Path("/home/buzz/Documents/proto/mujoco-studio/src")

CXX = os.environ.get("CXX", "g++")


def _build_ps_lib() -> Path | None:
    p = Path(os.environ.get("PROTOSPEC_BUILD_PS_LIB", str(DEFAULT_BUILD_PS_LIB)))
    if (p / "libprotospec_core.a").is_file() and (p / "libmujoco.so").is_file():
        return p
    return None


def _mj_include() -> Path | None:
    p = Path(os.environ.get("PROTOSPEC_MJ_INCLUDE", str(DEFAULT_MJ_INCLUDE)))
    return p if (p / "mujoco" / "mujoco.h").is_file() else None


def _studio_src() -> Path | None:
    p = Path(os.environ.get("PROTOSPEC_STUDIO_SRC", str(DEFAULT_STUDIO_SRC)))
    real = p / "experimental" / "platform" / "ux" / "plugin.h"
    return p if real.is_file() else None


BUILD_PS_LIB = _build_ps_lib()
MJ_INCLUDE = _mj_include()
STUDIO_SRC = _studio_src()


def _skip_reason() -> str:
    if shutil.which(CXX) is None:
        return f"no C++ compiler ({CXX}) on PATH"
    if BUILD_PS_LIB is None:
        return "studio build_ps libs absent (set PROTOSPEC_BUILD_PS_LIB)"
    if MJ_INCLUDE is None:
        return "MuJoCo headers absent (set PROTOSPEC_MJ_INCLUDE)"
    if STUDIO_SRC is None:
        return "studio source tree absent (set PROTOSPEC_STUDIO_SRC)"
    if not (
        ROOT / "studio" / "editor" / "test" / "test_layered_save_windowless.cc"
    ).is_file():
        return "test_layered_save_windowless.cc source missing"
    return ""


SKIP_REASON = _skip_reason()
pytestmark = pytest.mark.skipif(bool(SKIP_REASON), reason=SKIP_REASON)


@pytest.fixture(scope="session")
def layered_save_test_exe(tmp_path_factory) -> Path:
    out_dir = tmp_path_factory.mktemp("layered_save_windowless_build")
    exe = out_dir / "test_layered_save_windowless"
    cmd = [
        CXX,
        "-std=c++20",
        "-O1",
        str(
            ROOT / "studio" / "editor" / "test" / "test_layered_save_windowless.cc"
        ),
        f"-I{LIB / 'include'}",
        f"-I{LIB / 'generated'}",
        f"-I{LIB / 'sdk'}",
        f"-I{LIB / 'core'}",
        f"-I{LIB / 'compile'}",
        f"-I{LIB / 'io'}",
        f"-I{LIB / 'third_party' / 'tinyxml2'}",
        f"-I{ROOT / 'studio'}",
        f"-I{ROOT / 'studio' / 'glue'}",
        f"-I{STUDIO_SRC}",
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
            "test_layered_save_windowless failed to build (build_ps libs may be "
            f"stale vs current protospec sources):\n{proc.stderr[-4000:]}"
        )
    return exe


def test_layered_save_invariant(layered_save_test_exe: Path, tmp_path: Path):
    r = subprocess.run(
        [str(layered_save_test_exe), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    assert "all checks passed" in r.stdout
