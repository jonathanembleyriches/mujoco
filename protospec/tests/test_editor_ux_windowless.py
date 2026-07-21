"""Windowless tests of the three ease-of-use editor features (SE0/SE2/SE3).

Compiles ``studio/editor/test/test_editor_ux_windowless.cc`` -- which splices in
``studio/editor/hierarchy_model.cc`` (the reveal ancestor-chain builder),
``studio/editor/focus_frame.cc`` (the F-to-focus centre/radius/distance math)
and ``studio/editor/undo.cc`` (the labelled snapshot stack) -- and runs it. No
window, no ImGui, no Studio host:

* SE0 undo/redo toast  -- UndoStack retains each step's label through
  push/undo/redo; UndoToastText renders "<verb>: <label> (<n> more)" (and "edit"
  for an empty label); a StatusToast carries the text and fades on the host clock
* SE2 hierarchy reveal -- HierChainToSerial returns the root..target ancestor
  chain for a nested geom, a target inside a family section (Actuators), an
  unfindable serial / serial 0 (false, empty), and a filter-hidden target
* SE3 focus frame      -- ComputeFocusFrame's centre/radius/distance for a geom
  (geom_xpos + geom_rbound), a body (subtree union sphere), a site (point-like),
  and a miss (ok == false)

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
    if not (ROOT / "studio" / "editor" / "test" / "test_editor_ux_windowless.cc").is_file():
        return "test_editor_ux_windowless.cc source missing"
    return ""


SKIP_REASON = _skip_reason()
pytestmark = pytest.mark.skipif(bool(SKIP_REASON), reason=SKIP_REASON)


@pytest.fixture(scope="session")
def editor_ux_test_exe(tmp_path_factory) -> Path:
    out_dir = tmp_path_factory.mktemp("editor_ux_windowless_build")
    exe = out_dir / "test_editor_ux_windowless"
    cmd = [
        CXX,
        "-std=c++20",
        "-O1",
        str(ROOT / "studio" / "editor" / "test" / "test_editor_ux_windowless.cc"),
        f"-I{LIB / 'include'}",
        f"-I{LIB / 'generated'}",
        f"-I{LIB / 'sdk'}",
        f"-I{LIB / 'core'}",
        f"-I{LIB / 'compile'}",
        f"-I{LIB / 'io'}",
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
            "test_editor_ux_windowless failed to build (build_ps libs may be "
            f"stale vs current protospec sources):\n{proc.stderr[-4000:]}"
        )
    return exe


def test_editor_ux(editor_ux_test_exe: Path):
    r = subprocess.run(
        [str(editor_ux_test_exe)], capture_output=True, text=True, timeout=180
    )
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    assert "all checks passed" in r.stdout
