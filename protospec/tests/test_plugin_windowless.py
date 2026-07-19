"""Windowless tests of the editor's plugin surface.

Compiles ``studio/editor/test/test_plugin_windowless.cc`` -- which splices in
``studio/editor/protospec_editor.cc`` with a stub plugin registry and stub
editor-ops -- and runs it. This exercises the transport state machine and
adoption gating with no window, no ImGui and no Studio host:

* pre_compile camera conduit (cache + never-trigger-recompile)
* get_model_to_load once-per-generation gating, text/xml + source-path tagging,
  absolute asset-dir injection
* post_model_loaded own-adopt camera restore vs stock-load ingest
* do_update freeze semantics (Edit true / Play play_paused), reset-on-enter,
  pending-adopt anti-flicker hold, gizmo-drag deferral
* InjectAssetDirs on all three <compiler> shapes

Build strategy mirrors ``test_path_diff.py``: g++ against the prebuilt
``libprotospec_core.a`` + ``libmujoco.so`` from the studio ``build_ps`` tree,
plus the studio source tree for the upstream ``platform/ux/plugin.h``. Skips
cleanly when any of those are absent. The same staleness caveat applies: the
archive is a snapshot -- rebuild ``build_ps`` after touching compile/io sources.

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
    shim = p / "experimental" / "studio" / "protospec" / "platform" / "ux" / "plugin.h"
    return p if shim.is_file() else None


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
    if not (ROOT / "studio" / "editor" / "test" / "test_plugin_windowless.cc").is_file():
        return "test_plugin_windowless.cc source missing"
    return ""


SKIP_REASON = _skip_reason()
pytestmark = pytest.mark.skipif(bool(SKIP_REASON), reason=SKIP_REASON)


@pytest.fixture(scope="session")
def plugin_test_exe(tmp_path_factory) -> Path:
    out_dir = tmp_path_factory.mktemp("plugin_windowless_build")
    exe = out_dir / "test_plugin_windowless"
    cmd = [
        CXX,
        "-std=c++20",
        "-O1",
        str(ROOT / "studio" / "editor" / "test" / "test_plugin_windowless.cc"),
        f"-I{LIB / 'include'}",
        f"-I{LIB / 'generated'}",
        f"-I{LIB / 'compile'}",
        f"-I{LIB / 'io'}",
        f"-I{ROOT / 'studio'}",
        f"-I{STUDIO_SRC / 'experimental' / 'studio' / 'protospec'}",
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
            "test_plugin_windowless failed to build (build_ps libs may be "
            f"stale vs current protospec sources):\n{proc.stderr[-4000:]}"
        )
    return exe


def test_plugin_transport_and_adoption(plugin_test_exe: Path):
    r = subprocess.run(
        [str(plugin_test_exe)], capture_output=True, text=True, timeout=120
    )
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    assert "all checks passed" in r.stdout
