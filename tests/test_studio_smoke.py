"""Smoke tests for the ProtoSpec Studio thin shell (milestone SE0).

Drives the built ``protospec_studio.exe`` through its ``--smoke-frames`` CI hook:
a corpus model is loaded via ProtoSpec (ParseMjcf + bridge Compile), rendered
headless for N frames and stepped; the process must exit 0. Also covers the
no-model launch. Skips cleanly when the exe has not been built.

The exe is standalone (SDL2 static + classic mjr renderer); only mujoco.dll must
sit beside it, which the build copies post-build.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _find_exe() -> Path | None:
    env = os.environ.get("PROTOSPEC_STUDIO_EXE")
    if env and Path(env).is_file():
        return Path(env)
    matches = sorted(
        (ROOT / "apps" / "studio").rglob("protospec_studio.exe"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


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


STUDIO_EXE = _find_exe()
CORPUS_ROOT = _corpus_root()

pytestmark = pytest.mark.skipif(
    STUDIO_EXE is None,
    reason="protospec_studio.exe not built (cmake --build apps/studio/build)",
)


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(STUDIO_EXE), *args],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(STUDIO_EXE.parent),
    )


def test_smoke_no_model():
    proc = _run("--smoke-frames", "10")
    assert proc.returncode == 0, proc.stderr
    assert "rendered 10 frames" in proc.stdout
    assert "model=none" in proc.stdout


@pytest.mark.skipif(CORPUS_ROOT is None, reason="MuJoCo corpus not found")
def test_smoke_humanoid_loads_renders_steps():
    humanoid = CORPUS_ROOT / "model" / "humanoid" / "humanoid.xml"
    if not humanoid.is_file():
        pytest.skip(f"humanoid.xml not found under {CORPUS_ROOT}")
    proc = _run(str(humanoid), "--smoke-frames", "60")
    assert proc.returncode == 0, proc.stderr
    # The full plugin path ran: ProtoSpec compiled the model, the host adopted
    # it, and physics advanced.
    assert "model=loaded" in proc.stdout, proc.stdout
    assert "stepped=yes" in proc.stdout, proc.stdout
