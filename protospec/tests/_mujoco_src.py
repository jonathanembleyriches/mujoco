"""Shared discovery of a MuJoCo source checkout for the integration tests.

The bootstrap extractors (tools/bootstrap/extract_*.py) and the lift registry re-run
against a real MuJoCo source tree. That tree is not vendored into this repo, so the
integration tests that exercise the extractors are opt-in: point ``PROTOSPEC_MUJOCO_SRC``
at a MuJoCo checkout (its root, i.e. the directory containing ``include/`` and ``src/``)
and they run; leave it unset and they skip.

Historically these paths were hardcoded to the original author's Windows checkout, which
made the tests un-runnable anywhere else -- and one of them (``\\parents[2]`` on a Windows
path parsed under POSIX) crashed at *collection* time, taking the whole test session down.
Routing every extractor test through this helper keeps collection green everywhere while
still running the integration checks when the env var is set.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "PROTOSPEC_MUJOCO_SRC"

SKIP_REASON = f"set {ENV_VAR} to a MuJoCo source checkout to run this test"


def mujoco_src() -> Path | None:
    """Return the MuJoCo checkout root from ``PROTOSPEC_MUJOCO_SRC``, or ``None``.

    ``None`` is returned when the env var is unset or does not point at a plausible
    MuJoCo tree (missing ``include/mujoco/mjspec.h`` or ``src/``). Callers use this to
    drive ``pytest.mark.skipif`` so collection never depends on the tree existing.
    """
    raw = os.environ.get(ENV_VAR)
    if not raw:
        return None
    root = Path(raw).expanduser()
    if (root / "include" / "mujoco" / "mjspec.h").is_file() and (root / "src").is_dir():
        return root
    return None


MUJOCO_SRC = mujoco_src()
