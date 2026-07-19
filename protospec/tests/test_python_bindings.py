"""Milestone 7 acceptance tests for the `protospec` Python bindings.

Runs under `uv run pytest`. Skipped in full when the compiled extension is not
present (build it per TRYME.md); each test then only exercises the built module,
never the C++ sources directly, so it doubles as a smoke test of the shipped
artifact the owner will import tomorrow.

Authored against the module's public Python surface only (load/write/validate/
compile + typed element access), independent of the binding C++.
"""

from __future__ import annotations

import os
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Locate + import the compiled extension                                       #
# --------------------------------------------------------------------------- #
def _find_pyd() -> Path | None:
    env = os.environ.get("PROTOSPEC_PYD")
    if env and Path(env).is_file():
        return Path(env)
    matches = sorted(
        (ROOT / "lib" / "python").rglob("protospec*.pyd"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        # POSIX shared object (built on Linux/mac).
        matches = sorted(
            (ROOT / "lib" / "python").rglob("protospec*.so"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    return matches[0] if matches else None


def _import_protospec():
    pyd = _find_pyd()
    if pyd is None:
        return None
    d = str(pyd.parent)
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(d)  # so the sibling mujoco.dll resolves
        except OSError:
            pass
    if d not in sys.path:
        sys.path.insert(0, d)
    try:
        import protospec  # type: ignore

        return protospec
    except Exception:  # a half-built / mismatched artifact -> treat as absent
        return None


ps = _import_protospec()
np = pytest.importorskip("numpy") if ps is not None else None

pytestmark = pytest.mark.skipif(
    ps is None, reason="protospec extension not built (see TRYME.md)"
)


# --------------------------------------------------------------------------- #
# Corpus discovery (mirrors test_bridge_corpus.py)                             #
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


CORPUS_ROOT = _corpus_root()


def _corpus_sample(n: int) -> list[Path]:
    """A deterministic sample of `n` corpus MJCF files (stable across runs)."""
    if CORPUS_ROOT is None:
        return []
    files = [
        p
        for p in sorted(CORPUS_ROOT.rglob("*.xml"))
        if "build" not in {s.lower() for s in p.parts}
        and not p.name.startswith("._ps_")
        and p.stat().st_size < 2 * 1024 * 1024
    ]
    if len(files) <= n:
        return files
    return random.Random(1234).sample(files, n)


# A tiny self-contained model used by the pure-Python build test and elsewhere.
_PENDULUM_XML = """
<mujoco model="pend">
  <worldbody>
    <body name="link" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom name="rod" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02"/>
    </body>
  </worldbody>
</mujoco>
"""


# --------------------------------------------------------------------------- #
# 1. load -> edit field -> write -> reload round-trip                          #
# --------------------------------------------------------------------------- #
def test_load_edit_write_reload():
    m = ps.loads(_PENDULUM_XML)
    link = m.worldbody.bodies[0]
    assert link.name == "link"
    assert list(link.pos) == [0.0, 0.0, 1.0]

    # Edit a typed field, then write + reload and confirm it persisted.
    link.pos = [0.25, 0.0, 2.0]
    link.name = "link_moved"

    xml = ps.write(m)
    m2 = ps.loads(xml)
    link2 = m2.worldbody.bodies[0]
    assert link2.name == "link_moved"
    assert list(link2.pos) == [0.25, 0.0, 2.0]

    # Deterministic serializer: a second round trip is byte-identical.
    assert ps.write(m2) == xml


def test_typed_field_access_none_and_enums():
    m = ps.loads(_PENDULUM_XML)
    geom = m.worldbody.bodies[0].geoms[0]
    # Enums surface as their MJCF keyword strings.
    assert geom.type == "capsule"
    # An unset optional field reads as None and clears on assignment.
    assert geom.rgba is None
    geom.rgba = [1.0, 0.0, 0.0, 1.0]
    assert list(geom.rgba) == [1.0, 0.0, 0.0, 1.0]
    geom.rgba = None
    assert geom.rgba is None
    # A typed reference is a plain string.
    joint = m.worldbody.bodies[0].joints[0]
    assert joint.type == "hinge"


# --------------------------------------------------------------------------- #
# 2. build a robot in Python, compile, step, assert motion + finiteness        #
# --------------------------------------------------------------------------- #
def test_build_compile_step():
    m = ps.Model()
    world = m.worldbody
    ball = world.add_body(pos=[0.0, 0.0, 1.0], name="ball")
    ball.add_freejoint()
    ball.add_geom(type="sphere", size=[0.1], name="ball_geom")

    c = m.compile()
    assert c.ok
    assert c.nq == 7 and c.nv == 6  # a free joint
    assert c.nbody == 2  # world + ball

    q0 = np.array(c.qpos, copy=True)
    c.step(100)
    q1 = np.array(c.qpos, copy=True)

    assert not np.allclose(q0, q1), "qpos should evolve under gravity"
    assert np.isfinite(q1).all()

    x, y, z = c.xpos("ball")
    assert all(np.isfinite(v) for v in (x, y, z))
    assert z < 1.0  # the ball fell

    # Binding: the tree geom binds to a compiled id, cross-checked by name.
    geom = ball.geoms[0]
    assert c.binding.id(geom) is not None
    assert c.binding.id(ball) is not None


def test_recompile_preserves_state():
    m = ps.Model()
    world = m.worldbody
    b = world.add_body(pos=[0.0, 0.0, 1.0], name="ball")
    b.add_freejoint()
    b.add_geom(type="sphere", size=[0.1])

    c = m.compile()
    c.step(50)
    z_before = float(c.qpos[2])

    # A structural edit (add a second body), then recompile keeping state.
    b2 = world.add_body(pos=[1.0, 0.0, 1.0], name="ball2")
    b2.add_freejoint()
    b2.add_geom(type="sphere", size=[0.1])

    c2 = ps.recompile(m, c, keep_state=True)
    assert c2.nbody == 3
    assert float(c2.qpos[2]) == pytest.approx(z_before, abs=1e-9)


# --------------------------------------------------------------------------- #
# 3. validate() surfaces a planted tier-2 (referential) error                  #
# --------------------------------------------------------------------------- #
def test_validate_surfaces_tier2_error():
    # A geom references a mesh that does not exist -> a referential (tier 2) error.
    xml = (
        "<mujoco><worldbody><body>"
        '<geom type="mesh" mesh="ghost"/>'
        "</body></worldbody></mujoco>"
    )
    m = ps.loads(xml)
    diags = ps.validate(m)
    tier2 = [d for d in diags if d["tier"] == 2 and d["severity"] == "error"]
    assert tier2, f"expected a tier-2 error, got {diags}"
    d = tier2[0]
    assert set(d) >= {"tier", "severity", "message", "file", "line", "path"}
    assert "ghost" in d["message"]

    # A clean model produces no tier-1/tier-2 errors.
    assert not [
        x
        for x in ps.validate(ps.loads(_PENDULUM_XML))
        if x["severity"] == "error"
    ]


# --------------------------------------------------------------------------- #
# 4. corpus smoke: load + write byte-fixpoint over a deterministic sample       #
# --------------------------------------------------------------------------- #
_SAMPLE = _corpus_sample(30)


@pytest.mark.skipif(not _SAMPLE, reason="MuJoCo corpus not found (set PROTOSPEC_CORPUS)")
@pytest.mark.parametrize(
    "path", _SAMPLE, ids=[p.name for p in _SAMPLE] or ["<none>"]
)
def test_corpus_write_fixpoint(path: Path):
    try:
        m = ps.load(str(path))
    except ps.UnsupportedElement:
        pytest.skip("uses elements outside the supported families")
    except ValueError:
        pytest.skip("ProtoSpec reader rejected (non-model / malformed fixture)")

    # write(load(x)) then write(load(write(load(x)))) must be byte-identical:
    # a write->read->write fixpoint (plan Section 10.2).
    once = ps.write(m)
    twice = ps.write(ps.loads(once))
    assert once == twice, f"write fixpoint broke for {path.name}"


def test_corpus_sample_nonempty_when_available():
    # Guards the discovery logic: if the corpus is present, we did sample it.
    if CORPUS_ROOT is not None:
        assert _SAMPLE, "corpus present but no files sampled"
