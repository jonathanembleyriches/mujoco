"""Acceptance tests for the `protospec` Python bindings.

Runs under `uv run pytest`. Skipped in full when the compiled extension is not
present; each test then only exercises the built module, never the C++ sources
directly, so it doubles as a smoke test of the shipped artifact.

Authored against the module's public Python surface only (load/write/validate/
compile + typed element access + the schema-driven builders and the Model edit
verbs), independent of the binding C++.

Building the extension on Linux (the .so this suite imports):

    # ProtoSpec + MuJoCo object-model symbols come from the prebuilt static lib
    # libprotospec_core.a; the MuJoCo runtime from libmujoco.so. Both are built
    # by the studio build_ps tree -- rebuild the .so if those drift.
    # CRITICAL: ProtoSpec include dirs FIRST, Python/pybind11 includes LAST --
    # CPython ships a <compile.h> that shadows ProtoSpec's compile.h otherwise.
    PS=protospec/lib; STUDIO=/path/to/mujoco-studio
    g++ -O1 -std=c++20 -fPIC -shared -fvisibility=hidden \
      -o protospec.cpython-310-x86_64-linux-gnu.so \
      $PS/python/module.cc $PS/python/generated/py_*.cc \
      -I$PS/include -I$PS/generated -I$PS/core -I$PS/io -I$PS/validate \
      -I$PS/compile -I$PS/sdk -I$PS/sdk/protospec -I$PS/third_party/tinyxml2 \
      -I$PS/python -I$STUDIO/include \
      $(python -m pybind11 --includes) \
      $STUDIO/build_ps/lib/libprotospec_core.a \
      -L$STUDIO/build_ps/lib -lmujoco -Wl,-rpath,$STUDIO/build_ps/lib

Then point the suite at it: PROTOSPEC_PYD=/abs/path/to/that.so uv run pytest.
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


# --------------------------------------------------------------------------- #
# 5. Schema-driven builders: kwargs ctors + every authorable family in Python  #
# --------------------------------------------------------------------------- #
def _xml_tags(xml: str) -> set[str]:
    return {el.tag for el in ET.fromstring(xml).iter()}


def test_author_cartpole_with_material_pure_python():
    # The full cart-pole journey WITHOUT the old loads() workaround: the asset
    # <material> is authored in Python via the generated add_material builder.
    m = ps.Model()
    m.add_material(name="shiny", rgba=[0.2, 0.6, 0.9, 1.0], specular=0.8)

    m.worldbody.add_geom(name="floor", type="plane", size=[2, 2, 0.1])
    cart = m.worldbody.add_body(name="cart", pos=[0, 0, 0.2])
    cart.add_joint(name="slide_x", type="slide", axis=[1, 0, 0])
    cart.add_geom(name="cart_box", type="box", size=[0.15, 0.1, 0.05],
                  material="shiny", mass=1.0)
    pole = cart.add_body(name="pole", pos=[0, 0, 0.05])
    pole.add_joint(name="swing", type="hinge", axis=[0, 1, 0])
    pole.add_geom(name="rod", type="capsule", size=[0.02, 0.3],
                  material="shiny", mass=0.1)

    m.add_actuator("position", name="servo", joint="swing", kp=8.0)

    # The material is authored (not injected via XML) and referenced cleanly.
    xml = m.to_xml()
    assert "material" in _xml_tags(xml)
    assert 'name="shiny"' in xml and 'material="shiny"' in xml
    assert not [d for d in m.validate() if d["severity"] == "error"]

    c = m.compile()
    assert c.ok and c.nu == 1 and c.nq == 2
    c.ctrl[0] = 0.5
    c.step(50)
    assert np.isfinite(np.array(c.qpos)).all()


def test_material_kwargs_constructor_parity():
    # A material authored purely through the builder's kwargs surface.
    m = ps.Model()
    mat = m.add_material(name="steel", rgba=[0.5, 0.5, 0.5, 1.0], metallic=1.0)
    assert mat.name == "steel"
    assert list(mat.rgba) == [0.5, 0.5, 0.5, 1.0]
    assert mat.metallic == 1.0


def test_element_keyword_constructor():
    # The element class itself takes field keywords: ps.Material(name=..., ...).
    mat = ps.Material(name="brass", rgba=[0.8, 0.6, 0.2, 1.0], specular=0.5)
    assert mat.name == "brass"
    assert list(mat.rgba) == pytest.approx([0.8, 0.6, 0.2, 1.0])  # rgba is float32
    assert mat.specular == pytest.approx(0.5)
    # The no-arg form still works (empty kwargs).
    assert ps.Geom().type is None
    # Enum + list fields go through the same casters as attribute assignment.
    g = ps.Geom(type="box", size=[0.1, 0.2, 0.3])
    assert g.type == "box" and list(g.size) == pytest.approx([0.1, 0.2, 0.3])


def test_union_family_builders_sensor_tendon_equality():
    m = ps.Model()
    b1 = m.worldbody.add_body(name="b1", pos=[0, 0, 1])
    b1.add_joint(name="j1", type="hinge", axis=[0, 1, 0])
    b1.add_geom(type="sphere", size=[0.1], mass=1.0)
    b2 = m.worldbody.add_body(name="b2", pos=[1, 0, 1])
    b2.add_joint(name="j2", type="hinge", axis=[0, 1, 0])
    b2.add_geom(type="sphere", size=[0.1], mass=1.0)

    # Union builders dispatch by MJCF keyword.
    m.add_sensor("jointpos", name="s_j1", joint="j1")
    fixed = m.add_tendon("fixed", name="t1")
    m.add_equality("joint", name="eq1", joint1="j1", joint2="j2")

    xml = m.to_xml()
    tags = _xml_tags(xml)
    assert {"sensor", "jointpos", "tendon", "fixed", "equality", "joint"} <= tags
    # The tendon handle round-trips its name (typed element handle).
    assert fixed.name == "t1"

    # An unknown kind is a clean ValueError naming the alternatives.
    with pytest.raises(ValueError):
        m.add_sensor("no_such_sensor")


def test_owned_family_builders_key_and_custom():
    m = ps.Model()
    m.worldbody.add_body(name="b").add_geom(type="sphere", size=[0.1], mass=1.0)
    m.add_key(name="home")
    m.add_numeric(name="gain", data=[1.0, 2.0, 3.0])
    m.add_text(name="note")
    xml = m.to_xml()
    tags = _xml_tags(xml)
    assert {"keyframe", "key", "custom", "numeric", "text"} <= tags


# --------------------------------------------------------------------------- #
# 6. Model edit verbs: rename / delete / duplicate / reparent + name warning   #
# --------------------------------------------------------------------------- #
def _model_with_ref():
    """A model whose actuator references a joint by name (a live referrer)."""
    m = ps.Model()
    b = m.worldbody.add_body(name="link", pos=[0, 0, 1])
    j = b.add_joint(name="swing", type="hinge", axis=[0, 1, 0])
    b.add_geom(name="rod", type="capsule", size=[0.02, 0.25], mass=0.1)
    m.add_actuator("position", name="servo", joint="swing", kp=5.0)
    return m, b, j


def test_rename_rewrites_referrers():
    m, b, j = _model_with_ref()
    n = m.rename(j, "swing2")
    assert n == 1  # exactly one referrer (the actuator's joint=) rewritten
    assert j.name == "swing2"
    # The rewrite is visible in the serialized model: no stale "swing" ref.
    xml = m.to_xml()
    assert 'joint="swing2"' in xml and 'joint="swing"' not in xml
    # Still compiles (the reference resolves).
    assert m.compile().ok


def test_rename_collision_raises():
    m = ps.Model()
    b = m.worldbody.add_body(name="a")
    b2 = m.worldbody.add_body(name="b")
    with pytest.raises(ValueError):
        m.rename(b2, "a")  # name already held by another body
    assert b2.name == "b"  # unchanged on failure


def test_delete_reports_and_cascades_danglers():
    m, b, j = _model_with_ref()
    # Non-cascade delete of the referenced joint reports the dangling actuator ref.
    r = m.delete(j)
    assert r.removed is True and r.cascaded is False
    assert len(r.dangling) == 1
    d = r.dangling[0]
    assert d["field"] == "joint" and d["name"] == "swing"

    # A fresh model, deleting with cascade clears the dangling reference.
    m2, b2, j2 = _model_with_ref()
    r2 = m2.delete(j2, cascade=True)
    assert r2.removed and r2.cascaded
    # After cascade the actuator's joint ref is gone -> the model has no
    # referential errors from the delete (the actuator is simply untargeted).
    assert 'joint="swing"' not in m2.to_xml()


def test_delete_missing_element_raises():
    m, b, j = _model_with_ref()
    m.delete(j)
    with pytest.raises(ValueError):
        m.delete(j)  # already gone


def test_duplicate_returns_typed_clone_with_fresh_name():
    m = ps.Model()
    b = m.worldbody.add_body(name="widget", pos=[0, 0, 1])
    b.add_geom(name="widget_geom", type="box", size=[0.1, 0.1, 0.1], mass=1.0)

    clone = m.duplicate(b)
    # Same element type, distinct object, re-uniqued name.
    assert type(clone).__name__ == type(b).__name__ == "Body"
    assert clone.serial != b.serial
    assert clone.name and clone.name != "widget"
    # The world now holds both bodies.
    names = [x.name for x in m.worldbody.bodies]
    assert "widget" in names and clone.name in names
    assert m.compile().nbody == 3  # world + original + clone


def test_reparent_moves_child_and_rejects_cycle():
    m = ps.Model()
    a = m.worldbody.add_body(name="A", pos=[0, 0, 1])
    bee = m.worldbody.add_body(name="B", pos=[1, 0, 1])
    g = a.add_geom(name="g", type="sphere", size=[0.1], mass=1.0)

    m.reparent(g, bee)  # move g from A to B
    assert [x.name for x in a.geoms] == []
    assert [x.name for x in bee.geoms] == ["g"]

    # Reparenting a body into its own subtree is a cycle -> ValueError.
    inner = a.add_body(name="inner")
    with pytest.raises(ValueError):
        m.reparent(a, inner)


def test_reparent_to_world_none():
    m = ps.Model()
    a = m.worldbody.add_body(name="A", pos=[0, 0, 1])
    child = a.add_body(name="child", pos=[0, 0, 0.5])
    m.reparent(child, None)  # promote to a top-level (world) body
    assert "child" in [x.name for x in m.worldbody.bodies]
    assert [x.name for x in a.bodies] == []


def test_setting_name_warns_on_rename_but_not_on_first_name():
    import warnings

    m = ps.Model()
    b = m.worldbody.add_body(name="orig")

    # Renaming an already-named element via .name warns (referrers not rewritten).
    with pytest.warns(UserWarning):
        b.name = "renamed"
    assert b.name == "renamed"

    # Naming a previously-nameless element does NOT warn (nothing can dangle).
    fresh = m.worldbody.add_body()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here fails the test
        fresh.name = "first"
    assert fresh.name == "first"

    # Under a warnings-as-error filter, an in-place rename is rejected (raises)
    # and leaves the name untouched -- a rejected rename never half-applies.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning):
            fresh.name = "second"
    assert fresh.name == "first"
