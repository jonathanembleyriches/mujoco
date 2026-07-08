# ProtoSpec Python bindings — try it out

Build the `protospec` extension (pybind11 + MuJoCo, from the repo's uv venv):

```bash
uv sync                                         # installs pybind11 + numpy (dev deps)
cmake -S cpp/python -B cpp/python/build         # finds the venv Python + pybind11 + MuJoCo
cmake --build cpp/python/build --config Release  # -> cpp/python/build/Release/protospec.*.pyd (+ mujoco.dll)
```

Then, from the repo root, `uv run python`:

```python
import os, sys
d = os.path.abspath("cpp/python/build/Release")   # Linux: "cpp/python/build"
os.add_dll_directory(d); sys.path.insert(0, d)    # (add_dll_directory is Windows-only)
import protospec as ps

CORPUS = r"C:\Users\jonat\Documents\Unreal Projects\url_proj\Plugins\UnrealRoboticsLab\third_party\MuJoCo\src"
m = ps.load(os.path.join(CORPUS, "model", "humanoid", "humanoid.xml"))  # path or XML string
torso = m.worldbody.bodies[0]
print(torso.name, list(torso.pos), [b.name for b in torso.bodies])      # typed fields + child sequences
torso.pos = [0.0, 0.0, 1.5]                                             # tweak a field (opt<T> -> None-able)

c = m.compile()                                                         # -> mjModel + Binding + sim surface
print("nq", c.nq, "nv", c.nv, "nbody", c.nbody)
c.step(200)                                                            # advance the simulation
print("torso xpos", tuple(round(v, 3) for v in c.xpos("torso")), "t", round(c.time, 3))
print("torso id", c.binding.id(torso))                                 # element -> compiled id
m.save("humanoid_edited.xml")                                          # write MJCF back out
```

Build a model from scratch instead:

```python
m = ps.Model()
ball = m.worldbody.add_body(pos=[0, 0, 1], name="ball")
ball.add_freejoint(); ball.add_geom(type="sphere", size=[0.1])
c = m.compile(); c.step(100)
print("ball fell to z =", round(float(c.qpos[2]), 3))                   # qpos is a live numpy view
```
