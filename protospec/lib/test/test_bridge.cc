// Compile-bridge tests (plan milestone 5 exit criteria): compile a model, step
// it, cross-check the Binding against mj_name2id/mj_id2name, prove auto-naming
// and its opt-out, enforce the purity gate, reject the native path, and migrate
// simulation state across a structural edit via Recompile.
//
// Authored against the bridge's public surface only (compile.h/binding.h), never
// its internals -- a different concern than the compiler code it exercises.

#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include <mujoco/mujoco.h>

#include "compile.h"
#include "mjcf.h"
#include "types.h"
#include "visit.h"

using namespace ps::mjcf;
using ps::mjcf::Binding;
using ps::mjcf::Compile;
using ps::mjcf::CompileOptions;
using ps::mjcf::CompilePath;
using ps::mjcf::Compiled;
using ps::mjcf::Recompile;

static int g_failed = 0;
static int g_checks = 0;

#define CHECK(cond)                                                 \
  do {                                                              \
    ++g_checks;                                                     \
    if (!(cond)) {                                                  \
      ++g_failed;                                                   \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
    }                                                               \
  } while (0)

static bool Near(double a, double b) { return std::fabs(a - b) < 1e-9; }

// --- tree navigation helpers (same idiom as test_io.cc) ------------------- //
static const Body* World(const Model& m) {
  return m.worldbody.empty() ? nullptr : m.worldbody.front().get();
}
template <class T>
static const T* NthOf(const Body& parent, std::size_t n) {
  for (const auto& item : parent.subtree) {
    if (auto* p = std::get_if<std::unique_ptr<T>>(&item.node)) {
      if (n-- == 0) return p->get();
    }
  }
  return nullptr;
}
template <class T>
static const T* FirstOf(const Body& parent) {
  return NthOf<T>(parent, 0);
}

// A double-pendulum with an actuator; one unnamed geom exercises auto-naming.
static const char* kPendulum =
    "<mujoco>\n"
    "  <worldbody>\n"
    "    <body name='b1' pos='0 0 1'>\n"
    "      <joint name='j1' type='hinge' axis='0 1 0'/>\n"
    "      <geom name='g1' type='capsule' size='0.05 0.3' "
    "fromto='0 0 0 0 0 -0.6'/>\n"
    "      <body name='b2' pos='0 0 -0.6'>\n"
    "        <joint name='j2' type='hinge' axis='0 1 0'/>\n"
    "        <geom type='sphere' size='0.1'/>\n"  // unnamed -> auto-named
    "      </body>\n"
    "    </body>\n"
    "  </worldbody>\n"
    "  <actuator>\n"
    // Named m1 so the actuator Binding cross-check has a stable name. The default
    // Compile (Auto) now drives the mjSpec path, which supports dcmotor
    // (mjs_setToDCMotor); TestNativePathRejected forces NativePath separately.
    "    <dcmotor name='m1' joint='j1' motorconst='0.1 0.1' resistance='1'/>\n"
    "  </actuator>\n"
    "</mujoco>\n";

// --- purity sweep: (serial, loc) for every element, document order -------- //
struct Sweep {
  std::vector<std::tuple<std::uint64_t, std::string, int>>* out;
  template <class E>
  void operator()(const E& e) {
    out->emplace_back(e.serial, e.loc.file, e.loc.line);
    Rec r{this};
    Visit(e, r);
  }
  struct Rec {
    Sweep* s;
    template <class T>
    void field(int, const char*, const T&) {}
    template <class T>
    void child(int, const char*, const std::vector<std::unique_ptr<T>>& l) {
      for (const auto& p : l)
        if (p) (*s)(*p);
    }
    template <class U>
    void union_child(int, const char*, const std::vector<U>& l) {
      for (const auto& it : l)
        std::visit([&](const auto& p) { if (p) (*s)(*p); }, it.node);
    }
  };
};

static std::unique_ptr<Model> ParseOrDie(const char* xml) {
  auto r = io::ParseMjcfString(xml, "<test>");
  if (!r.ok()) {
    std::printf("PARSE FAILED:\n");
    for (const auto& d : r.errors) std::printf("  %s\n", d.Render().c_str());
  }
  return std::move(r.model);
}

// --------------------------------------------------------------------------- //
static void TestCompileStepAndBinding() {
  auto model = ParseOrDie(kPendulum);
  CHECK(model != nullptr);
  if (!model) return;

  Compiled c = Compile(*model);
  CHECK(c.ok());
  CHECK(c.model != nullptr);
  if (!c.model) return;
  mjModel* m = c.model.get();
  CHECK(m->njnt == 2);
  CHECK(m->nu == 1);

  // Step a few times: the model is dynamically valid.
  mjData* d = mj_makeData(m);
  CHECK(d != nullptr);
  for (int i = 0; i < 20 && d; ++i) mj_step(m, d);
  if (d) {
    CHECK(std::isfinite(d->qpos[0]));
    mj_deleteData(d);
  }

  // Element pointers from the tree.
  const Body* b1 = FirstOf<Body>(*World(*model));
  const Body* b2 = FirstOf<Body>(*b1);
  const Joint* j1 = FirstOf<Joint>(*b1);
  const Joint* j2 = FirstOf<Joint>(*b2);
  const Geom* g1 = FirstOf<Geom>(*b1);
  CHECK(b1 && b2 && j1 && j2 && g1);

  // Id() agrees with mj_name2id, and the reverse id->name agrees, for every
  // authored name (plan Section 10.6 binding cross-check).
  auto crosscheck = [&](auto id_opt, int objtype, const char* name) {
    CHECK(id_opt.has_value());
    if (!id_opt) return;
    CHECK(*id_opt == mj_name2id(m, objtype, name));
    const char* back = mj_id2name(m, objtype, *id_opt);
    CHECK(back != nullptr && std::string(back) == name);
  };
  crosscheck(c.binding.Id(*b1), mjOBJ_BODY, "b1");
  crosscheck(c.binding.Id(*b2), mjOBJ_BODY, "b2");
  crosscheck(c.binding.Id(*j1), mjOBJ_JOINT, "j1");
  crosscheck(c.binding.Id(*j2), mjOBJ_JOINT, "j2");
  crosscheck(c.binding.Id(*g1), mjOBJ_GEOM, "g1");

  // Typed reverse lookup.
  CHECK(c.binding.JointAt(*c.binding.Id(*j1)) == j1);
  CHECK(c.binding.BodyAt(*c.binding.Id(*b2)) == b2);
  CHECK(c.binding.GeomAt(*c.binding.Id(*g1)) == g1);

  // Address sugar matches the model's own address arrays.
  auto qadr = c.binding.QposAdr(*j1);
  auto dadr = c.binding.DofAdr(*j1);
  CHECK(qadr && *qadr == m->jnt_qposadr[mj_name2id(m, mjOBJ_JOINT, "j1")]);
  CHECK(dadr && *dadr == m->jnt_dofadr[mj_name2id(m, mjOBJ_JOINT, "j1")]);

  // Actuator via the union wrapper.
  const auto& act = model->actuators.front()->actuators.front();
  auto aid = c.binding.ActId(act);
  CHECK(aid && *aid == mj_name2id(m, mjOBJ_ACTUATOR, "m1"));
}

static void TestAutoNaming() {
  auto model = ParseOrDie(kPendulum);
  if (!model) return;
  const Geom* unnamed = FirstOf<Geom>(*FirstOf<Body>(*FirstOf<Body>(*World(*model))));
  CHECK(unnamed != nullptr);  // the sphere in b2, no authored name

  // Default: auto-named, hence bindable.
  Compiled on = Compile(*model);
  CHECK(on.ok());
  CHECK(on.binding.Id(*unnamed).has_value());

  // Opt out: unnamed elements become unbindable.
  auto model2 = ParseOrDie(kPendulum);
  CompileOptions opts;
  opts.auto_name = false;
  Compiled off = Compile(*model2, opts);
  CHECK(off.ok());
  const Geom* unnamed2 =
      FirstOf<Geom>(*FirstOf<Body>(*FirstOf<Body>(*World(*model2))));
  CHECK(!off.binding.Id(*unnamed2).has_value());

  // Pattern query still finds the auto-named geom by its reserved prefix.
  auto hits = on.binding.Find(mjOBJ_GEOM, "_ps:geom:*");
  CHECK(!hits.empty());
}

static void TestPurityGate() {
  auto model = ParseOrDie(kPendulum);
  if (!model) return;

  auto clone = Clone(*model);
  std::vector<std::tuple<std::uint64_t, std::string, int>> before, after;
  Sweep{&before}(*model);

  // Compile through both Auto and forced XmlPath; neither may mutate the tree.
  CompileOptions xml;
  xml.path = CompilePath::XmlPath;
  Compiled a = Compile(*model);
  Compiled b = Compile(*model, xml);
  CHECK(a.ok() && b.ok());

  Sweep{&after}(*model);
  CHECK(*model == *clone);              // content unchanged (== excludes serial/loc)
  CHECK(before == after);               // serial + loc unchanged
}

static void TestNativePathRejected() {
  auto model = ParseOrDie(kPendulum);
  if (!model) return;
  CompileOptions opts;
  opts.path = CompilePath::NativePath;
  Compiled c = Compile(*model, opts);
  CHECK(!c.ok());
  CHECK(c.model == nullptr);
  CHECK(!c.report.errors.empty());
  CHECK(c.report.taken == CompilePath::NativePath);
}

// Append a named body (hinge joint + geom) to the world; returns the new joint.
static const Joint* AppendBody(Model& m) {
  auto body = std::make_unique<Body>();
  body->name = "b3";
  body->pos = std::array<double, 3>{0.8, 0, 1};
  auto joint = std::make_unique<Joint>();
  joint->name = "j3";
  joint->type = JointType::hinge;
  joint->axis = std::array<double, 3>{0, 1, 0};
  const Joint* jptr = joint.get();
  auto geom = std::make_unique<Geom>();
  geom->type = GeomType::sphere;
  geom->size = ps::InlineVec<double, 3>{0.1};
  BodyChildAny cj;
  cj.node = std::move(joint);
  BodyChildAny cg;
  cg.node = std::move(geom);
  body->subtree.push_back(std::move(cj));
  body->subtree.push_back(std::move(cg));
  BodyChildAny cb;
  cb.node = std::move(body);
  m.worldbody.front()->subtree.push_back(std::move(cb));
  return jptr;
}

static void TestRecompileStateMigration() {
  auto model = ParseOrDie(kPendulum);
  if (!model) return;

  Compiled c0 = Compile(*model);
  CHECK(c0.ok());
  if (!c0.ok()) return;

  const Body* b1 = FirstOf<Body>(*World(*model));
  const Body* b2 = FirstOf<Body>(*b1);
  const Joint* j1 = FirstOf<Joint>(*b1);
  const Joint* j2 = FirstOf<Joint>(*b2);

  mjData* d0 = mj_makeData(c0.model.get());
  CHECK(d0 != nullptr);
  if (!d0) return;

  // Seed distinct state on the surviving joints + a nonzero time.
  const int qa1 = *c0.binding.QposAdr(*j1), qa2 = *c0.binding.QposAdr(*j2);
  const int da1 = *c0.binding.DofAdr(*j1), da2 = *c0.binding.DofAdr(*j2);
  d0->qpos[qa1] = 0.3;
  d0->qpos[qa2] = -0.4;
  d0->qvel[da1] = 0.7;
  d0->qvel[da2] = 1.1;
  d0->time = 1.25;

  // Structural edit on the same tree instance (surviving elements keep their
  // serials). Per the Binding contract this invalidates c0's id mappings; the
  // fresh binding comes from Recompile.
  const Joint* j3 = AppendBody(*model);

  mjData* d1 = nullptr;
  Compiled c1 = Recompile(*model, c0, d0, &d1);
  CHECK(c1.ok());
  CHECK(d1 != nullptr);
  if (c1.ok() && d1) {
    // Surviving joints keep their state at the new addresses.
    const int nqa1 = *c1.binding.QposAdr(*j1), nqa2 = *c1.binding.QposAdr(*j2);
    const int nda1 = *c1.binding.DofAdr(*j1), nda2 = *c1.binding.DofAdr(*j2);
    CHECK(Near(d1->qpos[nqa1], 0.3));
    CHECK(Near(d1->qpos[nqa2], -0.4));
    CHECK(Near(d1->qvel[nda1], 0.7));
    CHECK(Near(d1->qvel[nda2], 1.1));
    // The new joint falls back to qpos0 (zero for a hinge) and binds.
    auto j3id = c1.binding.Id(*j3);
    CHECK(j3id.has_value());
    const int nqa3 = *c1.binding.QposAdr(*j3);
    CHECK(Near(d1->qpos[nqa3], 0.0));
    // Time is preserved across the recompile.
    CHECK(Near(d1->time, 1.25));
    // The migrated data steps cleanly.
    mj_step(c1.model.get(), d1);
    CHECK(std::isfinite(d1->qpos[nqa1]));
  }
  if (d1) mj_deleteData(d1);
  mj_deleteData(d0);
}

// Full recompile cycle on a FORCED path: compile the pendulum, seed joint state
// + a nonzero time, append a body, recompile migrating state, step once. Returns
// the migrated slices so the two paths can be compared element-for-element.
// Recompile is path-agnostic (it re-enters Compile with the same opts), so the
// mjSpec and XML paths must migrate identical state.
struct RecompileResult {
  bool ok = false;
  double q1 = 0, q2 = 0, v1 = 0, v2 = 0, q3 = 0, time = 0;
  bool j3_bound = false;
  bool stepped_finite = false;
};

static RecompileResult RunRecompileCycle(CompilePath path) {
  RecompileResult r;
  auto model = ParseOrDie(kPendulum);
  if (!model) return r;
  CompileOptions opts;
  opts.path = path;

  Compiled c0 = Compile(*model, opts);
  if (!c0.ok() || c0.report.taken != path) return r;  // forced path honored

  const Body* b1 = FirstOf<Body>(*World(*model));
  const Body* b2 = FirstOf<Body>(*b1);
  const Joint* j1 = FirstOf<Joint>(*b1);
  const Joint* j2 = FirstOf<Joint>(*b2);

  mjData* d0 = mj_makeData(c0.model.get());
  if (!d0) return r;
  const int qa1 = *c0.binding.QposAdr(*j1), qa2 = *c0.binding.QposAdr(*j2);
  const int da1 = *c0.binding.DofAdr(*j1), da2 = *c0.binding.DofAdr(*j2);
  d0->qpos[qa1] = 0.3;
  d0->qpos[qa2] = -0.4;
  d0->qvel[da1] = 0.7;
  d0->qvel[da2] = 1.1;
  d0->time = 1.25;

  const Joint* j3 = AppendBody(*model);
  mjData* d1 = nullptr;
  Compiled c1 = Recompile(*model, c0, d0, &d1, opts);
  if (c1.ok() && d1 && c1.report.taken == path) {
    const int nqa1 = *c1.binding.QposAdr(*j1), nqa2 = *c1.binding.QposAdr(*j2);
    const int nda1 = *c1.binding.DofAdr(*j1), nda2 = *c1.binding.DofAdr(*j2);
    r.q1 = d1->qpos[nqa1];
    r.q2 = d1->qpos[nqa2];
    r.v1 = d1->qvel[nda1];
    r.v2 = d1->qvel[nda2];
    auto j3id = c1.binding.Id(*j3);
    r.j3_bound = j3id.has_value();
    if (r.j3_bound) r.q3 = d1->qpos[*c1.binding.QposAdr(*j3)];
    r.time = d1->time;
    mj_step(c1.model.get(), d1);
    r.stepped_finite = std::isfinite(d1->qpos[nqa1]);
    r.ok = true;
  }
  if (d1) mj_deleteData(d1);
  mj_deleteData(d0);
  return r;
}

static void TestRecompileParityBothPaths() {
  RecompileResult x = RunRecompileCycle(CompilePath::XmlPath);
  RecompileResult s = RunRecompileCycle(CompilePath::MjsPath);
  CHECK(x.ok);
  CHECK(s.ok);
  if (!x.ok || !s.ok) return;

  // Each path migrated the seeded state to its recompiled addresses.
  CHECK(Near(x.q1, 0.3) && Near(s.q1, 0.3));
  CHECK(Near(x.q2, -0.4) && Near(s.q2, -0.4));
  CHECK(Near(x.v1, 0.7) && Near(s.v1, 0.7));
  CHECK(Near(x.v2, 1.1) && Near(s.v2, 1.1));
  CHECK(x.j3_bound && s.j3_bound);
  CHECK(Near(x.q3, 0.0) && Near(s.q3, 0.0));
  CHECK(Near(x.time, 1.25) && Near(s.time, 1.25));
  CHECK(x.stepped_finite && s.stepped_finite);

  // Path-agnostic Recompile: the two forced paths agree element-for-element.
  CHECK(Near(x.q1, s.q1) && Near(x.q2, s.q2));
  CHECK(Near(x.v1, s.v1) && Near(x.v2, s.v2));
  CHECK(Near(x.q3, s.q3) && Near(x.time, s.time));
}

// First element of type T directly under any container with a `subtree` (Body
// or Frame); the generated union subtree holds Body/Geom/Site/... nodes.
template <class T, class P>
static const T* FirstIn(const P& parent) {
  for (const auto& item : parent.subtree)
    if (auto* p = std::get_if<std::unique_ptr<T>>(&item.node)) return p->get();
  return nullptr;
}

// --- Pose-patch: move a compiled element without recompiling -------------- //

// DR-S6 correctness proof at the library level. A mesh geom's compiled pose
// bakes the mesh recentering frame B into geom_pos. Capturing the PosePatch and
// re-applying A ∘ L_new ∘ B (never inverting B) must move geom_xpos by exactly
// the intended local delta -- the read-back-and-invert approach corrupts this.
static void TestPosePatchMeshGeom() {
  // A tetrahedron whose vertices are NOT centred on the origin, so MuJoCo
  // recenters it: mesh_pos/mesh_quat (folded into geom_pos) are non-identity.
  static const char* kMesh =
      "<mujoco>\n"
      "  <asset>\n"
      "    <mesh name='tet' vertex='0 0 0  2 0 0  0 2 0  0 0 2'/>\n"
      "  </asset>\n"
      "  <worldbody>\n"
      "    <body name='b'>\n"
      "      <geom name='g' type='mesh' mesh='tet' pos='0.3 0 0'"
      " quat='0.9238795 0 0 0.3826834'/>\n"
      "    </body>\n"
      "  </worldbody>\n"
      "</mujoco>";
  auto model = ParseOrDie(kMesh);
  CHECK(model != nullptr);
  if (!model) return;
  Compiled c = Compile(*model);
  CHECK(c.ok());
  if (!c.ok()) {
    for (const auto& e : c.report.errors) std::printf("  %s\n", e.Render().c_str());
    return;
  }
  mjModel* m = c.model.get();

  const Body* b = FirstOf<Body>(*World(*model));
  const Geom* g = b ? FirstIn<Geom>(*b) : nullptr;
  CHECK(g != nullptr);
  if (!g) return;

  auto pp = c.binding.PosePatchFor(*g);
  CHECK(pp.has_value());
  if (!pp) return;
  CHECK(pp->objtype == mjOBJ_GEOM);
  // The mesh recentering really is baked in (otherwise the proof is vacuous):
  // the suffix B is non-identity.
  const double bnorm = std::sqrt(pp->suffix.pos[0] * pp->suffix.pos[0] +
                                 pp->suffix.pos[1] * pp->suffix.pos[1] +
                                 pp->suffix.pos[2] * pp->suffix.pos[2]);
  CHECK(bnorm > 1e-2);

  const int gid = *c.binding.Id(*g);
  mjData* d = mj_makeData(m);
  mj_kinematics(m, d);
  double x0[3] = {d->geom_xpos[3 * gid], d->geom_xpos[3 * gid + 1],
                  d->geom_xpos[3 * gid + 2]};

  // L_new = the authored local pose, translated by delta (same orientation).
  const double delta[3] = {0.1, -0.2, 0.05};
  RigidPose L_new;
  for (int i = 0; i < 3; ++i) L_new.pos[i] = (*g->pos)[i] + delta[i];
  for (int i = 0; i < 4; ++i) L_new.quat[i] = (*g->quat)[i];

  CHECK(ApplyPosePatch(m, *pp, L_new));
  mj_kinematics(m, d);
  // geom_xpos moved by EXACTLY the local delta (body/frame identity): B was
  // re-applied, not inverted, so the mesh recentering term cancels in the delta.
  for (int i = 0; i < 3; ++i)
    CHECK(Near(d->geom_xpos[3 * gid + i] - x0[i], delta[i]));
  mj_deleteData(d);
}

// Free-jointed body: the rest pose at qpos0 is driven by qpos, so ApplyPosePatch
// must reseed qpos0 (reseed_width 7) for the patch to move the body.
static void TestPosePatchFreeBody() {
  static const char* kFree =
      "<mujoco>\n"
      "  <worldbody>\n"
      "    <body name='fb' pos='0 0 1'>\n"
      "      <freejoint/>\n"
      "      <geom name='fg' type='box' size='0.1 0.1 0.1'/>\n"
      "    </body>\n"
      "  </worldbody>\n"
      "</mujoco>";
  auto model = ParseOrDie(kFree);
  CHECK(model != nullptr);
  if (!model) return;
  Compiled c = Compile(*model);
  CHECK(c.ok());
  if (!c.ok()) return;
  mjModel* m = c.model.get();

  const Body* fb = FirstOf<Body>(*World(*model));
  const Geom* fg = fb ? FirstIn<Geom>(*fb) : nullptr;
  CHECK(fb && fg);
  if (!fb || !fg) return;

  auto pp = c.binding.PosePatchFor(*fb);
  CHECK(pp.has_value());
  if (!pp) return;
  CHECK(pp->objtype == mjOBJ_BODY);
  CHECK(pp->reseed_width == 7 && pp->reseed_qposadr >= 0);  // free-joint reseed

  const int gid = *c.binding.Id(*fg);
  mjData* d = mj_makeData(m);
  mj_resetData(m, d);
  mj_kinematics(m, d);
  double x0[3] = {d->geom_xpos[3 * gid], d->geom_xpos[3 * gid + 1],
                  d->geom_xpos[3 * gid + 2]};

  const double delta[3] = {0.4, -0.3, 0.2};
  RigidPose L_new;  // body authored pos + delta, identity orientation
  for (int i = 0; i < 3; ++i) L_new.pos[i] = (*fb->pos)[i] + delta[i];

  CHECK(ApplyPosePatch(m, *pp, L_new));
  mj_resetData(m, d);  // qpos <- reseeded qpos0
  mj_kinematics(m, d);
  for (int i = 0; i < 3; ++i)
    CHECK(Near(d->geom_xpos[3 * gid + i] - x0[i], delta[i]));
  mj_deleteData(d);
}

// Frame-nested geom: the enclosing <frame> is flattened into geom_pos as the
// prefix A. PosePatchFor must reconstruct A from the tree, and ApplyPosePatch
// must re-compose it, so a local edit maps through the frame.
static void TestPosePatchFrameNested() {
  static const char* kFrame =
      "<mujoco>\n"
      "  <worldbody>\n"
      "    <body name='fb2'>\n"
      "      <frame pos='0.5 0 0' quat='0.7071068 0 0 0.7071068'>\n"
      "        <geom name='fg2' type='box' size='0.1 0.1 0.1' pos='0.2 0 0'/>\n"
      "      </frame>\n"
      "    </body>\n"
      "  </worldbody>\n"
      "</mujoco>";
  auto model = ParseOrDie(kFrame);
  CHECK(model != nullptr);
  if (!model) return;
  Compiled c = Compile(*model);
  CHECK(c.ok());
  if (!c.ok()) return;
  mjModel* m = c.model.get();

  const Body* fb2 = FirstOf<Body>(*World(*model));
  const Frame* frame = fb2 ? FirstIn<Frame>(*fb2) : nullptr;
  const Geom* g = frame ? FirstIn<Geom>(*frame) : nullptr;
  CHECK(fb2 && frame && g);
  if (!g || !frame) return;

  auto pp = c.binding.PosePatchFor(*g);
  CHECK(pp.has_value());
  if (!pp) return;
  // The prefix A equals the authored frame pose (reconstructed from the tree).
  for (int i = 0; i < 3; ++i) CHECK(Near(pp->prefix.pos[i], (*frame->pos)[i]));
  for (int i = 0; i < 4; ++i) CHECK(Near(pp->prefix.quat[i], (*frame->quat)[i]));

  const int gid = *c.binding.Id(*g);
  mjData* d = mj_makeData(m);
  mj_kinematics(m, d);
  double x0[3] = {d->geom_xpos[3 * gid], d->geom_xpos[3 * gid + 1],
                  d->geom_xpos[3 * gid + 2]};

  const double delta[3] = {0.15, 0.0, 0.0};
  RigidPose L_new;
  for (int i = 0; i < 3; ++i) L_new.pos[i] = (*g->pos)[i] + delta[i];
  for (int i = 0; i < 4; ++i)
    L_new.quat[i] = g->quat ? (*g->quat)[i] : (i == 0 ? 1.0 : 0.0);

  CHECK(ApplyPosePatch(m, *pp, L_new));
  mj_kinematics(m, d);
  // geom_xpos matches A ∘ L_new ∘ B composed independently (body at identity).
  RigidPose expect = Compose(Compose(pp->prefix, L_new), pp->suffix);
  for (int i = 0; i < 3; ++i) CHECK(Near(d->geom_xpos[3 * gid + i], expect.pos[i]));
  // A 90deg-Z frame maps a +x local delta to a +y world move (A is in play).
  CHECK(Near(d->geom_xpos[3 * gid + 1] - x0[1], delta[0]));
  mj_deleteData(d);
}

int main() {
  TestCompileStepAndBinding();
  TestAutoNaming();
  TestPurityGate();
  TestNativePathRejected();
  TestRecompileStateMigration();
  TestRecompileParityBothPaths();
  TestPosePatchMeshGeom();
  TestPosePatchFreeBody();
  TestPosePatchFrameNested();

  std::printf("%d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
