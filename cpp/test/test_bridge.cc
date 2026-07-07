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
using ps::mjcf::bridge::Binding;
using ps::mjcf::bridge::Compile;
using ps::mjcf::bridge::CompileOptions;
using ps::mjcf::bridge::CompilePath;
using ps::mjcf::bridge::Compiled;
using ps::mjcf::bridge::Recompile;

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
    "    <motor name='m1' joint='j1'/>\n"
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

int main() {
  TestCompileStepAndBinding();
  TestAutoNaming();
  TestPurityGate();
  TestNativePathRejected();
  TestRecompileStateMigration();

  std::printf("%d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
