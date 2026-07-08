// Native-compiler NC0 tests. Authored against the compile/bridge public
// surfaces to exercise the scaffolding that lands before any compiler stage:
//
//   T0.4  purity gate on the native entry (clone/==/serial-loc sweep even in
//         fallback), feature-gate correctness (every model reports
//         UnsupportedNatively with a non-empty feature list), report shape.
//   T0.5  the mjModel allocator spike: our lifted MakeModel round-trips through
//         the engine's own mj_copyModel / mj_deleteModel (Open Q2).
//   T0.1  mjuu_* lifted smoke goldens (hand cases + mju_* analogues).

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include <mujoco/mujoco.h>

#include "compile.h"
#include "make_model.h"
#include "mjcf.h"
#include "mjuu_util.h"
#include "model_diff_lib.h"
#include "native.h"
#include "types.h"
#include "visit.h"

using namespace ps::mjcf;
using ps::mjcf::bridge::Compile;
using ps::mjcf::bridge::CompileOptions;
using ps::mjcf::bridge::CompilePath;
using ps::mjcf::bridge::Compiled;

static int g_failed = 0;
static int g_checks = 0;

#define CHECK(cond)                                                \
  do {                                                             \
    ++g_checks;                                                    \
    if (!(cond)) {                                                 \
      ++g_failed;                                                  \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
    }                                                              \
  } while (0)

static bool Near(double a, double b) { return std::fabs(a - b) < 1e-9; }

static std::unique_ptr<Model> ParseOrDie(const char* xml) {
  auto r = io::ParseMjcfString(xml, "<test>");
  if (!r.ok()) {
    std::printf("PARSE FAILED:\n");
    for (const auto& d : r.errors) std::printf("  %s\n", d.Render().c_str());
  }
  return std::move(r.model);
}

static const char* kPendulum =
    "<mujoco>\n"
    "  <worldbody>\n"
    "    <body name='b1' pos='0 0 1'>\n"
    "      <joint name='j1' type='hinge' axis='0 1 0'/>\n"
    "      <geom name='g1' type='capsule' size='0.05 0.3'/>\n"
    "      <body name='b2' pos='0 0 -0.3'>\n"
    "        <joint name='j2' type='hinge' axis='0 1 0'/>\n"
    "        <geom name='g2' type='sphere' size='0.05'/>\n"
    "      </body>\n"
    "    </body>\n"
    "  </worldbody>\n"
    "</mujoco>\n";

// A model whose <actuator> is outside the NC1/NC1b native slice: the gate routes
// the whole model to the XML fallback. (Sites, cameras, lights, frames, contact
// pairs/excludes and keyframes are native as of NC1b, so an unsupported example
// must reach for a still-unsupported family -- here, an actuator.)
static const char* kUnsupported =
    "<mujoco>\n"
    "  <worldbody>\n"
    "    <body name='b1' pos='0 0 1'>\n"
    "      <joint name='j1' type='hinge' axis='0 1 0'/>\n"
    "      <geom name='g1' type='sphere' size='0.05'/>\n"
    "      <site name='s1' pos='0 0 0'/>\n"
    "    </body>\n"
    "  </worldbody>\n"
    "  <actuator><motor name='m1' joint='j1'/></actuator>\n"
    "</mujoco>\n";

// --- purity sweep (serial, loc) for every element, document order ----------- //
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

// T0.4: the native entry is pure even when it falls back. Clone + sweep before,
// forced-NativePath and Auto compiles, sweep after -- tree byte-identical.
static void TestNativePurity() {
  auto model = ParseOrDie(kPendulum);
  if (!model) return;
  auto clone = Clone(*model);
  std::vector<std::tuple<std::uint64_t, std::string, int>> before, after;
  Sweep{&before}(*model);

  CompileOptions native;
  native.path = CompilePath::NativePath;
  Compiled a = Compile(*model, native);          // forced native (falls back null)
  Compiled b = Compile(*model);                  // Auto (native try -> XML)
  (void)a;
  (void)b;

  Sweep{&after}(*model);
  CHECK(*model == *clone);   // content unchanged (== excludes serial/loc)
  CHECK(before == after);    // serial + loc unchanged
}

// Forced NativePath on an out-of-slice model reports UnsupportedNatively with a
// non-empty feature list naming the offending family ("site").
static void TestGateUnsupported() {
  auto model = ParseOrDie(kUnsupported);
  if (!model) return;

  auto reasons = compile::CollectUnsupportedFeatures(*model);
  CHECK(!reasons.empty());
  bool saw_motor = false;
  for (const auto& r : reasons) {
    if (r.feature == "motor") saw_motor = true;
    CHECK(r.count > 0);
  }
  CHECK(saw_motor);

  CompileOptions native;
  native.path = CompilePath::NativePath;
  Compiled c = Compile(*model, native);
  CHECK(c.model == nullptr);
  CHECK(!c.ok());
  CHECK(c.report.taken == CompilePath::NativePath);
  CHECK(!c.report.fallback_reasons.empty());
  CHECK(!c.report.errors.empty());
  bool mentions = false;
  for (const auto& d : c.report.errors)
    if (d.message.find("UnsupportedNatively") != std::string::npos) mentions = true;
  CHECK(mentions);
}

// An in-slice model uses no unsupported feature -- the gate admits it whole.
static void TestGateAdmitsRigidBody() {
  auto model = ParseOrDie(kPendulum);
  if (!model) return;
  CHECK(compile::CollectUnsupportedFeatures(*model).empty());
}

// Auto on an out-of-slice model falls back LOUDLY: succeeds via XML but carries
// the native fallback reasons, and taken == XmlPath.
static void TestAutoLoudFallback() {
  auto model = ParseOrDie(kUnsupported);
  if (!model) return;
  Compiled c = Compile(*model);  // Auto
  CHECK(c.ok());
  CHECK(c.model != nullptr);
  CHECK(c.report.taken == CompilePath::XmlPath);
  CHECK(!c.report.fallback_reasons.empty());   // the fallback is not silent
}

// Auto on an in-slice model takes the native path with no fallback reasons.
static void TestAutoNativePath() {
  auto model = ParseOrDie(kPendulum);
  if (!model) return;
  Compiled c = Compile(*model);  // Auto
  CHECK(c.ok());
  CHECK(c.model != nullptr);
  CHECK(c.report.taken == CompilePath::NativePath);
  CHECK(c.report.fallback_reasons.empty());
}

// Report shape stability -- ok() == errors.empty(), requested recorded. The
// in-slice model now compiles natively (ok()).
static void TestReportShape() {
  auto model = ParseOrDie(kPendulum);
  if (!model) return;
  CompileOptions native;
  native.path = CompilePath::NativePath;
  Compiled c = Compile(*model, native);
  CHECK(c.report.requested == CompilePath::NativePath);
  CHECK(c.report.ok() == c.report.errors.empty());
  CHECK(c.report.ok());  // NC1: the rigid-body slice compiles natively
}

// T0.5: allocator spike. Build a plausible sizes header, allocate via the
// lifted MakeModel, write into the name arrays, and prove layout compatibility
// by round-tripping through the engine's own mj_copyModel + mj_deleteModel.
static void TestAllocatorRoundTrip() {
  mjModel sizes;
  std::memset(&sizes, 0, sizeof(sizes));
  sizes.nq = 2;
  sizes.nv = 2;
  sizes.nbody = 3;
  sizes.njnt = 2;
  sizes.ngeom = 2;
  sizes.nsite = 1;
  sizes.ntree = 1;
  sizes.nM = 3;
  sizes.nB = 4;
  sizes.nC = 3;
  sizes.nD = 4;
  sizes.nnames = 32;
  sizes.npaths = 1;

  mjModel* m = compile::lifted::MakeModel(sizes);
  CHECK(m != nullptr);
  if (!m) return;
  CHECK(m->nbody == 3);
  CHECK(m->nbuffer > 0);
  CHECK(m->buffer != nullptr);
  CHECK(m->qpos0 != nullptr);
  CHECK(m->names != nullptr);
  CHECK(Near(m->stat.extent, 2.0));

  // Write into buffer-backed arrays to prove the pointers are usable.
  const char* nm = "worldbody";
  std::memcpy(m->names, nm, std::strlen(nm) + 1);
  m->qpos0[0] = 0.25;
  m->qpos0[1] = -0.5;

  // Round-trip through the engine's own copy (which walks the same layout).
  mjModel* copy = mj_copyModel(nullptr, m);
  CHECK(copy != nullptr);
  if (copy) {
    CHECK(copy->nbuffer == m->nbuffer);
    CHECK(copy->nbody == m->nbody);
    CHECK(std::strcmp(copy->names, "worldbody") == 0);
    CHECK(Near(copy->qpos0[0], 0.25));
    CHECK(Near(copy->qpos0[1], -0.5));
    mj_deleteModel(copy);
  }
  mj_deleteModel(m);  // frees a MakeModel-allocated buffer via the engine
}

// T0.1: mjuu_* lifted smoke goldens -- hand cases + public mju_* analogues.
static void TestMjuuSmoke() {
  namespace L = ps::mjcf::compile::lifted;

  // hand case: dot product.
  const double a[3] = {1, 2, 3}, b[3] = {4, 5, 6};
  CHECK(Near(L::mjuu_dot3(a, b), 32.0));

  // mjuu_mulquat vs the public mju_mulQuat.
  double qa[4] = {0.5, 0.5, 0.5, 0.5};
  double qb[4] = {0.7071067811865476, 0.7071067811865476, 0, 0};
  double got[4], want[4];
  L::mjuu_mulquat(got, qa, qb);
  mju_mulQuat(want, qa, qb);
  for (int i = 0; i < 4; ++i) CHECK(Near(got[i], want[i]));

  // mjuu_rotVecQuat vs the public mju_rotVecQuat.
  double v[3] = {0.3, -0.7, 1.1}, gv[3], wv[3];
  L::mjuu_rotVecQuat(gv, v, qa);
  mju_rotVecQuat(wv, v, qa);
  for (int i = 0; i < 3; ++i) CHECK(Near(gv[i], wv[i]));

  // mjuu_normvec returns the previous length and unit-normalizes.
  double u[3] = {3, 0, 4};
  double len = L::mjuu_normvec(u, 3);
  CHECK(Near(len, 5.0));
  CHECK(Near(u[0], 0.6) && Near(u[1], 0.0) && Near(u[2], 0.8));
}

// NC1: a native compile is bit-identical to the XML oracle over the rigid-body
// slice -- from-scratch fixtures exercising orientation forms, primitive geoms,
// inertia inference, all joint types, and nested trees. Uses the harness's
// mjModel comparison (model_diff_lib), the same one ps_native_diff runs.
static void CheckBitIdentical(const char* label, const char* xml) {
  auto model = ParseOrDie(xml);
  if (!model) { CHECK(false); return; }
  CompileOptions native;
  native.path = CompilePath::NativePath;
  Compiled nat = Compile(*model, native);
  Compiled xmlc = Compile(*model, [] {
    CompileOptions o;
    o.path = CompilePath::XmlPath;
    return o;
  }());
  CHECK(nat.ok() && nat.model);
  CHECK(xmlc.ok() && xmlc.model);
  if (!nat.model || !xmlc.model) return;
  ps::harness::Tol tol;
  std::string err;
  ps::harness::DiffReport rep =
      ps::harness::DiffModels(xmlc.model.get(), nat.model.get(), tol, 4, err);
  if (!err.empty() || rep.Differs()) {
    std::printf("FAIL bit-identity [%s]: %s\n", label,
                rep.FirstDivergence().c_str());
    ++g_failed;
  }
  ++g_checks;
}

static void TestNc1BitIdentical() {
  CheckBitIdentical("empty", "<mujoco><worldbody/></mujoco>");
  CheckBitIdentical("pendulum", kPendulum);
  CheckBitIdentical(
      "euler",
      "<mujoco><worldbody><body pos='1 -0.5 0' euler='10 20 30'>"
      "<geom type='box' size='0.15 0.2 0.25' pos='0.1 0.2 0.3' euler='10 20 30'/>"
      "</body></worldbody></mujoco>");
  CheckBitIdentical(
      "primitives",
      "<mujoco><worldbody>"
      "<geom type='plane' size='1 1 0.1'/>"
      "<body pos='0 0 1'><geom type='sphere' size='0.2'/>"
      "<geom type='capsule' fromto='0 0 0 0 0 0.5' size='0.05'/>"
      "<geom type='cylinder' size='0.05 0.2'/>"
      "<geom type='ellipsoid' size='0.1 0.15 0.2'/></body>"
      "</worldbody></mujoco>");
  CheckBitIdentical(
      "joints",
      "<mujoco><worldbody>"
      "<body pos='0 0 1'><freejoint/><geom type='box' size='0.1 0.1 0.1'/>"
      "<body pos='0.3 0 0'><joint type='ball'/><geom type='sphere' size='0.05'/>"
      "<body pos='0.2 0 0'><joint type='slide' axis='1 0 0' range='-0.1 0.1'/>"
      "<geom type='sphere' size='0.04'/></body></body></body>"
      "</worldbody></mujoco>");
  CheckBitIdentical(
      "fullinertia",
      "<mujoco><worldbody><body pos='0 0 1'>"
      "<inertial pos='0 0 0' mass='1' fullinertia='0.3 0.3 0.3 0.01 0.01 0.01'/>"
      "<joint type='hinge' axis='0 1 0'/></body></worldbody></mujoco>");
}

// NC1b: per-family bit-identity vs the XML oracle for the families the native
// slice grew to cover -- frames as containers, sites/cameras/lights, mocap,
// keyframe padding, and contact pair sorting.
static void TestNc1bBitIdentical() {
  // Frames flatten: nested frames accumulate their pose into every child
  // (geoms, joints, sites, and child bodies).
  CheckBitIdentical(
      "frame_nesting",
      "<mujoco><worldbody>"
      "<frame pos='1 0 0' euler='0 0 30'>"
      "  <geom type='box' size='0.1 0.1 0.1' pos='0.2 0 0'/>"
      "  <frame pos='0 1 0' euler='45 0 0'>"
      "    <body pos='0 0 0.5'>"
      "      <joint type='hinge' axis='0 1 0' pos='0.1 0 0'/>"
      "      <geom type='capsule' size='0.05 0.3'/>"
      "      <site name='fs' type='ellipsoid' size='0.02 0.03 0.04' pos='0 0.1 0'/>"
      "    </body>"
      "  </frame>"
      "</frame></worldbody></mujoco>");

  // Sites: every shape form plus fromto, on a body with an inertial frame.
  CheckBitIdentical(
      "sites",
      "<mujoco><worldbody><body pos='0 0 1'>"
      "<joint type='hinge' axis='0 1 0'/>"
      "<geom type='sphere' size='0.1'/>"
      "<site name='sa' pos='0.1 0 0'/>"
      "<site name='sb' type='box' size='0.02 0.03 0.04' euler='10 20 30'/>"
      "<site name='sc' type='capsule' fromto='0 0 0 0 0 0.3' size='0.01'/>"
      "</body></worldbody></mujoco>");

  // Cameras: fovy, focal/sensorsize intrinsics, and targetbody tracking.
  CheckBitIdentical(
      "cameras",
      "<mujoco><worldbody>"
      "<body name='tgt' pos='1 1 1'><geom type='sphere' size='0.1'/></body>"
      "<body pos='0 0 1'><joint type='hinge' axis='0 1 0'/>"
      "<geom type='sphere' size='0.1'/>"
      "<camera name='ca' pos='2 0 0' fovy='55' euler='0 90 0'/>"
      "<camera name='cb' pos='0 -2 0' resolution='640 480' sensorsize='0.01 0.008'"
      "        focal='0.02 0.02' principal='0.001 0.001'/>"
      "<camera name='cc' mode='targetbody' target='tgt' pos='0 0 3'/>"
      "</body></worldbody></mujoco>");

  // Lights: directional and spot, with a targetbody.
  CheckBitIdentical(
      "lights",
      "<mujoco><worldbody>"
      "<body name='tgt2' pos='1 0 0'><geom type='sphere' size='0.1'/></body>"
      "<light name='la' directional='true' pos='0 0 3' dir='0 0 -1'/>"
      "<body pos='0 0 1'><joint type='hinge' axis='0 1 0'/>"
      "<geom type='sphere' size='0.1'/>"
      "<light name='lb' pos='0 0 1' dir='1 0 -1' mode='targetbody' target='tgt2'"
      "       diffuse='0.8 0.2 0.2' cutoff='30' attenuation='1 0.1 0.01'/>"
      "</body></worldbody></mujoco>");

  // Mocap bodies: mocapid assignment + nmocap + body_mocapid.
  CheckBitIdentical(
      "mocap",
      "<mujoco><worldbody>"
      "<body name='m1' mocap='true' pos='0.5 0 1' euler='0 0 45'/>"
      "<body name='m2' mocap='true' pos='-0.5 0 1'/>"
      "<body pos='0 0 1'><freejoint/><geom type='box' size='0.1 0.1 0.1'/></body>"
      "</worldbody></mujoco>");

  // Keyframes: explicit qpos/qvel plus padded-to-qpos0 defaults and mocap pose.
  CheckBitIdentical(
      "keyframes",
      "<mujoco><worldbody>"
      "<body name='mk' mocap='true' pos='0.2 0.3 0.4'/>"
      "<body pos='0 0 1'><freejoint/><geom type='box' size='0.1 0.1 0.1'/>"
      "<body pos='0.3 0 0'><joint type='hinge' axis='0 1 0'/>"
      "<geom type='sphere' size='0.05'/></body></body>"
      "</worldbody>"
      "<keyframe>"
      "<key name='full' qpos='0 0 1 1 0 0 0 0.5' qvel='0 0 0 0 0 0 0.1'/>"
      "<key name='empty'/>"
      "<key name='timed' time='1.5'/>"
      "</keyframe></mujoco>");

  // Contact pairs + excludes: compile-time stable-sort by body signature and id
  // reassignment (authored out of signature order to exercise the sort).
  CheckBitIdentical(
      "pairs_excludes",
      "<mujoco><worldbody>"
      "<body name='b1' pos='0 0 1'><freejoint/><geom name='g1' type='sphere' size='0.1'/></body>"
      "<body name='b2' pos='0.3 0 1'><joint type='hinge' axis='0 1 0'/>"
      "<geom name='g2' type='sphere' size='0.08'/></body>"
      "<body name='b3' pos='0.6 0 1'><joint type='slide' axis='1 0 0'/>"
      "<geom name='g3' type='sphere' size='0.05'/></body>"
      "</worldbody><contact>"
      "<pair geom1='g2' geom2='g3'/>"
      "<pair geom1='g1' geom2='g3' condim='1' friction='2 2 0.01 0.001 0.001'/>"
      "<pair geom1='g1' geom2='g2' solref='0.01 1'/>"
      "<exclude body1='b2' body2='b3'/>"
      "<exclude body1='b1' body2='b3'/>"
      "</contact></mujoco>");
}

int main() {
  TestNativePurity();
  TestGateUnsupported();
  TestGateAdmitsRigidBody();
  TestAutoLoudFallback();
  TestAutoNativePath();
  TestReportShape();
  TestNc1BitIdentical();
  TestNc1bBitIdentical();
  TestAllocatorRoundTrip();
  TestMjuuSmoke();
  std::printf("test_native: %d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
