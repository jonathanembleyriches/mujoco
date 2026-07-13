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

// A model whose actuator is outside the NC2 native slice: a dcmotor (still
// gated -- its lowering/state derivation is not lifted). The gate routes the
// whole model to the XML fallback. (Motors, position/velocity/etc. are native as
// of NC2, so the unsupported example must reach for a still-gated spelling.)
static const char* kUnsupported =
    "<mujoco>\n"
    "  <worldbody>\n"
    "    <body name='b1' pos='0 0 1'>\n"
    "      <joint name='j1' type='hinge' axis='0 1 0'/>\n"
    "      <geom name='g1' type='sphere' size='0.05'/>\n"
    "      <site name='s1' pos='0 0 0'/>\n"
    "    </body>\n"
    "  </worldbody>\n"
    "  <actuator><dcmotor name='m1' joint='j1' motorconst='0.1 0.1'"
  "    resistance='1'/></actuator>\n"
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
  bool saw_dcmotor = false;
  for (const auto& r : reasons) {
    if (r.feature == "dcmotor") saw_dcmotor = true;
    CHECK(r.count > 0);
  }
  CHECK(saw_dcmotor);

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
    for (const auto& f : rep.fields)
      for (const auto& e : f.examples)
        std::printf("    %s[%lld] xml=%.9g nat=%.9g\n", f.field.c_str(),
                    (long long)e.index, e.a, e.b);
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

// NC2 equality: connect/weld (body + site semantics), joint polycoef equality,
// anchor/relpose/torquescale packing, and class-default active/solref/solimp.
static void TestNc2Equality() {
  // Connect: body semantics (anchor) and site semantics.
  CheckBitIdentical(
      "eq_connect",
      "<mujoco><worldbody>"
      "<body name='b1' pos='0 0 1'><freejoint/><geom type='sphere' size='0.1'/>"
      "<site name='s1' pos='0.1 0 0'/></body>"
      "<body name='b2' pos='0.3 0 1'><joint type='hinge' axis='0 1 0'/>"
      "<geom type='sphere' size='0.08'/><site name='s2' pos='0 0 0'/></body>"
      "</worldbody><equality>"
      "<connect body1='b1' body2='b2' anchor='0.1 0.2 0.3'/>"
      "<connect site1='s1' site2='s2'/>"
      "</equality></mujoco>");

  // Weld: body semantics with relpose + torquescale, and site semantics.
  CheckBitIdentical(
      "eq_weld",
      "<mujoco><worldbody>"
      "<body name='b1' pos='0 0 1'><freejoint/><geom type='sphere' size='0.1'/>"
      "<site name='s1'/></body>"
      "<body name='b2' pos='0.3 0 1'><geom type='sphere' size='0.08'/>"
      "<site name='s2'/></body>"
      "</worldbody><equality>"
      "<weld body1='b1' body2='b2' relpose='0 0 0.1 1 0 0 0' torquescale='0.5'"
      "      anchor='0.05 0 0'/>"
      "<weld body1='b1'/>"
      "<weld site1='s1' site2='s2'/>"
      "</equality></mujoco>");

  // Joint equality: polycoef, plus a class-default solref/solimp/active.
  CheckBitIdentical(
      "eq_joint",
      "<mujoco><default><default class='eqc'>"
      "<equality solref='0.01 1' solimp='0.8 0.9 0.001 0.5 2'/>"
      "</default></default>"
      "<worldbody>"
      "<body pos='0 0 1'><joint name='j1' type='hinge' axis='0 1 0'/>"
      "<geom type='sphere' size='0.1'/>"
      "<body pos='0.2 0 0'><joint name='j2' type='slide' axis='1 0 0'/>"
      "<geom type='sphere' size='0.05'/></body></body>"
      "</worldbody><equality>"
      "<joint joint1='j1' joint2='j2' polycoef='0 1 0.5 0 0'/>"
      "<joint class='eqc' joint1='j1' polycoef='0.1 2 0 0 0' active='false'/>"
      "</equality></mujoco>");
}

// NC2 tendons: fixed (joint coefs), spatial (site/geom/pulley wraps with a side
// site + sphere->cylinder selection), limits/springlength, class defaults, and
// an equality-tendon coupling two tendons.
static void TestNc2Tendon() {
  // Fixed tendon: two joint coefficients, limited range, springlength.
  CheckBitIdentical(
      "ten_fixed",
      "<mujoco><worldbody>"
      "<body pos='0 0 1'><joint name='j1' type='slide' axis='1 0 0'/>"
      "<geom type='sphere' size='0.05'/>"
      "<body pos='0.2 0 0'><joint name='j2' type='slide' axis='1 0 0'/>"
      "<geom type='sphere' size='0.05'/></body></body>"
      "</worldbody><tendon>"
      "<fixed name='t1' limited='true' range='-0.5 0.5' stiffness='10'>"
      "<joint joint='j1' coef='1'/><joint joint='j2' coef='-1'/></fixed>"
      "</tendon></mujoco>");

  // Spatial tendon: sites, a sphere geom wrap with a side site, and a pulley.
  CheckBitIdentical(
      "ten_spatial",
      "<mujoco><worldbody>"
      "<site name='s0' pos='0 0 1'/>"
      "<geom name='wrapg' type='sphere' size='0.1' pos='0.2 0 1' contype='0' conaffinity='0'/>"
      "<site name='sside' pos='0.2 0.1 1'/>"
      "<body pos='0.4 0 1'><joint name='jj' type='hinge' axis='0 1 0'/>"
      "<geom type='sphere' size='0.05'/><site name='s1' pos='0 0 0'/></body>"
      "</worldbody><tendon>"
      "<spatial name='ts' width='0.005' rgba='1 0 0 1'>"
      "<site site='s0'/><geom geom='wrapg' sidesite='sside'/><site site='s1'/>"
      "</spatial></tendon></mujoco>");

  // Spatial tendon with a pulley branch and a class default.
  CheckBitIdentical(
      "ten_pulley",
      "<mujoco><default><default class='tc'>"
      "<tendon width='0.004' stiffness='5 0 0' rgba='0 1 0 1'/></default></default>"
      "<worldbody>"
      "<site name='a' pos='0 0 1'/><site name='b' pos='0.3 0 1'/>"
      "<body pos='0.5 0 1'><joint type='slide' axis='1 0 0'/>"
      "<geom type='sphere' size='0.05'/><site name='c' pos='0 0 0'/></body>"
      "</worldbody><tendon>"
      "<spatial class='tc' name='tp'>"
      "<site site='a'/><site site='b'/><pulley divisor='2'/>"
      "<site site='b'/><site site='c'/>"
      "</spatial></tendon></mujoco>");

  // Equality coupling two fixed tendons (polycoef), tendon ref resolution.
  CheckBitIdentical(
      "eq_tendon",
      "<mujoco><worldbody>"
      "<body pos='0 0 1'><joint name='j1' type='slide' axis='1 0 0'/>"
      "<geom type='sphere' size='0.05'/></body>"
      "<body pos='0.3 0 1'><joint name='j2' type='slide' axis='1 0 0'/>"
      "<geom type='sphere' size='0.05'/></body>"
      "</worldbody><tendon>"
      "<fixed name='ta'><joint joint='j1' coef='1'/></fixed>"
      "<fixed name='tb'><joint joint='j2' coef='1'/></fixed>"
      "</tendon><equality>"
      "<tendon tendon1='ta' tendon2='tb' polycoef='0 1 0 0 0'/>"
      "</equality></mujoco>");
}

// NC2 actuators: the lifted mjs_setTo* lowering per spelling, transmission
// resolution (joint/jointinparent/tendon/body), gear, tri-state limits,
// activation (na/actadr) for stateful dyntypes, inheritrange, and class defaults.
static void TestNc2Actuator() {
  // motor / position(kp,kv,timeconst) / velocity / damper / cylinder on joints.
  CheckBitIdentical(
      "act_basic",
      "<mujoco><worldbody>"
      "<body pos='0 0 1'><joint name='j1' type='hinge' axis='0 1 0' range='-1 1'/>"
      "<geom type='capsule' size='0.05 0.3'/>"
      "<body pos='0 0 -0.6'><joint name='j2' type='slide' axis='0 0 1'/>"
      "<geom type='sphere' size='0.05'/></body></body>"
      "</worldbody><actuator>"
      "<motor joint='j1' gear='2 0 0 0 0 0' ctrlrange='-1 1' ctrllimited='true'/>"
      "<position joint='j1' kp='100' kv='10'/>"
      "<position joint='j2' kp='50' dampratio='1' timeconst='0.01'/>"
      "<velocity joint='j2' kv='5'/>"
      "<damper joint='j1' kv='3' ctrlrange='0 1'/>"
      "<cylinder joint='j2' timeconst='0.2' area='0.5'/>"
      "</actuator></mujoco>");

  // intvelocity (stateful, actlimited) + inheritrange from the joint range.
  CheckBitIdentical(
      "act_intvel_inherit",
      "<mujoco><worldbody>"
      "<body pos='0 0 1'><joint name='j1' type='hinge' axis='0 1 0' range='-1.5 1.5'/>"
      "<geom type='sphere' size='0.1'/></body>"
      "</worldbody><actuator>"
      "<intvelocity joint='j1' kp='20' actrange='-2 2'/>"
      "<position joint='j1' kp='30' inheritrange='1'/>"
      "</actuator></mujoco>");

  // adhesion (body transmission) + a class default for a position servo.
  CheckBitIdentical(
      "act_adhesion_default",
      "<mujoco><default><default class='srv'>"
      "<position kp='80' kv='4'/></default></default>"
      "<worldbody>"
      "<body name='grip' pos='0 0 1'><joint name='j1' type='slide' axis='0 0 1'/>"
      "<geom type='box' size='0.1 0.1 0.1'/></body>"
      "</worldbody><actuator>"
      "<position class='srv' joint='j1'/>"
      "<adhesion body='grip' ctrlrange='0 1' gain='2'/>"
      "</actuator></mujoco>");

  // tendon transmission (motor + position on a fixed tendon).
  CheckBitIdentical(
      "act_tendon",
      "<mujoco><worldbody>"
      "<body pos='0 0 1'><joint name='j1' type='slide' axis='1 0 0'/>"
      "<geom type='sphere' size='0.05'/></body>"
      "</worldbody><tendon>"
      "<fixed name='t1' limited='true' range='-1 1'><joint joint='j1' coef='1'/></fixed>"
      "</tendon><actuator>"
      "<motor tendon='t1' gear='3'/>"
      "<position tendon='t1' kp='50'/>"
      "</actuator></mujoco>");
}

// NC2 sensors: site/joint/tendon/actuator/body targets, frame sensors with an
// explicit objtype (+optional reftype), and the scalar energy/clock sensors --
// verifying type/objtype/objid/datatype/needstage/dim/sensor_adr fill.
static void TestNc2Sensor() {
  CheckBitIdentical(
      "sensors",
      "<mujoco><worldbody>"
      "<body name='b1' pos='0 0 1'><joint name='j1' type='hinge' axis='0 1 0' range='-1 1'/>"
      "<geom name='g1' type='capsule' size='0.05 0.3'/>"
      "<site name='s1' pos='0 0 0'/>"
      "<body name='b2' pos='0 0 -0.6'><joint name='jb' type='ball'/>"
      "<geom type='sphere' size='0.05'/><site name='s2'/></body></body>"
      "</worldbody><tendon>"
      "<fixed name='t1' limited='true' range='-1 1'><joint joint='j1' coef='1'/></fixed>"
      "</tendon><actuator>"
      "<motor name='m1' joint='j1'/>"
      "</actuator><sensor>"
      "<accelerometer site='s1'/><gyro site='s1'/><force site='s2'/>"
      "<jointpos joint='j1'/><jointvel joint='j1'/>"
      "<jointlimitpos joint='j1'/>"
      "<ballquat joint='jb'/><ballangvel joint='jb'/>"
      "<tendonpos tendon='t1'/><tendonlimitfrc tendon='t1'/>"
      "<actuatorpos actuator='m1'/><actuatorfrc actuator='m1'/>"
      "<subtreecom body='b1'/><subtreelinvel body='b1'/>"
      "<framepos objtype='site' objname='s1'/>"
      "<framequat objtype='body' objname='b2'/>"
      "<framepos objtype='site' objname='s2' reftype='site' refname='s1'/>"
      "<framexaxis objtype='geom' objname='g1'/>"
      "<clock/><e_potential/><e_kinetic/>"
      "</sensor></mujoco>");
}

// NC3 meshes: user-vertex meshes compiled bit-identically vs the XML oracle,
// covering the convex-hull graph (collision geom), the visual (no-hull) path,
// and the inertia frame that a mesh geom binds.
static void TestNc3Meshes() {
  // Tetrahedron as a collision mesh: needhull -> qhull graph + polygons + BVH.
  CheckBitIdentical(
      "mesh_tetra",
      "<mujoco><asset>"
      "<mesh name='tetra' vertex='0 0 0  1 0 0  0 1 0  0 0 1'/>"
      "</asset><worldbody><body><freejoint/>"
      "<geom type='mesh' mesh='tetra'/></body></worldbody></mujoco>");
  // Unit cube (8 verts): drives the hull, coplanar-face polygon merge, and the
  // principal-axis inertia frame.
  CheckBitIdentical(
      "mesh_box",
      "<mujoco><asset>"
      "<mesh name='box' vertex='-1 -1 -1  1 -1 -1  -1 1 -1  1 1 -1  "
      "-1 -1 1  1 -1 1  -1 1 1  1 1 1'/>"
      "</asset><worldbody><body pos='0 0 1'><freejoint/>"
      "<geom type='mesh' mesh='box' pos='0.1 0.2 0.3' euler='10 20 30'/>"
      "</body></worldbody></mujoco>");
  // Visual mesh (contype=conaffinity=0): no hull graph, but faces, normals,
  // polygons(none) and BVH still built; geom binds mesh aamm/frame.
  CheckBitIdentical(
      "mesh_visual",
      "<mujoco><asset>"
      "<mesh name='box' vertex='-1 -1 -1  1 -1 -1  -1 1 -1  1 1 -1  "
      "-1 -1 1  1 -1 1  -1 1 1  1 1 1'/>"
      "</asset><worldbody><body pos='0 0 1'>"
      "<geom type='mesh' mesh='box' contype='0' conaffinity='0'/>"
      "</body></worldbody></mujoco>");
  // Scaled + refpos/refquat transform, mesh with explicit density.
  CheckBitIdentical(
      "mesh_transform",
      "<mujoco><asset>"
      "<mesh name='tetra' scale='2 1 0.5' refpos='0.1 0 0' refquat='1 0 0 1' "
      "vertex='0 0 0  1 0 0  0 1 0  0 0 1'/>"
      "</asset><worldbody><body><freejoint/>"
      "<geom type='mesh' mesh='tetra' density='500'/></body></worldbody></mujoco>");
}

// NC4.1: the deprecated arena/constraint sizing. <size memory> selects the
// explicit-bytes narena branch (with power-of-two suffix parsing), <size nstack>
// the legacy stack branch, and njmax/nconmax become mjModel members feeding both.
static void TestNc4Size() {
  CheckBitIdentical(
      "size_memory_plain",
      "<mujoco><size memory='1048576'/><worldbody>"
      "<body><freejoint/><geom size='0.1'/></body></worldbody></mujoco>");
  CheckBitIdentical(
      "size_memory_suffix",
      "<mujoco><size memory='10M'/><worldbody>"
      "<body><joint type='hinge'/><geom size='0.1'/></body></worldbody></mujoco>");
  CheckBitIdentical(
      "size_memory_giga",
      "<mujoco><size memory='1G'/><worldbody>"
      "<body pos='0 0 1'><freejoint/><geom type='box' size='0.1 0.1 0.1'/>"
      "</body></worldbody></mujoco>");
  CheckBitIdentical(
      "size_nstack",
      "<mujoco><size nstack='100000'/><worldbody>"
      "<body><joint type='hinge'/><geom size='0.1'/></body></worldbody></mujoco>");
  CheckBitIdentical(
      "size_njmax_nconmax",
      "<mujoco><size njmax='300' nconmax='50'/><worldbody>"
      "<body><freejoint/><geom size='0.1'/></body></worldbody></mujoco>");

  // A well-formed <size memory> is admitted (native compiles it).
  {
    auto model = ParseOrDie(
        "<mujoco><size memory='5M'/><worldbody><geom size='1'/></worldbody></mujoco>");
    if (model) CHECK(compile::CollectUnsupportedFeatures(*model).empty());
  }

  // Several top-level <default> blocks all merge field-wise into the one `main`
  // (the norm once <include> fuses two files' top-level blocks). The SDK
  // DefaultIndex reproduces the parse-time snapshot (Roots()/MergeClassChain), so
  // a class-free element sees every block merged -- native-identical to the XML
  // reader's single `main`.
  CheckBitIdentical(
      "multi_block_default",
      "<mujoco><default><motor ctrlrange='-1 1' ctrllimited='true'/></default>"
      "<default><geom size='0.1'/></default>"
      "<worldbody><body><joint name='j' type='hinge'/><geom/></body></worldbody>"
      "<actuator><motor name='m' joint='j'/></actuator></mujoco>");
}

// NC4.2: custom fields. Numeric size-padding, text NUL packing, and tuple
// obj-ref resolution by (objtype, name) across the supported families.
static void TestNc4Custom() {
  // Numeric: authored size zero-pads the data tail; unset size takes the length.
  CheckBitIdentical(
      "custom_numeric_pad",
      "<mujoco><custom>"
      "<numeric name='gain' size='5' data='1 2 3'/>"
      "<numeric name='raw' data='0.5 0.25'/>"
      "</custom><worldbody><geom size='1'/></worldbody></mujoco>");
  // Text field.
  CheckBitIdentical(
      "custom_text",
      "<mujoco><custom>"
      "<text name='note' data='hello world'/>"
      "</custom><worldbody><geom size='1'/></worldbody></mujoco>");
  // Tuple referencing body/geom/site/joint/actuator.
  CheckBitIdentical(
      "custom_tuple",
      "<mujoco><worldbody>"
      "<body name='b1'><joint name='j1' type='hinge'/><geom name='g1' size='0.1'/>"
      "<site name='s1'/></body></worldbody>"
      "<actuator><motor name='a1' joint='j1'/></actuator>"
      "<custom><tuple name='grp'>"
      "<element objtype='body' objname='b1' prm='1'/>"
      "<element objtype='geom' objname='g1' prm='2'/>"
      "<element objtype='site' objname='s1'/>"
      "<element objtype='joint' objname='j1'/>"
      "<element objtype='actuator' objname='a1' prm='3.5'/>"
      "</tuple></custom></mujoco>");
  // xbody objtype resolves through the body id space.
  CheckBitIdentical(
      "custom_tuple_xbody",
      "<mujoco><worldbody><body name='b1'><freejoint/><geom size='0.1'/></body>"
      "</worldbody><custom><tuple name='t'>"
      "<element objtype='xbody' objname='b1'/></tuple></custom></mujoco>");

  // An out-of-scope tuple objtype (mesh) routes to the XML fallback.
  {
    auto model = ParseOrDie(
        "<mujoco><asset><mesh name='m' vertex='0 0 0 1 0 0 0 1 0 0 0 1'/></asset>"
        "<worldbody><body><freejoint/><geom type='mesh' mesh='m'/></body></worldbody>"
        "<custom><tuple name='t'>"
        "<element objtype='mesh' objname='m'/></tuple></custom></mujoco>");
    if (model) {
      auto reasons = compile::CollectUnsupportedFeatures(*model);
      bool saw = false;
      for (const auto& r : reasons)
        if (r.feature == "tuple.objtype") saw = true;
      CHECK(saw);
    }
  }

  // discardvisual="true" routes to the XML fallback (native keeps all geoms).
  {
    auto model = ParseOrDie(
        "<mujoco><compiler discardvisual='true'/><worldbody>"
        "<body><joint type='hinge'/><geom size='0.1'/>"
        "<geom type='box' size='0.1 0.1 0.1' contype='0' conaffinity='0' group='2'/>"
        "</body></worldbody></mujoco>");
    if (model) {
      auto reasons = compile::CollectUnsupportedFeatures(*model);
      bool saw = false;
      for (const auto& r : reasons)
        if (r.feature == "compiler.discardvisual") saw = true;
      CHECK(saw);
    }
  }
}

// NC4.3: native <replicate> expansion (tree-clone with accumulating pose + zero-
// padded name suffix). Covers offset/euler poses, sep, nesting, and the gates.
static void TestNc4Replicate() {
  // Simple count with offset: 3 copies of a free body along +x.
  CheckBitIdentical(
      "replicate_offset",
      "<mujoco><worldbody><replicate count='3' offset='0.5 0 0'>"
      "<body><freejoint/><geom type='box' size='0.1 0.1 0.1'/></body>"
      "</replicate></worldbody></mujoco>");
  // euler accumulation + named elements (suffix appended to authored names).
  CheckBitIdentical(
      "replicate_euler_named",
      "<mujoco><worldbody><replicate count='4' offset='0.2 0 0' euler='0 0 30'>"
      "<body name='b'><joint name='j' type='hinge'/><geom name='g' size='0.1'/>"
      "<site name='s'/></body></replicate></worldbody></mujoco>");
  // Custom separator + zero padding (count 12 -> two digits).
  CheckBitIdentical(
      "replicate_sep",
      "<mujoco><worldbody><replicate count='12' offset='0.1 0 0' sep='_'>"
      "<body><freejoint/><geom size='0.05'/></body>"
      "</replicate></worldbody></mujoco>");
  // Nested replicates compose (name = base + inner-suffix + outer-suffix).
  CheckBitIdentical(
      "replicate_nested",
      "<mujoco><worldbody>"
      "<replicate count='3' offset='0.3 0 0'>"
      "<replicate count='2' offset='0 0.3 0'>"
      "<body name='cell'><freejoint/><geom type='sphere' size='0.05'/></body>"
      "</replicate></replicate></worldbody></mujoco>");
  // Unnamed elements get auto-named from the original serial + suffix.
  CheckBitIdentical(
      "replicate_unnamed",
      "<mujoco><worldbody><replicate count='5' offset='0 0.2 0'>"
      "<body pos='0 0 1'><freejoint/><geom type='capsule' fromto='0 0 0 0 0 0.2' "
      "size='0.03'/></body></replicate></worldbody></mujoco>");

  // A replicate whose subtree carries a childclass routes to the XML fallback.
  {
    auto model = ParseOrDie(
        "<mujoco><default><default class='c'><geom size='0.2'/></default></default>"
        "<worldbody><replicate count='2' offset='1 0 0'>"
        "<body childclass='c'><freejoint/><geom/></body>"
        "</replicate></worldbody></mujoco>");
    if (model) {
      auto reasons = compile::CollectUnsupportedFeatures(*model);
      bool saw = false;
      for (const auto& r : reasons)
        if (r.feature == "replicate.childclass") saw = true;
      CHECK(saw);
    }
  }
  // A model-level tendon referencing a site inside a replicate routes to fallback.
  {
    auto model = ParseOrDie(
        "<mujoco><worldbody><replicate count='3' offset='0.1 0 0'>"
        "<site name='a'/><body name='bd'><freejoint/><geom size='0.05'/>"
        "<site name='b'/></body></replicate></worldbody>"
        "<tendon><spatial><site site='a'/><site site='b'/></spatial></tendon>"
        "</mujoco>");
    if (model) {
      auto reasons = compile::CollectUnsupportedFeatures(*model);
      bool saw = false;
      for (const auto& r : reasons)
        if (r.feature == "replicate.referencing_element") saw = true;
      CHECK(saw);
    }
  }
}

// NC4.4: ellipsoid fluid model. fluidshape='ellipsoid' fills geom_fluid from the
// geom semiaxes; fluidcoef overrides the default drag/lift coefficients.
static void TestNc4Fluid() {
  CheckBitIdentical(
      "fluid_sphere",
      "<mujoco><worldbody><body><freejoint/>"
      "<geom type='sphere' size='0.1' fluidshape='ellipsoid'/>"
      "</body></worldbody></mujoco>");
  CheckBitIdentical(
      "fluid_capsule_coef",
      "<mujoco><worldbody><body><freejoint/>"
      "<geom type='capsule' size='0.05 0.2' fluidshape='ellipsoid' "
      "fluidcoef='0.4 0.3 1.2 0.9 0.8'/></body></worldbody></mujoco>");
  CheckBitIdentical(
      "fluid_box",
      "<mujoco><worldbody><body pos='0 0 1'><freejoint/>"
      "<geom type='box' size='0.1 0.2 0.05' fluidshape='ellipsoid'/>"
      "</body></worldbody></mujoco>");
}

// NC5 Wave 1: direct <flex> (young=0, non-interpolated edge-only). Bit-identical
// across dim 1/2/3, rigid/centered/offset vertices, contact/edge overrides, and
// texcoord; gated sub-features route the whole model to the XML fallback.
static void TestNc5Flex() {
  // dim=2 triangle, centered, three free bodies.
  CheckBitIdentical(
      "flex_tri2d",
      "<mujoco><worldbody>"
      "<body name='b0'><freejoint/><geom size='0.05'/></body>"
      "<body name='b1' pos='1 0 0'><freejoint/><geom size='0.05'/></body>"
      "<body name='b2' pos='0 1 0'><freejoint/><geom size='0.05'/></body>"
      "</worldbody><deformable>"
      "<flex name='tri' dim='2' body='b0 b1 b2' element='0 1 2'/>"
      "</deformable></mujoco>");
  // dim=1 rope (two line elements sharing a vertex).
  CheckBitIdentical(
      "flex_line1d",
      "<mujoco><worldbody>"
      "<body name='a'><freejoint/><geom size='0.05'/></body>"
      "<body name='b' pos='1 0 0'><freejoint/><geom size='0.05'/></body>"
      "<body name='c' pos='2 0 0'><freejoint/><geom size='0.05'/></body>"
      "</worldbody><deformable>"
      "<flex name='rope' dim='1' body='a b c' element='0 1 1 2'/>"
      "</deformable></mujoco>");
  // dim=3 single tetrahedron.
  CheckBitIdentical(
      "flex_tet3d",
      "<mujoco><worldbody>"
      "<body name='b0'><freejoint/><geom size='0.05'/></body>"
      "<body name='b1' pos='1 0 0'><freejoint/><geom size='0.05'/></body>"
      "<body name='b2' pos='0 1 0'><freejoint/><geom size='0.05'/></body>"
      "<body name='b3' pos='0 0 1'><freejoint/><geom size='0.05'/></body>"
      "</worldbody><deformable>"
      "<flex name='jelly' dim='3' body='b0 b1 b2 b3' element='0 1 2 3'/>"
      "</deformable></mujoco>");
  // rigid: single body carrying every vertex (vertex offsets, non-centered).
  CheckBitIdentical(
      "flex_rigid",
      "<mujoco><worldbody>"
      "<body name='only' pos='0 0 1'><freejoint/><geom size='0.05'/></body>"
      "</worldbody><deformable>"
      "<flex name='patch' dim='2' body='only' radius='0.01' "
      "vertex='0 0 0  1 0 0  0 1 0' element='0 1 2'/>"
      "</deformable></mujoco>");
  // non-centered multi-body with a hinge dof (weld-based edge rigidity) plus
  // contact/edge child overrides.
  CheckBitIdentical(
      "flex_offset_overrides",
      "<mujoco><worldbody>"
      "<body name='p0'><freejoint/><geom size='0.05'/></body>"
      "<body name='p1' pos='1 0 0'><freejoint/><geom size='0.05'/></body>"
      "<body name='p2' pos='0 1 0'><joint type='hinge' axis='0 0 1'/>"
      "<geom size='0.05'/></body>"
      "</worldbody><deformable>"
      "<flex name='m' dim='2' body='p0 p1 p2' radius='0.02' "
      "vertex='0.1 0 0  0.1 0 0  -0.1 0 0' element='0 1 2'>"
      "<contact condim='1' internal='true' selfcollide='none' activelayers='2'/>"
      "<edge damping='0.1'/></flex>"
      "</deformable></mujoco>");
  // quad = two triangles sharing an edge, with per-vertex texcoord.
  CheckBitIdentical(
      "flex_quad_texcoord",
      "<mujoco><worldbody>"
      "<body name='v0' pos='0 0 1'><freejoint/><geom size='0.05'/></body>"
      "<body name='v1' pos='1 0 1'><freejoint/><geom size='0.05'/></body>"
      "<body name='v2' pos='1 1 1'><freejoint/><geom size='0.05'/></body>"
      "<body name='v3' pos='0 1 1'><freejoint/><geom size='0.05'/></body>"
      "</worldbody><deformable>"
      "<flex name='cloth' dim='2' body='v0 v1 v2 v3' element='0 1 2 0 2 3' "
      "texcoord='0 0 1 0 1 1 0 1'/>"
      "</deformable></mujoco>");

  // Gated sub-features: each routes the whole model to the XML fallback.
  auto gated = [](const char* label, const char* key, const char* xml) {
    auto model = ParseOrDie(xml);
    if (!model) { CHECK(false); return; }
    auto reasons = compile::CollectUnsupportedFeatures(*model);
    bool saw = false;
    for (const auto& r : reasons)
      if (r.feature == key) saw = true;
    if (!saw) std::printf("FAIL flex gate [%s]: missing %s\n", label, key);
    CHECK(saw);
  };
  gated("young", "flex.elasticity",
        "<mujoco><worldbody>"
        "<body name='b0'><freejoint/><geom size='0.05'/></body>"
        "<body name='b1' pos='1 0 0'><freejoint/><geom size='0.05'/></body>"
        "<body name='b2' pos='0 1 0'><freejoint/><geom size='0.05'/></body>"
        "<body name='b3' pos='0 0 1'><freejoint/><geom size='0.05'/></body>"
        "</worldbody><deformable>"
        "<flex name='j' dim='3' body='b0 b1 b2 b3' element='0 1 2 3'>"
        "<elasticity young='1000' poisson='0.3'/></flex></deformable></mujoco>");
  gated("equality_flex", "equality.flex",
        "<mujoco><worldbody>"
        "<body name='b0'><freejoint/><geom size='0.05'/></body>"
        "<body name='b1' pos='1 0 0'><freejoint/><geom size='0.05'/></body>"
        "<body name='b2' pos='0 1 0'><freejoint/><geom size='0.05'/></body>"
        "</worldbody><deformable>"
        "<flex name='tri' dim='2' body='b0 b1 b2' element='0 1 2'/>"
        "</deformable><equality><flex flex='tri'/></equality></mujoco>");
  gated("interpolated", "flex.interpolated",
        "<mujoco><worldbody>"
        "<body name='b0'><freejoint/><geom size='0.05'/></body>"
        "<body name='b1' pos='1 0 0'><freejoint/><geom size='0.05'/></body>"
        "<body name='b2' pos='0 1 0'><freejoint/><geom size='0.05'/></body>"
        "<body name='b3' pos='0 0 1'><freejoint/><geom size='0.05'/></body>"
        "</worldbody><deformable>"
        "<flex name='j' dim='3' dof='trilinear' cellcount='1 1 1' "
        "node='b0' body='b0 b1 b2 b3' element='0 1 2 3'/>"
        "</deformable></mujoco>");
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
  TestNc2Equality();
  TestNc2Tendon();
  TestNc2Actuator();
  TestNc2Sensor();
  TestNc3Meshes();
  TestNc4Size();
  TestNc4Custom();
  TestNc4Replicate();
  TestNc4Fluid();
  TestNc5Flex();
  TestAllocatorRoundTrip();
  TestMjuuSmoke();
  std::printf("test_native: %d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
