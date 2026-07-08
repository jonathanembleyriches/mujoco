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

// T0.4: forced NativePath reports UnsupportedNatively with a non-empty feature
// list; the gate names the families the model uses.
static void TestGateUnsupported() {
  auto model = ParseOrDie(kPendulum);
  if (!model) return;

  auto reasons = compile::CollectUnsupportedFeatures(*model);
  CHECK(!reasons.empty());
  bool saw_body = false, saw_geom = false, saw_joint = false;
  for (const auto& r : reasons) {
    if (r.feature == "body") saw_body = true;
    if (r.feature == "geom") saw_geom = true;
    if (r.feature == "joint") saw_joint = true;
    CHECK(r.count > 0);
  }
  CHECK(saw_body && saw_geom && saw_joint);

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

// T0.4: Auto falls back LOUDLY -- succeeds via XML but carries the native
// fallback reasons, and taken == XmlPath.
static void TestAutoLoudFallback() {
  auto model = ParseOrDie(kPendulum);
  if (!model) return;
  Compiled c = Compile(*model);  // Auto
  CHECK(c.ok());
  CHECK(c.model != nullptr);
  CHECK(c.report.taken == CompilePath::XmlPath);
  CHECK(!c.report.fallback_reasons.empty());   // the fallback is not silent
}

// T0.4: report shape stability -- ok() == errors.empty(), requested recorded.
static void TestReportShape() {
  auto model = ParseOrDie(kPendulum);
  if (!model) return;
  CompileOptions native;
  native.path = CompilePath::NativePath;
  Compiled c = Compile(*model, native);
  CHECK(c.report.requested == CompilePath::NativePath);
  CHECK(c.report.ok() == c.report.errors.empty());
  CHECK(!c.report.ok());  // native fails today
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

int main() {
  TestNativePurity();
  TestGateUnsupported();
  TestAutoLoudFallback();
  TestReportShape();
  TestAllocatorRoundTrip();
  TestMjuuSmoke();
  std::printf("test_native: %d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
