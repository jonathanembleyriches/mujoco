// ProtoSpec Studio SE2 windowless MOVABILITY AUDIT.
//
// The gizmo's job is simple to state and easy to get subtly wrong: dragging an
// element must move it. This audit enforces that invariant for EVERY body / geom
// / site / camera / light in a corpus of models, driving each through the real
// gizmo core path (BuildDragFrame -> ApplyTranslate -> recompile) and asserting
// the element's COMPILED world position moved by the applied delta. Any element
// that does not move -- and is not on an explicit, documented expected-immovable
// list -- is a test failure.
//
// This is the class-catching test. The motivating bug: geoms/sites authored with
// `fromto` store their endpoints in the `shape` variant and leave pos/orient
// unset; the compiler DERIVES pos/quat from the endpoints and ignores authored
// pos/quat (cpp/compile/build.cc geom/site Compile). A gizmo that edits pos is a
// silent no-op on 14 of the humanoid's 29 geoms. The audit would have caught it
// on day one; it exists so the whole class stays caught.
//
// Corpus: inline fixtures (deterministic, dependency-free -- they always run and
// cover every element class the audit reasons about) PLUS real corpus models when
// the vendored MuJoCo model dir is present (humanoid = the fromto-heavy / class /
// camera case; car / arm26 / slider_crank = site-heavy; cards = fromto + camera).

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "binding.h"
#include "compile.h"
#include "editor/transform_math.h"
#include "editor/undo.h"
#include "mjcf.h"
#include "protospec/classes.h"
#include "protospec/detail.h"
#include "protospec/traversal.h"
#include "types.h"

namespace mj = ps::mjcf;
namespace sdkd = ps::sdk::detail;
using namespace ps::studio;

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

// ------------------------------------------------------------------------- //
// A compiled snapshot of a tree (owns mjData at qpos0).
// ------------------------------------------------------------------------- //
struct Compiled {
  mj::Compiled c;
  mjData* d = nullptr;
  Compiled() = default;
  Compiled(const Compiled&) = delete;
  Compiled& operator=(const Compiled&) = delete;
  Compiled(Compiled&& o) noexcept : c(std::move(o.c)), d(o.d) { o.d = nullptr; }
  Compiled& operator=(Compiled&& o) noexcept {
    if (this != &o) {
      if (d) mj_deleteData(d);
      c = std::move(o.c);
      d = o.d;
      o.d = nullptr;
    }
    return *this;
  }
  ~Compiled() {
    if (d) mj_deleteData(d);
  }
  bool ok() const { return c.ok() && d; }
};

static Compiled CompileAt(mj::Model& tree, const std::string& base_dir) {
  Compiled out;
  mj::CompileOptions opts;
  opts.path = mj::CompilePath::Auto;
  opts.base_dir = base_dir;
  out.c = mj::Compile(tree, opts);
  if (!out.c.ok()) return out;
  out.d = mj_makeData(out.c.model.get());
  mj_resetData(out.c.model.get(), out.d);
  mj_forward(out.c.model.get(), out.d);
  return out;
}

// Compiled world position of `serial` via the Binding, for the element kinds that
// have one (body/geom/site/camera/light). Returns false when the serial has no
// spatial compiled counterpart.
static bool WorldPos(const Compiled& s, std::uint64_t serial, double out[3]) {
  for (const mj::Binding::Entry& e : s.c.binding.entries()) {
    if (e.serial != serial || e.id < 0) continue;
    const mjData* d = s.d;
    switch (e.etype) {
      case mj::ElementType::Geom:
        for (int i = 0; i < 3; ++i) out[i] = d->geom_xpos[3 * e.id + i];
        return true;
      case mj::ElementType::Body:
        for (int i = 0; i < 3; ++i) out[i] = d->xpos[3 * e.id + i];
        return true;
      case mj::ElementType::Site:
        for (int i = 0; i < 3; ++i) out[i] = d->site_xpos[3 * e.id + i];
        return true;
      case mj::ElementType::Camera:
        for (int i = 0; i < 3; ++i) out[i] = d->cam_xpos[3 * e.id + i];
        return true;
      case mj::ElementType::Light:
        for (int i = 0; i < 3; ++i) out[i] = d->light_xpos[3 * e.id + i];
        return true;
      default:
        break;
    }
  }
  return false;
}

// ------------------------------------------------------------------------- //
// Element enumeration: every spatial element (with metadata) in a tree.
// ------------------------------------------------------------------------- //
struct ElemInfo {
  std::uint64_t serial = 0;
  mj::ElementType type = mj::ElementType::Model;
  std::string name;
  bool is_fromto = false;     // geom/site authored (or class-inherited) with fromto
  bool is_worldbody = false;  // the implicit world body (fixed at origin)
  bool is_template = false;   // lives under a <default> class subtree (not a scene
                              // object; the compiler never binds it)
};

static const char* TypeName(mj::ElementType t) {
  switch (t) {
    case mj::ElementType::Body: return "Body";
    case mj::ElementType::Geom: return "Geom";
    case mj::ElementType::Site: return "Site";
    case mj::ElementType::Camera: return "Camera";
    case mj::ElementType::Light: return "Light";
    case mj::ElementType::Frame: return "Frame";
    default: return "?";
  }
}

// True when a geom/site resolves (directly or via its class) to a fromto shape.
template <class E>
static bool ElemFromTo(mj::Model& tree, E& e) {
  if constexpr (requires { e.shape; }) {
    ps::opt<mj::GeomShape> shape;
    if (e.shape) {
      shape = e.shape;
    } else if constexpr (ps::sdk::HasDefaultFamily<E>::value) {
      std::unique_ptr<E> eff = ps::sdk::Effective(tree, e, true);
      shape = eff->shape;
    }
    if (shape) return std::holds_alternative<mj::FromTo>(*shape);
  }
  return false;
}

static std::vector<ElemInfo> Enumerate(mj::Model& tree) {
  std::vector<ElemInfo> out;
  ps::sdk::ParentMap pm(tree);
  sdkd::WalkModelAll(tree, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if constexpr (requires { e.serial; }) {
      if constexpr (!std::is_same_v<E, mj::Model>) {
        const mj::ElementType t = mj::element_type_of<E>::value;
        const bool spatial =
            t == mj::ElementType::Body || t == mj::ElementType::Geom ||
            t == mj::ElementType::Site || t == mj::ElementType::Camera ||
            t == mj::ElementType::Light;
        if (!spatial) return;
        ElemInfo info;
        info.serial = e.serial;
        info.type = t;
        if (const std::string* n = sdkd::NameOf(e)) info.name = *n;
        info.is_fromto = ElemFromTo(tree, e);
        // Walk ancestors: a Default ancestor => class template; a parentless Body
        // is the implicit world body.
        const ps::sdk::ParentMap::Node* self = pm.Lookup(&e);
        const void* p = self ? self->parent : nullptr;
        // The implicit world body sits directly under the Model root: its parent
        // is not a Body/Frame (every real scene body descends from it). Its own
        // pose is fixed at the origin, so it has no compiled scene object to move.
        if (t == mj::ElementType::Body) {
          const ps::sdk::ParentMap::Node* pn = p ? pm.Lookup(p) : nullptr;
          const bool parent_spatial =
              pn && (pn->type == mj::ElementType::Body ||
                     pn->type == mj::ElementType::Frame);
          if (!parent_spatial) info.is_worldbody = true;
        }
        while (p) {
          const ps::sdk::ParentMap::Node* n = pm.Lookup(p);
          if (!n) break;
          if (n->type == mj::ElementType::Default) {
            info.is_template = true;
            break;
          }
          p = n->parent;
        }
        out.push_back(info);
      }
    }
  });
  return out;
}

// ------------------------------------------------------------------------- //
// The audit engine: for one model, translate-test every element.
// ------------------------------------------------------------------------- //
struct AuditStats {
  int elements = 0;
  int moved = 0;
  int immovable = 0;   // did not move AND not explicitly excused
  int excused = 0;     // movable in principle; this delta hit a model constraint
  int nonscene = 0;    // world body / class template: no compiled scene object
};

// An element that legitimately cannot move under a world translate, with the
// documented reason. The audit consults this before flagging a failure. After the
// fromto fix this list is EMPTY: every enumerated element must move. It exists as
// the sanctioned escape hatch, so a future genuinely-immovable class is recorded
// here with a reason rather than silently tolerated.
static const char* ExpectedImmovableReason(const std::string& /*model*/,
                                           const ElemInfo& /*e*/) {
  return nullptr;
}

static void AuditModel(const std::string& model_name, mj::Model& pristine,
                       const std::string& base_dir, AuditStats& st) {
  const double D[3] = {0.13, -0.07, 0.19};
  const double kEps = 1e-4;

  std::vector<ElemInfo> elems = Enumerate(pristine);
  for (const ElemInfo& info : elems) {
    ++st.elements;

    // Work on a serial-preserving clone so each element starts from the pristine
    // tree (the real editor snapshots via this same clone in BeginEdit).
    std::unique_ptr<mj::Model> clone = CloneWithSerials(pristine);
    Compiled before = CompileAt(*clone, base_dir);
    if (!before.ok()) {
      std::printf("  [%s] compile failed, skipping model\n", model_name.c_str());
      return;
    }
    double p0[3];
    if (!WorldPos(before, info.serial, p0)) {
      // No compiled scene object. This is ONLY legitimate for the implicit world
      // body (fixed at origin) and <default>-class templates (which the compiler
      // never binds). Any OTHER unbound element is a real scene element the gizmo
      // could not track -- a failure, not a silent skip.
      if (info.is_worldbody || info.is_template) {
        ++st.nonscene;
        continue;
      }
      std::printf("  [%s] UNBOUND-SCENE-ELEMENT %s '%s' (serial %llu)\n",
                  model_name.c_str(), TypeName(info.type), info.name.c_str(),
                  static_cast<unsigned long long>(info.serial));
      ++st.immovable;
      CHECK(false && "real scene element has no compiled binding");
      continue;
    }

    // Drive the real gizmo core path: build the drag frame from the compiled
    // snapshot, apply the world-space translate onto the tree.
    DragFrame f = BuildDragFrame(before.c.model.get(), before.d,
                                 before.c.binding, *clone, info.serial);
    CHECK(f.valid);
    ApplyTranslate(*clone, info.serial, f, D);

    Compiled after = CompileAt(*clone, base_dir);
    if (after.ok()) {
      double p1[3];
      CHECK(WorldPos(after, info.serial, p1));
      const bool moved = std::fabs(p1[0] - p0[0] - D[0]) < kEps &&
                         std::fabs(p1[1] - p0[1] - D[1]) < kEps &&
                         std::fabs(p1[2] - p0[2] - D[2]) < kEps;
      if (moved) {
        ++st.moved;
        continue;
      }
    } else {
      // The edit produced a model MuJoCo rejects (e.g. a spatial-tendon wrap that
      // no longer has a valid solution once a routing site/body is displaced this
      // far). That is a LOUD failure the editor surfaces in Diagnostics while
      // keeping the last good compile (DR-S1) -- categorically different from the
      // silent fromto no-op. Confirm the element is still movable in principle
      // with a small delta before deciding anything: only a genuine
      // move-does-nothing is a bug.
      std::unique_ptr<mj::Model> c2 = CloneWithSerials(pristine);
      Compiled b2 = CompileAt(*c2, base_dir);
      double q0[3];
      if (b2.ok() && WorldPos(b2, info.serial, q0)) {
        const double d2[3] = {0.01, 0.006, 0.0};
        DragFrame f2 = BuildDragFrame(b2.c.model.get(), b2.d, b2.c.binding, *c2,
                                      info.serial);
        ApplyTranslate(*c2, info.serial, f2, d2);
        Compiled a2 = CompileAt(*c2, base_dir);
        double q1[3];
        if (a2.ok() && WorldPos(a2, info.serial, q1) &&
            std::fabs(q1[0] - q0[0] - d2[0]) < kEps &&
            std::fabs(q1[1] - q0[1] - d2[1]) < kEps &&
            std::fabs(q1[2] - q0[2] - d2[2]) < kEps) {
          ++st.excused;
          std::printf(
              "  [%s] EXCUSED %s '%s': movable (small delta ok); full delta "
              "rejected by compiler [%s]\n",
              model_name.c_str(), TypeName(info.type), info.name.c_str(),
              after.c.report.errors.empty()
                  ? "?"
                  : after.c.report.errors.front().Render().c_str());
          continue;
        }
      }
    }
    double p1[3] = {p0[0], p0[1], p0[2]};
    if (after.ok()) WorldPos(after, info.serial, p1);
    const bool moved = false;

    const char* reason = ExpectedImmovableReason(model_name, info);
    if (reason) {
      ++st.excused;
      std::printf("  [%s] EXCUSED-IMMOVABLE %s '%s' (%s)\n", model_name.c_str(),
                  TypeName(info.type), info.name.c_str(), reason);
      continue;
    }
    ++st.immovable;
    std::printf(
        "  [%s] IMMOVABLE %s '%s'%s  moved (%.4f %.4f %.4f) expected "
        "(%.2f %.2f %.2f)\n",
        model_name.c_str(), TypeName(info.type), info.name.c_str(),
        info.is_fromto ? " [fromto]" : "", p1[0] - p0[0], p1[1] - p0[1],
        p1[2] - p0[2], D[0], D[1], D[2]);
    CHECK(moved && "element did not move under translate");
  }
  std::printf("  [%s] %d elements: %d moved, %d immovable, %d excused, %d nonscene\n",
              model_name.c_str(), st.elements, st.moved, st.immovable, st.excused,
              st.nonscene);
}

// ------------------------------------------------------------------------- //
// Inline fixtures: one per element class the audit reasons about. These always
// run (no corpus dependency) and pin the behaviour deterministically.
// ------------------------------------------------------------------------- //
static std::unique_ptr<mj::Model> Parse(const char* xml) {
  mj::io::ParseResult r = mj::io::ParseMjcfString(xml, "movability");
  if (!r.ok()) {
    std::printf("FATAL parse:\n");
    for (const auto& e : r.errors) std::printf("  %s\n", e.Render().c_str());
  }
  return std::move(r.model);
}

// The fromto class in isolation: a capsule limb, a fromto site, plus explicit and
// mesh-free primitives, on rotated bodies. Pre-fix, the two fromto elements are
// immovable; post-fix all move.
static const char* kFromToXml = R"(
<mujoco>
  <worldbody>
    <body name="b" pos="0.5 0.2 1" euler="0 0 35">
      <freejoint/>
      <geom name="limb" type="capsule" size="0.05" fromto="0 0 0 0.3 0.1 0.2"/>
      <geom name="ball" type="sphere" size="0.1" pos="0.2 0 0"/>
      <site name="ftsite" type="capsule" size="0.02" fromto="0 0 0 0 0 0.2"/>
      <site name="dot" pos="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>)";

// Class-inherited fromto: the shape variant lives on the default class, so the
// geom's own `shape` is unset and must be resolved via Effective.
static const char* kClassFromToXml = R"(
<mujoco>
  <default>
    <default class="limb">
      <geom type="capsule" size="0.04" fromto="0 0 0 0 0 0.4"/>
    </default>
  </default>
  <worldbody>
    <body name="b" pos="0 0 1" euler="0 20 0">
      <geom name="inh" class="limb"/>
    </body>
  </worldbody>
</mujoco>)";

// A <frame> (absent from mjModel): moving the frame must move its child geom, and
// moving the child (whose parent pose composes the frame) must move it too.
static const char* kFrameXml = R"(
<mujoco>
  <worldbody>
    <body name="b" pos="0 0 0">
      <frame name="fr" pos="1 0 0.5" euler="0 0 90">
        <geom name="fg" type="box" size="0.1 0.1 0.1" pos="0.2 0 0"/>
        <geom name="flimb" type="capsule" size="0.03" fromto="0 0 0 0.2 0 0"/>
      </frame>
    </body>
  </worldbody>
</mujoco>)";

// A mocap body (xpos at qpos0 comes from mocap_pos, seeded from body pos) and a
// welded child geom.
static const char* kMocapXml = R"(
<mujoco>
  <worldbody>
    <body name="mb" mocap="true" pos="0.3 0.4 0.5">
      <geom name="mg" type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>)";

// Worldbody-direct geom + camera + light (no parent body -> P is identity).
static const char* kWorldElemsXml = R"(
<mujoco>
  <worldbody>
    <geom name="floorbox" type="box" size="1 1 0.1" pos="0 0 0"/>
    <camera name="cam" pos="0 -3 1" euler="60 0 0"/>
    <light name="lamp" pos="0 0 3" dir="0 0 -1"/>
    <body name="b" pos="0 0 1">
      <camera name="bcam" pos="0 -1 0"/>
      <light name="blight" pos="0 0 1" dir="0 0 -1"/>
    </body>
  </worldbody>
</mujoco>)";

// Frame-child movability needs a per-child check (a frame has no compiled entity).
static void TestFrameChildMoves() {
  std::unique_ptr<mj::Model> pristine = Parse(kFrameXml);
  const double D[3] = {0.13, -0.07, 0.19};

  // Move the FRAME; its child geom must move by D.
  const mj::Frame* fr = ps::sdk::Find<mj::Frame>(*pristine, "fr");
  const mj::Geom* fg = ps::sdk::Find<mj::Geom>(*pristine, "fg");
  CHECK(fr && fg);
  std::unique_ptr<mj::Model> clone = CloneWithSerials(*pristine);
  Compiled before = CompileAt(*clone, "");
  CHECK(before.ok());
  double p0[3];
  CHECK(WorldPos(before, fg->serial, p0));
  DragFrame f = BuildDragFrame(before.c.model.get(), before.d, before.c.binding,
                               *clone, fr->serial);
  CHECK(f.valid);
  ApplyTranslate(*clone, fr->serial, f, D);
  Compiled after = CompileAt(*clone, "");
  CHECK(after.ok());
  double p1[3];
  CHECK(WorldPos(after, fg->serial, p1));
  for (int i = 0; i < 3; ++i) CHECK(std::fabs(p1[i] - p0[i] - D[i]) < 1e-4);
}

// ------------------------------------------------------------------------- //
// fromto rotate + scale spot-checks (the audit translates; rotate/scale are
// checked here on the fromto class specifically).
// ------------------------------------------------------------------------- //
static void GeomEndpoints(mj::Model& tree, const char* name, double from[3],
                          double to[3]) {
  const mj::Geom* g = ps::sdk::Find<mj::Geom>(tree, name);
  const mj::FromTo* ft = std::get_if<mj::FromTo>(&*g->shape);
  for (int i = 0; i < 3; ++i) {
    from[i] = ft->fromto[i];
    to[i] = ft->fromto[i + 3];
  }
}

static void TestFromToRotate() {
  std::unique_ptr<mj::Model> pristine = Parse(kFromToXml);
  std::unique_ptr<mj::Model> clone = CloneWithSerials(*pristine);
  Compiled before = CompileAt(*clone, "");
  CHECK(before.ok());
  const std::uint64_t g = ps::sdk::Find<mj::Geom>(*clone, "limb")->serial;
  double p0[3];
  CHECK(WorldPos(before, g, p0));

  DragFrame f = BuildDragFrame(before.c.model.get(), before.d, before.c.binding,
                               *clone, g);
  CHECK(f.valid);
  CHECK(f.is_fromto);
  // Rotate 90 deg about world z. The compiled MIDPOINT (world pos) is invariant
  // (rotation pivots on the capsule centre); endpoints rotate about the midpoint.
  double mid_before[3];
  {
    double fr[3], to[3];
    GeomEndpoints(*clone, "limb", fr, to);
    for (int i = 0; i < 3; ++i) mid_before[i] = 0.5 * (fr[i] + to[i]);
  }
  const double axis[3] = {0, 0, 1};
  ApplyRotate(*clone, g, f, axis, 1.5707963267948966);
  // Still fromto-authored (no pos/orient written).
  const mj::Geom* gp = ps::sdk::Find<mj::Geom>(*clone, "limb");
  CHECK(gp->shape.has_value());
  CHECK(std::holds_alternative<mj::FromTo>(*gp->shape));
  CHECK(!gp->pos.has_value());
  double mid_after[3];
  {
    double fr[3], to[3];
    GeomEndpoints(*clone, "limb", fr, to);
    for (int i = 0; i < 3; ++i) mid_after[i] = 0.5 * (fr[i] + to[i]);
  }
  for (int i = 0; i < 3; ++i) CHECK(std::fabs(mid_after[i] - mid_before[i]) < 1e-9);

  Compiled after = CompileAt(*clone, "");
  CHECK(after.ok());
  double p1[3];
  CHECK(WorldPos(after, g, p1));
  for (int i = 0; i < 3; ++i) CHECK(std::fabs(p1[i] - p0[i]) < 1e-4);  // pos fixed
}

// A rotation about the fromto axis itself is a documented no-op (axisymmetric):
// the endpoints are unchanged.
static void TestFromToTwistIsNoOp() {
  std::unique_ptr<mj::Model> pristine = Parse(kFromToXml);
  std::unique_ptr<mj::Model> clone = CloneWithSerials(*pristine);
  Compiled before = CompileAt(*clone, "");
  CHECK(before.ok());
  const std::uint64_t g = ps::sdk::Find<mj::Geom>(*clone, "limb")->serial;
  DragFrame f = BuildDragFrame(before.c.model.get(), before.d, before.c.binding,
                               *clone, g);
  CHECK(f.valid && f.is_fromto);
  double fr0[3], to0[3];
  GeomEndpoints(*clone, "limb", fr0, to0);
  // world_quat's local z is the capsule axis; twist about it in WORLD space.
  double local_z[3] = {0, 0, 1};
  double world_axis[3];
  QuatRotate(f.world_quat, local_z, world_axis);
  ApplyRotate(*clone, g, f, world_axis, 0.9);
  double fr1[3], to1[3];
  GeomEndpoints(*clone, "limb", fr1, to1);
  for (int i = 0; i < 3; ++i) {
    CHECK(std::fabs(fr1[i] - fr0[i]) < 1e-9);
    CHECK(std::fabs(to1[i] - to0[i]) < 1e-9);
  }
}

static void TestFromToScale() {
  std::unique_ptr<mj::Model> pristine = Parse(kFromToXml);
  std::unique_ptr<mj::Model> clone = CloneWithSerials(*pristine);
  Compiled before = CompileAt(*clone, "");
  CHECK(before.ok());
  const std::uint64_t g = ps::sdk::Find<mj::Geom>(*clone, "limb")->serial;
  // half-length before (compiled geom size[1] for a capsule).
  int gid = -1;
  for (const auto& e : before.c.binding.entries())
    if (e.serial == g && e.etype == mj::ElementType::Geom) gid = e.id;
  CHECK(gid >= 0);
  const double half0 = before.c.model->geom_size[3 * gid + 1];
  const double rad0 = before.c.model->geom_size[3 * gid + 0];

  ScaleBase base = BuildScaleBase(*clone, g);
  CHECK(base.valid);
  CHECK(base.is_fromto);
  // Axis (local z) x2 -> half-length doubles; radius (x) x2 -> radius doubles.
  const double factor[3] = {2.0, 2.0, 2.0};
  ApplyScale(*clone, g, base, factor);
  Compiled after = CompileAt(*clone, "");
  CHECK(after.ok());
  gid = -1;
  for (const auto& e : after.c.binding.entries())
    if (e.serial == g && e.etype == mj::ElementType::Geom) gid = e.id;
  CHECK(gid >= 0);
  const double half1 = after.c.model->geom_size[3 * gid + 1];
  const double rad1 = after.c.model->geom_size[3 * gid + 0];
  CHECK(std::fabs(half1 - 2 * half0) < 1e-6);
  CHECK(std::fabs(rad1 - 2 * rad0) < 1e-6);
  // Still fromto-authored.
  CHECK(std::holds_alternative<mj::FromTo>(*ps::sdk::Find<mj::Geom>(*clone, "limb")->shape));
}

// ------------------------------------------------------------------------- //
// Corpus driver (optional: runs when the vendored model dir is present).
// ------------------------------------------------------------------------- //
static std::string CorpusRoot() {
  std::string root =
      "C:/Users/jonat/Documents/Unreal Projects/url_proj/Plugins/"
      "UnrealRoboticsLab/third_party/MuJoCo/src";
  if (const char* env = std::getenv("PROTOSPEC_CORPUS")) root = env;
  return root;
}

static void AuditCorpusFile(const std::string& rel, AuditStats& total) {
  const std::filesystem::path path =
      std::filesystem::path(CorpusRoot()) / "model" / rel;
  if (!std::filesystem::exists(path)) {
    std::printf("  corpus absent: %s (skipped)\n", rel.c_str());
    return;
  }
  mj::io::ParseResult r = mj::io::ParseMjcfFile(path.string());
  if (!r.ok()) {
    std::printf("  corpus parse failed: %s (skipped)\n", rel.c_str());
    return;
  }
  AuditStats st;
  AuditModel(rel, *r.model, path.parent_path().string(), st);
  total.elements += st.elements;
  total.moved += st.moved;
  total.immovable += st.immovable;
  total.excused += st.excused;
  total.nonscene += st.nonscene;
}

// ------------------------------------------------------------------------- //
int main() {
  std::printf("=== inline fixtures ===\n");
  AuditStats inline_st;
  struct { const char* name; const char* xml; } fixtures[] = {
      {"fixture:fromto", kFromToXml},
      {"fixture:class-fromto", kClassFromToXml},
      {"fixture:frame", kFrameXml},
      {"fixture:mocap", kMocapXml},
      {"fixture:world-elems", kWorldElemsXml},
  };
  for (const auto& fx : fixtures) {
    std::unique_ptr<mj::Model> m = Parse(fx.xml);
    if (!m) continue;
    AuditStats st;
    AuditModel(fx.name, *m, "", st);
    inline_st.elements += st.elements;
    inline_st.moved += st.moved;
    inline_st.immovable += st.immovable;
    inline_st.excused += st.excused;
    inline_st.nonscene += st.nonscene;
  }

  std::printf("=== per-class rotate / scale spot-checks ===\n");
  TestFrameChildMoves();
  TestFromToRotate();
  TestFromToTwistIsNoOp();
  TestFromToScale();

  std::printf("=== corpus models ===\n");
  AuditStats corpus_st;
  AuditCorpusFile("humanoid/humanoid.xml", corpus_st);
  AuditCorpusFile("car/car.xml", corpus_st);
  AuditCorpusFile("tendon_arm/arm26.xml", corpus_st);
  AuditCorpusFile("slider_crank/slider_crank.xml", corpus_st);
  AuditCorpusFile("cards/cards.xml", corpus_st);

  std::printf(
      "TOTAL inline: %d moved / %d immovable / %d excused / %d nonscene (of %d)\n",
      inline_st.moved, inline_st.immovable, inline_st.excused, inline_st.nonscene,
      inline_st.elements);
  std::printf(
      "TOTAL corpus: %d moved / %d immovable / %d excused / %d nonscene (of %d)\n",
      corpus_st.moved, corpus_st.immovable, corpus_st.excused, corpus_st.nonscene,
      corpus_st.elements);
  std::printf("%d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
