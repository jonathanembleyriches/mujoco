// Windowless tests of the three ease-of-use editor features (ps::studio, ours):
//
//   * SE0 undo/redo toast   -- UndoStack stores/retrieves the step label through
//                              push/undo/redo cycles; UndoToastText formats the
//                              "<verb>: <label> (<n> more)" string; a StatusToast
//                              carries it and fades on the pure host clock.
//   * SE2 hierarchy reveal  -- HierChainToSerial computes the ancestor chain
//                              (root..target inclusive) for an element deep in a
//                              body, one inside a family section, an unfindable
//                              serial, and a filter-hidden target.
//   * SE3 focus frame       -- ComputeFocusFrame's centre/radius/distance for a
//                              geom, a body (subtree union), a site, and a miss.
//
// Splices undo.cc / hierarchy_model.cc / focus_frame.cc so their functions are
// reachable directly; links libprotospec_core.a + libmujoco.so, same recipe as
// test_rigger_windowless. Driven by protospec/tests/test_editor_ux_windowless.py.

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "bridge.h"  // ps::mjcf::Compile / CompileOptions
#include "editor/editor_context.h"
#include "editor/focus_frame.h"
#include "editor/hierarchy_panel.h"
#include "mjcf.h"  // ps::mjcf::io::ParseMjcfString

// Splice the units under test (not in the linked core lib).
#include "editor/hierarchy_model.cc"
#include "editor/focus_frame.cc"
#include "editor/undo.cc"

namespace ps::studio {
namespace {

namespace mj = ps::mjcf;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

constexpr char kFixture[] = R"(<mujoco model="ux">
  <compiler autolimits="true"/>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <geom name="basegeom" type="box" size="0.1 0.1 0.1"/>
      <body name="arm" pos="0.2 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0" range="-30 45"/>
        <geom name="armgeom" type="capsule" fromto="0 0 0 0.4 0 0" size="0.03"/>
        <geom name="tipgeom" type="sphere" pos="0.4 0 0" size="0.05"/>
        <site name="tipsite" pos="0.4 0 0" size="0.01"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="m1" joint="hinge"/>
  </actuator>
</mujoco>
)";

struct Fixture {
  EditorContext ctx;
  ~Fixture() {
    if (ctx.sim_data) mj_deleteData(ctx.sim_data);
  }
};

bool BuildFixture(Fixture& f) {
  auto parsed = mj::io::ParseMjcfString(kFixture);
  if (!parsed.ok()) return false;
  f.ctx.tree = std::move(parsed.model);
  mj::CompileOptions opts;
  opts.path = mj::CompilePath::XmlPath;
  mj::Compiled c = mj::Compile(*f.ctx.tree, opts);
  if (!c.ok()) return false;
  mjModel* me = c.model.get();
  f.ctx.compiled = std::move(c);
  f.ctx.model_ready = true;
  f.ctx.sim_data = mj_makeData(me);
  mj_resetData(me, f.ctx.sim_data);
  mj_forward(me, f.ctx.sim_data);
  return true;
}

std::uint64_t SerialFor(const mj::Binding& b, mj::ElementType et, int id) {
  if (id < 0) return 0;
  for (const mj::Binding::Entry& e : b.entries()) {
    if (e.etype == et && e.id == id) return e.serial;
  }
  return 0;
}

// Serial by MuJoCo object type (mjOBJ_*), so families with several concrete
// element types (actuators) resolve without naming each one.
std::uint64_t SerialForObj(const mj::Binding& b, int objtype, int id) {
  if (id < 0) return 0;
  for (const mj::Binding::Entry& e : b.entries()) {
    if (mj::ObjTypeOf(e.etype) == objtype && e.id == id) return e.serial;
  }
  return 0;
}

// --- SE0: undo/redo toast --------------------------------------------------- //

int TestUndoToast() {
  Fixture f;
  CHECK(BuildFixture(f));

  // Labels survive push/undo/redo cycles. Two committed edits, then walk back
  // and forward reading each step's label off the stack.
  UndoStack us;
  us.BeginEdit(*f.ctx.tree);
  us.CommitEdit("range");
  us.BeginEdit(*f.ctx.tree);
  us.CommitEdit("rename");
  CHECK(us.undo_depth() == 2);

  std::unique_ptr<mj::Model> cur = CloneWithSerials(*f.ctx.tree);
  std::string lab;
  cur = us.Undo(std::move(cur), &lab);
  CHECK(lab == "rename");
  CHECK(us.undo_depth() == 1);  // one more undo left -> "(1 more)"
  cur = us.Undo(std::move(cur), &lab);
  CHECK(lab == "range");
  CHECK(us.undo_depth() == 0);
  cur = us.Redo(std::move(cur), &lab);
  CHECK(lab == "range");
  cur = us.Redo(std::move(cur), &lab);
  CHECK(lab == "rename");
  std::printf("  undo labels round-trip: ok\n");

  // Toast text formatting: verb, label, depth hint, empty-label fallback.
  CHECK(UndoToastText("Undo", "range", 3) == "Undo: range (3 more)");
  CHECK(UndoToastText("Undo", "range", 0) == "Undo: range");
  CHECK(UndoToastText("Redo", "", 0) == "Redo: edit");
  CHECK(UndoToastText("Redo", "", 1) == "Redo: edit (1 more)");
  std::printf("  toast text formatting: ok\n");

  // The StatusToast (windowless) carries the text and fades on the host clock.
  StatusToast t;
  t.Post(UndoToastText("Undo", "range", 2), StatusToast::Kind::Info, 100.0);
  CHECK(t.message == "Undo: range (2 more)");
  CHECK(t.Alpha(100.0) == 1.0f);        // fully opaque when just posted
  CHECK(t.Visible(100.0 + StatusToast::kHoldSeconds));
  CHECK(!t.Visible(100.0 + StatusToast::kHoldSeconds +
                   StatusToast::kFadeSeconds + 0.01));  // fully faded
  std::printf("  status toast carries text + fades: ok\n");
  return 0;
}

// --- SE2: hierarchy auto-reveal ancestor chain ------------------------------ //

const HierNode* FindNamed(const std::vector<const HierNode*>& chain,
                          const std::string& name) {
  for (const HierNode* n : chain) {
    if (n->name == name) return n;
  }
  return nullptr;
}

bool ChainHasSection(const std::vector<const HierNode*>& chain,
                     const std::string& tag) {
  for (const HierNode* n : chain) {
    if (n->is_section && n->tag == tag) return true;
  }
  return false;
}

int TestRevealChain() {
  Fixture f;
  CHECK(BuildFixture(f));
  const mjModel* m = f.ctx.compiled.model.get();
  const mj::Binding& b = f.ctx.compiled.binding;
  const HierNode model = BuildHierarchyModel(*f.ctx.tree);

  // A geom deep in the body tree: chain is root -> Body Tree -> base -> arm -> geom.
  const std::uint64_t tip_serial =
      SerialFor(b, mj::ElementType::Geom, mj_name2id(m, mjOBJ_GEOM, "tipgeom"));
  CHECK(tip_serial != 0);
  std::vector<const HierNode*> chain;
  CHECK(HierChainToSerial(model, tip_serial, chain));
  CHECK(chain.front() == &model);              // root first
  CHECK(chain.back()->serial == tip_serial);   // target last
  CHECK(ChainHasSection(chain, "Body Tree"));
  CHECK(FindNamed(chain, "base") != nullptr);
  CHECK(FindNamed(chain, "arm") != nullptr);
  std::printf("  chain to nested geom: ok\n");

  // A target inside a NON-body family section (Actuators) puts that section on
  // the path so the reveal can force it open.
  const std::uint64_t act_serial =
      SerialForObj(b, mjOBJ_ACTUATOR, mj_name2id(m, mjOBJ_ACTUATOR, "m1"));
  CHECK(act_serial != 0);
  chain.clear();
  CHECK(HierChainToSerial(model, act_serial, chain));
  CHECK(ChainHasSection(chain, "Actuators"));
  CHECK(chain.back()->serial == act_serial);
  std::printf("  chain into a family section: ok\n");

  // Unfindable serial / serial 0 -> false, chain left empty.
  chain.clear();
  CHECK(!HierChainToSerial(model, 0xDEADBEEFULL, chain));
  CHECK(chain.empty());
  chain.clear();
  CHECK(!HierChainToSerial(model, 0, chain));
  CHECK(chain.empty());
  std::printf("  unfindable serial: ok\n");

  // Filter-hidden target: absent from a filtered view, present in the full
  // model -- the exact condition the panel uses to clear the filter.
  const HierNode hidden = FilterHierarchy(model, "no-such-name");
  chain.clear();
  CHECK(!HierChainToSerial(hidden, tip_serial, chain));
  chain.clear();
  CHECK(HierChainToSerial(model, tip_serial, chain));
  std::printf("  filter-hidden target detectable: ok\n");
  return 0;
}

// --- SE3: focus frame ------------------------------------------------------- //

int TestFocusFrame() {
  Fixture f;
  CHECK(BuildFixture(f));
  const mjModel* m = f.ctx.compiled.model.get();
  const mjData* d = f.ctx.sim_data;
  const mj::Binding& b = f.ctx.compiled.binding;
  const double extent = m->stat.extent;

  // Geom: centre == geom_xpos, radius == geom_rbound, distance == the formula.
  const int gid = mj_name2id(m, mjOBJ_GEOM, "tipgeom");
  const std::uint64_t gser = SerialFor(b, mj::ElementType::Geom, gid);
  FocusFrame fg = ComputeFocusFrame(m, d, b, gser);
  CHECK(fg.ok);
  for (int k = 0; k < 3; ++k)
    CHECK(std::fabs(fg.center[k] - d->geom_xpos[3 * gid + k]) < 1e-9);
  CHECK(std::fabs(fg.radius - m->geom_rbound[gid]) < 1e-9);
  CHECK(std::fabs(fg.distance - std::max(fg.radius * 2.5, 0.15 * extent)) < 1e-9);
  std::printf("  geom frame: ok\n");

  // Body: the union sphere must enclose every subtree geom's bounding sphere.
  const int bid = mj_name2id(m, mjOBJ_BODY, "arm");
  const std::uint64_t bser = SerialFor(b, mj::ElementType::Body, bid);
  FocusFrame fb = ComputeFocusFrame(m, d, b, bser);
  CHECK(fb.ok);
  CHECK(fb.radius > 0.0);
  int checked = 0;
  for (int g = 0; g < m->ngeom; ++g) {
    // arm subtree = bodies bid.. whose parent chain includes bid.
    int body = m->geom_bodyid[g];
    bool in = false;
    for (int walk = body; walk > 0; walk = m->body_parentid[walk]) {
      if (walk == bid) { in = true; break; }
    }
    if (!in) continue;
    double dd = 0;
    for (int k = 0; k < 3; ++k) {
      double t = d->geom_xpos[3 * g + k] - fb.center[k];
      dd += t * t;
    }
    CHECK(std::sqrt(dd) + m->geom_rbound[g] <= fb.radius + 1e-9);
    ++checked;
  }
  CHECK(checked >= 2);  // armgeom + tipgeom
  std::printf("  body union frame encloses %d subtree geoms: ok\n", checked);

  // Site: point-like -> centre == site_xpos, default radius from extent.
  const int sid = mj_name2id(m, mjOBJ_SITE, "tipsite");
  const std::uint64_t sser = SerialFor(b, mj::ElementType::Site, sid);
  FocusFrame fs = ComputeFocusFrame(m, d, b, sser);
  CHECK(fs.ok);
  for (int k = 0; k < 3; ++k)
    CHECK(std::fabs(fs.center[k] - d->site_xpos[3 * sid + k]) < 1e-9);
  CHECK(std::fabs(fs.radius - 0.05 * extent) < 1e-9);
  std::printf("  site frame: ok\n");

  // Misses: serial 0 and an unbound serial both report ok == false.
  CHECK(!ComputeFocusFrame(m, d, b, 0).ok);
  CHECK(!ComputeFocusFrame(m, d, b, 0xDEADBEEFULL).ok);
  std::printf("  unframable serial: ok\n");
  return 0;
}

int RunTests() {
  if (TestUndoToast()) return 1;
  if (TestRevealChain()) return 1;
  if (TestFocusFrame()) return 1;
  std::printf("editor_ux_windowless: all checks passed\n");
  return 0;
}

}  // namespace
}  // namespace ps::studio

int main() { return ps::studio::RunTests(); }
