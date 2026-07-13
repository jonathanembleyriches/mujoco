// ProtoSpec Studio windowless logic tests (ps::studio). Exercises the editor
// cluster's pure core with no SDL/ImGui/plugin registry: the pending-load state
// machine, the ProtoSpec load pipeline (ParseMjcf -> Validate -> Compile), and
// the pick -> Binding -> element resolution path (the SE0 proof), driving the
// same LoadModel/ResolvePick the plugins call.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "binding.h"
#include "editor/editor_context.h"
#include "editor/editor_ops.h"
#include "editor/hierarchy_panel.h"
#include "editor/undo.h"
#include "mjcf.h"
#include "platform/ux/registry.inc.h"
#include "protospec/refs.h"
#include "protospec/traversal.h"
#include "types.h"

using ps::studio::EditorContext;
using ps::studio::LoadModel;
using ps::studio::PendingLoad;
using ps::studio::PickResolution;
using ps::studio::ResolvePick;
namespace mj = ps::mjcf;

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

static std::string CorpusRoot() {
  if (const char* env = std::getenv("PROTOSPEC_CORPUS")) {
    return env;
  }
  return "C:/Users/jonat/Documents/Unreal Projects/url_proj/Plugins/"
         "UnrealRoboticsLab/third_party/MuJoCo/src";
}

static std::string HumanoidPath() {
  return (std::filesystem::path(CorpusRoot()) / "model" / "humanoid" /
          "humanoid.xml")
      .string();
}

// The deferred load slot: request -> pending -> take clears it.
static void TestPendingLoadStateMachine() {
  PendingLoad slot;
  CHECK(!slot.pending());
  slot.Request("a.xml");
  CHECK(slot.pending());
  CHECK(slot.Take() == "a.xml");
  CHECK(!slot.pending());
  // A second take yields empty (nothing pending).
  CHECK(slot.Take().empty());
  // Latest request wins.
  slot.Request("one.xml");
  slot.Request("two.xml");
  CHECK(slot.Take() == "two.xml");
}

// A bad path fails cleanly and leaves no model ready.
static void TestLoadFailureIsClean() {
  EditorContext ctx;
  const bool ok = LoadModel(ctx, "does_not_exist_qwerty.xml");
  CHECK(!ok);
  CHECK(!ctx.model_ready);
  CHECK(ctx.compiled.model == nullptr);
  CHECK(!ctx.diagnostics.empty());
}

// The full ProtoSpec load pipeline on the humanoid corpus model.
static void TestLoadHumanoid(EditorContext& ctx) {
  const std::string path = HumanoidPath();
  if (!std::filesystem::exists(path)) {
    std::printf("SKIP humanoid (corpus not found at %s)\n", path.c_str());
    return;
  }
  const bool ok = LoadModel(ctx, path);
  CHECK(ok);
  CHECK(ctx.model_ready);
  CHECK(ctx.tree != nullptr);
  CHECK(ctx.compiled.ok());
  if (ctx.compiled.model) {
    CHECK(ctx.compiled.model->nq > 0);
    CHECK(ctx.compiled.model->nbody > 1);
  }
}

// pick -> Binding reverse -> element: resolve every bound geom id back to its
// authored ProtoSpec element and confirm the serial round-trips. This is the
// windowless equivalent of a viewport double-click (full Pick needs a live
// mjvScene/camera; the Binding reverse-lookup + selection layer is what the pick
// logger relies on and is what we assert here).
static void TestPickResolvesToElement(EditorContext& ctx) {
  if (!ctx.model_ready || !ctx.compiled.model) {
    return;  // humanoid not loaded (corpus absent)
  }
  namespace bridge = ps::mjcf::bridge;

  int geoms_checked = 0;
  for (const bridge::Binding::Entry& e : ctx.compiled.binding.entries()) {
    if (e.etype != ps::mjcf::ElementType::Geom || e.id < 0) {
      continue;
    }
    const PickResolution r = ResolvePick(ctx, e.id, -1);
    CHECK(r.hit);
    CHECK(r.type == "Geom");
    CHECK(r.serial == e.serial);
    CHECK(ctx.selected_serial == e.serial);
    ++geoms_checked;
    if (geoms_checked >= 5) {
      break;
    }
  }
  CHECK(geoms_checked > 0);

  // A body id resolves as a Body.
  for (const bridge::Binding::Entry& e : ctx.compiled.binding.entries()) {
    if (e.etype != ps::mjcf::ElementType::Body || e.id < 0) {
      continue;
    }
    const PickResolution r = ResolvePick(ctx, -1, e.id);
    CHECK(r.hit);
    CHECK(r.type == "Body");
    CHECK(r.serial == e.serial);
    break;
  }

  // A miss (no ids) resolves to nothing.
  const PickResolution miss = ResolvePick(ctx, -1, -1);
  CHECK(!miss.hit);
}

// The plugin registry dispatches in registration order and dedups by name --
// the dispatch contract the host relies on (ForEachPlugin over each hook type).
namespace {
struct FakePlugin {
  const char* name = "";
  int tag = 0;
};
}  // namespace

static void TestRegistryOrderAndDedup() {
  ps::studio::RegisterPlugin<FakePlugin>({"a", 1});
  ps::studio::RegisterPlugin<FakePlugin>({"b", 2});
  ps::studio::RegisterPlugin<FakePlugin>({"a", 99});  // duplicate name ignored
  ps::studio::RegisterPlugin<FakePlugin>({"c", 3});

  std::string order;
  int count = 0;
  ps::studio::ForEachPlugin<FakePlugin>(
      [&](FakePlugin* p) {
        order += p->name;
        ++count;
        (void)p->tag;
      });
  CHECK(count == 3);          // "a" registered once
  CHECK(order == "abc");      // registration order preserved

  // Mutation through the iterated pointer persists (the host toggles GuiPlugin
  // active this way).
  ps::studio::ForEachPlugin<FakePlugin>(
      [](FakePlugin* p) { p->tag = -1; });
  int sum = 0;
  ps::studio::ForEachPlugin<FakePlugin>([&](FakePlugin* p) { sum += p->tag; });
  CHECK(sum == -3);
}

// --- SE1 editor-core batteries -------------------------------------------- //

// A compact but structurally rich model: a body tree, a replicate macro, two
// typed referrers of one joint (an actuator and a sensor), and a default class.
static const char* kCorpusXml = R"MJCF(
<mujoco model="studio_test">
  <default>
    <default class="big">
      <geom size="2 0 0"/>
    </default>
  </default>
  <worldbody>
    <body name="torso" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom name="torso_geom" type="box" size="1 1 1"/>
      <body name="arm" pos="1 0 0">
        <geom name="arm_geom" type="capsule" size="0.1 0.5"/>
        <site name="tip"/>
      </body>
    </body>
    <replicate count="3" offset="1 0 0">
      <body name="clone">
        <geom name="cg" type="sphere" size="0.1"/>
      </body>
    </replicate>
  </worldbody>
  <actuator>
    <motor name="act1" joint="hinge"/>
  </actuator>
  <sensor>
    <jointpos name="hinge_pos" joint="hinge"/>
  </sensor>
</mujoco>
)MJCF";

static std::unique_ptr<mj::Model> ParseCorpus() {
  ps::mjcf::io::ParseResult r = ps::mjcf::io::ParseMjcfString(kCorpusXml, "corpus");
  if (!r.ok()) {
    std::printf("FATAL: corpus parse failed\n");
    for (const auto& e : r.errors) std::printf("  %s\n", e.Render().c_str());
  }
  return std::move(r.model);
}

static std::uint64_t SerialOfJoint(mj::Model& m, const char* name) {
  const mj::Joint* j = ps::sdk::Find<mj::Joint>(m, name);
  return j ? j->serial : 0;
}

static int CountNodes(const ps::studio::HierNode& n) {
  int c = n.is_section ? 0 : 1;
  for (const auto& ch : n.children) c += CountNodes(ch);
  return c;
}

static const ps::studio::HierNode* FindSection(const ps::studio::HierNode& root,
                                               const char* title) {
  for (const auto& s : root.children) {
    if (s.tag == title) return &s;
  }
  return nullptr;
}

static bool AnyNodeNamed(const ps::studio::HierNode& n, const std::string& name) {
  if (n.name == name) return true;
  for (const auto& ch : n.children) {
    if (AnyNodeNamed(ch, name)) return true;
  }
  return false;
}

static int CountMacroNodes(const ps::studio::HierNode& n) {
  int c = n.is_macro ? 1 : 0;
  for (const auto& ch : n.children) c += CountMacroNodes(ch);
  return c;
}

// CloneWithSerials must reproduce every serial (the undo-selection contract),
// whereas the generated Clone deliberately mints fresh ones.
static void TestCloneWithSerials() {
  std::unique_ptr<mj::Model> m = ParseCorpus();
  CHECK(m != nullptr);
  if (!m) return;
  const std::uint64_t hinge = SerialOfJoint(*m, "hinge");
  CHECK(hinge != 0);

  std::unique_ptr<mj::Model> faithful = ps::studio::CloneWithSerials(*m);
  CHECK(faithful != nullptr);
  CHECK(*faithful == *m);  // structurally equal
  CHECK(SerialOfJoint(*faithful, "hinge") == hinge);  // serial preserved

  // Baseline: the generated Clone mints fresh serials (so the mechanism matters).
  std::unique_ptr<mj::Model> fresh = mj::Clone(*m);
  CHECK(SerialOfJoint(*fresh, "hinge") != hinge);
}

// The authored tree model: section presence/counts and macro-node singularity.
static void TestHierarchyModel() {
  std::unique_ptr<mj::Model> m = ParseCorpus();
  if (!m) return;
  const ps::studio::HierNode root = ps::studio::BuildHierarchyModel(*m);

  const ps::studio::HierNode* body = FindSection(root, "Body Tree");
  CHECK(body != nullptr);
  if (body) {
    // Top level of the hoisted world: torso + the replicate macro.
    CHECK(body->children.size() == 2);
    CHECK(AnyNodeNamed(*body, "torso"));
    CHECK(AnyNodeNamed(*body, "arm"));
    CHECK(AnyNodeNamed(*body, "torso_geom"));
    CHECK(AnyNodeNamed(*body, "tip"));
    // The replicate macro is a single node and never expands its clones.
    CHECK(CountMacroNodes(*body) == 1);
  }

  const ps::studio::HierNode* act = FindSection(root, "Actuators");
  CHECK(act != nullptr && act->children.size() == 1);
  CHECK(act && AnyNodeNamed(*act, "act1"));

  const ps::studio::HierNode* sen = FindSection(root, "Sensors");
  CHECK(sen != nullptr && sen->children.size() == 1);
  CHECK(sen && AnyNodeNamed(*sen, "hinge_pos"));

  const ps::studio::HierNode* def = FindSection(root, "Defaults");
  CHECK(def != nullptr);
  CHECK(def && AnyNodeNamed(*def, "big"));

  // Sections that have no content are omitted (Tendons/Equality/Contact/...).
  CHECK(FindSection(root, "Tendons") == nullptr);

  // Filter keeps a match plus its ancestors; empty query is identity.
  const ps::studio::HierNode all = ps::studio::FilterHierarchy(root, "");
  CHECK(CountNodes(all) == CountNodes(root));
  const ps::studio::HierNode arm_only = ps::studio::FilterHierarchy(root, "arm");
  CHECK(AnyNodeNamed(arm_only, "arm"));
  CHECK(AnyNodeNamed(arm_only, "torso"));         // ancestor retained
  CHECK(!AnyNodeNamed(arm_only, "torso_geom"));   // non-matching sibling pruned
}

// Rename through the editor path rewrites every typed referrer.
static void TestRenameUpdatesReferrers() {
  EditorContext ctx;
  ctx.tree = ParseCorpus();
  if (!ctx.tree) return;
  const std::uint64_t hinge = SerialOfJoint(*ctx.tree, "hinge");
  CHECK(hinge != 0);

  ctx.BeginEdit();
  const int updated = ps::studio::RenameBySerial(ctx, hinge, "hinge2");
  ctx.CommitEdit("rename hinge");
  CHECK(updated == 2);  // motor.joint + jointpos.joint
  CHECK(ps::sdk::Find<mj::Joint>(*ctx.tree, "hinge2") != nullptr);
  CHECK(ps::sdk::Find<mj::Joint>(*ctx.tree, "hinge") == nullptr);
  CHECK(ctx.dirty);
  CHECK(ctx.recompile_requested);

  // Referrers now name the new joint.
  const auto refs = ps::sdk::FindReferrers(*ctx.tree, "hinge2", mj::ElementType::Joint);
  CHECK(refs.size() == 2);

  const int missing = ps::studio::RenameBySerial(ctx, 999999, "nope");
  CHECK(missing == -1);
}

// Delete surfaces the referrers that would dangle, and cascades on confirm.
static void TestDeleteSurfacesReferrers() {
  EditorContext ctx;
  ctx.tree = ParseCorpus();
  if (!ctx.tree) return;
  const std::uint64_t hinge = SerialOfJoint(*ctx.tree, "hinge");
  CHECK(hinge != 0);

  // Read-only preview: both referrers reported, tree untouched.
  const std::vector<std::string> preview =
      ps::studio::PreviewDeleteReferrers(ctx, hinge);
  CHECK(preview.size() == 2);
  CHECK(ps::sdk::Find<mj::Joint>(*ctx.tree, "hinge") != nullptr);  // not deleted

  // Confirmed cascade delete: element gone, referrers cleared.
  ctx.BeginEdit();
  const ps::studio::DeleteResult res = ps::studio::DeleteBySerial(ctx, hinge, true);
  ctx.CommitEdit("delete hinge");
  CHECK(res.found);
  CHECK(res.removed);
  CHECK(ps::sdk::Find<mj::Joint>(*ctx.tree, "hinge") == nullptr);
  CHECK(!ps::studio::FindBySerial(ctx, hinge));
  // No dangling references survive (they were cascaded).
  const auto refs = ps::sdk::FindReferrers(*ctx.tree, "hinge", mj::ElementType::Joint);
  CHECK(refs.empty());
}

// The full undo/redo battery, including serial stability of the selection across
// the model swap -- the subtle mechanism this milestone hinges on.
static void TestUndoRedoBattery() {
  EditorContext ctx;
  ctx.tree = ParseCorpus();
  if (!ctx.tree) return;
  const std::uint64_t hinge = SerialOfJoint(*ctx.tree, "hinge");
  CHECK(hinge != 0);
  ctx.selected_serial = hinge;

  CHECK(!ctx.history.can_undo());
  CHECK(!ctx.history.can_redo());

  // Edit 1: rename.
  ctx.BeginEdit();
  ps::studio::RenameBySerial(ctx, hinge, "renamed");
  ctx.CommitEdit("rename");
  CHECK(ctx.history.can_undo());
  CHECK(ps::sdk::Find<mj::Joint>(*ctx.tree, "renamed") != nullptr);

  // Undo restores the pre-edit name AND preserves the serial (different Model
  // object, same identity) so the selection still resolves.
  CHECK(ps::studio::Undo(ctx));
  CHECK(ps::sdk::Find<mj::Joint>(*ctx.tree, "hinge") != nullptr);
  CHECK(ps::sdk::Find<mj::Joint>(*ctx.tree, "renamed") == nullptr);
  CHECK(SerialOfJoint(*ctx.tree, "hinge") == hinge);       // serial preserved
  CHECK(ctx.selected_serial == hinge);                     // selection survives
  CHECK(ps::studio::FindBySerial(ctx, hinge));             // resolves in new tree
  CHECK(ctx.history.can_redo());
  CHECK(!ctx.history.can_undo());

  // Redo re-applies and keeps the same serial.
  CHECK(ps::studio::Redo(ctx));
  CHECK(ps::sdk::Find<mj::Joint>(*ctx.tree, "renamed") != nullptr);
  CHECK(SerialOfJoint(*ctx.tree, "renamed") == hinge);
  CHECK(ctx.selected_serial == hinge);

  // A fresh commit clears the redo branch.
  ctx.BeginEdit();
  ps::studio::RenameBySerial(ctx, hinge, "renamed3");
  ctx.CommitEdit("rename again");
  CHECK(!ctx.history.can_redo());
  CHECK(ctx.history.can_undo());

  // CancelEdit reverts an in-progress mutation from the snapshot.
  ctx.BeginEdit();
  ps::studio::RenameBySerial(ctx, hinge, "temporary");
  CHECK(ps::sdk::Find<mj::Joint>(*ctx.tree, "temporary") != nullptr);
  ctx.CancelEdit();
  CHECK(ps::sdk::Find<mj::Joint>(*ctx.tree, "temporary") == nullptr);
  CHECK(ps::sdk::Find<mj::Joint>(*ctx.tree, "renamed3") != nullptr);
  CHECK(SerialOfJoint(*ctx.tree, "renamed3") == hinge);
}

// Save through the editor path round-trips to a byte-stable fixpoint.
static void TestSaveRoundTripFixpoint() {
  EditorContext ctx;
  ctx.tree = ParseCorpus();
  if (!ctx.tree) return;
  ctx.dirty = true;

  const std::filesystem::path tmp =
      std::filesystem::temp_directory_path() / "protospec_studio_save_test.xml";
  const bool saved = ps::studio::SaveModel(ctx, tmp.string());
  CHECK(saved);
  CHECK(!ctx.dirty);  // save clears the dirty flag

  // Read the saved file back and re-serialize: authored form is a fixpoint.
  ps::mjcf::io::ParseResult reread = ps::mjcf::io::ParseMjcfFile(tmp.string());
  CHECK(reread.ok());
  if (reread.ok()) {
    const std::string once = ps::mjcf::io::WriteMjcf(*ctx.tree);
    const std::string twice = ps::mjcf::io::WriteMjcf(*reread.model);
    CHECK(once == twice);
    CHECK(*ctx.tree == *reread.model);
  }
  std::error_code ec;
  std::filesystem::remove(tmp, ec);
}

int main() {
  TestPendingLoadStateMachine();
  TestRegistryOrderAndDedup();
  TestLoadFailureIsClean();

  EditorContext ctx;
  TestLoadHumanoid(ctx);
  TestPickResolvesToElement(ctx);

  TestCloneWithSerials();
  TestHierarchyModel();
  TestRenameUpdatesReferrers();
  TestDeleteSurfacesReferrers();
  TestUndoRedoBattery();
  TestSaveRoundTripFixpoint();

  std::printf("%d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
