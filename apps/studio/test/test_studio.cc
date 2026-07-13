// ProtoSpec Studio windowless logic tests (ps::studio). Exercises the editor
// cluster's pure core with no SDL/ImGui/plugin registry: the pending-load state
// machine, the ProtoSpec load pipeline (ParseMjcf -> Validate -> Compile), and
// the pick -> Binding -> element resolution path (the SE0 proof), driving the
// same LoadModel/ResolvePick the plugins call.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

#include <mujoco/mujoco.h>

#include "binding.h"
#include "editor/editor_context.h"
#include "editor/editor_ops.h"
#include "platform/ux/registry.inc.h"
#include "types.h"

using ps::studio::EditorContext;
using ps::studio::LoadModel;
using ps::studio::PendingLoad;
using ps::studio::PickResolution;
using ps::studio::ResolvePick;

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

int main() {
  TestPendingLoadStateMachine();
  TestRegistryOrderAndDedup();
  TestLoadFailureIsClean();

  EditorContext ctx;
  TestLoadHumanoid(ctx);
  TestPickResolvesToElement(ctx);

  std::printf("%d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
