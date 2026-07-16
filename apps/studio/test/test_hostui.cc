// ProtoSpec Studio host-UI certification tests (headless ImGui, no window).
//
// Two host behaviors depend on the live ImGui frame and so escape the windowless
// batteries: the Q/W/E/R viewport-hover key gate (G2) and the real ray-pick +
// gizmo-vs-orbit priority path (G3). We drive the REAL registered plugin vtables
// (ViewportPlugin::on_mouse, the KeyHandlerPlugins) against a headless ImGui
// context -- so the actual gate predicate (ViewportFocused == !WantCaptureMouse)
// and the actual pick / gizmo-hit code run, just with no SDL window or renderer.
//
// The registry dedups by name, so the viewport editor is registered ONCE against
// one shared EditorContext, which every scenario below reuses.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include <mujoco/mujoco.h>
#include <imgui.h>

#include "binding.h"
#include "compile.h"
#include "editor/editor_context.h"
#include "editor/editor_ops.h"
#include "editor/gizmo_math.h"
#include "editor/joint_overlay.h"
#include "editor/plugins.h"
#include "mjcf.h"
#include "platform/ux/plugin.h"
#include "platform/ux/ps_plugin_ext.h"
#include "protospec/traversal.h"
#include "types.h"

namespace mj = ps::mjcf;
using namespace ps::studio;

// The plugin structs and RegisterPlugin/ForEachPlugin live in ps::studio here
// (the standalone host); `platform::` keeps the test body identical to the
// canonical mujoco-studio copy.
namespace platform = ps::studio;

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

// Retrieve a registered plugin vtable by exact name.
template <typename T>
static T* PluginNamed(const char* name) {
  T* found = nullptr;
  platform::ForEachPlugin<T>([&](T* p) {
    if (!found && p->name && std::strcmp(p->name, name) == 0) found = p;
  });
  return found;
}

// Compile `xml` into `ctx` (tree + compiled) and hand back fresh mjData at qpos0.
// The caller owns the returned data.
static mjData* LoadInto(EditorContext& ctx, const char* xml) {
  mj::io::ParseResult r = mj::io::ParseMjcfString(xml, "hostui_test");
  ctx.tree = std::move(r.model);
  mj::CompileOptions opts;
  opts.path = mj::CompilePath::Auto;
  ctx.compiled = mj::Compile(*ctx.tree, opts);
  if (!ctx.compiled.ok()) return nullptr;
  mjData* d = mj_makeData(ctx.compiled.model.get());
  mj_resetData(ctx.compiled.model.get(), d);
  mj_forward(ctx.compiled.model.get(), d);
  ctx.model_ready = true;
  ctx.selected_serial = 0;
  return d;
}

static platform::ViewportInput MakeInput(EditorContext& ctx, mjData* d,
                                         const mjvCamera& cam, float aspect,
                                         float x, float y, bool left_down) {
  platform::ViewportInput in;
  in.model = ctx.compiled.model.get();
  in.data = d;
  in.camera = &cam;
  in.vis_option = nullptr;
  in.aspect_ratio = aspect;
  in.x = x;
  in.y = y;
  in.left_down = left_down;
  return in;
}

// A press+release at (x,y) with no travel: one synthetic left click.
static void Click(platform::ViewportPlugin* vp, EditorContext& ctx, mjData* d,
                  const mjvCamera& cam, float aspect, float x, float y) {
  vp->on_mouse(vp, MakeInput(ctx, d, cam, aspect, x, y, true));
  vp->on_mouse(vp, MakeInput(ctx, d, cam, aspect, x, y, false));
}

// ------------------------------------------------------------------------- //
// G2: the Q/W/E/R tool keys act only while the viewport is hovered (mouse not
// captured by a panel). ViewportFocused() == !ImGui::GetIO().WantCaptureMouse.
// ------------------------------------------------------------------------- //
static void TestKeyRoutingViewportGate(EditorContext& ctx) {
  ctx.gizmo.tool = GizmoTool::Select;

  auto* w = PluginNamed<platform::KeyHandlerPlugin>("Tool Translate");
  auto* e = PluginNamed<platform::KeyHandlerPlugin>("Tool Rotate");
  CHECK(w != nullptr && w->on_key_pressed != nullptr);
  CHECK(e != nullptr);

  ImGuiIO& io = ImGui::GetIO();

  // Viewport hovered (panel not capturing the mouse): W switches to Translate.
  io.WantCaptureMouse = false;
  w->on_key_pressed(w);
  CHECK(ctx.gizmo.tool == GizmoTool::Translate);

  // A panel is focused/hovered (ImGui captures the mouse): a QWER press is NOT
  // allowed to switch tools -- it belongs to the panel / Studio's vis toggles.
  ctx.gizmo.tool = GizmoTool::Select;
  io.WantCaptureMouse = true;
  w->on_key_pressed(w);
  CHECK(ctx.gizmo.tool == GizmoTool::Select);  // gated out
  e->on_key_pressed(e);
  CHECK(ctx.gizmo.tool == GizmoTool::Select);  // still gated

  // Back over the viewport: keys act again.
  io.WantCaptureMouse = false;
  e->on_key_pressed(e);
  CHECK(ctx.gizmo.tool == GizmoTool::Rotate);
}

// ------------------------------------------------------------------------- //
// G3a: a synthetic click at the projected screen position of a known geom (and
// of a known joint overlay) selects it through the real ray-pick / joint-pick
// path in ViewportPlugin::on_mouse.
// ------------------------------------------------------------------------- //
static void TestOverlayAndGeomPick(EditorContext& ctx,
                                   platform::ViewportPlugin* vp,
                                   const mjvCamera& cam, float aspect) {
  mjData* d = LoadInto(ctx, R"(
  <mujoco>
    <worldbody>
      <body name="b" pos="0 0 1">
        <joint name="j" type="hinge" axis="0 1 0"/>
        <geom name="g" type="box" size="0.2 0.2 0.2"/>
      </body>
    </worldbody>
  </mujoco>)");
  CHECK(d != nullptr);
  ctx.mode = EditorMode::Edit;
  ctx.gizmo.tool = GizmoTool::Select;  // clicks are picks, not gizmo grabs

  const std::uint64_t g = ps::sdk::Find<mj::Geom>(*ctx.tree, "g")->serial;
  const std::uint64_t j = ps::sdk::Find<mj::Joint>(*ctx.tree, "j")->serial;

  ViewProj view = BuildViewProj(ctx.compiled.model.get(), d, &cam, aspect);

  // --- Geom pick: no joints drawn, click the geom's projected centre. ---
  ctx.show_all_joints = false;
  ctx.selected_serial = 0;
  double gpos[3];
  for (int i = 0; i < 3; ++i) gpos[i] = d->geom_xpos[i];  // geom 0 == g
  ScreenPt gs = WorldToScreen(view, gpos);
  CHECK(gs.visible);
  Click(vp, ctx, d, cam, aspect, gs.x, gs.y);
  CHECK(ctx.selected_serial == g);

  // --- Joint overlay pick: joints shown, click the joint's world anchor. The
  // overlay takes priority over the geom underneath (both project to the body
  // origin here). ---
  ctx.show_all_joints = true;
  std::vector<JointVis> jv =
      CollectJointVis(ctx.compiled.model.get(), d, ctx.compiled.binding, 0, true);
  CHECK(!jv.empty());
  ScreenPt js = WorldToScreen(view, jv[0].anchor);
  CHECK(js.visible);
  ctx.selected_serial = 0;
  Click(vp, ctx, d, cam, aspect, js.x, js.y);
  CHECK(ctx.selected_serial == j);

  mj_deleteData(d);
}

// ------------------------------------------------------------------------- //
// G3b: with a spatial selection and an active transform tool, a press on the
// gizmo is consumed (the host suppresses camera orbit); with the Select tool the
// same press is NOT consumed, so orbit/pick proceed.
// ------------------------------------------------------------------------- //
static void TestGizmoGrabPriority(EditorContext& ctx,
                                  platform::ViewportPlugin* vp,
                                  const mjvCamera& cam, float aspect) {
  mjData* d = LoadInto(ctx, R"(
  <mujoco>
    <worldbody>
      <body name="b" pos="0 0 1">
        <geom name="g" type="box" size="0.2 0.2 0.2"/>
      </body>
    </worldbody>
  </mujoco>)");
  CHECK(d != nullptr);
  ctx.mode = EditorMode::Edit;
  ctx.show_all_joints = false;

  const std::uint64_t g = ps::sdk::Find<mj::Geom>(*ctx.tree, "g")->serial;
  CHECK(SelectBySerial(ctx, g));

  ViewProj view = BuildViewProj(ctx.compiled.model.get(), d, &cam, aspect);
  double gpos[3];
  for (int i = 0; i < 3; ++i) gpos[i] = d->geom_xpos[i];
  ScreenPt gs = WorldToScreen(view, gpos);  // the gizmo anchor projects here
  CHECK(gs.visible);

  // Translate tool active: pressing on the gizmo centre handle is consumed
  // (returns true) so the host suppresses camera orbit for this drag.
  ctx.gizmo.tool = GizmoTool::Translate;
  const bool consumed = vp->on_mouse(vp, MakeInput(ctx, d, cam, aspect, gs.x, gs.y, true));
  CHECK(consumed);
  CHECK(ctx.gizmo_active);
  vp->on_mouse(vp, MakeInput(ctx, d, cam, aspect, gs.x, gs.y, false));  // release

  // Select tool: no gizmo, so the same press is NOT consumed -- orbit is free.
  ctx.gizmo.tool = GizmoTool::Select;
  ctx.gizmo_active = false;
  const bool consumed_sel = vp->on_mouse(vp, MakeInput(ctx, d, cam, aspect, gs.x, gs.y, true));
  CHECK(!consumed_sel);
  vp->on_mouse(vp, MakeInput(ctx, d, cam, aspect, gs.x, gs.y, false));

  mj_deleteData(d);
}

int main() {
  // Headless ImGui context: enough for GetIO()/GetMainViewport() with no backend.
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize = ImVec2(1200, 900);
  io.DeltaTime = 1.0f / 60.0f;
  unsigned char* pixels = nullptr;
  int tw = 0, th = 0;
  io.Fonts->GetTexDataAsRGBA32(&pixels, &tw, &th);  // build the atlas
  io.Fonts->SetTexID((ImTextureID)(intptr_t)1);
  ImGui::NewFrame();   // establishes the main viewport size from DisplaySize
  ImGui::EndFrame();

  const float aspect = io.DisplaySize.x / io.DisplaySize.y;
  mjvCamera cam;
  mjv_defaultCamera(&cam);
  cam.type = mjCAMERA_FREE;
  cam.lookat[0] = 0;
  cam.lookat[1] = 0;
  cam.lookat[2] = 1;
  cam.distance = 3.0;
  cam.azimuth = 90;
  cam.elevation = -15;

  // The registry dedups by name -> register the viewport editor exactly once,
  // against one context every scenario reuses.
  EditorContext ctx;
  RegisterViewportEditor(ctx);
  auto* vp = PluginNamed<platform::ViewportPlugin>("ProtoSpec Viewport");
  CHECK(vp != nullptr && vp->on_mouse != nullptr);

  TestKeyRoutingViewportGate(ctx);
  TestOverlayAndGeomPick(ctx, vp, cam, aspect);
  TestGizmoGrabPriority(ctx, vp, cam, aspect);

  ImGui::DestroyContext();
  std::printf("%d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
