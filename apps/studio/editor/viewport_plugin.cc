// ProtoSpec Studio: the Viewport editor plugin (ps::studio, ours). Replaces the
// SE0 pick logger. It owns the whole viewport interaction organism:
//   ViewportPlugin      mouse -> gizmo drag (via GizmoController) or pick/select;
//   ViewportGuiPlugin   draws the screen-space gizmo over the scene each frame;
//   OverlayPlugin       a selection outline (mjv wireframe box at the aabb);
//   KeyHandlerPlugin    Q/W/E/R tool select, F frame, Del delete, +/- etc.
// A single GizmoController is shared by the mouse and draw hooks.

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <mujoco/mujoco.h>

#include <imgui.h>

#include "editor/editor_ops.h"
#include "editor/gizmo.h"
#include "editor/gizmo_math.h"
#include "editor/plugins.h"
#include "platform/ux/plugin.h"
#include "platform/ux/ps_plugin_ext.h"

namespace ps::studio {
namespace {

namespace bridge = ps::mjcf::bridge;
namespace mj = ps::mjcf;

// Everything the viewport plugin cluster shares: the editor context, the gizmo,
// and the click-through cycling state. One instance, pointed to by every plugin.
struct ViewportEditor {
  EditorContext* ctx = nullptr;
  GizmoController gizmo;

  // Click-through cycling: successive clicks at ~the same pixel walk overlapping
  // geoms front to back.
  float last_click_x = -1, last_click_y = -1;
  int cycle_index = 0;
  bool press_x_set = false;
  float press_x = 0, press_y = 0;
  bool prev_left = false;
};

// Collect the geoms a ray passes through, front to back, by re-casting past each
// hit. Returns (geom_id, body_id) pairs.
std::vector<std::pair<int, int>> PickAlongRay(const mjModel* m, const mjData* d,
                                              const mjvOption* opt,
                                              const double o[3],
                                              const double dir[3]) {
  std::vector<std::pair<int, int>> hits;
  mjtNum pnt[3] = {o[0], o[1], o[2]};
  const mjtNum vec[3] = {dir[0], dir[1], dir[2]};
  const mjtByte* group = opt ? opt->geomgroup : nullptr;
  const mjtByte flg_static = opt ? opt->flags[mjVIS_STATIC] : 1;
  for (int guard = 0; guard < 32; ++guard) {
    int geom = -1;
    const mjtNum dist =
        mj_ray(m, d, pnt, vec, group, flg_static, -1, &geom, nullptr);
    if (geom < 0 || dist < 0) break;
    hits.emplace_back(geom, m->geom_bodyid[geom]);
    // Advance just past the hit to find the next geom behind it.
    for (int i = 0; i < 3; ++i) pnt[i] = pnt[i] + vec[i] * (dist + 1e-4);
  }
  return hits;
}

void HandlePick(ViewportEditor& ve, const ViewportInput& in) {
  EditorContext& ctx = *ve.ctx;
  const ViewProj vp =
      BuildViewProj(in.model, in.data, in.camera, in.aspect_ratio);
  double o[3], dir[3];
  ScreenToRay(vp, in.x, in.y, o, dir);
  std::vector<std::pair<int, int>> hits =
      PickAlongRay(in.model, in.data, in.vis_option, o, dir);

  if (hits.empty()) {
    ctx.selected_serial = 0;
    ctx.selected_desc.clear();
    ctx.Log("pick -> deselect (empty)");
    ve.cycle_index = 0;
    return;
  }

  // Click-through: same pixel as last click advances the cycle; a moved click
  // resets to the front geom.
  const float dx = in.x - ve.last_click_x, dy = in.y - ve.last_click_y;
  const bool same_spot =
      ve.last_click_x >= 0 && (dx * dx + dy * dy) < (0.01f * 0.01f);
  ve.cycle_index = same_spot ? (ve.cycle_index + 1) % static_cast<int>(hits.size())
                             : 0;
  ve.last_click_x = in.x;
  ve.last_click_y = in.y;

  const auto& [geom, body] = hits[ve.cycle_index];
  ResolvePick(ctx, geom, body);
}

bool OnMouse(ViewportPlugin* self, const ViewportInput& in) {
  ViewportEditor* ve = static_cast<ViewportEditor*>(self->data);
  EditorContext& ctx = *ve->ctx;
  if (in.model == nullptr || in.data == nullptr ||
      ctx.compiled.model.get() != in.model) {
    return false;
  }

  // The gizmo gets first crack (grab / drag); it consumes the mouse while active.
  const bool consumed = ve->gizmo.HandleMouse(ctx, in);

  // Single-click selection: a left press+release that neither dragged the gizmo
  // nor orbited the camera (little travel) is a pick.
  const bool press = in.left_down && !ve->prev_left;
  const bool release = !in.left_down && ve->prev_left;
  if (press && !consumed) {
    ve->press_x = in.x;
    ve->press_y = in.y;
    ve->press_x_set = true;
  }
  if (release && ve->press_x_set && !ctx.gizmo_active) {
    const float dx = in.x - ve->press_x, dy = in.y - ve->press_y;
    if ((dx * dx + dy * dy) < (0.006f * 0.006f)) {  // a click, not a drag
      HandlePick(*ve, in);
    }
    ve->press_x_set = false;
  }
  ve->prev_left = in.left_down;
  return consumed;
}

void OnDraw(ViewportGuiPlugin* self, const ViewportGuiPlugin::Context& vc) {
  ViewportEditor* ve = static_cast<ViewportEditor*>(self->data);
  ve->gizmo.Draw(*ve->ctx, vc);
}

// Selection outline: a wireframe box at the selected element's world aabb,
// appended to the scene as mjv line geoms (§5 / deliverable 5). Modest but clear.
void AppendBoxWire(mjvScene* s, const mjtNum center[3], const mjtNum half[3],
                   const float rgba[4]) {
  const int c[8][3] = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
                       {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};
  const int edges[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
                            {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
  mjtNum v[8][3];
  for (int i = 0; i < 8; ++i)
    for (int k = 0; k < 3; ++k) v[i][k] = center[k] + c[i][k] * half[k];
  for (int e = 0; e < 12; ++e) {
    if (s->ngeom >= s->maxgeom) break;
    mjvGeom* g = &s->geoms[s->ngeom];
    mjv_initGeom(g, mjGEOM_LINE, nullptr, nullptr, nullptr, rgba);
    mjv_connector(g, mjGEOM_LINE, 3.0, v[edges[e][0]], v[edges[e][1]]);
    s->ngeom++;
  }
}

void OnOverlay(OverlayPlugin* self, const mjModel* m, const mjData* d,
               mjvScene* s) {
  EditorContext* ctx = static_cast<EditorContext*>(self->data);
  if (!ctx->tree || ctx->selected_serial == 0 || ctx->compiled.model.get() != m)
    return;
  const bridge::Binding& b = ctx->compiled.binding;
  const float rgba[4] = {1.0f, 0.85f, 0.1f, 1.0f};

  for (const bridge::Binding::Entry& e : b.entries()) {
    if (e.serial != ctx->selected_serial || e.id < 0) continue;
    if (e.etype == mj::ElementType::Geom) {
      const int gid = e.id;
      const mjtNum* R = d->geom_xmat + 9 * gid;
      mjtNum half[3];
      for (int k = 0; k < 3; ++k) half[k] = m->geom_aabb[6 * gid + 3 + k] + 0.005;
      // aabb centre is in the geom frame; transform to world.
      mjtNum wc[3];
      mju_mulMatVec3(wc, R, m->geom_aabb + 6 * gid);
      for (int k = 0; k < 3; ++k) wc[k] += d->geom_xpos[3 * gid + k];
      // World AABB half-extent of the oriented local box: abs(R) * half.
      mjtNum wch[3];
      for (int r = 0; r < 3; ++r) {
        wch[r] = std::fabs(R[3 * r + 0]) * half[0] +
                 std::fabs(R[3 * r + 1]) * half[1] +
                 std::fabs(R[3 * r + 2]) * half[2];
      }
      AppendBoxWire(s, wc, wch, rgba);
      return;
    }
    if (e.etype == mj::ElementType::Body) {
      mjtNum half[3] = {0.06, 0.06, 0.06};
      AppendBoxWire(s, d->xpos + 3 * e.id, half, rgba);
      return;
    }
    if (e.etype == mj::ElementType::Site) {
      mjtNum half[3] = {0.04, 0.04, 0.04};
      AppendBoxWire(s, d->site_xpos + 3 * e.id, half, rgba);
      return;
    }
  }
}

void SetTool(EditorContext& c, GizmoTool t) { c.gizmo.tool = t; }

void RegisterKey(const char* name, int chord,
                 KeyHandlerPlugin::OnKeyPressedFn fn, ViewportEditor* ve) {
  KeyHandlerPlugin p;
  p.name = name;
  p.key_chord = chord;
  p.on_key_pressed = fn;
  p.data = ve;
  RegisterPlugin<KeyHandlerPlugin>(p);
}

}  // namespace

void RegisterViewportEditor(EditorContext& ctx) {
  // Leaked for the app lifetime (mirrors the plugin registry's static storage).
  ViewportEditor* ve = new ViewportEditor{&ctx};

  ViewportPlugin mouse;
  mouse.name = "ProtoSpec Viewport";
  mouse.on_mouse = OnMouse;
  mouse.data = ve;
  RegisterPlugin<ViewportPlugin>(mouse);

  ViewportGuiPlugin draw;
  draw.name = "ProtoSpec Gizmo";
  draw.draw = OnDraw;
  draw.data = ve;
  RegisterPlugin<ViewportGuiPlugin>(draw);

  OverlayPlugin overlay;
  overlay.name = "ProtoSpec Selection";
  overlay.add_overlay = OnOverlay;
  overlay.data = &ctx;
  RegisterPlugin<OverlayPlugin>(overlay);

  RegisterKey("Tool Select", ImGuiKey_Q,
              [](KeyHandlerPlugin* s) {
                SetTool(*static_cast<ViewportEditor*>(s->data)->ctx,
                        GizmoTool::Select);
              }, ve);
  RegisterKey("Tool Translate", ImGuiKey_W,
              [](KeyHandlerPlugin* s) {
                SetTool(*static_cast<ViewportEditor*>(s->data)->ctx,
                        GizmoTool::Translate);
              }, ve);
  RegisterKey("Tool Rotate", ImGuiKey_E,
              [](KeyHandlerPlugin* s) {
                SetTool(*static_cast<ViewportEditor*>(s->data)->ctx,
                        GizmoTool::Rotate);
              }, ve);
  RegisterKey("Tool Scale", ImGuiKey_R,
              [](KeyHandlerPlugin* s) {
                SetTool(*static_cast<ViewportEditor*>(s->data)->ctx,
                        GizmoTool::Scale);
              }, ve);
  RegisterKey("Delete", ImGuiKey_Delete,
              [](KeyHandlerPlugin* s) {
                EditorContext& c = *static_cast<ViewportEditor*>(s->data)->ctx;
                if (c.selected_serial == 0) return;
                // Route through the SE1a referrer-confirm flow: the panels layer
                // owns the modal, so request it via the shared context.
                c.delete_request_serial = c.selected_serial;
              }, ve);
}

}  // namespace ps::studio
