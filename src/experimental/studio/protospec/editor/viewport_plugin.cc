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
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include <imgui.h>

#include "editor/authoring_ops.h"
#include "editor/editor_ops.h"
#include "editor/gizmo.h"
#include "editor/gizmo_math.h"
#include "editor/joint_overlay.h"
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

  // Right-click "Add here" drop menu: a clean right-click (press+release, little
  // travel) captures the world point under the cursor and raises the menu.
  bool prev_right = false;
  bool press_r_set = false;
  float press_rx = 0, press_ry = 0;
  bool drop_pending = false;
  double drop_point[3] = {0, 0, 0};
};

// The world point under a screen ray: the nearest geom hit, else the ground
// plane (z = 0). Falls back to a point 2 m down the ray when both miss.
void ComputeDropPoint(const ViewportInput& in, double out[3]) {
  const ViewProj vp =
      BuildViewProj(in.model, in.data, in.camera, in.aspect_ratio);
  double o[3], dir[3];
  ScreenToRay(vp, in.x, in.y, o, dir);
  int geom = -1;
  const mjtByte* group = in.vis_option ? in.vis_option->geomgroup : nullptr;
  const mjtByte flg_static =
      in.vis_option ? in.vis_option->flags[mjVIS_STATIC] : 1;
  const mjtNum pnt[3] = {o[0], o[1], o[2]};
  const mjtNum vec[3] = {dir[0], dir[1], dir[2]};
  const mjtNum dist =
      mj_ray(in.model, in.data, pnt, vec, group, flg_static, -1, &geom, nullptr);
  if (geom >= 0 && dist >= 0) {
    for (int k = 0; k < 3; ++k) out[k] = o[k] + dir[k] * dist;
    return;
  }
  const double po[3] = {0, 0, 0}, pn[3] = {0, 0, 1};
  double t = 0, hit[3];
  if (RayPlaneIntersect(o, dir, po, pn, &t, hit)) {
    for (int k = 0; k < 3; ++k) out[k] = hit[k];
    return;
  }
  for (int k = 0; k < 3; ++k) out[k] = o[k] + dir[k] * 2.0;
}

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

// Joint overlays are pickable (deliverable 3b): a click near a drawn joint's
// axis segment or anchor selects that joint (its serial), taking priority over
// the geom underneath. Only the currently-drawn joints are candidates.
bool TryPickJoint(ViewportEditor& ve, const ViewportInput& in) {
  EditorContext& ctx = *ve.ctx;
  std::vector<JointVis> joints = CollectJointVis(
      in.model, in.data, ctx.compiled.binding, ctx.selected_serial,
      ctx.show_all_joints);
  if (joints.empty()) return false;
  const ViewProj vp =
      BuildViewProj(in.model, in.data, in.camera, in.aspect_ratio);
  double len = 0.12 * in.model->stat.extent;
  if (!(len > 1e-4)) len = 0.15;
  const double tol = 0.018;  // normalized-screen pick radius
  double best = tol * tol;
  std::uint64_t best_serial = 0;
  for (const JointVis& jv : joints) {
    if (jv.serial == 0) continue;
    // Sample the axis segment (hinge/slide) or just the anchor (ball/free).
    const int nsamp =
        (jv.type == mjJNT_HINGE || jv.type == mjJNT_SLIDE) ? 9 : 1;
    const double lo = (jv.type == mjJNT_HINGE) ? -1.0 : 0.0;
    for (int i = 0; i < nsamp; ++i) {
      const double t = nsamp > 1 ? lo + (1.0 - lo) * i / (nsamp - 1) : 0.0;
      double p[3];
      for (int k = 0; k < 3; ++k) p[k] = jv.anchor[k] + jv.axis[k] * len * t;
      ScreenPt sp = WorldToScreen(vp, p);
      if (!sp.visible) continue;
      const double dx = sp.x - in.x, dy = sp.y - in.y;
      const double d2 = dx * dx + dy * dy;
      if (d2 < best) {
        best = d2;
        best_serial = jv.serial;
      }
    }
  }
  if (best_serial != 0) {
    SelectBySerial(ctx, best_serial);
    ctx.Log("pick -> joint serial " + std::to_string(best_serial));
    ve.cycle_index = 0;
    return true;
  }
  return false;
}

void HandlePick(ViewportEditor& ve, const ViewportInput& in) {
  EditorContext& ctx = *ve.ctx;
  if (TryPickJoint(ve, in)) return;
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

  // Right-click "Add here": a clean click (not a fly-mode drag) in Edit mode.
  const bool rpress = in.right_down && !ve->prev_right;
  const bool rrelease = !in.right_down && ve->prev_right;
  if (rpress) {
    ve->press_rx = in.x;
    ve->press_ry = in.y;
    ve->press_r_set = true;
  }
  if (rrelease && ve->press_r_set) {
    const float dx = in.x - ve->press_rx, dy = in.y - ve->press_ry;
    if ((dx * dx + dy * dy) < (0.006f * 0.006f) &&
        ctx.mode == EditorMode::Edit) {
      ComputeDropPoint(in, ve->drop_point);
      ve->drop_pending = true;
    }
    ve->press_r_set = false;
  }
  ve->prev_right = in.right_down;
  return consumed;
}

// The right-click drop-add menu: a new world-parented body+geom at the captured
// world point (deliverable 1, viewport "drop" add).
void DrawDropMenu(ViewportEditor* ve) {
  EditorContext& ctx = *ve->ctx;
  if (ve->drop_pending) {
    ImGui::OpenPopup("ViewportAdd");
    ve->drop_pending = false;
  }
  if (ImGui::BeginPopup("ViewportAdd")) {
    ImGui::TextDisabled("Add at (%.2f, %.2f, %.2f)", ve->drop_point[0],
                        ve->drop_point[1], ve->drop_point[2]);
    ImGui::Separator();
    struct Item { const char* label; mj::GeomType type; };
    const Item items[] = {{"Sphere", mj::GeomType::sphere},
                          {"Box", mj::GeomType::box},
                          {"Capsule", mj::GeomType::capsule},
                          {"Cylinder", mj::GeomType::cylinder},
                          {"Ellipsoid", mj::GeomType::ellipsoid}};
    for (const Item& it : items) {
      if (ImGui::MenuItem(it.label)) {
        AddDropBodyGeomOp(ctx, it.type, ve->drop_point);
      }
    }
    ImGui::EndPopup();
  }
}

void OnDraw(ViewportGuiPlugin* self, const ViewportGuiPlugin::Context& vc) {
  ViewportEditor* ve = static_cast<ViewportEditor*>(self->data);
  ve->gizmo.Draw(*ve->ctx, vc);
  DrawDropMenu(ve);
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

// --- Joint rigging overlay (deliverable 3a) ------------------------------- //
// Distinct colour per joint type; the selected joint is drawn in highlight.
constexpr float kColHinge[4] = {0.20f, 0.90f, 0.35f, 1.0f};
constexpr float kColSlide[4] = {0.25f, 0.70f, 1.00f, 1.0f};
constexpr float kColBall[4] = {0.95f, 0.35f, 0.90f, 1.0f};
constexpr float kColFree[4] = {1.00f, 0.60f, 0.15f, 1.0f};
constexpr float kColSel[4] = {1.00f, 0.92f, 0.15f, 1.0f};

void AddConnector(mjvScene* s, int type, double width, const double a[3],
                  const double b[3], const float rgba[4]) {
  if (s->ngeom >= s->maxgeom) return;
  mjvGeom* g = &s->geoms[s->ngeom];
  mjv_initGeom(g, type, nullptr, nullptr, nullptr, rgba);
  const mjtNum from[3] = {a[0], a[1], a[2]};
  const mjtNum to[3] = {b[0], b[1], b[2]};
  mjv_connector(g, type, width, from, to);
  s->ngeom++;
}

void AddSphere(mjvScene* s, const double c[3], double r, const float rgba[4]) {
  if (s->ngeom >= s->maxgeom) return;
  mjvGeom* g = &s->geoms[s->ngeom];
  const mjtNum size[3] = {r, r, r};
  const mjtNum pos[3] = {c[0], c[1], c[2]};
  mjv_initGeom(g, mjGEOM_SPHERE, size, pos, nullptr, rgba);
  s->ngeom++;
}

// Two orthonormal vectors spanning the plane normal to unit `n`.
void PlaneBasis(const double n[3], double u[3], double v[3]) {
  double seed[3] = {1, 0, 0};
  if (std::fabs(n[0]) > 0.9) { seed[0] = 0; seed[1] = 1; }
  u[0] = seed[1] * n[2] - seed[2] * n[1];
  u[1] = seed[2] * n[0] - seed[0] * n[2];
  u[2] = seed[0] * n[1] - seed[1] * n[0];
  double un = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
  if (un < 1e-9) un = 1;
  for (int k = 0; k < 3; ++k) u[k] /= un;
  v[0] = n[1] * u[2] - n[2] * u[1];
  v[1] = n[2] * u[0] - n[0] * u[2];
  v[2] = n[0] * u[1] - n[1] * u[0];
}

void DrawJoint(mjvScene* s, const JointVis& jv, double len) {
  const float* col = jv.selected ? kColSel
                     : jv.type == mjJNT_HINGE ? kColHinge
                     : jv.type == mjJNT_SLIDE ? kColSlide
                     : jv.type == mjJNT_BALL  ? kColBall
                                              : kColFree;
  const double w = (jv.selected ? 0.010 : 0.006) * (len / 0.15);
  if (jv.type == mjJNT_HINGE || jv.type == mjJNT_SLIDE) {
    // Axis arrow through the anchor (both directions for a hinge pivot).
    double tip[3], tail[3];
    for (int k = 0; k < 3; ++k) {
      tip[k] = jv.anchor[k] + jv.axis[k] * len;
      tail[k] = jv.anchor[k] - jv.axis[k] * len * (jv.type == mjJNT_HINGE ? 1 : 0);
    }
    AddConnector(s, mjGEOM_ARROW, w, tail, tip, col);
    AddSphere(s, jv.anchor, w * 1.6, col);
    if (jv.has_range) {
      if (jv.type == mjJNT_HINGE) {
        double u[3], v[3];
        PlaneBasis(jv.axis, u, v);
        const double r = len * 0.75;
        const int seg = 24;
        double prev[3];
        for (int i = 0; i <= seg; ++i) {
          const double a = jv.range[0] + (jv.range[1] - jv.range[0]) * i / seg;
          double p[3];
          for (int k = 0; k < 3; ++k)
            p[k] = jv.anchor[k] + r * (std::cos(a) * u[k] + std::sin(a) * v[k]);
          if (i > 0) AddConnector(s, mjGEOM_LINE, 2.0, prev, p, col);
          for (int k = 0; k < 3; ++k) prev[k] = p[k];
        }
      } else {  // slide: the travel extent along the axis.
        double a[3], b[3];
        for (int k = 0; k < 3; ++k) {
          a[k] = jv.anchor[k] + jv.axis[k] * jv.range[0];
          b[k] = jv.anchor[k] + jv.axis[k] * jv.range[1];
        }
        AddConnector(s, mjGEOM_LINE, 3.0, a, b, col);
      }
    }
  } else if (jv.type == mjJNT_BALL) {
    AddSphere(s, jv.anchor, len * 0.35, col);
  } else {  // free: a frame triad at the anchor.
    const float rc[4] = {0.95f, 0.3f, 0.3f, 1.0f};
    const float gc[4] = {0.3f, 0.9f, 0.3f, 1.0f};
    const float bc[4] = {0.4f, 0.55f, 1.0f, 1.0f};
    const float* tri[3] = {jv.selected ? kColSel : rc,
                           jv.selected ? kColSel : gc,
                           jv.selected ? kColSel : bc};
    for (int ax = 0; ax < 3; ++ax) {
      double e[3] = {0, 0, 0};
      e[ax] = 1;
      double tip[3];
      for (int k = 0; k < 3; ++k) tip[k] = jv.anchor[k] + e[k] * len * 0.9;
      AddConnector(s, mjGEOM_ARROW, w, jv.anchor, tip, tri[ax]);
    }
  }
}

void DrawJointOverlays(EditorContext* ctx, const mjModel* m, const mjData* d,
                       mjvScene* s) {
  std::vector<JointVis> joints = CollectJointVis(
      m, d, ctx->compiled.binding, ctx->selected_serial, ctx->show_all_joints);
  if (joints.empty()) return;
  double len = 0.12 * m->stat.extent;
  if (!(len > 1e-4)) len = 0.15;
  for (const JointVis& jv : joints) DrawJoint(s, jv, len);
}

void OnOverlay(OverlayPlugin* self, const mjModel* m, const mjData* d,
               mjvScene* s) {
  EditorContext* ctx = static_cast<EditorContext*>(self->data);
  if (!ctx->tree || ctx->compiled.model.get() != m) return;
  DrawJointOverlays(ctx, m, d, s);
  if (ctx->selected_serial == 0) return;
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
  RegisterKey("Duplicate", ImGuiMod_Ctrl | ImGuiKey_D,
              [](KeyHandlerPlugin* s) {
                EditorContext& c = *static_cast<ViewportEditor*>(s->data)->ctx;
                if (c.selected_serial != 0) DuplicateOp(c, c.selected_serial);
              }, ve);
}

}  // namespace ps::studio
