// ProtoSpec Studio: the Viewport editor plugin (ps::studio, ours). R1 re-target:
// the whole viewport interaction organism now runs on the stock plugin surface.
//   * a dedicated ModelPlugin ("ProtoSpec Viewport") whose do_update runs every
//     tick INSIDE the active ImGui frame (window_->NewFrame calls ImGui::NewFrame
//     at App::Update start) but with NO window wrapper, so it can read
//     ImGui::GetIO() for the mouse, draw the gizmo/overlay on the foreground draw
//     list, and SetNextFrameWantCaptureMouse(true) to stand the host camera down
//     (app.cc HandleMouseEvents early-returns on io.WantCaptureMouse). It returns
//     false -- the model ModelPlugin owns the Edit-mode freeze.
//   * KeyHandlerPlugins for Q/W/E/R/F/Del/Ctrl+D/End (unchanged).
// The viewport math runs on the EDITOR's own compiled model + preview mjData
// (ctx.compiled.model / ctx.sim_data), id-consistent with ctx.compiled.binding
// and decoupled from the host's model-reload latency; the camera is the cached
// host mjvCamera*. The selection outline + joint overlays project to the
// foreground draw list (accepted degradation: no depth occlusion).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include <imgui.h>

#include "editor/authoring_ops.h"
#include "editor/editor_ops.h"
#include "editor/gizmo.h"
#include "editor/gizmo_math.h"
#include "editor/placement.h"
#include "editor/joint_overlay.h"
#include "editor/layers.h"
#include "editor/plugin_abi.h"
#include "editor/plugins.h"
#include "editor/viewport_input.h"
#include "platform/ux/plugin.h"

namespace ps::studio {
namespace {

namespace mj = ps::mjcf;

// Everything the viewport plugin shares: the editor context, the gizmo, the
// click-through cycling state, and a private mjvOption for picking (the stock
// surface hands the editor no host vis_option, so pick honours all groups).
struct ViewportEditor {
  EditorContext* ctx = nullptr;
  GizmoController gizmo;
  mjvOption vis_option;

  // Click-through cycling: successive clicks at ~the same pixel walk overlapping
  // geoms front to back.
  float last_click_x = -1, last_click_y = -1;
  int cycle_index = 0;
  bool press_x_set = false;
  float press_x = 0, press_y = 0;
  bool prev_left = false;

  // Right-click "Add here" drop menu.
  bool prev_right = false;
  bool press_r_set = false;
  float press_rx = 0, press_ry = 0;
  bool drop_pending = false;
  double drop_point[3] = {0, 0, 0};

  // Last-seen compiled snapshot pointers, latched every frame so key handlers
  // (End = drop to ground) can raycast without a mouse event in flight.
  const mjModel* last_model = nullptr;
  const mjData* last_data = nullptr;
  const mjvOption* last_opt = nullptr;
};

// --- Screen-space projection (foreground draw list) ------------------------ //

struct Pixel {
  ImVec2 p;
  bool visible = false;
};

Pixel ToPixel(const ViewProj& vp, const double world[3]) {
  const ScreenPt s = WorldToScreen(vp, world);
  const ImGuiViewport* v = ImGui::GetMainViewport();
  Pixel out;
  out.visible = s.visible;
  out.p = ImVec2(v->Pos.x + static_cast<float>(s.x) * v->Size.x,
                 v->Pos.y + static_cast<float>(s.y) * v->Size.y);
  return out;
}

void DrawLine3(ImDrawList* dl, const ViewProj& vp, const double a[3],
               const double b[3], ImU32 col, float thick) {
  const Pixel pa = ToPixel(vp, a), pb = ToPixel(vp, b);
  if (pa.visible && pb.visible) dl->AddLine(pa.p, pb.p, col, thick);
}

// (The old screen-space DrawWireBox is retired -- the selection box is now a
// depth-occluded ScenePlugin outline; see AddSelectionOutlineGeoms below.)

constexpr ImU32 kColHinge = IM_COL32(51, 230, 89, 255);
constexpr ImU32 kColSlide = IM_COL32(64, 179, 255, 255);
constexpr ImU32 kColBall = IM_COL32(242, 89, 230, 255);
constexpr ImU32 kColFree = IM_COL32(255, 153, 38, 255);
constexpr ImU32 kColSel = IM_COL32(255, 235, 38, 255);

// Joint overlays, projected (deliverable 3a). Screen-space simplification of the
// old mjvScene rig: axis segment + anchor dot; the range arc is dropped.
void DrawJointsScreen(EditorContext& ctx, const mjModel* m, const mjData* d,
                      const ViewProj& vp, ImDrawList* dl) {
  std::vector<JointVis> joints = CollectJointVis(
      m, d, ctx.compiled.binding, ctx.selected_serial, ctx.show_all_joints);
  if (joints.empty()) return;
  double len = 0.12 * m->stat.extent;
  if (!(len > 1e-4)) len = 0.15;
  for (const JointVis& jv : joints) {
    const ImU32 col = jv.selected ? kColSel
                      : jv.type == mjJNT_HINGE ? kColHinge
                      : jv.type == mjJNT_SLIDE ? kColSlide
                      : jv.type == mjJNT_BALL  ? kColBall
                                               : kColFree;
    if (jv.type == mjJNT_HINGE || jv.type == mjJNT_SLIDE) {
      double tip[3], tail[3];
      for (int k = 0; k < 3; ++k) {
        tip[k] = jv.anchor[k] + jv.axis[k] * len;
        tail[k] = jv.anchor[k] -
                  jv.axis[k] * len * (jv.type == mjJNT_HINGE ? 1 : 0);
      }
      DrawLine3(dl, vp, tail, tip, col, jv.selected ? 3.0f : 2.0f);
    }
    const Pixel a = ToPixel(vp, jv.anchor);
    if (a.visible) dl->AddCircleFilled(a.p, jv.selected ? 5.0f : 4.0f, col);
    if (jv.type == mjJNT_FREE) {  // frame triad
      const ImU32 tri[3] = {IM_COL32(242, 77, 77, 255), IM_COL32(77, 230, 77, 255),
                            IM_COL32(102, 140, 255, 255)};
      for (int ax = 0; ax < 3; ++ax) {
        double e[3] = {0, 0, 0};
        e[ax] = 1;
        double tip[3];
        for (int k = 0; k < 3; ++k) tip[k] = jv.anchor[k] + e[k] * len * 0.9;
        DrawLine3(dl, vp, jv.anchor, tip, jv.selected ? kColSel : tri[ax], 2.0f);
      }
    }
  }
}

// The selection overlay: joint rigging gizmos only. The selection BOX moved to
// a real depth-occluded ScenePlugin (AddSelectionOutlineGeoms / EnhanceScene
// below) now that upstream ships ScenePlugin (5637f743) -- so the outline is
// occluded by nearer geometry instead of always drawing on top. This retires
// the min-ask doc's worst accepted degradation (the non-occluded screen-space
// box) at zero upstream ask.
void DrawSelectionOverlay(EditorContext& ctx, const mjModel* m, const mjData* d,
                          const ViewProj& vp, ImDrawList* dl) {
  DrawJointsScreen(ctx, m, d, vp, dl);
}

// Adds one mjGEOM_LINEBOX outline geom (world pos/half-extents/rotation) to the
// host scene, honouring the scene's geom budget. mjCAT_DECOR keeps it out of
// selection/segmentation passes.
void AddOutlineBox(mjvScene* scn, const mjtNum pos[3], const mjtNum half[3],
                   const mjtNum mat[9]) {
  if (!scn || scn->ngeom >= scn->maxgeom) return;
  const float rgba[4] = {1.0f, 0.85f, 0.1f, 1.0f};
  mjvGeom* g = &scn->geoms[scn->ngeom];
  mjv_initGeom(g, mjGEOM_LINEBOX, half, pos, mat, rgba);
  g->category = mjCAT_DECOR;
  ++scn->ngeom;
}

// ScenePlugin callback: draw the selected element's outline as a depth-occluded
// wire box. Uses the HOST model/data (which, in Edit, is the editor's adopted &
// forwarded compile, so binding ids/kinematics correspond) keyed by the
// selection serial through the compiled Binding -- exactly the resolution the
// old screen-space box used.
void AddSelectionOutlineGeoms(EditorContext& ctx, const mjModel* m,
                              const mjData* d, mjvScene* scn) {
  if (!ctx.CanEdit() || ctx.selected_serial == 0 || !m || !d || !scn) return;
  for (const mj::Binding::Entry& e : ctx.compiled.binding.entries()) {
    if (e.serial != ctx.selected_serial || e.id < 0) continue;
    if (e.etype == mj::ElementType::Geom) {
      const int gid = e.id;
      const mjtNum* R = d->geom_xmat + 9 * gid;
      mjtNum half[3];
      for (int k = 0; k < 3; ++k) half[k] = m->geom_aabb[6 * gid + 3 + k] + 0.005;
      mjtNum wc[3];
      mju_mulMatVec3(wc, R, m->geom_aabb + 6 * gid);
      for (int k = 0; k < 3; ++k) wc[k] += d->geom_xpos[3 * gid + k];
      AddOutlineBox(scn, wc, half, R);
      return;
    }
    if (e.etype == mj::ElementType::Body) {
      const mjtNum half[3] = {0.06, 0.06, 0.06};
      AddOutlineBox(scn, d->xpos + 3 * e.id, half, d->xmat + 9 * e.id);
      return;
    }
    if (e.etype == mj::ElementType::Site) {
      mjtNum half[3];
      for (int k = 0; k < 3; ++k) half[k] = m->site_size[3 * e.id + k] + 0.01;
      AddOutlineBox(scn, d->site_xpos + 3 * e.id, half, d->site_xmat + 9 * e.id);
      return;
    }
  }
}

void EnhanceScene(ScenePlugin* self, const mjModel* m, mjData* d,
                  mjvScene* scn) {
  ViewportEditor* ve = ctx_cast<ViewportEditor>(self);
  AddSelectionOutlineGeoms(*ve->ctx, m, d, scn);
}

// --- Picking (unchanged math on the editor's own model/data) --------------- //

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
    for (int i = 0; i < 3; ++i) pnt[i] = pnt[i] + vec[i] * (dist + 1e-4);
  }
  return hits;
}

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
  const double tol = 0.018;
  double best = tol * tol;
  std::uint64_t best_serial = 0;
  for (const JointVis& jv : joints) {
    if (jv.serial == 0) continue;
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
    ctx.Diagnose(DiagEntry{DiagEntry::Severity::Info,
                           "pick -> joint serial " + std::to_string(best_serial),
                           best_serial, {}});
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
  const float dx = in.x - ve.last_click_x, dy = in.y - ve.last_click_y;
  const bool same_spot =
      ve.last_click_x >= 0 && (dx * dx + dy * dy) < (0.01f * 0.01f);
  ve.cycle_index = same_spot
                       ? (ve.cycle_index + 1) % static_cast<int>(hits.size())
                       : 0;
  ve.last_click_x = in.x;
  ve.last_click_y = in.y;
  const auto& [geom, body] = hits[ve.cycle_index];
  ResolvePick(ctx, geom, body);
}

// The mouse organism: gizmo grab/drag first, then click-select, then right-click
// "add here". Returns true when the gizmo owns the mouse this frame.
bool HandleViewportMouse(ViewportEditor& ve, const ViewportInput& in) {
  EditorContext& ctx = *ve.ctx;
  const bool consumed = ve.gizmo.HandleMouse(ctx, in);

  const bool press = in.left_down && !ve.prev_left;
  const bool release = !in.left_down && ve.prev_left;
  if (press && !consumed) {
    ve.press_x = in.x;
    ve.press_y = in.y;
    ve.press_x_set = true;
  }
  if (release && ve.press_x_set && !ctx.gizmo_active) {
    const float dx = in.x - ve.press_x, dy = in.y - ve.press_y;
    if ((dx * dx + dy * dy) < (0.006f * 0.006f)) HandlePick(ve, in);
    ve.press_x_set = false;
  }
  ve.prev_left = in.left_down;

  const bool rpress = in.right_down && !ve.prev_right;
  const bool rrelease = !in.right_down && ve.prev_right;
  if (rpress) {
    ve.press_rx = in.x;
    ve.press_ry = in.y;
    ve.press_r_set = true;
  }
  if (rrelease && ve.press_r_set) {
    const float dx = in.x - ve.press_rx, dy = in.y - ve.press_ry;
    if ((dx * dx + dy * dy) < (0.006f * 0.006f) && ctx.CanEdit()) {
      ComputeDropPoint(in, ve.drop_point);
      ve.drop_pending = true;
    }
    ve.press_r_set = false;
  }
  ve.prev_right = in.right_down;
  return consumed;
}

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

bool WorldPosOfSerial(const mjModel* m, const mjData* d,
                      const mj::Binding& binding, std::uint64_t serial,
                      double out[3]) {
  for (const mj::Binding::Entry& e : binding.entries()) {
    if (e.serial != serial || e.id < 0) continue;
    const double* src = nullptr;
    switch (e.etype) {
      case mj::ElementType::Body: src = d->xpos + 3 * e.id; break;
      case mj::ElementType::Geom: src = d->geom_xpos + 3 * e.id; break;
      case mj::ElementType::Site: src = d->site_xpos + 3 * e.id; break;
      case mj::ElementType::Camera: src = d->cam_xpos + 3 * e.id; break;
      case mj::ElementType::Joint:
      case mj::ElementType::FreeJoint:
        if (e.id < m->njnt) src = d->xanchor + 3 * e.id;
        break;
      default: break;
    }
    if (src) {
      for (int k = 0; k < 3; ++k) out[k] = src[k];
      return true;
    }
  }
  return false;
}

// F / Hierarchy-double-click focus: recentre the camera lookat on the requested
// element (punt #4). The cached host camera is mutable via the pre_compile
// const_cast, so writing lookat is well-defined (App::camera_ is non-const).
void ServiceFocus(EditorContext& ctx, const ViewportContext& vc) {
  if (ctx.focus_request_serial == 0 || !vc.camera || !vc.model || !vc.data)
    return;
  const std::uint64_t serial = ctx.focus_request_serial;
  ctx.focus_request_serial = 0;
  if (ctx.compiled.model.get() != vc.model) return;
  double p[3];
  if (WorldPosOfSerial(vc.model, vc.data, ctx.compiled.binding, serial, p)) {
    for (int k = 0; k < 3; ++k) vc.camera->lookat[k] = p[k];
  }
}

void ServiceDiagnosticsReveal(EditorContext& ctx) {
  if (!ctx.focus_diagnostics_request) return;
  ForEachPlugin<GuiPlugin>([](GuiPlugin* p) {
    if (p->name && std::string(p->name) == "Diagnostics") p->active = true;
  });
}

// The empty-state welcome: an actionable entry point instead of a dead hint.
void DrawWelcome(EditorContext& ctx) {
  const ImGuiViewport* v = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(
      ImVec2(v->Pos.x + v->Size.x * 0.5f, v->Pos.y + v->Size.y * 0.45f),
      ImGuiCond_Always, ImVec2(0.5f, 0.5f));
  ImGui::SetNextWindowBgAlpha(0.9f);
  if (ImGui::Begin("##welcome", nullptr,
                   ImGuiWindowFlags_NoDecoration |
                       ImGuiWindowFlags_AlwaysAutoResize |
                       ImGuiWindowFlags_NoSavedSettings |
                       ImGuiWindowFlags_NoDocking)) {
    ImGui::TextDisabled("No model loaded");
    ImGui::Spacing();
    if (ImGui::Button("New Model", ImVec2(180, 0))) NewModelOp(ctx);
    if (ImGui::Button("Open...", ImVec2(180, 0))) {
      ctx.file_dialog.Request(FileDialogState::Kind::Open, "");
    }
    ImGui::Spacing();
    ImGui::TextDisabled("%s", "or drag & drop an MJCF file");
  }
  ImGui::End();
}

ViewportInput MakeInput(ViewportEditor& ve) {
  ViewportInput in;
  in.model = ve.ctx->compiled.model.get();
  in.data = ve.ctx->sim_data;
  in.camera = ve.ctx->camera;
  in.vis_option = &ve.vis_option;
  const ImGuiIO& io = ImGui::GetIO();
  const float w = io.DisplaySize.x > 0 ? io.DisplaySize.x : 1.0f;
  const float h = io.DisplaySize.y > 0 ? io.DisplaySize.y : 1.0f;
  in.x = io.MousePos.x / w;
  in.y = io.MousePos.y / h;
  in.dx = io.MouseDelta.x / w;
  in.dy = io.MouseDelta.y / h;
  in.scroll = io.MouseWheel / 50.0f;
  in.aspect_ratio = w / h;
  in.left_down = ImGui::IsMouseDown(ImGuiMouseButton_Left);
  in.right_down = ImGui::IsMouseDown(ImGuiMouseButton_Right);
  in.middle_down = ImGui::IsMouseDown(ImGuiMouseButton_Middle);
  in.left_double = ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);
  in.right_double = ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Right);
  in.ctrl = io.KeyCtrl;
  in.shift = io.KeyShift;
  in.alt = io.KeyAlt;
  return in;
}

// Live gizmo-drag preview on the HOST scene. The gizmo's LivePatch keeps the
// editor's own sim_data current (patches ctx.compiled.model + mj_kinematics) so
// the outline tracks the drag, but the host owns a SEPARATE model/data copy since
// bytes adoption -- the patch never reached it, so welded bodies/geoms only moved
// on the commit reload. mjv_updateScene reads d->*_xpos / *_xmat directly (it does
// not re-run kinematics), so mirroring the mj_kinematics outputs from sim_data
// onto host_data makes the host track the drag for EVERY element kind, with no
// host_model mutation and no host forward. Only called with matching dims (mid-
// drag there is no structural change), so no copy ever spans a size mismatch.
void MirrorDragKinematics(const mjModel* m, const mjData* src, mjData* dst) {
  mju_copy(dst->xpos, src->xpos, 3 * m->nbody);
  mju_copy(dst->xquat, src->xquat, 4 * m->nbody);
  mju_copy(dst->xmat, src->xmat, 9 * m->nbody);
  mju_copy(dst->xipos, src->xipos, 3 * m->nbody);
  mju_copy(dst->ximat, src->ximat, 9 * m->nbody);
  if (m->ngeom) {
    mju_copy(dst->geom_xpos, src->geom_xpos, 3 * m->ngeom);
    mju_copy(dst->geom_xmat, src->geom_xmat, 9 * m->ngeom);
  }
  if (m->nsite) {
    mju_copy(dst->site_xpos, src->site_xpos, 3 * m->nsite);
    mju_copy(dst->site_xmat, src->site_xmat, 9 * m->nsite);
  }
  if (m->ncam) {
    mju_copy(dst->cam_xpos, src->cam_xpos, 3 * m->ncam);
    mju_copy(dst->cam_xmat, src->cam_xmat, 9 * m->ncam);
  }
  if (m->nlight) {
    mju_copy(dst->light_xpos, src->light_xpos, 3 * m->nlight);
    mju_copy(dst->light_xdir, src->light_xdir, 3 * m->nlight);
  }
  if (m->njnt) {
    mju_copy(dst->xanchor, src->xanchor, 3 * m->njnt);
    mju_copy(dst->xaxis, src->xaxis, 3 * m->njnt);
  }
}

// Whether the host's adopted model matches the editor's compiled model in every
// dimension the drag-preview mirror touches (true mid-drag: no structural edit).
bool DragDimsMatch(const mjModel* a, const mjModel* b) {
  return a && b && a->nbody == b->nbody && a->ngeom == b->ngeom &&
         a->nsite == b->nsite && a->ncam == b->ncam &&
         a->nlight == b->nlight && a->njnt == b->njnt;
}

// The per-tick viewport driver (a ModelPlugin do_update; ImGui frame is active,
// no window wrapper). Returns false: the model ModelPlugin owns Edit-mode freeze.
bool ViewportUpdate(ModelPlugin* self, mjModel* host_model, mjData* host_data) {
  ViewportEditor* ve = ctx_cast<ViewportEditor>(self);
  EditorContext& ctx = *ve->ctx;

  // Empty state (keyed on the EDITOR's state, not any host placeholder model).
  if (!ctx.model_ready || !ctx.compiled.model || !ctx.sim_data || !ctx.camera) {
    if (!ctx.model_ready && !ctx.tree) DrawWelcome(ctx);
    return false;
  }

  const mjModel* m = ctx.compiled.model.get();
  ve->last_model = m;
  ve->last_data = ctx.sim_data;
  ve->last_opt = &ve->vis_option;

  const ViewportInput in = MakeInput(*ve);

  // Interact only over the 3D scene (no ImGui window hovered), OR while a drag is
  // already in flight (we set WantCaptureMouse ourselves, so the flag is true).
  const ImGuiIO& io = ImGui::GetIO();
  const bool over_scene = !io.WantCaptureMouse || ctx.gizmo_active;
  if (over_scene) {
    HandleViewportMouse(*ve, in);
  } else {
    ve->prev_left = in.left_down;  // no phantom press/release edges from a panel
    ve->prev_right = in.right_down;
  }

  // Live drag preview: HandleViewportMouse just ran the gizmo (LivePatch updated
  // sim_data this frame), so push the patched kinematics onto the host scene now,
  // BEFORE the host renders. Runs here (this plugin holds the gizmo + sim_data),
  // not in the model plugin -- which dispatches first, one frame stale. The model
  // plugin defers host writes while gizmo_active so this mirror is authoritative.
  if (ctx.gizmo_active && host_model && host_data &&
      DragDimsMatch(host_model, m)) {
    MirrorDragKinematics(m, ctx.sim_data, host_data);
  }

  // Draw the gizmo + selection overlay + drop menu (foreground draw list).
  ViewportContext vc;
  vc.model = m;
  vc.data = ctx.sim_data;
  vc.camera = ctx.camera;
  vc.aspect_ratio = in.aspect_ratio;
  vc.edit_mode = ctx.CanEdit();
  ve->gizmo.Draw(ctx, vc);
  if (vc.edit_mode) {
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    const ViewProj vp = BuildViewProj(m, ctx.sim_data, ctx.camera, in.aspect_ratio);
    DrawSelectionOverlay(ctx, m, ctx.sim_data, vp, dl);
  }
  ServiceFocus(ctx, vc);
  ServiceDiagnosticsReveal(ctx);
  DrawDropMenu(ve);

  // Suppress the host camera next frame while the gizmo owns the mouse (hover a
  // handle or drag). 1-frame handoff; app.cc gates camera on WantCaptureMouse.
  if (over_scene && (ctx.gizmo_active || ve->gizmo.hot())) {
    ImGui::SetNextFrameWantCaptureMouse(true);
  }
  return false;
}

// --- Key handlers ---------------------------------------------------------- //

bool ViewportFocused() { return !ImGui::GetIO().WantCaptureMouse; }

void SetTool(EditorContext& c, GizmoTool t) {
  if (ViewportFocused()) c.gizmo.tool = t;
}

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
  mjv_defaultOption(&ve->vis_option);
  // Pick honours ALL geom groups. mjv_defaultOption enables only groups 0-2
  // (groups 3-5 off), which silently excludes them from mj_ray -- so a coacd-
  // style body whose collision hulls sit in group 3 (or any group>=3 geom) was
  // selectable in the Hierarchy but NOT by viewport click. The editor has no
  // access to the host's live render vis_option (accepted R1 degradation), so
  // the pick must not second-guess visibility by group: enable every group.
  for (int i = 0; i < mjNGROUP; ++i) ve->vis_option.geomgroup[i] = 1;

  ModelPlugin viewport;
  viewport.name = "ProtoSpec Viewport";
  viewport.do_update = ViewportUpdate;  // per-tick draw/mouse; returns false
  viewport.data = ve;
  RegisterPlugin<ModelPlugin>(viewport);

  // Depth-occluded selection outline (adopts upstream ScenePlugin, 5637f743).
  ScenePlugin outline;
  outline.name = "ProtoSpec Selection Outline";
  outline.enhance_scene = EnhanceScene;
  outline.data = ve;
  RegisterPlugin<ScenePlugin>(outline);

  RegisterKey("Tool Select", ImGuiKey_Q,
              [](KeyHandlerPlugin* s) {
                SetTool(*ctx_cast<ViewportEditor>(s)->ctx, GizmoTool::Select);
              }, ve);
  RegisterKey("Tool Translate", ImGuiKey_W,
              [](KeyHandlerPlugin* s) {
                SetTool(*ctx_cast<ViewportEditor>(s)->ctx, GizmoTool::Translate);
              }, ve);
  RegisterKey("Tool Rotate", ImGuiKey_E,
              [](KeyHandlerPlugin* s) {
                SetTool(*ctx_cast<ViewportEditor>(s)->ctx, GizmoTool::Rotate);
              }, ve);
  RegisterKey("Tool Scale", ImGuiKey_R,
              [](KeyHandlerPlugin* s) {
                SetTool(*ctx_cast<ViewportEditor>(s)->ctx, GizmoTool::Scale);
              }, ve);
  RegisterKey("Frame Selection", ImGuiKey_F,
              [](KeyHandlerPlugin* s) {
                EditorContext& c = *ctx_cast<ViewportEditor>(s)->ctx;
                if (ViewportFocused() && c.selected_serial != 0)
                  c.focus_request_serial = c.selected_serial;
              }, ve);
  RegisterKey("Delete", ImGuiKey_Delete,
              [](KeyHandlerPlugin* s) {
                EditorContext& c = *ctx_cast<ViewportEditor>(s)->ctx;
                if (c.selected_serial == 0 || !c.CanEdit()) return;
                c.delete_request_serial = c.selected_serial;
              }, ve);
  RegisterKey("Duplicate", ImGuiMod_Ctrl | ImGuiKey_D,
              [](KeyHandlerPlugin* s) {
                EditorContext& c = *ctx_cast<ViewportEditor>(s)->ctx;
                if (c.selected_serial != 0 && c.CanEdit() &&
                    SerialInActiveLayer(c, c.selected_serial)) {
                  DuplicateOp(c, c.selected_serial);
                }
              }, ve);
  RegisterKey("Drop To Ground", ImGuiKey_End,
              [](KeyHandlerPlugin* s) {
                ViewportEditor* ve = ctx_cast<ViewportEditor>(s);
                EditorContext& c = *ve->ctx;
                if (!ViewportFocused() || c.selected_serial == 0 ||
                    !c.CanEdit() ||
                    !SerialInActiveLayer(c, c.selected_serial)) {
                  return;
                }
                if (ve->last_model != c.compiled.model.get() || !ve->last_data)
                  return;
                if (DropToGroundOp(c, ve->last_data, ve->last_opt)) {
                  c.status_toast.Post("drop to ground", StatusToast::Kind::Info,
                                      ImGui::GetTime());
                }
              }, ve);
}

}  // namespace ps::studio
