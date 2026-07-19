// ProtoSpec Studio: the transform gizmo (ps::studio) -- THE core deliverable.
//
// A hand-rolled, ImGuizmo-style screen-space gizmo drawn with the ImGui draw
// list over the mjr-rendered scene (deliverable 3). It owns the drag interaction
// state machine and defers the actual tree edit to transform_math (the DR-S6
// delta rule) so the load-bearing math stays windowless and tested. One
// GizmoController instance is shared by the viewport's per-tick driver: HandleMouse
// (mouse) and Draw (screen-space drawing), both run from the viewport ModelPlugin
// do_update inside the active ImGui frame.

#ifndef PS_STUDIO_EDITOR_GIZMO_H_
#define PS_STUDIO_EDITOR_GIZMO_H_

#include <cstdint>
#include <optional>

#include "editor/editor_context.h"
#include "editor/gizmo_math.h"
#include "editor/placement.h"
#include "editor/transform_math.h"
#include "editor/viewport_input.h"  // ps::studio::ViewportInput / ViewportContext
#include "pose.h"  // ps::mjcf::PosePatch / ApplyPosePatch (the drag fast path)

namespace ps::studio {

// Which part of the gizmo a screen point is over / grabbed. axis 0/1/2 == x/y/z;
// a plane's axis is its normal axis.
enum class HandleKind {
  None,
  TransAxis,
  TransPlane,
  TransScreen,
  RotAxis,
  RotScreen,
  ScaleAxis,
  ScaleUniform,
};
struct GizmoHandle {
  HandleKind kind = HandleKind::None;
  int axis = 0;
  bool operator==(const GizmoHandle& o) const {
    return kind == o.kind && axis == o.axis;
  }
};

class GizmoController {
 public:
  // Processes viewport mouse input (called before camera handling). Returns true
  // when the gizmo owns the mouse this frame, so the host suppresses camera
  // orbit/pan. Only acts in Edit mode with a spatial selection and an active tool.
  bool HandleMouse(EditorContext& ctx, const ViewportInput& in);

  // Draws the gizmo + hover highlight into the active ImGui frame. Also services
  // an Esc-cancel of an in-progress drag.
  void Draw(EditorContext& ctx, const ViewportContext& vc);

  bool dragging() const { return dragging_; }
  // Whether the last Draw found the cursor over a handle (not yet dragging). The
  // viewport do_update reads this to SetNextFrameWantCaptureMouse so the press
  // lands on the gizmo instead of orbiting the host camera (1-frame handoff).
  bool hot() const { return hot_; }

 private:
  void Begin(EditorContext& ctx, const ViewportInput& in, const ViewProj& vp,
             const GizmoHandle& h);
  void UpdateDrag(EditorContext& ctx, const ViewportInput& in,
                  const ViewProj& vp);
  // Preview the current drag frame on the live compiled model without a recompile
  // (returns false when the element is not live-patchable, so the caller
  // recompiles). LivePatchJoint handles the joint anchor/axis case.
  bool LivePatch(EditorContext& ctx, const ViewportInput& in);
  void ArmMeshScaleFixup(EditorContext& ctx);
  bool LivePatchJoint(EditorContext& ctx, mjModel* m, mjData* d);
  // Surface-snap translate (placement.h): glide the dragged element on the
  // surface under the cursor. Returns false when no surface is hit (the caller
  // falls back to the normal constrained translate).
  bool UpdateSurfaceGlide(EditorContext& ctx, const ViewportInput& in,
                          const double ro[3], const double rd[3]);
  void Cancel(EditorContext& ctx);

  bool prev_left_ = false;
  bool dragging_ = false;
  bool hot_ = false;  // cursor over a handle at last Draw (see hot())
  GizmoHandle grabbed_;
  std::uint64_t drag_serial_ = 0;

  DragFrame frame_;        // captured at grab (translate/rotate)
  ScaleBase scale_base_;   // captured at grab (scale)
  SupportCache surf_cache_;  // captured at grab (translate): the element's
                             // support footprint for surface snapping

  // Joint rigging (SE4): when the selection is a joint, W translates its anchor
  // and E reorients its axis, applied through the separable joint functions in
  // transform_math. joint_frame_ is the grab-time joint state; frame_ carries a
  // synthetic display frame (anchor + world axes) so HitTest/Draw are reused.
  bool joint_mode_ = false;
  JointDragFrame joint_frame_;

  // Grab-time references for stable, drift-free cumulative deltas.
  double grab_axis_t_ = 0;       // param along the drag axis at grab
  double grab_hit_[3] = {0, 0, 0};   // world plane-hit at grab
  double grab_angle_ = 0;        // reference angle for rotation
  double gizmo_size_ = 1;        // world length of the gizmo at grab

  // Pose-patch fast path (deliverable 1): for a pure Translate/Rotate drag of a
  // bound Body/Geom/Site/Camera/Light, captured at grab. Each drag frame patches
  // the live mjModel pose field + mj_kinematics instead of recompiling; the drag
  // release does one real Compile (via CommitEdit) to reconcile. Empty => the
  // element is unpatchable (joint, fromto-authored, a light being rotated, or
  // unbound), so the drag falls back to the per-frame recompile preview.
  std::optional<ps::mjcf::PosePatch> pose_patch_;
};

}  // namespace ps::studio

#endif  // PS_STUDIO_EDITOR_GIZMO_H_
