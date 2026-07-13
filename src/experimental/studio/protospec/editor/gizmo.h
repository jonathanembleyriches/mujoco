// ProtoSpec Studio: the transform gizmo (ps::studio) -- THE core deliverable.
//
// A hand-rolled, ImGuizmo-style screen-space gizmo drawn with the ImGui draw
// list over the mjr-rendered scene (deliverable 3). It owns the drag interaction
// state machine and defers the actual tree edit to transform_math (the DR-S6
// delta rule) so the load-bearing math stays windowless and tested. One
// GizmoController instance is shared between the ViewportPlugin (mouse, before
// camera handling) and the ViewportGuiPlugin (drawing, during the ImGui frame).

#ifndef PS_STUDIO_EDITOR_GIZMO_H_
#define PS_STUDIO_EDITOR_GIZMO_H_

#include <cstdint>

#include "editor/editor_context.h"
#include "editor/gizmo_math.h"
#include "editor/transform_math.h"
#include "platform/ux/ps_plugin_ext.h"

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
  void Draw(EditorContext& ctx, const ViewportGuiPlugin::Context& vc);

  bool dragging() const { return dragging_; }

 private:
  void Begin(EditorContext& ctx, const ViewportInput& in, const ViewProj& vp,
             const GizmoHandle& h);
  void UpdateDrag(EditorContext& ctx, const ViewportInput& in,
                  const ViewProj& vp);
  void Cancel(EditorContext& ctx);

  bool prev_left_ = false;
  bool dragging_ = false;
  GizmoHandle grabbed_;
  std::uint64_t drag_serial_ = 0;

  DragFrame frame_;        // captured at grab (translate/rotate)
  ScaleBase scale_base_;   // captured at grab (scale)

  // Grab-time references for stable, drift-free cumulative deltas.
  double grab_axis_t_ = 0;       // param along the drag axis at grab
  double grab_hit_[3] = {0, 0, 0};   // world plane-hit at grab
  double grab_angle_ = 0;        // reference angle for rotation
  double gizmo_size_ = 1;        // world length of the gizmo at grab
};

}  // namespace ps::studio

#endif  // PS_STUDIO_EDITOR_GIZMO_H_
