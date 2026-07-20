// ProtoSpec Studio: interactive joint-rig handles (ps::studio, ours). Plan P2.
//
// The rigger's viewport interaction, kept in a SIBLING of the gizmo (not folded
// into gizmo.cc) so an upstream gizmo sync merges without colliding -- the same
// separation rationale transform_math.h:214 states for the joint math. Two
// interactions live here, both grounded in the P1 correctness doctrine (every
// DISPLAYED pose is mj_forward output; only the cursor->value MAPPING is
// approximate, docs/rigger_plan.md §2c/§6 P2):
//
//   * RANGE-ENDPOINT HANDLES -- the hinge arc's two endpoint spokes / the slide
//     travel line's two ends are draggable. A drag maps the cursor ray to a new
//     range value, previews it by patching the compiled m->jnt_range (the arc +
//     ghosts re-pose live off it), and COMMITS on release through the normal
//     gesture flow (BeginEdit -> authored range write -> CommitEdit -> undo +
//     debounced recompile). Esc cancels. THE single unit conversion (compiled
//     radians -> authored compiler-angle units) is DofToAuthored, round-tripped
//     by test_rigger_windowless.
//   * DIRECT LIMB-DRAG SCRUB -- dragging a subtree geom of the selected joint
//     scrubs the dof: the cursor maps to a candidate qpos, then SetJointPreview
//     (joint_rig.h) writes it and mj_forward's the preview data -- identical
//     preview-only/no-commit semantics as the panel slider, so the two produce
//     bitwise-identical poses for the same target value (pinned by the test).
//
// The cursor->value mapping functions are PURE (camera-ray + joint state in,
// dof out) so they are unit-tested windowless with no window/ImGui; the ImGui
// controller (below the PS_RIG_HANDLES_NO_CONTROLLER guard in the .cc) drives
// them from the viewport's input/draw flow.

#ifndef PS_STUDIO_EDITOR_RIG_HANDLES_H_
#define PS_STUDIO_EDITOR_RIG_HANDLES_H_

#include <array>
#include <cstdint>

#include "editor/editor_context.h"
#include "editor/gizmo_math.h"       // ViewProj (controller only; plain struct)
#include "editor/viewport_input.h"   // ViewportInput / ViewportContext

struct mjModel_;
struct mjData_;
typedef struct mjModel_ mjModel;
typedef struct mjData_ mjData;

namespace ps::studio {

// --- Pure cursor->value mapping (windowless) -------------------------------- //

// The range-handle geometry for a limited hinge/slide: the arc/travel reference
// child point p_ref (at the live dof value q_now), and the two endpoint world
// positions e[0]/e[1] where p_ref lands at jnt_range[0]/[1]. Endpoints come from
// JointLimitChildPoint (joint_rig.h) -- the one pinned piece of derived joint
// geometry -- so they coincide with the drawn arc ends. `valid` is false when the
// joint is not a limited hinge/slide or the subtree has no arc reference geom.
struct RangeEndpoints {
  bool valid = false;
  int jnt_type = 0;
  double p_ref[3] = {0, 0, 0};
  double q_now = 0.0;
  double e[2][3] = {{0, 0, 0}, {0, 0, 0}};
};
RangeEndpoints ComputeRangeEndpoints(const mjModel* m, const mjData* d, int jid);

// Map a cursor ray to a candidate dof value (compiled units: radians hinge,
// metres slide), referenced so the ray THROUGH p_ref yields q_now (continuous at
// grab). Hinge: angle, about xaxis at xanchor, of the ray's intersection with the
// joint plane, measured from the p_ref direction. Slide: signed axial travel of
// the ray's closest approach to the axis line, relative to p_ref. Returns q_now
// for a degenerate ray (parallel to the plane / axis, or p_ref on the axis).
double HingeDofFromRay(const double xanchor[3], const double xaxis[3],
                       const double p_ref[3], double q_now,
                       const double ray_o[3], const double ray_d[3]);
double SlideDofFromRay(const double xanchor[3], const double xaxis[3],
                       const double p_ref[3], double q_now,
                       const double ray_o[3], const double ray_d[3]);
double JointDofFromRay(int jnt_type, const double xanchor[3],
                       const double xaxis[3], const double p_ref[3], double q_now,
                       const double ray_o[3], const double ray_d[3]);

// The authored range-field value for a compiled dof value: THE single conversion
// (compiled radians -> authored compiler-angle units for hinge; metres verbatim
// for slide -- never converted). This is JointDofToDisplay's write-direction use
// (authored units == display units): the display helper P1 established, applied
// on the commit path. Round-tripped through a real recompile by the test.
double DofToAuthored(int jnt_type, double compiled, bool angle_is_degree);

// Optional snapping of a mapped dof value, consistent with the editor's gizmo
// snap settings: rounds to snap_rot (radians) for a hinge, snap_trans (metres)
// for a slide, when `on`. Identity when off. Pure so the monotonicity/rounding
// is windowless-testable.
double SnapDof(int jnt_type, double q, bool on, double snap_rot_rad,
               double snap_trans_m);

#ifndef PS_RIG_HANDLES_NO_CONTROLLER

// --- The ImGui controller (viewport-side) ----------------------------------- //

// Drives the range-endpoint handles + limb-scrub from the viewport's per-tick
// mouse/draw flow. One instance lives in the ViewportEditor beside the gizmo.
// Pick priority (stated + tested in viewport_plugin): gizmo handles > rig
// handles > limb-scrub > selection click -- so HandleMouse runs only after the
// gizmo declines the mouse, and it defers to selection when neither a handle nor
// a scrub engages.
class RigHandleController {
 public:
  // Process viewport mouse for the rig handles. Returns true when the rig owns
  // the mouse this frame (an endpoint drag or a limb-scrub is active, or a grab
  // just released) so the caller suppresses camera + selection. Acts only in
  // Edit mode with a selected hinge/slide joint.
  bool HandleMouse(EditorContext& ctx, const ViewportInput& in);

  // Draw the endpoint-handle spheres + hover highlight, the readout text during
  // a drag/scrub, and service an Esc-cancel of an in-flight endpoint drag.
  void Draw(EditorContext& ctx, const ViewportContext& vc);

  // Cursor over an endpoint handle at the last Draw (not yet dragging): the
  // viewport reads this to SetNextFrameWantCaptureMouse (1-frame grab handoff).
  bool hot() const { return hot_; }
  // An endpoint drag or a limb-scrub is in flight this frame.
  bool active() const { return mode_ == Mode::Endpoint || mode_ == Mode::Scrub; }
  // The range endpoint being dragged (0 = min, 1 = max), or -1 when no endpoint
  // drag is in flight. Feeds the interaction-scoped ghost emission: only the
  // dragged endpoint's ghost renders during a handle drag.
  int dragged_endpoint() const {
    return mode_ == Mode::Endpoint ? grab_endpoint_ : -1;
  }

 private:
  enum class Mode { None, ArmScrub, Endpoint, Scrub };

  void CancelEndpoint(EditorContext& ctx);
  void CommitEndpoint(EditorContext& ctx);

  bool prev_left_ = false;
  bool hot_ = false;
  Mode mode_ = Mode::None;

  std::uint64_t drag_serial_ = 0;
  int grab_jid_ = -1;
  int grab_endpoint_ = 0;        // which range endpoint (0/1) is being dragged
  bool began_edit_ = false;      // BeginEdit fired (deferred to first movement)
  double grab_range_[2] = {0, 0};  // compiled jnt_range at grab (endpoint cancel)
  double mapped_q_ = 0.0;        // last mapped dof (readout)

  // Limb-scrub grab state: the grabbed subtree geom's world centre is p_ref, the
  // grab-time dof is q_now; the scrub arms on press over a subtree geom and only
  // engages (sets the preview) once the cursor moves, so a bare click still
  // selects.
  RangeEndpoints scrub_geom_;    // p_ref/q_now for the limb-scrub mapping
  float press_x_ = 0, press_y_ = 0;
};

#endif  // PS_RIG_HANDLES_NO_CONTROLLER

}  // namespace ps::studio

#endif  // PS_STUDIO_EDITOR_RIG_HANDLES_H_
