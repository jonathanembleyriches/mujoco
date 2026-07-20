// ProtoSpec Studio: interactive joint rigger core (ps::studio, ours).
//
// Everything the rigger needs that can be a pure function over a compiled
// mjModel / forwarded mjData lives here, so it is unit-tested windowless against
// a real mj_loadXML fixture (studio/editor/test/test_rigger_windowless.cc). The
// ImGui "Rig" section (details_panel.cc) and the depth-occluded ScenePlugin
// emission (viewport_plugin.cc) only call into these functions.
//
// Correctness doctrine (docs/rigger_plan.md §3): every visual is a compiled or
// forwarded quantity. The ONE piece of derived geometry -- the hinge range arc /
// slide travel endpoints -- is JointLimitChildPoint, pinned to mj_forward to
// 1e-12 by the windowless test. Scrubbing writes qpos and lets MuJoCo articulate
// the subtree; the editor computes no kinematics of its own.

#ifndef PS_STUDIO_EDITOR_JOINT_RIG_H_
#define PS_STUDIO_EDITOR_JOINT_RIG_H_

#include <cstdint>
#include <vector>

#include <mujoco/mujoco.h>

#include "binding.h"
#include "editor/editor_context.h"
#include "types.h"

namespace ps::studio {

namespace mj = ps::mjcf;

// --- Units (display only) --------------------------------------------------- //
// Compiled DOF values are radians (hinge/ball) or metres (slide). Rotational DOFs
// display in degrees when the tree compiles angles in degrees (MuJoCo default);
// SLIDE is metres and is NEVER converted. This is the single conversion point;
// the scrub/render path elsewhere reads compiled units only (plan §1.6 drift trap).
double JointDofToDisplay(int jnt_type, double compiled, bool angle_is_degree);
double JointDisplayToDof(int jnt_type, double display, bool angle_is_degree);

// Whether the tree compiles angles in degrees. Mirrors ReadOrientContext(model)
// .degree (transform_math.h); duplicated here so the display-helper tests need
// neither a window nor the editor lib.
bool AngleIsDegree(const mj::Model& tree);

// --- Selection / scrub preview --------------------------------------------- //

// The compiled joint id bound to `serial`, or -1 (Joint or FreeJoint entries).
int JointIdForSerial(const mj::Binding& binding, std::uint64_t serial);

// Scrub preview: resolve the joint bound to `serial`, reset ctx.sim_data to
// qpos0, write `q` (compiled units) at jnt_qposadr, mj_forward. Marks
// ctx.rig_preview active with (serial, q) so DoUpdate defers the host forward
// and the viewport mirrors this pose. Returns false if `serial` is not a
// scrubbable hinge/slide joint (ball/free are not scrubbed in P1).
//
// INVARIANT: preview NEVER commits -- no BeginEdit, no ctx.dirty, no recompile,
// the authored tree is untouched. Pinned by test_rigger_windowless
// (no-commit invariants + scrub == reference forward).
bool SetJointPreview(EditorContext& ctx, std::uint64_t serial, double q);

// Snap ctx.sim_data back to qpos0 + forward and clear ctx.rig_preview. Leaves
// dirty / undo / recompile untouched. Idempotent (safe when inactive).
void ClearJointPreview(EditorContext& ctx);

// --- Limit ghosts ----------------------------------------------------------- //

// Compiled geom ids in the kinematic subtree rooted at joint jid's body (bodies
// whose parent chain passes through jnt_bodyid[jid]). Body ids are topologically
// ordered (parent < child), so one forward pass suffices.
std::vector<int> SubtreeGeoms(const mjModel* m, int jid);

// qpos0 with the single dof at jnt_qposadr[jid] set to `q_limit`, into `out`
// (length m->nq). Hinge/slide (1 qpos slot) only.
void GhostQpos(const mjModel* m, int jid, double q_limit, std::vector<double>& out);

// One transparent mjvGeom per subtree geom, posed from scratch_d->geom_x* (the
// caller forwarded scratch_d at a ghost qpos). mjCAT_DECOR (out of pick/segment),
// rgba as given (alpha < 1), mesh dataid re-encoded to the original mesh (2*id,
// matching engine_vis_visualize). Plane/hfield/sdf skipped (exotic, P1). At most
// `max_geoms` appended (the 2000-geom plugin-scene budget guard, plan §5 risk 4).
// Pose per geom equals an independent mj_forward at that qpos -- pinned by
// test_rigger_windowless (ghost pose/count/alpha/category).
std::vector<mjvGeom> CollectGhostGeoms(const mjModel* m, const mjData* scratch_d,
                                       int jid, const float rgba[4],
                                       int max_geoms);

// --- Depth-occluded selected-joint glyph + THE arc contract ----------------- //

// THE pinned joint geometry (plan §3): where a child point `p_ref` (measured at
// the current dof value `q_now`) lands when the dof reaches `q_limit`.
//   hinge: Rot(xaxis, q_limit - q_now) . (p_ref - xanchor) + xanchor
//   slide: p_ref + xaxis . (q_limit - q_now)
// A hinge moves the whole subtree rigidly about xaxis through xanchor; a slide
// translates it along xaxis -- so this equals mj_forward(geom_xpos) at
// qpos=q_limit to 1e-12 for any p_ref/q_now. This is the ONLY derived joint
// geometry in the rigger and it kills the old PlaneBasis arbitrary-zero defect
// (plan §1.1). Pinned by test_rigger_windowless (arc/travel endpoint contract),
// on a degree fixture, a radian twin, and a slide.
void JointLimitChildPoint(int jnt_type, const double xanchor[3],
                          const double xaxis[3], double q_now, double q_limit,
                          const double p_ref[3], double out[3]);

// The subtree geom whose centre is farthest (perpendicular) from the joint axis
// line -- the arc/travel reference with the best radius. -1 when the subtree has
// no geom (or all sit on the axis).
int PickArcReferenceGeom(const mjModel* m, const mjData* d, int jid);

// The depth-occluded glyph for the selected joint: axis arrow through the anchor
// (d->xanchor / d->xaxis), an anchor sphere, and -- when jnt_limited -- a hinge
// range ARC (24-segment fan pinned via JointLimitChildPoint) or a slide travel
// LINE. All mjCAT_DECOR mjvGeoms. q_now is read from d->qpos[jnt_qposadr] so the
// arc tracks the live scrub. Ball -> anchor sphere only; free -> empty. At most
// `max_geoms` appended.
std::vector<mjvGeom> CollectJointGlyph(const mjModel* m, const mjData* d, int jid,
                                       int max_geoms);

}  // namespace ps::studio

#endif  // PS_STUDIO_EDITOR_JOINT_RIG_H_
