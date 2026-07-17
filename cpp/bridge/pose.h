// ProtoSpec pose-patch API: move a compiled element without recompiling.
//
// The compiled pose field of every spatial element factors as
//     field = A ∘ L_authored ∘ B
// where A (a left prefix: the enclosing <frame> chain, and for an align-free
// free joint the inverse-inertial frame) and B (a right suffix: mesh/fit
// recentering, or a body's free-joint inertial fold) are frames the compiler
// bakes in. Both are CONSTANT across a pose-only edit. So to move an element
// during a gizmo drag you do NOT recompile: capture A and B once (PosePatch),
// then write A ∘ L_new ∘ B into the mjModel field and run mj_kinematics /
// mj_forward. See docs/drag_perf_investigation.md and docs/public_api.md.
//
// The read-back that "behaves weirdly for mesh geoms" fails only because it
// tries to INVERT B (the mesh recentering) to recover L. This API never inverts
// B: it re-applies the same captured B to the freshly authored L_new, which is
// exactly correct for mesh geoms.
//
// This header forward-declares mjModel and never includes <mujoco/...>: a
// consumer can hold a PosePatch without linking the engine. ApplyPosePatch lives
// in the MuJoCo quarantine zone.
#ifndef PROTOSPEC_BRIDGE_POSE_H
#define PROTOSPEC_BRIDGE_POSE_H

struct mjModel_;
typedef struct mjModel_ mjModel;

namespace ps::mjcf {

// A rigid transform acting as T(x) = R(quat)*x + pos. quat is a unit quaternion
// stored [w, x, y, z]. This is the public pose type for the pose-patch API; a
// consumer composes L_new from it and hands it to ApplyPosePatch.
struct RigidPose {
  double pos[3] = {0, 0, 0};
  double quat[4] = {1, 0, 0, 0};
};

// out = a ∘ b (b applied first, then a). Public so a caller can build L_new by
// composing rigid transforms without re-deriving the quaternion algebra.
RigidPose Compose(const RigidPose& a, const RigidPose& b);
// out = a^-1.
RigidPose Invert(const RigidPose& a);

// A per-element pose-patch descriptor captured from a compiled model via
// Binding::PosePatchFor. Holds the two baked frames A (prefix) and B (suffix)
// plus, for a FREE-jointed body, the qpos slice that must be reseeded instead
// of body_pos (kinematics reads a free body's pose from qpos, and its rest
// pose from qpos0). Ball and other joints need no reseed: their rest pose
// lives in body_pos/body_quat and their qpos0 is joint-relative (identity
// quat / `ref`), not a body pose.
struct PosePatch {
  int objtype = 0;  // mjOBJ_BODY / GEOM / SITE / CAMERA / LIGHT
  int id = -1;      // mjModel id of the element
  RigidPose prefix;  // A: enclosing <frame> chain (body-local)
  RigidPose suffix;  // B: mesh/fit recentering; body free-joint inertial fold

  // >=0 => ApplyPosePatch also reseeds model->qpos0[adr .. adr+7] for the
  // body's free joint (width is always 7: pos+quat).
  int reseed_qposadr = -1;
  int reseed_width = 0;
};

// Write A ∘ L_new ∘ B into the element's mjModel pose field (body/geom/site/cam
// _pos and _quat; for a light, light_pos only), and reseed model->qpos0 for a
// free-jointed body. Pure array write -- it never inverts B. The caller
// then runs mj_kinematics / mj_forward (and, for a reseeded body, mj_resetData
// or a qpos0->qpos copy so the reseeded rest pose takes effect). Returns false
// when the patch's objtype is not a writable pose field or its id is unset.
bool ApplyPosePatch(mjModel* m, const PosePatch& p, const RigidPose& L_new);

}  // namespace ps::mjcf

#endif  // PROTOSPEC_BRIDGE_POSE_H
