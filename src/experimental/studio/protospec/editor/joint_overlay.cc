// ProtoSpec Studio: joint visualization collect step. See joint_overlay.h.

#include "editor/joint_overlay.h"

#include <mujoco/mujoco.h>

namespace ps::studio {

namespace mj = ps::mjcf;
namespace bridge = ps::mjcf::bridge;

// The compiled body id that owns the selected element, or -1. Bodies map to
// themselves; a joint/geom/site/camera/light maps to its parent body.
static int BodyOfSelection(const mjModel* m,
                           const bridge::Binding& binding,
                           std::uint64_t selected_serial) {
  if (selected_serial == 0) return -1;
  for (const bridge::Binding::Entry& e : binding.entries()) {
    if (e.serial != selected_serial || e.id < 0) continue;
    switch (e.etype) {
      case mj::ElementType::Body:
        return e.id;
      case mj::ElementType::Joint:
      case mj::ElementType::FreeJoint:
        return (e.id < m->njnt) ? m->jnt_bodyid[e.id] : -1;
      case mj::ElementType::Geom:
        return (e.id < m->ngeom) ? m->geom_bodyid[e.id] : -1;
      case mj::ElementType::Site:
        return (e.id < m->nsite) ? m->site_bodyid[e.id] : -1;
      case mj::ElementType::Camera:
        return (e.id < m->ncam) ? m->cam_bodyid[e.id] : -1;
      case mj::ElementType::Light:
        return (e.id < m->nlight) ? m->light_bodyid[e.id] : -1;
      default:
        return -1;
    }
  }
  return -1;
}

// The tree serial bound to compiled joint id `jid`, or 0.
static std::uint64_t SerialOfJoint(const bridge::Binding& binding, int jid) {
  for (const bridge::Binding::Entry& e : binding.entries()) {
    if (e.id == jid && (e.etype == mj::ElementType::Joint ||
                        e.etype == mj::ElementType::FreeJoint)) {
      return e.serial;
    }
  }
  return 0;
}

std::vector<JointVis> CollectJointVis(const mjModel* m, const mjData* d,
                                      const bridge::Binding& binding,
                                      std::uint64_t selected_serial,
                                      bool show_all) {
  std::vector<JointVis> out;
  if (!m || !d) return out;
  const int active_body =
      show_all ? -1 : BodyOfSelection(m, binding, selected_serial);
  if (!show_all && active_body < 0) return out;

  for (int jid = 0; jid < m->njnt; ++jid) {
    if (!show_all && m->jnt_bodyid[jid] != active_body) continue;
    JointVis jv;
    jv.jnt_id = jid;
    jv.type = m->jnt_type[jid];
    jv.serial = SerialOfJoint(binding, jid);
    jv.selected = (jv.serial != 0 && jv.serial == selected_serial);
    for (int k = 0; k < 3; ++k) {
      jv.anchor[k] = d->xanchor[3 * jid + k];
      jv.axis[k] = d->xaxis[3 * jid + k];
    }
    if (m->jnt_limited[jid] &&
        (jv.type == mjJNT_HINGE || jv.type == mjJNT_SLIDE)) {
      jv.has_range = true;
      jv.range[0] = m->jnt_range[2 * jid + 0];
      jv.range[1] = m->jnt_range[2 * jid + 1];
    }
    out.push_back(jv);
  }
  return out;
}

}  // namespace ps::studio
