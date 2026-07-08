// Native build pipeline implementation (impl-plan T1.x). Inside the MuJoCo
// quarantine zone: this TU includes mujoco.h. See build.h for the contract and
// native.cc for the gate that guards which models reach here.
//
// Purity (CDR-14): every function here takes the tree by const reference and
// writes only into a compile-scoped side table and the mjModel under
// construction.
//
// Lifted from MuJoCo (CDR-3, pin mjVERSION_HEADER 3010000, Apache-2.0, (c)
// DeepMind Technologies Limited -- see NOTICE). The numeric kernels below are
// lifted verbatim or near-verbatim and registered in snapshots/lifted_code.json
// (ids: resolve_orientation, geom_set_inertia, geom_compute_aabb, geom_get_rbound,
// inertia_from_geom, joint_compile, checklimited, bvh_makebvh, body_compile,
// compute_sparse_sizes, set_sizes, copy_tree, copy_names, finalize_simple,
// hash_string); their sources are user_objects.cc / user_model.cc /
// engine_name.c. Per the reuse ledger these are class-C passes: the algorithm,
// constants, and iteration order are upstream; the data plumbing is retargeted
// to ProtoSpec compiled structs and the mjModel arrays.

#include "build.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <mujoco/mujoco.h>

#include "binding.h"       // bridge::ObjTypeOf / FamilyToken
#include "classes.h"       // ps::sdk::Effective (pure effective-defaults query)
#include "context.h"
#include "make_model.h"    // lifted::MakeModel
#include "mesh_pipeline.h" // lifted mesh compile
#include "mjuu_util.h"     // lifted math
#include "reflect.h"
#include "visit.h"

// Exported engine-internal builders (MJAPI in engine_io.h, not surfaced in
// mujoco.h). Own declarations per CDR-1 (drift-gated); called as-is.
extern "C" {
void mj_makeDofDofSparse(int nv, int nC, int nD, int nM, const int* dof_parentid,
                         const int* dof_simplenum, int* rownnz, int* rowadr,
                         int* diag, int* colind, int reduced, int upper,
                         int* remaining);
void mj_makeBSparse(int nv, int nbody, int nB, const int* body_dofnum,
                    const int* body_parentid, const int* body_dofadr,
                    int* B_rownnz, int* B_rowadr, int* B_colind, int* count);
void mj_makeDofDofMaps(int nv, int nM, int nC, int nD, const int* dof_Madr,
                       const int* dof_simplenum, const int* dof_parentid,
                       const int* D_rownnz, const int* D_rowadr, const int* D_colind,
                       const int* M_rownnz, const int* M_rowadr, const int* M_colind,
                       int* mapM2D, int* mapD2M, int* mapM2M, int* M, int* scratch);
}

namespace ps::mjcf::compile {
namespace {

namespace lift = ps::mjcf::compile::lifted;

// --------------------------------------------------------------------------- //
// Compiler settings (S1 front matter): the authored compiler block, resolved.  //
// --------------------------------------------------------------------------- //
struct CompilerSettings {
  bool degree = true;           // MuJoCo default: angle="degree"
  std::string eulerseq = "xyz";
  bool autolimits = true;
  double boundmass = 0;
  double boundinertia = 0;
  bool balanceinertia = false;
  double settotalmass = -1;
  InertiaFromGeom inertiafromgeom = InertiaFromGeom::auto_;
  int inertiagrouprange[2] = {0, mjNGROUP - 1};
  std::string meshdir;          // <compiler meshdir> (falls back to assetdir)
  bool strippath = false;       // <compiler strippath>
};

CompilerSettings ReadCompiler(const Model& m) {
  CompilerSettings s;
  for (const auto& c : m.compilers) {
    if (!c) continue;
    if (c->angle) s.degree = (*c->angle == AngleUnit::degree);
    if (c->eulerseq) s.eulerseq = *c->eulerseq;
    if (c->autolimits) s.autolimits = *c->autolimits;
    if (c->boundmass) s.boundmass = *c->boundmass;
    if (c->boundinertia) s.boundinertia = *c->boundinertia;
    if (c->balanceinertia) s.balanceinertia = *c->balanceinertia;
    if (c->settotalmass) s.settotalmass = *c->settotalmass;
    if (c->inertiafromgeom) s.inertiafromgeom = *c->inertiafromgeom;
    if (c->inertiagrouprange) {
      s.inertiagrouprange[0] = (*c->inertiagrouprange)[0];
      s.inertiagrouprange[1] = (*c->inertiagrouprange)[1];
    }
    if (c->assetdir) s.meshdir = *c->assetdir;
    if (c->meshdir) s.meshdir = *c->meshdir;
    if (c->strippath) s.strippath = *c->strippath;
  }
  return s;
}

// --------------------------------------------------------------------------- //
// Frame transform (S1/S7). Frames flatten away: a <frame> accumulates its pose //
// into every child, nested frames compose (mjCFrame::Compile). `present` marks //
// an enclosing frame so we mirror upstream's `if (frame)` guards exactly.       //
// --------------------------------------------------------------------------- //
struct FrameXform {
  bool present = false;
  double pos[3] = {0, 0, 0};
  double quat[4] = {1, 0, 0, 0};
};

// --------------------------------------------------------------------------- //
// Per-body compiled state (S7). Side table; the tree is never mutated.         //
// --------------------------------------------------------------------------- //
struct CBody {
  const Body* src = nullptr;
  int parentid = 0;
  int weldid = 0;                  // self if it has joints, else parent's weldid
  double pos[3] = {0, 0, 0};       // frame relative to parent
  double quat[4] = {1, 0, 0, 0};
  double ipos[3] = {0, 0, 0};      // inertial frame relative to body
  double iquat[4] = {1, 0, 0, 0};
  double mass = 0;
  double inertia[3] = {0, 0, 0};
  bool has_inertial = false;       // <inertial> authored (ipos "defined")
  bool mocap = false;
  std::vector<double> user;
  int geomadr = -1, geomnum = 0;   // range in the global geom list
  int jntadr = -1, jntnum = 0;     // range in the global joint list
  int dofadr = -1, dofnum = 0;
  bool has_joints = false;
  int contype = 0, conaffinity = 0;
  double margin = 0;
  // BVH result arrays (flattened at fill time).
  std::vector<double> bvh;
  std::vector<int> bvh_child, bvh_level, bvh_nodeid;
  int bvhadr = -1;
};

// --------------------------------------------------------------------------- //
// Pose helpers, lifted from user_model.cc (kFrameEps=1e-6): IsSameVec/Quat/Pose //
// /NullPose, used by the sameframe classification in CopyTree.                  //
// --------------------------------------------------------------------------- //
constexpr double kFrameEps = 1e-6;

bool IsSameVec(const double a[3], const double b[3]) {
  return std::abs(a[0] - b[0]) < kFrameEps && std::abs(a[1] - b[1]) < kFrameEps &&
         std::abs(a[2] - b[2]) < kFrameEps;
}
bool IsSameQuat(const double a[4], const double b[4]) {
  bool minus = std::abs(a[0] - b[0]) < kFrameEps && std::abs(a[1] - b[1]) < kFrameEps &&
               std::abs(a[2] - b[2]) < kFrameEps && std::abs(a[3] - b[3]) < kFrameEps;
  bool plus = std::abs(a[0] + b[0]) < kFrameEps && std::abs(a[1] + b[1]) < kFrameEps &&
              std::abs(a[2] + b[2]) < kFrameEps && std::abs(a[3] + b[3]) < kFrameEps;
  return minus || plus;
}
bool IsSamePose(const double* p1, const double* p2, const double* q1, const double* q2) {
  if (p1 && p2 && !IsSameVec(p1, p2)) return false;
  if (q1 && q2 && !IsSameQuat(q1, q2)) return false;
  return true;
}
bool IsNullPose(const double* pos, const double* quat) {
  double zero[3] = {0, 0, 0}, qunit[4] = {1, 0, 0, 0};
  return IsSamePose(pos, zero, quat, qunit);
}

// --------------------------------------------------------------------------- //
// Per-geom compiled state (S7).                                                //
// --------------------------------------------------------------------------- //
struct CGeom {
  const Geom* src = nullptr;
  int bodyid = 0;
  int type = 0;                    // mjtGeom (ProtoSpec GeomType casts directly)
  double size[3] = {0, 0, 0};
  double pos[3] = {0, 0, 0};
  double quat[4] = {1, 0, 0, 0};
  double aabb[6] = {0, 0, 0, 0, 0, 0};
  double friction[3] = {1, 0.005, 0.0001};
  double solref[2] = {0.02, 1};
  double solimp[5] = {0.9, 0.95, 0.001, 0.5, 2};
  double fluid[mjNFLUID] = {0};
  float rgba[4] = {0.5f, 0.5f, 0.5f, 1.0f};
  int contype = 1, conaffinity = 1, condim = 3, group = 0, priority = 0;
  double solmix = 1, margin = 0, gap = 0;
  double mass_ = 0, density = 1000;
  double inertia[3] = {0, 0, 0};
  double rbound = 0;
  bool has_mass = false;
  bool inferinertia = false;
  std::vector<double> user;
  std::string material_name;       // authored material ref (resolved to matid)
  int matid = -1;
  int dataid = -1;                 // mesh/hfield id (-1: primitive)
};

// Asset geometry a mesh/hfield geom binds to during collection (assets are
// compiled BEFORE the body walk so geom size/aabb/rbound/inertia and body
// inertia inference see the final mesh/hfield geometry, exactly as MuJoCo
// compiles assets before bodies). Keyed by asset name.
struct HfieldBind {
  int id = -1;
  double size[4] = {0, 0, 0, 0};
};
// A mesh geom binds its size/aabb/rbound/inertia and frame to the compiled mesh
// (mjCGeom::Compile mesh branch): aamm gives size/aabb/rbound, pos/quat the
// frame to accumulate, volume + boxsz the GetVolume/SetInertia inputs.
struct MeshBind {
  int id = -1;
  double aamm[6] = {0, 0, 0, 0, 0, 0};
  double pos[3] = {0, 0, 0};
  double quat[4] = {1, 0, 0, 0};
  double volume = 0;
  double boxsz[3] = {0, 0, 0};
};
struct AssetBinds {
  std::unordered_map<std::string, HfieldBind> hfield;
  std::unordered_map<std::string, MeshBind> mesh;
};

// --------------------------------------------------------------------------- //
// Per-joint compiled state (S7). JointType casts directly to mjtJoint.         //
// --------------------------------------------------------------------------- //
struct CJoint {
  const void* src = nullptr;       // Joint* or FreeJoint*
  std::string name;
  std::vector<double> user;
  int bodyid = 0;
  int type = mjJNT_HINGE;
  int group = 0;
  int qposadr = 0, dofadr = 0;
  double pos[3] = {0, 0, 0};
  double axis[3] = {0, 0, 1};
  double range[2] = {0, 0};
  double actfrcrange[2] = {0, 0};
  double solref[2] = {0.02, 1};
  double solimp[5] = {0.9, 0.95, 0.001, 0.5, 2};
  double solref_friction[2] = {0.02, 1};
  double solimp_friction[5] = {0.9, 0.95, 0.001, 0.5, 2};
  double stiffness[3] = {0, 0, 0};   // [scalar, poly0, poly1]
  double damping[3] = {0, 0, 0};
  double margin = 0, ref = 0, springref = 0, armature = 0, frictionloss = 0;
  bool limited = false, actfrclimited = false, actgravcomp = false;

  int nq() const {
    return type == mjJNT_FREE ? 7 : type == mjJNT_BALL ? 4 : 1;
  }
  int nv() const {
    return type == mjJNT_FREE ? 6 : type == mjJNT_BALL ? 3 : 1;
  }
};

// --------------------------------------------------------------------------- //
// Orientation resolver (S5, T1.2), lifted from ResolveOrientation              //
// (user_objects.cc:241-349). All five forms + degree + eulerseq.               //
// --------------------------------------------------------------------------- //
void ResolveQuat(const ps::opt<Orientation>& orient, const CompilerSettings& cs,
                 double quat[4]) {
  quat[0] = 1; quat[1] = quat[2] = quat[3] = 0;
  if (!orient) return;
  const bool degree = cs.degree;

  if (const Quat* q = std::get_if<Quat>(&*orient)) {
    quat[0] = q->w; quat[1] = q->x; quat[2] = q->y; quat[3] = q->z;
    lift::mjuu_normvec(quat, 4);
  } else if (const AxisAngle* aa = std::get_if<AxisAngle>(&*orient)) {
    double ax[4] = {aa->axis[0], aa->axis[1], aa->axis[2], aa->angle};
    if (degree) ax[3] = ax[3] / 180.0 * mjPI;
    if (lift::mjuu_normvec(ax, 3) < lift::mjEPS) return;
    double ang2 = ax[3] / 2;
    quat[0] = std::cos(ang2); quat[1] = std::sin(ang2) * ax[0];
    quat[2] = std::sin(ang2) * ax[1]; quat[3] = std::sin(ang2) * ax[2];
  } else if (const XYAxes* xy = std::get_if<XYAxes>(&*orient)) {
    double a[6]; for (int k = 0; k < 6; ++k) a[k] = xy->xyaxes[k];
    if (lift::mjuu_normvec(a, 3) < lift::mjEPS) return;
    double d = lift::mjuu_dot3(a, a + 3);
    a[3] -= a[0] * d; a[4] -= a[1] * d; a[5] -= a[2] * d;
    if (lift::mjuu_normvec(a + 3, 3) < lift::mjEPS) return;
    double z[3];
    lift::mjuu_crossvec(z, a, a + 3);
    if (lift::mjuu_normvec(z, 3) < lift::mjEPS) return;
    lift::mjuu_frame2quat(quat, a, a + 3, z);
  } else if (const ZAxis* za = std::get_if<ZAxis>(&*orient)) {
    double z[3] = {za->zaxis[0], za->zaxis[1], za->zaxis[2]};
    if (lift::mjuu_normvec(z, 3) < lift::mjEPS) return;
    lift::mjuu_z2quat(quat, z);
  } else if (const Euler* eu = std::get_if<Euler>(&*orient)) {
    double e[3] = {eu->angles[0], eu->angles[1], eu->angles[2]};
    if (degree) for (int i = 0; i < 3; ++i) e[i] = e[i] / 180.0 * mjPI;
    lift::mjuu_setvec(quat, 1, 0, 0, 0);
    const std::string& seq = cs.eulerseq;
    for (int i = 0; i < 3 && i < static_cast<int>(seq.size()); ++i) {
      double tmp[4], qrot[4] = {std::cos(e[i] / 2), 0, 0, 0};
      double sa = std::sin(e[i] / 2);
      char c = seq[i];
      if (c == 'x' || c == 'X') qrot[1] = sa;
      else if (c == 'y' || c == 'Y') qrot[2] = sa;
      else if (c == 'z' || c == 'Z') qrot[3] = sa;
      if (c == 'x' || c == 'y' || c == 'z') lift::mjuu_mulquat(tmp, quat, qrot);
      else lift::mjuu_mulquat(tmp, qrot, quat);
      lift::mjuu_copyvec(quat, tmp, 4);
    }
    lift::mjuu_normvec(quat, 4);
  }
}

// Resolve a <frame>'s pose, composing the enclosing frame (mjCFrame::Compile):
// the child frame's own pos/quat is accumulated under the parent frame, then
// normalized. An absent parent frame is the identity, so the accumulation is a
// no-op and the result is the frame's own resolved pose.
FrameXform ResolveFrame(const Frame& f, const CompilerSettings& cs,
                        const FrameXform& parent) {
  FrameXform out;
  out.present = true;
  if (f.pos) { out.pos[0] = (*f.pos)[0]; out.pos[1] = (*f.pos)[1];
               out.pos[2] = (*f.pos)[2]; }
  ResolveQuat(f.orient, cs, out.quat);
  lift::mjuu_frameaccumChild(parent.pos, parent.quat, out.pos, out.quat);
  lift::mjuu_normvec(out.quat, 4);
  return out;
}

// --------------------------------------------------------------------------- //
// Inertial resolution (S7). Explicit <inertial> only at this stage.            //
// --------------------------------------------------------------------------- //
void ResolveInertial(const Inertial& in, const CompilerSettings& cs, CBody& cb) {
  cb.has_inertial = true;
  if (in.pos) { cb.ipos[0] = (*in.pos)[0]; cb.ipos[1] = (*in.pos)[1];
                cb.ipos[2] = (*in.pos)[2]; }
  cb.mass = in.mass ? *in.mass : 0;
  ResolveQuat(in.iorient, cs, cb.iquat);
  if (in.inertia) {
    if (const DiagInertia* d = std::get_if<DiagInertia>(&*in.inertia)) {
      cb.inertia[0] = d->diaginertia[0];
      cb.inertia[1] = d->diaginertia[1];
      cb.inertia[2] = d->diaginertia[2];
    } else if (const FullInertia* f = std::get_if<FullInertia>(&*in.inertia)) {
      double q[4];
      lift::mjuu_fullInertia(q, cb.inertia, f->fullinertia.data());
      // Compose: iquat carries the eigenframe (iorient must be absent for full).
      cb.iquat[0] = q[0]; cb.iquat[1] = q[1];
      cb.iquat[2] = q[2]; cb.iquat[3] = q[3];
    }
  }
}

// --------------------------------------------------------------------------- //
// Geom closed-form volume/inertia/aabb/rbound, lifted verbatim from            //
// user_objects.cc (mjCGeom::GetVolume/SetInertia/ComputeAABB/GetRBound). Only  //
// primitive types (no mesh/hfield/sdf) reach here (gated). typeinertia is       //
// mjINERTIA_VOLUME(0) or mjINERTIA_SHELL(1) from geom shellinertia.             //
// --------------------------------------------------------------------------- //
double GeomVolume(int type, const double* size, int ti) {
  const double PI = mjPI;
  switch (type) {
    case mjGEOM_SPHERE: {
      double r = size[0];
      return ti == mjINERTIA_SHELL ? 4 * PI * r * r : 4 * PI * r * r * r / 3;
    }
    case mjGEOM_CAPSULE: {
      double h = 2 * size[1], r = size[0];
      return ti == mjINERTIA_SHELL ? 4 * PI * r * r + 2 * PI * r * h
                                   : PI * (r * r * h + 4 * r * r * r / 3);
    }
    case mjGEOM_CYLINDER: {
      double h = 2 * size[1], r = size[0];
      return ti == mjINERTIA_SHELL ? 2 * PI * r * r + 2 * PI * r * h
                                   : PI * r * r * h;
    }
    case mjGEOM_ELLIPSOID: {
      if (ti == mjINERTIA_SHELL) {
        double p = 1.6075;
        double tmp = std::pow(size[0] * size[1], p) + std::pow(size[1] * size[2], p) +
                     std::pow(size[2] * size[0], p);
        return 4 * PI * std::pow(tmp / 3, 1 / p);
      }
      return 4 * PI * size[0] * size[1] * size[2] / 3;
    }
    case mjGEOM_HFIELD:
    case mjGEOM_BOX:
      return ti == mjINERTIA_SHELL
                 ? 8 * (size[0] * size[1] + size[1] * size[2] + size[2] * size[0])
                 : size[0] * size[1] * size[2] * 8;
    default:
      return 0;
  }
}

void GeomSetInertia(int type, const double* size, double mass_, int ti,
                    double inertia[3]) {
  switch (type) {
    case mjGEOM_SPHERE:
      inertia[0] = inertia[1] = inertia[2] =
          (ti == mjINERTIA_SHELL ? 2 * mass_ * size[0] * size[0] / 3
                                 : 2 * mass_ * size[0] * size[0] / 5);
      return;
    case mjGEOM_CAPSULE: {
      double halfheight = size[1], height = 2 * size[1], radius = size[0];
      if (ti == mjINERTIA_SHELL) {
        double Asphere = 4 * mjPI * radius * radius;
        double Acylinder = 2 * mjPI * radius * height;
        double Atotal = Asphere + Acylinder;
        double sphere_mass = mass_ * Asphere / Atotal;
        double cylinder_mass = mass_ - sphere_mass;
        inertia[0] = inertia[1] = cylinder_mass * (6 * radius * radius + height * height) / 12;
        inertia[2] = cylinder_mass * radius * radius;
        double sphere_inertia = 2 * sphere_mass * radius * radius / 3;
        double hs_com = radius / 2;
        double hs_pos = halfheight + hs_com;
        inertia[0] += sphere_inertia + sphere_mass * (hs_pos * hs_pos - hs_com * hs_com);
        inertia[1] += sphere_inertia + sphere_mass * (hs_pos * hs_pos - hs_com * hs_com);
        inertia[2] += sphere_inertia;
      } else {
        double sphere_mass = mass_ * 4 * radius / (4 * radius + 3 * height);
        double cylinder_mass = mass_ - sphere_mass;
        inertia[0] = inertia[1] = cylinder_mass * (3 * radius * radius + height * height) / 12;
        inertia[2] = cylinder_mass * radius * radius / 2;
        double sphere_inertia = 2 * sphere_mass * radius * radius / 5;
        inertia[0] += sphere_inertia + sphere_mass * height * (3 * radius + 2 * height) / 8;
        inertia[1] += sphere_inertia + sphere_mass * height * (3 * radius + 2 * height) / 8;
        inertia[2] += sphere_inertia;
      }
      return;
    }
    case mjGEOM_CYLINDER: {
      double halfheight = size[1], height = 2 * halfheight, radius = size[0];
      if (ti == mjINERTIA_SHELL) {
        double Adisk = mjPI * radius * radius;
        double Acylinder = 2 * mjPI * radius * height;
        double Atotal = 2 * Adisk + Acylinder;
        double mass_disk = mass_ * Adisk / Atotal;
        double mass_cylinder = mass_ - 2 * mass_disk;
        inertia[0] = inertia[1] = mass_cylinder * (6 * radius * radius + height * height) / 12;
        inertia[2] = mass_cylinder * radius * radius;
        double inertia_disk_x = mass_disk * radius * radius / 4 + mass_disk * halfheight * halfheight;
        double inertia_disk_z = mass_disk * radius * radius / 2;
        inertia[0] += 2 * inertia_disk_x;
        inertia[1] += 2 * inertia_disk_x;
        inertia[2] += 2 * inertia_disk_z;
      } else {
        inertia[0] = inertia[1] = mass_ * (3 * radius * radius + height * height) / 12;
        inertia[2] = mass_ * radius * radius / 2;
      }
      return;
    }
    case mjGEOM_ELLIPSOID: {
      double s00 = size[0] * size[0], s11 = size[1] * size[1], s22 = size[2] * size[2];
      if (ti == mjINERTIA_SHELL) {
        double eps = 1e-6;
        double Va = 4 * mjPI * size[0] * size[1] * size[2] / 3;
        double ae = size[0] + eps, be = size[1] + eps, ce = size[2] + eps;
        double Vb = 4 * mjPI * ae * be * ce / 3;
        double density = mass_ / (Vb - Va);
        double mass_a = Va * density, ia[3];
        ia[0] = mass_a * (s11 + s22) / 5; ia[1] = mass_a * (s00 + s22) / 5;
        ia[2] = mass_a * (s00 + s11) / 5;
        double mass_b = Vb * density, ib[3];
        ib[0] = mass_b * (be * be + ce * ce) / 5; ib[1] = mass_b * (ae * ae + ce * ce) / 5;
        ib[2] = mass_b * (ae * ae + be * be) / 5;
        inertia[0] = ib[0] - ia[0]; inertia[1] = ib[1] - ia[1]; inertia[2] = ib[2] - ia[2];
      } else {
        inertia[0] = mass_ * (s11 + s22) / 5; inertia[1] = mass_ * (s00 + s22) / 5;
        inertia[2] = mass_ * (s00 + s11) / 5;
      }
      return;
    }
    case mjGEOM_HFIELD:
    case mjGEOM_BOX: {
      double s00 = size[0] * size[0], s11 = size[1] * size[1], s22 = size[2] * size[2];
      if (ti == mjINERTIA_SHELL) {
        double lx = 2 * size[0], ly = 2 * size[1], lz = 2 * size[2];
        double A0 = lx * ly, A1 = ly * lz, A2 = lz * lx;
        double Atotal = 2 * (A0 + A1 + A2);
        double mass0 = mass_ * A0 / Atotal;
        double Ix0 = mass0 * ly * ly / 12, Iy0 = mass0 * lx * lx / 12,
               Iz0 = mass0 * (lx * lx + ly * ly) / 12;
        double mass1 = mass_ * A1 / Atotal;
        double Ix1 = mass1 * (ly * ly + lz * lz) / 12, Iy1 = mass1 * lz * lz / 12,
               Iz1 = mass1 * ly * ly / 12;
        double mass2 = mass_ * A2 / Atotal;
        double Ix2 = mass2 * lz * lz / 12, Iy2 = mass2 * (lx * lx + lz * lz) / 12,
               Iz2 = mass2 * lx * lx / 12;
        inertia[0] = 2 * (mass0 * s22 + mass2 * s11 + Ix0 + Ix1 + Ix2);
        inertia[1] = 2 * (mass0 * s22 + mass1 * s00 + Iy0 + Iy1 + Iy2);
        inertia[2] = 2 * (mass1 * s00 + mass2 * s11 + Iz0 + Iz1 + Iz2);
      } else {
        inertia[0] = mass_ * (s11 + s22) / 3; inertia[1] = mass_ * (s00 + s22) / 3;
        inertia[2] = mass_ * (s00 + s11) / 3;
      }
      return;
    }
    default:
      inertia[0] = inertia[1] = inertia[2] = 0;
      return;
  }
}

void GeomComputeAABB(int type, const double* size, double aabb[6]) {
  double aamm[6];
  switch (type) {
    case mjGEOM_SPHERE:
      aamm[3] = aamm[4] = aamm[5] = size[0];
      aamm[0] = -aamm[3]; aamm[1] = -aamm[4]; aamm[2] = -aamm[5];
      break;
    case mjGEOM_CAPSULE:
      aamm[3] = aamm[4] = size[0]; aamm[5] = size[0] + size[1];
      aamm[0] = -aamm[3]; aamm[1] = -aamm[4]; aamm[2] = -aamm[5];
      break;
    case mjGEOM_CYLINDER:
      aamm[3] = aamm[4] = size[0]; aamm[5] = size[1];
      aamm[0] = -aamm[3]; aamm[1] = -aamm[4]; aamm[2] = -aamm[5];
      break;
    case mjGEOM_PLANE:
      aamm[0] = aamm[1] = aamm[2] = -mjMAXVAL;
      aamm[3] = aamm[4] = mjMAXVAL; aamm[5] = 0;
      break;
    default:
      aamm[3] = size[0]; aamm[4] = size[1]; aamm[5] = size[2];
      aamm[0] = -size[0]; aamm[1] = -size[1]; aamm[2] = -size[2];
      break;
  }
  aabb[0] = (aamm[3] + aamm[0]) / 2; aabb[1] = (aamm[4] + aamm[1]) / 2;
  aabb[2] = (aamm[5] + aamm[2]) / 2; aabb[3] = (aamm[3] - aamm[0]) / 2;
  aabb[4] = (aamm[4] - aamm[1]) / 2; aabb[5] = (aamm[5] - aamm[2]) / 2;
}

double GeomRBound(int type, const double* size) {
  switch (type) {
    case mjGEOM_SPHERE: return size[0];
    case mjGEOM_CAPSULE: return size[0] + size[1];
    case mjGEOM_CYLINDER: return std::sqrt(size[0] * size[0] + size[1] * size[1]);
    case mjGEOM_ELLIPSOID: return std::max(std::max(size[0], size[1]), size[2]);
    case mjGEOM_BOX:
      return std::sqrt(size[0] * size[0] + size[1] * size[1] + size[2] * size[2]);
    default: return 0;
  }
}

// Compile one geom (mjCGeom::Compile, primitive path). `cs` supplies degree;
// `assets` binds hfield/mesh geoms (assets compiled before the body walk).
CGeom GeomCompile(const Model& model, const Geom& g, const CompilerSettings& cs,
                  int bodyid, bool inferinertia, const AssetBinds& assets) {
  std::unique_ptr<Geom> eff = ps::sdk::Effective(model, g);
  CGeom cg;
  cg.src = &g;
  cg.bodyid = bodyid;
  cg.type = eff->type ? static_cast<int>(*eff->type) : mjGEOM_SPHERE;
  if (eff->size) for (std::size_t k = 0; k < eff->size->size() && k < 3; ++k)
    cg.size[k] = (*eff->size)[k];
  if (eff->pos) { cg.pos[0] = (*eff->pos)[0]; cg.pos[1] = (*eff->pos)[1];
                  cg.pos[2] = (*eff->pos)[2]; }
  if (eff->contype) cg.contype = *eff->contype;
  if (eff->conaffinity) cg.conaffinity = *eff->conaffinity;
  if (eff->condim) cg.condim = *eff->condim;
  if (eff->group) cg.group = *eff->group;
  if (eff->priority) cg.priority = *eff->priority;
  if (eff->solmix) cg.solmix = *eff->solmix;
  if (eff->margin) cg.margin = *eff->margin;
  if (eff->gap) cg.gap = *eff->gap;
  if (eff->friction) for (std::size_t k = 0; k < eff->friction->size() && k < 3; ++k)
    cg.friction[k] = (*eff->friction)[k];
  if (eff->solref) for (std::size_t k = 0; k < eff->solref->size() && k < 2; ++k)
    cg.solref[k] = (*eff->solref)[k];
  if (eff->solimp) for (std::size_t k = 0; k < eff->solimp->size() && k < 5; ++k)
    cg.solimp[k] = (*eff->solimp)[k];
  if (eff->rgba) for (int k = 0; k < 4; ++k) cg.rgba[k] = (*eff->rgba)[k];
  if (eff->material) cg.material_name = eff->material->name;
  if (eff->density) cg.density = *eff->density;
  if (eff->user) cg.user = *eff->user;
  cg.has_mass = eff->mass.has_value();
  double mass = cg.has_mass ? *eff->mass : 0;
  const int ti = (eff->shellinertia && *eff->shellinertia) ? mjINERTIA_SHELL
                                                           : mjINERTIA_VOLUME;

  // normalize quaternion / resolve orientation
  ResolveQuat(eff->orient, cs, cg.quat);
  lift::mjuu_normvec(cg.quat, 4);

  // fromto path
  const FromTo* ft = nullptr;
  if (eff->shape) ft = std::get_if<FromTo>(&*eff->shape);
  if (ft) {
    const double* f = ft->fromto.data();
    double vec[3] = {f[0] - f[3], f[1] - f[4], f[2] - f[5]};
    cg.size[1] = lift::mjuu_normvec(vec, 3) / 2;
    if (cg.type == mjGEOM_ELLIPSOID || cg.type == mjGEOM_BOX) {
      cg.size[2] = cg.size[1];
      cg.size[1] = cg.size[0];
    }
    cg.pos[0] = (f[0] + f[3]) / 2; cg.pos[1] = (f[1] + f[4]) / 2;
    cg.pos[2] = (f[2] + f[5]) / 2;
    lift::mjuu_z2quat(cg.quat, vec);
  }

  // hfield geom: size/aabb/rbound come from the bound hfield, not the geom's own
  // size (mjCGeom::Compile hfield branch + ComputeAABB/GetRBound hfield cases).
  if (cg.type == mjGEOM_HFIELD) {
    const std::string hn = eff->hfield ? eff->hfield->name : std::string();
    auto it = assets.hfield.find(hn);
    if (it != assets.hfield.end()) {
      const HfieldBind& hb = it->second;
      cg.dataid = hb.id;
      cg.size[0] = hb.size[0];
      cg.size[1] = hb.size[1];
      cg.size[2] = 0.25 * hb.size[2] + 0.5 * hb.size[3];
      // ComputeAABB hfield: aamm=[-s0,-s1,-s3, s0,s1,s2] -> (center, half).
      const double aamm[6] = {-hb.size[0], -hb.size[1], -hb.size[3],
                              hb.size[0], hb.size[1], hb.size[2]};
      for (int k = 0; k < 3; ++k) {
        cg.aabb[k] = (aamm[k + 3] + aamm[k]) / 2;
        cg.aabb[k + 3] = (aamm[k + 3] - aamm[k]) / 2;
      }
      // GetRBound hfield.
      cg.rbound = std::sqrt(hb.size[0] * hb.size[0] + hb.size[1] * hb.size[1] +
                            std::max(hb.size[2] * hb.size[2],
                                     hb.size[3] * hb.size[3]));
      cg.inferinertia = inferinertia;
      // hfield GetVolume/SetInertia use the box formula on the geom size.
      if (inferinertia) {
        if (cg.has_mass) {
          if (mass == 0) { cg.mass_ = 0; cg.density = 0; }
          else if (GeomVolume(cg.type, cg.size, ti) > lift::mjEPS) {
            cg.mass_ = mass;
            cg.density = mass / GeomVolume(cg.type, cg.size, ti);
            GeomSetInertia(cg.type, cg.size, cg.mass_, ti, cg.inertia);
          }
        } else if (cg.density != 0) {
          cg.mass_ = cg.density * GeomVolume(cg.type, cg.size, ti);
          GeomSetInertia(cg.type, cg.size, cg.mass_, ti, cg.inertia);
        }
      }
      return cg;
    }
  }

  // mesh geom: accumulate the mesh frame, then take size/aabb/rbound from the
  // mesh aamm and inertia from the mesh volume/box (mjCGeom::Compile mesh branch
  // + GetVolume/SetInertia/GetRBound/ComputeAABB mesh cases).
  if (cg.type == mjGEOM_MESH) {
    // The geom's mesh ref carries the referenced name, which is exactly the key
    // the asset table is built under (authored name, else the file stem).
    const std::string mn = eff->mesh ? eff->mesh->name : std::string();
    auto it = assets.mesh.find(mn);
    if (it != assets.mesh.end()) {
      const MeshBind& mb = it->second;
      cg.dataid = mb.id;
      // rotate center (0 for a mesh geom) to the geom frame, add the mesh pos,
      // then accumulate the mesh frame into the geom frame.
      double center[3] = {0, 0, 0}, meshpos[3];
      lift::mjuu_rotVecQuat(meshpos, center, mb.quat);
      lift::mjuu_addtovec(meshpos, mb.pos, 3);
      lift::mjuu_frameaccum(cg.pos, cg.quat, meshpos, mb.quat);
      // size = per-axis max |aamm|.
      cg.size[0] = std::max(std::abs(mb.aamm[0]), std::abs(mb.aamm[3]));
      cg.size[1] = std::max(std::abs(mb.aamm[1]), std::abs(mb.aamm[4]));
      cg.size[2] = std::max(std::abs(mb.aamm[2]), std::abs(mb.aamm[5]));
      // aabb = mesh aamm as (center, half).
      for (int k = 0; k < 3; ++k) {
        cg.aabb[k] = (mb.aamm[k + 3] + mb.aamm[k]) / 2;
        cg.aabb[k + 3] = (mb.aamm[k + 3] - mb.aamm[k]) / 2;
      }
      cg.rbound = std::sqrt(cg.size[0] * cg.size[0] + cg.size[1] * cg.size[1] +
                            cg.size[2] * cg.size[2]);
      cg.inferinertia = inferinertia;
      if (inferinertia) {
        auto set_mesh_inertia = [&]() {
          cg.inertia[0] = cg.mass_ * (mb.boxsz[1] * mb.boxsz[1] +
                                      mb.boxsz[2] * mb.boxsz[2]) / 3;
          cg.inertia[1] = cg.mass_ * (mb.boxsz[0] * mb.boxsz[0] +
                                      mb.boxsz[2] * mb.boxsz[2]) / 3;
          cg.inertia[2] = cg.mass_ * (mb.boxsz[0] * mb.boxsz[0] +
                                      mb.boxsz[1] * mb.boxsz[1]) / 3;
        };
        if (cg.has_mass) {
          if (mass == 0) { cg.mass_ = 0; cg.density = 0; }
          else if (mb.volume > lift::mjEPS) {
            cg.mass_ = mass;
            cg.density = mass / mb.volume;
            set_mesh_inertia();
          }
        } else if (cg.density != 0) {
          cg.mass_ = cg.density * mb.volume;
          set_mesh_inertia();
        }
      }
      return cg;
    }
  }

  GeomComputeAABB(cg.type, cg.size, cg.aabb);
  cg.inferinertia = inferinertia;
  if (inferinertia) {
    if (cg.has_mass) {
      if (mass == 0) { cg.mass_ = 0; cg.density = 0; }
      else if (GeomVolume(cg.type, cg.size, ti) > lift::mjEPS) {
        cg.mass_ = mass;
        cg.density = mass / GeomVolume(cg.type, cg.size, ti);
        GeomSetInertia(cg.type, cg.size, cg.mass_, ti, cg.inertia);
      }
    } else {
      if (cg.density == 0) cg.mass_ = 0;
      else {
        cg.mass_ = cg.density * GeomVolume(cg.type, cg.size, ti);
        GeomSetInertia(cg.type, cg.size, cg.mass_, ti, cg.inertia);
      }
    }
  }
  cg.rbound = GeomRBound(cg.type, cg.size);
  return cg;
}

// islimited (user_objects.cc:186-191): TrueState/range -> active limit.
bool IsLimited(int limited, const double range[2]) {
  return limited == mjLIMITED_TRUE ||
         (limited == mjLIMITED_AUTO && range[0] < range[1]);
}

// Compile one <joint> (mjCJoint::Compile). `cs` supplies degree/autolimits;
// `xf` is the enclosing frame (identity when none), applied to axis and pos.
CJoint JointCompile(const Model& model, const Joint& j, const CompilerSettings& cs,
                    int bodyid, const FrameXform& xf) {
  std::unique_ptr<Joint> eff = ps::sdk::Effective(model, j);
  CJoint cj;
  cj.src = &j;
  cj.bodyid = bodyid;
  cj.type = eff->type ? static_cast<int>(*eff->type) : mjJNT_HINGE;
  if (eff->group) cj.group = *eff->group;
  if (eff->pos) { cj.pos[0] = (*eff->pos)[0]; cj.pos[1] = (*eff->pos)[1];
                  cj.pos[2] = (*eff->pos)[2]; }
  if (eff->axis) { cj.axis[0] = (*eff->axis)[0]; cj.axis[1] = (*eff->axis)[1];
                   cj.axis[2] = (*eff->axis)[2]; }
  if (eff->range) { cj.range[0] = (*eff->range)[0]; cj.range[1] = (*eff->range)[1]; }
  if (eff->actuatorfrcrange) {
    cj.actfrcrange[0] = (*eff->actuatorfrcrange)[0];
    cj.actfrcrange[1] = (*eff->actuatorfrcrange)[1];
  }
  if (eff->margin) cj.margin = *eff->margin;
  if (eff->ref) cj.ref = *eff->ref;
  if (eff->springref) cj.springref = *eff->springref;
  if (eff->armature) cj.armature = *eff->armature;
  if (eff->frictionloss) cj.frictionloss = *eff->frictionloss;
  if (eff->actuatorgravcomp) cj.actgravcomp = *eff->actuatorgravcomp;
  if (eff->user) cj.user = *eff->user;
  if (eff->stiffness)
    for (std::size_t k = 0; k < eff->stiffness->size() && k < 3; ++k)
      cj.stiffness[k] = (*eff->stiffness)[k];
  if (eff->damping)
    for (std::size_t k = 0; k < eff->damping->size() && k < 3; ++k)
      cj.damping[k] = (*eff->damping)[k];
  if (eff->solreflimit)
    for (std::size_t k = 0; k < eff->solreflimit->size() && k < 2; ++k)
      cj.solref[k] = (*eff->solreflimit)[k];
  if (eff->solimplimit)
    for (std::size_t k = 0; k < eff->solimplimit->size() && k < 5; ++k)
      cj.solimp[k] = (*eff->solimplimit)[k];
  if (eff->solreffriction)
    for (std::size_t k = 0; k < eff->solreffriction->size() && k < 2; ++k)
      cj.solref_friction[k] = (*eff->solreffriction)[k];
  if (eff->solimpfriction)
    for (std::size_t k = 0; k < eff->solimpfriction->size() && k < 5; ++k)
      cj.solimp_friction[k] = (*eff->solimpfriction)[k];

  int limited = eff->limited ? static_cast<int>(*eff->limited) : mjLIMITED_AUTO;
  int actfrclimited =
      eff->actuatorfrclimited ? static_cast<int>(*eff->actuatorfrclimited)
                              : mjLIMITED_AUTO;

  // free joints cannot be limited
  if (cj.type == mjJNT_FREE) limited = mjLIMITED_FALSE;
  cj.limited = IsLimited(limited, cj.range);
  if (cj.limited && cs.degree &&
      (cj.type == mjJNT_HINGE || cj.type == mjJNT_BALL)) {
    if (cj.range[0]) cj.range[0] *= mjPI / 180.0;
    if (cj.range[1]) cj.range[1] *= mjPI / 180.0;
  }

  if (cj.type == mjJNT_FREE || cj.type == mjJNT_BALL)
    actfrclimited = mjLIMITED_FALSE;
  cj.actfrclimited = IsLimited(actfrclimited, cj.actfrcrange);

  // axis: FREE/BALL fixed to (0,0,1); HINGE/SLIDE accumulate frame rotation
  if (cj.type == mjJNT_FREE || cj.type == mjJNT_BALL) {
    cj.axis[0] = cj.axis[1] = 0; cj.axis[2] = 1;
  } else if (xf.present) {
    double ax[3] = {cj.axis[0], cj.axis[1], cj.axis[2]};
    lift::mjuu_rotVecQuat(cj.axis, ax, xf.quat);
  }
  lift::mjuu_normvec(cj.axis, 3);
  // pos: FREE fixed to (0,0,0); BALL/HINGE/SLIDE accumulate frame translation
  if (cj.type == mjJNT_FREE) {
    cj.pos[0] = cj.pos[1] = cj.pos[2] = 0;
  } else if (xf.present) {
    double qunit[4] = {1, 0, 0, 0};
    lift::mjuu_frameaccumChild(xf.pos, xf.quat, cj.pos, qunit);
  }

  // ref/springref to radians for hinge joints
  if (cj.type == mjJNT_HINGE && cs.degree) {
    cj.ref *= mjPI / 180.0;
    cj.springref *= mjPI / 180.0;
  }
  return cj;
}

// --------------------------------------------------------------------------- //
// Per-site compiled state (S7). Lifted from mjCSite::Compile.                  //
// --------------------------------------------------------------------------- //
struct CSite {
  const Site* src = nullptr;
  int bodyid = 0;
  int type = mjGEOM_SPHERE;
  int group = 0;
  int matid = -1;
  std::string material_name;
  double size[3] = {0.005, 0.005, 0.005};
  double pos[3] = {0, 0, 0};
  double quat[4] = {1, 0, 0, 0};
  float rgba[4] = {0.5f, 0.5f, 0.5f, 1.0f};
  std::vector<double> user;
};

CSite SiteCompile(const Model& model, const Site& s, const CompilerSettings& cs,
                  int bodyid, const FrameXform& xf) {
  std::unique_ptr<Site> eff = ps::sdk::Effective(model, s);
  CSite cs2;
  cs2.src = &s;
  cs2.bodyid = bodyid;
  cs2.type = eff->type ? static_cast<int>(*eff->type) : mjGEOM_SPHERE;
  if (eff->group) cs2.group = *eff->group;
  if (eff->size)
    for (std::size_t k = 0; k < eff->size->size() && k < 3; ++k)
      cs2.size[k] = (*eff->size)[k];
  if (eff->pos) { cs2.pos[0] = (*eff->pos)[0]; cs2.pos[1] = (*eff->pos)[1];
                  cs2.pos[2] = (*eff->pos)[2]; }
  if (eff->rgba) for (int k = 0; k < 4; ++k) cs2.rgba[k] = (*eff->rgba)[k];
  if (eff->material) cs2.material_name = eff->material->name;
  if (eff->user) cs2.user = *eff->user;

  // 'fromto' path (mjCSite::Compile): compute pos, quat, size like geoms.
  const FromTo* ft = nullptr;
  if (eff->shape) ft = std::get_if<FromTo>(&*eff->shape);
  if (ft) {
    const double* f = ft->fromto.data();
    double vec[3] = {f[0] - f[3], f[1] - f[4], f[2] - f[5]};
    cs2.size[1] = lift::mjuu_normvec(vec, 3) / 2;
    if (cs2.type == mjGEOM_ELLIPSOID || cs2.type == mjGEOM_BOX) {
      cs2.size[2] = cs2.size[1];
      cs2.size[1] = cs2.size[0];
    }
    cs2.pos[0] = (f[0] + f[3]) / 2; cs2.pos[1] = (f[1] + f[4]) / 2;
    cs2.pos[2] = (f[2] + f[5]) / 2;
    lift::mjuu_z2quat(cs2.quat, vec);
  } else {
    ResolveQuat(eff->orient, cs, cs2.quat);
  }
  // frame accumulation, then normalize (mjCSite::Compile order).
  if (xf.present) lift::mjuu_frameaccumChild(xf.pos, xf.quat, cs2.pos, cs2.quat);
  lift::mjuu_normvec(cs2.quat, 4);
  return cs2;
}

// --------------------------------------------------------------------------- //
// Per-camera compiled state (S7). Lifted from mjCCamera::Compile.              //
// --------------------------------------------------------------------------- //
struct CCamera {
  const Camera* src = nullptr;
  int bodyid = 0;
  int mode = mjCAMLIGHT_FIXED;
  int targetbodyid = -1;
  std::string targetbody;
  double pos[3] = {0, 0, 0};
  double quat[4] = {1, 0, 0, 0};
  int proj = 0;                       // mjtProjection (perspective)
  double fovy = 45;
  double ipd = 0.068;
  int resolution[2] = {1, 1};
  int output = mjCAMOUT_RGB;
  float sensorsize[2] = {0, 0};
  float intrinsic[4] = {0, 0, 0, 0};
  std::vector<double> user;
};

CCamera CameraCompile(const Model& model, const Camera& c,
                      const CompilerSettings& cs, int bodyid,
                      const FrameXform& xf, double znear) {
  std::unique_ptr<Camera> eff = ps::sdk::Effective(model, c);
  CCamera cc;
  cc.src = &c;
  cc.bodyid = bodyid;
  if (eff->pos) { cc.pos[0] = (*eff->pos)[0]; cc.pos[1] = (*eff->pos)[1];
                  cc.pos[2] = (*eff->pos)[2]; }
  if (eff->mode) cc.mode = static_cast<int>(*eff->mode);
  if (eff->ipd) cc.ipd = *eff->ipd;
  if (eff->resolution) { cc.resolution[0] = (*eff->resolution)[0];
                         cc.resolution[1] = (*eff->resolution)[1]; }
  if (eff->projection) cc.proj = static_cast<int>(*eff->projection);
  if (eff->target) cc.targetbody = eff->target->name;
  if (eff->user) cc.user = *eff->user;

  // orientation, frame, normalize (mjCCamera::Compile order).
  ResolveQuat(eff->orient, cs, cc.quat);
  if (xf.present) lift::mjuu_frameaccumChild(xf.pos, xf.quat, cc.pos, cc.quat);
  lift::mjuu_normvec(cc.quat, 4);

  // output bit flags (default RGB); intrinsics (fovy / focal / principal).
  if (eff->output && !eff->output->empty()) {
    cc.output = 0;
    for (CameraOutput o : *eff->output) cc.output |= (1 << static_cast<int>(o));
  }
  float focal_length[2] = {0, 0};  // mjsCamera.focal_length is float
  if (eff->intrinsics) {
    if (const Fovy* fv = std::get_if<Fovy>(&*eff->intrinsics)) cc.fovy = fv->fovy;
    else if (const Focal* fl = std::get_if<Focal>(&*eff->intrinsics)) {
      focal_length[0] = static_cast<float>(fl->focal[0]);
      focal_length[1] = static_cast<float>(fl->focal[1]);
    }
  }
  float focal_pixel[2] = {0, 0}, principal_length[2] = {0, 0},
        principal_pixel[2] = {0, 0}, sensor_size[2] = {0, 0};
  if (eff->focalpixel) { focal_pixel[0] = (*eff->focalpixel)[0];
                         focal_pixel[1] = (*eff->focalpixel)[1]; }
  if (eff->principal) { principal_length[0] = (*eff->principal)[0];
                        principal_length[1] = (*eff->principal)[1]; }
  if (eff->principalpixel) { principal_pixel[0] = (*eff->principalpixel)[0];
                             principal_pixel[1] = (*eff->principalpixel)[1]; }
  if (eff->sensorsize) { sensor_size[0] = (*eff->sensorsize)[0];
                         sensor_size[1] = (*eff->sensorsize)[1]; }
  cc.sensorsize[0] = sensor_size[0]; cc.sensorsize[1] = sensor_size[1];

  // compute intrinsic + fovy (user_objects.cc:4380-4430).
  if (sensor_size[0] > 0 && sensor_size[1] > 0) {
    float pixel_density[2] = {
        static_cast<float>(cc.resolution[0]) / sensor_size[0],
        static_cast<float>(cc.resolution[1]) / sensor_size[1]};
    cc.intrinsic[0] = focal_pixel[0] ? focal_pixel[0] / pixel_density[0]
                                     : focal_length[0];
    cc.intrinsic[1] = focal_pixel[1] ? focal_pixel[1] / pixel_density[1]
                                     : focal_length[1];
    cc.intrinsic[2] = principal_pixel[0] ? principal_pixel[0] / pixel_density[0]
                                         : principal_length[0];
    cc.intrinsic[3] = principal_pixel[1] ? principal_pixel[1] / pixel_density[1]
                                         : principal_length[1];
    cc.fovy = std::atan2(sensor_size[1] / 2, cc.intrinsic[1]) * 360.0 / mjPI;
  } else {
    cc.intrinsic[0] = static_cast<float>(znear);
    cc.intrinsic[1] = static_cast<float>(znear);
  }
  return cc;
}

// --------------------------------------------------------------------------- //
// Per-light compiled state (S7). Lifted from mjCLight::Compile.                //
// --------------------------------------------------------------------------- //
struct CLight {
  const Light* src = nullptr;
  int bodyid = 0;
  int mode = mjCAMLIGHT_FIXED;
  int targetbodyid = -1;
  std::string targetbody;
  int type = mjLIGHT_SPOT;
  int texid = -1;
  bool castshadow = true;
  bool active = true;
  double pos[3] = {0, 0, 0};
  double dir[3] = {0, 0, -1};
  float bulbradius = 0.02f, intensity = 0.0f, range = 10.0f;
  float attenuation[3] = {1, 0, 0};
  float cutoff = 45.0f, exponent = 10.0f;
  float ambient[3] = {0, 0, 0};
  float diffuse[3] = {0.7f, 0.7f, 0.7f};
  float specular[3] = {0.3f, 0.3f, 0.3f};
};

CLight LightCompile(const Model& model, const Light& l,
                    const CompilerSettings& cs, int bodyid, const FrameXform& xf) {
  (void)cs;
  std::unique_ptr<Light> eff = ps::sdk::Effective(model, l);
  CLight cl;
  cl.src = &l;
  cl.bodyid = bodyid;
  if (eff->mode) cl.mode = static_cast<int>(*eff->mode);
  // type: `directional` (legacy bool) maps to spot/directional; else `type`.
  if (eff->directional) cl.type = *eff->directional ? mjLIGHT_DIRECTIONAL
                                                     : mjLIGHT_SPOT;
  else if (eff->type) cl.type = static_cast<int>(*eff->type);
  if (eff->castshadow) cl.castshadow = *eff->castshadow;
  if (eff->active) cl.active = *eff->active;
  if (eff->pos) { cl.pos[0] = (*eff->pos)[0]; cl.pos[1] = (*eff->pos)[1];
                  cl.pos[2] = (*eff->pos)[2]; }
  if (eff->dir) { cl.dir[0] = (*eff->dir)[0]; cl.dir[1] = (*eff->dir)[1];
                  cl.dir[2] = (*eff->dir)[2]; }
  if (eff->bulbradius) cl.bulbradius = *eff->bulbradius;
  if (eff->intensity) cl.intensity = *eff->intensity;
  if (eff->range) cl.range = *eff->range;
  if (eff->attenuation) for (int k = 0; k < 3; ++k)
    cl.attenuation[k] = (*eff->attenuation)[k];
  if (eff->cutoff) cl.cutoff = *eff->cutoff;
  if (eff->exponent) cl.exponent = *eff->exponent;
  if (eff->ambient) for (int k = 0; k < 3; ++k) cl.ambient[k] = (*eff->ambient)[k];
  if (eff->diffuse) for (int k = 0; k < 3; ++k) cl.diffuse[k] = (*eff->diffuse)[k];
  if (eff->specular) for (int k = 0; k < 3; ++k)
    cl.specular[k] = (*eff->specular)[k];
  if (eff->target) cl.targetbody = eff->target->name;

  // frame: accumulate pos, rotate dir (mjCLight::Compile), then normalize dir.
  if (xf.present) {
    double qunit[4] = {1, 0, 0, 0};
    lift::mjuu_frameaccumChild(xf.pos, xf.quat, cl.pos, qunit);
    double d[3] = {cl.dir[0], cl.dir[1], cl.dir[2]};
    lift::mjuu_rotVecQuat(cl.dir, d, xf.quat);
  }
  lift::mjuu_normvec(cl.dir, 3);
  return cl;
}

// --------------------------------------------------------------------------- //
// Bounding volume hierarchy, lifted from mjCBoundingVolumeHierarchy            //
// (user_objects.cc:356-546). Consumes plain leaf arrays, emits bvh_* 1:1.      //
// --------------------------------------------------------------------------- //
struct BVHLeaf {
  int id;
  int contype, conaffinity;
  const double* pos;
  const double* quat;   // may be null (identity)
  const double* aabb;
};

class BVH {
 public:
  void Set(const double* ipos, const double* iquat) {
    lift::mjuu_copyvec(ipos_, ipos, 3);
    lift::mjuu_copyvec(iquat_, iquat, 4);
  }
  void Add(const BVHLeaf& l) { leaves_.push_back(l); }

  void Create() {
    struct BVElement { const BVHLeaf* e; double lpos[3]; };
    std::vector<BVElement> elems;
    elems.reserve(leaves_.size());
    double qinv[4] = {iquat_[0], -iquat_[1], -iquat_[2], -iquat_[3]};
    for (const BVHLeaf& l : leaves_) {
      if (l.conaffinity || l.contype) {
        BVElement e; e.e = &l;
        double vert[3] = {l.pos[0] - ipos_[0], l.pos[1] - ipos_[1], l.pos[2] - ipos_[2]};
        lift::mjuu_rotVecQuat(e.lpos, vert, qinv);
        elems.push_back(e);
      }
    }
    if (!elems.empty()) MakeBVH(elems.begin(), elems.end(), 0);
  }

  int nbvh() const { return static_cast<int>(child_.size() / 2); }
  const std::vector<double>& bvh() const { return bvh_; }
  const std::vector<int>& child() const { return child_; }
  const std::vector<int>& level() const { return level_; }
  const std::vector<int>& nodeid() const { return nodeid_; }

 private:
  template <class It>
  int MakeBVH(It begin, It end, int lev) {
    int nelements = static_cast<int>(end - begin);
    if (nelements == 0) return -1;
    constexpr double kMaxVal = std::numeric_limits<double>::max();
    double AAMM[6] = {kMaxVal, kMaxVal, kMaxVal, -kMaxVal, -kMaxVal, -kMaxVal};
    double qinv[4] = {iquat_[0], -iquat_[1], -iquat_[2], -iquat_[3]};
    for (auto it = begin; it != end; ++it) {
      const BVHLeaf* e = it->e;
      double aamm[6] = {e->aabb[0] - e->aabb[3], e->aabb[1] - e->aabb[4],
                        e->aabb[2] - e->aabb[5], e->aabb[0] + e->aabb[3],
                        e->aabb[1] + e->aabb[4], e->aabb[2] + e->aabb[5]};
      for (int v = 0; v < 8; ++v) {
        double vert[3], box[3];
        vert[0] = (v & 1 ? aamm[3] : aamm[0]);
        vert[1] = (v & 2 ? aamm[4] : aamm[1]);
        vert[2] = (v & 4 ? aamm[5] : aamm[2]);
        if (e->quat) {
          lift::mjuu_rotVecQuat(box, vert, e->quat);
          box[0] += e->pos[0] - ipos_[0]; box[1] += e->pos[1] - ipos_[1];
          box[2] += e->pos[2] - ipos_[2];
          lift::mjuu_rotVecQuat(vert, box, qinv);
        }
        AAMM[0] = std::min(AAMM[0], vert[0]); AAMM[1] = std::min(AAMM[1], vert[1]);
        AAMM[2] = std::min(AAMM[2], vert[2]); AAMM[3] = std::max(AAMM[3], vert[0]);
        AAMM[4] = std::max(AAMM[4], vert[1]); AAMM[5] = std::max(AAMM[5], vert[2]);
      }
    }
    for (int i = 0; i < 3; ++i)
      if (std::abs(AAMM[i] - AAMM[i + 3]) < lift::mjEPS) {
        AAMM[i] -= lift::mjEPS; AAMM[i + 3] += lift::mjEPS;
      }
    int index = static_cast<int>(child_.size() / 2);
    child_.push_back(-1); child_.push_back(-1);
    nodeid_.push_back(-1); level_.push_back(lev);
    bvh_.push_back((AAMM[3] + AAMM[0]) / 2); bvh_.push_back((AAMM[4] + AAMM[1]) / 2);
    bvh_.push_back((AAMM[5] + AAMM[2]) / 2); bvh_.push_back((AAMM[3] - AAMM[0]) / 2);
    bvh_.push_back((AAMM[4] - AAMM[1]) / 2); bvh_.push_back((AAMM[5] - AAMM[2]) / 2);
    if (nelements == 1) {
      nodeid_[index] = begin->e->id;
      return index;
    }
    int axis = 0;
    double edges[3] = {AAMM[3] - AAMM[0], AAMM[4] - AAMM[1], AAMM[5] - AAMM[2]};
    if (edges[1] >= edges[0] + lift::mjEPS) axis = 1;
    if (edges[2] >= edges[axis] + lift::mjEPS) axis = 2;
    auto compare = [&](const auto& e1, const auto& e2) {
      if (std::abs(e1.lpos[axis] - e2.lpos[axis]) > lift::mjEPS)
        return e1.lpos[axis] < e2.lpos[axis];
      return e1.e < e2.e;
    };
    int m = nelements / 2;
    std::nth_element(begin, begin + m, end, compare);
    if (m > 0) child_[2 * index + 0] = MakeBVH(begin, begin + m, lev + 1);
    if (m != nelements) child_[2 * index + 1] = MakeBVH(begin + m, end, lev + 1);
    return index;
  }

  double ipos_[3] = {0, 0, 0}, iquat_[4] = {1, 0, 0, 0};
  std::vector<BVHLeaf> leaves_;
  std::vector<double> bvh_;
  std::vector<int> child_, level_, nodeid_;
};

// --------------------------------------------------------------------------- //
// Frame flattening (S1): a body's direct children plus everything reachable    //
// through nested <frame> containers, in document order, each paired with its   //
// accumulated frame transform. Frames flatten away -- their pose lives in the  //
// child transforms, matching upstream where every element carries a `frame`.   //
// --------------------------------------------------------------------------- //
struct FramedChildren {
  std::vector<std::pair<const BodyChildAny*, FrameXform>> joints;  // Joint/Free
  std::vector<std::pair<const Geom*, FrameXform>> geoms;
  std::vector<std::pair<const Site*, FrameXform>> sites;
  std::vector<std::pair<const Camera*, FrameXform>> cameras;
  std::vector<std::pair<const Light*, FrameXform>> lights;
  std::vector<std::pair<const Body*, FrameXform>> bodies;
};

void FlattenChildren(const std::vector<BodyChildAny>& subtree,
                     const FrameXform& xf, const CompilerSettings& cs,
                     FramedChildren& out) {
  for (const BodyChildAny& child : subtree) {
    switch (child.kind()) {
      case BodyChildAny::Kind::Joint:
      case BodyChildAny::Kind::FreeJoint:
        out.joints.emplace_back(&child, xf);
        break;
      case BodyChildAny::Kind::Geom: {
        const auto& g = std::get<std::unique_ptr<Geom>>(child.node);
        if (g) out.geoms.emplace_back(g.get(), xf);
        break;
      }
      case BodyChildAny::Kind::Site: {
        const auto& s = std::get<std::unique_ptr<Site>>(child.node);
        if (s) out.sites.emplace_back(s.get(), xf);
        break;
      }
      case BodyChildAny::Kind::Camera: {
        const auto& c = std::get<std::unique_ptr<Camera>>(child.node);
        if (c) out.cameras.emplace_back(c.get(), xf);
        break;
      }
      case BodyChildAny::Kind::Light: {
        const auto& l = std::get<std::unique_ptr<Light>>(child.node);
        if (l) out.lights.emplace_back(l.get(), xf);
        break;
      }
      case BodyChildAny::Kind::Body: {
        const auto& b = std::get<std::unique_ptr<Body>>(child.node);
        if (b) out.bodies.emplace_back(b.get(), xf);
        break;
      }
      case BodyChildAny::Kind::Frame: {
        const auto& f = std::get<std::unique_ptr<Frame>>(child.node);
        if (f) {
          FrameXform sub = ResolveFrame(*f, cs, xf);
          FlattenChildren(f->subtree, sub, cs, out);
        }
        break;
      }
      default:
        break;  // PluginRef/Composite/Flexcomp/Replicate/Attach: gated out
    }
  }
}

// --------------------------------------------------------------------------- //
// S1 Collect: DFS pre-order over the body tree, document order == id order.    //
// --------------------------------------------------------------------------- //
class BodyCollector {
 public:
  BodyCollector(const Model& model, const CompilerSettings& cs,
                const bridge::CompileOptions& opts, double znear,
                const AssetBinds& assets)
      : model_(model), cs_(cs), opts_(opts), znear_(znear), assets_(assets) {}

  // Auto-name mirror (bridge Collector): authored name else _ps:joint:<serial>.
  std::string JointName(const ps::opt<std::string>& nm, std::uint64_t serial) {
    if (nm) return *nm;
    if (opts_.auto_name)
      return opts_.auto_name_prefix + "joint:" + std::to_string(serial);
    return "";
  }

  // Collect the world body (slot 0) and the subtree. MuJoCo merges every
  // <worldbody> block (e.g. from includes) into the single world body, so all
  // entries' direct geoms belong to body 0 (contiguous, in document order) and
  // all entries' direct child bodies are the top-level bodies.
  void Run(const std::vector<std::unique_ptr<Body>>& worldbodies) {
    CBody world;
    world.src = worldbodies.empty() ? nullptr : worldbodies.front().get();
    world.parentid = 0;
    world.weldid = 0;
    bodies_.push_back(world);  // reserve id 0

    // Flatten every <worldbody> block (merged into world by MuJoCo includes),
    // descending frames, in document order.
    FramedChildren fc;
    FrameXform identity;
    for (const auto& wb : worldbodies)
      if (wb) FlattenChildren(wb->subtree, identity, cs_, fc);

    // World's geoms (id 0, no inertia inference), sites, cameras, lights.
    const int gstart = static_cast<int>(geoms_.size());
    for (const auto& [g, xf] : fc.geoms) {
      CGeom cg = GeomCompile(model_, *g, cs_, 0, false, assets_);
      if (xf.present) lift::mjuu_frameaccumChild(xf.pos, xf.quat, cg.pos, cg.quat);
      geoms_.push_back(cg);
    }
    CBody& w = bodies_[0];
    w.geomnum = static_cast<int>(geoms_.size()) - gstart;
    w.geomadr = w.geomnum ? gstart : -1;
    for (int j = gstart; j < static_cast<int>(geoms_.size()); ++j) {
      w.contype |= geoms_[j].contype;
      w.conaffinity |= geoms_[j].conaffinity;
      w.margin = std::max(w.margin, geoms_[j].margin);
    }
    CollectVisual(fc, 0);
    // world (id 0): ipos = body frame (0, identity); no inertia inference.
    lift::mjuu_copyvec(w.ipos, w.pos, 3);
    lift::mjuu_copyvec(w.iquat, w.quat, 4);
    BuildBVH(w);

    // Top-level bodies (each carrying its enclosing frame), in document order.
    for (const auto& [b, xf] : fc.bodies) Collect(*b, 0, xf);
  }

  std::vector<CBody>& bodies() { return bodies_; }
  std::vector<CGeom>& geoms() { return geoms_; }
  std::vector<CSite>& sites() { return sites_; }
  std::vector<CCamera>& cameras() { return cameras_; }
  std::vector<CLight>& lights() { return lights_; }

 private:
  // Compile this body's sites, cameras, lights (mjCBody child lists), appending
  // to the global lists in document order (== id order).
  void CollectVisual(const FramedChildren& fc, int bodyid) {
    for (const auto& [s, xf] : fc.sites)
      sites_.push_back(SiteCompile(model_, *s, cs_, bodyid, xf));
    for (const auto& [c, xf] : fc.cameras)
      cameras_.push_back(CameraCompile(model_, *c, cs_, bodyid, xf, znear_));
    for (const auto& [l, xf] : fc.lights)
      lights_.push_back(LightCompile(model_, *l, cs_, bodyid, xf));
  }

  void BuildBVH(CBody& cb) {
    if (cb.geomnum <= 0) return;
    BVH tree;
    tree.Set(cb.ipos, cb.iquat);
    for (int j = cb.geomadr; j < cb.geomadr + cb.geomnum; ++j)
      tree.Add(BVHLeaf{j, geoms_[j].contype, geoms_[j].conaffinity,
                       geoms_[j].pos, geoms_[j].quat, geoms_[j].aabb});
    tree.Create();
    cb.bvh = tree.bvh();
    cb.bvh_child = tree.child();
    cb.bvh_level = tree.level();
    cb.bvh_nodeid = tree.nodeid();
  }

  // Accumulate body inertial frame from its geoms (mjCBody::InertiaFromGeom).
  void InertiaFromGeom(CBody& cb) {
    std::vector<const CGeom*> sel;
    for (int j = cb.geomadr; j < cb.geomadr + cb.geomnum; ++j)
      if (geoms_[j].group >= cs_.inertiagrouprange[0] &&
          geoms_[j].group <= cs_.inertiagrouprange[1] &&
          geoms_[j].mass_ > lift::mjEPS)
        sel.push_back(&geoms_[j]);
    const int sz = static_cast<int>(sel.size());
    if (sz == 1) {
      lift::mjuu_copyvec(cb.ipos, sel[0]->pos, 3);
      lift::mjuu_copyvec(cb.iquat, sel[0]->quat, 4);
      cb.mass = sel[0]->mass_;
      lift::mjuu_copyvec(cb.inertia, sel[0]->inertia, 3);
    } else if (sz > 1) {
      double com[3] = {0, 0, 0}, toti[6] = {0, 0, 0, 0, 0, 0};
      cb.mass = 0;
      for (int i = 0; i < sz; ++i) {
        cb.mass += sel[i]->mass_;
        com[0] += sel[i]->mass_ * sel[i]->pos[0];
        com[1] += sel[i]->mass_ * sel[i]->pos[1];
        com[2] += sel[i]->mass_ * sel[i]->pos[2];
      }
      cb.ipos[0] = com[0] / cb.mass; cb.ipos[1] = com[1] / cb.mass;
      cb.ipos[2] = com[2] / cb.mass;
      for (int i = 0; i < sz; ++i) {
        double inert0[6], inert1[6];
        double dpos[3] = {sel[i]->pos[0] - cb.ipos[0], sel[i]->pos[1] - cb.ipos[1],
                          sel[i]->pos[2] - cb.ipos[2]};
        lift::mjuu_globalinertia(inert0, sel[i]->inertia, sel[i]->quat);
        lift::mjuu_offcenter(inert1, sel[i]->mass_, dpos);
        for (int k = 0; k < 6; ++k) toti[k] += inert0[k] + inert1[k];
      }
      lift::mjuu_fullInertia(cb.iquat, cb.inertia, toti);
    }
    cb.has_inertial = (sz > 0);  // ipos now "defined" if any geom contributed
  }

  int Collect(const Body& b, int parentid, const FrameXform& body_xf) {
    const int id = static_cast<int>(bodies_.size());
    CBody cb;
    cb.src = &b;
    cb.parentid = parentid < 0 ? 0 : parentid;

    // Flatten this body's children through nested frames, in document order.
    FramedChildren fc;
    FrameXform identity;
    FlattenChildren(b.subtree, identity, cs_, fc);

    // has_joints: any <joint>/<freejoint> (frames flattened). Determines weld.
    cb.has_joints = !fc.joints.empty();
    cb.weldid = cb.has_joints ? id : (id == 0 ? 0 : bodies_[cb.parentid].weldid);
    if (b.pos) { cb.pos[0] = (*b.pos)[0]; cb.pos[1] = (*b.pos)[1];
                 cb.pos[2] = (*b.pos)[2]; }
    ResolveQuat(b.orient, cs_, cb.quat);
    lift::mjuu_normvec(cb.quat, 4);
    cb.mocap = b.mocap && *b.mocap;
    if (b.user) cb.user = *b.user;
    const bool explicitinertial = !b.inertial.empty() && b.inertial.front();
    if (explicitinertial) ResolveInertial(*b.inertial.front(), cs_, cb);

    // Compile this body's joints (grouped by body); assign global jnt id and
    // qpos/dof addresses via running cursors (CopyTree cursor order).
    const int jstart = static_cast<int>(joints_.size());
    for (const auto& [child, xf] : fc.joints) {
      CJoint cj;
      if (child->kind() == BodyChildAny::Kind::Joint) {
        const auto& j = std::get<std::unique_ptr<Joint>>(child->node);
        if (!j) continue;
        cj = JointCompile(model_, *j, cs_, id, xf);
        cj.name = JointName(j->name, j->serial);
      } else {
        const auto& fj = std::get<std::unique_ptr<FreeJoint>>(child->node);
        if (!fj) continue;
        cj.src = fj.get();
        cj.bodyid = id;
        cj.type = mjJNT_FREE;
        if (fj->group) cj.group = *fj->group;
        cj.axis[0] = cj.axis[1] = 0; cj.axis[2] = 1;
        cj.name = JointName(fj->name, fj->serial);
      }
      cj.qposadr = qpos_cursor_;
      cj.dofadr = dof_cursor_;
      qpos_cursor_ += cj.nq();
      dof_cursor_ += cj.nv();
      joints_.push_back(cj);
    }
    cb.jntnum = static_cast<int>(joints_.size()) - jstart;
    cb.jntadr = cb.jntnum ? jstart : -1;
    cb.dofnum = 0;
    for (int j = jstart; j < static_cast<int>(joints_.size()); ++j)
      cb.dofnum += joints_[j].nv();
    cb.dofadr = cb.dofnum ? joints_[jstart].dofadr : -1;

    // Compile this body's own geoms (grouped by body, MakeTreeLists order).
    const int gstart = static_cast<int>(geoms_.size());
    for (const auto& [g, xf] : fc.geoms) {
      const bool infer =
          id > 0 &&
          (!explicitinertial || cs_.inertiafromgeom == InertiaFromGeom::true_);
      CGeom cg = GeomCompile(model_, *g, cs_, id, false, assets_);
      cg.inferinertia = infer && cg.group >= cs_.inertiagrouprange[0] &&
                        cg.group <= cs_.inertiagrouprange[1];
      if (cg.inferinertia) {
        // recompute mass/inertia now that the group gate is known
        cg = GeomCompile(model_, *g, cs_, id, true, assets_);
      }
      // frame accumulation is the final touch on geom pos/quat (upstream).
      if (xf.present) lift::mjuu_frameaccumChild(xf.pos, xf.quat, cg.pos, cg.quat);
      geoms_.push_back(cg);
    }
    cb.geomnum = static_cast<int>(geoms_.size()) - gstart;
    cb.geomadr = cb.geomnum ? gstart : -1;

    // Sites, cameras, lights on this body (document order == id order).
    CollectVisual(fc, id);

    // Accumulate contype/conaffinity/margin over geoms (mjCBody::Compile).
    for (int j = gstart; j < static_cast<int>(geoms_.size()); ++j) {
      cb.contype |= geoms_[j].contype;
      cb.conaffinity |= geoms_[j].conaffinity;
      cb.margin = std::max(cb.margin, geoms_[j].margin);
    }

    // Inertial frame from geoms if needed (mjCBody::Compile).
    if (id > 0 && (cs_.inertiafromgeom == InertiaFromGeom::true_ ||
                   (!explicitinertial &&
                    cs_.inertiafromgeom == InertiaFromGeom::auto_))) {
      InertiaFromGeom(cb);
    }
    // ipos undefined: copy body frame into inertial.
    if (!cb.has_inertial) {
      lift::mjuu_copyvec(cb.ipos, cb.pos, 3);
      lift::mjuu_copyvec(cb.iquat, cb.quat, 4);
    } else {
      lift::mjuu_normvec(cb.iquat, 4);
    }
    if (id > 0) {
      cb.mass = std::max(cb.mass, cs_.boundmass);
      for (int k = 0; k < 3; ++k)
        cb.inertia[k] = std::max(cb.inertia[k], cs_.boundinertia);
      if (cb.inertia[0] + cb.inertia[1] < cb.inertia[2] ||
          cb.inertia[0] + cb.inertia[2] < cb.inertia[1] ||
          cb.inertia[1] + cb.inertia[2] < cb.inertia[0]) {
        if (cs_.balanceinertia)
          cb.inertia[0] = cb.inertia[1] = cb.inertia[2] =
              (cb.inertia[0] + cb.inertia[1] + cb.inertia[2]) / 3.0;
      }
    }

    // frame: accumulate the enclosing frame into the body frame, after the
    // inertial-frame copy (mjCBody::Compile order).
    if (body_xf.present)
      lift::mjuu_frameaccumChild(body_xf.pos, body_xf.quat, cb.pos, cb.quat);

    // BVH over this body's geoms (mjCBody::ComputeBVH).
    BuildBVH(cb);

    bodies_.push_back(cb);

    // Recurse into child bodies in document order (frames flattened).
    for (const auto& [child, xf] : fc.bodies) Collect(*child, id, xf);
    return id;
  }

  const Model& model_;
  const CompilerSettings& cs_;
  const bridge::CompileOptions& opts_;
  double znear_ = 0.01;
  const AssetBinds& assets_;
  std::vector<CBody> bodies_;
  std::vector<CGeom> geoms_;
  std::vector<CJoint> joints_;
  std::vector<CSite> sites_;
  std::vector<CCamera> cameras_;
  std::vector<CLight> lights_;
  int qpos_cursor_ = 0, dof_cursor_ = 0;

 public:
  std::vector<CJoint>& joints() { return joints_; }
};

// --------------------------------------------------------------------------- //
// S9/S7 dof tree + sparse sizes. Computes dof_parentid/bodyid, body_simple      //
// (all demotions + level-2), dof_simplenum, tree ids, subtreemass precursors,   //
// and nM/nD/nB/nC/ntree -- lifted from ComputeSparseSizes + CopyTree +          //
// FinalizeSimple so fill and sizes agree by construction.                       //
// --------------------------------------------------------------------------- //
struct DofTree {
  std::vector<int> dof_parentid, dof_bodyid, dof_jntid, dof_simplenum, dof_treeid;
  std::vector<int> body_rootid, body_simple, body_treeid, body_lastdof;
  std::vector<int> body_subtreedofs;
  int nM = 0, nD = 0, nB = 0, nC = 0, ntree = 0;
};

DofTree ComputeDofTree(const std::vector<CBody>& cbs,
                       const std::vector<CJoint>& joints, int nv) {
  const int nbody = static_cast<int>(cbs.size());
  DofTree t;
  t.dof_parentid.assign(nv, -1); t.dof_bodyid.assign(nv, -1);
  t.dof_jntid.assign(nv, -1); t.dof_simplenum.assign(nv, 0);
  t.dof_treeid.assign(nv, 0);
  t.body_rootid.assign(nbody, 0); t.body_simple.assign(nbody, 0);
  t.body_treeid.assign(nbody, -1); t.body_lastdof.assign(nbody, -1);
  t.body_subtreedofs.assign(nbody, 0);

  // Per-body joint index ranges.
  auto sameframe = [&](int i) {
    const CBody& cb = cbs[i];
    if (IsNullPose(cb.ipos, cb.iquat)) return mjSAMEFRAME_BODY;
    if (IsNullPose(nullptr, cb.iquat)) return mjSAMEFRAME_BODYROT;
    return mjSAMEFRAME_NONE;
  };

  // dof_parentid / dof_bodyid / dof_jntid via body lastdof (CopyTree).
  for (int i = 0; i < nbody; ++i) {
    const CBody& cb = cbs[i];
    t.body_lastdof[i] = (i == 0) ? -1 : t.body_lastdof[cb.parentid];
    for (int jj = cb.jntadr; jj < cb.jntadr + cb.jntnum && cb.jntadr >= 0; ++jj) {
      for (int k = 0; k < joints[jj].nv(); ++k) {
        int dofadr = joints[jj].dofadr + k;
        t.dof_bodyid[dofadr] = i;
        t.dof_jntid[dofadr] = jj;
        t.dof_parentid[dofadr] = t.body_lastdof[i];
        t.body_lastdof[i] = dofadr;
      }
    }
  }

  // body_rootid, body_simple (initial + joint demotions + level-2), treeids.
  for (int i = 0; i < nbody; ++i) {
    const CBody& cb = cbs[i];
    int parentid = cb.parentid;
    if (i == 0 || parentid == 0) t.body_rootid[i] = i;
    else t.body_rootid[i] = t.body_rootid[parentid];

    int sf = sameframe(i);
    t.body_simple[i] =
        (sf == mjSAMEFRAME_BODY &&
         (t.body_rootid[i] == i ||
          (cbs[parentid].parentid == 0 && cbs[parentid].dofnum == 0))) ? 1 : 0;
    if (parentid > 0) t.body_simple[parentid] = 0;

    int rotfound = 0;
    for (int jj = cb.jntadr; jj < cb.jntadr + cb.jntnum && cb.jntadr >= 0; ++jj) {
      const CJoint& pj = joints[jj];
      bool axis_aligned = ((std::abs(pj.axis[0]) > lift::mjEPS) +
                           (std::abs(pj.axis[1]) > lift::mjEPS) +
                           (std::abs(pj.axis[2]) > lift::mjEPS)) == 1;
      if (rotfound || !IsNullPose(pj.pos, nullptr) ||
          ((pj.type == mjJNT_HINGE || pj.type == mjJNT_SLIDE) && !axis_aligned))
        t.body_simple[i] = 0;
      if (pj.type == mjJNT_BALL || pj.type == mjJNT_HINGE) rotfound = 1;
    }
    if (t.body_simple[i] && cb.dofnum) {
      t.body_simple[i] = 2;
      for (int jj = cb.jntadr; jj < cb.jntadr + cb.jntnum; ++jj)
        if (joints[jj].type != mjJNT_SLIDE) { t.body_simple[i] = 1; break; }
    }
  }

  // nM, nD
  for (int i = 0; i < nv; ++i) {
    int j = i;
    while (j != -1) { t.nM++; j = t.dof_parentid[j]; }
  }
  t.nD = 2 * t.nM - nv;

  // subtreedofs, nB
  for (int i = nbody - 1; i >= 0; --i) {
    t.body_subtreedofs[i] += cbs[i].dofnum;
    if (i > 0) t.body_subtreedofs[cbs[i].parentid] += t.body_subtreedofs[i];
  }
  for (int i = 0; i < nbody; ++i) {
    t.nB += t.body_subtreedofs[i];
    int p = cbs[i].parentid;
    while (p > 0) { t.nB += cbs[p].dofnum; p = cbs[p].parentid; }
  }

  // dof_simplenum (FinalizeSimple), nC
  int count = 0;
  for (int i = nv - 1; i >= 0; --i) {
    if (t.body_simple[t.dof_bodyid[i]]) count++;
    else count = 0;
    t.dof_simplenum[i] = count;
  }
  int nOD = 0;
  for (int i = 0; i < nv; ++i)
    if (!t.dof_simplenum[i]) {
      int j = i;
      while (j >= 0) { if (j != i) nOD++; j = t.dof_parentid[j]; }
    }
  t.nC = nOD + nv;

  // dof_treeid, ntree, body_treeid
  int ntree = 0;
  for (int i = 0; i < nv; ++i) {
    if (t.dof_parentid[i] == -1) ntree++;
    t.dof_treeid[i] = ntree - 1;
  }
  t.ntree = ntree;
  for (int i = 0; i < nbody; ++i) {
    int weldid = cbs[i].weldid;
    if (cbs[weldid].dofnum) t.body_treeid[i] = t.dof_treeid[cbs[weldid].dofadr];
    else t.body_treeid[i] = -1;
  }
  return t;
}

// --------------------------------------------------------------------------- //
// S11 tree fill: body arrays.                                                  //
// --------------------------------------------------------------------------- //
void FillTree(mjModel* m, const std::vector<CBody>& cbs,
              const std::vector<CGeom>& geoms, const std::vector<CJoint>& joints,
              const DofTree& dt) {
  int bvh_adr = 0;
  for (int i = 0; i < static_cast<int>(cbs.size()); ++i) {
    const CBody& cb = cbs[i];
    m->body_parentid[i] = cb.parentid;
    m->body_weldid[i] = cb.weldid;
    m->body_mocapid[i] = -1;            // assigned in a later pass
    m->body_jntnum[i] = cb.jntnum;
    m->body_jntadr[i] = cb.jntadr;
    m->body_dofnum[i] = cb.dofnum;
    m->body_dofadr[i] = cb.dofadr;
    m->body_geomnum[i] = cb.geomnum;
    m->body_geomadr[i] = cb.geomadr;
    m->body_contype[i] = cb.contype;
    m->body_conaffinity[i] = cb.conaffinity;
    m->body_margin[i] = cb.margin;
    const int nbvh = static_cast<int>(cb.bvh_child.size() / 2);
    m->body_bvhadr[i] = nbvh ? bvh_adr : -1;
    m->body_bvhnum[i] = nbvh;
    for (int k = 0; k < nbvh; ++k) {
      lift::mjuu_copyvec(m->bvh_aabb + 6 * (bvh_adr + k), cb.bvh.data() + 6 * k, 6);
      m->bvh_child[2 * (bvh_adr + k)] = cb.bvh_child[2 * k];
      m->bvh_child[2 * (bvh_adr + k) + 1] = cb.bvh_child[2 * k + 1];
      m->bvh_depth[bvh_adr + k] = cb.bvh_level[k];
      m->bvh_nodeid[bvh_adr + k] = cb.bvh_nodeid[k];
    }
    bvh_adr += nbvh;
    m->body_plugin[i] = -1;
    lift::mjuu_setvec(m->body_pos + 3 * i, cb.pos[0], cb.pos[1], cb.pos[2]);
    lift::mjuu_setvec(m->body_quat + 4 * i, cb.quat[0], cb.quat[1], cb.quat[2],
                      cb.quat[3]);
    lift::mjuu_setvec(m->body_ipos + 3 * i, cb.ipos[0], cb.ipos[1], cb.ipos[2]);
    lift::mjuu_setvec(m->body_iquat + 4 * i, cb.iquat[0], cb.iquat[1],
                      cb.iquat[2], cb.iquat[3]);
    m->body_mass[i] = cb.mass;
    lift::mjuu_setvec(m->body_inertia + 3 * i, cb.inertia[0], cb.inertia[1],
                      cb.inertia[2]);
    m->body_gravcomp[i] = cb.src && cb.src->gravcomp ? *cb.src->gravcomp : 0;
    for (int k = 0; k < m->nuser_body && k < static_cast<int>(cb.user.size()); ++k)
      m->body_user[m->nuser_body * i + k] = cb.user[k];
  }

  // body_rootid, body_sameframe (CopyTree), and body_simple/body_treeid from the
  // precomputed dof tree (so sizes and fill agree).
  const int nbody = static_cast<int>(cbs.size());
  for (int i = 0; i < nbody; ++i) {
    m->body_rootid[i] = dt.body_rootid[i];
    int sameframe;
    if (IsNullPose(m->body_ipos + 3 * i, m->body_iquat + 4 * i))
      sameframe = mjSAMEFRAME_BODY;
    else if (IsNullPose(nullptr, m->body_iquat + 4 * i))
      sameframe = mjSAMEFRAME_BODYROT;
    else
      sameframe = mjSAMEFRAME_NONE;
    m->body_sameframe[i] = static_cast<mjtByte>(sameframe);
    m->body_simple[i] = static_cast<mjtByte>(dt.body_simple[i]);
    m->body_treeid[i] = dt.body_treeid[i];
  }

  // tree sleep policy (AUTO for all trees; NC1 has no non-default <body sleep>).
  for (int i = 0; i < dt.ntree; ++i) m->tree_sleep_policy[i] = mjSLEEP_AUTO;

  // body_subtreemass: sum of self + descendants (leaf-up).
  for (int i = 0; i < nbody; ++i) m->body_subtreemass[i] = m->body_mass[i];
  for (int i = nbody - 1; i > 0; --i)
    m->body_subtreemass[m->body_parentid[i]] += m->body_subtreemass[i];

  // Joint fill (CopyTree jnt loop) + qpos0/qpos_spring.
  for (int jid = 0; jid < static_cast<int>(joints.size()); ++jid) {
    const CJoint& pj = joints[jid];
    m->jnt_type[jid] = pj.type;
    m->jnt_group[jid] = pj.group;
    m->jnt_limited[jid] = static_cast<mjtByte>(pj.limited);
    m->jnt_actfrclimited[jid] = static_cast<mjtByte>(pj.actfrclimited);
    m->jnt_actgravcomp[jid] = pj.actgravcomp ? 1 : 0;
    m->jnt_qposadr[jid] = pj.qposadr;
    m->jnt_dofadr[jid] = pj.dofadr;
    m->jnt_bodyid[jid] = pj.bodyid;
    lift::mjuu_copyvec(m->jnt_pos + 3 * jid, pj.pos, 3);
    lift::mjuu_copyvec(m->jnt_axis + 3 * jid, pj.axis, 3);
    m->jnt_stiffness[jid] = pj.stiffness[0];
    lift::mjuu_copyvec(m->jnt_stiffnesspoly + mjNPOLY * jid, pj.stiffness + 1, mjNPOLY);
    lift::mjuu_copyvec(m->jnt_range + 2 * jid, pj.range, 2);
    lift::mjuu_copyvec(m->jnt_actfrcrange + 2 * jid, pj.actfrcrange, 2);
    lift::mjuu_copyvec(m->jnt_solref + mjNREF * jid, pj.solref, mjNREF);
    lift::mjuu_copyvec(m->jnt_solimp + mjNIMP * jid, pj.solimp, mjNIMP);
    m->jnt_margin[jid] = pj.margin;
    for (int k = 0; k < m->nuser_jnt && k < static_cast<int>(pj.user.size()); ++k)
      m->jnt_user[m->nuser_jnt * jid + k] = pj.user[k];

    const int qa = pj.qposadr;
    const CBody& pb = cbs[pj.bodyid];
    switch (pj.type) {
      case mjJNT_FREE:
        lift::mjuu_copyvec(m->qpos0 + qa, pb.pos, 3);
        lift::mjuu_copyvec(m->qpos0 + qa + 3, pb.quat, 4);
        lift::mjuu_copyvec(m->qpos_spring + qa, m->qpos0 + qa, 7);
        break;
      case mjJNT_BALL:
        m->qpos0[qa] = 1; m->qpos0[qa + 1] = 0; m->qpos0[qa + 2] = 0;
        m->qpos0[qa + 3] = 0;
        lift::mjuu_copyvec(m->qpos_spring + qa, m->qpos0 + qa, 4);
        break;
      default:  // slide / hinge
        m->qpos0[qa] = pj.ref;
        m->qpos_spring[qa] = pj.springref;
        break;
    }
  }

  // Dof fill (CopyTree dof loop) + dof_parentid/treeid/simplenum.
  for (int d = 0; d < static_cast<int>(dt.dof_parentid.size()); ++d) {
    const CJoint& pj = joints[dt.dof_jntid[d]];
    m->dof_bodyid[d] = dt.dof_bodyid[d];
    m->dof_jntid[d] = dt.dof_jntid[d];
    m->dof_parentid[d] = dt.dof_parentid[d];
    m->dof_treeid[d] = dt.dof_treeid[d];
    m->dof_simplenum[d] = dt.dof_simplenum[d];
    lift::mjuu_copyvec(m->dof_solref + mjNREF * d, pj.solref_friction, mjNREF);
    lift::mjuu_copyvec(m->dof_solimp + mjNIMP * d, pj.solimp_friction, mjNIMP);
    m->dof_frictionloss[d] = pj.frictionloss;
    m->dof_armature[d] = pj.armature;
    m->dof_damping[d] = pj.damping[0];
    lift::mjuu_copyvec(m->dof_dampingpoly + mjNPOLY * d, pj.damping + 1, mjNPOLY);
  }
  // dof_Madr (CopyTree): running ancestor-chain address.
  int nM_adr = 0;
  for (int d = 0; d < static_cast<int>(dt.dof_parentid.size()); ++d) {
    m->dof_Madr[d] = nM_adr;
    int j = d;
    while (j >= 0) { nM_adr++; j = dt.dof_parentid[j]; }
  }

  // Geom fill (CopyTree geom loop). matid/dataid are -1 in NC1 scope (no
  // material/mesh/hfield refs -- gated). sameframe uses the body's inertial pose.
  for (int gid = 0; gid < static_cast<int>(geoms.size()); ++gid) {
    const CGeom& cg = geoms[gid];
    const int b = cg.bodyid;
    m->geom_type[gid] = cg.type;
    m->geom_contype[gid] = cg.contype;
    m->geom_conaffinity[gid] = cg.conaffinity;
    m->geom_condim[gid] = cg.condim;
    m->geom_bodyid[gid] = b;
    m->geom_dataid[gid] = cg.dataid;
    m->geom_matid[gid] = cg.matid;
    m->geom_plugin[gid] = -1;
    m->geom_group[gid] = cg.group;
    m->geom_priority[gid] = cg.priority;
    lift::mjuu_copyvec(m->geom_size + 3 * gid, cg.size, 3);
    lift::mjuu_copyvec(m->geom_aabb + 6 * gid, cg.aabb, 6);
    lift::mjuu_copyvec(m->geom_pos + 3 * gid, cg.pos, 3);
    lift::mjuu_copyvec(m->geom_quat + 4 * gid, cg.quat, 4);
    lift::mjuu_copyvec(m->geom_friction + 3 * gid, cg.friction, 3);
    m->geom_solmix[gid] = cg.solmix;
    lift::mjuu_copyvec(m->geom_solref + mjNREF * gid, cg.solref, mjNREF);
    lift::mjuu_copyvec(m->geom_solimp + mjNIMP * gid, cg.solimp, mjNIMP);
    m->geom_margin[gid] = cg.margin;
    m->geom_gap[gid] = cg.gap;
    lift::mjuu_copyvec(m->geom_fluid + mjNFLUID * gid, cg.fluid, mjNFLUID);
    for (int k = 0; k < 4; ++k) m->geom_rgba[4 * gid + k] = cg.rgba[k];
    for (int k = 0; k < m->nuser_geom && k < static_cast<int>(cg.user.size()); ++k)
      m->geom_user[m->nuser_geom * gid + k] = cg.user[k];

    const double* ipos = m->body_ipos + 3 * b;
    const double* iquat = m->body_iquat + 4 * b;
    int sf;
    if (IsNullPose(cg.pos, cg.quat)) sf = mjSAMEFRAME_BODY;
    else if (IsNullPose(nullptr, cg.quat)) sf = mjSAMEFRAME_BODYROT;
    else if (IsSamePose(cg.pos, ipos, cg.quat, iquat)) sf = mjSAMEFRAME_INERTIA;
    else if (IsSamePose(nullptr, nullptr, cg.quat, iquat)) sf = mjSAMEFRAME_INERTIAROT;
    else sf = mjSAMEFRAME_NONE;
    m->geom_sameframe[gid] = static_cast<mjtByte>(sf);
    m->geom_rbound[gid] = cg.rbound;
  }
}

// --------------------------------------------------------------------------- //
// S11 visual fill: sites, cameras, lights (CopyObjects). targetbodyid resolved //
// via the body name -> id map. cam_pos0/mat0/light_pos0/dir0 are the reference  //
// configuration and are filled by mj_setConst (finalize).                       //
// --------------------------------------------------------------------------- //
void FillVisual(mjModel* m, const std::vector<CSite>& sites,
                const std::vector<CCamera>& cameras,
                const std::vector<CLight>& lights,
                const std::unordered_map<std::string, int>& bodyid) {
  auto resolve = [&](const std::string& nm) -> int {
    if (nm.empty()) return -1;
    auto it = bodyid.find(nm);
    return it == bodyid.end() ? -1 : it->second;
  };

  for (int sid = 0; sid < static_cast<int>(sites.size()); ++sid) {
    const CSite& cs = sites[sid];
    const int b = cs.bodyid;
    m->site_type[sid] = cs.type;
    m->site_bodyid[sid] = b;
    m->site_matid[sid] = cs.matid;
    m->site_group[sid] = cs.group;
    lift::mjuu_copyvec(m->site_size + 3 * sid, cs.size, 3);
    lift::mjuu_copyvec(m->site_pos + 3 * sid, cs.pos, 3);
    lift::mjuu_copyvec(m->site_quat + 4 * sid, cs.quat, 4);
    for (int k = 0; k < 4; ++k) m->site_rgba[4 * sid + k] = cs.rgba[k];
    for (int k = 0; k < m->nuser_site && k < static_cast<int>(cs.user.size()); ++k)
      m->site_user[m->nuser_site * sid + k] = cs.user[k];

    const double* ipos = m->body_ipos + 3 * b;
    const double* iquat = m->body_iquat + 4 * b;
    int sf;
    if (IsNullPose(cs.pos, cs.quat)) sf = mjSAMEFRAME_BODY;
    else if (IsNullPose(nullptr, cs.quat)) sf = mjSAMEFRAME_BODYROT;
    else if (IsSamePose(cs.pos, ipos, cs.quat, iquat)) sf = mjSAMEFRAME_INERTIA;
    else if (IsSamePose(nullptr, nullptr, cs.quat, iquat)) sf = mjSAMEFRAME_INERTIAROT;
    else sf = mjSAMEFRAME_NONE;
    m->site_sameframe[sid] = static_cast<mjtByte>(sf);
  }

  for (int cid = 0; cid < static_cast<int>(cameras.size()); ++cid) {
    const CCamera& cc = cameras[cid];
    m->cam_bodyid[cid] = cc.bodyid;
    m->cam_mode[cid] = cc.mode;
    m->cam_targetbodyid[cid] = resolve(cc.targetbody);
    lift::mjuu_copyvec(m->cam_pos + 3 * cid, cc.pos, 3);
    lift::mjuu_copyvec(m->cam_quat + 4 * cid, cc.quat, 4);
    m->cam_projection[cid] = cc.proj;
    m->cam_fovy[cid] = cc.fovy;
    m->cam_ipd[cid] = cc.ipd;
    m->cam_resolution[2 * cid] = cc.resolution[0];
    m->cam_resolution[2 * cid + 1] = cc.resolution[1];
    m->cam_output[cid] = cc.output;
    m->cam_sensorsize[2 * cid] = cc.sensorsize[0];
    m->cam_sensorsize[2 * cid + 1] = cc.sensorsize[1];
    for (int k = 0; k < 4; ++k) m->cam_intrinsic[4 * cid + k] = cc.intrinsic[k];
    for (int k = 0; k < m->nuser_cam && k < static_cast<int>(cc.user.size()); ++k)
      m->cam_user[m->nuser_cam * cid + k] = cc.user[k];
  }

  for (int lid = 0; lid < static_cast<int>(lights.size()); ++lid) {
    const CLight& cl = lights[lid];
    m->light_bodyid[lid] = cl.bodyid;
    m->light_mode[lid] = cl.mode;
    m->light_targetbodyid[lid] = resolve(cl.targetbody);
    m->light_type[lid] = cl.type;
    m->light_texid[lid] = cl.texid;
    m->light_castshadow[lid] = static_cast<mjtByte>(cl.castshadow);
    m->light_active[lid] = static_cast<mjtByte>(cl.active);
    lift::mjuu_copyvec(m->light_pos + 3 * lid, cl.pos, 3);
    lift::mjuu_copyvec(m->light_dir + 3 * lid, cl.dir, 3);
    m->light_bulbradius[lid] = cl.bulbradius;
    m->light_intensity[lid] = cl.intensity;
    m->light_range[lid] = cl.range;
    for (int k = 0; k < 3; ++k) m->light_attenuation[3 * lid + k] = cl.attenuation[k];
    m->light_cutoff[lid] = cl.cutoff;
    m->light_exponent[lid] = cl.exponent;
    for (int k = 0; k < 3; ++k) m->light_ambient[3 * lid + k] = cl.ambient[k];
    for (int k = 0; k < 3; ++k) m->light_diffuse[3 * lid + k] = cl.diffuse[k];
    for (int k = 0; k < 3; ++k) m->light_specular[3 * lid + k] = cl.specular[k];
  }
}

// --------------------------------------------------------------------------- //
// S10 Names: the 23-list name blob + hash + adr tables.                        //
// --------------------------------------------------------------------------- //

constexpr int kLoadMultiple = 2;  // mjLOAD_MULTIPLE (engine-internal)

// Lifted verbatim: mj_hashString (engine_name.c:228-236, djb2-xor).
std::uint64_t HashString(const char* s, std::uint64_t n) {
  std::uint64_t h = 5381;
  int c;
  while ((c = *s++)) {
    h = ((h << 5) + h) ^ static_cast<std::uint64_t>(c);
  }
  return h % n;
}

// Lifted from addtolist (user_model.cc:2546-2558): append name+NUL, record adr.
int AddToList(const std::string& in, int adr, int* adr_field, char* names) {
  *adr_field = adr;
  std::memcpy(names + adr, in.c_str(), in.size());
  adr += static_cast<int>(in.size());
  names[adr] = 0;
  adr++;
  return adr;
}

// Lifted from namelist (user_model.cc:2560-2583): hash fill + concat.
int NameList(const std::vector<std::string>& list, int adr, int* name_adr,
             char* names, int* map) {
  const int map_size = kLoadMultiple * static_cast<int>(list.size());
  for (unsigned i = 0; i < list.size(); ++i) {
    if (list[i].empty()) continue;
    std::uint64_t j = HashString(list[i].c_str(), map_size);
    for (; map[j] != -1; j = (j + 1) % map_size) {}
    map[j] = static_cast<int>(i);
  }
  for (unsigned i = 0; i < list.size(); ++i)
    adr = AddToList(list[i], adr, &name_adr[i], names);
  return adr;
}

// The ordered per-family name lists (23-list order, CopyNames). NC1 populates
// the rigid-body families; the rest stay empty (size 0) so their name_*adr
// tables and map segments are trivially correct.
struct NameLists {
  std::string modelname;
  std::vector<std::string> body, jnt, geom, site, cam, light, mesh, skin, hfield,
      tex, mat, pair, exclude, eq, tendon, actuator, sensor, key;

  int TotalNames() const {
    int n = static_cast<int>(modelname.size()) + 1;
    auto add = [&](const std::vector<std::string>& v) {
      for (const auto& s : v) n += static_cast<int>(s.size()) + 1;
    };
    add(body); add(jnt); add(geom); add(site); add(cam); add(light);
    add(mesh); add(skin); add(hfield); add(tex); add(mat);
    add(pair); add(exclude); add(eq); add(tendon); add(actuator); add(sensor);
    add(key);
    return n;
  }
};

void FillNames(mjModel* m, const NameLists& nl) {
  int adr = static_cast<int>(nl.modelname.size()) + 1;
  int* map_adr = m->names_map;
  mju_strncpy(m->names, nl.modelname.c_str(), m->nnames);
  std::memset(m->names_map, -1, sizeof(int) * m->nnames_map);

  auto pass = [&](const std::vector<std::string>& v, int* name_adr) {
    adr = NameList(v, adr, name_adr, m->names, map_adr);
    map_adr += kLoadMultiple * static_cast<int>(v.size());
  };
  // 23-list order (CopyNames); families empty in NC1b scope are skipped -- an
  // empty list adds nothing to the blob and advances the map cursor by 0, so
  // segment offsets match MuJoCo exactly.
  pass(nl.body, m->name_bodyadr);
  pass(nl.jnt, m->name_jntadr);
  pass(nl.geom, m->name_geomadr);
  pass(nl.site, m->name_siteadr);
  pass(nl.cam, m->name_camadr);
  pass(nl.light, m->name_lightadr);
  pass(nl.mesh, m->name_meshadr);
  pass(nl.skin, m->name_skinadr);
  pass(nl.hfield, m->name_hfieldadr);
  pass(nl.tex, m->name_texadr);
  pass(nl.mat, m->name_matadr);
  pass(nl.pair, m->name_pairadr);
  pass(nl.exclude, m->name_excludeadr);
  pass(nl.eq, m->name_eqadr);
  pass(nl.tendon, m->name_tendonadr);
  pass(nl.actuator, m->name_actuatoradr);
  pass(nl.sensor, m->name_sensoradr);
  pass(nl.key, m->name_keyadr);
}

// Effective name of a nameable element: authored name, else the auto-name the
// XML path would inject (mirrors bridge Collector so name tables match leg B).
template <class E>
std::string EffectiveName(const E& e, const bridge::CompileOptions& opts) {
  if (e.name) return *e.name;
  if (opts.auto_name) {
    return opts.auto_name_prefix +
           std::string(bridge::FamilyToken(element_type_of<E>::value)) + ":" +
           std::to_string(e.serial);
  }
  return "";
}

// Arena size heuristic, lifted verbatim from user_model.cc:5219-5252 (the
// memory/nstack unset branch; NC1 scope has no <size memory/nstack>).
void SetNarena(mjModel* m) {
  const int nconmax = m->nconmax == -1 ? 100 : m->nconmax;
  const int njmax = m->njmax == -1 ? 500 : m->njmax;
  m->narena = sizeof(mjtNum) * static_cast<size_t>(mjMAX(
      1000,
      5 * (njmax + m->neq + m->nv) * (njmax + m->neq + m->nv) +
      20 * (m->nq + m->nv + m->nu + m->na + m->nbody + m->njnt +
            m->ngeom + m->nsite + m->neq + m->ntendon + m->nwrap)));
  const std::size_t arena_bytes = (
      nconmax * sizeof(mjContact) +
      njmax * (8 * sizeof(int) + 14 * sizeof(mjtNum)) +
      m->nv * (3 * sizeof(int)) +
      njmax * m->nv * (2 * sizeof(int) + 2 * sizeof(mjtNum)) +
      njmax * njmax * (sizeof(int) + sizeof(mjtNum)));
  m->narena += arena_bytes;
  constexpr std::size_t kMegabyte = 1 << 20;
  std::size_t nstack_mb = m->narena / kMegabyte;
  std::size_t residual_mb = m->narena % kMegabyte ? 1 : 0;
  m->narena = kMegabyte * (nstack_mb + residual_mb);
}

// --------------------------------------------------------------------------- //
// Contact pairs + excludes (S8). Compiled, then stable-sorted by body          //
// signature with ids reassigned (user_model.cc:5143-5146). Pair parameters are //
// the mjs_defaultPair base overridden by class/element authored values; the    //
// mjCPair::Compile geom-derivation never fires on XML-authored pairs (their     //
// fields are always defined), so leg B and the native path agree here.          //
// --------------------------------------------------------------------------- //
struct CPair {
  const Pair* src = nullptr;
  int geom1 = -1, geom2 = -1;
  int condim = 3;
  double solref[2] = {0.02, 1};
  double solreffriction[2] = {0, 0};
  double solimp[5] = {0.9, 0.95, 0.001, 0.5, 2};
  double friction[5] = {1, 1, 0.005, 0.0001, 0.0001};
  double margin = 0, gap = 0;
  int signature = 0;
};

struct CExclude {
  const Exclude* src = nullptr;
  int signature = 0;
};

CPair PairCompile(const Model& model, const Pair& p,
                  const std::unordered_map<std::string, int>& geomid,
                  const std::vector<CGeom>& geoms) {
  std::unique_ptr<Pair> eff = ps::sdk::Effective(model, p);
  CPair cp;
  cp.src = &p;
  int g1 = -1, g2 = -1;
  if (eff->geom1) { auto it = geomid.find(eff->geom1->name);
                    if (it != geomid.end()) g1 = it->second; }
  if (eff->geom2) { auto it = geomid.find(eff->geom2->name);
                    if (it != geomid.end()) g2 = it->second; }
  // swap so body1 <= body2 (mjCPair::ResolveReferences).
  if (g1 >= 0 && g2 >= 0 && geoms[g1].bodyid > geoms[g2].bodyid)
    std::swap(g1, g2);
  cp.geom1 = g1; cp.geom2 = g2;
  if (eff->condim) cp.condim = *eff->condim;
  if (eff->friction)
    for (int k = 0; k < 5 && k < static_cast<int>(eff->friction->size()); ++k)
      cp.friction[k] = (*eff->friction)[k];
  if (eff->solref)
    for (int k = 0; k < 2 && k < static_cast<int>(eff->solref->size()); ++k)
      cp.solref[k] = (*eff->solref)[k];
  if (eff->solreffriction)
    for (int k = 0; k < 2 && k < static_cast<int>(eff->solreffriction->size()); ++k)
      cp.solreffriction[k] = (*eff->solreffriction)[k];
  if (eff->solimp)
    for (int k = 0; k < 5 && k < static_cast<int>(eff->solimp->size()); ++k)
      cp.solimp[k] = (*eff->solimp)[k];
  if (eff->margin) cp.margin = *eff->margin;
  if (eff->gap) cp.gap = *eff->gap;
  const int b1 = g1 >= 0 ? geoms[g1].bodyid : 0;
  const int b2 = g2 >= 0 ? geoms[g2].bodyid : 0;
  cp.signature = (b1 << 16) + b2;
  return cp;
}

CExclude ExcludeCompile(const Exclude& e,
                        const std::unordered_map<std::string, int>& bodyid) {
  CExclude ce;
  ce.src = &e;
  int b1 = 0, b2 = 0;
  if (e.body1) { auto it = bodyid.find(e.body1->name);
                 if (it != bodyid.end()) b1 = it->second; }
  if (e.body2) { auto it = bodyid.find(e.body2->name);
                 if (it != bodyid.end()) b2 = it->second; }
  if (b1 > b2) std::swap(b1, b2);  // mjCBodyPair::ResolveReferences
  ce.signature = (b1 << 16) + b2;
  return ce;
}

void FillPairs(mjModel* m, const std::vector<CPair>& pairs) {
  for (int i = 0; i < static_cast<int>(pairs.size()); ++i) {
    const CPair& cp = pairs[i];
    m->pair_dim[i] = cp.condim;
    m->pair_geom1[i] = cp.geom1;
    m->pair_geom2[i] = cp.geom2;
    m->pair_signature[i] = cp.signature;
    lift::mjuu_copyvec(m->pair_solref + mjNREF * i, cp.solref, mjNREF);
    lift::mjuu_copyvec(m->pair_solreffriction + mjNREF * i, cp.solreffriction, mjNREF);
    lift::mjuu_copyvec(m->pair_solimp + mjNIMP * i, cp.solimp, mjNIMP);
    m->pair_margin[i] = cp.margin;
    m->pair_gap[i] = cp.gap;
    lift::mjuu_copyvec(m->pair_friction + 5 * i, cp.friction, 5);
  }
}

void FillExcludes(mjModel* m, const std::vector<CExclude>& excludes) {
  for (int i = 0; i < static_cast<int>(excludes.size()); ++i)
    m->exclude_signature[i] = excludes[i].signature;
}

// --------------------------------------------------------------------------- //
// Equality constraints (S8). The typed -> data[mjNEQDATA] packing lives only    //
// in MuJoCo's XML reader (OneEquality, xml_native_reader.cc:2192-2311); on the  //
// native path it is the fill rule for eq_data. Object resolution mirrors        //
// mjCEquality::ResolveReferences (user_objects.cc:6267): connect/weld target    //
// body|site, joint/tendon equalities target joints|tendons, a missing body2 is  //
// the world (id 0). Defaults use a distinct EqualityDefault partial that        //
// ps::sdk::Effective does not merge, so the active/solref/solimp class chain is  //
// resolved here (top-level elements: governing class is the element's own class  //
// or root, no body childclass). Lifted: equality_pack (registry).               //
// --------------------------------------------------------------------------- //
struct CEquality {
  const void* src = nullptr;   // Connect/Weld/EqualityJoint/EqualityTendon*
  std::uint64_t serial = 0;
  ps::opt<std::string> name;
  int type = mjEQ_CONNECT;
  int objtype = 0;             // mjtObj (BODY/SITE for connect/weld)
  int obj1id = -1, obj2id = -1;
  bool active = true;
  double solref[2] = {0.02, 1};
  double solimp[5] = {0.9, 0.95, 0.001, 0.5, 2};
  double data[mjNEQDATA] = {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1};
};

// Resolve active/solref/solimp from the EqualityDefault class chain (element's
// own class then ancestors up to root; IDL defaults are already the CEquality
// initializers). Fill-unauthored order: authored value wins, else nearest class.
void MergeEqualityClassChain(const ps::sdk::detail::DefaultIndex& idx,
                             const std::string& cls,
                             const ps::opt<bool>& a_auth,
                             const ps::opt<ps::InlineVec<double, 2>>& sr_auth,
                             const ps::opt<ps::InlineVec<double, 5>>& si_auth,
                             CEquality& ce) {
  // Partial solref/solimp keep their defaults for the unspecified entries
  // (ReadAttr overwrites only the values present), so copy only the authored
  // length, never the full array.
  auto copy_ref = [](double* dst, const ps::InlineVec<double, 2>& v) {
    for (std::size_t k = 0; k < v.size() && k < 2; ++k) dst[k] = v[k];
  };
  auto copy_imp = [](double* dst, const ps::InlineVec<double, 5>& v) {
    for (std::size_t k = 0; k < v.size() && k < 5; ++k) dst[k] = v[k];
  };
  bool have_a = a_auth.has_value(), have_sr = sr_auth.has_value(),
       have_si = si_auth.has_value();
  if (have_a) ce.active = *a_auth;
  if (have_sr) copy_ref(ce.solref, *sr_auth);
  if (have_si) copy_imp(ce.solimp, *si_auth);
  for (const ps::mjcf::Default* d = idx.ByNameOrRoot(cls); d;
       d = idx.ParentOf(d)) {
    if (d->equality.empty() || !d->equality.front()) continue;
    const EqualityDefault& ed = *d->equality.front();
    if (!have_a && ed.active) { ce.active = *ed.active; have_a = true; }
    if (!have_sr && ed.solref) { copy_ref(ce.solref, *ed.solref); have_sr = true; }
    if (!have_si && ed.solimp) { copy_imp(ce.solimp, *ed.solimp); have_si = true; }
  }
}

// Pack a connect equality (OneEquality mjEQ_CONNECT branch).
void PackConnect(const Connect& c, CEquality& ce) {
  ce.type = mjEQ_CONNECT;
  const bool body_semantic = c.body1.has_value();
  ce.objtype = body_semantic ? mjOBJ_BODY : mjOBJ_SITE;
  if (c.anchor) for (int k = 0; k < 3; ++k) ce.data[k] = (*c.anchor)[k];
}

// Pack a weld equality (OneEquality mjEQ_WELD branch).
void PackWeld(const Weld& w, CEquality& ce) {
  ce.type = mjEQ_WELD;
  const bool body_semantic = w.body1.has_value();
  const bool has_anchor = w.anchor.has_value();
  ce.objtype = body_semantic ? mjOBJ_BODY : mjOBJ_SITE;
  if (has_anchor) for (int k = 0; k < 3; ++k) ce.data[k] = (*w.anchor)[k];
  if (w.relpose) for (int k = 0; k < 7; ++k) ce.data[3 + k] = (*w.relpose)[k];
  // body semantics with no anchor: zero the anchor slot (upstream mjuu_zerovec).
  if (body_semantic && !has_anchor) ce.data[0] = ce.data[1] = ce.data[2] = 0;
  if (w.torquescale) ce.data[10] = *w.torquescale;
}

// Pack joint/tendon polycoef into data[0..4] (default {0,1,0,0,0}).
void PackPolycoef(const ps::opt<std::vector<double>>& poly, CEquality& ce) {
  if (poly)
    for (int k = 0; k < 5 && k < static_cast<int>(poly->size()); ++k)
      ce.data[k] = (*poly)[k];
}

using NameIdMap = std::unordered_map<std::string, int>;

// Compile one equality element of any spelling.
CEquality EqualityCompile(const ps::sdk::detail::DefaultIndex& idx,
                          const EqualityAny& any, const NameIdMap& bodyid,
                          const NameIdMap& siteid, const NameIdMap& jointid,
                          const NameIdMap& tendonid) {
  CEquality ce;
  auto resolve = [](const NameIdMap& m, const std::string& nm) -> int {
    auto it = m.find(nm);
    return it == m.end() ? -1 : it->second;
  };
  switch (any.kind()) {
    case EqualityAny::Kind::Connect: {
      const Connect& c = *std::get<std::unique_ptr<Connect>>(any.node);
      ce.src = &c; ce.serial = c.serial; ce.name = c.name;
      PackConnect(c, ce);
      MergeEqualityClassChain(idx, c.dclass ? c.dclass->name : "", c.active,
                              c.solref, c.solimp, ce);
      const NameIdMap& m = (ce.objtype == mjOBJ_BODY) ? bodyid : siteid;
      if (c.body1) ce.obj1id = resolve(m, c.body1->name);
      else if (c.site1) ce.obj1id = resolve(m, c.site1->name);
      if (c.body2) ce.obj2id = resolve(m, c.body2->name);
      else if (c.site2) ce.obj2id = resolve(m, c.site2->name);
      break;
    }
    case EqualityAny::Kind::Weld: {
      const Weld& w = *std::get<std::unique_ptr<Weld>>(any.node);
      ce.src = &w; ce.serial = w.serial; ce.name = w.name;
      PackWeld(w, ce);
      MergeEqualityClassChain(idx, w.dclass ? w.dclass->name : "", w.active,
                              w.solref, w.solimp, ce);
      const NameIdMap& m = (ce.objtype == mjOBJ_BODY) ? bodyid : siteid;
      if (w.body1) ce.obj1id = resolve(m, w.body1->name);
      else if (w.site1) ce.obj1id = resolve(m, w.site1->name);
      if (w.body2) ce.obj2id = resolve(m, w.body2->name);
      else if (w.site2) ce.obj2id = resolve(m, w.site2->name);
      break;
    }
    case EqualityAny::Kind::EqualityJoint: {
      const EqualityJoint& j = *std::get<std::unique_ptr<EqualityJoint>>(any.node);
      ce.src = &j; ce.serial = j.serial; ce.name = j.name;
      ce.type = mjEQ_JOINT;  // objtype stays 0: OneEquality sets it only for
                             // connect/weld (joint/tendon leave it unset).
      PackPolycoef(j.polycoef, ce);
      MergeEqualityClassChain(idx, j.dclass ? j.dclass->name : "", j.active,
                              j.solref, j.solimp, ce);
      if (j.joint1) ce.obj1id = resolve(jointid, j.joint1->name);
      if (j.joint2) ce.obj2id = resolve(jointid, j.joint2->name);
      break;
    }
    case EqualityAny::Kind::EqualityTendon: {
      const EqualityTendon& t = *std::get<std::unique_ptr<EqualityTendon>>(any.node);
      ce.src = &t; ce.serial = t.serial; ce.name = t.name;
      ce.type = mjEQ_TENDON;  // objtype stays 0 (see EqualityJoint).
      PackPolycoef(t.polycoef, ce);
      MergeEqualityClassChain(idx, t.dclass ? t.dclass->name : "", t.active,
                              t.solref, t.solimp, ce);
      if (t.tendon1) ce.obj1id = resolve(tendonid, t.tendon1->name);
      if (t.tendon2) ce.obj2id = resolve(tendonid, t.tendon2->name);
      break;
    }
    default:
      break;  // flex equalities gated out
  }
  // missing body2 -> world (mjCEquality::ResolveReferences).
  if (ce.objtype == mjOBJ_BODY && ce.obj2id == -1) ce.obj2id = 0;
  return ce;
}

void FillEqualities(mjModel* m, const std::vector<CEquality>& eqs) {
  for (int i = 0; i < static_cast<int>(eqs.size()); ++i) {
    const CEquality& ce = eqs[i];
    m->eq_type[i] = ce.type;
    m->eq_obj1id[i] = ce.obj1id;
    m->eq_obj2id[i] = ce.obj2id;
    m->eq_objtype[i] = ce.objtype;
    m->eq_active0[i] = static_cast<mjtByte>(ce.active);
    lift::mjuu_copyvec(m->eq_solref + mjNREF * i, ce.solref, mjNREF);
    lift::mjuu_copyvec(m->eq_solimp + mjNIMP * i, ce.solimp, mjNIMP);
    lift::mjuu_copyvec(m->eq_data + mjNEQDATA * i, ce.data, mjNEQDATA);
  }
}

// --------------------------------------------------------------------------- //
// Tendons (S8), spatial + fixed. The wrap path is built from the ProtoSpec      //
// PathItemAny / FixedJoint lists (WrapSite/WrapGeom/WrapPulley/WrapJoint,        //
// user_objects.cc:6470-6540); wrap type/objid/prm and geom sphere->cylinder     //
// selection + sidesite follow mjCWrap::ResolveReferences (:6800). Limits use    //
// islimited (checklimited/Q-AUTO). Defaults use the distinct TendonDefault       //
// partial (not merged by ps::sdk::Effective), resolved here over the class       //
// chain. Length range / actuatorfrcrange are copied through; nJten is computed   //
// as in ComputeSparseSizes (:1088). Material/armature route to fallback (gate).  //
// --------------------------------------------------------------------------- //
struct CWrap {
  int type = 0;      // mjtWrap
  int objid = -1;
  double prm = 0;
  int sideid = -1;
};

struct CTendon {
  const void* src = nullptr;
  std::uint64_t serial = 0;
  ps::opt<std::string> name;
  bool spatial = true;
  std::vector<CWrap> path;
  int group = 0;
  bool limited = false, actfrclimited = false;
  double width = 0.003;
  double solref_lim[2] = {0.02, 1};
  double solimp_lim[5] = {0.9, 0.95, 0.001, 0.5, 2};
  double solref_fri[2] = {0.02, 1};
  double solimp_fri[5] = {0.9, 0.95, 0.001, 0.5, 2};
  double range[2] = {0, 0}, actfrcrange[2] = {0, 0};
  double margin = 0;
  double stiffness[3] = {0, 0, 0};   // [scalar, poly0, poly1]
  double damping[3] = {0, 0, 0};
  double armature = 0, frictionloss = 0;
  double springlength[2] = {-1, -1};
  float rgba[4] = {0.5f, 0.5f, 0.5f, 1.0f};
  std::vector<double> user;
  int matid = -1;
  std::string material_name;
};

// Common authored-field bag so Spatial and Fixed share one resolver. `has_*`
// gate is the opt presence; class-chain / IDL fill unauthored fields.
struct TendonAuthored {
  ps::opt<int32_t> group;
  ps::opt<TriState> limited, actfrclimited;
  ps::opt<std::array<double, 2>> range, actfrcrange;
  ps::opt<ps::InlineVec<double, 2>> solreflimit, solreffriction;
  ps::opt<ps::InlineVec<double, 5>> solimplimit, solimpfriction;
  ps::opt<double> frictionloss;
  ps::opt<ps::InlineVec<double, 2>> springlength;
  ps::opt<double> width, margin, armature;
  ps::opt<ps::InlineVec<double, 3>> stiffness, damping;
  ps::opt<std::array<float, 4>> rgba;
  ps::opt<std::vector<double>> user;
  ps::opt<ps::Ref<Material>> material;
};

template <int N>
void CopyVecOpt(double* dst, const ps::InlineVec<double, N>& v) {
  for (std::size_t k = 0; k < v.size() && k < static_cast<std::size_t>(N); ++k)
    dst[k] = v[k];
}

// Resolve authored -> class chain -> IDL (CTendon initializers) for the scalar
// and small-array fields shared by spatial and fixed tendons.
void ResolveTendonFields(const ps::sdk::detail::DefaultIndex& idx,
                         const std::string& cls, const TendonAuthored& a,
                         CTendon& ct) {
  // gather class-chain TendonDefault snapshots (nearest first).
  std::vector<const TendonDefault*> chain;
  for (const ps::mjcf::Default* d = idx.ByNameOrRoot(cls); d;
       d = idx.ParentOf(d))
    if (!d->tendon.empty() && d->tendon.front())
      chain.push_back(d->tendon.front().get());

  // group
  if (a.group) ct.group = *a.group;
  else for (const auto* td : chain) if (td->group) { ct.group = *td->group; break; }
  // width
  if (a.width) ct.width = *a.width;
  else for (const auto* td : chain) if (td->width) { ct.width = *td->width; break; }
  // margin
  if (a.margin) ct.margin = *a.margin;
  else for (const auto* td : chain) if (td->margin) { ct.margin = *td->margin; break; }
  // armature (no TendonDefault field; gated non-zero) -> authored only
  if (a.armature) ct.armature = *a.armature;
  // frictionloss
  if (a.frictionloss) ct.frictionloss = *a.frictionloss;
  else for (const auto* td : chain)
    if (td->frictionloss) { ct.frictionloss = *td->frictionloss; break; }
  // range
  if (a.range) { ct.range[0] = (*a.range)[0]; ct.range[1] = (*a.range)[1]; }
  else for (const auto* td : chain)
    if (td->range) { ct.range[0] = (*td->range)[0]; ct.range[1] = (*td->range)[1]; break; }
  // actfrcrange (no TendonDefault field) -> authored only
  if (a.actfrcrange) { ct.actfrcrange[0] = (*a.actfrcrange)[0];
                       ct.actfrcrange[1] = (*a.actfrcrange)[1]; }
  // springlength
  if (a.springlength) CopyVecOpt<2>(ct.springlength, *a.springlength);
  else for (const auto* td : chain)
    if (td->springlength) { ct.springlength[0] = (*td->springlength)[0];
                            ct.springlength[1] = (*td->springlength)[1]; break; }
  // solref/solimp limit + friction
  auto resolve_ref = [&](double* dst, const ps::opt<ps::InlineVec<double, 2>>& au,
                         auto pick) {
    if (au) { CopyVecOpt<2>(dst, *au); return; }
    for (const auto* td : chain) { const auto& f = pick(td);
      if (f) { CopyVecOpt<2>(dst, *f); return; } }
  };
  auto resolve_imp = [&](double* dst, const ps::opt<ps::InlineVec<double, 5>>& au,
                         auto pick) {
    if (au) { CopyVecOpt<5>(dst, *au); return; }
    for (const auto* td : chain) { const auto& f = pick(td);
      if (f) { CopyVecOpt<5>(dst, *f); return; } }
  };
  resolve_ref(ct.solref_lim, a.solreflimit, [](const TendonDefault* t){ return t->solreflimit; });
  resolve_imp(ct.solimp_lim, a.solimplimit, [](const TendonDefault* t){ return t->solimplimit; });
  resolve_ref(ct.solref_fri, a.solreffriction, [](const TendonDefault* t){ return t->solreffriction; });
  resolve_imp(ct.solimp_fri, a.solimpfriction, [](const TendonDefault* t){ return t->solimpfriction; });
  // stiffness / damping (InlineVec<3> authored; array<3> in default)
  if (a.stiffness) CopyVecOpt<3>(ct.stiffness, *a.stiffness);
  else for (const auto* td : chain) if (td->stiffness) {
    for (int k = 0; k < 3; ++k) ct.stiffness[k] = (*td->stiffness)[k]; break; }
  if (a.damping) CopyVecOpt<3>(ct.damping, *a.damping);
  else for (const auto* td : chain) if (td->damping) {
    for (int k = 0; k < 3; ++k) ct.damping[k] = (*td->damping)[k]; break; }
  // rgba
  if (a.rgba) for (int k = 0; k < 4; ++k) ct.rgba[k] = (*a.rgba)[k];
  else for (const auto* td : chain) if (td->rgba) {
    for (int k = 0; k < 4; ++k) ct.rgba[k] = (*td->rgba)[k]; break; }
  // user
  if (a.user) ct.user = *a.user;
  else for (const auto* td : chain) if (td->user) { ct.user = *td->user; break; }
  // material (authored -> class chain); resolved to matid after materials compile
  if (a.material) ct.material_name = a.material->name;
  else for (const auto* td : chain)
    if (td->material) { ct.material_name = td->material->name; break; }

  // limits (islimited over resolved limited state + range).
  auto tri = [](const ps::opt<TriState>& t) {
    return t ? static_cast<int>(*t) : mjLIMITED_AUTO;
  };
  int lim = a.limited ? static_cast<int>(*a.limited) : mjLIMITED_AUTO;
  if (!a.limited)
    for (const auto* td : chain) if (td->limited) { lim = static_cast<int>(*td->limited); break; }
  ct.limited = IsLimited(lim, ct.range);
  ct.actfrclimited = IsLimited(tri(a.actfrclimited), ct.actfrcrange);
}

void FillTendonAuthoredSpatial(const Spatial& s, TendonAuthored& a) {
  a.group = s.group; a.limited = s.limited; a.actfrclimited = s.actuatorfrclimited;
  a.range = s.range; a.actfrcrange = s.actuatorfrcrange;
  a.solreflimit = s.solreflimit; a.solimplimit = s.solimplimit;
  a.solreffriction = s.solreffriction; a.solimpfriction = s.solimpfriction;
  a.frictionloss = s.frictionloss; a.springlength = s.springlength;
  a.width = s.width; a.margin = s.margin; a.armature = s.armature;
  a.stiffness = s.stiffness; a.damping = s.damping; a.rgba = s.rgba; a.user = s.user;
  a.material = s.material;
}

void FillTendonAuthoredFixed(const Fixed& f, TendonAuthored& a) {
  a.group = f.group; a.limited = f.limited; a.actfrclimited = f.actuatorfrclimited;
  a.range = f.range; a.actfrcrange = f.actuatorfrcrange;
  a.solreflimit = f.solreflimit; a.solimplimit = f.solimplimit;
  a.solreffriction = f.solreffriction; a.solimpfriction = f.solimpfriction;
  a.frictionloss = f.frictionloss; a.springlength = f.springlength;
  a.margin = f.margin; a.armature = f.armature;
  a.stiffness = f.stiffness; a.damping = f.damping; a.user = f.user;
}

CTendon TendonCompile(const ps::sdk::detail::DefaultIndex& idx,
                      const TendonAny& any, const NameIdMap& siteid,
                      const NameIdMap& geomid, const NameIdMap& jointid,
                      const std::vector<CGeom>& geoms) {
  CTendon ct;
  auto resolve = [](const NameIdMap& m, const std::string& nm) -> int {
    auto it = m.find(nm);
    return it == m.end() ? -1 : it->second;
  };
  TendonAuthored a;
  if (any.kind() == TendonAny::Kind::Spatial) {
    const Spatial& s = *std::get<std::unique_ptr<Spatial>>(any.node);
    ct.src = &s; ct.serial = s.serial; ct.name = s.name; ct.spatial = true;
    FillTendonAuthoredSpatial(s, a);
    ResolveTendonFields(idx, s.dclass ? s.dclass->name : "", a, ct);
    for (const PathItemAny& item : s.path) {
      CWrap w;
      switch (item.kind()) {
        case PathItemAny::Kind::SpatialSite: {
          const SpatialSite& ss = *std::get<std::unique_ptr<SpatialSite>>(item.node);
          w.type = mjWRAP_SITE;
          if (ss.site) w.objid = resolve(siteid, ss.site->name);
          break;
        }
        case PathItemAny::Kind::SpatialGeom: {
          const SpatialGeom& sg = *std::get<std::unique_ptr<SpatialGeom>>(item.node);
          w.type = mjWRAP_SPHERE;
          if (sg.geom) w.objid = resolve(geomid, sg.geom->name);
          if (w.objid >= 0 && geoms[w.objid].type == mjGEOM_CYLINDER)
            w.type = mjWRAP_CYLINDER;
          if (sg.sidesite) w.sideid = resolve(siteid, sg.sidesite->name);
          break;
        }
        case PathItemAny::Kind::Pulley: {
          const Pulley& p = *std::get<std::unique_ptr<Pulley>>(item.node);
          w.type = mjWRAP_PULLEY;
          w.prm = p.divisor ? *p.divisor : 0;
          break;
        }
        default:
          break;
      }
      ct.path.push_back(w);
    }
  } else {
    const Fixed& f = *std::get<std::unique_ptr<Fixed>>(any.node);
    ct.src = &f; ct.serial = f.serial; ct.name = f.name; ct.spatial = false;
    FillTendonAuthoredFixed(f, a);
    ResolveTendonFields(idx, f.dclass ? f.dclass->name : "", a, ct);
    for (const auto& fj : f.fixedJoints) {
      if (!fj) continue;
      CWrap w;
      w.type = mjWRAP_JOINT;
      if (fj->joint) w.objid = resolve(jointid, fj->joint->name);
      w.prm = fj->coef ? *fj->coef : 0;
      ct.path.push_back(w);
    }
  }
  return ct;
}

// nJten (ComputeSparseSizes user_model.cc:1088): fixed tendons contribute one
// per joint wrap; spatial tendons contribute the number of dofs on the ancestor
// chains of every wrapped body.
int ComputeNJten(const std::vector<CTendon>& tendons,
                 const std::vector<CBody>& cbs, const std::vector<CGeom>& geoms,
                 const std::vector<CSite>& sites, int nv) {
  int nJten = 0;
  if (nv <= 0) return 0;
  std::vector<char> bitmap(nv, 0);
  for (const CTendon& t : tendons) {
    if (!t.path.empty() && t.path[0].type == mjWRAP_JOINT) {
      nJten += static_cast<int>(t.path.size());
      continue;
    }
    std::fill(bitmap.begin(), bitmap.end(), 0);
    for (const CWrap& w : t.path) {
      int bodyid = -1;
      if (w.type == mjWRAP_SITE && w.objid >= 0) bodyid = sites[w.objid].bodyid;
      else if ((w.type == mjWRAP_SPHERE || w.type == mjWRAP_CYLINDER) && w.objid >= 0)
        bodyid = geoms[w.objid].bodyid;
      if (bodyid <= 0) continue;
      for (int b = bodyid; b > 0; b = cbs[b].parentid)
        for (int d = cbs[b].dofadr; d < cbs[b].dofadr + cbs[b].dofnum && cbs[b].dofadr >= 0; ++d)
          bitmap[d] = 1;
    }
    for (int j = 0; j < nv; ++j) nJten += bitmap[j];
  }
  return nJten;
}

void FillTendons(mjModel* m, const std::vector<CTendon>& tendons) {
  int adr = 0;
  for (int i = 0; i < static_cast<int>(tendons.size()); ++i) {
    const CTendon& t = tendons[i];
    m->tendon_adr[i] = adr;
    m->tendon_num[i] = static_cast<int>(t.path.size());
    m->tendon_matid[i] = t.matid;
    m->tendon_group[i] = t.group;
    m->tendon_limited[i] = static_cast<mjtByte>(t.limited);
    m->tendon_actfrclimited[i] = static_cast<mjtByte>(t.actfrclimited);
    m->tendon_width[i] = t.width;
    lift::mjuu_copyvec(m->tendon_solref_lim + mjNREF * i, t.solref_lim, mjNREF);
    lift::mjuu_copyvec(m->tendon_solimp_lim + mjNIMP * i, t.solimp_lim, mjNIMP);
    lift::mjuu_copyvec(m->tendon_solref_fri + mjNREF * i, t.solref_fri, mjNREF);
    lift::mjuu_copyvec(m->tendon_solimp_fri + mjNIMP * i, t.solimp_fri, mjNIMP);
    m->tendon_range[2 * i] = t.range[0];
    m->tendon_range[2 * i + 1] = t.range[1];
    m->tendon_actfrcrange[2 * i] = t.actfrcrange[0];
    m->tendon_actfrcrange[2 * i + 1] = t.actfrcrange[1];
    m->tendon_margin[i] = t.margin;
    m->tendon_stiffness[i] = t.stiffness[0];
    lift::mjuu_copyvec(m->tendon_stiffnesspoly + mjNPOLY * i, t.stiffness + 1, mjNPOLY);
    m->tendon_damping[i] = t.damping[0];
    lift::mjuu_copyvec(m->tendon_dampingpoly + mjNPOLY * i, t.damping + 1, mjNPOLY);
    m->tendon_armature[i] = t.armature;
    m->tendon_frictionloss[i] = t.frictionloss;
    m->tendon_lengthspring[2 * i] = t.springlength[0];
    m->tendon_lengthspring[2 * i + 1] = t.springlength[1];
    for (int k = 0; k < m->nuser_tendon && k < static_cast<int>(t.user.size()); ++k)
      m->tendon_user[m->nuser_tendon * i + k] = t.user[k];
    for (int k = 0; k < 4; ++k) m->tendon_rgba[4 * i + k] = t.rgba[k];

    for (int j = 0; j < static_cast<int>(t.path.size()); ++j) {
      const CWrap& w = t.path[j];
      m->wrap_type[adr + j] = w.type;
      m->wrap_objid[adr + j] = w.objid;
      m->wrap_prm[adr + j] = (w.type == mjWRAP_SPHERE || w.type == mjWRAP_CYLINDER)
                                 ? static_cast<double>(w.sideid)
                                 : w.prm;
    }
    adr += static_cast<int>(t.path.size());
  }
}

// --------------------------------------------------------------------------- //
// Actuators (S8). ProtoSpec stores the TYPED shortcut variants (motor/position  //
// /.../muscle); the shortcut->general lowering math lives only in the mjs_setTo* //
// helpers (user_api.cc:1133-1294) and is LIFTED verbatim below, operating on a   //
// plain ActParams that maps 1:1 to the mjsActuator gain/bias/dyn fields. The     //
// per-shortcut defaults are mjs_defaultActuator's (gainprm[0]=1, dynprm[0]=1),   //
// which the reader reads through before overriding, so an unauthored shortcut    //
// inherits those. Transmission resolution + actdim + inheritrange follow         //
// mjCActuator::Compile/ResolveReferences (user_objects.cc:7099,6910). Lifted:    //
// actuator_lower (registry). Scope: motor/position/velocity/intvelocity/damper/  //
// cylinder/adhesion/muscle with joint/jointinparent/tendon/body transmission;    //
// dcmotor, plugin, site/refsite/slidercrank transmission, and delay are gated.   //
// --------------------------------------------------------------------------- //
struct ActParams {
  int dyntype = mjDYN_NONE, gaintype = mjGAIN_FIXED, biastype = mjBIAS_NONE;
  double dynprm[mjNDYN] = {0};
  double gainprm[mjNGAIN] = {0};
  double biasprm[mjNBIAS] = {0};
  double ctrlrange[2] = {0, 0}, actrange[2] = {0, 0}, forcerange[2] = {0, 0};
  int ctrllimited = mjLIMITED_AUTO, forcelimited = mjLIMITED_AUTO,
      actlimited = mjLIMITED_AUTO;
  double inheritrange = 0;
  int actdim = -1, actearly = 0;

  // mjs_defaultActuator base (user_init.c): the values an unauthored shortcut
  // reads through before lowering.
  ActParams() { gainprm[0] = 1; dynprm[0] = 1; }
};

namespace actlower {  // lifted from user_api.cc mjs_setTo* (verbatim math)

void SetToMotor(ActParams& a) {
  a.gainprm[0] = 1;
  a.dyntype = mjDYN_NONE;
  a.gaintype = mjGAIN_FIXED;
  a.biastype = mjBIAS_NONE;
}

void SetToPosition(ActParams& a, double kp, const double* kv,
                   const double* dampratio, const double* timeconst,
                   double inheritrange) {
  a.gainprm[0] = kp;
  a.biasprm[1] = -kp;
  if (kv) a.biasprm[2] = -(*kv);
  if (dampratio) a.biasprm[2] = *dampratio;
  if (timeconst) {
    a.dynprm[0] = *timeconst;
    a.dyntype = *timeconst == 0 ? mjDYN_NONE : mjDYN_FILTEREXACT;
  }
  a.inheritrange = inheritrange;
  a.gaintype = mjGAIN_FIXED;
  a.biastype = mjBIAS_AFFINE;
}

void SetToIntVelocity(ActParams& a, double kp, const double* kv,
                      const double* dampratio, const double* timeconst,
                      double inheritrange) {
  SetToPosition(a, kp, kv, dampratio, timeconst, inheritrange);
  a.dyntype = mjDYN_INTEGRATOR;
  a.actlimited = 1;
}

void SetToVelocity(ActParams& a, double kv) {
  for (int i = 0; i < mjNBIAS; ++i) a.biasprm[i] = 0;
  a.gainprm[0] = kv;
  a.biasprm[2] = -kv;
  a.dyntype = mjDYN_NONE;
  a.gaintype = mjGAIN_FIXED;
  a.biastype = mjBIAS_AFFINE;
}

void SetToDamper(ActParams& a, double kv) {
  for (int i = 0; i < mjNGAIN; ++i) a.gainprm[i] = 0;
  a.gainprm[2] = -kv;
  a.ctrllimited = 1;
  a.dyntype = mjDYN_NONE;
  a.gaintype = mjGAIN_AFFINE;
  a.biastype = mjBIAS_NONE;
}

void SetToCylinder(ActParams& a, double timeconst, double bias, double area,
                   double diameter) {
  a.dynprm[0] = timeconst;
  a.biasprm[0] = bias;
  a.gainprm[0] = area;
  if (diameter >= 0) a.gainprm[0] = mjPI / 4 * diameter * diameter;
  a.dyntype = mjDYN_FILTER;
  a.gaintype = mjGAIN_FIXED;
  a.biastype = mjBIAS_AFFINE;
}

void SetToMuscle(ActParams& a, const double timeconst[2], double tausmooth,
                 const double range[2], double force, double scale, double lmin,
                 double lmax, double vmax, double fpmax, double fvmax) {
  if (a.dynprm[0] == 1) a.dynprm[0] = 0.01;
  if (a.dynprm[1] == 0) a.dynprm[1] = 0.04;
  if (a.gainprm[0] == 1) a.gainprm[0] = 0.75;
  if (a.gainprm[1] == 0) a.gainprm[1] = 1.05;
  if (a.gainprm[2] == 0) a.gainprm[2] = -1;
  if (a.gainprm[3] == 0) a.gainprm[3] = 200;
  if (a.gainprm[4] == 0) a.gainprm[4] = 0.5;
  if (a.gainprm[5] == 0) a.gainprm[5] = 1.6;
  if (a.gainprm[6] == 0) a.gainprm[6] = 1.5;
  if (a.gainprm[7] == 0) a.gainprm[7] = 1.3;
  if (a.gainprm[8] == 0) a.gainprm[8] = 1.2;
  a.dynprm[2] = tausmooth;
  if (timeconst[0] >= 0) a.dynprm[0] = timeconst[0];
  if (timeconst[1] >= 0) a.dynprm[1] = timeconst[1];
  if (range[0] >= 0) a.gainprm[0] = range[0];
  if (range[1] >= 0) a.gainprm[1] = range[1];
  if (force >= 0) a.gainprm[2] = force;
  if (scale >= 0) a.gainprm[3] = scale;
  if (lmin >= 0) a.gainprm[4] = lmin;
  if (lmax >= 0) a.gainprm[5] = lmax;
  if (vmax >= 0) a.gainprm[6] = vmax;
  if (fpmax >= 0) a.gainprm[7] = fpmax;
  if (fvmax >= 0) a.gainprm[8] = fvmax;
  for (int n = 0; n < 9; ++n) a.biasprm[n] = a.gainprm[n];
  a.dyntype = mjDYN_MUSCLE;
  a.gaintype = mjGAIN_MUSCLE;
  a.biastype = mjBIAS_MUSCLE;
}

void SetToAdhesion(ActParams& a, double gain) {
  a.gainprm[0] = gain;
  a.ctrllimited = 1;
  a.gaintype = mjGAIN_FIXED;
  a.biastype = mjBIAS_NONE;
}

}  // namespace actlower

struct CActuator {
  const void* src = nullptr;
  std::uint64_t serial = 0;
  ps::opt<std::string> name;
  int trntype = mjTRN_UNDEFINED;
  int trnid[2] = {-1, -1};
  int gaintype = mjGAIN_FIXED, biastype = mjBIAS_NONE, dyntype = mjDYN_NONE;
  double dynprm[mjNDYN] = {0}, gainprm[mjNGAIN] = {0}, biasprm[mjNBIAS] = {0};
  double gear[6] = {1, 0, 0, 0, 0, 0};
  double ctrlrange[2] = {0, 0}, forcerange[2] = {0, 0}, actrange[2] = {0, 0},
         lengthrange[2] = {0, 0};
  double damping[3] = {0, 0, 0}, armature = 0, cranklength = 0;
  int group = 0, actdim = 0, actadr = -1;
  bool ctrllimited = false, forcelimited = false, actlimited = false;
  bool actearly = false;
  std::vector<double> user;
};

using OptStr = ps::opt<std::string>;

// Resolve the effective actuator element (class chain + IDL) via ps::sdk, lower
// its typed shortcuts through the lifted math, resolve the transmission target,
// and finish actdim / limits / inheritrange like mjCActuator::Compile.
template <class T>
void FillActuatorCommon(const T& eff, CActuator& ca, ActParams& ap) {
  if (eff.group) ca.group = *eff.group;
  if constexpr (requires { eff.gear; })
    if (eff.gear) for (std::size_t k = 0; k < eff.gear->size() && k < 6; ++k)
      ca.gear[k] = (*eff.gear)[k];
  if constexpr (requires { eff.ctrlrange; })
    if (eff.ctrlrange) { ca.ctrlrange[0] = (*eff.ctrlrange)[0];
                         ca.ctrlrange[1] = (*eff.ctrlrange)[1];
                         ap.ctrlrange[0] = ca.ctrlrange[0];
                         ap.ctrlrange[1] = ca.ctrlrange[1]; }
  if constexpr (requires { eff.forcerange; })
    if (eff.forcerange) { ca.forcerange[0] = (*eff.forcerange)[0];
                          ca.forcerange[1] = (*eff.forcerange)[1];
                          ap.forcerange[0] = ca.forcerange[0];
                          ap.forcerange[1] = ca.forcerange[1]; }
  if constexpr (requires { eff.lengthrange; })
    if (eff.lengthrange) { ca.lengthrange[0] = (*eff.lengthrange)[0];
                           ca.lengthrange[1] = (*eff.lengthrange)[1]; }
  if constexpr (requires { eff.damping; }) {
    if (eff.damping) for (std::size_t k = 0; k < eff.damping->size() && k < 3; ++k)
      ca.damping[k] = (*eff.damping)[k];
  }
  if constexpr (requires { eff.armature; }) if (eff.armature) ca.armature = *eff.armature;
  if constexpr (requires { eff.cranklength; }) if (eff.cranklength) ca.cranklength = *eff.cranklength;
  if (eff.user) ca.user = *eff.user;
  if constexpr (requires { eff.ctrllimited; })
    if (eff.ctrllimited) ap.ctrllimited = static_cast<int>(*eff.ctrllimited);
  if constexpr (requires { eff.forcelimited; })
    if (eff.forcelimited) ap.forcelimited = static_cast<int>(*eff.forcelimited);
}

// Transmission target from the first present ref (reader precedence:
// joint > jointinparent > tendon > cranksite(slidercrank) > site > body).
template <class T>
void ResolveTransmission(const T& eff, CActuator& ca, const NameIdMap& jointid,
                         const NameIdMap& tendonid, const NameIdMap& bodyid) {
  auto id = [](const NameIdMap& m, const std::string& n) {
    auto it = m.find(n); return it == m.end() ? -1 : it->second;
  };
  if (eff.joint) { ca.trntype = mjTRN_JOINT; ca.trnid[0] = id(jointid, eff.joint->name); }
  else if (eff.jointinparent) { ca.trntype = mjTRN_JOINTINPARENT;
                                ca.trnid[0] = id(jointid, eff.jointinparent->name); }
  else if (eff.tendon) { ca.trntype = mjTRN_TENDON; ca.trnid[0] = id(tendonid, eff.tendon->name); }
  else if constexpr (requires { eff.body; }) {
    if (eff.body) { ca.trntype = mjTRN_BODY; ca.trnid[0] = id(bodyid, eff.body->name); }
  }
}

// Range accessors for inheritrange: a target joint or tendon's compiled range.
struct RangeLookup {
  const std::vector<CJoint>* joints;
  const std::vector<CTendon>* tendons;
  const NameIdMap* jointid;
  const NameIdMap* tendonid;
};

CActuator ActuatorCompile(const Model& model, const ActuatorAny& any,
                          const NameIdMap& jointid, const NameIdMap& tendonid,
                          const NameIdMap& bodyid, const RangeLookup& rl) {
  CActuator ca;
  ActParams ap;
  auto ptr = [](const double* v) { return v; };
  (void)ptr;

  switch (any.kind()) {
    case ActuatorAny::Kind::ActuatorGeneral: {
      const ActuatorGeneral& g = *std::get<std::unique_ptr<ActuatorGeneral>>(any.node);
      std::unique_ptr<ActuatorGeneral> eff = ps::sdk::Effective(model, g);
      ca.src = &g; ca.serial = g.serial; ca.name = g.name;
      FillActuatorCommon(*eff, ca, ap);
      if (eff->dyntype) ap.dyntype = static_cast<int>(*eff->dyntype);
      if (eff->gaintype) ap.gaintype = static_cast<int>(*eff->gaintype);
      if (eff->biastype) ap.biastype = static_cast<int>(*eff->biastype);
      if (eff->actearly) ap.actearly = *eff->actearly ? 1 : 0;
      if (eff->dynprm) for (std::size_t k = 0; k < eff->dynprm->size() && k < mjNDYN; ++k)
        ap.dynprm[k] = (*eff->dynprm)[k];
      if (eff->gainprm) for (std::size_t k = 0; k < eff->gainprm->size() && k < mjNGAIN; ++k)
        ap.gainprm[k] = (*eff->gainprm)[k];
      if (eff->biasprm) for (std::size_t k = 0; k < eff->biasprm->size() && k < mjNBIAS; ++k)
        ap.biasprm[k] = (*eff->biasprm)[k];
      if (eff->actrange) { ca.actrange[0] = (*eff->actrange)[0]; ca.actrange[1] = (*eff->actrange)[1];
                           ap.actrange[0] = ca.actrange[0]; ap.actrange[1] = ca.actrange[1]; }
      if (eff->actlimited) ap.actlimited = static_cast<int>(*eff->actlimited);
      if (eff->actdim) ap.actdim = *eff->actdim;
      ResolveTransmission(*eff, ca, jointid, tendonid, bodyid);
      break;
    }
    case ActuatorAny::Kind::Motor: {
      const Motor& e = *std::get<std::unique_ptr<Motor>>(any.node);
      std::unique_ptr<Motor> eff = ps::sdk::Effective(model, e);
      ca.src = &e; ca.serial = e.serial; ca.name = e.name;
      FillActuatorCommon(*eff, ca, ap);
      actlower::SetToMotor(ap);
      ResolveTransmission(*eff, ca, jointid, tendonid, bodyid);
      break;
    }
    case ActuatorAny::Kind::Position:
    case ActuatorAny::Kind::IntVelocity: {
      const bool intvel = any.kind() == ActuatorAny::Kind::IntVelocity;
      // Position and IntVelocity share the same shortcut set.
      auto lower = [&](auto& eff) {
        ca.name = eff.name;
        FillActuatorCommon(eff, ca, ap);
        double kp = eff.kp ? *eff.kp : ap.gainprm[0];
        double kvv = 0, drr = 0, tcc = 0;
        const double* kv = nullptr; const double* dr = nullptr; const double* tc = nullptr;
        if (eff.kv) { kvv = *eff.kv; kv = &kvv; }
        if (eff.dampratio) { drr = *eff.dampratio; dr = &drr; }
        if constexpr (requires { eff.timeconst; })
          if (eff.timeconst) { tcc = *eff.timeconst; tc = &tcc; }
        double inherit = eff.inheritrange ? *eff.inheritrange : ap.inheritrange;
        if constexpr (requires { eff.actrange; }) {
          if (eff.actrange) { ca.actrange[0] = (*eff.actrange)[0]; ca.actrange[1] = (*eff.actrange)[1];
                              ap.actrange[0] = ca.actrange[0]; ap.actrange[1] = ca.actrange[1]; }
        }
        if (intvel) {
          actlower::SetToIntVelocity(ap, kp, kv, dr, tc, inherit);
        } else {
          actlower::SetToPosition(ap, kp, kv, dr, tc, inherit);
        }
        ResolveTransmission(eff, ca, jointid, tendonid, bodyid);
      };
      if (intvel) {
        const IntVelocity& e = *std::get<std::unique_ptr<IntVelocity>>(any.node);
        std::unique_ptr<IntVelocity> eff = ps::sdk::Effective(model, e);
        ca.src = &e; ca.serial = e.serial; lower(*eff);
      } else {
        const Position& e = *std::get<std::unique_ptr<Position>>(any.node);
        std::unique_ptr<Position> eff = ps::sdk::Effective(model, e);
        ca.src = &e; ca.serial = e.serial; lower(*eff);
      }
      break;
    }
    case ActuatorAny::Kind::Velocity: {
      const Velocity& e = *std::get<std::unique_ptr<Velocity>>(any.node);
      std::unique_ptr<Velocity> eff = ps::sdk::Effective(model, e);
      ca.src = &e; ca.serial = e.serial; ca.name = e.name;
      FillActuatorCommon(*eff, ca, ap);
      actlower::SetToVelocity(ap, eff->kv ? *eff->kv : ap.gainprm[0]);
      ResolveTransmission(*eff, ca, jointid, tendonid, bodyid);
      break;
    }
    case ActuatorAny::Kind::Damper: {
      const Damper& e = *std::get<std::unique_ptr<Damper>>(any.node);
      std::unique_ptr<Damper> eff = ps::sdk::Effective(model, e);
      ca.src = &e; ca.serial = e.serial; ca.name = e.name;
      FillActuatorCommon(*eff, ca, ap);
      actlower::SetToDamper(ap, eff->kv ? *eff->kv : 0);
      ResolveTransmission(*eff, ca, jointid, tendonid, bodyid);
      break;
    }
    case ActuatorAny::Kind::Cylinder: {
      const Cylinder& e = *std::get<std::unique_ptr<Cylinder>>(any.node);
      std::unique_ptr<Cylinder> eff = ps::sdk::Effective(model, e);
      ca.src = &e; ca.serial = e.serial; ca.name = e.name;
      FillActuatorCommon(*eff, ca, ap);
      double timeconst = eff->timeconst ? *eff->timeconst : ap.dynprm[0];
      double bias = eff->bias ? (*eff->bias)[0] : ap.biasprm[0];
      double area = eff->area ? *eff->area : ap.gainprm[0];
      double diameter = eff->diameter ? *eff->diameter : -1;
      actlower::SetToCylinder(ap, timeconst, bias, area, diameter);
      ResolveTransmission(*eff, ca, jointid, tendonid, bodyid);
      break;
    }
    case ActuatorAny::Kind::Muscle: {
      const Muscle& e = *std::get<std::unique_ptr<Muscle>>(any.node);
      std::unique_ptr<Muscle> eff = ps::sdk::Effective(model, e);
      ca.src = &e; ca.serial = e.serial; ca.name = e.name;
      FillActuatorCommon(*eff, ca, ap);
      double tausmooth = eff->tausmooth ? *eff->tausmooth : ap.dynprm[2];
      double tc[2] = {eff->timeconst ? (*eff->timeconst)[0] : -1,
                      eff->timeconst ? (*eff->timeconst)[1] : -1};
      double range[2] = {eff->range ? (*eff->range)[0] : -1,
                         eff->range ? (*eff->range)[1] : -1};
      auto g = [](const ps::opt<double>& v) { return v ? *v : -1.0; };
      actlower::SetToMuscle(ap, tc, tausmooth, range, g(eff->force), g(eff->scale),
                            g(eff->lmin), g(eff->lmax), g(eff->vmax), g(eff->fpmax),
                            g(eff->fvmax));
      ResolveTransmission(*eff, ca, jointid, tendonid, bodyid);
      break;
    }
    case ActuatorAny::Kind::Adhesion: {
      const Adhesion& e = *std::get<std::unique_ptr<Adhesion>>(any.node);
      std::unique_ptr<Adhesion> eff = ps::sdk::Effective(model, e);
      ca.src = &e; ca.serial = e.serial; ca.name = e.name;
      FillActuatorCommon(*eff, ca, ap);
      if (eff->ctrlrange) { ca.ctrlrange[0] = (*eff->ctrlrange)[0];
                            ca.ctrlrange[1] = (*eff->ctrlrange)[1];
                            ap.ctrlrange[0] = ca.ctrlrange[0]; ap.ctrlrange[1] = ca.ctrlrange[1]; }
      actlower::SetToAdhesion(ap, eff->gain ? *eff->gain : ap.gainprm[0]);
      if (eff->body) { auto it = bodyid.find(eff->body->name);
                       ca.trntype = mjTRN_BODY; ca.trnid[0] = it == bodyid.end() ? -1 : it->second; }
      break;
    }
    default:
      break;  // dcmotor / plugin gated out
  }

  // Copy lowered gain/bias/dyn state into the compiled actuator.
  ca.gaintype = ap.gaintype; ca.biastype = ap.biastype; ca.dyntype = ap.dyntype;
  for (int k = 0; k < mjNDYN; ++k) ca.dynprm[k] = ap.dynprm[k];
  for (int k = 0; k < mjNGAIN; ++k) ca.gainprm[k] = ap.gainprm[k];
  for (int k = 0; k < mjNBIAS; ++k) ca.biasprm[k] = ap.biasprm[k];
  ca.actearly = ap.actearly != 0;

  // inheritrange (mjCActuator::Compile): position/intvelocity with matching
  // affine semantics inherit the target's range into ctrlrange/actrange.
  if (ap.gaintype == mjGAIN_FIXED && ap.biastype == mjBIAS_AFFINE &&
      ap.gainprm[0] == -ap.biasprm[1] && ap.inheritrange > 0) {
    double* range = (ap.dyntype == mjDYN_NONE || ap.dyntype == mjDYN_FILTEREXACT)
                        ? ca.ctrlrange
                        : ca.actrange;
    const double* tr = nullptr;
    if (ca.trntype == mjTRN_JOINT) {
      auto it = rl.jointid->find(""); (void)it;
      if (ca.trnid[0] >= 0) tr = (*rl.joints)[ca.trnid[0]].range;
    } else if (ca.trntype == mjTRN_TENDON) {
      if (ca.trnid[0] >= 0) tr = (*rl.tendons)[ca.trnid[0]].range;
    }
    if (tr && tr[0] != tr[1]) {
      double mean = 0.5 * (tr[1] + tr[0]);
      double radius = 0.5 * (tr[1] - tr[0]) * ap.inheritrange;
      range[0] = mean - radius;
      range[1] = mean + radius;
    }
  }

  // actdim (mjCActuator::Compile): default -1 becomes 1 for a stateful dyntype.
  int actdim = ap.actdim;
  if (actdim < 0)
    actdim = (ap.dyntype != mjDYN_NONE && ap.dyntype != mjDYN_DCMOTOR) ? 1 : 0;
  ca.actdim = actdim;

  // tri-state limits -> booleans (islimited).
  ca.ctrllimited = IsLimited(ap.ctrllimited, ca.ctrlrange);
  ca.forcelimited = IsLimited(ap.forcelimited, ca.forcerange);
  ca.actlimited = IsLimited(ap.actlimited, ca.actrange);
  return ca;
}

// nJmom (CountNJmom user_model.cc:3248) over the built transmission set; scope
// covers joint/jointinparent/tendon/body (site/slidercrank are gated).
int ComputeNJmom(const std::vector<CActuator>& acts,
                 const std::vector<CJoint>& joints,
                 const std::vector<CTendon>& tendons,
                 const std::vector<CBody>& cbs, const std::vector<CGeom>& geoms,
                 const std::vector<CSite>& sites, int nv) {
  int count = 0;
  for (const CActuator& a : acts) {
    switch (a.trntype) {
      case mjTRN_JOINT:
      case mjTRN_JOINTINPARENT: {
        int jt = a.trnid[0] >= 0 ? joints[a.trnid[0]].type : mjJNT_HINGE;
        count += jt == mjJNT_BALL ? 3 : jt == mjJNT_FREE ? 6 : 1;
        break;
      }
      case mjTRN_TENDON: {
        std::vector<CTendon> one;
        if (a.trnid[0] >= 0) one.push_back(tendons[a.trnid[0]]);
        count += ComputeNJten(one, cbs, geoms, sites, nv);
        break;
      }
      case mjTRN_BODY:
        count += nv;
        break;
      default:
        break;
    }
  }
  return count;
}

void FillActuators(mjModel* m, const std::vector<CActuator>& acts) {
  int adr = 0;
  for (int i = 0; i < static_cast<int>(acts.size()); ++i) {
    const CActuator& a = acts[i];
    m->actuator_trntype[i] = a.trntype;
    m->actuator_dyntype[i] = a.dyntype;
    m->actuator_gaintype[i] = a.gaintype;
    m->actuator_biastype[i] = a.biastype;
    m->actuator_trnid[2 * i] = a.trnid[0];
    m->actuator_trnid[2 * i + 1] = a.trnid[1];
    m->actuator_actnum[i] = a.actdim;
    m->actuator_actadr[i] = a.actdim ? adr : -1;
    adr += a.actdim;
    m->actuator_group[i] = a.group;
    m->actuator_plugin[i] = -1;
    m->actuator_delay[i] = 0;
    m->actuator_history[2 * i] = 0;
    m->actuator_history[2 * i + 1] = 0;
    m->actuator_historyadr[i] = -1;
    m->actuator_ctrllimited[i] = static_cast<mjtByte>(a.ctrllimited);
    m->actuator_forcelimited[i] = static_cast<mjtByte>(a.forcelimited);
    m->actuator_actlimited[i] = static_cast<mjtByte>(a.actlimited);
    m->actuator_actearly[i] = static_cast<mjtByte>(a.actearly);
    m->actuator_cranklength[i] = a.cranklength;
    lift::mjuu_copyvec(m->actuator_gear + 6 * i, a.gear, 6);
    m->actuator_damping[i] = a.damping[0];
    lift::mjuu_copyvec(m->actuator_dampingpoly + mjNPOLY * i, a.damping + 1, mjNPOLY);
    m->actuator_armature[i] = a.armature;
    lift::mjuu_copyvec(m->actuator_dynprm + mjNDYN * i, a.dynprm, mjNDYN);
    lift::mjuu_copyvec(m->actuator_gainprm + mjNGAIN * i, a.gainprm, mjNGAIN);
    lift::mjuu_copyvec(m->actuator_biasprm + mjNBIAS * i, a.biasprm, mjNBIAS);
    lift::mjuu_copyvec(m->actuator_ctrlrange + 2 * i, a.ctrlrange, 2);
    lift::mjuu_copyvec(m->actuator_forcerange + 2 * i, a.forcerange, 2);
    lift::mjuu_copyvec(m->actuator_actrange + 2 * i, a.actrange, 2);
    lift::mjuu_copyvec(m->actuator_lengthrange + 2 * i, a.lengthrange, 2);
    for (int k = 0; k < m->nuser_actuator && k < static_cast<int>(a.user.size()); ++k)
      m->actuator_user[m->nuser_actuator * i + k] = a.user[k];
  }
}

// --------------------------------------------------------------------------- //
// Sensors (S8). type/objtype/objid/reftype/refid resolution + the per-type      //
// datatype/needstage/dim tables lifted from sensorDatatype/sensorNeedstage      //
// (user_objects.cc:7481-7604) and mjs_sensorDim (user_api.cc:1694). Objtype is   //
// implied by the sensor type (site/joint/tendon/actuator/body sensors) or an     //
// explicit string for frame sensors (mju_str2Type). Plugin/user/tactile/contact //
// /rangefinder/camprojection/insidesite/geom-distance sensors are gated (their   //
// dims need runtime/plugin state or extra refs). Lifted: sensor_tables.          //
// --------------------------------------------------------------------------- //
int SensorDatatype(int type) {
  switch (type) {
    case mjSENS_TOUCH: case mjSENS_INSIDESITE:
      return mjDATATYPE_POSITIVE;
    case mjSENS_FRAMEXAXIS: case mjSENS_FRAMEYAXIS: case mjSENS_FRAMEZAXIS:
    case mjSENS_GEOMNORMAL:
      return mjDATATYPE_AXIS;
    case mjSENS_BALLQUAT: case mjSENS_FRAMEQUAT:
      return mjDATATYPE_QUATERNION;
    default:
      return mjDATATYPE_REAL;
  }
}

int SensorNeedstage(int type) {
  switch (type) {
    case mjSENS_TOUCH: case mjSENS_ACCELEROMETER: case mjSENS_FORCE:
    case mjSENS_TORQUE: case mjSENS_ACTUATORFRC: case mjSENS_JOINTACTFRC:
    case mjSENS_TENDONACTFRC: case mjSENS_JOINTLIMITFRC: case mjSENS_TENDONLIMITFRC:
    case mjSENS_FRAMELINACC: case mjSENS_FRAMEANGACC: case mjSENS_CONTACT:
    case mjSENS_TACTILE:
      return mjSTAGE_ACC;
    case mjSENS_VELOCIMETER: case mjSENS_GYRO: case mjSENS_JOINTVEL:
    case mjSENS_TENDONVEL: case mjSENS_ACTUATORVEL: case mjSENS_BALLANGVEL:
    case mjSENS_JOINTLIMITVEL: case mjSENS_TENDONLIMITVEL: case mjSENS_FRAMELINVEL:
    case mjSENS_FRAMEANGVEL: case mjSENS_SUBTREELINVEL: case mjSENS_SUBTREEANGMOM:
      return mjSTAGE_VEL;
    default:
      return mjSTAGE_POS;
  }
}

// Fixed dim for the in-scope (non-variable-dim) sensor types.
int SensorDim(int type) {
  switch (type) {
    case mjSENS_CAMPROJECTION: return 2;
    case mjSENS_ACCELEROMETER: case mjSENS_VELOCIMETER: case mjSENS_GYRO:
    case mjSENS_FORCE: case mjSENS_TORQUE: case mjSENS_MAGNETOMETER:
    case mjSENS_BALLANGVEL: case mjSENS_FRAMEPOS: case mjSENS_FRAMEXAXIS:
    case mjSENS_FRAMEYAXIS: case mjSENS_FRAMEZAXIS: case mjSENS_FRAMELINVEL:
    case mjSENS_FRAMEANGVEL: case mjSENS_FRAMELINACC: case mjSENS_FRAMEANGACC:
    case mjSENS_SUBTREECOM: case mjSENS_SUBTREELINVEL: case mjSENS_SUBTREEANGMOM:
    case mjSENS_GEOMNORMAL:
      return 3;
    case mjSENS_GEOMFROMTO: return 6;
    case mjSENS_BALLQUAT: case mjSENS_FRAMEQUAT: return 4;
    default: return 1;  // touch/joint*/tendon*/actuator*/clock/energy
  }
}

struct CSensor {
  const void* src = nullptr;
  std::uint64_t serial = 0;
  ps::opt<std::string> name;
  int type = 0;
  int datatype = mjDATATYPE_REAL, needstage = mjSTAGE_POS, dim = 0;
  int objtype = mjOBJ_UNKNOWN, objid = -1;
  int reftype = mjOBJ_UNKNOWN, refid = -1;
  double cutoff = 0, noise = 0;
  double interval[2] = {0, 0};
  std::vector<double> user;
};

// Resolve an object id by mjtObj + name over the per-family id maps.
struct SensorMaps {
  const NameIdMap* body;
  const NameIdMap* geom;
  const NameIdMap* site;
  const NameIdMap* cam;
  const NameIdMap* joint;
  const NameIdMap* tendon;
  const NameIdMap* actuator;
};

int ResolveObj(const SensorMaps& sm, int objtype, const std::string& name) {
  const NameIdMap* m = nullptr;
  switch (objtype) {
    case mjOBJ_BODY: case mjOBJ_XBODY: m = sm.body; break;
    case mjOBJ_GEOM: m = sm.geom; break;
    case mjOBJ_SITE: m = sm.site; break;
    case mjOBJ_CAMERA: m = sm.cam; break;
    case mjOBJ_JOINT: m = sm.joint; break;
    case mjOBJ_TENDON: m = sm.tendon; break;
    case mjOBJ_ACTUATOR: m = sm.actuator; break;
    default: return -1;
  }
  if (!m || name.empty()) return -1;
  auto it = m->find(name);
  return it == m->end() ? -1 : it->second;
}

CSensor SensorCompile(const Model& model, const SensorAny& any,
                      const SensorMaps& sm) {
  CSensor cs;
  std::string objname, refname;

  auto site_sensor = [&](auto& e, int t) {
    cs.type = t; cs.objtype = mjOBJ_SITE;
    if (e.site) objname = e.site->name;
  };
  auto joint_sensor = [&](auto& e, int t) {
    cs.type = t; cs.objtype = mjOBJ_JOINT;
    if (e.joint) objname = e.joint->name;
  };
  auto tendon_sensor = [&](auto& e, int t) {
    cs.type = t; cs.objtype = mjOBJ_TENDON;
    if (e.tendon) objname = e.tendon->name;
  };
  auto actuator_sensor = [&](auto& e, int t) {
    cs.type = t; cs.objtype = mjOBJ_ACTUATOR;
    if (e.actuator) objname = *e.actuator;
  };
  auto body_sensor = [&](auto& e, int t) {
    cs.type = t; cs.objtype = mjOBJ_BODY;
    if (e.body) objname = *e.body;
  };
  auto frame_sensor = [&](auto& e, int t) {
    cs.type = t;
    if (e.objtype) cs.objtype = mju_str2Type(e.objtype->c_str());
    if (e.objname) objname = *e.objname;
    if constexpr (requires { e.reftype; }) {
      if (e.reftype) { cs.reftype = mju_str2Type(e.reftype->c_str());
                       if (e.refname) refname = *e.refname; }
    }
  };

  using K = SensorAny::Kind;
  auto& node = any.node;
  auto get = [&](auto tag) -> auto& {
    return *std::get<std::unique_ptr<std::decay_t<decltype(tag)>>>(node);
  };
  switch (any.kind()) {
    case K::Touch: { auto& e = get(Touch{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     site_sensor(e, mjSENS_TOUCH); cs.cutoff=e.cutoff?*e.cutoff:0;
                     cs.noise=e.noise?*e.noise:0; if(e.user)cs.user=*e.user;
                     if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Accelerometer: { auto& e=get(Accelerometer{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     site_sensor(e, mjSENS_ACCELEROMETER); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Velocimeter: { auto& e=get(Velocimeter{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     site_sensor(e, mjSENS_VELOCIMETER); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Gyro: { auto& e=get(Gyro{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     site_sensor(e, mjSENS_GYRO); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Force: { auto& e=get(Force{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     site_sensor(e, mjSENS_FORCE); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Torque: { auto& e=get(Torque{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     site_sensor(e, mjSENS_TORQUE); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Magnetometer: { auto& e=get(Magnetometer{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     site_sensor(e, mjSENS_MAGNETOMETER); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Jointpos: { auto& e=get(Jointpos{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     joint_sensor(e, mjSENS_JOINTPOS); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Jointvel: { auto& e=get(Jointvel{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     joint_sensor(e, mjSENS_JOINTVEL); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Jointactuatorfrc: { auto& e=get(Jointactuatorfrc{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     joint_sensor(e, mjSENS_JOINTACTFRC); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Ballquat: { auto& e=get(Ballquat{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     joint_sensor(e, mjSENS_BALLQUAT); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Ballangvel: { auto& e=get(Ballangvel{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     joint_sensor(e, mjSENS_BALLANGVEL); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Jointlimitpos: { auto& e=get(Jointlimitpos{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     joint_sensor(e, mjSENS_JOINTLIMITPOS); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Jointlimitvel: { auto& e=get(Jointlimitvel{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     joint_sensor(e, mjSENS_JOINTLIMITVEL); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Jointlimitfrc: { auto& e=get(Jointlimitfrc{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     joint_sensor(e, mjSENS_JOINTLIMITFRC); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Tendonpos: { auto& e=get(Tendonpos{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     tendon_sensor(e, mjSENS_TENDONPOS); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Tendonvel: { auto& e=get(Tendonvel{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     tendon_sensor(e, mjSENS_TENDONVEL); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Tendonactuatorfrc: { auto& e=get(Tendonactuatorfrc{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     tendon_sensor(e, mjSENS_TENDONACTFRC); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Tendonlimitpos: { auto& e=get(Tendonlimitpos{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     tendon_sensor(e, mjSENS_TENDONLIMITPOS); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Tendonlimitvel: { auto& e=get(Tendonlimitvel{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     tendon_sensor(e, mjSENS_TENDONLIMITVEL); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Tendonlimitfrc: { auto& e=get(Tendonlimitfrc{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     tendon_sensor(e, mjSENS_TENDONLIMITFRC); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Actuatorpos: { auto& e=get(Actuatorpos{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     actuator_sensor(e, mjSENS_ACTUATORPOS); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Actuatorvel: { auto& e=get(Actuatorvel{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     actuator_sensor(e, mjSENS_ACTUATORVEL); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Actuatorfrc: { auto& e=get(Actuatorfrc{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     actuator_sensor(e, mjSENS_ACTUATORFRC); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Subtreecom: { auto& e=get(Subtreecom{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     body_sensor(e, mjSENS_SUBTREECOM); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Subtreelinvel: { auto& e=get(Subtreelinvel{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     body_sensor(e, mjSENS_SUBTREELINVEL); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Subtreeangmom: { auto& e=get(Subtreeangmom{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     body_sensor(e, mjSENS_SUBTREEANGMOM); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Framepos: { auto& e=get(Framepos{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     frame_sensor(e, mjSENS_FRAMEPOS); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Framequat: { auto& e=get(Framequat{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     frame_sensor(e, mjSENS_FRAMEQUAT); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Framexaxis: { auto& e=get(Framexaxis{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     frame_sensor(e, mjSENS_FRAMEXAXIS); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Frameyaxis: { auto& e=get(Frameyaxis{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     frame_sensor(e, mjSENS_FRAMEYAXIS); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Framezaxis: { auto& e=get(Framezaxis{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     frame_sensor(e, mjSENS_FRAMEZAXIS); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Framelinvel: { auto& e=get(Framelinvel{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     frame_sensor(e, mjSENS_FRAMELINVEL); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Frameangvel: { auto& e=get(Frameangvel{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     frame_sensor(e, mjSENS_FRAMEANGVEL); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Framelinacc: { auto& e=get(Framelinacc{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     frame_sensor(e, mjSENS_FRAMELINACC); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Frameangacc: { auto& e=get(Frameangacc{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     frame_sensor(e, mjSENS_FRAMEANGACC); cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Clock: { auto& e=get(Clock{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     cs.type=mjSENS_CLOCK; cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::EPotential: { auto& e=get(EPotential{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     cs.type=mjSENS_E_POTENTIAL; cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::EKinetic: { auto& e=get(EKinetic{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     cs.type=mjSENS_E_KINETIC; cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    default:
      break;  // gated types (plugin/user/tactile/contact/rangefinder/...)
  }
  (void)model;
  cs.objid = ResolveObj(sm, cs.objtype, objname);
  cs.refid = ResolveObj(sm, cs.reftype, refname);
  cs.datatype = SensorDatatype(cs.type);
  cs.needstage = SensorNeedstage(cs.type);
  cs.dim = SensorDim(cs.type);
  return cs;
}

void FillSensors(mjModel* m, const std::vector<CSensor>& sensors) {
  int adr = 0;
  for (int i = 0; i < static_cast<int>(sensors.size()); ++i) {
    const CSensor& s = sensors[i];
    m->sensor_type[i] = s.type;
    m->sensor_datatype[i] = s.datatype;
    m->sensor_needstage[i] = s.needstage;
    m->sensor_objtype[i] = s.objtype;
    m->sensor_objid[i] = s.objid;
    m->sensor_reftype[i] = s.reftype;
    m->sensor_refid[i] = s.refid;
    m->sensor_plugin[i] = -1;
    for (int k = 0; k < mjNSENS; ++k) m->sensor_intprm[i * mjNSENS + k] = 0;
    m->sensor_dim[i] = s.dim;
    m->sensor_cutoff[i] = s.cutoff;
    m->sensor_noise[i] = s.noise;
    m->sensor_delay[i] = 0;
    m->sensor_history[2 * i] = 0;
    m->sensor_history[2 * i + 1] = 0;
    m->sensor_interval[2 * i] = s.interval[0];
    m->sensor_interval[2 * i + 1] = s.interval[1];
    m->sensor_historyadr[i] = -1;
    for (int k = 0; k < m->nuser_sensor && k < static_cast<int>(s.user.size()); ++k)
      m->sensor_user[m->nuser_sensor * i + k] = s.user[k];
    m->sensor_adr[i] = adr;
    adr += s.dim;
  }
}

// --------------------------------------------------------------------------- //
// Keyframe fill (S12), mirroring ExpandKeyframe + mjCKey::Compile: an absent    //
// (empty) array defaults wholesale (qpos->qpos0, qvel/act/ctrl->0, mpos/mquat-> //
// mocap ref pose); a SHORTER-than-full authored array keeps its prefix and the  //
// tail is padded (ExpandKeyframe: qpos tail=qpos0, qvel/act/ctrl tail=0,        //
// mpos/mquat new-mocap tail=ref pose). Ball/free qpos quats and mocap quats are //
// normalized. (A longer-than-full array is a hard error upstream, so the model  //
// fails leg B too and never reaches here as a claimed-native compile.)          //
// --------------------------------------------------------------------------- //
void FillKeyframes(mjModel* m, const std::vector<const Key*>& keys) {
  const int nq = m->nq, nv = m->nv, na = m->na, nu = m->nu, nmocap = m->nmocap;
  for (int i = 0; i < static_cast<int>(keys.size()); ++i) {
    const Key& k = *keys[i];
    m->key_time[i] = k.time ? *k.time : 0;

    // qpos: authored prefix (pad tail with qpos0); empty -> all qpos0.
    {
      const int n0 = (k.qpos && !k.qpos->empty())
                         ? std::min(static_cast<int>(k.qpos->size()), nq)
                         : 0;
      for (int j = 0; j < n0; ++j) m->key_qpos[i * nq + j] = (*k.qpos)[j];
      for (int j = n0; j < nq; ++j) m->key_qpos[i * nq + j] = m->qpos0[j];
    }

    // qvel: authored prefix (pad tail with 0); empty -> all 0.
    {
      const int n0 = (k.qvel && !k.qvel->empty())
                         ? std::min(static_cast<int>(k.qvel->size()), nv)
                         : 0;
      for (int j = 0; j < n0; ++j) m->key_qvel[i * nv + j] = (*k.qvel)[j];
      for (int j = n0; j < nv; ++j) m->key_qvel[i * nv + j] = 0;
    }

    if (na) {
      const int n0 = (k.act && !k.act->empty())
                         ? std::min(static_cast<int>(k.act->size()), na)
                         : 0;
      for (int j = 0; j < n0; ++j) m->key_act[i * na + j] = (*k.act)[j];
      for (int j = n0; j < na; ++j) m->key_act[i * na + j] = 0;
    }

    if (nmocap) {
      // mpos/mquat: authored prefix by mocapid, then pad new mocaps (mocapid >=
      // authored count) with the body reference pose (ExpandKeyframe).
      const int mpos0 = (k.mpos && !k.mpos->empty())
                            ? static_cast<int>(k.mpos->size()) / 3 : 0;
      const int mquat0 = (k.mquat && !k.mquat->empty())
                             ? static_cast<int>(k.mquat->size()) / 4 : 0;
      for (int j = 0; j < 3 * std::min(mpos0, nmocap); ++j)
        m->key_mpos[i * 3 * nmocap + j] = (*k.mpos)[j];
      for (int j = 0; j < 4 * std::min(mquat0, nmocap); ++j)
        m->key_mquat[i * 4 * nmocap + j] = (*k.mquat)[j];
      for (int bi = 0; bi < m->nbody; ++bi) {
        const int mid = m->body_mocapid[bi];
        if (mid < 0) continue;
        if (mid >= mpos0)
          lift::mjuu_copyvec(m->key_mpos + i * 3 * nmocap + 3 * mid,
                             m->body_pos + 3 * bi, 3);
        if (mid >= mquat0)
          lift::mjuu_copyvec(m->key_mquat + i * 4 * nmocap + 4 * mid,
                             m->body_quat + 4 * bi, 4);
      }
    }

    // normalize ball/free quats in qpos.
    for (int j = 0; j < m->njnt; ++j)
      if (m->jnt_type[j] == mjJNT_BALL || m->jnt_type[j] == mjJNT_FREE)
        lift::mjuu_normvec(m->key_qpos + i * nq + m->jnt_qposadr[j] +
                               3 * (m->jnt_type[j] == mjJNT_FREE), 4);
    for (int j = 0; j < nmocap; ++j)
      lift::mjuu_normvec(m->key_mquat + i * 4 * nmocap + 4 * j, 4);

    if (nu) {
      const int n0 = (k.ctrl && !k.ctrl->empty())
                         ? std::min(static_cast<int>(k.ctrl->size()), nu)
                         : 0;
      for (int j = 0; j < n0; ++j) m->key_ctrl[i * nu + j] = (*k.ctrl)[j];
      for (int j = n0; j < nu; ++j) m->key_ctrl[i * nu + j] = 0;
    }
  }
}

// --------------------------------------------------------------------------- //
// Assets: textures. Builtin procedural generation (gradient/checker/flat + edge/ //
// cross/random marks, 2D and cube) lifted from mjCTexture::Builtin2D/BuiltinCube //
// + the file-local helpers randomdot/interp/checker (user_objects.cc:4973-5218). //
// Data is a flat rgb byte blob; colorspace is recorded, never applied. File and  //
// cube-face textures are a later wave (gated by the finer scan). ProtoSpec's     //
// TextureBuiltin/TextureMark/TextureType/ColorSpace enums share MuJoCo's integer //
// values, so casts are direct (mjBUILTIN_*/mjMARK_*/mjTEXTURE_*/mjCOLORSPACE_*). //
// --------------------------------------------------------------------------- //

// Lifted verbatim (user_objects.cc:4974-4990): random dots at fixed seed 42.
void TexRandomdot(unsigned char* rgb, const double* markrgb, int width,
                  int height, double probability) {
  std::mt19937_64 rng;
  rng.seed(42);
  std::uniform_real_distribution<double> dist(0, 1);
  for (int r = 0; r < height; r++) {
    for (int c = 0; c < width; c++) {
      if (dist(rng) < probability) {
        for (int j = 0; j < 3; j++)
          rgb[3 * (r * width + c) + j] =
              static_cast<unsigned char>(255 * markrgb[j]);
      }
    }
  }
}

// Lifted verbatim (user_objects.cc:4996-5008): sigmoid interp of two colors.
void TexInterp(unsigned char* rgb, const double* rgb1, const double* rgb2,
               double pos) {
  const double correction = 1.0 / std::sqrt(2);
  double alpha = 0.5 * (1 + pos / std::sqrt(1 + pos * pos) / correction);
  if (alpha < 0) alpha = 0;
  else if (alpha > 1) alpha = 1;
  for (int j = 0; j < 3; j++)
    rgb[j] = static_cast<unsigned char>(255 * (alpha * rgb1[j] + (1 - alpha) * rgb2[j]));
}

// Lifted verbatim (user_objects.cc:5013-5035): checker pattern for one face.
void TexChecker(unsigned char* rgb, const unsigned char* RGB1,
                const unsigned char* RGB2, int width, int height) {
  for (int r = 0; r < height / 2; r++)
    for (int c = 0; c < width / 2; c++)
      std::memcpy(rgb + 3 * (r * width + c), RGB1, 3);
  for (int r = height / 2; r < height; r++)
    for (int c = width / 2; c < width; c++)
      std::memcpy(rgb + 3 * (r * width + c), RGB1, 3);
  for (int r = 0; r < height / 2; r++)
    for (int c = width / 2; c < width; c++)
      std::memcpy(rgb + 3 * (r * width + c), RGB2, 3);
  for (int r = height / 2; r < height; r++)
    for (int c = 0; c < width / 2; c++)
      std::memcpy(rgb + 3 * (r * width + c), RGB2, 3);
}

struct CTexBuiltin {
  int builtin = 0, mark = 0, width = 0, height = 0, nchannel = 3, type = 1;
  double rgb1[3] = {0.8, 0.8, 0.8};
  double rgb2[3] = {0.5, 0.5, 0.5};
  double markrgb[3] = {0, 0, 0};
  double random = 0.01;
};

// mjCTexture::Builtin2D (user_objects.cc:5040-5108). Data is nchannel*w*h bytes,
// pre-zeroed; the rgb helpers write stride-3.
void TexBuiltin2D(const CTexBuiltin& t, unsigned char* data) {
  unsigned char RGB1[3], RGB2[3], RGBm[3];
  for (int j = 0; j < 3; j++) {
    RGB1[j] = static_cast<unsigned char>(255 * t.rgb1[j]);
    RGB2[j] = static_cast<unsigned char>(255 * t.rgb2[j]);
    RGBm[j] = static_cast<unsigned char>(255 * t.markrgb[j]);
  }
  const int width = t.width, height = t.height;
  if (t.builtin == 1) {  // gradient
    for (int r = 0; r < height; r++)
      for (int c = 0; c < width; c++) {
        double x = 2.0 * c / (width - 1) - 1;
        double y = 1 - 2.0 * r / (height - 1);
        double pos = 2 * std::sqrt(x * x + y * y) - 1;
        TexInterp(data + 3 * (r * width + c), t.rgb2, t.rgb1, pos);
      }
  } else if (t.builtin == 2) {  // checker
    TexChecker(data, RGB1, RGB2, width, height);
  } else if (t.builtin == 3) {  // flat
    for (int r = 0; r < height; r++)
      for (int c = 0; c < width; c++)
        std::memcpy(data + 3 * (r * width + c), RGB1, 3);
  }
  if (t.mark == 1) {  // edge
    for (int r = 0; r < height; r++) {
      std::memcpy(data + 3 * (r * width + 0), RGBm, 3);
      std::memcpy(data + 3 * (r * width + width - 1), RGBm, 3);
    }
    for (int c = 0; c < width; c++) {
      std::memcpy(data + 3 * (0 * width + c), RGBm, 3);
      std::memcpy(data + 3 * ((height - 1) * width + c), RGBm, 3);
    }
  } else if (t.mark == 2) {  // cross
    for (int r = 0; r < height; r++)
      std::memcpy(data + 3 * (r * width + width / 2), RGBm, 3);
    for (int c = 0; c < width; c++)
      std::memcpy(data + 3 * (height / 2 * width + c), RGBm, 3);
  } else if (t.mark == 3 && t.random > 0) {  // random dots
    TexRandomdot(data, t.markrgb, width, height, t.random);
  }
}

// mjCTexture::BuiltinCube (user_objects.cc:5113-5218). Six faces stacked, face j
// at byte offset j*3*ww; face order R,L,U,D,F,B = 0..5.
void TexBuiltinCube(const CTexBuiltin& t, unsigned char* data) {
  unsigned char RGB1[3], RGB2[3], RGBm[3], RGBi[3];
  const int w = t.width;
  const std::int64_t ww = static_cast<std::int64_t>(w) * w;
  for (int j = 0; j < 3; j++) {
    RGB1[j] = static_cast<unsigned char>(255 * t.rgb1[j]);
    RGB2[j] = static_cast<unsigned char>(255 * t.rgb2[j]);
    RGBm[j] = static_cast<unsigned char>(255 * t.markrgb[j]);
  }
  if (t.builtin == 1) {  // gradient
    for (int r = 0; r < w; r++)
      for (int c = 0; c < w; c++) {
        double x = 2.0 * c / (w - 1) - 1;
        double y = 1 - 2.0 * r / (w - 1);
        double elside = std::asin(y / std::sqrt(1 + x * x + y * y)) / (0.5 * mjPI);
        double elup = 1 - std::acos(1.0 / std::sqrt(1 + x * x + y * y)) / (0.5 * mjPI);
        TexInterp(RGBi, t.rgb1, t.rgb2, elside);
        std::memcpy(data + 0 * 3 * ww + 3 * (r * w + c), RGBi, 3);
        std::memcpy(data + 1 * 3 * ww + 3 * (r * w + c), RGBi, 3);
        std::memcpy(data + 4 * 3 * ww + 3 * (r * w + c), RGBi, 3);
        std::memcpy(data + 5 * 3 * ww + 3 * (r * w + c), RGBi, 3);
        TexInterp(data + 2 * 3 * ww + 3 * (r * w + c), t.rgb1, t.rgb2, elup);
        TexInterp(data + 3 * 3 * ww + 3 * (r * w + c), t.rgb1, t.rgb2, -elup);
      }
  } else if (t.builtin == 2) {  // checker
    TexChecker(data + 0 * 3 * ww, RGB1, RGB2, w, w);
    TexChecker(data + 1 * 3 * ww, RGB1, RGB2, w, w);
    TexChecker(data + 2 * 3 * ww, RGB1, RGB2, w, w);
    TexChecker(data + 3 * 3 * ww, RGB1, RGB2, w, w);
    TexChecker(data + 4 * 3 * ww, RGB2, RGB1, w, w);
    TexChecker(data + 5 * 3 * ww, RGB2, RGB1, w, w);
  } else if (t.builtin == 3) {  // flat
    for (int r = 0; r < w; r++)
      for (int c = 0; c < w; c++) {
        std::memcpy(data + 0 * 3 * ww + 3 * (r * w + c), RGB1, 3);
        std::memcpy(data + 1 * 3 * ww + 3 * (r * w + c), RGB1, 3);
        std::memcpy(data + 2 * 3 * ww + 3 * (r * w + c), RGB1, 3);
        std::memcpy(data + 4 * 3 * ww + 3 * (r * w + c), RGB1, 3);
        std::memcpy(data + 5 * 3 * ww + 3 * (r * w + c), RGB1, 3);
        std::memcpy(data + 3 * 3 * ww + 3 * (r * w + c), RGB2, 3);
      }
  }
  if (t.mark == 1) {  // edge
    for (int j = 0; j < 6; j++) {
      for (int r = 0; r < w; r++) {
        std::memcpy(data + j * 3 * ww + 3 * (r * w + 0), RGBm, 3);
        std::memcpy(data + j * 3 * ww + 3 * (r * w + w - 1), RGBm, 3);
      }
      for (int c = 0; c < w; c++) {
        std::memcpy(data + j * 3 * ww + 3 * (0 * w + c), RGBm, 3);
        std::memcpy(data + j * 3 * ww + 3 * ((w - 1) * w + c), RGBm, 3);
      }
    }
  } else if (t.mark == 2) {  // cross
    for (int j = 0; j < 6; j++) {
      for (int r = 0; r < w; r++)
        std::memcpy(data + j * 3 * ww + 3 * (r * w + w / 2), RGBm, 3);
      for (int c = 0; c < w; c++)
        std::memcpy(data + j * 3 * ww + 3 * (w / 2 * w + c), RGBm, 3);
    }
  } else if (t.mark == 3 && t.random > 0) {  // random dots
    TexRandomdot(data, t.markrgb, w, t.height, t.random);
  }
}

struct CTexture {
  const Texture* src = nullptr;
  ps::opt<std::string> name;
  int type = 1;         // mjtTexture (TextureType casts directly)
  int colorspace = 0;   // mjtColorSpace
  int width = 0, height = 0, nchannel = 3;
  std::vector<unsigned char> data;
};

// mjCTexture::Compile builtin path (user_objects.cc:5632-5686). Returns false and
// records a diagnostic on invalid dimensions; the gate only admits builtin
// textures so file/cube-face branches are not reached here.
bool TextureCompile(const Model& model, const Texture& tx, CTexture& out,
                    std::vector<bridge::Diagnostic>& diags) {
  std::unique_ptr<Texture> eff = ps::sdk::Effective(model, tx);
  out.src = &tx;
  out.name = tx.name;
  out.type = eff->type ? static_cast<int>(*eff->type) : 1;
  out.colorspace = eff->colorspace ? static_cast<int>(*eff->colorspace) : 0;
  out.nchannel = eff->nchannel ? *eff->nchannel : 3;

  CTexBuiltin b;
  b.type = out.type;
  b.nchannel = out.nchannel;
  if (const TextureBuiltin* bi =
          eff->source ? std::get_if<TextureBuiltin>(&*eff->source) : nullptr)
    b.builtin = static_cast<int>(*bi);
  b.mark = eff->mark ? static_cast<int>(*eff->mark) : 0;
  if (eff->rgb1) for (int k = 0; k < 3; ++k) b.rgb1[k] = (*eff->rgb1)[k];
  if (eff->rgb2) for (int k = 0; k < 3; ++k) b.rgb2[k] = (*eff->rgb2)[k];
  if (eff->markrgb) for (int k = 0; k < 3; ++k) b.markrgb[k] = (*eff->markrgb)[k];
  if (eff->random) b.random = *eff->random;
  b.width = eff->width ? *eff->width : 0;
  b.height = eff->height ? *eff->height : 0;

  // dimension checks (mjCTexture::Compile:5652-5667).
  if (b.width < 1) {
    diags.push_back({bridge::Diagnostic::Severity::Error, "texture",
                     "Invalid width of builtin texture", tx.loc});
    return false;
  }
  if (out.type != 0 /* 2D */) {
    b.height = 6 * b.width;
  } else if (b.height < 1) {
    diags.push_back({bridge::Diagnostic::Severity::Error, "texture",
                     "Invalid height of builtin texture", tx.loc});
    return false;
  }
  out.width = b.width;
  out.height = b.height;
  out.data.assign(static_cast<std::size_t>(out.nchannel) * b.width * b.height, 0);
  if (out.type == 0) TexBuiltin2D(b, out.data.data());
  else TexBuiltinCube(b, out.data.data());
  return true;
}

void FillTextures(mjModel* m, const std::vector<CTexture>& texs) {
  mjtSize data_adr = 0;
  for (int i = 0; i < static_cast<int>(texs.size()); ++i) {
    const CTexture& t = texs[i];
    m->tex_type[i] = t.type;
    m->tex_colorspace[i] = t.colorspace;
    m->tex_height[i] = t.height;
    m->tex_width[i] = t.width;
    m->tex_nchannel[i] = t.nchannel;
    m->tex_adr[i] = data_adr;
    const mjtSize nbytes =
        static_cast<mjtSize>(t.nchannel) * t.width * t.height;
    if (nbytes) std::memcpy(m->tex_data + data_adr, t.data.data(), nbytes);
    data_adr += nbytes;
  }
}

// --------------------------------------------------------------------------- //
// Assets: materials (CopyObjects, user_model.cc:3760-3777). mjCMaterial::Compile //
// is a no-op copy; the only resolution is the per-role texture id array         //
// (mat_texid): the legacy `texture` attr maps to role mjTEXROLE_RGB and each     //
// <layer role=.. texture=..> to its role (mjXReader::OneMaterial). Class-chain + //
// IDL defaults come from ps::sdk::Effective<Material> (metallic/roughness default //
// -1, which the IDL layer does not set, so the struct carries it).               //
// --------------------------------------------------------------------------- //
struct CMaterial {
  const Material* src = nullptr;
  ps::opt<std::string> name;
  std::uint64_t serial = 0;
  int texid[mjNTEXROLE];
  int texuniform = 0;
  float texrepeat[2] = {1, 1};
  float emission = 0, specular = 0.5f, shininess = 0.5f, reflectance = 0;
  float metallic = -1, roughness = -1;
  float rgba[4] = {1, 1, 1, 1};
  CMaterial() { for (int r = 0; r < mjNTEXROLE; ++r) texid[r] = -1; }
};

CMaterial MaterialCompile(const Model& model, const Material& mat,
                          const NameIdMap& texid_of) {
  std::unique_ptr<Material> eff = ps::sdk::Effective(model, mat);
  CMaterial cm;
  cm.src = &mat;
  cm.name = mat.name;
  cm.serial = mat.serial;
  auto resolve_tex = [&](const std::string& nm) -> int {
    auto it = texid_of.find(nm);
    return it == texid_of.end() ? -1 : it->second;
  };
  // legacy `texture` attr -> role RGB; <layer> entries -> their role. ProtoSpec's
  // TexRole enum omits the USER role (rgb=0..orm=8), so it is offset +1 from
  // mjtTextureRole (mjTEXROLE_USER=0, RGB=1..ORM=9).
  if (eff->texture) cm.texid[mjTEXROLE_RGB] = resolve_tex(eff->texture->name);
  for (const auto& layer : eff->layers) {
    if (!layer) continue;
    const int role =
        layer->role ? static_cast<int>(*layer->role) + 1 : mjTEXROLE_USER;
    if (layer->texture) cm.texid[role] = resolve_tex(layer->texture->name);
  }
  if (eff->texuniform) cm.texuniform = *eff->texuniform ? 1 : 0;
  if (eff->texrepeat) { cm.texrepeat[0] = (*eff->texrepeat)[0];
                        cm.texrepeat[1] = (*eff->texrepeat)[1]; }
  if (eff->emission) cm.emission = *eff->emission;
  if (eff->specular) cm.specular = *eff->specular;
  if (eff->shininess) cm.shininess = *eff->shininess;
  if (eff->reflectance) cm.reflectance = *eff->reflectance;
  if (eff->metallic) cm.metallic = *eff->metallic;
  if (eff->roughness) cm.roughness = *eff->roughness;
  if (eff->rgba) for (int k = 0; k < 4; ++k) cm.rgba[k] = (*eff->rgba)[k];
  return cm;
}

void FillMaterials(mjModel* m, const std::vector<CMaterial>& mats) {
  for (int i = 0; i < static_cast<int>(mats.size()); ++i) {
    const CMaterial& cm = mats[i];
    for (int j = 0; j < mjNTEXROLE; ++j)
      m->mat_texid[mjNTEXROLE * i + j] = cm.texid[j];
    m->mat_texuniform[i] = static_cast<mjtByte>(cm.texuniform);
    m->mat_texrepeat[2 * i] = cm.texrepeat[0];
    m->mat_texrepeat[2 * i + 1] = cm.texrepeat[1];
    m->mat_emission[i] = cm.emission;
    m->mat_specular[i] = cm.specular;
    m->mat_shininess[i] = cm.shininess;
    m->mat_reflectance[i] = cm.reflectance;
    m->mat_metallic[i] = cm.metallic;
    m->mat_roughness[i] = cm.roughness;
    for (int k = 0; k < 4; ++k) m->mat_rgba[4 * i + k] = cm.rgba[k];
  }
}

// --------------------------------------------------------------------------- //
// Assets: height fields (mjCHField::Compile, user_objects.cc). The user-data    //
// path (elevation authored inline): copy nrow*ncol floats, validate size>0,     //
// normalize to [0,1] (subtract emin, divide by emax-emin when > mjEPS). File    //
// (PNG/custom) hfields are a later wave (gated). geom_dataid + the hfield geom's //
// size/aabb/rbound come from the hfield in GeomCompile (bound before the body    //
// walk).                                                                         //
// --------------------------------------------------------------------------- //
struct CHField {
  const Hfield* src = nullptr;
  ps::opt<std::string> name;
  int nrow = 0, ncol = 0;
  double size[4] = {0, 0, 0, 0};
  std::vector<float> data;  // normalized elevation, row-major
};

bool HfieldCompile(const Hfield& hf, CHField& out,
                   std::vector<bridge::Diagnostic>& diags) {
  out.src = &hf;
  out.name = hf.name;
  out.nrow = hf.nrow ? *hf.nrow : 0;
  out.ncol = hf.ncol ? *hf.ncol : 0;
  if (hf.size) for (int k = 0; k < 4; ++k) out.size[k] = (*hf.size)[k];

  // user elevation data. The XML reader (xml_native_reader.cc OneHField) copies
  // elevation in REVERSE row order ("so XML string is top-to-bottom"); leg B
  // round-trips through that reader, so leg C reproduces the same row flip before
  // mjCHField::Compile's normalization.
  if (hf.elevation && !hf.elevation->empty()) {
    const auto& e = *hf.elevation;
    if (out.nrow * out.ncol != static_cast<int>(e.size())) {
      diags.push_back({bridge::Diagnostic::Severity::Error, "hfield",
                       "elevation data length must match nrow*ncol", hf.loc});
      return false;
    }
    out.data.resize(e.size());
    for (int i = 0; i < out.nrow; ++i) {
      const int flip = out.nrow - 1 - i;
      for (int j = 0; j < out.ncol; ++j)
        out.data[flip * out.ncol + j] =
            static_cast<float>(e[i * out.ncol + j]);
    }
  }

  // size parameters must be positive.
  for (int k = 0; k < 4; ++k)
    if (out.size[k] <= 0) {
      diags.push_back({bridge::Diagnostic::Severity::Error, "hfield",
                       "size parameter is not positive in hfield", hf.loc});
      return false;
    }
  if (out.nrow < 1 || out.ncol < 1 || out.data.empty()) {
    diags.push_back({bridge::Diagnostic::Severity::Error, "hfield",
                     "hfield not specified", hf.loc});
    return false;
  }

  // normalize elevation to [0, 1] (mjCHField::Compile).
  float emin = 1e10f, emax = -1e10f;
  for (int i = 0; i < out.nrow * out.ncol; ++i) {
    emin = std::min(emin, out.data[i]);
    emax = std::max(emax, out.data[i]);
  }
  for (int i = 0; i < out.nrow * out.ncol; ++i) {
    out.data[i] -= emin;
    if (emax - emin > static_cast<float>(lift::mjEPS)) out.data[i] /= (emax - emin);
  }
  return true;
}

void FillHfields(mjModel* m, const std::vector<CHField>& hfields) {
  int adr = 0;
  for (int i = 0; i < static_cast<int>(hfields.size()); ++i) {
    const CHField& hf = hfields[i];
    m->hfield_nrow[i] = hf.nrow;
    m->hfield_ncol[i] = hf.ncol;
    for (int k = 0; k < 4; ++k) m->hfield_size[4 * i + k] = hf.size[k];
    m->hfield_adr[i] = adr;
    m->hfield_pathadr[i] = -1;  // user data: no file path
    for (int j = 0; j < hf.nrow * hf.ncol; ++j)
      m->hfield_data[adr + j] = hf.data[j];
    adr += hf.nrow * hf.ncol;
  }
}

// --------------------------------------------------------------------------- //
// Meshes (S8/CopyObjects). Compiled before the body walk (like hfields) so a    //
// mesh geom binds the mesh aamm/frame/inertia. The heavy lifting -- parse,      //
// convex hull, inertia, polygons -- is the lifted pipeline; build.cc resolves   //
// files, drives the shared face BVH, and lays out the mjModel mesh_* arrays.    //
// --------------------------------------------------------------------------- //
struct CMesh {
  ps::opt<std::string> name;           // authored name (bind key), may be empty
  std::string listname;                // name table entry (authored else file stem)
  std::string file;                    // File() for m->paths (empty: user mesh)
  lift::MeshResult r;
  std::vector<double> bvh;
  std::vector<int> bvh_child, bvh_level, bvh_nodeid;
  int nbvh() const { return static_cast<int>(bvh_child.size() / 2); }
};

// ProtoSpec MeshInertia -> mjtMeshInertia (the enum orderings differ: ProtoSpec
// is convex/legacy/exact/shell, mjtMeshInertia is convex/exact/legacy/shell).
int MeshInertiaToMj(const ps::opt<MeshInertia>& mi) {
  if (!mi) return mjMESH_INERTIA_LEGACY;
  switch (*mi) {
    case MeshInertia::convex: return mjMESH_INERTIA_CONVEX;
    case MeshInertia::exact:  return mjMESH_INERTIA_EXACT;
    case MeshInertia::legacy: return mjMESH_INERTIA_LEGACY;
    case MeshInertia::shell:  return mjMESH_INERTIA_SHELL;
  }
  return mjMESH_INERTIA_LEGACY;
}

// Which meshes a collision (or contact-pair) mesh geom references -- these force
// the convex-hull graph even when the mesh's own inertia is not convex
// (mjCModel::IndexAssets SetNeedHull, run before mesh compile).
void CollectMeshHullRefs(const Model& m, const std::set<std::string>& pair_geoms,
                         std::unordered_map<std::string, bool>& out) {
  std::function<void(const Body&)> walk_body;
  std::function<void(const std::vector<BodyChildAny>&)> walk_children;
  walk_children = [&](const std::vector<BodyChildAny>& subtree) {
    for (const BodyChildAny& c : subtree) {
      switch (c.kind()) {
        case BodyChildAny::Kind::Geom: {
          const auto& g = std::get<std::unique_ptr<Geom>>(c.node);
          if (!g) break;
          std::unique_ptr<Geom> eff = ps::sdk::Effective(m, *g);
          if (!eff->mesh) break;
          const int type = eff->type ? static_cast<int>(*eff->type) : mjGEOM_SPHERE;
          const int contype = eff->contype ? *eff->contype : 1;
          const int conaff = eff->conaffinity ? *eff->conaffinity : 1;
          const bool in_pair =
              eff->name && pair_geoms.count(*eff->name) > 0;
          const bool coll =
              type == mjGEOM_MESH && (contype || conaff || in_pair);
          out[eff->mesh->name] = out[eff->mesh->name] || coll;
          break;
        }
        case BodyChildAny::Kind::Body: {
          const auto& b = std::get<std::unique_ptr<Body>>(c.node);
          if (b) walk_body(*b);
          break;
        }
        case BodyChildAny::Kind::Frame: {
          const auto& f = std::get<std::unique_ptr<Frame>>(c.node);
          if (f) walk_children(f->subtree);
          break;
        }
        default:
          break;
      }
    }
  };
  walk_body = [&](const Body& b) { walk_children(b.subtree); };
  for (const auto& b : m.worldbody)
    if (b) walk_body(*b);
}

// Strip any directory prefix (mjuu_strippath).
std::string MeshStripPath(const std::string& f) {
  size_t s = f.find_last_of("/\\");
  return s == std::string::npos ? f : f.substr(s + 1);
}

// Extension -> content type (mjuu_extToContentType, mesh formats only).
std::string MeshExtToContentType(const std::string& file) {
  size_t dot = file.find_last_of('.');
  std::string ext = dot == std::string::npos ? std::string() : file.substr(dot);
  for (char& c : ext) c = static_cast<char>(std::tolower(c));
  if (ext == ".stl") return "model/stl";
  if (ext == ".obj") return "model/obj";
  if (ext == ".msh") return "model/vnd.mujoco.msh";
  return "";
}

// Join meshdir with a (possibly already stripped) file (FilePath::Combine).
std::string MeshCombine(const std::string& dir, const std::string& file) {
  if (!file.empty() && (file[0] == '/' || file[0] == '\\' ||
                        file.find(":/") != std::string::npos ||
                        file.find(":\\") != std::string::npos)) {
    return file;
  }
  if (dir.empty()) return file;
  char back = dir.back();
  if (back != '/' && back != '\\') return dir + "/" + file;
  return dir + file;
}

// Compile one mesh into a CMesh (files resolved via the resource API).
bool MeshCompile(const Model& model, const Mesh& mesh, const CompilerSettings& cs,
                 const std::string& base_dir,
                 const std::unordered_map<std::string, bool>& mesh_hull,
                 CMesh& out, std::vector<bridge::Diagnostic>& diags) {
  std::unique_ptr<Mesh> eff = ps::sdk::Effective(model, mesh);
  out.name = eff->name;

  lift::MeshInput in;
  // Name table entry: authored name, else the file stem (strippath + stripext),
  // reproducing mjCMesh::CopyFromSpec's file-derived naming.
  if (eff->name) {
    out.listname = *eff->name;
  } else if (eff->file) {
    std::string stem = MeshStripPath(*eff->file);
    size_t dot = stem.find_last_of('.');
    out.listname = dot == std::string::npos ? stem : stem.substr(0, dot);
  }

  // needhull: a collision/pair mesh geom or convex inertia forces the hull graph
  // (SetNeedHull is keyed by the mesh's canonical name).
  bool needhull = false;
  if (!out.listname.empty()) {
    auto it = mesh_hull.find(out.listname);
    if (it != mesh_hull.end()) needhull = it->second;
  }

  if (eff->scale) for (int k = 0; k < 3; ++k) in.scale[k] = (*eff->scale)[k];
  if (eff->refpos) for (int k = 0; k < 3; ++k) in.refpos[k] = (*eff->refpos)[k];
  if (eff->refquat) for (int k = 0; k < 4; ++k) in.refquat[k] = (*eff->refquat)[k];
  if (eff->smoothnormal) in.smoothnormal = *eff->smoothnormal;
  if (eff->maxhullvert) in.maxhullvert = *eff->maxhullvert;
  in.inertia = MeshInertiaToMj(eff->inertia);
  in.needhull = needhull || in.inertia == mjMESH_INERTIA_CONVEX;

  if (eff->file) {
    std::string file = *eff->file;
    if (cs.strippath) file = MeshStripPath(file);
    out.file = file;
    const std::string combined = MeshCombine(cs.meshdir, file);
    char err[1024] = {0};
    mjResource* res = mju_openResource(base_dir.empty() ? nullptr : base_dir.c_str(),
                                       combined.c_str(), nullptr, err, sizeof(err));
    if (!res) {
      diags.push_back({bridge::Diagnostic::Severity::Error, "mesh",
                       std::string("could not open mesh file '") + combined + "'",
                       mesh.loc});
      return false;
    }
    const void* bytes = nullptr;
    int n = mju_readResource(res, &bytes);
    if (n < 0) {
      mju_closeResource(res);
      diags.push_back({bridge::Diagnostic::Severity::Error, "mesh",
                       std::string("could not read mesh file '") + combined + "'",
                       mesh.loc});
      return false;
    }
    in.content_type = MeshExtToContentType(file);
    if (in.content_type == "model/vnd.mujoco.msh") in.format = lift::MeshFormat::Msh;
    else if (in.content_type == "model/obj") in.format = lift::MeshFormat::Obj;
    else if (in.content_type == "model/stl") in.format = lift::MeshFormat::Stl;
    else {
      mju_closeResource(res);
      diags.push_back({bridge::Diagnostic::Severity::Error, "mesh",
                       "unsupported mesh file format", mesh.loc});
      return false;
    }
    in.filebytes.assign(static_cast<const char*>(bytes),
                        static_cast<const char*>(bytes) + n);
    mju_closeResource(res);
  } else {
    in.format = lift::MeshFormat::UserVertex;
    if (eff->vertex)
      for (double v : *eff->vertex) in.uservert.push_back(static_cast<float>(v));
    if (eff->normal)
      for (double v : *eff->normal) in.usernormal.push_back(static_cast<float>(v));
    if (eff->texcoord)
      for (float v : *eff->texcoord) in.usertexcoord.push_back(v);
    if (eff->face)
      for (int v : *eff->face) in.userface.push_back(v);
  }

  std::string err;
  if (!lift::CompileMesh(in, out.r, err)) {
    diags.push_back({bridge::Diagnostic::Severity::Error, "mesh",
                     "mesh compile failed: " + err, mesh.loc});
    return false;
  }

  // Face BVH: one leaf per face (contype/conaffinity = 1, identity frame, pos =
  // face circumcenter, aabb = face aabb), built by the shared BVH kernel.
  BVH tree;
  double zero[3] = {0, 0, 0}, ident[4] = {1, 0, 0, 0};
  tree.Set(zero, ident);
  for (int i = 0; i < out.r.nface(); ++i) {
    BVHLeaf leaf;
    leaf.id = i;
    leaf.contype = 1;
    leaf.conaffinity = 1;
    leaf.pos = out.r.center.data() + 3 * i;
    leaf.quat = nullptr;
    leaf.aabb = out.r.face_aabb.data() + 6 * i;
    tree.Add(leaf);
  }
  tree.Create();
  out.bvh = tree.bvh();
  out.bvh_child = tree.child();
  out.bvh_level = tree.level();
  out.bvh_nodeid = tree.nodeid();
  return true;
}

// Fill mesh_* arrays and append the mesh face-BVH nodes after all body BVH nodes
// (CopyObjects mesh loop). Returns the number of mesh BVH nodes written.
void FillMeshes(mjModel* m, const std::vector<CMesh>& meshes, int bvh_start) {
  int vert_adr = 0, normal_adr = 0, texcoord_adr = 0, face_adr = 0;
  int graph_adr = 0, poly_adr = 0, polyvert_adr = 0, polymap_adr = 0;
  int bvh_adr = bvh_start;
  for (int i = 0; i < static_cast<int>(meshes.size()); ++i) {
    const lift::MeshResult& r = meshes[i].r;
    m->mesh_polyadr[i] = poly_adr;
    m->mesh_polynum[i] = r.npolygon();
    m->mesh_vertadr[i] = vert_adr;
    m->mesh_vertnum[i] = r.nvert();
    m->mesh_normaladr[i] = normal_adr;
    m->mesh_normalnum[i] = r.nnormal();
    m->mesh_texcoordadr[i] = r.has_texcoord() ? texcoord_adr : -1;
    m->mesh_texcoordnum[i] = r.ntexcoord();
    m->mesh_faceadr[i] = face_adr;
    m->mesh_facenum[i] = r.nface();
    m->mesh_graphadr[i] = r.szgraph() ? graph_adr : -1;
    m->mesh_bvhnum[i] = meshes[i].nbvh();
    m->mesh_bvhadr[i] = meshes[i].nbvh() ? bvh_adr : -1;
    m->mesh_octadr[i] = -1;
    m->mesh_octnum[i] = 0;
    for (int k = 0; k < 3; ++k) m->mesh_scale[3 * i + k] = r.scale[k];
    for (int k = 0; k < 3; ++k) m->mesh_pos[3 * i + k] = r.pos[k];
    for (int k = 0; k < 4; ++k) m->mesh_quat[4 * i + k] = r.quat[k];

    for (int k = 0; k < 3 * r.nvert(); ++k) m->mesh_vert[3 * vert_adr + k] = r.vert[k];
    for (int k = 0; k < 3 * r.nnormal(); ++k)
      m->mesh_normal[3 * normal_adr + k] = r.normal[k];
    for (int k = 0; k < 3 * r.nface(); ++k) m->mesh_face[3 * face_adr + k] = r.face[k];
    for (int k = 0; k < 3 * r.nface(); ++k)
      m->mesh_facenormal[3 * face_adr + k] = r.facenormal[k];
    if (r.has_texcoord()) {
      for (int k = 0; k < 2 * r.ntexcoord(); ++k)
        m->mesh_texcoord[2 * texcoord_adr + k] = r.texcoord[k];
      for (int k = 0; k < 3 * r.nface(); ++k)
        m->mesh_facetexcoord[3 * face_adr + k] = r.facetexcoord[k];
    } else {
      std::memset(m->mesh_facetexcoord + 3 * face_adr, 0,
                  3 * r.nface() * sizeof(int));
    }
    if (r.szgraph())
      for (int k = 0; k < r.szgraph(); ++k) m->mesh_graph[graph_adr + k] = r.graph[k];

    for (int k = 0; k < 3 * r.npolygon(); ++k)
      m->mesh_polynormal[3 * poly_adr + k] = r.polygon_normals[k];
    int pv = polyvert_adr;
    for (int p = 0; p < r.npolygon(); ++p) {
      int sz = static_cast<int>(r.polygons[p].size());
      m->mesh_polyvertnum[poly_adr + p] = sz;
      m->mesh_polyvertadr[poly_adr + p] = pv;
      for (int j = 0; j < sz; ++j) m->mesh_polyvert[pv + j] = r.polygons[p][j];
      pv += sz;
    }
    int pm = polymap_adr;
    for (int v = 0; v < r.nvert(); ++v) {
      int sz = static_cast<int>(r.polygon_map[v].size());
      m->mesh_polymapnum[vert_adr + v] = sz;
      m->mesh_polymapadr[vert_adr + v] = pm;
      for (int j = 0; j < sz; ++j) m->mesh_polymap[pm + j] = r.polygon_map[v][j];
      pm += sz;
    }

    const int nb = meshes[i].nbvh();
    for (int k = 0; k < nb; ++k) {
      lift::mjuu_copyvec(m->bvh_aabb + 6 * (bvh_adr + k),
                         meshes[i].bvh.data() + 6 * k, 6);
      m->bvh_child[2 * (bvh_adr + k)] = meshes[i].bvh_child[2 * k];
      m->bvh_child[2 * (bvh_adr + k) + 1] = meshes[i].bvh_child[2 * k + 1];
      m->bvh_depth[bvh_adr + k] = meshes[i].bvh_level[k];
      m->bvh_nodeid[bvh_adr + k] = meshes[i].bvh_nodeid[k];
    }

    poly_adr += r.npolygon();
    polyvert_adr += r.npolygonvert();
    polymap_adr += r.npolygonmap();
    vert_adr += r.nvert();
    normal_adr += r.nnormal();
    texcoord_adr += r.has_texcoord() ? r.ntexcoord() : 0;
    face_adr += r.nface();
    graph_adr += r.szgraph();
    bvh_adr += nb;
  }
}

// Asset file paths, in CopyPaths order (hfields, meshes, skins, textures). Only
// file meshes carry a path in the native scope; the rest are -1.
void FillMeshPaths(mjModel* m, const std::vector<CMesh>& meshes) {
  m->paths[0] = 0;
  int adr = 0;
  for (int i = 0; i < m->nhfield; ++i) m->hfield_pathadr[i] = -1;
  for (int i = 0; i < static_cast<int>(meshes.size()); ++i) {
    if (meshes[i].file.empty()) {
      m->mesh_pathadr[i] = -1;
      continue;
    }
    m->mesh_pathadr[i] = adr;
    const std::string& f = meshes[i].file;
    std::memcpy(m->paths + adr, f.c_str(), f.size());
    adr += static_cast<int>(f.size());
    m->paths[adr] = 0;
    adr++;
  }
  for (int i = 0; i < m->ntex; ++i) m->tex_pathadr[i] = -1;
}

}  // namespace

mjModel* BuildNativeModel(const Model& m, const bridge::CompileOptions& opts,
                          std::vector<bridge::Diagnostic>& diags) {
  const CompilerSettings cs = ReadCompiler(m);

  // <visual><map znear> feeds a camera's intrinsic fallback (no sensorsize).
  double znear = 0.01;  // mjs_defaultVisual (engine_init.c:176)
  for (const auto& v : m.visuals) {
    if (!v) continue;
    for (const auto& mp : v->visualMaps)
      if (mp && mp->znear) znear = *mp->znear;
  }

  // Assets that geoms bind to (hfields; meshes in a later wave) are compiled
  // BEFORE the body walk so a mesh/hfield geom's size/aabb/rbound/inertia -- and
  // the body inertia it feeds -- see the final asset geometry, exactly as MuJoCo
  // compiles assets before bodies.
  std::vector<CHField> hfields;
  for (const auto& as : m.assets) {
    if (!as) continue;
    for (const auto& hf : as->hfields) {
      if (!hf) continue;
      CHField ch;
      if (!HfieldCompile(*hf, ch, diags)) return nullptr;
      hfields.push_back(std::move(ch));
    }
  }
  AssetBinds asset_binds;
  for (int i = 0; i < static_cast<int>(hfields.size()); ++i) {
    if (!hfields[i].name) continue;
    HfieldBind hb;
    hb.id = i;
    for (int k = 0; k < 4; ++k) hb.size[k] = hfields[i].size[k];
    asset_binds.hfield[*hfields[i].name] = hb;
  }

  // Meshes: determine per-mesh needhull (collision / pair / convex-inertia mesh
  // geoms force the hull graph), then compile in document order.
  std::set<std::string> pair_geoms;
  for (const auto& ct : m.contacts) {
    if (!ct) continue;
    for (const auto& p : ct->pairs) {
      if (!p) continue;
      if (p->geom1) pair_geoms.insert(p->geom1->name);
      if (p->geom2) pair_geoms.insert(p->geom2->name);
    }
  }
  std::unordered_map<std::string, bool> mesh_hull;
  CollectMeshHullRefs(m, pair_geoms, mesh_hull);

  std::vector<CMesh> meshes;
  for (const auto& as : m.assets) {
    if (!as) continue;
    for (const auto& mesh : as->meshs) {
      if (!mesh) continue;
      CMesh cm;
      if (!MeshCompile(m, *mesh, cs, opts.base_dir, mesh_hull, cm, diags))
        return nullptr;
      meshes.push_back(std::move(cm));
    }
  }
  for (int i = 0; i < static_cast<int>(meshes.size()); ++i) {
    if (meshes[i].listname.empty()) continue;
    MeshBind mb;
    mb.id = i;
    lift::mjuu_copyvec(mb.aamm, meshes[i].r.aamm, 6);
    lift::mjuu_copyvec(mb.pos, meshes[i].r.pos, 3);
    lift::mjuu_copyvec(mb.quat, meshes[i].r.quat, 4);
    mb.volume = meshes[i].r.volume_ref;
    lift::mjuu_copyvec(mb.boxsz, meshes[i].r.boxsz, 3);
    asset_binds.mesh[meshes[i].listname] = mb;
  }

  // S1 Collect. The world body (id 0) is implicit; every <worldbody> block's
  // direct children are merged into it (MuJoCo include semantics).
  BodyCollector collector(m, cs, opts, znear, asset_binds);
  collector.Run(m.worldbody);
  const std::vector<CBody>& cbs = collector.bodies();
  std::vector<CGeom> geoms = collector.geoms();
  const std::vector<CJoint>& joints = collector.joints();
  std::vector<CSite> sites = collector.sites();
  const std::vector<CCamera>& cameras = collector.cameras();
  const std::vector<CLight>& lights = collector.lights();

  // Body / geom name -> id maps for targetbody, pair, and exclude resolution.
  std::unordered_map<std::string, int> bodyid_of, geomid_of;
  for (int i = 0; i < static_cast<int>(cbs.size()); ++i) {
    const std::string nm =
        i == 0 ? std::string("world")
               : (cbs[i].src && cbs[i].src->name ? *cbs[i].src->name : std::string());
    if (!nm.empty()) bodyid_of[nm] = i;
  }
  for (int i = 0; i < static_cast<int>(geoms.size()); ++i)
    if (geoms[i].src && geoms[i].src->name) geomid_of[*geoms[i].src->name] = i;

  // Site / joint name -> id maps for equality (and later actuator/sensor) ref
  // resolution. Authored names only (referenceable targets are always named).
  std::unordered_map<std::string, int> siteid_of, jointid_of, tendonid_of;
  for (int i = 0; i < static_cast<int>(sites.size()); ++i)
    if (sites[i].src && sites[i].src->name) siteid_of[*sites[i].src->name] = i;
  for (int i = 0; i < static_cast<int>(joints.size()); ++i)
    if (!joints[i].name.empty()) jointid_of[joints[i].name] = i;

  // Contact pairs + excludes (S8): compile, stable-sort by body signature,
  // reassign ids (their position after sorting is the id).
  std::vector<CPair> pairs;
  std::vector<CExclude> excludes;
  for (const auto& ct : m.contacts) {
    if (!ct) continue;
    for (const auto& p : ct->pairs)
      if (p) pairs.push_back(PairCompile(m, *p, geomid_of, geoms));
    for (const auto& e : ct->excludes)
      if (e) excludes.push_back(ExcludeCompile(*e, bodyid_of));
  }
  std::stable_sort(pairs.begin(), pairs.end(),
                   [](const CPair& a, const CPair& b) {
                     return a.signature < b.signature;
                   });
  std::stable_sort(excludes.begin(), excludes.end(),
                   [](const CExclude& a, const CExclude& b) {
                     return a.signature < b.signature;
                   });

  ps::sdk::detail::DefaultIndex default_idx(m);

  // Tendons (S8): document order across all <tendon> sections. Compiled before
  // equalities so equality-tendon refs (and later actuator tendon targets) can
  // resolve tendon ids by name.
  std::vector<CTendon> tendons;
  for (const auto& tn : m.tendons) {
    if (!tn) continue;
    for (const auto& any : tn->tendons)
      tendons.push_back(TendonCompile(default_idx, any, siteid_of, geomid_of,
                                      jointid_of, geoms));
  }
  for (int i = 0; i < static_cast<int>(tendons.size()); ++i)
    if (tendons[i].name) tendonid_of[*tendons[i].name] = i;

  // Equality constraints (S8): document order across all <equality> sections.
  std::vector<CEquality> equalities;
  for (const auto& eq : m.equalitys) {
    if (!eq) continue;
    for (const auto& any : eq->equalities)
      equalities.push_back(EqualityCompile(default_idx, any, bodyid_of,
                                           siteid_of, jointid_of, tendonid_of));
  }

  // Actuators (S8): document order across all <actuator> sections. Transmission
  // targets resolve against joint/tendon/body id maps built above.
  RangeLookup rlook{&joints, &tendons, &jointid_of, &tendonid_of};
  std::vector<CActuator> actuators;
  for (const auto& ac : m.actuators) {
    if (!ac) continue;
    for (const auto& any : ac->actuators)
      actuators.push_back(ActuatorCompile(m, any, jointid_of, tendonid_of,
                                          bodyid_of, rlook));
  }

  // Actuator / camera name -> id maps for sensor targets.
  std::unordered_map<std::string, int> actuatorid_of, cameraid_of;
  for (int i = 0; i < static_cast<int>(actuators.size()); ++i)
    if (actuators[i].name) actuatorid_of[*actuators[i].name] = i;
  for (int i = 0; i < static_cast<int>(cameras.size()); ++i)
    if (cameras[i].src && cameras[i].src->name) cameraid_of[*cameras[i].src->name] = i;

  // Sensors (S8): document order across all <sensor> sections.
  SensorMaps smaps{&bodyid_of, &geomid_of, &siteid_of, &cameraid_of,
                   &jointid_of, &tendonid_of, &actuatorid_of};
  std::vector<CSensor> sensors;
  for (const auto& sn : m.sensors) {
    if (!sn) continue;
    for (const auto& any : sn->sensors)
      sensors.push_back(SensorCompile(m, any, smaps));
  }

  // Keyframes (S12): flattened key list, document order.
  std::vector<const Key*> keys;
  for (const auto& kf : m.keyframes) {
    if (!kf) continue;
    for (const auto& k : kf->keys)
      if (k) keys.push_back(k.get());
  }

  // Assets (S8/CopyObjects). Order: textures -> materials -> meshes/hfields, so
  // materials resolve texture ids by name and geoms/sites/tendons resolve
  // material ids by name. Meshes/hfields land in later NC3 waves; those maps stay
  // empty until then (the gate blocks references to uncompiled assets).
  std::unordered_map<std::string, int> texid_of, matid_of;

  std::vector<CTexture> textures;
  for (const auto& as : m.assets) {
    if (!as) continue;
    for (const auto& tx : as->textures) {
      if (!tx) continue;
      CTexture ct;
      if (!TextureCompile(m, *tx, ct, diags)) return nullptr;
      textures.push_back(std::move(ct));
    }
  }
  for (int i = 0; i < static_cast<int>(textures.size()); ++i)
    if (textures[i].name) texid_of[*textures[i].name] = i;

  std::vector<CMaterial> materials;
  for (const auto& as : m.assets) {
    if (!as) continue;
    for (const auto& mat : as->materials)
      if (mat) materials.push_back(MaterialCompile(m, *mat, texid_of));
  }
  for (int i = 0; i < static_cast<int>(materials.size()); ++i)
    if (materials[i].name) matid_of[*materials[i].name] = i;

  // Resolve material refs to ids (geoms/sites/tendons; CopyObjects/IndexAssets).
  auto resolve_mat = [&](const std::string& nm) -> int {
    if (nm.empty()) return -1;
    auto it = matid_of.find(nm);
    return it == matid_of.end() ? -1 : it->second;
  };
  for (CGeom& cg : geoms) cg.matid = resolve_mat(cg.material_name);
  for (CSite& cs2 : sites) cs2.matid = resolve_mat(cs2.material_name);
  for (CTendon& t : tendons) t.matid = resolve_mat(t.material_name);

  // S9 sizes: nq/nv from joints.
  int nq = 0, nv = 0;
  for (const CJoint& cj : joints) { nq += cj.nq(); nv += cj.nv(); }

  // nuser_* (SetNuser): the <size nuser_X> override, else max authored user
  // length over the family (auto).
  int nuser_body = -1, nuser_jnt = -1, nuser_geom = -1, nuser_site = -1,
      nuser_cam = -1, nuser_tendon = -1, nuser_actuator = -1, nuser_sensor = -1;
  for (const auto& sz : m.sizes) {
    if (!sz) continue;
    if (sz->nuser_body) nuser_body = *sz->nuser_body;
    if (sz->nuser_jnt) nuser_jnt = *sz->nuser_jnt;
    if (sz->nuser_geom) nuser_geom = *sz->nuser_geom;
    if (sz->nuser_site) nuser_site = *sz->nuser_site;
    if (sz->nuser_cam) nuser_cam = *sz->nuser_cam;
    if (sz->nuser_tendon) nuser_tendon = *sz->nuser_tendon;
    if (sz->nuser_actuator) nuser_actuator = *sz->nuser_actuator;
    if (sz->nuser_sensor) nuser_sensor = *sz->nuser_sensor;
  }
  auto maxlen = [](int cur, std::size_t n) {
    return cur > static_cast<int>(n) ? cur : static_cast<int>(n);
  };
  if (nuser_body == -1) { nuser_body = 0;
    for (const CBody& cb : cbs) nuser_body = maxlen(nuser_body, cb.user.size()); }
  if (nuser_jnt == -1) { nuser_jnt = 0;
    for (const CJoint& cj : joints) nuser_jnt = maxlen(nuser_jnt, cj.user.size()); }
  if (nuser_geom == -1) { nuser_geom = 0;
    for (const CGeom& cg : geoms) nuser_geom = maxlen(nuser_geom, cg.user.size()); }
  if (nuser_site == -1) { nuser_site = 0;
    for (const CSite& cs2 : sites) nuser_site = maxlen(nuser_site, cs2.user.size()); }
  if (nuser_cam == -1) { nuser_cam = 0;
    for (const CCamera& cc : cameras) nuser_cam = maxlen(nuser_cam, cc.user.size()); }
  if (nuser_tendon == -1) { nuser_tendon = 0;
    for (const CTendon& t : tendons) nuser_tendon = maxlen(nuser_tendon, t.user.size()); }
  if (nuser_actuator == -1) { nuser_actuator = 0;
    for (const CActuator& a : actuators) nuser_actuator = maxlen(nuser_actuator, a.user.size()); }
  if (nuser_sensor == -1) { nuser_sensor = 0;
    for (const CSensor& s : sensors) nuser_sensor = maxlen(nuser_sensor, s.user.size()); }

  // mocap ids (SetSizes/CopyTree): count mocap bodies, assign in body order.
  int nmocap = 0;
  std::vector<int> mocapid(cbs.size(), -1);
  for (int i = 0; i < static_cast<int>(cbs.size()); ++i)
    if (cbs[i].mocap) mocapid[i] = nmocap++;

  // Dof tree + sparse sizes (the crown-jewel pass).
  DofTree dt = ComputeDofTree(cbs, joints, nv);

  // S10 names: effective names (authored or XML-path auto-name). Body 0 is the
  // implicit world body, always named "world".
  NameLists nl;
  nl.modelname = m.model ? *m.model : std::string("MuJoCo Model");
  for (int i = 0; i < static_cast<int>(cbs.size()); ++i) {
    nl.body.push_back(i == 0 ? std::string("world")
                             : EffectiveName(*cbs[i].src, opts));
  }
  for (const CJoint& cj : joints) nl.jnt.push_back(cj.name);
  for (const CGeom& cg : geoms) nl.geom.push_back(EffectiveName(*cg.src, opts));
  for (const CSite& cs2 : sites) nl.site.push_back(EffectiveName(*cs2.src, opts));
  for (const CCamera& cc : cameras) nl.cam.push_back(EffectiveName(*cc.src, opts));
  for (const CLight& cl : lights) nl.light.push_back(EffectiveName(*cl.src, opts));
  // Assets (texture/material/mesh/hfield/skin) are NOT auto-named on the XML
  // path (bridge AutoNameableFamily), so leg C uses the authored name only --
  // an unnamed asset keeps an empty name, exactly like leg B.
  for (const CMesh& cm : meshes) nl.mesh.push_back(cm.listname);
  for (const CHField& hf : hfields)
    nl.hfield.push_back(hf.name ? *hf.name : std::string());
  for (const CTexture& ct : textures)
    nl.tex.push_back(ct.name ? *ct.name : std::string());
  for (const CMaterial& cm : materials)
    nl.mat.push_back(cm.name ? *cm.name : std::string());
  for (const CPair& cp : pairs) nl.pair.push_back(EffectiveName(*cp.src, opts));
  for (const CExclude& ce : excludes)
    nl.exclude.push_back(EffectiveName(*ce.src, opts));
  // Equality (all spellings share the auto-name token "eq").
  auto serial_name = [&](const ps::opt<std::string>& nm, std::uint64_t serial,
                         const char* tok) -> std::string {
    if (nm) return *nm;
    if (opts.auto_name)
      return opts.auto_name_prefix + tok + ":" + std::to_string(serial);
    return "";
  };
  for (const CEquality& ce : equalities)
    nl.eq.push_back(serial_name(ce.name, ce.serial, "eq"));
  for (const CTendon& t : tendons)
    nl.tendon.push_back(serial_name(t.name, t.serial, "tendon"));
  for (const CActuator& a : actuators)
    nl.actuator.push_back(serial_name(a.name, a.serial, "act"));
  for (const CSensor& s : sensors)
    nl.sensor.push_back(serial_name(s.name, s.serial, "sensor"));
  for (const Key* k : keys) nl.key.push_back(EffectiveName(*k, opts));

  // nbvh census (SetSizes): static BVH nodes over all bodies, then meshes. Mesh
  // face-BVH nodes are laid out after every body node (CopyObjects bvh_adr).
  int nbvh_body = 0;
  for (const CBody& cb : cbs) nbvh_body += static_cast<int>(cb.bvh_child.size() / 2);
  int nbvh_mesh = 0;
  for (const CMesh& cm : meshes) nbvh_mesh += cm.nbvh();
  int nbvhstatic = nbvh_body + nbvh_mesh;

  // S9 Sizes.
  mjModel sizes;
  std::memset(&sizes, 0, sizeof(sizes));
  sizes.nbody = static_cast<int>(cbs.size());
  sizes.njnt = static_cast<int>(joints.size());
  sizes.ngeom = static_cast<int>(geoms.size());
  sizes.nsite = static_cast<int>(sites.size());
  sizes.ncam = static_cast<int>(cameras.size());
  sizes.nlight = static_cast<int>(lights.size());
  sizes.ntex = static_cast<int>(textures.size());
  mjtSize ntexdata = 0;
  for (const CTexture& ct : textures) ntexdata += static_cast<mjtSize>(ct.data.size());
  sizes.ntexdata = ntexdata;
  sizes.nmat = static_cast<int>(materials.size());
  sizes.nmesh = static_cast<int>(meshes.size());
  int nmeshvert = 0, nmeshnormal = 0, nmeshtexcoord = 0, nmeshface = 0,
      nmeshgraph = 0, nmeshpoly = 0, nmeshpolyvert = 0, nmeshpolymap = 0;
  for (const CMesh& cm : meshes) {
    const lift::MeshResult& r = cm.r;
    nmeshvert += r.nvert();
    nmeshnormal += r.nnormal();
    nmeshtexcoord += r.has_texcoord() ? r.ntexcoord() : 0;
    nmeshface += r.nface();
    nmeshgraph += r.szgraph();
    nmeshpoly += r.npolygon();
    nmeshpolyvert += r.npolygonvert();
    nmeshpolymap += r.npolygonmap();
  }
  sizes.nmeshvert = nmeshvert;
  sizes.nmeshnormal = nmeshnormal;
  sizes.nmeshtexcoord = nmeshtexcoord;
  sizes.nmeshface = nmeshface;
  sizes.nmeshgraph = nmeshgraph;
  sizes.nmeshpoly = nmeshpoly;
  sizes.nmeshpolyvert = nmeshpolyvert;
  sizes.nmeshpolymap = nmeshpolymap;
  sizes.nhfield = static_cast<int>(hfields.size());
  int nhfielddata = 0;
  for (const CHField& hf : hfields) nhfielddata += hf.nrow * hf.ncol;
  sizes.nhfielddata = nhfielddata;
  sizes.npair = static_cast<int>(pairs.size());
  sizes.nexclude = static_cast<int>(excludes.size());
  sizes.neq = static_cast<int>(equalities.size());
  // nemax (SetSizes user_model.cc:2358): connect +3, weld +7, else +1.
  int nemax = 0;
  for (const CEquality& ce : equalities)
    nemax += ce.type == mjEQ_CONNECT ? 3 : ce.type == mjEQ_WELD ? 7 : 1;
  sizes.nemax = nemax;
  sizes.ntendon = static_cast<int>(tendons.size());
  int nwrap = 0;
  for (const CTendon& t : tendons) nwrap += static_cast<int>(t.path.size());
  sizes.nwrap = nwrap;
  sizes.nJten = ComputeNJten(tendons, cbs, geoms, sites, nv);
  sizes.nuser_tendon = nuser_tendon;
  sizes.nu = static_cast<int>(actuators.size());
  int na = 0;
  for (const CActuator& a : actuators) na += a.actdim;
  sizes.na = na;
  sizes.nJmom = ComputeNJmom(actuators, joints, tendons, cbs, geoms, sites, nv);
  sizes.nuser_actuator = nuser_actuator;
  sizes.nsensor = static_cast<int>(sensors.size());
  int nsensordata = 0;
  for (const CSensor& s : sensors) nsensordata += s.dim;
  sizes.nsensordata = nsensordata;
  sizes.nuser_sensor = nuser_sensor;
  sizes.nkey = static_cast<int>(keys.size());
  sizes.nbvh = nbvhstatic;
  sizes.nbvhstatic = nbvhstatic;
  sizes.nq = nq;
  sizes.nv = nv;
  sizes.ntree = dt.ntree;
  sizes.nmocap = nmocap;
  sizes.nM = dt.nM;
  sizes.nD = dt.nD;
  sizes.nB = dt.nB;
  sizes.nC = dt.nC;
  sizes.ngravcomp = 0;
  sizes.nuser_body = nuser_body;
  sizes.nuser_jnt = nuser_jnt;
  sizes.nuser_geom = nuser_geom;
  sizes.nuser_site = nuser_site;
  sizes.nuser_cam = nuser_cam;
  sizes.nnames = nl.TotalNames();
  // npaths: concatenated asset file paths + NUL each (getpathslength over
  // hfields/meshes/skins/textures); only file meshes contribute here. Defaults
  // to 1 when there are none (SetSizes).
  int npaths = 0;
  for (const CMesh& cm : meshes)
    if (!cm.file.empty()) npaths += static_cast<int>(cm.file.size()) + 1;
  sizes.npaths = npaths == 0 ? 1 : npaths;

  // S11 Allocate.
  mjModel* out = lift::MakeModel(sizes);
  if (!out) {
    diags.push_back({bridge::Diagnostic::Severity::Error, "alloc",
                     "native: mj_makeModel lift failed", {}});
    return nullptr;
  }

  // Default constraint-size fields (mjCModel ctor -> m at user_model.cc:3335).
  // Unset in NC1 scope (no <size njmax/nconmax>); their -1 drives the narena
  // heuristic's 500/100 fallbacks.
  out->nconmax = -1;
  out->njmax = -1;

  // S11 Fill.
  FillNames(out, nl);
  FillTextures(out, textures);
  for (int i = 0; i < out->ntex; ++i) out->tex_pathadr[i] = -1;  // builtin: no path
  FillMaterials(out, materials);
  FillHfields(out, hfields);
  FillMeshes(out, meshes, nbvh_body);
  FillMeshPaths(out, meshes);
  FillTree(out, cbs, geoms, joints, dt);
  for (int i = 0; i < static_cast<int>(cbs.size()); ++i)
    out->body_mocapid[i] = mocapid[i];
  FillVisual(out, sites, cameras, lights, bodyid_of);
  FillPairs(out, pairs);
  FillExcludes(out, excludes);
  FillEqualities(out, equalities);
  FillTendons(out, tendons);
  FillActuators(out, actuators);
  FillSensors(out, sensors);
  // Keyframe padding uses qpos0 / body_pos / body_quat / body_mocapid, all set
  // by FillTree above (mjCKey::Compile).
  FillKeyframes(out, keys);

  // Arena size heuristic (ledger 3.7, user_model.cc:5219-5252). memory/nstack
  // are unset (-1) in NC1 scope (no <size memory/nstack>); njmax/nconmax default
  // to -1 -> 500/100. Lifted verbatim.
  SetNarena(out);

  // Sparsity structures (D/B/M rows + M<->D maps), public MJAPI builders called
  // exactly as TryCompile does (user_model.cc:5254-5280).
  if (nv > 0) {
    std::vector<int> scratch(out->nv);
    std::vector<int> count(out->nbody);
    std::vector<int> Mtmp(out->nM);
    mj_makeDofDofSparse(out->nv, out->nC, out->nD, out->nM, out->dof_parentid,
                        out->dof_simplenum, out->D_rownnz, out->D_rowadr,
                        out->D_diag, out->D_colind, 0, 1, scratch.data());
    mj_makeBSparse(out->nv, out->nbody, out->nB, out->body_dofnum,
                   out->body_parentid, out->body_dofadr, out->B_rownnz,
                   out->B_rowadr, out->B_colind, count.data());
    mj_makeDofDofSparse(out->nv, out->nC, out->nD, out->nM, out->dof_parentid,
                        out->dof_simplenum, out->M_rownnz, out->M_rowadr, nullptr,
                        out->M_colind, 1, 0, scratch.data());
    mj_makeDofDofMaps(out->nv, out->nM, out->nC, out->nD, out->dof_Madr,
                      out->dof_simplenum, out->dof_parentid, out->D_rownnz,
                      out->D_rowadr, out->D_colind, out->M_rownnz, out->M_rowadr,
                      out->M_colind, out->mapM2D, out->mapD2M, out->mapM2M,
                      Mtmp.data(), scratch.data());
  }

  // S12 Finalize: derived constants, statistics, sparsity. mj_setConst needs a
  // scratch mjData (public post-build pass; ledger 3.8).
  mjData* d = mj_makeData(out);
  if (!d) {
    diags.push_back({bridge::Diagnostic::Severity::Error, "finalize",
                     "native: mj_makeData failed during finalize", {}});
    mj_deleteModel(out);
    return nullptr;
  }
  mj_setConst(out, d);
  mj_deleteData(d);

  return out;
}

}  // namespace ps::mjcf::compile
