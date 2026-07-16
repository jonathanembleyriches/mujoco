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
// (ids: geom_set_inertia, geom_compute_aabb, geom_get_rbound,
// inertia_from_geom, joint_compile, checklimited, bvh_makebvh, body_compile,
// compute_sparse_sizes, set_sizes, copy_tree, copy_names, finalize_simple,
// hash_string; flexcomp expansion: flexcomp_make, flexcomp_makegrid,
// flexcomp_makesquare, flexcomp_makebox, flexcomp_boxproject, flexcomp_boxid,
// flexcomp_makemesh, flexcomp_loadgmsh, flexcomp_loadgmsh41, flexcomp_loadgmsh22);
// their sources are user_objects.cc / user_model.cc / user_flexcomp.cc /
// engine_name.c. Per the reuse ledger these are class-C passes: the algorithm,
// constants, and iteration order are upstream; the data plumbing is retargeted
// to ProtoSpec compiled structs and the mjModel arrays.

#include "build.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <mujoco/mujoco.h>

#include "binding.h"       // ps::mjcf::ObjTypeOf / FamilyToken
#include "classes.h"       // ps::sdk::Effective (pure effective-defaults query)
#include "context.h"
#include "builtin_mesh.h"  // lifted procedural mesh generators
#include "image_pipeline.h" // lifted PNG decode
#include "make_model.h"    // lifted::MakeModel
#include "marching_cube.h" // vendored MarchingCubeCpp (mjCMesh::LoadSDF)
#include "mesh_pipeline.h" // lifted mesh compile
#include "mjuu_util.h"     // lifted math
#include "reflect.h"       // reflection tables
#include "resolve.h"       // core::ResolveOrientation (canonical orientation)
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
  std::string texturedir;       // <compiler texturedir> (falls back to assetdir)
  bool strippath = false;       // <compiler strippath>
  bool fitaabb = false;         // <compiler fitaabb> (fitgeom: aabb vs inertia box)
  bool alignfree = false;       // <compiler alignfree> (free-joint frame alignment)
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
    if (c->assetdir) { s.meshdir = *c->assetdir; s.texturedir = *c->assetdir; }
    if (c->meshdir) s.meshdir = *c->meshdir;
    if (c->texturedir) s.texturedir = *c->texturedir;
    if (c->strippath) s.strippath = *c->strippath;
    if (c->fitaabb) s.fitaabb = *c->fitaabb;
    if (c->alignfree) s.alignfree = *c->alignfree;
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
  double xpos0[3] = {0, 0, 0};     // global body pose at qpos0 (no joint xform)
  double xquat0[4] = {1, 0, 0, 0}; // mjCBody::Compile (user_objects.cc:2844-2849)
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
  double springdamper[2] = {0, 0};   // [timeconst, dampratio]; AutoSpringDamper
  bool limited = false, actfrclimited = false, actgravcomp = false;

  int nq() const {
    return type == mjJNT_FREE ? 7 : type == mjJNT_BALL ? 4 : 1;
  }
  int nv() const {
    return type == mjJNT_FREE ? 6 : type == mjJNT_BALL ? 3 : 1;
  }
};

// --------------------------------------------------------------------------- //
// Orientation copy (Q-ORIENT). Orientation is canonicalized to a unit quaternion
// at read (cpp/core/resolve.cc, the same lifted ResolveOrientation math), so the
// native compiler consumes the pre-resolved `quat` field and only renormalizes.
// The degree/eulerseq fold no longer lives here; ReplicateEulerQuat (below) is
// the one native site that still folds euler, via core::ResolveOrientation.
// --------------------------------------------------------------------------- //
void ResolveQuat(const ps::opt<std::array<double, 4>>& quat, double out[4]) {
  if (quat) {
    out[0] = (*quat)[0]; out[1] = (*quat)[1];
    out[2] = (*quat)[2]; out[3] = (*quat)[3];
    lift::mjuu_normvec(out, 4);
  } else {
    out[0] = 1; out[1] = out[2] = out[3] = 0;
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
  (void)cs;
  ResolveQuat(f.quat, out.quat);
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
  (void)cs;
  // Q-INERTIA: inertia is canonicalized at read to diaginertia + iquat (the
  // reader eigendecomposes fullinertia into both), so the native compiler copies
  // the principal moments and the pre-resolved inertial-frame quaternion.
  ResolveQuat(in.iquat, cb.iquat);
  if (in.diaginertia) {
    cb.inertia[0] = (*in.diaginertia)[0];
    cb.inertia[1] = (*in.diaginertia)[1];
    cb.inertia[2] = (*in.diaginertia)[2];
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

// GetAddedMassKappa (user_objects.cc:3759): 15-point Gauss-Kronrod quadrature of
// the ellipsoid added-mass integral. Tables + change of variables lifted verbatim.
double GeomAddedMassKappa(double dx, double dy, double dz) {
  static constexpr double kronrod_w[15] = {
    0.01146766, 0.03154605, 0.05239501, 0.07032663, 0.08450236,
    0.09517529, 0.10221647, 0.10474107, 0.10221647, 0.09517529,
    0.08450236, 0.07032663, 0.05239501, 0.03154605, 0.01146766};
  static constexpr double kronrod_l[15] = {
    7.865151709349917e-08, 1.7347976913907274e-05, 0.0003548008144506193,
    0.002846636252924549, 0.014094260903596077, 0.053063261727396636,
    0.17041978741317773, 0.5, 1.4036301548686991, 3.9353484827022642,
    11.644841677041734, 39.53187807410903, 177.5711362220801,
    1429.4772912937397, 54087.416549217705};
  static constexpr double kronrod_d[15] = {
    5.538677720489877e-05, 0.002080868285293228, 0.016514126520723166,
    0.07261900344370877, 0.23985243401862602, 0.6868318249020725,
    1.8551129519182894, 5.0, 14.060031152313941, 43.28941239611009,
    156.58546376397112, 747.9826085305024, 5827.4042950027115,
    116754.0197944512, 25482945.327264845};
  const double invdx2 = 1.0 / (dx * dx);
  const double invdy2 = 1.0 / (dy * dy);
  const double invdz2 = 1.0 / (dz * dz);
  const double scale = std::pow(dx * dx * dx * dy * dz, 0.4);
  double kappa = 0.0;
  for (int i = 0; i < 15; ++i) {
    const double lambda = scale * kronrod_l[i];
    const double denom = (1 + lambda * invdx2) * std::sqrt(
        (1 + lambda * invdx2) * (1 + lambda * invdy2) * (1 + lambda * invdz2));
    kappa += scale * kronrod_d[i] / denom * kronrod_w[i];
  }
  return kappa * invdx2;
}

// SetFluidCoefs (user_objects.cc:3809) + writeFluidGeomInteraction (engine_
// passive.c:1346): fill the 12-entry geom_fluid array for an ellipsoid-model
// fluid geom from its semiaxes and the five user drag/lift coefficients.
void GeomFluidCoefs(int type, const double size[3], const double coefs[5],
                    double fluid[mjNFLUID]) {
  double dx, dy, dz;
  switch (type) {
    case mjGEOM_SPHERE: dx = dy = dz = size[0]; break;
    case mjGEOM_CAPSULE: dx = size[0]; dy = size[0]; dz = size[1] + size[0]; break;
    case mjGEOM_CYLINDER: dx = size[0]; dy = size[0]; dz = size[1]; break;
    default: dx = size[0]; dy = size[1]; dz = size[2];
  }
  const double volume = 4.0 / 3.0 * mjPI * dx * dy * dz;
  const double kx = GeomAddedMassKappa(dx, dy, dz);
  const double ky = GeomAddedMassKappa(dy, dz, dx);
  const double kz = GeomAddedMassKappa(dz, dx, dy);
  const auto pow2 = [](double v) { return v * v; };
  const double Ixfac = pow2(dy * dy - dz * dz) * std::abs(kz - ky) / std::max(
      lift::mjEPS, std::abs(2 * (dy * dy - dz * dz) + (dy * dy + dz * dz) * (ky - kz)));
  const double Iyfac = pow2(dz * dz - dx * dx) * std::abs(kx - kz) / std::max(
      lift::mjEPS, std::abs(2 * (dz * dz - dx * dx) + (dz * dz + dx * dx) * (kz - kx)));
  const double Izfac = pow2(dx * dx - dy * dy) * std::abs(ky - kx) / std::max(
      lift::mjEPS, std::abs(2 * (dx * dx - dy * dy) + (dx * dx + dy * dy) * (kx - ky)));
  double vmass[3] = {volume * kx / std::max(lift::mjEPS, 2 - kx),
                     volume * ky / std::max(lift::mjEPS, 2 - ky),
                     volume * kz / std::max(lift::mjEPS, 2 - kz)};
  double vinertia[3] = {volume * Ixfac / 5, volume * Iyfac / 5, volume * Izfac / 5};
  fluid[0] = 1.0;  // fluid_ellipsoid (ellipsoid model active)
  for (int k = 0; k < 5; ++k) fluid[1 + k] = coefs[k];
  for (int k = 0; k < 3; ++k) { fluid[6 + k] = vmass[k]; fluid[9 + k] = vinertia[k]; }
}

// Eager per-slot `size`: MuJoCo pre-fills the 3-vector from the governing
// default chain, then ReadAttr overwrites only the element's authored prefix
// (user_objects.cc: the mjCGeom/mjCSite default carries a full size[3]). So a
// geom/site authoring FEWER size values than a class-chain default provides
// inherits that default's tail -- which ProtoSpec's whole-field presence merge
// (Effective takes the first-present WHOLE vector) does not reproduce. Rebuild
// the eager result per slot: the highest-priority level (element, then leaf
// class up to the root blocks, high rank first) that authors slot j wins it.
// Only needed when the element authored a partial (1-2 value) size; the full /
// unauthored cases already agree with Effective, so the hot path is untouched.
// `out` enters holding the element-type constructor default (mjCGeom {0,0,0},
// mjCSite {0.005,0.005,0.005}) -- the eager base MuJoCo pre-fills before any
// overwrite. Slots no level authors keep that default; a slot any level authors
// takes the highest-priority authored value.
template <class T>
void EagerSizeArray(const Model& m, const T& e, double out[3]) {
  bool set[3] = {false, false, false};
  auto overlay = [&](const auto& sz) {
    if (!sz) return;
    for (std::size_t k = 0; k < sz->size() && k < 3; ++k)
      if (!set[k]) { out[k] = (*sz)[k]; set[k] = true; }
  };
  auto node_size = [&](const ps::mjcf::Default& d) {
    const auto* vec = ps::sdk::DefaultVec<T>(d);
    if (vec && !vec->empty() && vec->front()) overlay(vec->front()->size);
  };
  overlay(e.size);  // element authored prefix is highest priority
  ps::sdk::ParentMap pm(m);
  ps::sdk::detail::DefaultIndex idx(m);
  const std::string cls =
      ps::sdk::detail::ResolveClassName(pm, ps::sdk::detail::OwnClass(e), &e);
  for (const ps::mjcf::Default* d = idx.ByNameOrRoot(cls); d;
       d = idx.ParentOf(d)) {
    if (idx.ParentOf(d) == nullptr) {  // terminal: top-level block(s) -> `main`
      const int rank = idx.RootRank(d);
      if (rank < 0) {
        node_size(*d);
      } else {
        const auto& roots = idx.Roots();
        for (int i = rank; i >= 0; --i)
          if (roots[i]) node_size(*roots[i]);
      }
      break;
    }
    node_size(*d);
  }
}

// True when the element authored a partial size (1-2 values), the only case
// EagerSizeArray must correct.
template <class T>
bool AuthoredPartialSize(const T& e) {
  const int n = e.size ? static_cast<int>(e.size->size()) : -1;
  return n == 1 || n == 2;
}

// Compile one geom (mjCGeom::Compile, primitive path). `cs` supplies degree;
// `assets` binds hfield/mesh geoms (assets compiled before the body walk).
CGeom GeomCompile(const Model& model, const Geom& g, const CompilerSettings& cs,
                  int bodyid, bool inferinertia, const AssetBinds& assets) {
  std::unique_ptr<Geom> eff = ps::sdk::Effective(model, g);
  // orientation is pre-resolved (Q-ORIENT); cs is used for the fitgeom branch.
  CGeom cg;
  cg.src = &g;
  cg.bodyid = bodyid;
  cg.type = eff->type ? static_cast<int>(*eff->type) : mjGEOM_SPHERE;
  if (AuthoredPartialSize(g))
    EagerSizeArray(model, g, cg.size);
  else if (eff->size)
    for (std::size_t k = 0; k < eff->size->size() && k < 3; ++k)
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
  // Ellipsoid fluid model (mjCGeom::Compile: SetFluidCoefs when fluid_ellipsoid>0),
  // computed from the FINAL size/type just before returning cg. mjs_defaultGeom's
  // drag/lift coefficients {0.5,0.25,1.5,1.0,1.0} apply when fluidcoef is unset.
  const bool fluid_on = eff->fluidshape &&
                        *eff->fluidshape == FluidShape::ellipsoid;
  double fluidcoef[5] = {0.5, 0.25, 1.5, 1.0, 1.0};
  if (eff->fluidcoef)
    for (std::size_t k = 0; k < eff->fluidcoef->size() && k < 5; ++k)
      fluidcoef[k] = (*eff->fluidcoef)[k];
  auto finalize_fluid = [&]() {
    if (fluid_on) GeomFluidCoefs(cg.type, cg.size, fluidcoef, cg.fluid);
  };
  cg.has_mass = eff->mass.has_value();
  double mass = cg.has_mass ? *eff->mass : 0;
  const int ti = (eff->shellinertia && *eff->shellinertia) ? mjINERTIA_SHELL
                                                           : mjINERTIA_VOLUME;

  // normalize quaternion / resolve orientation
  ResolveQuat(eff->quat, cg.quat);
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
      finalize_fluid();
      return cg;
    }
  }

  // fitgeom: a primitive geom (type != mesh/sdf) that references a mesh takes its
  // size from the mesh's inertia box (default) or aabb (compiler fitaabb), scaled
  // by fitscale, then accumulates the mesh frame + fit center into its own frame
  // and proceeds as a primitive (mjCGeom::Compile mesh branch + mjCMesh::FitGeom,
  // user_mesh.cc:944). The mesh ref is dropped, so no dataid/mesh inertia here.
  if (cg.type != mjGEOM_MESH && cg.type != mjGEOM_SDF && eff->mesh) {
    auto it = assets.mesh.find(eff->mesh->name);
    if (it != assets.mesh.end()) {
      const MeshBind& mb = it->second;
      // geom_dataid keeps the source mesh id even after fitting (MuJoCo retains
      // the reference for visualization; only size/frame come from the fit).
      cg.dataid = mb.id;
      const double fitscale = eff->fitscale ? *eff->fitscale : 1.0;
      double center[3] = {0, 0, 0};
      if (!cs.fitaabb) {  // inertia box
        switch (cg.type) {
          case mjGEOM_SPHERE:
            cg.size[0] = (mb.boxsz[0] + mb.boxsz[1] + mb.boxsz[2]) / 3;
            break;
          case mjGEOM_CAPSULE:
            cg.size[0] = (mb.boxsz[0] + mb.boxsz[1]) / 2;
            cg.size[1] = std::max(0.0, mb.boxsz[2] - cg.size[0] / 2);
            break;
          case mjGEOM_CYLINDER:
            cg.size[0] = (mb.boxsz[0] + mb.boxsz[1]) / 2;
            cg.size[1] = mb.boxsz[2];
            break;
          case mjGEOM_ELLIPSOID:
          case mjGEOM_BOX:
            cg.size[0] = mb.boxsz[0];
            cg.size[1] = mb.boxsz[1];
            cg.size[2] = mb.boxsz[2];
            break;
          default:
            break;
        }
      } else {  // aabb
        for (int k = 0; k < 3; ++k) center[k] = (mb.aamm[k] + mb.aamm[k + 3]) / 2;
        double sz[3] = {mb.aamm[3] - center[0], mb.aamm[4] - center[1],
                        mb.aamm[5] - center[2]};
        switch (cg.type) {
          case mjGEOM_SPHERE:
            cg.size[0] = std::max(std::max(sz[0], sz[1]), sz[2]);
            break;
          case mjGEOM_CAPSULE:
          case mjGEOM_CYLINDER:
            cg.size[0] = std::max(sz[0], sz[1]);
            cg.size[1] = sz[2];
            if (cg.type == mjGEOM_CAPSULE) cg.size[1] -= cg.size[0];
            break;
          case mjGEOM_ELLIPSOID:
          case mjGEOM_BOX:
            cg.size[0] = sz[0];
            cg.size[1] = sz[1];
            cg.size[2] = sz[2];
            break;
          default:
            break;
        }
      }
      cg.size[0] *= fitscale;
      cg.size[1] *= fitscale;
      cg.size[2] *= fitscale;
      // rotate the fit center to the geom frame, add the mesh pos, then
      // accumulate the mesh frame into the geom frame.
      double meshpos[3];
      lift::mjuu_rotVecQuat(meshpos, center, mb.quat);
      lift::mjuu_addtovec(meshpos, mb.pos, 3);
      lift::mjuu_frameaccum(cg.pos, cg.quat, meshpos, mb.quat);
      // fall through to the primitive path (size fitted, mesh ref dropped).
    }
  }

  // mesh geom: accumulate the mesh frame, then take size/aabb/rbound from the
  // mesh aamm and inertia from the mesh volume/box (mjCGeom::Compile mesh branch
  // + GetVolume/SetInertia/GetRBound/ComputeAABB mesh cases). An SDF geom binds
  // its plugin-generated mesh identically (upstream treats mjGEOM_SDF as mjGEOM_MESH
  // for bounds/inertia, user_objects.cc:3417/3496/3743/3902).
  if (cg.type == mjGEOM_MESH || cg.type == mjGEOM_SDF) {
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
      finalize_fluid();
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
  finalize_fluid();
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
  if (eff->springdamper) {
    cj.springdamper[0] = (*eff->springdamper)[0];
    cj.springdamper[1] = (*eff->springdamper)[1];
  }
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
  (void)cs;  // orientation is pre-resolved (Q-ORIENT)
  CSite cs2;
  cs2.src = &s;
  cs2.bodyid = bodyid;
  cs2.type = eff->type ? static_cast<int>(*eff->type) : mjGEOM_SPHERE;
  if (eff->group) cs2.group = *eff->group;
  if (AuthoredPartialSize(s))
    EagerSizeArray(model, s, cs2.size);
  else if (eff->size)
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
    ResolveQuat(eff->quat, cs2.quat);
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
  ResolveQuat(eff->quat, cc.quat);
  if (xf.present) lift::mjuu_frameaccumChild(xf.pos, xf.quat, cc.pos, cc.quat);
  lift::mjuu_normvec(cc.quat, 4);

  // output bit flags (default RGB); intrinsics (fovy / focal / principal).
  if (eff->output && !eff->output->empty()) {
    cc.output = 0;
    for (CameraOutput o : *eff->output) cc.output |= (1 << static_cast<int>(o));
  }
  // R1: fovy and focal are plain KEEP-semantics fields (the CameraIntrinsics
  // variant was dissolved), each set independently as authored.
  float focal_length[2] = {0, 0};  // mjsCamera.focal_length is float
  if (eff->fovy) cc.fovy = *eff->fovy;
  if (eff->focal) {
    focal_length[0] = static_cast<float>((*eff->focal)[0]);
    focal_length[1] = static_cast<float>((*eff->focal)[1]);
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
  std::string texture_name;
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
  // type is canonical (the legacy `directional` bool is folded to it at read).
  if (eff->type) cl.type = static_cast<int>(*eff->type);
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
  if (eff->texture) cl.texture_name = eff->texture->name;

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

// --------------------------------------------------------------------------- //
// Native <replicate> expansion (NC4), a pure ProtoSpec tree-clone reproducing  //
// the XML reader's mjs_attach loop (xml_native_reader.cc:3806-3874): the        //
// replicate subtree is cloned `count` times through an accumulating offset/euler //
// pose (mjuu_frameaccum) with a zero-padded name suffix (sep). Nested replicates //
// expand inner-first so names compose base+inner+outer in document order. Name   //
// suffixing follows mjs_attach with prefix="" and same-model semantics: every    //
// non-empty element name plus camera/light targetbody and light texture gets the //
// suffix (mjCBase/mjCCamera/mjCLight::NameSpace); class and material/mesh/hfield  //
// refs are left alone (their `model != m` guard). The clones live in a compile-  //
// owned arena so the flatten's element pointers stay valid; the source tree is   //
// never mutated (CDR-14 purity).                                                 //
// --------------------------------------------------------------------------- //
using RepArena = std::vector<std::unique_ptr<std::vector<BodyChildAny>>>;

// Zero-padded instance suffix (UpdateString, xml_native_reader.cc:155).
std::string ReplicateSuffix(const std::string& sep, int count, int i) {
  const int ndigits = static_cast<int>(std::to_string(count).size());
  const std::string is = std::to_string(i);
  std::string pad(std::max(0, ndigits - static_cast<int>(is.size())), '0');
  return sep + pad + is;
}

// scale*euler -> quat. Replicate's per-copy euler is #28 KEEP (a family
// parameterization, not a redundant encoding): copy i re-resolves i*euler. The
// fold uses the shared core resolver over the effective compiler context.
void ReplicateEulerQuat(const ps::opt<std::array<double, 3>>& euler, double scale,
                        const CompilerSettings& cs, double q[4]) {
  double raw[3] = {0, 0, 0};
  if (euler) for (int k = 0; k < 3; ++k) raw[k] = (*euler)[k] * scale;
  ps::core::OrientContext ctx;
  ctx.degree = cs.degree;
  ctx.eulerseq = cs.eulerseq;
  std::array<double, 4> out =
      ps::core::ResolveOrientation(ps::core::OrientKind::Euler, raw, ctx);
  q[0] = out[0]; q[1] = out[1]; q[2] = out[2]; q[3] = out[3];
}

BodyChildAny CloneBodyChild(const BodyChildAny& c);

void NamespaceSubtree(std::vector<BodyChildAny>& subtree,
                      const std::string& suffix) {
  auto add_name = [&](ps::opt<std::string>& nm) {
    if (nm && !nm->empty()) *nm += suffix;
  };
  auto add_ref = [&](auto& ref) {
    if (ref && !ref->name.empty()) ref->name += suffix;
  };
  for (BodyChildAny& c : subtree) {
    switch (c.kind()) {
      case BodyChildAny::Kind::Geom:
        add_name(std::get<std::unique_ptr<Geom>>(c.node)->name); break;
      case BodyChildAny::Kind::Joint:
        add_name(std::get<std::unique_ptr<Joint>>(c.node)->name); break;
      case BodyChildAny::Kind::FreeJoint:
        add_name(std::get<std::unique_ptr<FreeJoint>>(c.node)->name); break;
      case BodyChildAny::Kind::Site:
        add_name(std::get<std::unique_ptr<Site>>(c.node)->name); break;
      case BodyChildAny::Kind::Camera: {
        auto& cam = std::get<std::unique_ptr<Camera>>(c.node);
        add_name(cam->name); add_ref(cam->target); break;
      }
      case BodyChildAny::Kind::Light: {
        auto& l = std::get<std::unique_ptr<Light>>(c.node);
        add_name(l->name); add_ref(l->target); add_ref(l->texture); break;
      }
      case BodyChildAny::Kind::Body: {
        auto& b = std::get<std::unique_ptr<Body>>(c.node);
        add_name(b->name); NamespaceSubtree(b->subtree, suffix); break;
      }
      case BodyChildAny::Kind::Frame: {
        auto& f = std::get<std::unique_ptr<Frame>>(c.node);
        add_name(f->name); NamespaceSubtree(f->subtree, suffix); break;
      }
      default:
        break;  // PluginRef/Composite/Flexcomp/Attach: gated out of native
    }
  }
}

// Base auto-name a nameable element would receive on the XML path (mirrors
// EffectiveName / the bridge Collector): authored name, else the injected
// _ps:<family>:<serial> using the ORIGINAL element's serial. mjs_attach runs
// AFTER the XML path auto-names, so the base is fixed by the source element's
// serial and the replicate suffix is appended to it -- reproduced here by baking
// the base name into the clone before suffixing (the clone's own serial differs).
template <class E>
std::string ReplicateBaseName(const E& e, const ps::mjcf::CompileOptions& opts) {
  if (e.name) return *e.name;
  if (opts.auto_name)
    return opts.auto_name_prefix +
           std::string(ps::mjcf::FamilyToken(element_type_of<E>::value)) + ":" +
           std::to_string(e.serial);
  return "";
}

// Bake the base auto-name (from the ORIGINAL element `orig`) onto the clone, so a
// later suffix pass names it base+suffix exactly as the XML path would.
void BakeName(const BodyChildAny& orig, BodyChildAny& clone,
              const ps::mjcf::CompileOptions& opts) {
  auto set = [](ps::opt<std::string>& nm, std::string v) {
    if (!v.empty()) nm = std::move(v);
  };
  switch (orig.kind()) {
    case BodyChildAny::Kind::Geom:
      set(std::get<std::unique_ptr<Geom>>(clone.node)->name,
          ReplicateBaseName(*std::get<std::unique_ptr<Geom>>(orig.node), opts));
      break;
    case BodyChildAny::Kind::Joint:
      set(std::get<std::unique_ptr<Joint>>(clone.node)->name,
          ReplicateBaseName(*std::get<std::unique_ptr<Joint>>(orig.node), opts));
      break;
    case BodyChildAny::Kind::FreeJoint:
      set(std::get<std::unique_ptr<FreeJoint>>(clone.node)->name,
          ReplicateBaseName(*std::get<std::unique_ptr<FreeJoint>>(orig.node), opts));
      break;
    case BodyChildAny::Kind::Site:
      set(std::get<std::unique_ptr<Site>>(clone.node)->name,
          ReplicateBaseName(*std::get<std::unique_ptr<Site>>(orig.node), opts));
      break;
    case BodyChildAny::Kind::Camera:
      set(std::get<std::unique_ptr<Camera>>(clone.node)->name,
          ReplicateBaseName(*std::get<std::unique_ptr<Camera>>(orig.node), opts));
      break;
    case BodyChildAny::Kind::Light:
      set(std::get<std::unique_ptr<Light>>(clone.node)->name,
          ReplicateBaseName(*std::get<std::unique_ptr<Light>>(orig.node), opts));
      break;
    case BodyChildAny::Kind::Body:
      set(std::get<std::unique_ptr<Body>>(clone.node)->name,
          ReplicateBaseName(*std::get<std::unique_ptr<Body>>(orig.node), opts));
      break;
    default:
      break;  // Frame: not a named mjModel object; nested handled by recursion
  }
}

std::vector<BodyChildAny> ExpandTree(const std::vector<BodyChildAny>& subtree,
                                     const CompilerSettings& cs,
                                     const ps::mjcf::CompileOptions& opts);

// Expand one <replicate> into a replicate-free list of pose-carrying frames.
std::vector<BodyChildAny> ExpandReplicateNode(const Replicate& rep,
                                              const CompilerSettings& cs,
                                              const ps::mjcf::CompileOptions& opts) {
  std::vector<BodyChildAny> inner = ExpandTree(rep.subtree, cs, opts);
  double rot[4];
  ReplicateEulerQuat(rep.euler, 1.0, cs, rot);
  double offset[3] = {0, 0, 0};
  if (rep.offset) for (int k = 0; k < 3; ++k) offset[k] = (*rep.offset)[k];
  double pos[3] = {0, 0, 0}, quat[4] = {1, 0, 0, 0};
  const std::string sep = rep.sep ? *rep.sep : std::string();

  std::vector<BodyChildAny> out;
  for (int i = 0; i < rep.count; ++i) {
    double quat_i[4];
    ReplicateEulerQuat(rep.euler, static_cast<double>(i), cs, quat_i);
    auto frame = std::make_unique<Frame>();
    frame->pos = std::array<double, 3>{pos[0], pos[1], pos[2]};
    frame->quat = std::array<double, 4>{quat_i[0], quat_i[1], quat_i[2], quat_i[3]};
    for (const BodyChildAny& ic : inner)
      frame->subtree.push_back(CloneBodyChild(ic));  // base names already baked
    NamespaceSubtree(frame->subtree, ReplicateSuffix(sep, rep.count, i));
    out.push_back(BodyChildAny{std::move(frame)});
    lift::mjuu_frameaccum(pos, quat, offset, rot);
  }
  return out;
}

// Recursively replace every <replicate> in `subtree` with its expansion,
// descending into bodies and frames so nested replicates expand inner-first.
// Every cloned element carries the base auto-name computed from its ORIGINAL
// serial, so the later suffix pass reproduces the XML path's name+suffix.
std::vector<BodyChildAny> ExpandTree(const std::vector<BodyChildAny>& subtree,
                                     const CompilerSettings& cs,
                                     const ps::mjcf::CompileOptions& opts) {
  std::vector<BodyChildAny> out;
  for (const BodyChildAny& c : subtree) {
    switch (c.kind()) {
      case BodyChildAny::Kind::Replicate: {
        const auto& rp = std::get<std::unique_ptr<Replicate>>(c.node);
        if (!rp) break;
        std::vector<BodyChildAny> exp = ExpandReplicateNode(*rp, cs, opts);
        for (auto& e : exp) out.push_back(std::move(e));
        break;
      }
      case BodyChildAny::Kind::Body: {
        BodyChildAny clone{Clone(*std::get<std::unique_ptr<Body>>(c.node))};
        BakeName(c, clone, opts);
        std::get<std::unique_ptr<Body>>(clone.node)->subtree =
            ExpandTree(std::get<std::unique_ptr<Body>>(c.node)->subtree, cs, opts);
        out.push_back(std::move(clone));
        break;
      }
      case BodyChildAny::Kind::Frame: {
        BodyChildAny clone{Clone(*std::get<std::unique_ptr<Frame>>(c.node))};
        std::get<std::unique_ptr<Frame>>(clone.node)->subtree =
            ExpandTree(std::get<std::unique_ptr<Frame>>(c.node)->subtree, cs, opts);
        out.push_back(std::move(clone));
        break;
      }
      default: {
        BodyChildAny clone = CloneBodyChild(c);
        BakeName(c, clone, opts);
        out.push_back(std::move(clone));
      }
    }
  }
  return out;
}

BodyChildAny CloneBodyChild(const BodyChildAny& c) {
  switch (c.kind()) {
    case BodyChildAny::Kind::Geom:
      return BodyChildAny{Clone(*std::get<std::unique_ptr<Geom>>(c.node))};
    case BodyChildAny::Kind::Joint:
      return BodyChildAny{Clone(*std::get<std::unique_ptr<Joint>>(c.node))};
    case BodyChildAny::Kind::FreeJoint:
      return BodyChildAny{Clone(*std::get<std::unique_ptr<FreeJoint>>(c.node))};
    case BodyChildAny::Kind::Site:
      return BodyChildAny{Clone(*std::get<std::unique_ptr<Site>>(c.node))};
    case BodyChildAny::Kind::Camera:
      return BodyChildAny{Clone(*std::get<std::unique_ptr<Camera>>(c.node))};
    case BodyChildAny::Kind::Light:
      return BodyChildAny{Clone(*std::get<std::unique_ptr<Light>>(c.node))};
    case BodyChildAny::Kind::PluginRef:
      return BodyChildAny{Clone(*std::get<std::unique_ptr<PluginRef>>(c.node))};
    case BodyChildAny::Kind::Body:
      return BodyChildAny{Clone(*std::get<std::unique_ptr<Body>>(c.node))};
    case BodyChildAny::Kind::Frame:
      return BodyChildAny{Clone(*std::get<std::unique_ptr<Frame>>(c.node))};
    case BodyChildAny::Kind::Replicate:
      return BodyChildAny{Clone(*std::get<std::unique_ptr<Replicate>>(c.node))};
    case BodyChildAny::Kind::Composite:
      return BodyChildAny{Clone(*std::get<std::unique_ptr<Composite>>(c.node))};
    case BodyChildAny::Kind::Flexcomp:
      return BodyChildAny{Clone(*std::get<std::unique_ptr<Flexcomp>>(c.node))};
    case BodyChildAny::Kind::Attach:
      return BodyChildAny{Clone(*std::get<std::unique_ptr<Attach>>(c.node))};
  }
  return BodyChildAny{};
}

// --------------------------------------------------------------------------- //
// Flexcomp expansion (NC5 Wave 2). Reproduces mjCFlexcomp::Make for the         //
// procedural grid/box/square family at young=0, non-interpolated: generates the //
// point/element geometry (MakeGrid/MakeSquare/MakeBox), applies pins, then       //
// synthesizes one Body (explicit inertia + orthogonal sliders) per free point   //
// plus a single <flex> spec routed to the wave-1 flex compile. Lifted from       //
// user_flexcomp.cc (Make :148, MakeGrid :861, MakeSquare :1073, MakeBox :1110,   //
// GridID/BoxID/BoxProject); the numeric generators are verbatim (class-C).       //
// --------------------------------------------------------------------------- //
// Collect point for flexcomp expansion: synthesized flexes and their equalities,
// owned for the compile's lifetime and routed to the flex / equality compiles.
// A flexcomp-synthesized flex equality plus its mjtEq kind. The kind is not a
// field of EqualityFlex (the standalone <equality><flex> element is always
// mjEQ_FLEX); the flexcomp <edge equality="edge|vert"> synthesis records the
// mjEQ_FLEX / mjEQ_FLEXVERT distinction here (Make :788).
struct SynthEquality {
  std::unique_ptr<EqualityFlex> eq;
  int type = mjEQ_FLEX;
  bool has_data = false;      // strain: cell coords packed in data[0..2]
  double data[3] = {0, 0, 0};
};

struct FlexcompSink {
  std::vector<std::unique_ptr<Flex>> flexes;
  // interpolated node local offsets and empty-cell mask, parallel to `flexes`
  // (empty for non-interpolated). Consumed by FlexCompile.
  std::vector<std::vector<double>> node_local;
  std::vector<std::vector<char>> cell_empty;
  std::vector<SynthEquality> equalities;
};

struct FcompGeom {
  FlexcompType type = FlexcompType::grid;
  int count[3] = {10, 10, 10};
  double spacing[3] = {0.02, 0.02, 0.02};
  int dim = 2;
  bool needtex = false;
  std::vector<double> point;
  std::vector<int> element;
  std::vector<float> texcoord;

  int GridID(int ix, int iy) const { return ix * count[1] + iy; }
  int GridID(int ix, int iy, int iz) const {
    return ix * count[1] * count[2] + iy * count[2] + iz;
  }

  // point id from box coordinates and side (user_flexcomp.cc:996)
  int BoxID(int ix, int iy, int iz) const {
    if (iz == 0) {
      return ix * count[1] + iy + 1;
    } else if (iz == count[2] - 1) {
      return count[0] * count[1] + ix * count[1] + iy + 1;
    } else if (iy == 0) {
      return 2 * count[0] * count[1] + ix * (count[2] - 2) + iz - 1 + 1;
    } else if (iy == count[1] - 1) {
      return 2 * count[0] * count[1] + count[0] * (count[2] - 2) +
             ix * (count[2] - 2) + iz - 1 + 1;
    } else if (ix == 0) {
      return 2 * count[0] * count[1] + 2 * count[0] * (count[2] - 2) +
             (iy - 1) * (count[2] - 2) + iz - 1 + 1;
    } else {
      return 2 * count[0] * count[1] + 2 * count[0] * (count[2] - 2) +
             (count[1] - 2) * (count[2] - 2) + (iy - 1) * (count[2] - 2) + iz - 1 +
             1;
    }
  }

  // project from box to other shape (user_flexcomp.cc:1032)
  void BoxProject(double* pos, int ix, int iy, int iz) const {
    pos[0] = 2.0 * ix / (count[0] - 1) - 1;
    pos[1] = 2.0 * iy / (count[1] - 1) - 1;
    pos[2] = 2.0 * iz / (count[2] - 1) - 1;
    double size[3] = {0.5 * spacing[0] * (count[0] - 1),
                      0.5 * spacing[1] * (count[1] - 1),
                      0.5 * spacing[2] * (count[2] - 1)};
    if (type == FlexcompType::box) {
      pos[0] *= size[0]; pos[1] *= size[1]; pos[2] *= size[2];
    } else if (type == FlexcompType::cylinder) {
      double L0 = std::max(std::abs(pos[0]), std::abs(pos[1]));
      lift::mjuu_normvec(pos, 2);
      pos[0] *= size[0] * L0; pos[1] *= size[1] * L0; pos[2] *= size[2];
    } else if (type == FlexcompType::ellipsoid) {
      lift::mjuu_normvec(pos, 3);
      pos[0] *= size[0]; pos[1] *= size[1]; pos[2] *= size[2];
    }
  }

  // make grid (user_flexcomp.cc:861)
  void MakeGrid() {
    if (dim == 1) {
      for (int ix = 0; ix < count[0]; ix++) {
        if (type == FlexcompType::circle) {
          if (ix >= count[0] - 1) continue;
          double theta = 2 * mjPI / (count[0] - 1);
          double radius = spacing[0] / std::sin(theta / 2) / 2;
          point.push_back(radius * std::cos(theta * ix));
          point.push_back(radius * std::sin(theta * ix));
          point.push_back(0);
          element.push_back(ix);
          element.push_back(ix == count[0] - 2 ? 0 : ix + 1);
        } else {
          point.push_back(spacing[0] * (ix - 0.5 * (count[0] - 1)));
          point.push_back(0);
          point.push_back(0);
          if (ix < count[0] - 1) {
            element.push_back(ix);
            element.push_back(ix + 1);
          }
        }
      }
    } else if (dim == 2) {
      for (int ix = 0; ix < count[0]; ix++) {
        for (int iy = 0; iy < count[1]; iy++) {
          int quad2tri[2][3] = {{0, 1, 2}, {0, 2, 3}};
          double pos[2] = {spacing[0] * (ix - 0.5 * (count[0] - 1)),
                           spacing[1] * (iy - 0.5 * (count[1] - 1))};
          point.push_back(pos[0]);
          point.push_back(pos[1]);
          point.push_back(0);
          if (needtex) {
            texcoord.push_back(ix / (double)std::max(count[0] - 1, 1));
            texcoord.push_back(iy / (double)std::max(count[1] - 1, 1));
          }
          if (((pos[0] < -lift::mjEPS && pos[1] > -lift::mjEPS) ||
               (pos[0] > -lift::mjEPS && pos[1] < -lift::mjEPS)) &&
              type == FlexcompType::disc) {
            quad2tri[0][2] = 3;
            quad2tri[1][0] = 1;
          }
          if (ix < count[0] - 1 && iy < count[1] - 1) {
            int vert[4] = {
                count[2] * count[1] * (ix + 0) + count[2] * (iy + 0),
                count[2] * count[1] * (ix + 1) + count[2] * (iy + 0),
                count[2] * count[1] * (ix + 1) + count[2] * (iy + 1),
                count[2] * count[1] * (ix + 0) + count[2] * (iy + 1),
            };
            for (int s = 0; s < 2; s++)
              for (int v = 0; v < 3; v++)
                element.push_back(vert[quad2tri[s][v]]);
          }
        }
      }
    } else {
      int cube2tets[6][4] = {{0, 3, 1, 7}, {0, 1, 4, 7}, {1, 3, 2, 7},
                             {1, 2, 6, 7}, {1, 5, 4, 7}, {1, 6, 5, 7}};
      for (int ix = 0; ix < count[0]; ix++) {
        for (int iy = 0; iy < count[1]; iy++) {
          for (int iz = 0; iz < count[2]; iz++) {
            point.push_back(spacing[0] * (ix - 0.5 * (count[0] - 1)));
            point.push_back(spacing[1] * (iy - 0.5 * (count[1] - 1)));
            point.push_back(spacing[2] * (iz - 0.5 * (count[2] - 1)));
            if (needtex) {
              texcoord.push_back(ix / (float)std::max(count[0] - 1, 1));
              texcoord.push_back(iy / (float)std::max(count[1] - 1, 1));
            }
            if (ix < count[0] - 1 && iy < count[1] - 1 && iz < count[2] - 1) {
              int vert[8] = {
                  count[2] * count[1] * (ix + 0) + count[2] * (iy + 0) + iz + 0,
                  count[2] * count[1] * (ix + 1) + count[2] * (iy + 0) + iz + 0,
                  count[2] * count[1] * (ix + 1) + count[2] * (iy + 1) + iz + 0,
                  count[2] * count[1] * (ix + 0) + count[2] * (iy + 1) + iz + 0,
                  count[2] * count[1] * (ix + 0) + count[2] * (iy + 0) + iz + 1,
                  count[2] * count[1] * (ix + 1) + count[2] * (iy + 0) + iz + 1,
                  count[2] * count[1] * (ix + 1) + count[2] * (iy + 1) + iz + 1,
                  count[2] * count[1] * (ix + 0) + count[2] * (iy + 1) + iz + 1,
              };
              for (int s = 0; s < 6; s++)
                for (int v = 0; v < 4; v++)
                  element.push_back(vert[cube2tets[s][v]]);
            }
          }
        }
      }
    }
  }

  // make 2d square or disc (user_flexcomp.cc:1073)
  void MakeSquare() {
    dim = 2;
    MakeGrid();
    if (type == FlexcompType::disc) {
      double size[2] = {0.5 * spacing[0] * (count[0] - 1),
                        0.5 * spacing[1] * (count[1] - 1)};
      for (int i = 0; i < (int)point.size() / 3; i++) {
        double* pos = point.data() + i * 3;
        double L0 = std::max(std::abs(pos[0]), std::abs(pos[1]));
        lift::mjuu_normvec(pos, 2);
        pos[0] *= size[0] * L0;
        pos[1] *= size[1] * L0;
      }
    }
  }

  // make 3d box, ellipsoid or cylinder (user_flexcomp.cc:1110, open=true)
  void MakeBox() {
    const bool open = true;
    double pos[3];
    if (dim == 3) {
      point.push_back(0);
      point.push_back(0);
      point.push_back(0);
    }
    if (needtex) {
      texcoord.push_back(0);
      texcoord.push_back(0);
    }
    int n = 0;
    std::vector<int> idx(count[0] * count[1] * count[2]);
    auto mat2lin = [&](int ix, int iy, int iz) {
      return ix * count[1] * count[2] + iy * count[2] + iz;
    };
    // iz=0/max
    for (int iz = 0; iz < count[2]; iz += count[2] - 1) {
      for (int ix = 0; ix < count[0]; ix++) {
        for (int iy = 0; iy < count[1]; iy++) {
          if (open && dim == 2 && iz != 0) continue;
          BoxProject(pos, ix, iy, iz);
          point.push_back(pos[0]); point.push_back(pos[1]); point.push_back(pos[2]);
          idx[mat2lin(ix, iy, iz)] = n++;
          if (needtex) {
            texcoord.push_back(ix / (float)std::max(count[0] - 1, 1));
            texcoord.push_back(iy / (float)std::max(count[1] - 1, 1));
          }
        }
      }
    }
    // iy=0/max
    for (int iy = 0; iy < count[1]; iy += count[1] - 1) {
      for (int ix = 0; ix < count[0]; ix++) {
        for (int iz = 0; iz < count[2]; iz++) {
          if (iz > 0 && ((open && dim == 2) || (iz < count[2] - 1))) {
            BoxProject(pos, ix, iy, iz);
            point.push_back(pos[0]); point.push_back(pos[1]); point.push_back(pos[2]);
            idx[mat2lin(ix, iy, iz)] = n++;
            if (needtex) {
              texcoord.push_back(ix / (float)std::max(count[0] - 1, 1));
              texcoord.push_back(iz / (float)std::max(count[2] - 1, 1));
            }
          }
        }
      }
    }
    // ix=0/max
    for (int ix = 0; ix < count[0]; ix += count[0] - 1) {
      for (int iy = 0; iy < count[1]; iy++) {
        for (int iz = 0; iz < count[2]; iz++) {
          if (iz > 0 && ((open && dim == 2) || (iz < count[2] - 1)) && iy > 0 &&
              iy < count[1] - 1) {
            BoxProject(pos, ix, iy, iz);
            point.push_back(pos[0]); point.push_back(pos[1]); point.push_back(pos[2]);
            idx[mat2lin(ix, iy, iz)] = n++;
            if (needtex) {
              texcoord.push_back(iy / (float)std::max(count[1] - 1, 1));
              texcoord.push_back(iz / (float)std::max(count[2] - 1, 1));
            }
          }
        }
      }
    }
    // elements: iz=0/max
    for (int iz = 0; iz < count[2]; iz += count[2] - 1) {
      for (int ix = 0; ix < count[0]; ix++) {
        for (int iy = 0; iy < count[1]; iy++) {
          if (open && dim == 2 && iz != 0) continue;
          if (ix < count[0] - 1 && iy < count[1] - 1) {
            if (dim == 3) {
              element.push_back(0);
              element.push_back(BoxID(ix, iy, iz));
              element.push_back(BoxID(ix + 1, iy, iz));
              element.push_back(BoxID(ix + 1, iy + 1, iz));
              element.push_back(0);
              element.push_back(BoxID(ix, iy, iz));
              element.push_back(BoxID(ix, iy + 1, iz));
              element.push_back(BoxID(ix + 1, iy + 1, iz));
            } else {
              int step1 = iz == 0 ? 1 : 0;
              int step2 = iz == 0 ? 0 : 1;
              element.push_back(idx[mat2lin(ix, iy, iz)]);
              element.push_back(idx[mat2lin(ix + 1, iy + step1, iz)]);
              element.push_back(idx[mat2lin(ix + 1, iy + step2, iz)]);
              element.push_back(idx[mat2lin(ix, iy, iz)]);
              element.push_back(idx[mat2lin(ix + step2, iy + 1, iz)]);
              element.push_back(idx[mat2lin(ix + step1, iy + 1, iz)]);
            }
          }
        }
      }
    }
    // elements: iy=0/max
    for (int iy = 0; iy < count[1]; iy += count[1] - 1) {
      for (int ix = 0; ix < count[0]; ix++) {
        for (int iz = 0; iz < count[2]; iz++) {
          if (ix < count[0] - 1 && iz < count[2] - 1) {
            if (dim == 3) {
              element.push_back(0);
              element.push_back(BoxID(ix, iy, iz));
              element.push_back(BoxID(ix + 1, iy, iz));
              element.push_back(BoxID(ix + 1, iy, iz + 1));
              element.push_back(0);
              element.push_back(BoxID(ix, iy, iz));
              element.push_back(BoxID(ix, iy, iz + 1));
              element.push_back(BoxID(ix + 1, iy, iz + 1));
            } else {
              int ix0 = iy == 0 ? ix : ix + 1;
              int dx = iy == 0 ? 1 : -1;
              element.push_back(idx[mat2lin(ix0, iy, iz)]);
              element.push_back(idx[mat2lin(ix0 + dx, iy, iz)]);
              element.push_back(idx[mat2lin(ix0 + dx, iy, iz + 1)]);
              element.push_back(idx[mat2lin(ix0, iy, iz)]);
              element.push_back(idx[mat2lin(ix0 + dx, iy, iz + 1)]);
              element.push_back(idx[mat2lin(ix0, iy, iz + 1)]);
            }
          }
        }
      }
    }
    // elements: ix=0/max
    for (int ix = 0; ix < count[0]; ix += count[0] - 1) {
      for (int iy = 0; iy < count[1]; iy++) {
        for (int iz = 0; iz < count[2]; iz++) {
          if (iy < count[1] - 1 && iz < count[2] - 1) {
            if (dim == 3) {
              element.push_back(0);
              element.push_back(BoxID(ix, iy, iz));
              element.push_back(BoxID(ix, iy + 1, iz));
              element.push_back(BoxID(ix, iy + 1, iz + 1));
              element.push_back(0);
              element.push_back(BoxID(ix, iy, iz));
              element.push_back(BoxID(ix, iy, iz + 1));
              element.push_back(BoxID(ix, iy + 1, iz + 1));
            } else {
              int iy0 = ix != 0 ? iy : iy + 1;
              int dy = ix != 0 ? 1 : -1;
              element.push_back(idx[mat2lin(ix, iy0, iz)]);
              element.push_back(idx[mat2lin(ix, iy0 + dy, iz)]);
              element.push_back(idx[mat2lin(ix, iy0 + dy, iz + 1)]);
              element.push_back(idx[mat2lin(ix, iy0, iz)]);
              element.push_back(idx[mat2lin(ix, iy0 + dy, iz + 1)]);
              element.push_back(idx[mat2lin(ix, iy0, iz + 1)]);
            }
          }
        }
      }
    }
  }
};

// Mesh file-path helpers (defined later with the mesh pipeline); forward-declared
// so the flexcomp gmsh/mesh file arms can resolve asset paths like mesh geoms.
std::string MeshStripPath(const std::string& f);
std::string MeshCombine(const std::string& dir, const std::string& file);
std::string MeshExtToContentType(const std::string& file);

// --------------------------------------------------------------------------- //
// GMSH loader (NC5 Wave 5). Lifted from user_flexcomp.cc: the .msh reader that   //
// feeds the direct-points path. Upstream LoadGMSH/LoadGMSH41/LoadGMSH22 are      //
// member functions writing this->point/element and def.spec.flex->dim; here they //
// write a GmshOut (point/element/dim). Upstream `throw mjCError(NULL, msg)` is    //
// `throw GmshError{msg}`; the algorithm, byte offsets, and validation are        //
// verbatim (ASCII + binary, v4.1 + v2.2). Registry: gmsh_load*.                   //
// --------------------------------------------------------------------------- //
namespace gmsh {

struct GmshOut {
  std::vector<double> point;
  std::vector<int> element;
  int dim = 0;
};

struct GmshError { const char* msg; };

// read data of type T from a potentially unaligned buffer pointer
template <typename T>
void ReadFromBuffer(T* dst, const char* src) { std::memcpy(dst, src, sizeof(T)); }

void ReadStrFromBuffer(char* dest, const char* src, int maxlen) {
  std::strncpy(dest, src, maxlen);
}

bool IsValidElementOrNodeHeader22(const std::string& line) {
  for (char c : line)
    if (!std::isdigit(static_cast<unsigned char>(c))) return false;
  return true;
}

// find string in buffer, return position or -1 if not found
int GmshFindString(const char* buffer, int buffer_sz, const char* str) {
  int len = (int)std::strlen(str);
  for (int i = 0; i < buffer_sz - len; i++) {
    bool found = true;
    for (int k = 0; k < len; k++)
      if (buffer[i + k] != str[k]) { found = false; break; }
    if (found) return i;
  }
  return -1;
}

// load GMSH format 4.1
void LoadGMSH41(GmshOut& o, char* buffer, int binary, int nodeend,
                int nodebegin, int elemend, int elembegin) {
  constexpr int kGmsh41HeaderSize = 52;
  size_t minNodeTag, numEntityBlocks, numNodes, maxNodeTag, numNodesInBlock, tag;
  int entityDim, entityTag, parametric;

  // ascii nodes
  if (binary == 0) {
    std::stringstream ss(std::string(buffer + nodebegin, nodeend - nodebegin));
    ss >> numEntityBlocks >> numNodes >> minNodeTag >> maxNodeTag;
    ss >> entityDim >> entityTag >> parametric >> numNodesInBlock;
    if (!ss.good()) throw GmshError{"Error reading Nodes header"};
    if (numNodes < 0) throw GmshError{"Invalid number of nodes"};
    if (numEntityBlocks != 1 || numNodes != numNodesInBlock)
      throw GmshError{"All nodes must be in single block"};
    if (maxNodeTag != numNodesInBlock)
      throw GmshError{"Maximum number of nodes must be equal to number of nodes in a block"};
    if (entityDim < 1 || entityDim > 3)
      throw GmshError{"Entity must be 1D, 2D or 3D"};
    o.dim = entityDim;
    for (size_t i = 0; i < numNodes; i++) {
      size_t t;
      ss >> t;
      if (!ss.good()) throw GmshError{"Error reading node tags"};
      if (t != i + minNodeTag) throw GmshError{"Node tags must be sequential"};
    }
    if (numNodes < 0 || numNodes >= INT_MAX / 3)
      throw GmshError{"Invalid number of nodes."};
    o.point.reserve(3 * numNodes);
    for (size_t i = 0; i < 3 * numNodes; i++) {
      double x;
      ss >> x;
      if (!ss.good()) throw GmshError{"Error reading node coordinates"};
      o.point.push_back(x);
    }
  }
  // binary nodes
  else {
    if (nodeend - nodebegin < kGmsh41HeaderSize) throw GmshError{"Invalid nodes header"};
    ReadFromBuffer(&numEntityBlocks, buffer + nodebegin);
    ReadFromBuffer(&numNodes, buffer + nodebegin + 8);
    ReadFromBuffer(&minNodeTag, buffer + nodebegin + 16);
    ReadFromBuffer(&maxNodeTag, buffer + nodebegin + 24);
    ReadFromBuffer(&entityDim, buffer + nodebegin + 32);
    ReadFromBuffer(&entityTag, buffer + nodebegin + 36);
    ReadFromBuffer(&parametric, buffer + nodebegin + 40);
    ReadFromBuffer(&numNodesInBlock, buffer + nodebegin + 44);
    if (numEntityBlocks != 1 || numNodes != numNodesInBlock)
      throw GmshError{"All nodes must be in single block"};
    if (numNodes < 0) throw GmshError{"Invalid number of nodes"};
    if (entityDim < 1 || entityDim > 3) throw GmshError{"Entity must be 1D, 2D or 3D"};
    o.dim = entityDim;
    constexpr int numNodeComponents = 4;
    constexpr int componentSize = 8;
    int nodeDataSize = numNodeComponents * componentSize;
    if (nodeend - nodebegin < kGmsh41HeaderSize + (int)numNodes * nodeDataSize)
      throw GmshError{"Insufficient byte size of Nodes"};
    const char* tagbuffer = buffer + nodebegin + kGmsh41HeaderSize;
    for (size_t i = 0; i < numNodes; i++) {
      ReadFromBuffer(&tag, tagbuffer + i * componentSize);
      if (tag != i + minNodeTag) throw GmshError{"Node tags must be sequential"};
    }
    if (numNodes < 0 || numNodes >= INT_MAX / 3)
      throw GmshError{"Invalid number of nodes."};
    o.point.reserve(3 * numNodes);
    const char* pointbuffer =
        buffer + nodebegin + kGmsh41HeaderSize + componentSize * numNodes;
    for (size_t i = 0; i < 3 * numNodes; i++) {
      double x;
      ReadFromBuffer(&x, pointbuffer + i * componentSize);
      o.point.push_back(x);
    }
  }

  size_t numElements, minElementTag, maxElementTag, numElementsInBlock;
  int elementType;

  // ascii elements
  if (binary == 0) {
    buffer[elemend] = 0;
    std::stringstream ss(std::string(buffer + elembegin, elemend - elembegin));
    ss >> numEntityBlocks >> numElements >> minElementTag >> maxElementTag;
    ss >> entityDim >> entityTag >> elementType >> numElementsInBlock;
    if (!ss.good()) throw GmshError{"Error reading Elements header"};
    if (numEntityBlocks != 1 || numElements != numElementsInBlock)
      throw GmshError{"All elements must be in single block"};
    if (numElements < 0) throw GmshError{"Invalid number of elements"};
    if (entityDim != o.dim) throw GmshError{"Inconsistent dimensionality in Elements"};
    if (numElements < 0 || numElements >= INT_MAX / 4)
      throw GmshError{"Invalid numElements."};
    if ((entityDim == 1 && elementType != 1) ||
        (entityDim == 2 && elementType != 2) ||
        (entityDim == 3 && elementType != 4))
      throw GmshError{"Element type inconsistent with dimensionality"};
    o.element.reserve((entityDim + 1) * numElements);
    for (size_t i = 0; i < numElements; i++) {
      size_t t, nodeid;
      ss >> t;
      for (int k = 0; k <= entityDim; k++) {
        ss >> nodeid;
        if (!ss.good()) throw GmshError{"Error reading Elements"};
        o.element.push_back((int)(nodeid - minNodeTag));
      }
    }
  }
  // binary elements
  else {
    if (elemend - elembegin < kGmsh41HeaderSize) throw GmshError{"Invalid elements header"};
    ReadFromBuffer(&numEntityBlocks, buffer + elembegin);
    ReadFromBuffer(&numElements, buffer + elembegin + 8);
    ReadFromBuffer(&minElementTag, buffer + elembegin + 16);
    ReadFromBuffer(&maxElementTag, buffer + elembegin + 24);
    ReadFromBuffer(&entityDim, buffer + elembegin + 32);
    ReadFromBuffer(&entityTag, buffer + elembegin + 36);
    ReadFromBuffer(&elementType, buffer + elembegin + 40);
    ReadFromBuffer(&numElementsInBlock, buffer + elembegin + 44);
    if (numEntityBlocks != 1 || numElements != numElementsInBlock)
      throw GmshError{"All elements must be in single block"};
    if (numElements < 0) throw GmshError{"Invalid number of elements"};
    if (entityDim != o.dim) throw GmshError{"Inconsistent dimensionality in Elements"};
    if ((entityDim == 1 && elementType != 1) ||
        (entityDim == 2 && elementType != 2) ||
        (entityDim == 3 && elementType != 4))
      throw GmshError{"Element type inconsistent with dimensionality"};
    if (numElements < 0 || numElements >= INT_MAX / 4)
      throw GmshError{"Invalid numElements."};
    int numElementComponents = (entityDim + 2);
    constexpr int componentSize = 8;
    int elementDataSize = numElementComponents * componentSize;
    if (elemend - elembegin < kGmsh41HeaderSize + (int)numElements * elementDataSize)
      throw GmshError{"Insufficient byte size of Elements"};
    o.element.reserve((entityDim + 1) * numElements);
    const char* elembuffer = buffer + elembegin + kGmsh41HeaderSize;
    for (size_t i = 0; i < numElements; i++) {
      elembuffer += componentSize;
      size_t elemid;
      for (int k = 0; k <= entityDim; k++) {
        ReadFromBuffer(&elemid, elembuffer);
        int elementid = elemid - minNodeTag;
        o.element.push_back(elementid);
        elembuffer += componentSize;
      }
    }
  }
}

// load GMSH format 2.2
void LoadGMSH22(GmshOut& o, char* buffer, int binary, int nodeend,
                int nodebegin, int elemend, int elembegin) {
  size_t numNodes = 0;

  // ascii nodes
  if (binary == 0) {
    std::stringstream ss(std::string(buffer + nodebegin, nodeend - nodebegin));
    std::string line;
    std::getline(ss, line);
    if (!IsValidElementOrNodeHeader22(line)) throw GmshError{"Invalid node header"};
    ss.seekg(-(long)(line.size() + 1), std::ios::cur);
    size_t maxNodeTag = 0;
    ss >> maxNodeTag;
    if (!ss.good()) throw GmshError{"Error reading Nodes header"};
    numNodes = maxNodeTag;
    if (numNodes < 0 || numNodes >= INT_MAX / 3)
      throw GmshError{"Invalid number of nodes."};
    o.point.reserve(3 * numNodes);
    for (size_t i = 0; i < numNodes; i++) {
      size_t tag;
      double x;
      ss >> tag;
      if (!ss.good()) throw GmshError{"Error reading node tags"};
      for (int k = 0; k < 3; k++) {
        ss >> x;
        if (!ss.good()) throw GmshError{"Error reading node coordinates"};
        o.point.push_back(x);
      }
    }
  }
  // binary nodes
  else {
    constexpr int nodeHeaderSizeGmshApp = 5;
    constexpr int nodeHeaderSize = nodeHeaderSizeGmshApp - 1;
    if (nodeend - nodebegin < nodeHeaderSize) throw GmshError{"Invalid nodes header"};
    char maxNodeTagChar[11] = {0};
    ReadStrFromBuffer(maxNodeTagChar, buffer + nodebegin, std::min(10, nodeend - nodebegin));
    size_t measuredHeaderSize = strnlen(maxNodeTagChar, 10) - 1;
    size_t maxNodeTag;
    try {
      maxNodeTag = std::stoi(maxNodeTagChar);
    } catch (const std::out_of_range&) {
      throw GmshError{"Invalid number of nodes"};
    }
    numNodes = maxNodeTag;
    if (numNodes < 0) throw GmshError{"Invalid number of nodes"};
    int nodeSize = sizeof(double);
    int indexSize = sizeof(int);
    int nodeDataSize = indexSize + 3 * nodeSize;
    if (nodeend - nodebegin < nodeHeaderSize + (int)numNodes * nodeDataSize)
      throw GmshError{"Insufficient byte size of Nodes"};
    if (numNodes < 0 || numNodes >= INT_MAX / 3)
      throw GmshError{"Invalid number of nodes."};
    o.point.reserve(3 * numNodes);
    const char* tagBuffer = buffer + nodebegin + measuredHeaderSize;
    for (int i = 0; i < (int)numNodes; i++) {
      int tag;
      int offset = i * (sizeof(int) + sizeof(double) * 3);
      ReadFromBuffer(&tag, tagBuffer + offset);
      for (int k = 0; k < 3; k++) {
        double x;
        const char* nodeBuffer = tagBuffer + sizeof(int) + sizeof(double) * k;
        ReadFromBuffer(&x, nodeBuffer + offset);
        o.point.push_back(x);
      }
    }
  }

  // ascii elements
  if (binary == 0) {
    buffer[elemend] = 0;
    std::stringstream ss(std::string(buffer + elembegin, elemend - elembegin));
    std::string line;
    std::getline(ss, line);
    if (!IsValidElementOrNodeHeader22(line)) throw GmshError{"Invalid elements header"};
    ss.seekg(-(long)(line.size() + 1), std::ios::cur);
    size_t maxElementTag = 0;
    ss >> maxElementTag;
    if (!ss.good()) throw GmshError{"Error reading Elements header"};
    size_t numElements = maxElementTag;
    if (numElements < 0 || numElements >= INT_MAX / 4)
      throw GmshError{"Invalid number of elements."};
    int tag = 0, elementType = 0, numTags = 0;
    ss >> tag >> elementType >> numTags;
    if (!ss.good()) throw GmshError{"Error reading Elements"};
    size_t entityDim = 0;
    int numNodeTags = 0;
    if (elementType == 2) { entityDim = 2; numNodeTags = 3; }
    else if (elementType == 4) { entityDim = 3; numNodeTags = 4; }
    if (numNodeTags < 1 || numNodeTags > 4) throw GmshError{"Invalid number of node tags"};
    o.dim = entityDim;
    o.element.reserve(numNodeTags * numElements);
    for (size_t i = 0; i < numElements; i++) {
      int nodeTag = 0, physicalEntityTag = 0, elementModelEntityTag = 0;
      if (i != 0) {
        ss >> tag >> elementType >> numTags;
        if (!ss.good()) throw GmshError{"Error reading Elements"};
      }
      if (numTags > 0) {
        ss >> physicalEntityTag >> elementModelEntityTag;
        if (!ss.good()) throw GmshError{"Error reading Elements"};
      }
      for (int k = 0; k < numNodeTags; k++) {
        ss >> nodeTag;
        if (!ss.good()) throw GmshError{"Error reading Elements"};
        if (nodeTag > (int)numNodes || nodeTag < 1) throw GmshError{"Invalid node tag"};
        o.element.push_back((int)(nodeTag - 1));
      }
    }
  }
  // binary elements
  else {
    constexpr int elementHeaderSizeGmshApp = 4;
    if (elemend - elembegin < elementHeaderSizeGmshApp)
      throw GmshError{"Invalid elements header"};
    char maxElementTagChar[11] = {0};
    ReadStrFromBuffer(maxElementTagChar, buffer + elembegin, std::min(10, elemend - elembegin));
    int measuredHeaderSize = strnlen(maxElementTagChar, 10) - 1;
    int maxElementTag;
    try {
      maxElementTag = std::stoi(maxElementTagChar);
    } catch (const std::out_of_range&) {
      throw GmshError{"Invalid number of elements"};
    }
    int numElements = maxElementTag;
    int tag, numTags;
    int nodeTag;
    int elementType;
    if (numElements < 0) throw GmshError{"Invalid number of elements"};
    int componentSize = sizeof(int);
    const char* elementsBuffer = buffer + elembegin + measuredHeaderSize;
    ReadFromBuffer(&elementType, elementsBuffer);
    ReadFromBuffer(&numTags, elementsBuffer + componentSize * 2);
    ReadFromBuffer(&tag, elementsBuffer + componentSize * 3);
    int numNodeTags = 0;
    size_t entityDim = 0;
    if (elementType == 2) { entityDim = 2; numNodeTags = 3; }
    else if (elementType == 4) { entityDim = 3; numNodeTags = 4; }
    if (numNodeTags < 1 || numNodeTags > 4) throw GmshError{"Invalid number of node tags"};
    o.dim = entityDim;
    constexpr int numComponentsFtetwild = 5;
    constexpr int numInfoComponents = 4;
    constexpr int numEntityTagComponents = 2;
    constexpr int elementHeaderSizeFtetwild = 17;
    int numComponentsGmshApp = numInfoComponents + numEntityTagComponents + numNodeTags;
    int elementDataSizeFtetwild = numComponentsFtetwild * componentSize;
    int elementDataSizeGmshApp = numComponentsGmshApp * componentSize;
    int elementsBufferSizeFtetwild =
        elementHeaderSizeFtetwild + numElements * elementDataSizeFtetwild;
    int elementsBufferSizeGmshApp =
        elementHeaderSizeGmshApp + numElements * elementDataSizeGmshApp;
    if (elemend - elembegin < elementsBufferSizeFtetwild)
      throw GmshError{"Insufficient byte size of Elements"};
    if (numTags > 0) {
      if (elemend - elembegin < elementsBufferSizeGmshApp)
        throw GmshError{"Insufficient byte size of Elements"};
      for (int k = 0; k < numNodeTags; k++) {
        ReadFromBuffer(&nodeTag, elementsBuffer + componentSize * (6 + k));
        if (nodeTag > (int)numNodes || nodeTag < 1) throw GmshError{"Invalid node tag"};
        o.element.push_back(nodeTag - 1);
      }
      for (int i = 1; i < numElements; i++) {
        const char* numTagsBuffer = elementsBuffer + componentSize * 2;
        const char* tagBuffer = elementsBuffer + componentSize * 3;
        int offset = i * elementDataSizeGmshApp;
        ReadFromBuffer(&numTags, numTagsBuffer + offset);
        ReadFromBuffer(&tag, tagBuffer + offset);
        for (int k = 0; k < numNodeTags; k++) {
          const char* nodeTagBuffer = elementsBuffer + componentSize * (6 + k);
          ReadFromBuffer(&nodeTag, nodeTagBuffer + offset);
          if (nodeTag > numElements || nodeTag < 1) throw GmshError{"Invalid node tag"};
          o.element.push_back(nodeTag - 1);
        }
      }
    } else {
      for (int k = 0; k < numNodeTags; k++) {
        const char* nodeTagBuffer = elementsBuffer + componentSize * (4 + k);
        ReadFromBuffer(&nodeTag, nodeTagBuffer);
        if (nodeTag > (int)numNodes || nodeTag < 1) throw GmshError{"Invalid node tag"};
        o.element.push_back(nodeTag - 1);
      }
      for (int i = 0; i < numElements - 1; i++) {
        int offset = componentSize * (4 + 2) + i * elementDataSizeFtetwild;
        const char* tagBuffer = elementsBuffer + componentSize * 2;
        ReadFromBuffer(&tag, tagBuffer + offset);
        for (int k = 0; k < numNodeTags; k++) {
          const char* nodeTagBuffer = elementsBuffer + componentSize * (3 + k);
          ReadFromBuffer(&nodeTag, nodeTagBuffer + offset);
          if (nodeTag > numElements || nodeTag < 1) throw GmshError{"Invalid node tag"};
          o.element.push_back(nodeTag - 1);
        }
      }
    }
  }
}

// load GMSH file from a mutable buffer (LoadGMSH). Returns false with `err` set
// on any parse error (mirroring the mjCError throws upstream).
bool LoadGMSH(GmshOut& o, char* buffer, int buffer_sz, std::string& err) {
  try {
    if (buffer_sz < 0) throw GmshError{"Could not read GMSH file"};
    if (buffer_sz == 0) throw GmshError{"Empty GMSH file"};
    if (buffer_sz < 11 || std::strncmp(buffer, "$MeshFormat", 11))
      throw GmshError{"GMSH file must begin with $MeshFormat"};
    double version;
    int binary;
    if (sscanf(buffer + 11, "%lf %d", &version, &binary) != 2)
      throw GmshError{"Could not read GMSH file header"};
    if (mju_round(100 * version) != 220 && mju_round(100 * version) != 410)
      throw GmshError{"Only GMSH file format versions 4.1 and 2.2 are supported"};
    int nodebegin = GmshFindString(buffer, buffer_sz, "$Nodes");
    int nodeend = GmshFindString(buffer, buffer_sz, "$EndNodes");
    int elembegin = GmshFindString(buffer, buffer_sz, "$Elements");
    int elemend = GmshFindString(buffer, buffer_sz, "$EndElements");
    nodebegin += (int)std::strlen("$Nodes") + 1;
    elembegin += (int)std::strlen("$Elements") + 1;
    if (nodebegin < 0) throw GmshError{"GMSH file missing $Nodes"};
    if (nodeend < nodebegin) throw GmshError{"GMSH file missing $EndNodes after $Nodes"};
    if (elembegin < 0) throw GmshError{"GMSH file missing $Elements"};
    if (elemend < elembegin) throw GmshError{"GMSH file missing $EndElements after $Elements"};
    if (mju_round(100 * version) == 410)
      LoadGMSH41(o, buffer, binary, nodeend, nodebegin, elemend, elembegin);
    else if (mju_round(100 * version) == 220)
      LoadGMSH22(o, buffer, binary, nodeend, nodebegin, elemend, elembegin);
    else
      throw GmshError{"Unsupported GMSH file format version"};
  } catch (const GmshError& e) {
    err = e.msg;
    return false;
  } catch (...) {
    err = "exception while reading GMSH file";
    return false;
  }
  return true;
}

}  // namespace gmsh

// Forward decl (defined with the flex compile helpers): empty-cell mask from
// vertex/element geometry, needed by the interpolated flexcomp expansion.
void FlexComputeCellEmpty(const double* vpos, const int* elems, int nv, int ne,
                          int fdim, const int cellcount[3],
                          std::vector<char>& cell_empty,
                          const double* bbox);

// Synthesize the flexcomp's bodies (appended to `out_bodies`) and its <flex>
// spec (returned; null when the expansion is unsupported/degenerate). Mirrors
// mjCFlexcomp::Make for the grid/box/square family, including the interpolated
// (trilinear/quadratic) nodal finite-element mesh.
std::unique_ptr<Flex> ExpandFlexcompInto(const Flexcomp& fc,
                                         const CompilerSettings& cs,
                                         const ps::mjcf::CompileOptions& opts,
                                         const std::string& parent_name,
                                         std::vector<BodyChildAny>& out_bodies,
                                         std::vector<SynthEquality>& out_eqs,
                                         std::vector<double>& out_node_local,
                                         std::vector<char>& out_cell_empty) {
  const FlexcompType type = fc.type ? *fc.type : FlexcompType::grid;
  const FlexDof dof = fc.dof ? *fc.dof : FlexDof::full;
  int dim = fc.dim ? *fc.dim : 2;
  const std::string name = fc.name ? *fc.name : std::string();

  // Interpolated (trilinear/quadratic) dof drives the nodal finite-element path
  // (NC5 Wave 6): all vertices attach to the parent, a separate nodal mesh of
  // slider bodies carries the dofs, and stiffness/bending come from the FE
  // kernels. Reduced dof -- radial (one slider per vertex) and 2d (two) -- is a
  // per-vertex joint change only. The `direct` type (authored inline points,
  // Wave 4), `gmsh` (.msh file, Wave 5), and `mesh` (OBJ/STL, Wave 5b) reach
  // here; upstream treats DIRECT/MESH/GMSH uniformly (the "direct" family).
  const bool interpolated =
      (dof == FlexDof::trilinear || dof == FlexDof::quadratic);
  const int order = dof == FlexDof::trilinear ? 1
                    : dof == FlexDof::quadratic ? 2
                                                : 0;
  const bool is_direct = (type == FlexcompType::direct ||
                          type == FlexcompType::gmsh ||
                          type == FlexcompType::mesh);
  // elemtexcoord (per-element-vertex texcoord ids) only the mesh path produces.
  std::vector<int> elemtexcoord;

  FcompGeom g;
  g.type = type;
  g.count[0] = g.count[1] = g.count[2] = 10;
  if (fc.count) for (int k = 0; k < 3; ++k) g.count[k] = (*fc.count)[k];
  g.spacing[0] = g.spacing[1] = g.spacing[2] = 0.02;
  if (fc.spacing) for (int k = 0; k < 3; ++k) g.spacing[k] = (*fc.spacing)[k];
  g.dim = dim;
  const bool has_material = fc.material && !fc.material->empty();
  g.needtex = has_material;  // no explicit flexcomp texcoord on this path

  // type-specific geometry (also fixes dim for square/disc/box)
  switch (type) {
    case FlexcompType::grid:
    case FlexcompType::circle:
      g.MakeGrid();
      break;
    case FlexcompType::box:
    case FlexcompType::cylinder:
    case FlexcompType::ellipsoid:
      g.MakeBox();
      break;
    case FlexcompType::square:
    case FlexcompType::disc:
      g.MakeSquare();
      break;
    case FlexcompType::direct:
      // Points/elements authored inline (Make :226 sets res=true; the reader
      // parsed them into point/element/texcoord). dim stays as authored.
      if (fc.point) g.point = *fc.point;
      if (fc.element) g.element = *fc.element;
      if (fc.texcoord) g.texcoord = *fc.texcoord;
      break;
    case FlexcompType::gmsh: {
      // Load points/elements from the .msh file (MakeGMSH). The file resolves
      // like a mesh asset (strippath + meshdir + base-dir resource open). dim is
      // taken from the file's entity dimensionality, overriding the authored dim.
      if (!fc.file || fc.file->empty()) return nullptr;
      std::string file = *fc.file;
      if (cs.strippath) file = MeshStripPath(file);
      const std::string combined = MeshCombine(cs.meshdir, file);
      char oerr[1024] = {0};
      mjResource* res = mju_openResource(
          opts.base_dir.empty() ? nullptr : opts.base_dir.c_str(),
          combined.c_str(), nullptr, oerr, sizeof(oerr));
      if (!res) return nullptr;
      const void* bytes = nullptr;
      int n = mju_readResource(res, &bytes);
      if (n < 0) { mju_closeResource(res); return nullptr; }
      std::vector<char> buf(static_cast<const char*>(bytes),
                            static_cast<const char*>(bytes) + n);
      mju_closeResource(res);
      gmsh::GmshOut go;
      std::string gerr;
      if (!gmsh::LoadGMSH(go, buf.data(), static_cast<int>(buf.size()), gerr))
        return nullptr;
      g.point = std::move(go.point);
      g.element = std::move(go.element);
      g.dim = go.dim;
      break;
    }
    case FlexcompType::mesh: {
      // Load raw verts/faces from an OBJ/STL file (MakeMesh). dim is authored
      // (>=1); legacy .msh is rejected in the flexcomp mesh path upstream.
      if (!fc.file || fc.file->empty()) return nullptr;
      if (dim < 1) return nullptr;  // "Flex dim must be at least 1 for mesh"
      std::string file = *fc.file;
      if (cs.strippath) file = MeshStripPath(file);
      lift::MeshInput min;
      const std::string ct = MeshExtToContentType(file);
      if (ct == "model/obj") min.format = lift::MeshFormat::Obj;
      else if (ct == "model/stl") min.format = lift::MeshFormat::Stl;
      else return nullptr;  // legacy .msh / unknown extension
      const std::string combined = MeshCombine(cs.meshdir, file);
      char oerr[1024] = {0};
      mjResource* res = mju_openResource(
          opts.base_dir.empty() ? nullptr : opts.base_dir.c_str(),
          combined.c_str(), nullptr, oerr, sizeof(oerr));
      if (!res) return nullptr;
      const void* bytes = nullptr;
      int n = mju_readResource(res, &bytes);
      if (n < 0) { mju_closeResource(res); return nullptr; }
      min.content_type = ct;
      min.filebytes.assign(static_cast<const char*>(bytes),
                           static_cast<const char*>(bytes) + n);
      mju_closeResource(res);
      lift::MeshRawResult mr;
      std::string merr;
      if (!lift::LoadMeshRaw(min, mr, merr)) return nullptr;
      if (mr.vert.empty() || mr.face.empty()) return nullptr;
      // ProcessVertices(remove_repeated=true) (user_mesh.cc:539): the flexcomp
      // mesh path loads with remove_repeated, merging position-duplicate
      // vertices (new index by first occurrence) and remapping faces. texcoord /
      // facetexcoord are per-corner and unaffected.
      {
        const int nv = static_cast<int>(mr.vert.size()) / 3;
        std::map<std::array<float, 3>, int> vmap;
        int index = 0;
        for (int i = 0; i < nv; i++) {
          std::array<float, 3> key = {mr.vert[3 * i], mr.vert[3 * i + 1],
                                      mr.vert[3 * i + 2]};
          if (vmap.find(key) == vmap.end()) vmap.emplace(key, index++);
        }
        if (index != nv) {
          for (int& fi : mr.face) {
            std::array<float, 3> key = {mr.vert[3 * fi], mr.vert[3 * fi + 1],
                                        mr.vert[3 * fi + 2]};
            fi = vmap[key];
          }
          std::vector<float> newvert(3 * index);
          for (const auto& [key, idx] : vmap) {
            newvert[3 * idx + 0] = key[0];
            newvert[3 * idx + 1] = key[1];
            newvert[3 * idx + 2] = key[2];
          }
          mr.vert = std::move(newvert);
        }
      }
      // copy vertices (float -> double)
      g.point.assign(mr.vert.begin(), mr.vert.end());
      if (mr.has_texcoord()) {
        g.texcoord = mr.texcoord;
        elemtexcoord = mr.facetexcoord;
      }
      // faces or 3D tets (MakeMesh :1382-1418)
      if (dim == 2) {
        g.element = mr.face;
      } else if (dim == 1) {
        // edge pairs from degenerate triangles (i1, i2, i2)
        g.element.clear();
        g.element.reserve(mr.face.size() * 2 / 3);
        for (std::size_t i = 0; i < mr.face.size(); i += 3) {
          g.element.push_back(mr.face[i]);
          g.element.push_back(mr.face[i + 1]);
        }
      } else {
        // dim >= 3: origin (required) prepended as vertex 0, then one tetra per
        // positive-volume surface triangle.
        if (!fc.origin) return nullptr;
        const auto& org = *fc.origin;
        double origin[3] = {org[0], org[1], org[2]};
        g.point.insert(g.point.begin() + 0, origin[0]);
        g.point.insert(g.point.begin() + 1, origin[1]);
        g.point.insert(g.point.begin() + 2, origin[2]);
        std::vector<int> tetel;
        for (std::size_t i = 0; i + 2 < mr.face.size(); i += 3) {
          int tet[3] = {mr.face[i + 0] + 1, mr.face[i + 1] + 1, mr.face[i + 2] + 1};
          double edge1[3], edge2[3], edge3[3];
          for (int k = 0; k < 3; k++) {
            edge1[k] = g.point[3 * tet[0] + k] - origin[k];
            edge2[k] = g.point[3 * tet[1] + k] - origin[k];
            edge3[k] = g.point[3 * tet[2] + k] - origin[k];
          }
          double normal[3];
          lift::mjuu_crossvec(normal, edge1, edge2);
          if (lift::mjuu_dot3(normal, edge3) < mjMINVAL) continue;
          tetel.push_back(0);
          tetel.push_back(tet[0]);
          tetel.push_back(tet[1]);
          tetel.push_back(tet[2]);
        }
        g.element = std::move(tetel);
      }
      g.dim = dim;
      break;
    }
    default:
      return nullptr;
  }
  dim = g.dim;
  if (g.element.empty() || g.point.empty()) return nullptr;
  // element sizing + vertex-id range (Make :272-287); a malformed direct spec
  // routes the whole model to the XML fallback (leg B raises the same error).
  if (g.point.size() % 3 ||
      g.element.size() % static_cast<std::size_t>(dim + 1))
    return nullptr;
  {
    const int np = static_cast<int>(g.point.size()) / 3;
    for (int v : g.element)
      if (v < 0 || v >= np) return nullptr;
  }
  // scaling applies only to direct types (Make :289-296), before the pose xform.
  if (is_direct && fc.scale) {
    const auto& s = *fc.scale;
    if (s[0] != 1 || s[1] != 1 || s[2] != 1) {
      const int np = static_cast<int>(g.point.size()) / 3;
      for (int i = 0; i < np; i++) {
        g.point[3 * i + 0] *= s[0];
        g.point[3 * i + 1] *= s[1];
        g.point[3 * i + 2] *= s[2];
      }
    }
  }

  // force flatskin for box, cylinder, and 3D grid (Make :237)
  const bool force_flatskin =
      type == FlexcompType::box || type == FlexcompType::cylinder ||
      (type == FlexcompType::grid && dim == 3);

  int npnt = static_cast<int>(g.point.size()) / 3;

  // pose transform to points (Make :298). scale applies only to direct types.
  double pos[3] = {0, 0, 0}, quat[4] = {1, 0, 0, 0};
  if (fc.pos) for (int k = 0; k < 3; ++k) pos[k] = (*fc.pos)[k];
  ResolveQuat(fc.quat, quat);
  for (int i = 0; i < npnt; i++) {
    double oldp[3] = {g.point[3 * i], g.point[3 * i + 1], g.point[3 * i + 2]};
    double newp[3];
    lift::mjuu_trnVecPose(newp, pos, quat, oldp);
    g.point[3 * i] = newp[0];
    g.point[3 * i + 1] = newp[1];
    g.point[3 * i + 2] = newp[2];
  }

  // bounding box of the (pose-transformed) points (Make :309).
  double minmax[6] = {mjMAXVAL,  mjMAXVAL,  mjMAXVAL,
                      -mjMAXVAL, -mjMAXVAL, -mjMAXVAL};
  for (int i = 0; i < npnt; i++)
    for (int j = 0; j < 3; j++) {
      minmax[j + 0] = std::min(minmax[j + 0], g.point[3 * i + j]);
      minmax[j + 3] = std::max(minmax[j + 3], g.point[3 * i + j]);
    }

  // flex cellcount (default 1,1,1; overridden by the authored attribute). The
  // pinned-array node count (Make :317) uses cellcount only for mesh/direct/gmsh,
  // else a single cell -- a quirk we must reproduce for the shared pinned array.
  int cc[3] = {1, 1, 1};
  const bool has_cellcount = fc.cellcount && (*fc.cellcount)[0] >= 0;
  if (interpolated && has_cellcount)
    for (int k = 0; k < 3; ++k) cc[k] = (*fc.cellcount)[k];
  int nnode_early = 0;
  if (interpolated) {
    int cx = 1, cy = 1, cz = 1;
    if ((type == FlexcompType::mesh || type == FlexcompType::direct ||
         type == FlexcompType::gmsh) &&
        has_cellcount) {
      cx = (*fc.cellcount)[0];
      cy = (*fc.cellcount)[1];
      cz = (*fc.cellcount)[2];
    }
    nnode_early = (cx * order + 1) * (cy * order + 1) * (cz * order + 1);
  }

  // pinned array (Make :316). rigid forces all pinned. For interpolated the
  // array is shared between vertices [0,npnt) and nodes [0,nnode).
  const bool rigid_attr = fc.rigid && *fc.rigid;
  std::vector<char> pinned(std::max(npnt, nnode_early), rigid_attr ? 1 : 0);
  bool rigid = rigid_attr;
  bool centered = false;
  if (!rigid) {
    auto in_grid = [&](int idv) { return idv >= 0 && idv < npnt; };
    for (const auto& p : fc.flexcompPins) {
      if (!p) continue;
      // pin grid(range) is a grid/interpolated-only addressing (Make :253); a
      // direct flexcomp using it is an upstream hard error -> XML fallback.
      if (is_direct && (p->grid || p->gridrange)) return nullptr;
      if (p->id) for (int v : *p->id) { if (!in_grid(v)) return nullptr; pinned[v] = 1; }
      if (p->range) {
        const auto& r = *p->range;
        for (std::size_t i = 0; i + 1 < r.size(); i += 2) {
          if (!in_grid(r[i]) || !in_grid(r[i + 1])) return nullptr;
          for (int k = r[i]; k <= r[i + 1]; k++) pinned[k] = 1;
        }
      }
      if (p->grid) {
        const auto& gr = *p->grid;
        for (std::size_t i = 0; i + dim <= gr.size(); i += dim) {
          for (int k = 0; k < dim; k++)
            if (gr[i + k] < 0 || gr[i + k] >= g.count[k]) return nullptr;
          if (dim == 2) pinned[g.GridID(gr[i], gr[i + 1])] = 1;
          else if (dim == 3) pinned[g.GridID(gr[i], gr[i + 1], gr[i + 2])] = 1;
          else return nullptr;
        }
      }
      if (p->gridrange) {
        const auto& gr = *p->gridrange;
        const std::size_t stride = 2 * static_cast<std::size_t>(dim);
        for (std::size_t i = 0; i + stride <= gr.size(); i += stride) {
          for (int k = 0; k < 2 * dim; k++)
            if (gr[i + k] < 0 || gr[i + k] >= g.count[k % dim]) return nullptr;
          if (dim == 2) {
            for (int ix = gr[i]; ix <= gr[i + 2]; ix++)
              for (int iy = gr[i + 1]; iy <= gr[i + 3]; iy++)
                pinned[g.GridID(ix, iy)] = 1;
          } else if (dim == 3) {
            for (int ix = gr[i]; ix <= gr[i + 3]; ix++)
              for (int iy = gr[i + 1]; iy <= gr[i + 4]; iy++)
                for (int iz = gr[i + 2]; iz <= gr[i + 5]; iz++)
                  pinned[g.GridID(ix, iy, iz)] = 1;
          } else {
            return nullptr;
          }
        }
      }
    }
    // center of a radial body is always pinned (its radial axis is degenerate);
    // it attaches to the parent instead of getting a zero-axis slider (Make :413).
    if (dof == FlexDof::radial && npnt > 0) pinned[0] = 1;
    bool allpin = true, nopin = true;
    for (int i = 0; i < npnt; i++) {
      if (pinned[i]) nopin = false; else allpin = false;
    }
    if (allpin) rigid = true;
    else if (nopin) centered = true;
  }

  // remove unreferenced points for direct types (Make :438-493): compact
  // point/texcoord/pinned and reindex element so a point no element uses is
  // dropped (the procedural families reference every point, so this is a no-op
  // there and stays direct-only). npnt shrinks to the compacted count.
  if (is_direct) {
    std::vector<char> used(npnt, 0);
    for (int v : g.element) used[v] = 1;
    std::vector<int> reindex(npnt, 0);
    bool hasunused = false;
    for (int i = 0; i < npnt; i++)
      if (!used[i]) {
        hasunused = true;
        for (int k = i + 1; k < npnt; k++) reindex[k]--;
      }
    if (hasunused) {
      for (int& v : g.element) v += reindex[v];
      int new_npnt = 0;
      for (int i = 0; i < npnt; i++) {
        if (!used[i]) continue;
        g.point[3 * new_npnt + 0] = g.point[3 * i + 0];
        g.point[3 * new_npnt + 1] = g.point[3 * i + 1];
        g.point[3 * new_npnt + 2] = g.point[3 * i + 2];
        if (!g.texcoord.empty()) {
          g.texcoord[2 * new_npnt + 0] = g.texcoord[2 * i + 0];
          g.texcoord[2 * new_npnt + 1] = g.texcoord[2 * i + 1];
        }
        pinned[new_npnt] = pinned[i];
        new_npnt++;
      }
      g.point.resize(3 * new_npnt);
      if (!g.texcoord.empty()) g.texcoord.resize(2 * new_npnt);
      pinned.resize(std::max(new_npnt, nnode_early));
      npnt = new_npnt;
    }
  }

  // body mass/inertia matching specs (Make :525)
  const double mass = fc.mass ? *fc.mass : 1.0;
  const double inertiabox = fc.inertiabox ? *fc.inertiabox : 0.005;
  const double bodymass = mass / npnt;
  const double bodyinertia = bodymass * (2.0 * inertiabox * inertiabox) / 3.0;

  // create bodies + vertbody list. Every non-direct point is used. Interpolated
  // vertices all attach to the parent (their dofs come from the nodal mesh).
  std::vector<std::string> vertbody;
  vertbody.reserve(npnt);
  for (int i = 0; i < npnt; i++) {
    if (pinned[i] || interpolated) {
      vertbody.push_back(parent_name);
      continue;
    }
    auto b = std::make_unique<Body>();
    b->name = name + "_" + std::to_string(i);
    b->pos = std::array<double, 3>{g.point[3 * i], g.point[3 * i + 1],
                                   g.point[3 * i + 2]};
    auto in = std::make_unique<Inertial>();
    in->pos = std::array<double, 3>{0, 0, 0};
    in->mass = bodymass;
    in->diaginertia = std::array<double, 3>{bodyinertia, bodyinertia, bodyinertia};
    b->inertial.push_back(std::move(in));
    // per-vertex sliders keyed by dof (Make :571-605). radial = one slider along
    // the (normalized) vertex direction; 2d = two axis-aligned x/y sliders; full
    // = three orthogonal sliders. trilinear/quadratic never reach here (gated).
    auto add_slider = [&](const std::array<double, 3>& axis) {
      auto jnt = std::make_unique<Joint>();
      // upstream flexcomp joints are unnamed; force an empty (present) name so
      // the native collector does not auto-name them (the XML oracle never sees
      // these joints, so they stay unnamed there).
      jnt->name = std::string();
      jnt->type = JointType::slide;
      jnt->pos = std::array<double, 3>{0, 0, 0};
      jnt->axis = axis;
      b->subtree.push_back(BodyChildAny{std::move(jnt)});
    };
    if (dof == FlexDof::radial) {
      double ax[3] = {g.point[3 * i], g.point[3 * i + 1], g.point[3 * i + 2]};
      lift::mjuu_normvec(ax, 3);
      add_slider({ax[0], ax[1], ax[2]});
    } else if (dof == FlexDof::twod) {
      add_slider({1, 0, 0});
      add_slider({0, 1, 0});
    } else {
      for (int j = 0; j < 3; j++) {
        std::array<double, 3> ax{0, 0, 0};
        ax[j] = 1;
        add_slider(ax);
      }
    }
    vertbody.push_back(*b->name);
    // clear flex vertex coords for this (non-centered) point
    g.point[3 * i] = g.point[3 * i + 1] = g.point[3 * i + 2] = 0;
    out_bodies.push_back(BodyChildAny{std::move(b)});
  }

  // ------------------------------------------------------------------------- //
  // Nodal mesh for trilinear/quadratic interpolation (Make :631-782). Node    //
  // bodies (or the parent, for pinned nodes) carry the interpolation dofs.    //
  // ------------------------------------------------------------------------- //
  std::vector<std::string> nodebody;
  std::vector<double> node_local;
  const bool elastic2d_shell =
      !fc.flexElasticitys.empty() && fc.flexElasticitys.front() &&
      fc.flexElasticitys.front()->elastic2d &&
      *fc.flexElasticitys.front()->elastic2d != Elastic2D::none;
  const double mass_total_spec = mass;  // prescribed total mass
  if (interpolated && !rigid) {
    const int cx = cc[0], cy = cc[1], cz = cc[2];
    const int nx = cx * order + 1, ny = cy * order + 1, nz = cz * order + 1;
    const int nnode = nx * ny * nz;
    const int nelem = static_cast<int>(g.element.size()) / (dim + 1);

    // mark empty cells + pin exclusively-empty nodes (volume mode only)
    if (!elastic2d_shell) {
      FlexComputeCellEmpty(g.point.data(), g.element.data(), npnt, nelem, dim,
                           cc, out_cell_empty, minmax);
      for (int gi = 0; gi < nx; gi++)
        for (int gj = 0; gj < ny; gj++)
          for (int gk = 0; gk < nz; gk++) {
            bool all_empty = true;
            int ci_min = std::max(0, gi == 0 ? 0 : (gi - 1) / order);
            int ci_max = std::min(cx - 1, gi / order);
            int cj_min = std::max(0, gj == 0 ? 0 : (gj - 1) / order);
            int cj_max = std::min(cy - 1, gj / order);
            int ck_min = std::max(0, gk == 0 ? 0 : (gk - 1) / order);
            int ck_max = std::min(cz - 1, gk / order);
            for (int ci = ci_min; ci <= ci_max && all_empty; ci++)
              for (int cj = cj_min; cj <= cj_max && all_empty; cj++)
                for (int ck = ck_min; ck <= ck_max && all_empty; ck++)
                  if (!out_cell_empty[ci * cy * cz + cj * cz + ck])
                    all_empty = false;
            if (all_empty) pinned[gi * ny * nz + gj * nz + gk] = 1;
          }
    } else {
      // shell mode: pin all interior (non-boundary) nodes
      for (int gi = 0; gi < nx; gi++)
        for (int gj = 0; gj < ny; gj++)
          for (int gk = 0; gk < nz; gk++)
            if (!(gi == 0 || gi == nx - 1 || gj == 0 || gj == ny - 1 ||
                  gk == 0 || gk == nz - 1))
              pinned[gi * ny * nz + gj * nz + gk] = 1;
    }

    // if any node is pinned, node local positions must be saved
    if (centered)
      for (int i = 0; i < nnode; i++)
        if (pinned[i]) { centered = false; break; }

    node_local.assign(3 * nnode, 0);
    const double massP2[3] = {1. / 6., 2. / 3., 1. / 6.};
    std::vector<Inertial*> node_inertials;
    std::vector<double> node_masses;
    std::vector<double> node_pre_inertia;
    int idx = 0;
    for (int gi = 0; gi < nx; gi++) {
      for (int gj = 0; gj < ny; gj++) {
        for (int gk = 0; gk < nz; gk++) {
          double s = (double)gi / (cx * order);
          double t = (double)gj / (cy * order);
          double u = (double)gk / (cz * order);
          double px = minmax[0] + s * (minmax[3] - minmax[0]);
          double py = minmax[1] + t * (minmax[4] - minmax[1]);
          double pz = minmax[2] + u * (minmax[5] - minmax[2]);
          if (pinned[idx]) {
            node_local[3 * idx + 0] = px;
            node_local[3 * idx + 1] = py;
            node_local[3 * idx + 2] = pz;
            nodebody.push_back(parent_name);
            idx++;
            continue;
          }
          auto pb = std::make_unique<Body>();
          pb->name = name + "_" + std::to_string(gi) + "_" +
                     std::to_string(gj) + "_" + std::to_string(gk);
          pb->pos = std::array<double, 3>{px, py, pz};
          double bmass;
          if (order == 1) {
            bmass = 1.0;
          } else {
            int li = gi % order, lj = gj % order, lk = gk % order;
            int nci = (gi > 0 && gi < nx - 1 && li == 0) ? 2 : 1;
            int ncj = (gj > 0 && gj < ny - 1 && lj == 0) ? 2 : 1;
            int nck = (gk > 0 && gk < nz - 1 && lk == 0) ? 2 : 1;
            double wi = massP2[li == 0 ? 0 : li];
            double wj = massP2[lj == 0 ? 0 : lj];
            double wk = massP2[lk == 0 ? 0 : lk];
            bmass = wi * wj * wk * nci * ncj * nck;
          }
          auto in = std::make_unique<Inertial>();
          in->pos = std::array<double, 3>{0, 0, 0};
          in->mass = bmass;
          double bi = bmass * (2.0 * inertiabox * inertiabox) / 3.0;
          in->diaginertia = std::array<double, 3>{bi, bi, bi};
          node_inertials.push_back(in.get());
          node_masses.push_back(bmass);
          node_pre_inertia.push_back(bi);
          pb->inertial.push_back(std::move(in));
          for (int d = 0; d < 3; d++) {
            auto jnt = std::make_unique<Joint>();
            jnt->name = std::string();
            jnt->type = JointType::slide;
            jnt->pos = std::array<double, 3>{0, 0, 0};
            std::array<double, 3> ax{0, 0, 0};
            ax[d] = 1;
            jnt->axis = ax;
            pb->subtree.push_back(BodyChildAny{std::move(jnt)});
          }
          nodebody.push_back(*pb->name);
          out_bodies.push_back(BodyChildAny{std::move(pb)});
          idx++;
        }
      }
    }

    // normalize masses so the total equals the prescribed mass
    double total_mass = 0;
    for (double m : node_masses) total_mass += m;
    if (total_mass > 0) {
      double scale = mass_total_spec / total_mass;
      for (std::size_t k = 0; k < node_inertials.size(); ++k) {
        node_inertials[k]->mass = node_masses[k] * scale;
        double bi = node_pre_inertia[k] * scale;
        node_inertials[k]->diaginertia = std::array<double, 3>{bi, bi, bi};
      }
    }
  }

  // synthesize the <flex> spec
  auto flex = std::make_unique<Flex>();
  if (!name.empty()) flex->name = name;
  flex->dim = dim;
  if (fc.group) flex->group = *fc.group;
  if (fc.radius) flex->radius = *fc.radius;
  if (fc.material) flex->material = *fc.material;
  if (fc.rgba) flex->rgba = *fc.rgba;
  if (force_flatskin) flex->flatskin = true;
  else if (fc.flatskin) flex->flatskin = *fc.flatskin;

  // vertbody -> body="a b c ..."
  std::string bodies_str;
  for (std::size_t i = 0; i < vertbody.size(); ++i) {
    if (i) bodies_str += ' ';
    bodies_str += vertbody[i];
  }
  flex->body = bodies_str;
  flex->element = g.element;
  if (!g.texcoord.empty()) flex->texcoord = g.texcoord;
  if (!elemtexcoord.empty()) flex->elemtexcoord = elemtexcoord;

  // interpolated: nodebody name list -> "node" attribute; dof drives the order;
  // cellcount is the FE cell count. Node local offsets flow out-of-band (only
  // saved when not centered, Make :779).
  if (interpolated) {
    std::string node_str;
    for (std::size_t i = 0; i < nodebody.size(); ++i) {
      if (i) node_str += ' ';
      node_str += nodebody[i];
    }
    flex->node = node_str;
    flex->dof = dof;
    flex->cellcount = std::array<int32_t, 3>{cc[0], cc[1], cc[2]};
    if (!centered) out_node_local = node_local;
  }

  // vert coords are saved whenever not centered (Make :514/:784) or interpolated,
  // including the rigid case (all points on the parent). Rigid then collapses
  // vertbody to the single parent name (Make :519).
  if (!centered || interpolated) flex->vertex = g.point;
  if (rigid) flex->body = parent_name;

  // contact / edge / elasticity children carry over from the flexcomp def
  if (!fc.flexContacts.empty() && fc.flexContacts.front())
    flex->flexContacts.push_back(Clone(*fc.flexContacts.front()));
  if (!fc.flexElasticitys.empty() && fc.flexElasticitys.front())
    flex->flexElasticitys.push_back(Clone(*fc.flexElasticitys.front()));
  // edge stiffness/damping + equality (from <edge>) (Make :788). equality=edge
  // synthesizes one mjEQ_FLEX, equality=vert one mjEQ_FLEXVERT, strain one
  // mjEQ_FLEXSTRAIN per finite-element cell (volume) or boundary face (shell),
  // each referencing this flex by name; a rigid flex cannot carry one (upstream
  // hard error), so it is suppressed.
  if (!fc.flexcompEdges.empty() && fc.flexcompEdges.front()) {
    const FlexcompEdge& fe = *fc.flexcompEdges.front();
    if (fe.stiffness || fe.damping) {
      auto e = std::make_unique<FlexEdge>();
      e->stiffness = fe.stiffness;
      e->damping = fe.damping;
      flex->flexEdges.push_back(std::move(e));
    }
    const bool is_edge = fe.equality && *fe.equality == FlexEquality::true_;
    const bool is_vert = fe.equality && *fe.equality == FlexEquality::vert;
    const bool is_strain = fe.equality && *fe.equality == FlexEquality::strain;
    auto make_eq = [&]() {
      auto eq = std::make_unique<EqualityFlex>();
      // upstream's flexcomp equality is unnamed; force an empty (present) name so
      // the native name pass does not auto-name it (the XML oracle never sees it).
      eq->name = std::string();
      eq->flex = ps::Ref<Flex>(name);
      eq->active = true;
      eq->solref = fe.solref;
      eq->solimp = fe.solimp;
      return eq;
    };
    if (!rigid && (is_edge || is_vert) && !name.empty()) {
      SynthEquality se;
      se.eq = make_eq();
      se.type = is_vert ? mjEQ_FLEXVERT : mjEQ_FLEX;
      out_eqs.push_back(std::move(se));
    } else if (!rigid && is_strain && !name.empty() && interpolated) {
      const int cx = cc[0], cy = cc[1], cz = cc[2];
      if (elastic2d_shell) {
        int nelem_fe = 2 * (cy * cz + cx * cz + cx * cy);
        for (int f = 0; f < nelem_fe; f++) {
          SynthEquality se;
          se.eq = make_eq();
          se.type = mjEQ_FLEXSTRAIN;
          se.has_data = true;
          se.data[0] = f;
          se.data[1] = -1;  // sentinel: shell mode
          se.data[2] = -1;
          out_eqs.push_back(std::move(se));
        }
      } else {
        for (int ci = 0; ci < cx; ci++)
          for (int cj = 0; cj < cy; cj++)
            for (int ck = 0; ck < cz; ck++) {
              if (!out_cell_empty.empty() &&
                  out_cell_empty[ci * cy * cz + cj * cz + ck])
                continue;
              SynthEquality se;
              se.eq = make_eq();
              se.type = mjEQ_FLEXSTRAIN;
              se.has_data = true;
              se.data[0] = ci;
              se.data[1] = cj;
              se.data[2] = ck;
              out_eqs.push_back(std::move(se));
            }
      }
    }
  }
  (void)cs;
  return flex;
}

void FlattenChildren(const std::vector<BodyChildAny>& subtree,
                     const FrameXform& xf, const CompilerSettings& cs,
                     FramedChildren& out, RepArena* arena, FlexcompSink* sink,
                     const std::string& parent_name,
                     const ps::mjcf::CompileOptions& opts) {
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
          FlattenChildren(f->subtree, sub, cs, out, arena, sink,
                          parent_name, opts);
        }
        break;
      }
      case BodyChildAny::Kind::Replicate: {
        // Expand natively into compile-owned clones, then flatten each instance.
        const auto& rp = std::get<std::unique_ptr<Replicate>>(child.node);
        if (rp && arena) {
          auto owned = std::make_unique<std::vector<BodyChildAny>>(
              ExpandReplicateNode(*rp, cs, opts));
          std::vector<BodyChildAny>* stable = owned.get();
          arena->push_back(std::move(owned));
          FlattenChildren(*stable, xf, cs, out, arena, sink, parent_name,
                          opts);
        }
        break;
      }
      case BodyChildAny::Kind::Flexcomp: {
        // Expand natively into compile-owned bodies (inserted in place) plus a
        // synthesized <flex> (and optional edge equality) routed to the sink,
        // then flatten the bodies.
        const auto& fcn = std::get<std::unique_ptr<Flexcomp>>(child.node);
        if (fcn && arena && sink) {
          auto owned = std::make_unique<std::vector<BodyChildAny>>();
          std::vector<SynthEquality> eqs;
          std::vector<double> node_local;
          std::vector<char> cell_empty;
          std::unique_ptr<Flex> synth = ExpandFlexcompInto(
              *fcn, cs, opts, parent_name, *owned, eqs, node_local, cell_empty);
          if (synth) {
            sink->flexes.push_back(std::move(synth));
            sink->node_local.push_back(std::move(node_local));
            sink->cell_empty.push_back(std::move(cell_empty));
          }
          for (auto& se : eqs) sink->equalities.push_back(std::move(se));
          std::vector<BodyChildAny>* stable = owned.get();
          arena->push_back(std::move(owned));
          FlattenChildren(*stable, xf, cs, out, arena, sink, parent_name,
                          opts);
        }
        break;
      }
      default:
        break;  // PluginRef/Composite/Attach: gated out
    }
  }
}

// --------------------------------------------------------------------------- //
// S1 Collect: DFS pre-order over the body tree, document order == id order.    //
// --------------------------------------------------------------------------- //
class BodyCollector {
 public:
  BodyCollector(const Model& model, const CompilerSettings& cs,
                const ps::mjcf::CompileOptions& opts, double znear,
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
      if (wb)
        FlattenChildren(wb->subtree, identity, cs_, fc, &rep_arena_,
                        &flexcomp_sink_, "world", opts_);

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
    FlattenChildren(b.subtree, identity, cs_, fc, &rep_arena_, &flexcomp_sink_,
                    b.name ? *b.name : std::string(), opts_);

    // has_joints: any <joint>/<freejoint> (frames flattened). Determines weld.
    cb.has_joints = !fc.joints.empty();
    cb.weldid = cb.has_joints ? id : (id == 0 ? 0 : bodies_[cb.parentid].weldid);
    if (b.pos) { cb.pos[0] = (*b.pos)[0]; cb.pos[1] = (*b.pos)[1];
                 cb.pos[2] = (*b.pos)[2]; }
    ResolveQuat(b.quat, cb.quat);
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

    // Free-joint alignment precondition (mjCBody::Compile user_objects.cc:2787):
    // exactly one free joint, no child bodies, and align="true" (or align="auto"
    // with compiler alignfree). When set, phase 1 (below) folds the inertial
    // frame into the body frame and null-transforms child geoms; phase 2
    // transforms sites/cameras/lights.
    bool align_free = false;
    if (fc.joints.size() == 1 && fc.bodies.empty()) {
      const auto& [child, jxf] = fc.joints.front();
      if (child->kind() == BodyChildAny::Kind::FreeJoint) {
        const auto& fj = std::get<std::unique_ptr<FreeJoint>>(child->node);
        const TriState al = fj->align ? *fj->align : TriState::auto_;
        align_free = (al == TriState::true_) ||
                     (al == TriState::auto_ && cs_.alignfree);
      }
    }
    const int sstart = static_cast<int>(sites_.size());
    const int cstart = static_cast<int>(cameras_.size());
    const int lstart = static_cast<int>(lights_.size());

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

    // Free-joint alignment (mjCBody::Compile user_objects.cc:2795-2812 phase 1 +
    // :2877-2897 phase 2). Fold the inertial frame into the body frame so the
    // free joint's qpos0 rest pose sits at the CoM with principal axes, then
    // null-transform the children by the inverse inertial frame. Phase 1 runs
    // after the enclosing-frame accum and before xpos0/BVH (so both see the
    // aligned frame and the nulled inertial frame); phase 2 transforms the
    // already-collected sites/cameras/lights.
    if (align_free) {
      double ipos_inv[3], iquat_inv[4];
      lift::mjuu_frameaccum(cb.pos, cb.quat, cb.ipos, cb.iquat);
      lift::mjuu_frameinvert(ipos_inv, iquat_inv, cb.ipos, cb.iquat);
      lift::mjuu_setvec(cb.ipos, 0, 0, 0);
      lift::mjuu_setvec(cb.iquat, 1, 0, 0, 0);
      for (int j = cb.geomadr; j < cb.geomadr + cb.geomnum; ++j)
        lift::mjuu_frameaccumChild(ipos_inv, iquat_inv, geoms_[j].pos,
                                   geoms_[j].quat);
      // phase 2: sites, cameras, lights of this body.
      for (int j = sstart; j < static_cast<int>(sites_.size()); ++j)
        lift::mjuu_frameaccumChild(ipos_inv, iquat_inv, sites_[j].pos,
                                   sites_[j].quat);
      for (int j = cstart; j < static_cast<int>(cameras_.size()); ++j)
        lift::mjuu_frameaccumChild(ipos_inv, iquat_inv, cameras_[j].pos,
                                   cameras_[j].quat);
      for (int j = lstart; j < static_cast<int>(lights_.size()); ++j) {
        double qunit[4] = {1, 0, 0, 0};
        lift::mjuu_frameaccumChild(ipos_inv, iquat_inv, lights_[j].pos, qunit);
        lift::mjuu_rotVecQuat(lights_[j].dir, lights_[j].dir, iquat_inv);
      }
    }

    // Global rest pose (no joint xform in qpos0), accumulated through the parent
    // (mjCBody::Compile, user_objects.cc:2844-2849). Parent id < id (preorder),
    // so bodies_[parentid] is final; consumed by flex vertex placement.
    {
      const CBody& p = bodies_[cb.parentid];
      lift::mjuu_rotVecQuat(cb.xpos0, cb.pos, p.xquat0);
      lift::mjuu_addtovec(cb.xpos0, p.xpos0, 3);
      lift::mjuu_mulquat(cb.xquat0, p.xquat0, cb.quat);
    }

    // BVH over this body's geoms (mjCBody::ComputeBVH).
    BuildBVH(cb);

    bodies_.push_back(cb);

    // Recurse into child bodies in document order (frames flattened).
    for (const auto& [child, xf] : fc.bodies) Collect(*child, id, xf);
    return id;
  }

  const Model& model_;
  const CompilerSettings& cs_;
  const ps::mjcf::CompileOptions& opts_;
  double znear_ = 0.01;
  const AssetBinds& assets_;
  RepArena rep_arena_;  // owns replicate-expansion clones for the compile's life
  FlexcompSink flexcomp_sink_;  // flexcomp-synthesized flexes + equalities
  std::vector<CBody> bodies_;
  std::vector<CGeom> geoms_;
  std::vector<CJoint> joints_;
  std::vector<CSite> sites_;
  std::vector<CCamera> cameras_;
  std::vector<CLight> lights_;
  int qpos_cursor_ = 0, dof_cursor_ = 0;

 public:
  std::vector<CJoint>& joints() { return joints_; }
  std::vector<std::unique_ptr<Flex>>& synth_flexes() { return flexcomp_sink_.flexes; }
  std::vector<SynthEquality>& synth_equalities() {
    return flexcomp_sink_.equalities;
  }
  const std::vector<std::vector<double>>& synth_node_local() const {
    return flexcomp_sink_.node_local;
  }
  const std::vector<std::vector<char>>& synth_cell_empty() const {
    return flexcomp_sink_.cell_empty;
  }
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
                       const std::vector<CJoint>& joints, int nv,
                       const std::vector<int>& tendon_demote_bodies) {
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

  // Tendon-armature demotion (ComputeSparseSizes user_model.cc:1049 /
  // FinalizeSimple :4048): an inertia-bearing tendon (armature>0) couples the
  // dofs of every body its site/cylinder/sphere wraps touch, so those bodies
  // cannot use the diagonal "simple" fast path. Applied before dof_simplenum/nC.
  for (int b : tendon_demote_bodies)
    if (b >= 0 && b < nbody) t.body_simple[b] = 0;

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

  // tree sleep policy: AUTO for every tree, then each body's non-default <body
  // sleep> policy stamped onto its tree (user_model.cc:3065-3082). A non-AUTO
  // policy is only valid on a movable root body (validated at parse; leg B would
  // reject a misplaced one, so a flippable model always places it legally).
  for (int i = 0; i < dt.ntree; ++i) m->tree_sleep_policy[i] = mjSLEEP_AUTO;
  for (int i = 1; i < nbody; ++i) {
    const CBody& cb = cbs[i];
    if (!cb.src || !cb.src->sleep) continue;
    int policy = mjSLEEP_AUTO;
    switch (*cb.src->sleep) {
      case BodySleep::auto_:   policy = mjSLEEP_AUTO;    break;
      case BodySleep::never:   policy = mjSLEEP_NEVER;   break;
      case BodySleep::allowed: policy = mjSLEEP_ALLOWED; break;
      case BodySleep::init:    policy = mjSLEEP_INIT;    break;
    }
    if (policy != mjSLEEP_AUTO && m->body_treeid[i] >= 0)
      m->tree_sleep_policy[m->body_treeid[i]] = policy;
  }

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
  std::vector<std::string> body, jnt, geom, site, cam, light, flex, mesh, skin,
      hfield, tex, mat, pair, exclude, eq, tendon, actuator, sensor, numeric,
      text, tuple, key, plugin;

  int TotalNames() const {
    int n = static_cast<int>(modelname.size()) + 1;
    auto add = [&](const std::vector<std::string>& v) {
      for (const auto& s : v) n += static_cast<int>(s.size()) + 1;
    };
    add(body); add(jnt); add(geom); add(site); add(cam); add(light);
    add(flex); add(mesh); add(skin); add(hfield); add(tex); add(mat);
    add(pair); add(exclude); add(eq); add(tendon); add(actuator); add(sensor);
    add(numeric); add(text); add(tuple);
    add(key); add(plugin);
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
  pass(nl.flex, m->name_flexadr);
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
  pass(nl.numeric, m->name_numericadr);
  pass(nl.text, m->name_textadr);
  pass(nl.tuple, m->name_tupleadr);
  pass(nl.key, m->name_keyadr);
  pass(nl.plugin, m->name_pluginadr);
}

// Effective name of a nameable element: authored name, else the auto-name the
// XML path would inject (mirrors bridge Collector so name tables match leg B).
template <class E>
std::string EffectiveName(const E& e, const ps::mjcf::CompileOptions& opts) {
  if (e.name) return *e.name;
  if (opts.auto_name) {
    return opts.auto_name_prefix +
           std::string(ps::mjcf::FamilyToken(element_type_of<E>::value)) + ":" +
           std::to_string(e.serial);
  }
  return "";
}

// Arena size heuristic, lifted verbatim from user_model.cc:5219-5252. `memory`
// is the authored <size memory> byte count (-1 when unset) and `nstack` the
// legacy <size nstack> (-1 when unset); both branches match mj_loadXML. When
// memory is set the arena is exactly that many bytes; otherwise the nstack or
// conservative-heuristic branch feeds the pre-arena footprint + megabyte round.
void SetNarena(mjModel* m, long long memory, int nstack) {
  if (memory != -1) {
    // memory size is user-specified in bytes.
    m->narena = static_cast<std::size_t>(memory);
    return;
  }
  const int nconmax = m->nconmax == -1 ? 100 : m->nconmax;
  const int njmax = m->njmax == -1 ? 500 : m->njmax;
  if (nstack != -1) {
    // (legacy) stack size is user-specified as multiple of sizeof(mjtNum).
    m->narena = sizeof(mjtNum) * static_cast<std::size_t>(nstack);
  } else {
    m->narena = sizeof(mjtNum) * static_cast<size_t>(mjMAX(
        1000,
        5 * (njmax + m->neq + m->nv) * (njmax + m->neq + m->nv) +
        20 * (m->nq + m->nv + m->nu + m->na + m->nbody + m->njnt +
              m->ngeom + m->nsite + m->neq + m->ntendon + m->nwrap)));
  }
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
  std::string flex_name;       // referenced flex name (flex equalities)
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

// Compile a flex equality referencing a flex by name (mjCEquality::
// ResolveReferences flex branch). `type` is the mjtEq kind (mjEQ_FLEX for an
// edge equality, mjEQ_FLEXVERT for a vert equality); obj2 is unused (-1).
CEquality FlexEqualityCompile(const ps::sdk::detail::DefaultIndex& idx,
                              const EqualityFlex& e, const NameIdMap& flexid,
                              int type, const double* strain_data = nullptr) {
  CEquality ce;
  ce.src = &e;
  ce.serial = e.serial;
  ce.name = e.name;
  ce.type = type;       // objtype stays 0: the reader sets eq_objtype only for
                        // connect/weld (flex resolves object_type locally).
  ce.flex_name = e.flex ? e.flex->name : std::string();
  auto it = flexid.find(ce.flex_name);
  ce.obj1id = it == flexid.end() ? -1 : it->second;
  ce.obj2id = -1;
  MergeEqualityClassChain(idx, e.dclass ? e.dclass->name : "", e.active,
                          e.solref, e.solimp, ce);
  // strain: one constraint per FE cell; data[0..2] carries the cell coords
  // (volume) or [fe,-1,-1] (shell). The remaining data entries keep defaults.
  if (strain_data)
    for (int k = 0; k < 3; ++k) ce.data[k] = strain_data[k];
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
  ps::opt<std::array<double, 2>> springlength;
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
  // springlength (canonical two-value pair)
  if (a.springlength) { ct.springlength[0] = (*a.springlength)[0];
                        ct.springlength[1] = (*a.springlength)[1]; }
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
  int nsample = 0, interp = 0;
  double delay = 0;
  bool is_plugin = false;  // <plugin> actuator: force computed by a plugin
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
  // Delay/history buffer (mjCActuator: nsample rings, interp order, delay time).
  if constexpr (requires { eff.nsample; }) if (eff.nsample) ca.nsample = *eff.nsample;
  if constexpr (requires { eff.interp; }) if (eff.interp) ca.interp = static_cast<int>(*eff.interp);
  if constexpr (requires { eff.delay; }) if (eff.delay) ca.delay = *eff.delay;
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
                         const NameIdMap& tendonid, const NameIdMap& bodyid,
                         const NameIdMap& siteid) {
  auto id = [](const NameIdMap& m, const std::string& n) {
    auto it = m.find(n); return it == m.end() ? -1 : it->second;
  };
  if (eff.joint) { ca.trntype = mjTRN_JOINT; ca.trnid[0] = id(jointid, eff.joint->name); }
  else if (eff.jointinparent) { ca.trntype = mjTRN_JOINTINPARENT;
                                ca.trnid[0] = id(jointid, eff.jointinparent->name); }
  else if (eff.tendon) { ca.trntype = mjTRN_TENDON; ca.trnid[0] = id(tendonid, eff.tendon->name); }
  else {
    // slidercrank (cranksite) > site (+refsite) > body -- reader precedence
    // (OneActuator, xml_native_reader.cc:2417-2454). trnid[1] is the refsite /
    // slidersite id (else -1).
    bool set = false;
    if constexpr (requires { eff.cranksite; }) {
      if (eff.cranksite) {
        ca.trntype = mjTRN_SLIDERCRANK;
        ca.trnid[0] = id(siteid, eff.cranksite->name);
        if constexpr (requires { eff.slidersite; })
          if (eff.slidersite) ca.trnid[1] = id(siteid, eff.slidersite->name);
        set = true;
      }
    }
    if constexpr (requires { eff.site; }) {
      if (!set && eff.site) {
        ca.trntype = mjTRN_SITE;
        ca.trnid[0] = id(siteid, eff.site->name);
        if constexpr (requires { eff.refsite; })
          if (eff.refsite) ca.trnid[1] = id(siteid, eff.refsite->name);
        set = true;
      }
    }
    if constexpr (requires { eff.body; }) {
      if (!set && eff.body) {
        ca.trntype = mjTRN_BODY; ca.trnid[0] = id(bodyid, eff.body->name);
      }
    }
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
                          const NameIdMap& bodyid, const NameIdMap& siteid,
                          const RangeLookup& rl) {
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
      ResolveTransmission(*eff, ca, jointid, tendonid, bodyid, siteid);
      break;
    }
    case ActuatorAny::Kind::Motor: {
      const Motor& e = *std::get<std::unique_ptr<Motor>>(any.node);
      std::unique_ptr<Motor> eff = ps::sdk::Effective(model, e);
      ca.src = &e; ca.serial = e.serial; ca.name = e.name;
      FillActuatorCommon(*eff, ca, ap);
      actlower::SetToMotor(ap);
      ResolveTransmission(*eff, ca, jointid, tendonid, bodyid, siteid);
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
        ResolveTransmission(eff, ca, jointid, tendonid, bodyid, siteid);
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
      ResolveTransmission(*eff, ca, jointid, tendonid, bodyid, siteid);
      break;
    }
    case ActuatorAny::Kind::Damper: {
      const Damper& e = *std::get<std::unique_ptr<Damper>>(any.node);
      std::unique_ptr<Damper> eff = ps::sdk::Effective(model, e);
      ca.src = &e; ca.serial = e.serial; ca.name = e.name;
      FillActuatorCommon(*eff, ca, ap);
      actlower::SetToDamper(ap, eff->kv ? *eff->kv : 0);
      ResolveTransmission(*eff, ca, jointid, tendonid, bodyid, siteid);
      break;
    }
    case ActuatorAny::Kind::Cylinder: {
      const Cylinder& e = *std::get<std::unique_ptr<Cylinder>>(any.node);
      std::unique_ptr<Cylinder> eff = ps::sdk::Effective(model, e);
      ca.src = &e; ca.serial = e.serial; ca.name = e.name;
      FillActuatorCommon(*eff, ca, ap);
      double timeconst = eff->timeconst ? *eff->timeconst : ap.dynprm[0];
      double bias = eff->bias ? (*eff->bias)[0] : ap.biasprm[0];
      // area is canonical (a `diameter` spelling is folded to pi/4 d^2 at read),
      // so pass diameter=-1 to leave the area untouched inside SetToCylinder.
      double area = eff->area ? *eff->area : ap.gainprm[0];
      actlower::SetToCylinder(ap, timeconst, bias, area, -1);
      ResolveTransmission(*eff, ca, jointid, tendonid, bodyid, siteid);
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
      ResolveTransmission(*eff, ca, jointid, tendonid, bodyid, siteid);
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
    case ActuatorAny::Kind::ActuatorPlugin: {
      const ActuatorPlugin& e = *std::get<std::unique_ptr<ActuatorPlugin>>(any.node);
      std::unique_ptr<ActuatorPlugin> eff = ps::sdk::Effective(model, e);
      ca.src = &e; ca.serial = e.serial; ca.name = e.name;
      ca.is_plugin = true;
      FillActuatorCommon(*eff, ca, ap);
      if (eff->dyntype) ap.dyntype = static_cast<int>(*eff->dyntype);
      if (eff->actearly) ap.actearly = *eff->actearly ? 1 : 0;
      if (eff->dynprm) for (std::size_t k = 0; k < eff->dynprm->size() && k < mjNDYN; ++k)
        ap.dynprm[k] = (*eff->dynprm)[k];
      if (eff->actrange) { ca.actrange[0] = (*eff->actrange)[0]; ca.actrange[1] = (*eff->actrange)[1];
                           ap.actrange[0] = ca.actrange[0]; ap.actrange[1] = ca.actrange[1]; }
      if (eff->actlimited) ap.actlimited = static_cast<int>(*eff->actlimited);
      if (eff->actdim) ap.actdim = *eff->actdim;
      ResolveTransmission(*eff, ca, jointid, tendonid, bodyid, siteid);
      break;
    }
    default:
      break;  // dcmotor gated out
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
// mj_mergeChain (engine_core_util.c:55): the sorted union of the dof chains of
// two welded bodies. flg_skipcommon stops at the first shared ancestor dof (the
// relative chain, for site/refsite); else it climbs to the world (slidercrank).
// A dofless welded body (world after body_weldid) contributes da = -1.
int MergeChain(const std::vector<CBody>& cbs, const std::vector<int>& dof_parentid,
               int b1, int b2, bool skipcommon, std::vector<int>& chain) {
  b1 = cbs[b1].weldid;
  b2 = cbs[b2].weldid;
  if (b1 == 0 && b2 == 0) return 0;
  int da1 = cbs[b1].dofnum ? cbs[b1].dofadr + cbs[b1].dofnum - 1 : -1;
  int da2 = cbs[b2].dofnum ? cbs[b2].dofadr + cbs[b2].dofnum - 1 : -1;
  int NV = 0;
  while (da1 >= 0 || da2 >= 0) {
    const int da = std::max(da1, da2);
    if (skipcommon && da1 == da && da2 == da) break;
    chain[NV] = da;
    if (da1 == da) da1 = dof_parentid[da1];
    if (da2 == da) da2 = dof_parentid[da2];
    ++NV;
  }
  for (int i = 0; i < NV / 2; ++i) std::swap(chain[i], chain[NV - i - 1]);
  return NV;
}

int ComputeNJmom(const std::vector<CActuator>& acts,
                 const std::vector<CJoint>& joints,
                 const std::vector<CTendon>& tendons,
                 const std::vector<CBody>& cbs, const std::vector<CGeom>& geoms,
                 const std::vector<CSite>& sites,
                 const std::vector<int>& dof_parentid, int nv) {
  int count = 0;
  std::vector<int> chain(nv);
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
      case mjTRN_SITE: {
        // mergeChain(site body, refsite body | world, skipcommon=1).
        const int sb = a.trnid[0] >= 0 ? sites[a.trnid[0]].bodyid : 0;
        const int rb = a.trnid[1] >= 0 ? sites[a.trnid[1]].bodyid : 0;
        count += MergeChain(cbs, dof_parentid, sb, rb, true, chain);
        break;
      }
      case mjTRN_SLIDERCRANK: {
        // mergeChain(crank body, slider body, skipcommon=0).
        const int sb = a.trnid[0] >= 0 ? sites[a.trnid[0]].bodyid : 0;
        const int slb = a.trnid[1] >= 0 ? sites[a.trnid[1]].bodyid : 0;
        count += MergeChain(cbs, dof_parentid, sb, slb, false, chain);
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

void FillActuators(mjModel* m, const std::vector<CActuator>& acts,
                   int& delay_adr) {
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
    m->actuator_delay[i] = a.delay;
    m->actuator_history[2 * i] = a.nsample;
    m->actuator_history[2 * i + 1] = a.interp;
    if (a.nsample > 0) {
      m->actuator_historyadr[i] = delay_adr;
      delay_adr += 2 + 2 * a.nsample;  // [user, cursor, times, values]
    } else {
      m->actuator_historyadr[i] = -1;
    }
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

// Per-field element counts for a rangefinder dataspec (mjRAYDATA_SIZE,
// engine_support.c) and a contact dataspec (mjCONDATA_SIZE); sum over set bits.
constexpr int kRayDataSize[mjNRAYDATA] = {1, 3, 3, 3, 3, 1};
int RaydataSize(int dataspec) {
  int n = 0;
  for (int i = 0; i < mjNRAYDATA; ++i)
    if (dataspec & (1 << i)) n += kRayDataSize[i];
  return n;
}
constexpr int kConDataSize[mjNCONDATA] = {1, 3, 3, 1, 3, 3, 3};
int CondataSize(int dataspec) {
  int n = 0;
  for (int i = 0; i < mjNCONDATA; ++i)
    if (dataspec & (1 << i)) n += kConDataSize[i];
  return n;
}

struct CSensor {
  const void* src = nullptr;
  std::uint64_t serial = 0;
  ps::opt<std::string> name;
  int type = 0;
  int datatype = mjDATATYPE_REAL, needstage = mjSTAGE_POS, dim = 0;
  int objtype = mjOBJ_UNKNOWN, objid = -1;
  int reftype = mjOBJ_UNKNOWN, refid = -1;
  int intprm[mjNSENS] = {0, 0, 0};
  double cutoff = 0, noise = 0;
  double interval[2] = {0, 0};
  int nsample = 0, interp = 0;
  double delay = 0;
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
  const NameIdMap* mesh = nullptr;                    // tactile obj
  const std::unordered_map<std::string, int>* meshnvert = nullptr;  // tactile dim
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
    case mjOBJ_MESH: m = sm.mesh; break;
    default: return -1;
  }
  if (!m || name.empty()) return -1;
  auto it = m->find(name);
  return it == m->end() ? -1 : it->second;
}

// A camera-target rangefinder casts one ray per camera pixel, so its dim scales
// with the camera resolution (mjs_sensorDim user_api.cc:1758). Resolve the named
// camera's effective resolution (default 1x1, mjs_defaultCamera).
void CameraResolutionByName(const Model& m, const std::string& name, int res[2]) {
  res[0] = res[1] = 1;
  const Camera* found = nullptr;
  std::function<void(const std::vector<BodyChildAny>&)> walk =
      [&](const std::vector<BodyChildAny>& sub) {
        for (const auto& c : sub) {
          if (found) return;
          switch (c.kind()) {
            case BodyChildAny::Kind::Camera: {
              const auto& cam = std::get<std::unique_ptr<Camera>>(c.node);
              if (cam && cam->name && *cam->name == name) found = cam.get();
              break;
            }
            case BodyChildAny::Kind::Body:
              walk(std::get<std::unique_ptr<Body>>(c.node)->subtree); break;
            case BodyChildAny::Kind::Frame:
              walk(std::get<std::unique_ptr<Frame>>(c.node)->subtree); break;
            case BodyChildAny::Kind::Replicate:
              walk(std::get<std::unique_ptr<Replicate>>(c.node)->subtree); break;
            default: break;
          }
        }
      };
  for (const auto& wb : m.worldbody)
    if (wb && !found) walk(wb->subtree);
  if (!found) return;
  std::unique_ptr<Camera> eff = ps::sdk::Effective(m, *found);
  if (eff->resolution) { res[0] = (*eff->resolution)[0]; res[1] = (*eff->resolution)[1]; }
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
  // distance/normal/fromto: exactly one of geom1/body1 (obj) and geom2/body2
  // (ref); objtype/reftype follow which attr was authored (reader semantics).
  auto geom_pair_sensor = [&](auto& e, int t) {
    cs.type = t;
    if (e.geom1) { cs.objtype = mjOBJ_GEOM; objname = e.geom1->name; }
    else if (e.body1) { cs.objtype = mjOBJ_BODY; objname = e.body1->name; }
    if (e.geom2) { cs.reftype = mjOBJ_GEOM; refname = e.geom2->name; }
    else if (e.body2) { cs.reftype = mjOBJ_BODY; refname = e.body2->name; }
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
    case K::Rangefinder: { auto& e=get(Rangefinder{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     cs.type=mjSENS_RANGEFINDER;
                     if (e.site) { cs.objtype=mjOBJ_SITE; objname=e.site->name; }
                     else if (e.camera) { cs.objtype=mjOBJ_CAMERA; objname=e.camera->name; }
                     // dataspec bitmask in intprm[0] (default 1<<mjRAYDATA_DIST).
                     { int spec=0;
                       if (e.data && !e.data->empty())
                         for (RayData rd : *e.data) spec |= (1 << static_cast<int>(rd));
                       else spec = 1 << mjRAYDATA_DIST;
                       cs.intprm[0]=spec; }
                     cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Camprojection: { auto& e=get(Camprojection{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     cs.type=mjSENS_CAMPROJECTION; cs.objtype=mjOBJ_SITE;
                     if (e.site) objname=e.site->name;
                     cs.reftype=mjOBJ_CAMERA; if (e.camera) refname=e.camera->name;
                     cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Insidesite: { auto& e=get(Insidesite{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     cs.type=mjSENS_INSIDESITE;
                     if (e.objtype) cs.objtype=mju_str2Type(e.objtype->c_str());
                     if (e.objname) objname=*e.objname;
                     cs.reftype=mjOBJ_SITE; if (e.site) refname=e.site->name;
                     cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Distance: { auto& e=get(Distance{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     geom_pair_sensor(e, mjSENS_GEOMDIST);
                     cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Normal: { auto& e=get(Normal{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     geom_pair_sensor(e, mjSENS_GEOMNORMAL);
                     cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Fromto: { auto& e=get(Fromto{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     geom_pair_sensor(e, mjSENS_GEOMFROMTO);
                     cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::SensorContact: { auto& e=get(SensorContact{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     cs.type=mjSENS_CONTACT;
                     // first match criterion -> objtype/objname; second -> reftype/refname.
                     if (e.site) { cs.objtype=mjOBJ_SITE; objname=e.site->name; }
                     else if (e.body1) { cs.objtype=mjOBJ_BODY; objname=e.body1->name; }
                     else if (e.subtree1) { cs.objtype=mjOBJ_XBODY; objname=e.subtree1->name; }
                     else if (e.geom1) { cs.objtype=mjOBJ_GEOM; objname=e.geom1->name; }
                     if (e.body2) { cs.reftype=mjOBJ_BODY; refname=e.body2->name; }
                     else if (e.subtree2) { cs.reftype=mjOBJ_XBODY; refname=e.subtree2->name; }
                     else if (e.geom2) { cs.reftype=mjOBJ_GEOM; refname=e.geom2->name; }
                     { int spec=0;
                       if (e.data && !e.data->empty())
                         for (ContactData cd : *e.data) spec |= (1 << static_cast<int>(cd));
                       else spec = 1 << mjCONDATA_FOUND;
                       cs.intprm[0]=spec; }
                     cs.intprm[1] = e.reduce ? static_cast<int>(*e.reduce) : 0;
                     cs.intprm[2] = e.num ? *e.num : 1;
                     cs.cutoff=e.cutoff?*e.cutoff:0; cs.noise=e.noise?*e.noise:0;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::Tactile: { auto& e=get(Tactile{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     // obj = mesh, ref = geom (mjCSensor::Compile user_objects.cc:7915).
                     cs.type=mjSENS_TACTILE;
                     cs.objtype=mjOBJ_MESH; if (e.mesh) objname=e.mesh->name;
                     cs.reftype=mjOBJ_GEOM; if (e.geom) refname=e.geom->name;
                     if(e.user)cs.user=*e.user; if(e.interval){cs.interval[0]=(*e.interval)[0];cs.interval[1]=(*e.interval)[1];} break; }
    case K::SensorPlugin: { auto& e=get(SensorPlugin{}); cs.src=&e; cs.serial=e.serial; cs.name=e.name;
                     cs.type=mjSENS_PLUGIN;
                     // Typed obj/ref by explicit string (mju_str2Type); dim=0 and
                     // needstage are filled post-alloc from the plugin decl.
                     if (e.objtype) { cs.objtype = mju_str2Type(e.objtype->c_str());
                                      objname = e.objname ? *e.objname : std::string(); }
                     if (e.reftype) { cs.reftype = mju_str2Type(e.reftype->c_str());
                                      refname = e.refname ? *e.refname : std::string(); }
                     cs.cutoff=e.cutoff?*e.cutoff:0; if(e.user)cs.user=*e.user; break; }
    default:
      break;  // gated types (user/tactile)
  }
  // Delay/history buffer (nsample rings, interp order, delay time) is common to
  // every sensor family; read it generically from the source node.
  std::visit([&](const auto& p) {
    if (!p) return;
    const auto& e = *p;
    if constexpr (requires { e.nsample; }) if (e.nsample) cs.nsample = *e.nsample;
    if constexpr (requires { e.interp; }) if (e.interp) cs.interp = static_cast<int>(*e.interp);
    if constexpr (requires { e.delay; }) if (e.delay) cs.delay = *e.delay;
  }, node);
  (void)model;
  cs.objid = ResolveObj(sm, cs.objtype, objname);
  cs.refid = ResolveObj(sm, cs.reftype, refname);
  cs.datatype = SensorDatatype(cs.type);
  cs.needstage = SensorNeedstage(cs.type);
  if (cs.type == mjSENS_RANGEFINDER) {
    int num_rays = 1;
    if (cs.objtype == mjOBJ_CAMERA) {
      int res[2];
      CameraResolutionByName(model, objname, res);
      num_rays = res[0] * res[1];
    }
    cs.dim = RaydataSize(cs.intprm[0]) * num_rays;
  }
  else if (cs.type == mjSENS_CONTACT)
    cs.dim = cs.intprm[2] * CondataSize(cs.intprm[0]);
  else if (cs.type == mjSENS_TACTILE) {
    // dim = 3 * nvert of the referenced mesh (mjs_sensorDim user_api.cc:1753).
    int nvert = 0;
    if (sm.meshnvert) {
      auto it = sm.meshnvert->find(objname);
      if (it != sm.meshnvert->end()) nvert = it->second;
    }
    cs.dim = 3 * nvert;
  } else
    cs.dim = SensorDim(cs.type);
  return cs;
}

void FillSensors(mjModel* m, const std::vector<CSensor>& sensors,
                 int& delay_adr) {
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
    for (int k = 0; k < mjNSENS; ++k)
      m->sensor_intprm[i * mjNSENS + k] = s.intprm[k];
    m->sensor_dim[i] = s.dim;
    m->sensor_cutoff[i] = s.cutoff;
    m->sensor_noise[i] = s.noise;
    m->sensor_delay[i] = s.delay;
    m->sensor_history[2 * i] = s.nsample;
    m->sensor_history[2 * i + 1] = s.interp;
    m->sensor_interval[2 * i] = s.interval[0];
    m->sensor_interval[2 * i + 1] = s.interval[1];
    if (s.nsample > 0) {
      m->sensor_historyadr[i] = delay_adr;
      delay_adr += 2 + s.nsample + s.nsample * s.dim;  // [user, cursor, times, values]
    } else {
      m->sensor_historyadr[i] = -1;
    }
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
  // A <size nkey> pad slot (index >= keys.size()) has no source Key: it defaults
  // wholesale, exactly as an authored empty key would (AddKey). Use a shared
  // default Key so the existing prefix-pad logic produces the reference pose.
  static const Key kDefaultKey{};
  for (int i = 0; i < m->nkey; ++i) {
    const Key& k = i < static_cast<int>(keys.size()) ? *keys[i] : kDefaultKey;
    m->key_time[i] = k.time ? *k.time : 0;

    // qpos: authored prefix (pad tail with qpos0); empty -> all qpos0.
    {
      const int n0 = (k.qpos && !k.qpos->empty())
                         ? std::min(static_cast<int>(k.qpos->size()), nq)
                         : 0;
      // A NaN slot is an attach-imported keyframe's gap (a dof not owned by the
      // grafted child): it defaults to qpos0, exactly as mjCModel::RestoreState
      // uses pos0 for a joint the child keyframe did not define (user_model.cc
      // :4156-4161). Authored keyframes never carry NaN.
      for (int j = 0; j < n0; ++j)
        m->key_qpos[i * nq + j] =
            std::isnan((*k.qpos)[j]) ? m->qpos0[j] : (*k.qpos)[j];
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
// Custom fields (S8/CopyObjects): numeric / text / tuple. Numeric size padding  //
// is lifted from mjCNumeric::Compile (user_objects.cc:8024): an unauthored size  //
// takes the data length, an authored size zero-pads the tail. Text carries its   //
// string + a NUL. Tuple obj refs resolve by (objtype,name) exactly as            //
// mjCTuple::ResolveReferences (user_objects.cc:8214) -> FindObject, storing the   //
// raw mjtObj and the compiled id; the objtype string is mapped with the public   //
// mju_str2Type. Purity: reads the const Custom tree, writes only the C-structs.  //
// --------------------------------------------------------------------------- //
struct CNumeric {
  std::string name;
  int size = 0;
  std::vector<double> data;  // authored prefix (length <= size); tail zero-pads
};

struct CText {
  std::string name;
  std::string data;
};

struct CTupleEntry {
  int objtype = 0;  // raw mjtObj (mju_str2Type)
  int objid = -1;
  double prm = 0;
};

struct CTuple {
  std::string name;
  std::vector<CTupleEntry> entries;
};

// Numeric data is canonical: the reader already materialized it to the authored
// size (zero-padded/truncated, Q-NUM Wave B #8), so size == data length here and
// the data is copied verbatim.
CNumeric NumericCompile(const Numeric& n, const ps::mjcf::CompileOptions& opts) {
  CNumeric cn;
  cn.name = EffectiveName(n, opts);
  cn.size = n.data ? static_cast<int>(n.data->size()) : 0;
  if (n.data) cn.data = *n.data;
  return cn;
}

CText TextCompile(const Text& t, const ps::mjcf::CompileOptions& opts) {
  CText ct;
  ct.name = EffectiveName(t, opts);
  ct.data = t.data ? *t.data : std::string();
  return ct;
}

void FillNumerics(mjModel* m, const std::vector<CNumeric>& nums) {
  int adr = 0;
  for (int i = 0; i < static_cast<int>(nums.size()); ++i) {
    m->numeric_adr[i] = adr;
    m->numeric_size[i] = nums[i].size;
    const int have = static_cast<int>(nums[i].data.size());
    for (int j = 0; j < have; ++j) m->numeric_data[adr + j] = nums[i].data[j];
    for (int j = have; j < nums[i].size; ++j) m->numeric_data[adr + j] = 0;
    adr += nums[i].size;
  }
}

void FillTexts(mjModel* m, const std::vector<CText>& texts) {
  int adr = 0;
  for (int i = 0; i < static_cast<int>(texts.size()); ++i) {
    m->text_adr[i] = adr;
    m->text_size[i] = static_cast<int>(texts[i].data.size()) + 1;
    mju_strncpy(m->text_data + adr, texts[i].data.c_str(), m->ntextdata - adr);
    adr += m->text_size[i];
  }
}

void FillTuples(mjModel* m, const std::vector<CTuple>& tuples) {
  int adr = 0;
  for (int i = 0; i < static_cast<int>(tuples.size()); ++i) {
    m->tuple_adr[i] = adr;
    m->tuple_size[i] = static_cast<int>(tuples[i].entries.size());
    for (int j = 0; j < m->tuple_size[i]; ++j) {
      m->tuple_objtype[adr + j] = tuples[i].entries[j].objtype;
      m->tuple_objid[adr + j] = tuples[i].entries[j].objid;
      m->tuple_objprm[adr + j] = tuples[i].entries[j].prm;
    }
    adr += m->tuple_size[i];
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
  std::string listname;  // name-table entry: authored name, else file stem
  int type = 1;         // mjtTexture (TextureType casts directly)
  int colorspace = 0;   // mjtColorSpace
  int width = 0, height = 0, nchannel = 3;
  std::vector<unsigned char> data;
  std::string file;     // File() for m->paths (empty: builtin/user texture)
};

// --------------------------------------------------------------------------- //
// File textures (mjCTexture::Load2D/LoadCubeSingle + LoadPNG/LoadKTX/LoadCustom //
// + FlipIfNeeded, user_objects.cc:5221-5541). PNG decode is the lifted lodepng  //
// wrapper (lifted::DecodePNG); KTX/custom parsing and the 2D/cube composition   //
// are retargeted plumbing over a plain byte buffer. Cube-from-separate-files    //
// (LoadCubeSeparate) and an authored content_type stay gated to the XML         //
// fallback (native.cc CheckTexture). Face symbol order R,L,U,D,F,B = 0..5.      //
// --------------------------------------------------------------------------- //

std::string MeshCombine(const std::string& dir, const std::string& file);

// mjuu_extToContentType, image branch (user_util.cc:1033): extension -> MIME.
std::string TexExtContentType(const std::string& file) {
  std::size_t dot = file.find_last_of('.');
  std::string ext = dot == std::string::npos ? "" : file.substr(dot);
  for (char& c : ext) c = static_cast<char>(std::tolower(c));
  if (ext == ".png") return "image/png";
  if (ext == ".ktx") return "image/ktx";
  return "image/vnd.mujoco.texture";  // LoadFlip's empty->custom fallback
}

// mjCTexture::FlipIfNeeded (user_objects.cc:5305-5338). In-place channel swap.
void TexFlipIfNeeded(std::vector<unsigned char>& image, int w, int h,
                     int nchannel, bool hflip, bool vflip) {
  if (hflip) {
    for (int r = 0; r < h; r++)
      for (int c = 0; c < w / 2; c++) {
        int c1 = w - 1 - c;
        int val1 = nchannel * (r * w + c);
        int val2 = nchannel * (r * w + c1);
        for (int ch = 0; ch < nchannel; ch++)
          std::swap(image[val1 + ch], image[val2 + ch]);
      }
  }
  if (vflip) {
    for (int r = 0; r < h / 2; r++)
      for (int c = 0; c < w; c++) {
        int r1 = h - 1 - r;
        int val1 = nchannel * (r * w + c);
        int val2 = nchannel * (r1 * w + c);
        for (int ch = 0; ch < nchannel; ch++)
          std::swap(image[val1 + ch], image[val2 + ch]);
      }
  }
}

// mjCTexture::LoadKTX (user_objects.cc:5245): raw bytes, w=size, h=1, nchannel=1.
// mjCTexture::LoadCustom (:5267): [w:int][h:int][w*h*3 bytes].
// mjCTexture::LoadPNG (:5221) via the lifted decoder. `nchannel` is in/out
// (KTX forces 1). Resolves texturedir + strippath, opens the resource, dispatch
// by content type, then FlipIfNeeded. Returns false + diagnostic on any error.
bool TexLoadFile(const CompilerSettings& cs, const std::string& base_dir,
                 const std::string& file, int& nchannel, bool hflip, bool vflip,
                 std::vector<unsigned char>& image, int& w, int& h,
                 bool& is_srgb, const ps::SourceLoc& loc,
                 std::vector<ps::mjcf::Diagnostic>& diags) {
  const std::string asset_type = TexExtContentType(file);
  const std::string combined = MeshCombine(cs.texturedir, file);
  char err[1024] = {0};
  mjResource* res = mju_openResource(base_dir.empty() ? nullptr : base_dir.c_str(),
                                     combined.c_str(), nullptr, err, sizeof(err));
  if (!res) {
    diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "texture",
                     std::string("could not open texture file '") + combined + "'",
                     loc});
    return false;
  }
  const void* bytes = nullptr;
  int n = mju_readResource(res, &bytes);
  auto fail = [&](const std::string& msg) {
    mju_closeResource(res);
    diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "texture", msg, loc});
    return false;
  };
  if (n < 0) return fail("could not read texture file '" + combined + "'");
  if (n == 0) return fail("texture file is empty: '" + combined + "'");

  const unsigned char* buf = static_cast<const unsigned char*>(bytes);
  if (asset_type == "image/png") {
    std::string derr;
    if (!lift::DecodePNG(buf, n, nchannel, image, w, h, is_srgb, derr))
      return fail(derr + " '" + combined + "'");
  } else if (asset_type == "image/ktx") {
    if (hflip || vflip) return fail("cannot flip KTX textures");
    w = n; h = 1; nchannel = 1; is_srgb = false;
    image.assign(buf, buf + n);
  } else {  // custom binary
    if (n < static_cast<int>(2 * sizeof(int)))
      return fail("Non-PNG texture, unexpected file size in file '" + combined + "'");
    const int* pint = reinterpret_cast<const int*>(buf);
    w = pint[0]; h = pint[1]; is_srgb = false;
    if (w < 1 || h < 1)
      return fail("Non-PNG texture, non-positive dimensions in file '" + combined + "'");
    if (n != static_cast<int>(2 * sizeof(int)) + w * h * 3)
      return fail("Non-PNG texture, unexpected file size in file '" + combined + "'");
    image.assign(buf + 2 * sizeof(int), buf + n);
  }
  mju_closeResource(res);
  TexFlipIfNeeded(image, w, h, nchannel, hflip, vflip);
  return true;
}

// mjCTexture::Compile (user_objects.cc:5632-5729). Builtin path lifted above;
// the file path (single-file 2D / cube) resolves texturedir + strippath, decodes
// (PNG/KTX/custom), flips, and resolves colorspace=AUTO from the PNG sRGB chunk.
// Cube-from-separate-files and an authored content_type stay gated (native.cc).
bool TextureCompile(const Model& model, const Texture& tx, const CompilerSettings& cs,
                    const std::string& base_dir, CTexture& out,
                    std::vector<ps::mjcf::Diagnostic>& diags) {
  std::unique_ptr<Texture> eff = ps::sdk::Effective(model, tx);
  out.src = &tx;
  out.name = tx.name;
  out.type = eff->type ? static_cast<int>(*eff->type) : 1;
  out.colorspace = eff->colorspace ? static_cast<int>(*eff->colorspace) : 0;
  out.nchannel = eff->nchannel ? *eff->nchannel : 3;

  const TexFile* tf =
      eff->source ? std::get_if<TexFile>(&*eff->source) : nullptr;

  // Name table entry (mjCTexture::CopyFromSpec :4947): authored name, else the
  // file stem (strippath + stripext), independent of compiler strippath.
  if (tx.name) {
    out.listname = *tx.name;
  } else if (tf && !tf->file.empty()) {
    std::string stem = MeshStripPath(tf->file);
    std::size_t dot = stem.find_last_of('.');
    out.listname = dot == std::string::npos ? stem : stem.substr(0, dot);
  }
  const TextureBuiltin* bi =
      eff->source ? std::get_if<TextureBuiltin>(&*eff->source) : nullptr;

  // ----- single-file texture (2D or cube from one image) -----
  if (tf && !tf->file.empty()) {
    std::string file = tf->file;
    if (cs.strippath) file = MeshStripPath(file);
    out.file = file;
    const bool hflip = eff->hflip && *eff->hflip;
    const bool vflip = eff->vflip && *eff->vflip;

    std::vector<unsigned char> img;
    int w = 0, h = 0;
    bool is_srgb = false;
    if (!TexLoadFile(cs, base_dir, file, out.nchannel, hflip, vflip, img, w, h,
                     is_srgb, tx.loc, diags))
      return false;
    if (out.colorspace == 0 /* AUTO */)
      out.colorspace = is_srgb ? 2 /* SRGB */ : 1 /* LINEAR */;

    if (out.type == 0 /* 2D */) {                       // Load2D
      out.width = w;
      out.height = h;
      out.data = std::move(img);
      return true;
    }

    // LoadCubeSingle: repeated or grid layout (mjCTexture::LoadCubeSingle).
    int gs0 = 1, gs1 = 1;
    if (eff->gridsize) { gs0 = (*eff->gridsize)[0]; gs1 = (*eff->gridsize)[1]; }
    if (gs0 < 1 || gs1 < 1 || gs0 * gs1 > 12) {
      diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "texture",
                       "gridsize must be non-zero and no more than 12 squares "
                       "in texture", tx.loc});
      return false;
    }
    if (w / gs1 != h / gs0 || (w % gs1) || (h % gs0)) {
      diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "texture",
                       "PNG size must be integer multiple of gridsize in texture",
                       tx.loc});
      return false;
    }
    double rgb1[3] = {0.8, 0.8, 0.8};
    if (eff->rgb1) for (int k = 0; k < 3; ++k) rgb1[k] = (*eff->rgb1)[k];
    if (gs0 == 1 && gs1 == 1) {
      out.width = out.height = w;
      out.data.assign(static_cast<std::size_t>(3) * w * w, 0);
      std::memcpy(out.data.data(), img.data(),
                  static_cast<std::size_t>(3) * w * w);
    } else {
      const int width = w / gs1;
      out.width = width;
      out.height = 6 * width;
      out.data.assign(static_cast<std::size_t>(3) * width * out.height, 0);
      std::string layout = eff->gridlayout ? *eff->gridlayout : "";
      int loaded[6] = {0, 0, 0, 0, 0, 0};
      for (int k = 0; k < gs0 * gs1; k++) {
        char sym = k < static_cast<int>(layout.size()) ? layout[k] : '.';
        int i = -1;
        switch (sym) {
          case 'R': i = 0; break; case 'L': i = 1; break;
          case 'U': i = 2; break; case 'D': i = 3; break;
          case 'F': i = 4; break; case 'B': i = 5; break;
          case '.': break;
          default:
            diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "texture",
                             "gridlayout symbol is not among '.RLUDFB' in texture",
                             tx.loc});
            return false;
        }
        if (i >= 0) {
          int rstart = width * (k / gs1);
          int cstart = width * (k % gs1);
          for (int j = 0; j < width; j++)
            std::memcpy(out.data.data() + i * 3 * width * width + j * 3 * width,
                        img.data() + (j + rstart) * 3 * w + 3 * cstart, 3 * width);
          loaded[i] = 1;
        }
      }
      for (int i = 0; i < 6; i++)
        if (!loaded[i])
          for (int k = 0; k < width; k++)
            for (int s = 0; s < width; s++)
              for (int j = 0; j < 3; j++)
                out.data[i * 3 * width * width + 3 * (k * width + s) + j] =
                    static_cast<unsigned char>(255 * rgb1[j]);
    }
    return true;
  }

  // ----- builtin texture (gradient/checker/flat + marks) -----
  CTexBuiltin b;
  b.type = out.type;
  b.nchannel = out.nchannel;
  if (bi) b.builtin = static_cast<int>(*bi);
  b.mark = eff->mark ? static_cast<int>(*eff->mark) : 0;
  if (eff->rgb1) for (int k = 0; k < 3; ++k) b.rgb1[k] = (*eff->rgb1)[k];
  if (eff->rgb2) for (int k = 0; k < 3; ++k) b.rgb2[k] = (*eff->rgb2)[k];
  if (eff->markrgb) for (int k = 0; k < 3; ++k) b.markrgb[k] = (*eff->markrgb)[k];
  if (eff->random) b.random = *eff->random;
  b.width = eff->width ? *eff->width : 0;
  b.height = eff->height ? *eff->height : 0;

  // dimension checks (mjCTexture::Compile:5652-5667).
  if (b.width < 1) {
    diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "texture",
                     "Invalid width of builtin texture", tx.loc});
    return false;
  }
  if (out.type != 0 /* 2D */) {
    b.height = 6 * b.width;
  } else if (b.height < 1) {
    diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "texture",
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
  // <layer> entries -> their role (the legacy `texture` attr is folded to a
  // <layer role="rgb"> at read). ProtoSpec's TexRole enum omits the USER role
  // (rgb=0..orm=8), so it is offset +1 from mjtTextureRole (mjTEXROLE_USER=0,
  // RGB=1..ORM=9).
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
// Assets: height fields (mjCHField::Compile, user_objects.cc:4752). Inline       //
// elevation (user-data) OR a file (PNG grey via lifted DecodePNG, or custom      //
// binary [nrow,ncol,f32...]) -> row flip -> normalize to [0,1] (subtract emin,   //
// divide by emax-emin when > mjEPS). File hfields resolve via meshdir + strippath //
// and derive a file-stem name when unnamed. geom_dataid + the hfield geom's      //
// size/aabb/rbound bind from the hfield's authored `size` (bound before the body //
// walk), so file loading never feeds geom sizing.                                //
// --------------------------------------------------------------------------- //
struct CHField {
  const Hfield* src = nullptr;
  ps::opt<std::string> name;
  std::string listname;     // name-table entry: authored name, else file stem
  std::string file;         // File() for m->paths (empty: user-data hfield)
  int nrow = 0, ncol = 0;
  double size[4] = {0, 0, 0, 0};
  std::vector<float> data;  // normalized elevation, row-major
};

// mjCHField::LoadPNG (user_objects.cc:4735): grey PNG, ncol=w, nrow=h, rows
// reversed. mjCHField::LoadCustom (:4691): [nrow:int][ncol:int][nrow*ncol f32].
bool HfieldLoadFile(const CompilerSettings& cs, const std::string& base_dir,
                    const std::string& file, int& nrow, int& ncol,
                    std::vector<float>& data, const ps::SourceLoc& loc,
                    std::vector<ps::mjcf::Diagnostic>& diags) {
  std::size_t dot = file.find_last_of('.');
  std::string ext = dot == std::string::npos ? "" : file.substr(dot);
  for (char& c : ext) c = static_cast<char>(std::tolower(c));
  const bool is_png = (ext == ".png");
  const std::string combined = MeshCombine(cs.meshdir, file);
  char err[1024] = {0};
  mjResource* res = mju_openResource(base_dir.empty() ? nullptr : base_dir.c_str(),
                                     combined.c_str(), nullptr, err, sizeof(err));
  if (!res) {
    diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "hfield",
                     std::string("could not open hfield file '") + combined + "'",
                     loc});
    return false;
  }
  const void* bytes = nullptr;
  int n = mju_readResource(res, &bytes);
  auto fail = [&](const std::string& msg) {
    mju_closeResource(res);
    diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "hfield", msg, loc});
    return false;
  };
  if (n < 1) return fail("could not read hfield file '" + combined + "'");
  const unsigned char* buf = static_cast<const unsigned char*>(bytes);

  if (is_png) {
    std::vector<unsigned char> img;
    int w = 0, h = 0;
    bool is_srgb = false;
    std::string derr;
    if (!lift::DecodePNG(buf, n, 1, img, w, h, is_srgb, derr))
      return fail(derr + " '" + combined + "'");
    ncol = w;
    nrow = h;
    data.clear();
    data.reserve(static_cast<std::size_t>(nrow) * ncol);
    for (int r = 0; r < nrow; r++)
      for (int c = 0; c < ncol; c++)
        data.push_back(static_cast<float>(img[c + (nrow - 1 - r) * ncol]));
  } else {  // custom binary
    if (n < static_cast<int>(2 * sizeof(int)))
      return fail("hfield missing header '" + combined + "'");
    const int* pint = reinterpret_cast<const int*>(buf);
    nrow = pint[0];
    ncol = pint[1];
    if (nrow < 1 || ncol < 1)
      return fail("non-positive hfield dimensions in file '" + combined + "'");
    if (n != nrow * ncol * static_cast<int>(sizeof(float)) + 8)
      return fail("unexpected file size in file '" + combined + "'");
    data.assign(static_cast<std::size_t>(nrow) * ncol, 0);
    std::memcpy(data.data(), pint + 2,
                static_cast<std::size_t>(nrow) * ncol * sizeof(float));
  }
  mju_closeResource(res);
  return true;
}

bool HfieldCompile(const Hfield& hf, const CompilerSettings& cs,
                   const std::string& base_dir, CHField& out,
                   std::vector<ps::mjcf::Diagnostic>& diags) {
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
      diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "hfield",
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

  // size parameters must be positive (checked before file load, as upstream).
  for (int k = 0; k < 4; ++k)
    if (out.size[k] <= 0) {
      diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "hfield",
                       "size parameter is not positive in hfield", hf.loc});
      return false;
    }

  // file path: strip, derive name, load (PNG or custom). MuJoCo errors if a
  // file hfield also authors nrow/ncol/data -- the gate lets only file-only
  // hfields through, so we do not reach a mixed state here.
  if (hf.file && !hf.file->empty()) {
    std::string file = *hf.file;
    if (cs.strippath) file = MeshStripPath(file);
    out.file = file;
    if (!HfieldLoadFile(cs, base_dir, file, out.nrow, out.ncol, out.data, hf.loc,
                        diags))
      return false;
  }

  // A dynamic hfield -- nrow/ncol authored, no file, no elevation -- allocates
  // nrow*ncol zeros at read time (OneHField "user data not given, set to 0",
  // xml_native_reader.cc:3603), to be filled at runtime. leg B round-trips those
  // zeros, so leg C must too.
  if ((!hf.file || hf.file->empty()) && out.data.empty() && out.nrow > 0 &&
      out.ncol > 0)
    out.data.assign(static_cast<std::size_t>(out.nrow) * out.ncol, 0.0f);

  if (out.nrow < 1 || out.ncol < 1 || out.data.empty()) {
    diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "hfield",
                     "hfield not specified", hf.loc});
    return false;
  }

  // name-table entry (mjCHField::CopyFromSpec :4650): authored name, else stem.
  if (hf.name) {
    out.listname = *hf.name;
  } else if (hf.file && !hf.file->empty()) {
    std::string stem = MeshStripPath(*hf.file);
    std::size_t d2 = stem.find_last_of('.');
    out.listname = d2 == std::string::npos ? stem : stem.substr(0, d2);
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
    for (int j = 0; j < hf.nrow * hf.ncol; ++j)
      m->hfield_data[adr + j] = hf.data[j];
    adr += hf.nrow * hf.ncol;
  }
  // hfield_pathadr is filled by FillMeshPaths (CopyPaths order).
}

// --------------------------------------------------------------------------- //
// Assets: skins (mjCSkin::Compile, user_mesh.cc:3109 + LoadSKN :3262). Inline    //
// geometry (vert/texcoord/face + per-bone body/bindpos/bindquat/vertid/          //
// vertweight) OR a .skn binary (meshdir + strippath resolution). Compile         //
// validates sizes, resolves bone body ids + material id, accumulates per-vertex  //
// weights and normalizes them (per-bone weight / total-vertex-weight), and       //
// normalizes bindquat. Runs after the body walk (needs body ids) and materials.  //
// --------------------------------------------------------------------------- //
struct CSkin {
  const Skin* src = nullptr;
  ps::opt<std::string> name;
  std::string listname;     // authored name, else file stem
  std::string file;         // File() for m->paths (empty: inline skin)
  int matid = -1;
  int group = 0;
  float rgba[4] = {0.5f, 0.5f, 0.5f, 1.0f};
  float inflate = 0.0f;
  std::vector<float> vert;      // 3*nvert
  std::vector<float> texcoord;  // 2*ntexvert (empty if none)
  std::vector<int> face;        // 3*nface
  std::vector<int> bodyid;      // nbone
  std::vector<float> bindpos;   // 3*nbone
  std::vector<float> bindquat;  // 4*nbone
  std::vector<std::vector<int>> vertid;       // per bone
  std::vector<std::vector<float>> vertweight; // per bone
};

// mjCSkin::LoadSKN (user_mesh.cc:3262): [nvert,ntex,nface,nbone][verts][texs]
// [faces]{ name[40], bindpos[3], bindquat[4], vcount, vertid[], vertweight[] }.
bool SkinLoadSKN(const unsigned char* buf, int n, CSkin& out,
                 std::vector<std::string>& bonename, const ps::SourceLoc& loc,
                 std::vector<ps::mjcf::Diagnostic>& diags) {
  auto fail = [&](const std::string& msg) {
    diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "skin", msg, loc});
    return false;
  };
  if (n < 16) return fail("missing header in SKN file");
  const int* hdr = reinterpret_cast<const int*>(buf);
  int nvert = hdr[0], ntexcoord = hdr[1], nface = hdr[2], nbone = hdr[3];
  if (nvert < 0 || ntexcoord < 0 || nface < 0 || nbone < 0)
    return fail("negative size in header of SKN file");
  if (n < 16 + 12 * nvert + 8 * ntexcoord + 12 * nface)
    return fail("insufficient data in SKN file");
  const float* pdata = reinterpret_cast<const float*>(buf + 16);
  int cnt = 0;
  if (nvert) {
    out.vert.resize(3 * nvert);
    std::memcpy(out.vert.data(), pdata + cnt, 3 * nvert * sizeof(float));
    cnt += 3 * nvert;
  }
  if (ntexcoord) {
    out.texcoord.resize(2 * ntexcoord);
    std::memcpy(out.texcoord.data(), pdata + cnt, 2 * ntexcoord * sizeof(float));
    cnt += 2 * ntexcoord;
  }
  if (nface) {
    out.face.resize(3 * nface);
    std::memcpy(out.face.data(), pdata + cnt, 3 * nface * sizeof(int));
    cnt += 3 * nface;
  }
  out.bindpos.resize(3 * nbone);
  out.bindquat.resize(4 * nbone);
  out.vertid.resize(nbone);
  out.vertweight.resize(nbone);
  bonename.clear();
  for (int i = 0; i < nbone; i++) {
    if (n / 4 - 4 - cnt < 18)
      return fail("insufficient data in SKN file, bone " + std::to_string(i));
    char txt[40];
    std::strncpy(txt, reinterpret_cast<const char*>(pdata + cnt), 39);
    txt[39] = '\0';
    cnt += 10;
    bonename.push_back(txt);
    std::memcpy(out.bindpos.data() + 3 * i, pdata + cnt, 3 * sizeof(float));
    cnt += 3;
    std::memcpy(out.bindquat.data() + 4 * i, pdata + cnt, 4 * sizeof(float));
    cnt += 4;
    int vcount = *reinterpret_cast<const int*>(pdata + cnt);
    cnt += 1;
    if (vcount < 1)
      return fail("vertex count must be positive in SKN file, bone " +
                  std::to_string(i));
    if (n / 4 - 4 - cnt < 2 * vcount)
      return fail("insufficient vertex data in SKN file, bone " +
                  std::to_string(i));
    out.vertid[i].resize(vcount);
    std::memcpy(out.vertid[i].data(), pdata + cnt, vcount * sizeof(int));
    cnt += vcount;
    out.vertweight[i].resize(vcount);
    std::memcpy(out.vertweight[i].data(), pdata + cnt, vcount * sizeof(float));
    cnt += vcount;
  }
  if (n != 16 + 4 * cnt) return fail("unexpected buffer size in SKN file");
  return true;
}

bool SkinCompile(const Model& model, const Skin& sk, const CompilerSettings& cs,
                 const std::string& base_dir,
                 const std::unordered_map<std::string, int>& bodyid_of,
                 const NameIdMap& matid_of, CSkin& out,
                 std::vector<ps::mjcf::Diagnostic>& diags) {
  std::unique_ptr<Skin> eff = ps::sdk::Effective(model, sk);
  out.src = &sk;
  out.name = sk.name;
  if (eff->group) out.group = *eff->group;
  if (eff->inflate) out.inflate = *eff->inflate;
  if (eff->rgba) for (int k = 0; k < 4; ++k) out.rgba[k] = (*eff->rgba)[k];

  std::vector<std::string> bonename;

  // file vs inline (mjCSkin::Compile). The gate admits only one or the other.
  if (eff->file && !eff->file->empty()) {
    std::string file = *eff->file;
    if (cs.strippath) file = MeshStripPath(file);
    out.file = file;
    std::string ext = file.size() >= 4 ? file.substr(file.size() - 4) : "";
    for (char& c : ext) c = static_cast<char>(std::tolower(c));
    if (ext != ".skn") {
      diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "skin",
                       "Unknown skin file type: " + file, sk.loc});
      return false;
    }
    const std::string combined = MeshCombine(cs.meshdir, file);
    char err[1024] = {0};
    mjResource* res = mju_openResource(
        base_dir.empty() ? nullptr : base_dir.c_str(), combined.c_str(), nullptr,
        err, sizeof(err));
    if (!res) {
      diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "skin",
                       "could not open skin file '" + combined + "'", sk.loc});
      return false;
    }
    const void* bytes = nullptr;
    int nb = mju_readResource(res, &bytes);
    if (nb < 0) {
      mju_closeResource(res);
      diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "skin",
                       "could not read SKN file '" + combined + "'", sk.loc});
      return false;
    }
    bool ok = SkinLoadSKN(static_cast<const unsigned char*>(bytes), nb, out,
                          bonename, sk.loc, diags);
    mju_closeResource(res);
    if (!ok) return false;
  } else {
    if (eff->vertex) out.vert = *eff->vertex;
    if (eff->texcoord) out.texcoord = *eff->texcoord;
    if (eff->face) out.face = *eff->face;
    for (const auto& b : eff->bones) {
      if (!b) continue;
      bonename.push_back(b->body ? *b->body : std::string());
      if (b->bindpos) for (int k = 0; k < 3; ++k)
        out.bindpos.push_back(static_cast<float>((*b->bindpos)[k]));
      if (b->bindquat) for (int k = 0; k < 4; ++k)
        out.bindquat.push_back(static_cast<float>((*b->bindquat)[k]));
      out.vertid.push_back(b->vertid ? *b->vertid : std::vector<int>{});
      out.vertweight.push_back(b->vertweight ? *b->vertweight
                                             : std::vector<float>{});
    }
  }

  // name-table entry (mjCSkin::CopyFromSpec :3072): authored name, else stem.
  if (sk.name) {
    out.listname = *sk.name;
  } else if (eff->file && !eff->file->empty()) {
    std::string stem = MeshStripPath(*eff->file);
    std::size_t dot = stem.find_last_of('.');
    out.listname = dot == std::string::npos ? stem : stem.substr(0, dot);
  }

  const std::size_t nbone = bonename.size();
  auto fail = [&](const std::string& msg) {
    diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "skin", msg, sk.loc});
    return false;
  };
  if (out.vert.empty() || out.face.empty() || nbone == 0 ||
      out.bindpos.empty() || out.bindquat.empty() || out.vertid.empty() ||
      out.vertweight.empty())
    return fail("Missing data in skin");
  if (out.vert.size() % 3) return fail("Vertex data must be multiple of 3");
  if (!out.texcoord.empty() && out.texcoord.size() != 2 * out.vert.size() / 3)
    return fail("Vertex and texcoord data incompatible size");
  if (out.face.size() % 3) return fail("Face data must be multiple of 3");
  if (out.bindpos.size() != 3 * nbone) return fail("Unexpected bindpos size in skin");
  if (out.bindquat.size() != 4 * nbone) return fail("Unexpected bindquat size in skin");
  if (out.vertid.size() != nbone) return fail("Unexpected vertid size in skin");
  if (out.vertweight.size() != nbone) return fail("Unexpected vertweight size in skin");

  // resolve bone body ids.
  out.bodyid.resize(nbone);
  for (std::size_t i = 0; i < nbone; ++i) {
    auto it = bodyid_of.find(bonename[i]);
    if (it == bodyid_of.end())
      return fail("unknown body '" + bonename[i] + "' in skin");
    out.bodyid[i] = it->second;
  }

  // resolve material id.
  if (eff->material && !eff->material->name.empty()) {
    auto it = matid_of.find(eff->material->name);
    if (it == matid_of.end())
      return fail("unknown material '" + eff->material->name + "' in skin");
    out.matid = it->second;
  }

  // accumulate per-vertex weights, check coverage, normalize.
  const std::size_t nvert = out.vert.size() / 3;
  std::vector<float> vw(nvert, 0.0f);
  for (std::size_t i = 0; i < nbone; ++i) {
    const std::size_t nbv = out.vertid[i].size();
    if (out.vertweight[i].size() != nbv || nbv == 0)
      return fail("vertid and vertweight must have same non-zero size in skin");
    for (std::size_t j = 0; j < nbv; ++j) {
      int jj = out.vertid[i][j];
      if (jj < 0 || jj >= static_cast<int>(nvert))
        return fail("vertid " + std::to_string(jj) + " out of range in skin");
      vw[jj] += out.vertweight[i][j];
    }
  }
  for (std::size_t i = 0; i < nvert; ++i)
    if (vw[i] <= static_cast<float>(mjMINVAL))
      return fail("vertex " + std::to_string(i) +
                  " must have positive total weight in skin");
  for (std::size_t i = 0; i < nbone; ++i)
    for (std::size_t j = 0; j < out.vertid[i].size(); ++j)
      out.vertweight[i][j] /= vw[out.vertid[i][j]];

  // normalize bindquat.
  for (std::size_t i = 0; i < nbone; ++i) {
    double q[4] = {out.bindquat[4 * i], out.bindquat[4 * i + 1],
                   out.bindquat[4 * i + 2], out.bindquat[4 * i + 3]};
    lift::mjuu_normvec(q, 4);
    for (int k = 0; k < 4; ++k) out.bindquat[4 * i + k] = static_cast<float>(q[k]);
  }
  return true;
}

void FillSkins(mjModel* m, const std::vector<CSkin>& skins) {
  int vert_adr = 0, texcoord_adr = 0, face_adr = 0, bone_adr = 0, bonevert_adr = 0;
  for (int i = 0; i < static_cast<int>(skins.size()); ++i) {
    const CSkin& sk = skins[i];
    m->skin_matid[i] = sk.matid;
    m->skin_group[i] = sk.group;
    for (int k = 0; k < 4; ++k) m->skin_rgba[4 * i + k] = sk.rgba[k];
    m->skin_inflate[i] = sk.inflate;
    m->skin_vertadr[i] = vert_adr;
    m->skin_vertnum[i] = static_cast<int>(sk.vert.size() / 3);
    m->skin_texcoordadr[i] = sk.texcoord.empty() ? -1 : texcoord_adr;
    m->skin_faceadr[i] = face_adr;
    m->skin_facenum[i] = static_cast<int>(sk.face.size() / 3);
    m->skin_boneadr[i] = bone_adr;
    m->skin_bonenum[i] = static_cast<int>(sk.bodyid.size());
    if (!sk.vert.empty())
      std::memcpy(m->skin_vert + 3 * vert_adr, sk.vert.data(),
                  sk.vert.size() * sizeof(float));
    if (!sk.texcoord.empty())
      std::memcpy(m->skin_texcoord + 2 * texcoord_adr, sk.texcoord.data(),
                  sk.texcoord.size() * sizeof(float));
    if (!sk.face.empty())
      std::memcpy(m->skin_face + 3 * face_adr, sk.face.data(),
                  sk.face.size() * sizeof(int));
    if (!sk.bindpos.empty())
      std::memcpy(m->skin_bonebindpos + 3 * bone_adr, sk.bindpos.data(),
                  sk.bindpos.size() * sizeof(float));
    if (!sk.bindquat.empty())
      std::memcpy(m->skin_bonebindquat + 4 * bone_adr, sk.bindquat.data(),
                  sk.bindquat.size() * sizeof(float));
    if (!sk.bodyid.empty())
      std::memcpy(m->skin_bonebodyid + bone_adr, sk.bodyid.data(),
                  sk.bodyid.size() * sizeof(int));
    for (int j = 0; j < m->skin_bonenum[i]; ++j) {
      m->skin_bonevertadr[bone_adr + j] = bonevert_adr;
      m->skin_bonevertnum[bone_adr + j] = static_cast<int>(sk.vertid[j].size());
      if (!sk.vertid[j].empty()) {
        std::memcpy(m->skin_bonevertid + bonevert_adr, sk.vertid[j].data(),
                    sk.vertid[j].size() * sizeof(int));
        std::memcpy(m->skin_bonevertweight + bonevert_adr, sk.vertweight[j].data(),
                    sk.vertweight[j].size() * sizeof(float));
      }
      bonevert_adr += m->skin_bonevertnum[bone_adr + j];
    }
    vert_adr += m->skin_vertnum[i];
    texcoord_adr += static_cast<int>(sk.texcoord.size() / 2);
    face_adr += m->skin_facenum[i];
    bone_adr += m->skin_bonenum[i];
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
              (type == mjGEOM_MESH || type == mjGEOM_SDF) &&
              (contype || conaff || in_pair);
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
        case BodyChildAny::Kind::Replicate: {
          const auto& r = std::get<std::unique_ptr<Replicate>>(c.node);
          if (r) walk_children(r->subtree);
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

// Resolve a mesh/geom <plugin> reference to its plugin descriptor + config, from
// the model's <extension> instances. A ref by instance name resolves that
// instance's declared plugin + config; an inline plugin name carries its own
// config. Returns false if the plugin name is unknown (unresolved).
std::map<std::string, std::string> ConfigMap(
    const std::vector<std::unique_ptr<Config>>& config) {
  std::map<std::string, std::string> cfg;
  for (const auto& c : config)
    if (c && c->key) cfg[*c->key] = c->value ? *c->value : std::string();
  return cfg;
}

struct ResolvedPluginRef {
  const mjpPlugin* p = nullptr;
  std::map<std::string, std::string> config;
};
bool ResolvePluginRef(const Model& m, const PluginRef& pr, ResolvedPluginRef& out) {
  std::string plugin_name;
  const std::vector<std::unique_ptr<Config>>* config = nullptr;
  if (pr.instance && !pr.instance->name.empty()) {
    // Find the extension instance and the <plugin> that declares it.
    for (const auto& ext : m.extensions) {
      if (!ext) continue;
      for (const auto& pd : ext->pluginDefs) {
        if (!pd || !pd->plugin) continue;
        for (const auto& inst : pd->pluginInstances) {
          if (inst && inst->name && *inst->name == pr.instance->name) {
            plugin_name = *pd->plugin;
            config = &inst->config;
          }
        }
      }
    }
    if (plugin_name.empty()) return false;
  } else if (pr.plugin) {
    plugin_name = *pr.plugin;
    config = &pr.config;
  } else {
    return false;
  }
  int slot = -1;
  if (!mjp_getPlugin(plugin_name.c_str(), &slot)) return false;
  out.p = mjp_getPluginAtSlot(slot);
  if (config) out.config = ConfigMap(*config);
  return true;
}

// mjCMesh::LoadSDF (user_mesh.cc:356-442): a plugin (SDF) mesh has no vertices of
// its own -- the plugin defines a signed distance field, which is sampled on a
// regular grid and marching-cubed into a visualization mesh. The generated
// vert/normal/face feed the ordinary mesh pipeline (needreorient=false keeps the
// plugin frame). Returns false on a non-SDF/unresolved plugin.
bool LoadSdfMesh(const Model& model, const Mesh& mesh, lift::MeshInput& in,
                 std::vector<ps::mjcf::Diagnostic>& diags) {
  ResolvedPluginRef rp;
  if (mesh.plugin.empty() || !mesh.plugin.front() ||
      !ResolvePluginRef(model, *mesh.plugin.front(), rp)) {
    diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "mesh",
                     "native: unresolved SDF mesh plugin", mesh.loc});
    return false;
  }
  const mjpPlugin* p = rp.p;
  if (!(p->capabilityflags & mjPLUGIN_SDF)) {
    diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "mesh",
                     "native: mesh plugin is not an SDF plugin", mesh.loc});
    return false;
  }

  std::vector<mjtNum> attributes(p->nattribute, 0);
  std::vector<const char*> names(p->nattribute, nullptr);
  std::vector<std::string> valstore(p->nattribute);
  std::vector<const char*> values(p->nattribute, nullptr);
  for (int i = 0; i < p->nattribute; ++i) {
    names[i] = p->attributes[i];
    auto it = rp.config.find(p->attributes[i]);
    valstore[i] = it == rp.config.end() ? std::string() : it->second;
    values[i] = valstore[i].c_str();
  }
  if (p->sdf_attribute)
    p->sdf_attribute(attributes.data(), names.data(), values.data());

  mjtNum aabb[6] = {0};
  p->sdf_aabb(aabb, attributes.data());
  const mjtNum total = aabb[3] + aabb[4] + aabb[5];
  const double n = 300;
  const int nx = static_cast<int>(std::floor(n / total * aabb[3])) + 1;
  const int ny = static_cast<int>(std::floor(n / total * aabb[4])) + 1;
  const int nz = static_cast<int>(std::floor(n / total * aabb[5])) + 1;
  std::vector<MC::MC_FLOAT> field(static_cast<std::size_t>(nx) * ny * nz);
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k) {
        mjtNum point[] = {aabb[0] - aabb[3] + 2 * aabb[3] * i / (nx - 1),
                          aabb[1] - aabb[4] + 2 * aabb[4] * j / (ny - 1),
                          aabb[2] - aabb[5] + 2 * aabb[5] * k / (nz - 1)};
        field[(static_cast<std::size_t>(k) * ny + j) * nx + i] =
            p->sdf_staticdistance(point, attributes.data());
      }

  MC::mcMesh mc;
  MC::marching_cube(field.data(), nx, ny, nz, mc);
  in.format = lift::MeshFormat::UserVertex;
  in.needreorient = false;
  in.uservert.reserve(mc.vertices.size() * 3);
  for (auto& v : mc.vertices) {
    in.uservert.push_back(
        static_cast<float>(2 * aabb[3] * v.x / (nx - 1) + aabb[0] - aabb[3]));
    in.uservert.push_back(
        static_cast<float>(2 * aabb[4] * v.y / (ny - 1) + aabb[1] - aabb[4]));
    in.uservert.push_back(
        static_cast<float>(2 * aabb[5] * v.z / (nz - 1) + aabb[2] - aabb[5]));
  }
  in.usernormal.reserve(mc.normals.size() * 3);
  for (auto& nrm : mc.normals) {
    in.usernormal.push_back(nrm.x);
    in.usernormal.push_back(nrm.y);
    in.usernormal.push_back(nrm.z);
  }
  in.userface.reserve(mc.indices.size());
  for (unsigned int idx : mc.indices) in.userface.push_back(static_cast<int>(idx));
  return true;
}

// Compile one mesh into a CMesh (files resolved via the resource API).
bool MeshCompile(const Model& model, const Mesh& mesh, const CompilerSettings& cs,
                 const std::string& base_dir,
                 const std::unordered_map<std::string, bool>& mesh_hull,
                 CMesh& out, std::vector<ps::mjcf::Diagnostic>& diags) {
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

  if (!mesh.plugin.empty()) {
    // Plugin (SDF) mesh: geometry generated by marching cubes over the plugin's
    // signed distance field (mjCMesh::LoadSDF), then run through the pipeline.
    if (!LoadSdfMesh(model, mesh, in, diags)) return false;
  } else if (eff->builtin && *eff->builtin != MeshBuiltin::none) {
    // Builtin procedural mesh (sphere/hemisphere/cone/...): generate its
    // vert/normal/face (mjs_makeMesh + mjCMesh::Make*), then run the pipeline.
    lift::MeshBuiltinKind kind;
    switch (*eff->builtin) {
      case MeshBuiltin::sphere:      kind = lift::MeshBuiltinKind::Sphere; break;
      case MeshBuiltin::hemisphere:  kind = lift::MeshBuiltinKind::Hemisphere; break;
      case MeshBuiltin::cone:        kind = lift::MeshBuiltinKind::Cone; break;
      case MeshBuiltin::supertorus:  kind = lift::MeshBuiltinKind::Supertorus; break;
      case MeshBuiltin::supersphere: kind = lift::MeshBuiltinKind::Supersphere; break;
      case MeshBuiltin::wedge:       kind = lift::MeshBuiltinKind::Wedge; break;
      case MeshBuiltin::plate:       kind = lift::MeshBuiltinKind::Plate; break;
      default:
        diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "mesh",
                         "native: unsupported builtin mesh", mesh.loc});
        return false;
    }
    std::vector<double> params;
    if (eff->params) params = *eff->params;
    lift::BuiltinMeshResult br;
    std::string berr;
    if (!lift::MakeBuiltinMesh(kind, params, br, berr)) {
      diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "mesh",
                       "native: builtin mesh: " + berr, mesh.loc});
      return false;
    }
    in.format = lift::MeshFormat::UserVertex;
    in.uservert = std::move(br.uservert);
    in.usernormal = std::move(br.usernormal);
    in.userface = std::move(br.userface);
    if (br.inertia_shell) in.inertia = mjMESH_INERTIA_SHELL;
  } else if (eff->file) {
    std::string file = *eff->file;
    if (cs.strippath) file = MeshStripPath(file);
    out.file = file;
    const std::string combined = MeshCombine(cs.meshdir, file);
    char err[1024] = {0};
    mjResource* res = mju_openResource(base_dir.empty() ? nullptr : base_dir.c_str(),
                                       combined.c_str(), nullptr, err, sizeof(err));
    if (!res) {
      diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "mesh",
                       std::string("could not open mesh file '") + combined + "'",
                       mesh.loc});
      return false;
    }
    const void* bytes = nullptr;
    int n = mju_readResource(res, &bytes);
    if (n < 0) {
      mju_closeResource(res);
      diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "mesh",
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
      diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "mesh",
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
    diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "mesh",
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

// Asset file paths, in CopyPaths order (hfields, meshes, skins, textures). File
// meshes (mesh_pathadr) and file textures (tex_pathadr) carry a path; hfields in
// the native scope are user-data only (-1). One shared `paths` cursor, mirroring
// mjCModel::CopyPaths' pathlist() concatenation, so the addresses match leg B.
void FillMeshPaths(mjModel* m, const std::vector<CHField>& hfields,
                   const std::vector<CMesh>& meshes,
                   const std::vector<CSkin>& skins,
                   const std::vector<CTexture>& textures) {
  m->paths[0] = 0;
  int adr = 0;
  auto emit = [&](const std::string& f, int* pathadr) {
    if (f.empty()) { *pathadr = -1; return; }
    *pathadr = adr;
    std::memcpy(m->paths + adr, f.c_str(), f.size());
    adr += static_cast<int>(f.size());
    m->paths[adr] = 0;
    adr++;
  };
  for (int i = 0; i < static_cast<int>(hfields.size()); ++i)
    emit(hfields[i].file, &m->hfield_pathadr[i]);
  for (int i = 0; i < static_cast<int>(meshes.size()); ++i)
    emit(meshes[i].file, &m->mesh_pathadr[i]);
  for (int i = 0; i < static_cast<int>(skins.size()); ++i)
    emit(skins[i].file, &m->skin_pathadr[i]);
  for (int i = 0; i < static_cast<int>(textures.size()); ++i)
    emit(textures[i].file, &m->tex_pathadr[i]);
}

// --------------------------------------------------------------------------- //
// Flex (mjCFlex, NC5 Wave 1 geometry + Wave 3 elasticity, non-interpolated).    //
// Lifted from src/user/user_mesh.cc: mjCFlex::Compile (:4630), ResolveReferences //
// (:4275), CreateShellPair (:5460), CreateFlapStencil (:3605), CreateBVH (:5410),//
// the simplex tables (:3385 / user_objects.h:1050), and the young>0 elasticity  //
// kernels (:3382-3722, above). Node/dof interpolation and gmsh/mesh loaders are  //
// still gated to the XML fallback (native.cc).                                   //
// Sizing (nflex*/nJfe/nJfv, user_model.cc:2182-2241) and fill (CopyObjects       //
// :3432-3654) are retargeted below. BVH reuses the shared lifted BVH class.      //
// --------------------------------------------------------------------------- //

// Adjacent-triangle stencil (user_objects.h:963). Populated by the flap stencil
// even at young=0 for its edge-consistency assertion; edgeflap output feeds the
// young>0 bending kernel (ComputeBending<StencilFlap>, NC5 Wave 3).
struct StencilFlap {
  static constexpr int kNumVerts = 4;
  int vertices[kNumVerts];
};

// Simplex connectivity (user_mesh.cc:3385): local edges per element by dim-1.
constexpr int kFlexEledge[3][6][2] = {
    {{0, 1}, {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}},
    {{1, 2}, {2, 0}, {0, 1}, {-1, -1}, {-1, -1}, {-1, -1}},
    {{0, 1}, {1, 2}, {2, 0}, {2, 3}, {0, 3}, {1, 3}}};

// Edges per element indexed by dim (user_objects.h:1050).
constexpr int kFlexNumEdges[3] = {1, 3, 6};

// Local triangle edge order used by CreateFlapStencil (user_mesh.cc:3602).
constexpr int kFlapEdge[3][2] = {{1, 2}, {2, 0}, {0, 1}};

// std::pair hash (user_mesh.cc:2893 PairHash).
struct FlexPairHash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& p) const {
    return std::hash<T1>()(p.first) ^ std::hash<T2>()(p.second);
  }
};

// Per-flex compiled state (side table). Field set mirrors mjCFlex_ plus the
// resolved mjsFlex scalars needed by the fill; young/poisson/elastic2d drive the
// NC5 Wave 3 elasticity kernels (stiffness/bending) when young>0.
struct CFlex {
  std::string name;
  int dim = 2;
  double radius = 0.005;
  int group = 0;
  int matid = -1;
  std::string material_name;
  float rgba[4] = {0.5f, 0.5f, 0.5f, 1.0f};
  bool flatskin = false;
  // contact (FlexContact / mjs_defaultFlex)
  int contype = 1, conaffinity = 1, condim = 3, priority = 0;
  double friction[3] = {1, 0.005, 0.0001};
  double solmix = 1;
  double solref[2] = {0.02, 1};
  double solimp[5] = {0.9, 0.95, 0.001, 0.5, 2};
  double margin = 0, gap = 0;
  bool internal = false;
  int selfcollide = mjFLEXSELF_AUTO;
  int activelayers = 1;
  bool passive = false;
  // edge (FlexEdge)
  double edgestiffness = 0, edgedamping = 0;
  // elasticity (FlexElasticity); young==0 on this path
  double damping = 0, young = 0, poisson = 0, thickness = -1;
  int elastic2d = 0;
  int order = 0;            // spec.order (0 without dof) -> flex_interp
  int cellcount[3] = {1, 1, 1};
  // topology
  int nvert = 0, nnode = 0, nedge = 0, nelem = 0;
  bool rigid = false, centered = false, interpolated = false;
  bool has_strain_eq = false;            // a strain equality references this flex
  std::vector<int> vertbodyid;
  std::vector<int> nodebodyid;           // interpolated: node -> body id
  std::vector<double> node;              // interpolated: node local offsets (synth)
  std::vector<double> node0;             // interpolated: node0 (local frame)
  std::vector<char> cell_empty;          // interpolated volume: empty-cell mask
  std::vector<std::pair<int, int>> edge;
  std::vector<int> edgeidx;
  std::vector<int> shell, elemlayer, evpair;
  std::vector<int> elem;                 // reordered for dim==3
  std::vector<double> vert;              // authored offsets (empty if centered)
  std::vector<float> texcoord;
  std::vector<int> elemtexcoord;
  std::vector<double> vertxpos;
  std::vector<double> vert0;
  std::vector<StencilFlap> flaps;
  std::vector<double> stiffness;         // young>0: 21*nelem (0 if young==0)
  std::vector<double> bending;           // young>0 elastic2d bend: 17*nedge
  double size[3] = {0, 0, 0};
  int edgeequality = 0;                  // resolved at fill from equalities
  // BVH (dynamic; laid out after body+mesh static nodes)
  std::vector<double> bvh;
  std::vector<int> bvh_child, bvh_level, bvh_nodeid;
  int bvhadr = -1;
  bool HasTexcoord() const { return !texcoord.empty(); }
  int nbvh() const { return static_cast<int>(bvh_child.size() / 2); }
};

// Split a whitespace-separated attribute string into tokens (the flex `body`
// attribute is a name list; the reader stores it verbatim).
std::vector<std::string> SplitWhitespace(const std::string& s) {
  std::vector<std::string> out;
  std::istringstream is(s);
  std::string tok;
  while (is >> tok) out.push_back(tok);
  return out;
}

// --------------------------------------------------------------------------- //
// Nonlinear elasticity kernels (young>0), NC5 Wave 3. Lifted verbatim from      //
// user_mesh.cc:3382-3722 (the Stencil2D/Stencil3D vertex-edge stencils and the  //
// ComputeVolume / MetricTensor / ComputeBasis / ComputeStiffness template       //
// specializations, cot, the ComputeVolume triangle-area overload, and           //
// ComputeBending); the sole edit is qualifying mjuu_* with lift::. FlexCompile  //
// calls ComputeStiffness per element and ComputeBending per edge when young>0.  //
// --------------------------------------------------------------------------- //

// simplex vertex/edge stencils (user_mesh.cc:3392).
struct Stencil2D {
  static constexpr int kNumEdges = 3;
  static constexpr int kNumVerts = 3;
  static constexpr int kNumFaces = 2;
  static constexpr int edge[kNumEdges][2] = {{1, 2}, {2, 0}, {0, 1}};
  static constexpr int face[kNumVerts][2] = {{1, 2}, {2, 0}, {0, 1}};
  static constexpr int edge2face[kNumEdges][2] = {{1, 2}, {2, 0}, {0, 1}};
  int vertices[kNumVerts];
  int edges[kNumEdges];
};

struct Stencil3D {
  static constexpr int kNumEdges = 6;
  static constexpr int kNumVerts = 4;
  static constexpr int kNumFaces = 3;
  static constexpr int edge[kNumEdges][2] = {{0, 1}, {1, 2}, {2, 0},
                                             {2, 3}, {0, 3}, {1, 3}};
  static constexpr int face[kNumVerts][3] = {{2, 1, 0}, {0, 1, 3},
                                             {1, 2, 3}, {2, 0, 3}};
  static constexpr int edge2face[kNumEdges][2] = {{2, 3}, {1, 3}, {2, 1},
                                                  {1, 0}, {0, 2}, {0, 3}};
  int vertices[kNumVerts];
  int edges[kNumEdges];
};

template <typename T>
inline double ComputeVolume(const double* x, const int v[T::kNumVerts]);

template <>
inline double ComputeVolume<Stencil2D>(const double* x,
                                       const int v[Stencil2D::kNumVerts]) {
  double normal[3];
  const double* x0 = x + 3*v[0];
  const double* x1 = x + 3*v[1];
  const double* x2 = x + 3*v[2];
  double edge1[3] = {x1[0]-x0[0], x1[1]-x0[1], x1[2]-x0[2]};
  double edge2[3] = {x2[0]-x0[0], x2[1]-x0[1], x2[2]-x0[2]};
  lift::mjuu_crossvec(normal, edge1, edge2);
  return lift::mjuu_normvec(normal, 3) / 2;
}

template<>
inline double ComputeVolume<Stencil3D>(const double* x,
                                       const int v[Stencil3D::kNumVerts]) {
  double normal[3];
  const double* x0 = x + 3*v[0];
  const double* x1 = x + 3*v[1];
  const double* x2 = x + 3*v[2];
  const double* x3 = x + 3*v[3];
  double edge1[3] = {x1[0]-x0[0], x1[1]-x0[1], x1[2]-x0[2]};
  double edge2[3] = {x2[0]-x0[0], x2[1]-x0[1], x2[2]-x0[2]};
  double edge3[3] = {x3[0]-x0[0], x3[1]-x0[1], x3[2]-x0[2]};
  lift::mjuu_crossvec(normal, edge1, edge2);
  return lift::mjuu_dot3(normal, edge3) / 6;
}

// compute metric tensor of edge lengths inner product
template <typename T>
void inline MetricTensor(double* metric, int idx, double mu,
                         double la, const double basis[T::kNumEdges][9]) {
  double trE[T::kNumEdges] = {0};
  double trEE[T::kNumEdges*T::kNumEdges] = {0};
  double k[T::kNumEdges*T::kNumEdges];

  // compute first invariant i.e. trace(strain)
  for (int e = 0; e < T::kNumEdges; e++) {
    for (int i = 0; i < 3; i++) {
      trE[e] += basis[e][4*i];
    }
  }

  // compute second invariant i.e. trace(strain^2)
  for (int ed1 = 0; ed1 < T::kNumEdges; ed1++) {
    for (int ed2 = 0; ed2 < T::kNumEdges; ed2++) {
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          trEE[T::kNumEdges*ed1+ed2] += basis[ed1][3*i+j] * basis[ed2][3*j+i];
        }
      }
    }
  }

  // assembly of strain metric tensor
  for (int ed1 = 0; ed1 < T::kNumEdges; ed1++) {
    for (int ed2 = 0; ed2 < T::kNumEdges; ed2++) {
      k[T::kNumEdges*ed1 + ed2] = mu * trEE[T::kNumEdges * ed1 + ed2] +
                                  la * trE[ed2] * trE[ed1];
    }
  }

  // copy to triangular representation
  int id = 0;
  for (int ed1 = 0; ed1 < T::kNumEdges; ed1++) {
    for (int ed2 = ed1; ed2 < T::kNumEdges; ed2++) {
      metric[21*idx + id++] = k[T::kNumEdges*ed1 + ed2];
    }
  }

  if (id != T::kNumEdges*(T::kNumEdges+1)/2) {
    mju_error("incorrect stiffness matrix size");
  }
}

// compute local basis
template <typename T>
void inline ComputeBasis(double basis[9], const double* x,
                         const int v[T::kNumVerts],
                         const int faceL[T::kNumFaces],
                         const int faceR[T::kNumFaces], double volume);

template <>
void inline ComputeBasis<Stencil2D>(double basis[9], const double* x,
                                    const int v[Stencil2D::kNumVerts],
                                    const int faceL[Stencil2D::kNumFaces],
                                    const int faceR[Stencil2D::kNumFaces],
                                    double volume) {
  double basisL[3], basisR[3];
  double normal[3];

  const double* xL0 = x + 3*v[faceL[0]];
  const double* xL1 = x + 3*v[faceL[1]];
  const double* xR0 = x + 3*v[faceR[0]];
  const double* xR1 = x + 3*v[faceR[1]];
  double edgesL[3] = {xL0[0]-xL1[0], xL0[1]-xL1[1], xL0[2]-xL1[2]};
  double edgesR[3] = {xR1[0]-xR0[0], xR1[1]-xR0[1], xR1[2]-xR0[2]};

  lift::mjuu_crossvec(normal, edgesR, edgesL);
  lift::mjuu_normvec(normal, 3);
  lift::mjuu_crossvec(basisL, normal, edgesL);
  lift::mjuu_crossvec(basisR, edgesR, normal);

  // we use as basis the symmetrized tensor products of the edge normals of the
  // other two edges; this is shown in Weischedel "A discrete geometric view on
  // shear-deformable shell models" in the remark at the end of section 4.1;
  // equivalent to linear finite elements but in a coordinate-free formulation.

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      basis[3*i+j] = (basisL[i]*basisR[j] +
                      basisR[i]*basisL[j]) / (8*volume*volume);
    }
  }
}

// compute local basis
template <>
void inline ComputeBasis<Stencil3D>(double basis[9], const double* x,
                                    const int v[Stencil3D::kNumVerts],
                                    const int faceL[Stencil3D::kNumFaces],
                                    const int faceR[Stencil3D::kNumFaces],
                                    double volume) {
  const double* xL0 = x + 3*v[faceL[0]];
  const double* xL1 = x + 3*v[faceL[1]];
  const double* xL2 = x + 3*v[faceL[2]];
  const double* xR0 = x + 3*v[faceR[0]];
  const double* xR1 = x + 3*v[faceR[1]];
  const double* xR2 = x + 3*v[faceR[2]];
  double edgesL[6] = {xL1[0] - xL0[0], xL1[1] - xL0[1], xL1[2] - xL0[2],
                      xL2[0] - xL0[0], xL2[1] - xL0[1], xL2[2] - xL0[2]};
  double edgesR[6] = {xR1[0] - xR0[0], xR1[1] - xR0[1], xR1[2] - xR0[2],
                      xR2[0] - xR0[0], xR2[1] - xR0[1], xR2[2] - xR0[2]};

  double normalL[3], normalR[3];
  lift::mjuu_crossvec(normalL, edgesL, edgesL+3);
  lift::mjuu_crossvec(normalR, edgesR, edgesR+3);

  // we use as basis the symmetrized tensor products of the area normals of the
  // two faces not adjacent to the edge; this is the 3D equivalent to the basis
  // proposed in Weischedel "A discrete geometric view on shear-deformable shell
  // models" in the remark at the end of section 4.1. This is also equivalent to
  // linear finite elements but in a coordinate-free formulation.

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      basis[3*i+j] = (normalL[i]*normalR[j] +
                      normalR[i]*normalL[j]) / (36*2*volume*volume);
    }
  }
}

// compute stiffness for a single element
template <typename T>
void inline ComputeStiffness(std::vector<double>& stiffness,
                             const std::vector<double>& body_pos,
                             const int* v, int t, double E,
                             double nu, double thickness = 4) {
  // triangles area
  double volume = ComputeVolume<T>(body_pos.data(), v);

  // material parameters
  double mu = E / (2*(1+nu)) * std::abs(volume) / 4 * thickness;
  double la = E*nu / ((1+nu)*(1-2*nu)) * std::abs(volume) / 4 * thickness;

  // local geometric quantities
  double basis[T::kNumEdges][9] = {{0}};

  // compute edge basis
  for (int e = 0; e < T::kNumEdges; e++) {
    ComputeBasis<T>(basis[e], body_pos.data(), v,
                    T::face[T::edge2face[e][0]],
                    T::face[T::edge2face[e][1]], volume);
  }

  // compute metric tensor
  MetricTensor<T>(stiffness.data(), t, mu, la, basis);
}

// cotangent between two edges
double inline cot(const double* x, int v0, int v1, int v2) {
  double normal[3];
  double edge1[3] = {x[3*v1]-x[3*v0], x[3*v1+1]-x[3*v0+1], x[3*v1+2]-x[3*v0+2]};
  double edge2[3] = {x[3*v2]-x[3*v0], x[3*v2+1]-x[3*v0+1], x[3*v2+2]-x[3*v0+2]};

  lift::mjuu_crossvec(normal, edge1, edge2);
  return lift::mjuu_dot3(edge1, edge2) / sqrt(lift::mjuu_dot3(normal, normal));
}

// area of a triangle
double inline ComputeVolume(const double* x, const int v[Stencil2D::kNumVerts]) {
  double normal[3];
  double edge1[3] = {x[3*v[1]]-x[3*v[0]], x[3*v[1]+1]-x[3*v[0]+1], x[3*v[1]+2]-x[3*v[0]+2]};
  double edge2[3] = {x[3*v[2]]-x[3*v[0]], x[3*v[2]+1]-x[3*v[0]+1], x[3*v[2]+2]-x[3*v[0]+2]};

  lift::mjuu_crossvec(normal, edge1, edge2);
  return sqrt(lift::mjuu_dot3(normal, normal)) / 2;
}

// compute bending stiffness for a single edge
template <typename T>
void inline ComputeBending(double* bending, double* pos, const int v[4], double mu,
                           double thickness) {
  int vadj[3] = {v[1], v[0], v[3]};

  if (v[3]== -1) {
    // skip boundary edges
    return;
  }

  // cotangent operator from Wardetzky at al., "Discrete Quadratic Curvature
  // Energies", https://cims.nyu.edu/gcl/papers/wardetzky2007dqb.pdf

  double a01 = cot(pos, v[0], v[1], v[2]);
  double a02 = cot(pos, v[0], v[3], v[1]);
  double a03 = cot(pos, v[1], v[2], v[0]);
  double a04 = cot(pos, v[1], v[0], v[3]);
  double c[4] = {a03 + a04, a01 + a02, -(a01 + a03), -(a02 + a04)};
  double volume = ComputeVolume(pos, v) + ComputeVolume(pos, vadj);
  double stiffness = 3 * mu * pow(thickness, 3) / (24 * volume);

  // Garg et al., "Cubic Shells", https://cims.nyu.edu/gcl/papers/garg2007cs.pdf
  const double* v0 = pos + 3*v[0];
  const double* v1 = pos + 3*v[1];
  const double* v2 = pos + 3*v[2];
  const double* v3 = pos + 3*v[3];
  double e0[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
  double e1[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
  double e2[3] = {v3[0] - v0[0], v3[1] - v0[1], v3[2] - v0[2]};
  double e3[3] = {v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]};
  double e4[3] = {v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]};
  double t0[3] = {-(a03*e1[0] + a01*e3[0]), -(a03*e1[1] + a01*e3[1]), -(a03*e1[2] + a01*e3[2])};
  double t1[3] = {-(a04*e2[0] + a02*e4[0]), -(a04*e2[1] + a02*e4[1]), -(a04*e2[2] + a02*e4[2])};
  double sqr = lift::mjuu_dot3(e0, e0);
  double cos_theta = -lift::mjuu_dot3(t0, t1) / sqr;

  for (int v1 = 0; v1 < T::kNumVerts; v1++) {
    for (int v2 = 0; v2 < T::kNumVerts; v2++) {
      bending[4 * v1 + v2] += c[v1] * c[v2] * cos_theta * stiffness;
    }
  }

  double n[3];
  lift::mjuu_crossvec(n, e0, e1);
  bending[16] = lift::mjuu_dot3(n, e2) * (a01 - a03) * (a04 - a02) * stiffness / (sqr * sqrt(sqr));
}

// --------------------------------------------------------------------------- //
// Interpolated FE elasticity kernels (NC5 Wave 6). Lifted verbatim from         //
// user_mesh.cc:3728-4180 (quadratureGaussLegendre, phi/dphi, sym/inner/trace,   //
// ComputeLinearStiffness, ComputeLinearStiffness2D, ComputeWarpMode,            //
// ComputeWarpStiffness, EigendecomposeStiffness) and :4391 (ComputeInterpBending//
// ); the sole edits are qualifying mjuu_* with lift:: and lowering throw        //
// mjCError/mjERROR internal asserts (unreachable for well-formed grids) to      //
// mju_error, matching MetricTensor's existing convention. Called by FlexCompile //
// per finite-element cell (volume) or boundary quad (shell).                    //
// --------------------------------------------------------------------------- //

// Gauss Legendre quadrature points in 1 dimension on the interval [a, b]
void quadratureGaussLegendre(double* points, double* weights,
                             const int order, const double a, const double b) {
  if (order > 3)
    mju_error("Integration order > 3 not yet supported.");

  // x is on [-1, 1], p on [a, b]
  double p0 = (a+b)/2.;
  double dpdx = (b-a)/2;

  if (order == 2) {
    points[0] = -dpdx / sqrt(3) + p0;
    points[1] =  dpdx / sqrt(3) + p0;
    weights[0] = dpdx;
    weights[1] = dpdx;
  } else {
    points[0] = p0;
    points[1] = -dpdx / sqrt(3. / 5.) + p0;
    points[2] =  dpdx / sqrt(3. / 5.) + p0;
    weights[0] = 8. / 9. * dpdx;
    weights[1] = 5. / 9. * dpdx;
    weights[2] = 5. / 9. * dpdx;
  }
}

// evaluate 1-dimensional basis function
double phi(const double s, const int i, const int order) {
  if (order == 1) {
    return i == 0 ? 1 - s : s;
  } else if (order == 2) {
    switch (i) {
      case 0:
        return 2 * s * s - 3 * s + 1;
      case 1:
        return 4 * (s - s * s);
      case 2:
        return 2 * s * s - s;
      default:
        mju_error("invalid index %d", i);
        return 0;
    }
  } else {
    mju_error("Order must be 1 or 2.");
    return 0;
  }
}

// evaluate gradient of 1-dimensional basis function
double dphi(const double s, const int i, const int order) {
  if (order == 1) {
    return i == 0 ? -1 : 1;
  } else if (order == 2) {
    switch (i) {
      case 0:
        return 4 * s - 3;
      case 1:
        return 4 * (1 - 2 * s);
      case 2:
        return 4 * s - 1;
      default:
        mju_error("invalid index %d, must be 0, 1, or 2", i);
        return 0;
    }
  } else {
    mju_error("Order must be 1 or 2.");
    return 0;
  }
}

typedef std::array<std::array<double, 3>, 3> FeMatrix;

// symmetrize a tensor
FeMatrix inline sym(const FeMatrix& tensor) {
  FeMatrix eps;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      eps[i][j] = (tensor[i][j] + tensor[j][i]) / 2;
    }
  }
  return eps;
}

// compute tensor inner product
FeMatrix inline inner(const FeMatrix& tensor1, const FeMatrix& tensor2) {
  FeMatrix inner;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      inner[i][j] = tensor1[i][0] * tensor2[0][j] +
                    tensor1[i][1] * tensor2[1][j] +
                    tensor1[i][2] * tensor2[2][j];
    }
  }
  return inner;
}

// compute trace of a tensor
double inline trace(const FeMatrix& tensor) {
  return tensor[0][0] + tensor[1][1] + tensor[2][2];
}

void inline ComputeLinearStiffness(std::vector<double>& K,
                                   const double* pos,
                                   double E, double nu, int order) {
  int nbasis = order + 1;
  int n = pow(nbasis, 3);
  int ndof = 3*n;

  // compute quadrature points
  std::vector<double> points(nbasis);     // quadrature points
  std::vector<double> weight(nbasis);     // quadrature weights
  quadratureGaussLegendre(points.data(), weight.data(), nbasis, 0, 1);

  // compute element transformation
  double dx = (pos+3*(n-1))[0] - pos[0];
  double dy = (pos+3*(n-1))[1] - pos[1];
  double dz = (pos+3*(n-1))[2] - pos[2];
  double detJ = dx * dy * dz;
  double invJ[3] = {1.0 / dx, 1.0 / dy, 1.0 / dz};

  // compute stiffness matrix
  std::vector<std::array<double, 3> > F(n);
  double la = E * nu / (1 + nu) / (1 - 2 * nu);
  double mu = E / (2 * (1 + nu));

  // loop over quadrature points
  for (int ps=0; ps < nbasis; ps++) {
    for (int pt=0; pt < nbasis; pt++) {
      for (int pu=0; pu < nbasis; pu++) {
        double s = points[ps];
        double t = points[pt];
        double u = points[pu];
        double dvol = weight[ps] * weight[pt] * weight[pu] * detJ;
        int dof = 0;

        // cartesian product of basis functions
        for (int bx=0; bx < nbasis; bx++) {
          for (int by=0; by < nbasis; by++) {
            for (int bz=0; bz < nbasis; bz++) {
              std::array<double, 3> gradient;
              gradient[0] = dphi(s, bx, order) *  phi(t, by, order) *  phi(u, bz, order);
              gradient[1] =  phi(s, bx, order) * dphi(t, by, order) *  phi(u, bz, order);
              gradient[2] =  phi(s, bx, order) *  phi(t, by, order) * dphi(u, bz, order);
              F[dof++] = gradient;
            }
          }
        }

        if (dof != n) {  // SHOULD NOT OCCUR
          mju_error("incorrect number of basis functions");
        }

        // tensor contraction of the gradients of elastic strains
        // (d(F+F')/dx : d(F+F')/dx)
        for (int i=0; i < n; i++) {
          for (int j=0; j < n; j++) {
            FeMatrix du;
            FeMatrix dv;
            du.fill({0, 0, 0});
            dv.fill({0, 0, 0});
            for (int k=0; k < 3; k++) {
              for (int l=0; l < 3; l++) {
                du[k][0] = invJ[0] * F[i][0];
                du[k][1] = invJ[1] * F[i][1];
                du[k][2] = invJ[2] * F[i][2];
                dv[l][0] = invJ[0] * F[j][0];
                dv[l][1] = invJ[1] * F[j][1];
                dv[l][2] = invJ[2] * F[j][2];
                K[ndof*(3*i+k) + 3*j+l] -= la * trace(du) * trace(dv) * dvol;
                K[ndof*(3*i+k) + 3*j+l] -= mu * trace(inner(sym(du), sym(dv))) * dvol;
                lift::mjuu_zerovec(du[k].data(), 3);
                lift::mjuu_zerovec(dv[l].data(), 3);
              }
            }
          }
        }
      }
    }
  }
}

// compute the linear stiffness matrix for a flat 2D quad face element (membrane)
void inline ComputeLinearStiffness2D(std::vector<double>& K,
                                     const double* pos,
                                     double E, double nu, int order,
                                     double thickness, int normal_axis) {
  int nbasis = order + 1;
  int npe = nbasis * nbasis;        // nodes per face element
  int ndof = 3 * npe;

  // in-plane axes
  int axis0 = (normal_axis + 1) % 3;  // slow-varying
  int axis1 = (normal_axis + 2) % 3;  // fast-varying

  // compute quadrature points
  std::vector<double> points(nbasis);
  std::vector<double> weight(nbasis);
  quadratureGaussLegendre(points.data(), weight.data(), nbasis, 0, 1);

  // compute element transformation (diagonal Jacobian on flat face)
  double d0 = (pos + 3*(npe-1))[axis0] - pos[axis0];  // extent along axis0
  double d1 = (pos + 3*(npe-1))[axis1] - pos[axis1];  // extent along axis1
  if (d0 == 0 || d1 == 0) {
    mju_error("degenerate 2D element with zero extent");
  }
  double detJ = d0 * d1;
  double invJ0 = 1.0 / d0;
  double invJ1 = 1.0 / d1;

  // plane-stress Lamé parameter: lambda* = E*nu/(1 - nu^2)
  double la = E * nu / (1.0 - nu * nu);
  double mu = E / (2.0 * (1.0 + nu));

  // basis function gradients (2-component)
  std::vector<std::array<double, 2>> F(npe);

  // loop over quadrature points (2D)
  for (int ps = 0; ps < nbasis; ps++) {
    for (int pt = 0; pt < nbasis; pt++) {
      double s = points[ps];
      double t = points[pt];
      double dvol = weight[ps] * weight[pt] * detJ * thickness;
      int dof = 0;

      // cartesian product of 2D basis functions
      for (int b0 = 0; b0 < nbasis; b0++) {
        for (int b1 = 0; b1 < nbasis; b1++) {
          F[dof][0] = dphi(s, b0, order) *  phi(t, b1, order);
          F[dof][1] =  phi(s, b0, order) * dphi(t, b1, order);
          dof++;
        }
      }

      if (dof != npe) {
        mju_error("incorrect number of 2D basis functions");
      }

      // tensor contraction: pure membrane (in-plane strain only)
      int inplane[2] = {axis0, axis1};
      for (int i = 0; i < npe; i++) {
        for (int j = 0; j < npe; j++) {
          FeMatrix du;
          FeMatrix dv;
          du.fill({0, 0, 0});
          dv.fill({0, 0, 0});
          for (int ki = 0; ki < 2; ki++) {
            int k = inplane[ki];
            for (int li = 0; li < 2; li++) {
              int l = inplane[li];
              du[k][axis0] = invJ0 * F[i][0];
              du[k][axis1] = invJ1 * F[i][1];
              dv[l][axis0] = invJ0 * F[j][0];
              dv[l][axis1] = invJ1 * F[j][1];
              K[ndof*(3*i+k) + 3*j+l] -= la * trace(du) * trace(dv) * dvol;
              // mu (not 2*mu): same convention as 3D ComputeLinearStiffness
              K[ndof*(3*i+k) + 3*j+l] -= mu * trace(inner(sym(du), sym(dv))) * dvol;
              lift::mjuu_zerovec(du[k].data(), 3);
              lift::mjuu_zerovec(dv[l].data(), 3);
            }
          }
        }
      }
    }
  }
}

// compute the bilinear warp mode for a 2D face element
void ComputeWarpMode(double* warp, const double* pos,
                     int npe, int order, int normal_axis) {
  int ndof = 3 * npe;
  int nbasis = order + 1;

  // zero out
  std::fill(warp, warp + ndof, 0.0);

  // evaluate warp pattern (1-2s)(1-2t) at each node
  for (int b0 = 0; b0 < nbasis; b0++) {
    for (int b1 = 0; b1 < nbasis; b1++) {
      int node = b0 * nbasis + b1;
      double s = static_cast<double>(b0) / (nbasis - 1);
      double t = static_cast<double>(b1) / (nbasis - 1);
      warp[3*node + normal_axis] = (1 - 2*s) * (1 - 2*t);
    }
  }

  // orthogonalize against rigid body modes (6 modes: 3 translations + 3 rotations)
  double centroid[3] = {0, 0, 0};
  for (int n = 0; n < npe; n++) {
    for (int k = 0; k < 3; k++) {
      centroid[k] += pos[3*n + k];
    }
  }
  for (int k = 0; k < 3; k++) {
    centroid[k] /= npe;
  }

  // build and orthonormalize rigid body modes inline
  std::vector<double> rigid(6 * ndof, 0.0);

  // translations
  for (int n = 0; n < npe; n++) {
    rigid[0*ndof + 3*n + 0] = 1;
    rigid[1*ndof + 3*n + 1] = 1;
    rigid[2*ndof + 3*n + 2] = 1;
  }

  // rotations about centroid
  for (int n = 0; n < npe; n++) {
    double rx = pos[3*n + 0] - centroid[0];
    double ry = pos[3*n + 1] - centroid[1];
    double rz = pos[3*n + 2] - centroid[2];
    rigid[3*ndof + 3*n + 1] = -rz;
    rigid[3*ndof + 3*n + 2] =  ry;
    rigid[4*ndof + 3*n + 0] =  rz;
    rigid[4*ndof + 3*n + 2] = -rx;
    rigid[5*ndof + 3*n + 0] = -ry;
    rigid[5*ndof + 3*n + 1] =  rx;
  }

  // orthonormalize rigid modes via modified Gram-Schmidt
  for (int i = 0; i < 6; i++) {
    double* ri = rigid.data() + i * ndof;
    for (int j = 0; j < i; j++) {
      const double* rj = rigid.data() + j * ndof;
      double dot = 0;
      for (int k = 0; k < ndof; k++) dot += ri[k] * rj[k];
      for (int k = 0; k < ndof; k++) ri[k] -= dot * rj[k];
    }
    double norm2 = 0;
    for (int k = 0; k < ndof; k++) norm2 += ri[k] * ri[k];
    if (norm2 > 1e-20) {
      double inv_norm = 1.0 / std::sqrt(norm2);
      for (int k = 0; k < ndof; k++) ri[k] *= inv_norm;
    }
  }

  // project warp against rigid modes
  for (int i = 0; i < 6; i++) {
    const double* ri = rigid.data() + i * ndof;
    double dot = 0;
    for (int k = 0; k < ndof; k++) dot += warp[k] * ri[k];
    for (int k = 0; k < ndof; k++) warp[k] -= dot * ri[k];
  }

  // normalize
  double norm2 = 0;
  for (int k = 0; k < ndof; k++) norm2 += warp[k] * warp[k];
  if (norm2 > 1e-20) {
    double inv_norm = 1.0 / std::sqrt(norm2);
    for (int k = 0; k < ndof; k++) warp[k] *= inv_norm;
  }
}

// compute the warp bending stiffness for a 2D face element
double ComputeWarpStiffness(const double* pos, int npe, int normal_axis,
                            double E, double nu, double thickness) {
  int axis0 = (normal_axis + 1) % 3;
  int axis1 = (normal_axis + 2) % 3;
  double d0 = std::abs(pos[3*(npe-1) + axis0] - pos[axis0]);
  double d1 = std::abs(pos[3*(npe-1) + axis1] - pos[axis1]);

  if (d0 < 1e-30 || d1 < 1e-30) return 0;

  // plate bending rigidity: D = E*t³ / (12*(1-ν²))
  double D = E * thickness * thickness * thickness / (12.0 * (1.0 - nu * nu));

  // warp stiffness from twist curvature Rayleigh quotient:
  //   w^T K_bend w / |w|^2 = D*(1-ν)*4 / (d0*d1)
  return D * (1.0 - nu) * 4.0 / (d0 * d1);
}

// Eigendecompose cell stiffness matrix and store scaled eigenvectors.
int EigendecomposeStiffness(const double* K_cell_data,
                            double* out, int ndof) {
  // copy K_cell for in-place decomposition
  std::vector<double> mat(K_cell_data, K_cell_data + ndof * ndof);
  std::vector<double> eigval(ndof);
  std::vector<double> eigvec(ndof * ndof);

  lift::mjuu_eigendecompose(mat.data(), eigval.data(), eigvec.data(), ndof);

  // K_stored = -K_physical, so physical eigenvalue = -eigval[i]
  double max_eigval = 0;
  for (int i = 0; i < ndof; i++) {
    max_eigval = std::max(max_eigval, std::abs(eigval[i]));
  }

  double threshold = max_eigval * 1e-8;
  int neig = 0;
  for (int i = 0; i < ndof; i++) {
    double lambda_phys = -eigval[i];  // negate to get physical eigenvalue
    if (lambda_phys > threshold) {
      // store sqrt(λ) * eigenvector (column i of eigvec matrix)
      double scale = std::sqrt(lambda_phys);
      double* w = out + 1 + neig * ndof;
      for (int j = 0; j < ndof; j++) {
        w[j] = scale * eigvec[j * ndof + i];
      }
      neig++;
    }
  }

  out[0] = static_cast<double>(neig);
  return neig;
}

// compute interpolated shell bending edge data
void ComputeInterpBending(
    std::vector<double>& bending,
    const std::vector<double>& nodexpos_local,
    int order, const int cellcount[3],
    double young, double poisson, double thickness) {
  // bending modulus D = E * t^3 / (12 * (1 - nu^2))
  double D_bend = young * thickness * thickness * thickness /
                  (12.0 * (1.0 - poisson * poisson));

  int cx = cellcount[0], cy = cellcount[1], cz = cellcount[2];
  int ny_global = cy * order + 1;
  int nz_global = cz * order + 1;
  int npe = (order + 1) * (order + 1);  // nodes per 2D face element

  int face_sizes[6] = {cy*cz, cy*cz, cx*cz, cx*cz, cx*cy, cx*cy};
  int face_normal[6] = {0, 0, 1, 1, 2, 2};
  int face_count1[6] = {cz, cz, cx, cx, cy, cy};
  int face_fixed[6] = {0, cx*order, 0, cy*order, 0, cz*order};

  // gather node positions for one face element
  auto gather_face_nodes = [&](int face_id, int within_face,
                               std::vector<double>& fpos) {
    int nax = face_normal[face_id];
    int a0 = (nax + 1) % 3;
    int a1 = (nax + 2) % 3;
    int c1 = face_count1[face_id];
    int gf = face_fixed[face_id];
    int q0 = within_face / c1;
    int q1 = within_face % c1;
    fpos.resize(3 * npe);
    int loc = 0;
    for (int l0 = 0; l0 <= order; l0++) {
      for (int l1 = 0; l1 <= order; l1++) {
        int g[3];
        g[nax] = gf;
        g[a0] = q0 * order + l0;
        g[a1] = q1 * order + l1;
        int gidx = g[0] * ny_global * nz_global + g[1] * nz_global + g[2];
        lift::mjuu_copyvec(fpos.data() + 3*loc, &nodexpos_local[3*gidx], 3);
        loc++;
      }
    }
  };

  // compute unnormalized normal and tangents at a parametric point
  auto compute_normal = [&](const std::vector<double>& fpos,
                            const double local[2],
                            double normal[3], double t1[3], double t2[3]) {
    lift::mjuu_zerovec(t1, 3);
    lift::mjuu_zerovec(t2, 3);
    int idx = 0;
    for (int l0 = 0; l0 <= order; l0++) {
      for (int l1 = 0; l1 <= order; l1++) {
        double g0 = dphi(local[0], l0, order) * phi(local[1], l1, order);
        double g1 = phi(local[0], l0, order) * dphi(local[1], l1, order);
        for (int d = 0; d < 3; d++) {
          t1[d] += fpos[3*idx + d] * g0;
          t2[d] += fpos[3*idx + d] * g1;
        }
        idx++;
      }
    }
    lift::mjuu_crossvec(normal, t1, t2);
  };

  // face cumulative offsets
  int face_cumul[6];
  face_cumul[0] = 0;
  for (int f = 1; f < 6; f++) {
    face_cumul[f] = face_cumul[f-1] + face_sizes[f-1];
  }

  int face_count0[6];
  for (int f = 0; f < 6; f++) {
    face_count0[f] = face_sizes[f] / face_count1[f];
  }

  int cells[3] = {cx, cy, cz};

  auto get_neighbor = [&](int fid, int q0, int q1, int dir, int side, double local_A[2],
                          double local_B[2]) -> std::pair<int, int> {
    int nax = fid / 2, sign_f = fid % 2;
    int a0 = (nax+1)%3, a1 = (nax+2)%3;
    int nc1 = face_count1[fid];

    local_A[0] = (dir == 0) ? (side > 0 ? 1.0 : 0.0) : 0.5;
    local_A[1] = (dir == 1) ? (side > 0 ? 1.0 : 0.0) : 0.5;

    int q_nb = (dir == 0 ? q0 : q1) + side;
    int q_max = (dir == 0) ? face_count0[fid] : nc1;
    if (q_nb >= 0 && q_nb < q_max) {
      int q0_B = (dir == 0) ? q_nb : q0;
      int q1_B = (dir == 0) ? q1 : q_nb;
      local_B[0] = (dir == 0) ? (side > 0 ? 0.0 : 1.0) : 0.5;
      local_B[1] = (dir == 1) ? (side > 0 ? 0.0 : 1.0) : 0.5;
      return {fid, q0_B * nc1 + q1_B};
    }

    int ax = (dir == 0) ? a0 : a1;           // axis being crossed
    int fid_B = 2*ax + (side > 0 ? 1 : 0);  // neighboring face
    int nc1_B = face_count1[fid_B];

    int q_run = (dir == 0) ? q1 : q0;
    int q_boundary = sign_f ? (cells[nax]-1) : 0;
    int q0_B, q1_B;
    if (dir == 0) {
      q0_B = q_run;
      q1_B = q_boundary;
      local_B[0] = 0.5;
      local_B[1] = sign_f ? 1.0 : 0.0;
    } else {
      q0_B = q_boundary;
      q1_B = q_run;
      local_B[0] = sign_f ? 1.0 : 0.0;
      local_B[1] = 0.5;
    }
    return {fid_B, q0_B * nc1_B + q1_B};
  };

  struct BendEdge {
    int fe_A, fe_B;          // global face element indices (for runtime)
    int fid_A, fid_B;        // face id (0-5)
    int within_A, within_B;  // within-face element index
    double local_A[2];
    double local_B[2];
  };
  std::vector<BendEdge> edges;

  for (int f = 0; f < 6; f++) {
    int nc0 = face_count0[f];
    int nc1 = face_count1[f];
    for (int q0 = 0; q0 < nc0; q0++) {
      for (int q1 = 0; q1 < nc1; q1++) {
        int within_A = q0 * nc1 + q1;
        int fe_A = face_cumul[f] + within_A;

        for (int dir = 0; dir < 2; dir++) {
          for (int side = -1; side <= 1; side += 2) {
            double lA[2], lB[2];
            auto [fid_B, within_B] = get_neighbor(f, q0, q1, dir, side, lA, lB);
            int fe_B = face_cumul[fid_B] + within_B;
            if (fe_A < fe_B) {
              BendEdge e;
              e.fe_A = fe_A;  e.fid_A = f;      e.within_A = within_A;
              e.fe_B = fe_B;  e.fid_B = fid_B;  e.within_B = within_B;
              lift::mjuu_copyvec(e.local_A, lA, 2);
              lift::mjuu_copyvec(e.local_B, lB, 2);
              edges.push_back(e);
            }
          }
        }
      }
    }
  }

  // compute per-edge bending data
  const int BEND_EDGE_SIZE = 10;  // should match engine_passive.c
  bending.resize(1 + edges.size() * BEND_EDGE_SIZE, 0);
  bending[0] = static_cast<double>(edges.size());

  for (int e = 0; e < (int)edges.size(); e++) {
    const BendEdge& edge = edges[e];
    std::vector<double> fpos_A, fpos_B;
    gather_face_nodes(edge.fid_A, edge.within_A, fpos_A);
    gather_face_nodes(edge.fid_B, edge.within_B, fpos_B);

    double n_A[3], t1_A[3], t2_A[3];
    double n_B[3], t1_B[3], t2_B[3];
    compute_normal(fpos_A, edge.local_A, n_A, t1_A, t2_A);
    compute_normal(fpos_B, edge.local_B, n_B, t1_B, t2_B);

    double len_A = lift::mjuu_normvec(n_A, 3);
    double len_B = lift::mjuu_normvec(n_B, 3);
    if (len_A < 1e-12 || len_B < 1e-12) continue;

    double dn0[3] = {n_A[0]-n_B[0], n_A[1]-n_B[1], n_A[2]-n_B[2]};

    double h_A, l_A, h_B, l_B;
    if (edge.local_A[0] == 0.5) {
      l_A = lift::mjuu_normvec(t1_A, 3);
      h_A = lift::mjuu_normvec(t2_A, 3);
    } else {
      h_A = lift::mjuu_normvec(t1_A, 3);
      l_A = lift::mjuu_normvec(t2_A, 3);
    }
    if (edge.local_B[0] == 0.5) {
      l_B = lift::mjuu_normvec(t1_B, 3);
      h_B = lift::mjuu_normvec(t2_B, 3);
    } else {
      h_B = lift::mjuu_normvec(t1_B, 3);
      l_B = lift::mjuu_normvec(t2_B, 3);
    }
    double h_avg = (h_A + h_B) / 2;
    double l_avg = (l_A + l_B) / 2;
    double stiffness_coeff = D_bend * l_avg / mjMAX(h_avg, 1e-12);

    double* edata = bending.data() + 1 + e * BEND_EDGE_SIZE;
    edata[0] = static_cast<double>(edge.fe_A);
    edata[1] = static_cast<double>(edge.fe_B);
    edata[2] = edge.local_A[0];
    edata[3] = edge.local_A[1];
    edata[4] = edge.local_B[0];
    edata[5] = edge.local_B[1];
    edata[6] = stiffness_coeff;
    edata[7] = dn0[0];
    edata[8] = dn0[1];
    edata[9] = dn0[2];
  }
}

// ComputeCellEmpty (user_mesh.cc:5285): identify cells with no mesh content from
// vertex/element geometry. Retargeted from the mjCFlex method to a free function
// writing `cell_empty`; algorithm verbatim.
void FlexComputeCellEmpty(const double* vpos, const int* elems, int nv, int ne,
                          int fdim, const int cellcount[3],
                          std::vector<char>& cell_empty,
                          const double* bbox = nullptr) {
  int cx = cellcount[0];
  int cy = cellcount[1];
  int cz = cellcount[2];
  int ncells = cx * cy * cz;

  double minmax[6];
  if (bbox) {
    for (int j = 0; j < 6; j++) minmax[j] = bbox[j];
  } else {
    minmax[0] = minmax[1] = minmax[2] = 1e30;
    minmax[3] = minmax[4] = minmax[5] = -1e30;
    for (int i = 0; i < nv; i++) {
      for (int j = 0; j < 3; j++) {
        minmax[j+0] = std::min(minmax[j+0], vpos[3*i+j]);
        minmax[j+3] = std::max(minmax[j+3], vpos[3*i+j]);
      }
    }
  }

  double dx = minmax[3] - minmax[0];
  double dy = minmax[4] - minmax[1];
  double dz = minmax[5] - minmax[2];

  std::vector<bool> has_element(ncells, false);
  int nvpe = fdim + 1;

  if (nvpe > 0 && ne > 0) {
    for (int e = 0; e < ne; e++) {
      double elo[3] = {1e30, 1e30, 1e30};
      double ehi[3] = {-1e30, -1e30, -1e30};
      for (int v = 0; v < nvpe; v++) {
        int vid = elems[nvpe * e + v];
        for (int j = 0; j < 3; j++) {
          elo[j] = std::min(elo[j], vpos[3 * vid + j]);
          ehi[j] = std::max(ehi[j], vpos[3 * vid + j]);
        }
      }

      auto cellIdx = [](double coord, double lo, double d, int nc) {
        if (d <= 0) return 0;
        int c = (int)((coord - lo) / d * nc);
        return std::max(0, std::min(nc - 1, c));
      };

      int ci0 = cellIdx(elo[0], minmax[0], dx, cx);
      int ci1 = cellIdx(ehi[0], minmax[0], dx, cx);
      int cj0 = cellIdx(elo[1], minmax[1], dy, cy);
      int cj1 = cellIdx(ehi[1], minmax[1], dy, cy);
      int ck0 = cellIdx(elo[2], minmax[2], dz, cz);
      int ck1 = cellIdx(ehi[2], minmax[2], dz, cz);

      for (int ci = ci0; ci <= ci1; ci++) {
        for (int cj = cj0; cj <= cj1; cj++) {
          for (int ck = ck0; ck <= ck1; ck++) {
            has_element[ci * cy * cz + cj * cz + ck] = true;
          }
        }
      }
    }
  }

  cell_empty.assign(ncells, 0);

  if (fdim == 2 && nvpe == 3 && ne > 0) {
    std::vector<bool> visited(ncells, false);
    std::queue<std::array<int, 3>> bfs;

    for (int ci = 0; ci < cx; ci++) {
      for (int cj = 0; cj < cy; cj++) {
        for (int ck = 0; ck < cz; ck++) {
          if (ci == 0 || ci == cx - 1 ||
              cj == 0 || cj == cy - 1 ||
              ck == 0 || ck == cz - 1) {
            int idx = ci * cy * cz + cj * cz + ck;
            if (!has_element[idx] && !visited[idx]) {
              visited[idx] = true;
              cell_empty[idx] = 1;
              bfs.push({ci, cj, ck});
            }
          }
        }
      }
    }

    const int dirs[6][3] = {
        {-1, 0, 0}, {1, 0, 0},  {0, -1, 0},
        {0, 1, 0},  {0, 0, -1}, {0, 0, 1}};
    while (!bfs.empty()) {
      auto [ci, cj, ck] = bfs.front();
      bfs.pop();
      for (auto& d : dirs) {
        int ni = ci + d[0], nj = cj + d[1], nk = ck + d[2];
        if (ni < 0 || ni >= cx ||
            nj < 0 || nj >= cy ||
            nk < 0 || nk >= cz) {
          continue;
        }
        int nidx = ni * cy * cz + nj * cz + nk;
        if (!visited[nidx] && !has_element[nidx]) {
          visited[nidx] = true;
          cell_empty[nidx] = 1;
          bfs.push({ni, nj, nk});
        }
      }
    }
  } else {
    for (int c = 0; c < ncells; c++) {
      cell_empty[c] = !has_element[c];
    }
  }
}

// ComputeUnrotatedNodePositions (user_mesh.cc:5202): apply the inverse grid
// rotation R0 to node positions so stiffness eigenvectors see axis-aligned
// coordinates. Retargeted from the mjCFlex method; algorithm verbatim, the
// non-orthonormal-R0 throw lowered to a diagnostic (returns false).
bool FlexComputeUnrotatedNodePositions(const std::vector<double>& nodexpos,
                                       int nnode, int order,
                                       const int cellcount[3],
                                       const std::vector<char>& cell_empty,
                                       std::vector<double>& nodexpos_local,
                                       double* R0_out,
                                       std::vector<ps::mjcf::Diagnostic>& diags) {
  nodexpos_local.assign(3*nnode, 0);
  if (nnode <= 0) return true;
  int ny_global = cellcount[1] * order + 1;
  int nz_global = cellcount[2] * order + 1;

  int cx = cellcount[0], cy = cellcount[1], cz = cellcount[2];
  int ref_ci = 0, ref_cj = 0, ref_ck = 0;
  bool found = false;
  for (int ci = 0; ci < cx && !found; ci++) {
    for (int cj = 0; cj < cy && !found; cj++) {
      for (int ck = 0; ck < cz && !found; ck++) {
        int cell_idx = ci * cy * cz + cj * cz + ck;
        if (cell_empty.empty() || !cell_empty[cell_idx]) {
          ref_ci = ci; ref_cj = cj; ref_ck = ck;
          found = true;
        }
      }
    }
  }

  int g000 = (ref_ci * order) * ny_global * nz_global +
             (ref_cj * order) * nz_global +
             (ref_ck * order);
  int g100 = ((ref_ci * order) + order) * ny_global * nz_global +
             (ref_cj * order) * nz_global +
             (ref_ck * order);
  int g010 = (ref_ci * order) * ny_global * nz_global +
             ((ref_cj * order) + order) * nz_global +
             (ref_ck * order);
  int g001 = (ref_ci * order) * ny_global * nz_global +
             (ref_cj * order) * nz_global +
             ((ref_ck * order) + order);

  double R0[9];
  for (int d = 0; d < 3; d++) {
    R0[0+d] = nodexpos[3*g100 + d] - nodexpos[3*g000 + d];
    R0[3+d] = nodexpos[3*g010 + d] - nodexpos[3*g000 + d];
    R0[6+d] = nodexpos[3*g001 + d] - nodexpos[3*g000 + d];
  }

  double li = lift::mjuu_normvec(R0+0, 3);
  double lj = lift::mjuu_normvec(R0+3, 3);
  double lk = lift::mjuu_normvec(R0+6, 3);
  (void)li; (void)lj; (void)lk;

  for (int a = 0; a < 3; a++) {
    for (int b = a; b < 3; b++) {
      double dot = lift::mjuu_dot3(R0 + 3*a, R0 + 3*b);
      double expected = (a == b) ? 1.0 : 0.0;
      if (std::abs(dot - expected) > 1e-8) {
        diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "flex",
                         "flex grid rotation R0 is not orthonormal", {}});
        return false;
      }
    }
  }

  if (R0_out) {
    lift::mjuu_copyvec(R0_out, R0, 9);
  }

  for (int i = 0; i < nnode; i++) {
    const double* p = nodexpos.data() + 3*i;
    double* q = nodexpos_local.data() + 3*i;
    lift::mjuu_mulvecmat(q, p, R0);
  }
  return true;
}

// CreateFlapStencil (user_mesh.cc:3605): builds triangle flap adjacency and
// asserts the edge indexing matches the one computed in Compile. Returns false
// (with a diagnostic) on mismatch -- the only observable effect at young=0.
bool FlexCreateFlapStencil(std::vector<StencilFlap>& flaps,
                           const std::vector<int>& simplex,
                           const std::vector<int>& edgeidx,
                           std::vector<ps::mjcf::Diagnostic>& diags) {
  int ne = 0;
  const int nt = static_cast<int>(simplex.size()) / 3;
  std::vector<std::array<int, 3>> elem_verts(nt);
  for (int t = 0; t < nt; t++)
    for (int v = 0; v < 3; v++) elem_verts[t][v] = simplex[3 * t + v];

  std::unordered_map<std::pair<int, int>, int, FlexPairHash> edge_indices;
  for (int t = 0; t < nt; t++) {
    const int* v = elem_verts[t].data();
    for (int e = 0; e < 3; e++) {
      auto pair = std::pair(std::min(v[kFlapEdge[e][0]], v[kFlapEdge[e][1]]),
                            std::max(v[kFlapEdge[e][0]], v[kFlapEdge[e][1]]));
      auto [it, inserted] = edge_indices.insert({pair, ne});
      int edge_id;
      if (inserted) {
        StencilFlap flap;
        flap.vertices[0] = v[kFlapEdge[e][0]];
        flap.vertices[1] = v[kFlapEdge[e][1]];
        flap.vertices[2] = v[(kFlapEdge[e][1] + 1) % 3];
        flap.vertices[3] = -1;
        flaps.push_back(flap);
        edge_id = ne++;
      } else {
        edge_id = it->second;
        flaps[it->second].vertices[3] = v[(kFlapEdge[e][1] + 1) % 3];
      }
      if (!edgeidx.empty() && edge_id != edgeidx[3 * t + e]) {
        diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "flex",
                         "edge indices do not match in CreateFlapStencil", {}});
        return false;
      }
    }
  }
  return true;
}

// CreateShellPair (user_mesh.cc:5460): border fragments (shell), element layer
// distance-from-border (3D value iteration), and element-vertex collision pairs.
void FlexCreateShellPair(CFlex& f) {
  const int dim = f.dim;
  const int nelem = f.nelem;
  std::vector<std::vector<int>> fragspec(nelem * (dim + 1));
  std::vector<std::vector<int>> connectspec;
  std::vector<bool> border(nelem, false);
  std::vector<bool> borderfrag(nelem * (dim + 1), false);

  for (int e = 0; e < nelem; e++) {
    int n = e * (dim + 1);
    std::vector<int> el(f.elem.begin() + n, f.elem.begin() + n + dim + 1);
    if (dim == 1) {
      fragspec[n] = {el[0], e, el[0]};
      fragspec[n + 1] = {el[1], e, el[1]};
    } else if (dim == 2) {
      fragspec[n] = {el[0], el[1], e, el[0], el[1]};
      fragspec[n + 2] = {el[1], el[2], e, el[1], el[2]};
      fragspec[n + 1] = {el[2], el[0], e, el[2], el[0]};
    } else {
      fragspec[n] = {el[0], el[1], el[2], e, el[0], el[1], el[2]};
      fragspec[n + 2] = {el[0], el[2], el[3], e, el[0], el[2], el[3]};
      fragspec[n + 1] = {el[0], el[3], el[1], e, el[0], el[3], el[1]};
      fragspec[n + 3] = {el[1], el[3], el[2], e, el[1], el[3], el[2]};
    }
  }

  if (dim > 1)
    for (int n = 0; n < nelem * (dim + 1); n++)
      std::sort(fragspec[n].begin(), fragspec[n].begin() + dim);
  std::sort(fragspec.begin(), fragspec.end());

  int cnt = 1;
  for (int n = 1; n < nelem * (dim + 1); n++) {
    std::vector<int> previous(fragspec[n - 1].begin(), fragspec[n - 1].begin() + dim);
    std::vector<int> current(fragspec[n].begin(), fragspec[n].begin() + dim);
    if (previous == current) {
      std::vector<int> connect;
      connect.push_back(fragspec[n - 1][dim]);
      connect.push_back(fragspec[n][dim]);
      connect.insert(connect.end(), fragspec[n].begin(), fragspec[n].begin() + dim);
      connectspec.push_back(connect);
      cnt++;
    } else {
      if (cnt == 1) {
        border[fragspec[n - 1][dim]] = true;
        borderfrag[n - 1] = true;
      }
      cnt = 1;
    }
  }
  if (cnt == 1) {
    int n = nelem * (dim + 1);
    border[fragspec[n - 1][dim]] = true;
    borderfrag[n - 1] = true;
  }

  for (unsigned i = 0; i < borderfrag.size(); i++)
    if (borderfrag[i])
      f.shell.insert(f.shell.end(), fragspec[i].begin() + dim + 1, fragspec[i].end());

  if (dim < 3) {
    f.elemlayer.assign(nelem, 0);
  } else {
    f.elemlayer.assign(nelem, nelem + 1);
    for (int e = 0; e < nelem; e++)
      if (border[e]) f.elemlayer[e] = 0;
    bool change = true;
    while (change) {
      change = false;
      for (const auto& connect : connectspec) {
        int e1 = connect[0], e2 = connect[1];
        if (f.elemlayer[e1] > f.elemlayer[e2] + 1) {
          f.elemlayer[e1] = f.elemlayer[e2] + 1;
          change = true;
        } else if (f.elemlayer[e2] > f.elemlayer[e1] + 1) {
          f.elemlayer[e2] = f.elemlayer[e1] + 1;
          change = true;
        }
      }
    }
  }

  if (dim < 3) {
    for (const auto& connect : connectspec) {
      if (border[connect[0]] || border[connect[1]]) {
        std::vector<int> frag(connect.begin() + 2, connect.end());
        for (int ei = 0; ei < 2; ei++) {
          const int* edata = f.elem.data() + connect[ei] * (dim + 1);
          for (int i = 0; i <= dim; i++) {
            if (std::find(frag.begin(), frag.end(), edata[i]) == frag.end()) {
              f.evpair.push_back(connect[1 - ei]);
              f.evpair.push_back(edata[i]);
              break;
            }
          }
        }
      }
    }
  }
}

// mjCFlex::Compile (young=0, non-interpolated). Fills `out`; returns false with
// a diagnostic on a compile error (the differential harness pairs these with the
// XML reader's own mjCError so parity holds). `bodyid_of`/`matid_of` are the
// native name->id maps; `cbs` supplies body xpos0/xquat0.
bool FlexCompile(const Flex& fl, const std::vector<CBody>& cbs,
                 const std::unordered_map<std::string, int>& bodyid_of,
                 const std::unordered_map<std::string, int>& matid_of,
                 CFlex& out, std::vector<ps::mjcf::Diagnostic>& diags,
                 const std::vector<double>& node_local = {},
                 bool has_strain_eq = false,
                 const std::vector<char>& cell_empty = {}) {
  auto err = [&](const char* msg) {
    diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "flex", msg, fl.loc});
    return false;
  };

  // Flex is not auto-named on the XML path (bridge AutoNameableFamily), so an
  // unnamed flex keeps an empty name -- exactly like leg B.
  out.name = fl.name ? *fl.name : std::string();
  out.dim = fl.dim ? *fl.dim : 2;
  out.radius = fl.radius ? *fl.radius : 0.005;
  out.group = fl.group ? *fl.group : 0;
  out.flatskin = fl.flatskin && *fl.flatskin;
  if (fl.rgba) for (int k = 0; k < 4; ++k) out.rgba[k] = (*fl.rgba)[k];
  out.material_name = fl.material ? *fl.material : std::string();
  out.has_strain_eq = has_strain_eq;
  if (fl.cellcount)
    for (int k = 0; k < 3; ++k) out.cellcount[k] = (*fl.cellcount)[k];
  // interpolation order from dof (reader: quadratic=2, trilinear=1, else 0).
  if (fl.dof) {
    if (*fl.dof == FlexDof::quadratic) out.order = 2;
    else if (*fl.dof == FlexDof::trilinear) out.order = 1;
    else out.order = 0;
  }
  // nodebody name list (the flex "node" attribute is body names, not positions).
  std::vector<std::string> nodebody =
      SplitWhitespace(fl.node ? *fl.node : std::string());
  out.interpolated = !nodebody.empty();
  out.node = node_local;
  out.cell_empty = cell_empty;

  // contact / edge / elasticity children (single, optional).
  if (!fl.flexContacts.empty() && fl.flexContacts.front()) {
    const FlexContact& c = *fl.flexContacts.front();
    if (c.contype) out.contype = *c.contype;
    if (c.conaffinity) out.conaffinity = *c.conaffinity;
    if (c.condim) out.condim = *c.condim;
    if (c.priority) out.priority = *c.priority;
    if (c.friction)
      for (std::size_t k = 0; k < c.friction->size() && k < 3; ++k)
        out.friction[k] = (*c.friction)[k];
    if (c.solmix) out.solmix = *c.solmix;
    if (c.solref)
      for (std::size_t k = 0; k < c.solref->size() && k < 2; ++k)
        out.solref[k] = (*c.solref)[k];
    if (c.solimp)
      for (std::size_t k = 0; k < c.solimp->size() && k < 5; ++k)
        out.solimp[k] = (*c.solimp)[k];
    if (c.margin) out.margin = *c.margin;
    if (c.gap) out.gap = *c.gap;
    if (c.internal) out.internal = *c.internal;
    if (c.selfcollide) out.selfcollide = static_cast<int>(*c.selfcollide);
    if (c.activelayers) out.activelayers = *c.activelayers;
    if (c.passive) out.passive = *c.passive;
  }
  if (!fl.flexEdges.empty() && fl.flexEdges.front()) {
    const FlexEdge& e = *fl.flexEdges.front();
    if (e.stiffness) out.edgestiffness = *e.stiffness;
    if (e.damping) out.edgedamping = *e.damping;
  }
  if (!fl.flexElasticitys.empty() && fl.flexElasticitys.front()) {
    const FlexElasticity& e = *fl.flexElasticitys.front();
    if (e.young) out.young = *e.young;
    if (e.poisson) out.poisson = *e.poisson;
    if (e.damping) out.damping = *e.damping;
    out.thickness = e.thickness ? *e.thickness : -1;
    if (e.elastic2d) out.elastic2d = static_cast<int>(*e.elastic2d);
  }

  const int dim = out.dim;
  std::vector<std::string> vertbody =
      SplitWhitespace(fl.body ? *fl.body : std::string());
  if (fl.vertex) out.vert = *fl.vertex;
  if (fl.element) out.elem = *fl.element;
  if (fl.texcoord) out.texcoord = *fl.texcoord;
  if (fl.elemtexcoord) out.elemtexcoord = *fl.elemtexcoord;

  // checks (user_mesh.cc:4634-4672)
  if (dim < 1 || dim > 3) return err("dim must be 1, 2 or 3");
  if (out.elem.empty()) return err("elem is empty");
  if (out.elem.size() % (dim + 1)) return err("elem size must be multiple of (dim+1)");
  if (vertbody.empty() && !out.interpolated)
    return err("vertbody and nodebody are both empty");
  if (out.vert.size() % 3) return err("vert size must be a multiple of 3");
  if (out.edgestiffness > 0 && dim > 1)
    return err("edge stiffness only available for dim=1");
  if (out.interpolated && out.selfcollide != mjFLEXSELF_NONE)
    return err("trilinear interpolation cannot do self-collision");
  if (out.interpolated && out.internal)
    return err("trilinear interpolation cannot do internal collisions");
  out.nelem = static_cast<int>(out.elem.size()) / (dim + 1);

  // nvert, rigid (user_mesh.cc:4674-4689)
  if (out.vert.empty()) {
    out.centered = true;
    out.nvert = static_cast<int>(vertbody.size());
  } else {
    out.nvert = static_cast<int>(out.vert.size()) / 3;
    if (vertbody.size() == 1)
      out.rigid = true;
    else if (static_cast<int>(vertbody.size()) != out.nvert)
      return err("vertbody size must be 1 or nvert");
  }
  if (out.nvert < dim + 1) return err("not enough vertices");

  // nnode + interpolation order / cellcount checks (user_mesh.cc:4691-4712)
  out.nnode = static_cast<int>(nodebody.size());
  if (out.nnode && !out.order)
    return err("Interpolation order must be explicitly specified (dof is missing)");
  if (out.order > 0) {
    if (out.cellcount[0] == 0 || out.cellcount[1] == 0 || out.cellcount[2] == 0)
      return err("cellcount cannot be 0 in any dimension when interpolation order > 0");
    int expected_nodes = (out.cellcount[0] * out.order + 1) *
                         (out.cellcount[1] * out.order + 1) *
                         (out.cellcount[2] * out.order + 1);
    if (out.nnode != expected_nodes)
      return err("number of nodes does not match cellcount and dof");
  }

  // elem vertex-id range (user_mesh.cc:4714-4719)
  for (int v : out.elem)
    if (v < 0 || v >= out.nvert) return err("elem vertex id out of range");

  // texcoord check (user_mesh.cc:4721-4730)
  if (!out.texcoord.empty() &&
      static_cast<int>(out.texcoord.size()) != 2 * out.nvert &&
      out.elemtexcoord.empty())
    return err("two texture coordinates per vertex expected");
  if (out.elemtexcoord.empty() && !out.texcoord.empty()) {
    out.elemtexcoord.assign((dim + 1) * out.nelem, 0);
    std::memcpy(out.elemtexcoord.data(), out.elem.data(),
                (dim + 1) * out.nelem * sizeof(int));
  }

  // material (user_mesh.cc:4732-4738)
  if (!out.material_name.empty()) {
    auto it = matid_of.find(out.material_name);
    if (it == matid_of.end()) return err("unknown material in flex");
    out.matid = it->second;
  }

  // resolve vertex body ids (ResolveReferences, user_mesh.cc:4275)
  for (const std::string& nm : vertbody) {
    auto it = bodyid_of.find(nm);
    if (it == bodyid_of.end()) return err("unknown body in flex");
    out.vertbodyid.push_back(it->second);
  }
  // resolve node body ids (interpolated, user_mesh.cc:4292)
  for (const std::string& nm : nodebody) {
    auto it = bodyid_of.find(nm);
    if (it == bodyid_of.end()) return err("unknown body in flex");
    out.nodebodyid.push_back(it->second);
  }

  // repeated-vertex-in-element check (user_mesh.cc:4744-4756)
  for (int e = 0; e < out.nelem; e++) {
    std::vector<int> el(out.elem.begin() + e * (dim + 1),
                        out.elem.begin() + (e + 1) * (dim + 1));
    std::sort(el.begin(), el.end());
    for (int k = 0; k < dim; k++)
      if (el[k] == el[k + 1]) return err("repeated vertex in element");
  }

  // finalize rigid / centered (user_mesh.cc:4758-4778)
  if (!out.rigid && !out.interpolated) {
    out.rigid = true;
    for (std::size_t i = 1; i < out.vertbodyid.size(); i++)
      if (out.vertbodyid[i] != out.vertbodyid[0]) { out.rigid = false; break; }
  }
  if (!out.centered && !out.interpolated) {
    out.centered = true;
    for (double v : out.vert)
      if (v != 0) { out.centered = false; break; }
  }
  // interpolated: centered derives from the node local offsets (user_mesh.cc:4780)
  if (!out.centered && out.interpolated) {
    out.centered = true;
    for (double v : out.node)
      if (v != 0) { out.centered = false; break; }
  }

  // global vertex positions (user_mesh.cc:4790-4809)
  out.vertxpos.assign(3 * out.nvert, 0);
  for (int i = 0; i < out.nvert; i++) {
    int b = out.rigid ? out.vertbodyid[0] : out.vertbodyid[i];
    lift::mjuu_copyvec(out.vertxpos.data() + 3 * i, cbs[b].xpos0, 3);
    if (!out.centered || out.interpolated) {
      double offset[3];
      lift::mjuu_rotVecQuat(offset, out.vert.data() + 3 * i, cbs[b].xquat0);
      lift::mjuu_addtovec(out.vertxpos.data() + 3 * i, offset, 3);
    }
    // hack (user_mesh.cc:4823): interpolated vertbodyid is the parent, reset to -1
    if (out.interpolated) out.vertbodyid[i] = -1;
  }

  // global node positions (user_mesh.cc:4811-4824)
  std::vector<double> nodexpos(3 * out.nnode, 0);
  for (int i = 0; i < out.nnode; i++) {
    int b = out.nodebodyid[i];
    lift::mjuu_copyvec(nodexpos.data() + 3 * i, cbs[b].xpos0, 3);
    if (!out.centered) {
      double offset[3];
      lift::mjuu_rotVecQuat(offset, out.node.data() + 3 * i, cbs[b].xquat0);
      lift::mjuu_addtovec(nodexpos.data() + 3 * i, offset, 3);
    }
  }

  // unrotated node positions for stiffness (user_mesh.cc:4826-4828)
  double R0[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<double> nodexpos_local;
  if (out.interpolated) {
    if (!FlexComputeUnrotatedNodePositions(nodexpos, out.nnode, out.order,
                                           out.cellcount, out.cell_empty,
                                           nodexpos_local, R0, diags))
      return false;
  }

  // tetrahedron reorientation (user_mesh.cc:4832-4853)
  if (dim == 3) {
    for (int e = 0; e < out.nelem; e++) {
      const int* edata = out.elem.data() + e * (dim + 1);
      double* v0 = out.vertxpos.data() + 3 * edata[0];
      double* v1 = out.vertxpos.data() + 3 * edata[1];
      double* v2 = out.vertxpos.data() + 3 * edata[2];
      double* v3 = out.vertxpos.data() + 3 * edata[3];
      double v01[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
      double v02[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
      double v03[3] = {v3[0] - v0[0], v3[1] - v0[1], v3[2] - v0[2]};
      double nrm[3];
      lift::mjuu_crossvec(nrm, v01, v02);
      if (lift::mjuu_dot3(nrm, v03) > 0)
        std::swap(out.elem[e * (dim + 1) + 1], out.elem[e * (dim + 1) + 2]);
    }
  }

  // edges (user_mesh.cc:4855-4882)
  const int nedge_per = kFlexNumEdges[dim - 1];
  out.edgeidx.assign(out.elem.size() * nedge_per / (dim + 1), 0);
  std::unordered_map<std::pair<int, int>, int, FlexPairHash> edge_indices;
  int nedge = 0;
  for (unsigned f = 0; f < out.elem.size() / (dim + 1); f++) {
    int* v = out.elem.data() + f * (dim + 1);
    for (int e = 0; e < nedge_per; e++) {
      auto pair = std::pair(
          std::min(v[kFlexEledge[dim - 1][e][0]], v[kFlexEledge[dim - 1][e][1]]),
          std::max(v[kFlexEledge[dim - 1][e][0]], v[kFlexEledge[dim - 1][e][1]]));
      auto [it, inserted] = edge_indices.insert({pair, nedge});
      if (inserted) {
        out.edge.push_back(pair);
        out.edgeidx[f * nedge_per + e] = nedge++;
      } else {
        out.edgeidx[f * nedge_per + e] = it->second;
      }
    }
  }
  out.nedge = static_cast<int>(out.edge.size());

  // flap stencil (dim==2): throw-on-mismatch only at young=0 (user_mesh.cc:4885)
  if (dim == 2 &&
      !FlexCreateFlapStencil(out.flaps, out.elem, out.edgeidx, diags))
    return false;

  // elasticity (user_mesh.cc:4889-4925, non-interpolated). young>0 fills the
  // per-element strain metric (stiffness, 21*nelem stride: 6 of 21 used in 2D,
  // all 21 in 3D) and, for 2D bending (elastic2d 1 or 3), the per-edge bending
  // stiffness (17*nedge). Interpolated dof is gated to the XML fallback.
  if (out.young > 0) {
    if (out.poisson < 0 || out.poisson >= 0.5)
      return err("Poisson ratio must be in [0, 0.5)");

    // linear elasticity (non-interpolated)
    if (!out.interpolated) out.stiffness.assign(21 * out.nelem, 0);

    // geometrically nonlinear elasticity
    for (int t = 0; t < out.nelem; t++) {
      if (out.interpolated) continue;
      if (dim == 2 && out.elastic2d >= 2 && out.thickness > 0) {
        ComputeStiffness<Stencil2D>(out.stiffness, out.vertxpos,
                                    out.elem.data() + (dim + 1) * t, t,
                                    out.young, out.poisson, out.thickness);
      } else if (dim == 3) {
        ComputeStiffness<Stencil3D>(out.stiffness, out.vertxpos,
                                    out.elem.data() + (dim + 1) * t, t,
                                    out.young, out.poisson);
      }
    }

    // bending stiffness (2D only)
    if (dim == 2 && (out.elastic2d == 1 || out.elastic2d == 3) &&
        !out.interpolated) {
      out.bending.assign(out.nedge * 17, 0);
      for (int e = 0; e < out.nedge; e++) {
        ComputeBending<StencilFlap>(out.bending.data() + 17 * e,
                                    out.vertxpos.data(), out.flaps[e].vertices,
                                    out.young / (2 * (1 + out.poisson)),
                                    out.thickness);
      }
    }
  }

  // shells + element-vertex pairs + elemlayer (user_mesh.cc:4942)
  FlexCreateShellPair(out);

  // recompute cell_empty from geometry if not supplied by the flexcomp
  // (volume mode only; user_mesh.cc:4944-4949)
  if (out.interpolated && !out.elastic2d && out.cell_empty.empty()) {
    int cx = out.cellcount[0], cy = out.cellcount[1], cz = out.cellcount[2];
    if (cx * cy * cz > 1)
      FlexComputeCellEmpty(out.vertxpos.data(), out.elem.data(), out.nvert,
                           out.nelem, dim, out.cellcount, out.cell_empty);
  }

  // interpolated finite-element stiffness (user_mesh.cc:4954-5095). young=0 with
  // strain constraints still computes (eigendecomposed geometry modes).
  if (out.interpolated && (out.young > 0 || out.has_strain_eq)) {
    double K_young = out.has_strain_eq ? 1e1 : out.young;
    double K_poisson = out.has_strain_eq ? 0.3 : out.poisson;
    int cx = out.cellcount[0], cy = out.cellcount[1], cz = out.cellcount[2];
    int ny_global = cy * out.order + 1;
    int nz_global = cz * out.order + 1;
    bool shell_mode = out.elastic2d != 0;
    int npe, nelem_fe;
    if (shell_mode) {
      npe = static_cast<int>(pow(out.order + 1, 2));
      nelem_fe = 2 * (cy * cz + cx * cz + cx * cy);
    } else {
      npe = static_cast<int>(pow(out.order + 1, 3));
      nelem_fe = cx * cy * cz;
    }
    int ndof_elem = 3 * npe;
    out.stiffness.assign(static_cast<std::size_t>(nelem_fe) * ndof_elem *
                             ndof_elem,
                         0);

    int face_sizes[6] = {cy*cz, cy*cz, cx*cz, cx*cz, cx*cy, cx*cy};
    int face_normal[6] = {0, 0, 1, 1, 2, 2};
    int face_count1[6] = {cz, cz, cx, cx, cy, cy};
    int face_fixed[6] = {0, cx*out.order, 0, cy*out.order, 0, cz*out.order};

    for (int fe = 0; fe < nelem_fe; fe++) {
      std::vector<double> elem_pos(3 * npe);
      int normal_axis = -1;

      if (shell_mode) {
        int face_id = 0, within_face = fe, cumul = 0;
        for (int f = 0; f < 6; f++) {
          if (fe < cumul + face_sizes[f]) { face_id = f; within_face = fe - cumul; break; }
          cumul += face_sizes[f];
        }
        normal_axis = face_normal[face_id];
        int na0 = (normal_axis + 1) % 3;
        int na1 = (normal_axis + 2) % 3;
        int c1 = face_count1[face_id];
        int g_fixed = face_fixed[face_id];
        int q0 = within_face / c1;
        int q1 = within_face % c1;
        int local = 0;
        for (int l0 = 0; l0 <= out.order; l0++) {
          for (int l1 = 0; l1 <= out.order; l1++) {
            int g[3];
            g[normal_axis] = g_fixed;
            g[na0] = q0 * out.order + l0;
            g[na1] = q1 * out.order + l1;
            int global = g[0] * ny_global * nz_global + g[1] * nz_global + g[2];
            lift::mjuu_copyvec(elem_pos.data() + 3*local,
                               nodexpos_local.data() + 3*global, 3);
            local++;
          }
        }
      } else {
        int ci = fe / (cy * cz);
        int cj = (fe / cz) % cy;
        int ck = fe % cz;
        if (!out.cell_empty.empty() && out.cell_empty[fe]) continue;
        int local = 0;
        for (int li = 0; li <= out.order; li++) {
          for (int lj = 0; lj <= out.order; lj++) {
            for (int lk = 0; lk <= out.order; lk++) {
              int gi = ci * out.order + li;
              int gj = cj * out.order + lj;
              int gk = ck * out.order + lk;
              int global = gi * ny_global * nz_global + gj * nz_global + gk;
              lift::mjuu_copyvec(elem_pos.data() + 3*local,
                                 nodexpos_local.data() + 3*global, 3);
              local++;
            }
          }
        }
      }

      std::vector<double> K_elem(static_cast<std::size_t>(ndof_elem) * ndof_elem,
                                 0);
      if (shell_mode) {
        ComputeLinearStiffness2D(K_elem, elem_pos.data(), K_young, K_poisson,
                                 out.order, out.thickness, normal_axis);
      } else {
        ComputeLinearStiffness(K_elem, elem_pos.data(), K_young, K_poisson,
                               out.order);
      }
      double* dst = out.stiffness.data() +
                    static_cast<std::size_t>(fe) * ndof_elem * ndof_elem;

      if (out.has_strain_eq) {
        std::fill(dst, dst + static_cast<std::size_t>(ndof_elem) * ndof_elem, 0.0);
        if (shell_mode) {
          int neig = EigendecomposeStiffness(K_elem.data(), dst, ndof_elem);
          double warp_stiffness = ComputeWarpStiffness(
              elem_pos.data(), npe, normal_axis, K_young, K_poisson,
              out.thickness);
          if (warp_stiffness > 0) {
            double* warp_out = dst + 1 + neig * ndof_elem;
            ComputeWarpMode(warp_out, elem_pos.data(), npe, out.order,
                            normal_axis);
            double scale = std::sqrt(warp_stiffness);
            for (int j = 0; j < ndof_elem; j++) warp_out[j] *= scale;
            dst[0] = static_cast<double>(neig + 1);
          }
        } else {
          EigendecomposeStiffness(K_elem.data(), dst, ndof_elem);
        }
      } else {
        std::copy(K_elem.begin(), K_elem.end(), dst);
      }
    }
  }

  // interpolated shell bending edge data (user_mesh.cc:5097-5101)
  if (out.interpolated && (out.elastic2d == 1 || out.elastic2d == 3) &&
      out.thickness > 0 && out.young > 0) {
    ComputeInterpBending(out.bending, nodexpos_local, out.order, out.cellcount,
                         out.young, out.poisson, out.thickness);
  }

  // BVH over active element AABBs (user_mesh.cc:5410 CreateBVH). vertxpos is
  // global, so the BVH frame is identity; dim==3 inactive layers are skipped.
  BVH tree;
  double zero[3] = {0, 0, 0}, ident[4] = {1, 0, 0, 0};
  tree.Set(zero, ident);
  std::vector<double> elemaabb(6 * out.nelem);
  for (int e = 0; e < out.nelem; e++) {
    const int* edata = out.elem.data() + e * (dim + 1);
    if (dim == 3 && out.elemlayer[e] >= out.activelayers) continue;
    double xmin[3], xmax[3];
    lift::mjuu_copyvec(xmin, out.vertxpos.data() + 3 * edata[0], 3);
    lift::mjuu_copyvec(xmax, out.vertxpos.data() + 3 * edata[0], 3);
    for (int i = 1; i <= dim; i++)
      for (int j = 0; j < 3; j++) {
        xmin[j] = std::min(xmin[j], out.vertxpos[3 * edata[i] + j]);
        xmax[j] = std::max(xmax[j], out.vertxpos[3 * edata[i] + j]);
      }
    double* ab = elemaabb.data() + 6 * e;
    ab[0] = 0.5 * (xmax[0] + xmin[0]);
    ab[1] = 0.5 * (xmax[1] + xmin[1]);
    ab[2] = 0.5 * (xmax[2] + xmin[2]);
    ab[3] = 0.5 * (xmax[0] - xmin[0]) + out.radius;
    ab[4] = 0.5 * (xmax[1] - xmin[1]) + out.radius;
    ab[5] = 0.5 * (xmax[2] - xmin[2]) + out.radius;
    tree.Add(BVHLeaf{e, 1, 1, ab, nullptr, ab});
  }
  tree.Create();
  out.bvh = tree.bvh();
  out.bvh_child = tree.child();
  out.bvh_level = tree.level();
  out.bvh_nodeid = tree.nodeid();

  // bounding-box vertex coordinates (user_mesh.cc:5155-5199)
  out.vert0.assign(3 * out.nvert, 0);
  if (out.interpolated && out.nnode > 0) {
    // interpolated: vert0 in the unrotated local frame (parametric coords)
    std::vector<double> vertxpos_local(3 * out.nvert);
    for (int j = 0; j < out.nvert; j++)
      lift::mjuu_mulvecmat(vertxpos_local.data() + 3 * j,
                           out.vertxpos.data() + 3 * j, R0);
    double lo[3] = {1e30, 1e30, 1e30};
    double hi[3] = {-1e30, -1e30, -1e30};
    for (int i = 0; i < out.nnode; i++)
      for (int k = 0; k < 3; k++) {
        lo[k] = std::min(lo[k], nodexpos_local[3 * i + k]);
        hi[k] = std::max(hi[k], nodexpos_local[3 * i + k]);
      }
    for (int k = 0; k < 3; k++) out.size[k] = (hi[k] - lo[k]) / 2;
    for (int j = 0; j < out.nvert; j++)
      for (int k = 0; k < 3; k++) {
        double extent = hi[k] - lo[k];
        out.vert0[3 * j + k] =
            extent > mjMINVAL ? (vertxpos_local[3 * j + k] - lo[k]) / extent : 0.5;
      }
  } else if (out.nbvh() > 0) {
    const double* bvh = out.bvh.data();
    out.size[0] = bvh[3] - out.radius;
    out.size[1] = bvh[4] - out.radius;
    out.size[2] = bvh[5] - out.radius;
    for (int j = 0; j < out.nvert; j++)
      for (int k = 0; k < 3; k++)
        out.vert0[3 * j + k] =
            out.size[k] > mjMINVAL
                ? (out.vertxpos[3 * j + k] - bvh[k]) / (2 * out.size[k]) + 0.5
                : 0.5;
  }

  // node0 in the unrotated (body-local) frame (user_mesh.cc:5201-5205)
  out.node0.assign(3 * out.nnode, 0);
  for (int i = 0; i < out.nnode; i++)
    lift::mjuu_copyvec(out.node0.data() + 3 * i, nodexpos_local.data() + 3 * i, 3);
  return true;
}

// SetSizes flex edge-Jacobian nnz (user_model.cc:2198-2241): nJfe walks the body
// dof trees spanned by each edge's two vertex bodies; nJfv the trees spanned by a
// vertex and its edge-neighbours. Skipped for rigid/interpolated flexes.
void FlexJacobianCounts(const CFlex& f, const std::vector<CBody>& cbs,
                        int& nJfe, int& nJfv) {
  if (f.interpolated || f.rigid) return;
  for (const auto& e : f.edge) {
    int b1 = f.vertbodyid[e.first], b2 = f.vertbodyid[e.second];
    std::unordered_set<int> bodies_in_jac;
    while (b1 >= 0 || b2 >= 0) {
      if (b1 >= 0) { bodies_in_jac.insert(b1); b1 = b1 == 0 ? -1 : cbs[b1].parentid; }
      if (b2 >= 0) { bodies_in_jac.insert(b2); b2 = b2 == 0 ? -1 : cbs[b2].parentid; }
    }
    for (int b : bodies_in_jac) nJfe += cbs[b].dofnum;
  }
  std::vector<std::vector<int>> adj(f.nvert);
  for (const auto& e : f.edge) {
    adj[e.first].push_back(e.second);
    adj[e.second].push_back(e.first);
  }
  for (int j = 0; j < f.nvert; j++) {
    std::unordered_set<int> vert_bodies;
    vert_bodies.insert(f.vertbodyid[j]);
    for (int nb : adj[j]) vert_bodies.insert(f.vertbodyid[nb]);
    std::unordered_set<int> bodies_in_jac;
    for (int bid : vert_bodies) {
      int b = bid;
      while (b >= 0) { bodies_in_jac.insert(b); b = b == 0 ? -1 : cbs[b].parentid; }
    }
    for (int b : bodies_in_jac) nJfv += cbs[b].dofnum;
  }
}

// CopyObjects flex fill (user_model.cc:3432-3654). `bvh_start` is the first
// dynamic BVH slot (== nbvhstatic); flex nodes append after body+mesh nodes.
void FillFlexes(mjModel* m, const std::vector<CFlex>& flexes,
                const std::vector<CBody>& cbs, int bvh_start) {
  int vert_adr = 0, node_adr = 0, edge_adr = 0, elem_adr = 0, elemdata_adr = 0,
      elemedge_adr = 0, shelldata_adr = 0, evpair_adr = 0, texcoord_adr = 0;
  int stiffness_adr = 0, bending_adr = 0;
  int bvh_adr = bvh_start;
  for (int i = 0; i < static_cast<int>(flexes.size()); ++i) {
    const CFlex& f = flexes[i];
    const int dim = f.dim;
    m->flex_contype[i] = f.contype;
    m->flex_conaffinity[i] = f.conaffinity;
    m->flex_condim[i] = f.condim;
    m->flex_matid[i] = f.matid;
    m->flex_group[i] = f.group;
    m->flex_priority[i] = f.priority;
    m->flex_solmix[i] = f.solmix;
    lift::mjuu_copyvec(m->flex_solref + mjNREF * i, f.solref, mjNREF);
    lift::mjuu_copyvec(m->flex_solimp + mjNIMP * i, f.solimp, mjNIMP);
    m->flex_radius[i] = f.radius;
    lift::mjuu_copyvec(m->flex_size + 3 * i, f.size, 3);
    lift::mjuu_copyvec(m->flex_friction + 3 * i, f.friction, 3);
    m->flex_margin[i] = f.margin;
    m->flex_gap[i] = f.gap;
    for (int k = 0; k < 4; ++k) m->flex_rgba[4 * i + k] = f.rgba[k];

    // elasticity (young>0; empty vectors when young==0). user_model.cc:3465.
    if (f.stiffness.empty()) {
      m->flex_stiffnessadr[i] = -1;
    } else {
      m->flex_stiffnessadr[i] = stiffness_adr;
      lift::mjuu_copyvec(m->flex_stiffness + stiffness_adr, f.stiffness.data(),
                         static_cast<int>(f.stiffness.size()));
    }
    if (f.bending.empty()) {
      m->flex_bendingadr[i] = -1;
    } else {
      m->flex_bendingadr[i] = bending_adr;
      lift::mjuu_copyvec(m->flex_bending + bending_adr, f.bending.data(),
                         static_cast<int>(f.bending.size()));
    }
    m->flex_damping[i] = f.damping;

    m->flex_dim[i] = dim;
    m->flex_vertadr[i] = vert_adr;
    m->flex_vertnum[i] = f.nvert;
    m->flex_nodeadr[i] = node_adr;
    m->flex_nodenum[i] = f.nnode;
    m->flex_edgeadr[i] = edge_adr;
    m->flex_edgenum[i] = f.nedge;
    m->flex_elemadr[i] = elem_adr;
    m->flex_elemdataadr[i] = elemdata_adr;
    m->flex_elemedgeadr[i] = elemedge_adr;
    m->flex_shellnum[i] = static_cast<int>(f.shell.size()) / dim;
    m->flex_shelldataadr[i] = m->flex_shellnum[i] ? shelldata_adr : -1;
    if (f.evpair.empty()) {
      m->flex_evpairadr[i] = -1;
      m->flex_evpairnum[i] = 0;
    } else {
      m->flex_evpairadr[i] = evpair_adr;
      m->flex_evpairnum[i] = static_cast<int>(f.evpair.size()) / 2;
      std::memcpy(m->flex_evpair + 2 * evpair_adr, f.evpair.data(),
                  f.evpair.size() * sizeof(int));
    }
    if (f.texcoord.empty()) {
      m->flex_texcoordadr[i] = -1;
      std::memcpy(m->flex_elemtexcoord + elemdata_adr, f.elem.data(),
                  f.elem.size() * sizeof(int));
    } else {
      m->flex_texcoordadr[i] = texcoord_adr;
      std::memcpy(m->flex_texcoord + 2 * texcoord_adr, f.texcoord.data(),
                  f.texcoord.size() * sizeof(float));
      std::memcpy(m->flex_elemtexcoord + elemdata_adr, f.elemtexcoord.data(),
                  f.elemtexcoord.size() * sizeof(int));
    }
    m->flex_elemnum[i] = f.nelem;
    std::memcpy(m->flex_elem + elemdata_adr, f.elem.data(),
                f.elem.size() * sizeof(int));
    std::memcpy(m->flex_elemedge + elemedge_adr, f.edgeidx.data(),
                f.edgeidx.size() * sizeof(int));
    std::memcpy(m->flex_elemlayer + elem_adr, f.elemlayer.data(),
                f.nelem * sizeof(int));
    if (m->flex_shellnum[i])
      std::memcpy(m->flex_shell + shelldata_adr, f.shell.data(),
                  f.shell.size() * sizeof(int));
    m->flex_edgestiffness[i] = f.edgestiffness;
    m->flex_edgedamping[i] = f.edgedamping;
    m->flex_rigid[i] = f.rigid;
    m->flex_centered[i] = f.centered;
    m->flex_internal[i] = f.internal;
    m->flex_flatskin[i] = f.flatskin;
    m->flex_selfcollide[i] = f.selfcollide;
    m->flex_activelayers[i] = f.activelayers;
    m->flex_passive[i] = f.passive;
    m->flex_bvhnum[i] = f.nbvh();
    m->flex_bvhadr[i] = f.nbvh() ? bvh_adr : -1;
    m->flex_edgeequality[i] = f.edgeequality;

    // dynamic BVH nodes (aabb computed at runtime; only child/depth/nodeid here)
    for (int k = 0; k < f.nbvh(); ++k) {
      m->bvh_child[2 * (bvh_adr + k)] = f.bvh_child[2 * k];
      m->bvh_child[2 * (bvh_adr + k) + 1] = f.bvh_child[2 * k + 1];
      m->bvh_depth[bvh_adr + k] = f.bvh_level[k];
      m->bvh_nodeid[bvh_adr + k] = f.bvh_nodeid[k];
    }

    if (f.centered && !f.interpolated)
      lift::mjuu_zerovec(m->flex_vert + 3 * vert_adr, 3 * f.nvert);
    else
      lift::mjuu_copyvec(m->flex_vert + 3 * vert_adr, f.vert.data(), 3 * f.nvert);

    // node local offsets (interpolated only). user_model.cc:3576-3581.
    if (f.centered && f.interpolated)
      lift::mjuu_zerovec(m->flex_node + 3 * node_adr, 3 * f.nnode);
    else if (f.interpolated)
      lift::mjuu_copyvec(m->flex_node + 3 * node_adr, f.node.data(), 3 * f.nnode);

    lift::mjuu_copyvec(m->flex_vert0 + 3 * vert_adr, f.vert0.data(), 3 * f.nvert);
    lift::mjuu_copyvec(m->flex_node0 + 3 * node_adr, f.node0.data(), 3 * f.nnode);

    if (f.rigid)
      for (int k = 0; k < f.nvert; k++)
        m->flex_vertbodyid[vert_adr + k] = f.vertbodyid[0];
    else
      std::memcpy(m->flex_vertbodyid + vert_adr, f.vertbodyid.data(),
                  f.nvert * sizeof(int));

    // nodebodyid (interpolated). user_model.cc:3599-3606.
    if (f.rigid)
      for (int k = 0; k < f.nnode; k++)
        m->flex_nodebodyid[node_adr + k] = f.nodebodyid.empty() ? -1 : f.nodebodyid[0];
    else if (!f.nodebodyid.empty())
      std::memcpy(m->flex_nodebodyid + node_adr, f.nodebodyid.data(),
                  f.nnode * sizeof(int));

    m->flex_interp[i] = f.elastic2d ? -f.order : f.order;
    m->flex_cellnum[3 * i + 0] = f.cellcount[0];
    m->flex_cellnum[3 * i + 1] = f.cellcount[1];
    m->flex_cellnum[3 * i + 2] = f.cellcount[2];

    // edgeflap carries the two off-edge vertices of the adjacent triangles for
    // the 2D bending kernel; only populated for elastic2d bend (user_model.cc
    // :3618-3626), otherwise -1/-1.
    const bool has_edgeflap = dim == 2 && (f.elastic2d == 1 || f.elastic2d == 3);
    for (int k = 0; k < f.nedge; k++) {
      m->flex_edge[2 * (edge_adr + k)] = f.edge[k].first;
      m->flex_edge[2 * (edge_adr + k) + 1] = f.edge[k].second;
      if (has_edgeflap) {
        m->flex_edgeflap[2 * (edge_adr + k) + 0] = f.flaps[k].vertices[2];
        m->flex_edgeflap[2 * (edge_adr + k) + 1] = f.flaps[k].vertices[3];
      } else {
        m->flex_edgeflap[2 * (edge_adr + k) + 0] = -1;
        m->flex_edgeflap[2 * (edge_adr + k) + 1] = -1;
      }
      if (f.rigid) {
        m->flexedge_rigid[edge_adr + k] = 1;
      } else if (!f.interpolated) {
        int b1 = f.vertbodyid[f.edge[k].first];
        int b2 = f.vertbodyid[f.edge[k].second];
        m->flexedge_rigid[edge_adr + k] = (cbs[b1].weldid == cbs[b2].weldid);
      } else {
        m->flexedge_rigid[edge_adr + k] = 0;
      }
    }

    vert_adr += f.nvert;
    node_adr += f.nnode;
    edge_adr += f.nedge;
    elem_adr += f.nelem;
    elemdata_adr += (dim + 1) * f.nelem;
    elemedge_adr += kFlexNumEdges[dim - 1] * f.nelem;
    shelldata_adr += static_cast<int>(f.shell.size());
    evpair_adr += static_cast<int>(f.evpair.size()) / 2;
    texcoord_adr += static_cast<int>(f.texcoord.size()) / 2;
    stiffness_adr += static_cast<int>(f.stiffness.size());
    bending_adr += static_cast<int>(f.bending.size());
    bvh_adr += f.nbvh();
  }
}

}  // namespace

// Parse a <size memory> attribute string exactly as mjXReader::Size does
// (xml_native_reader.cc:1364-1454): an unsigned integer with an optional
// {K,M,G,T,P,E} power-of-two suffix, or the literal "-1"/empty for unset.
// Returns the byte count; -1 for unset; sets *valid=false for a malformed
// string (leg B throws on the same input, so such a model routes to fallback).
long long ParseSizeMemoryBytes(const std::string& raw, bool* valid) {
  *valid = true;
  // Trim: read one whitespace-delimited token, reject any trailing token.
  std::string trimmed;
  {
    std::istringstream strm(raw);
    strm >> trimmed;
    std::string trailing;
    strm >> trailing;
    if (!trailing.empty() || !strm.eof()) { *valid = false; return -1; }
  }
  if (trimmed.empty() || trimmed == "-1") return -1;  // unset

  std::istringstream strm(trimmed);
  if (strm.peek() == '-') { *valid = false; return -1; }
  std::size_t base_size = 0;
  strm >> base_size;
  if (strm.fail()) { *valid = false; return -1; }

  int multiplier_bit = 0;
  if (!strm.eof()) {
    char suffix = static_cast<char>(strm.get());
    switch (suffix) {
      case 'K': case 'k': multiplier_bit = 10; break;
      case 'M': case 'm': multiplier_bit = 20; break;
      case 'G': case 'g': multiplier_bit = 30; break;
      case 'T': case 't': multiplier_bit = 40; break;
      case 'P': case 'p': multiplier_bit = 50; break;
      case 'E': case 'e': multiplier_bit = 60; break;
      default: break;
    }
    strm.get();
    if (!multiplier_bit || !strm.eof()) { *valid = false; return -1; }
  }
  if (multiplier_bit + 1 > std::numeric_limits<std::size_t>::digits) {
    *valid = false; return -1;
  }
  const std::size_t max_base_size =
      (std::numeric_limits<std::size_t>::max() << multiplier_bit) >> multiplier_bit;
  if (base_size > max_base_size) { *valid = false; return -1; }
  const std::size_t total_size = base_size << multiplier_bit;
  if (total_size / sizeof(mjtNum) > std::numeric_limits<std::size_t>::max()) {
    *valid = false; return -1;
  }
  return static_cast<long long>(total_size);
}

// --------------------------------------------------------------------------- //
// Plugins (extension/instance resolution + mjModel plugin arrays). MuJoCo's     //
// plugin registry is process-global; the harness loads the first-party plugin   //
// DLLs before compile, so mjp_getPlugin / mjp_getPluginAtSlot resolve every      //
// corpus plugin. A plugin instance is a slot + config; the config is packed into //
// flattened_attributes in the plugin's declared attribute order (mjCPlugin::     //
// Compile, user_objects.cc:8489). Element-level plugin refs (actuator/sensor)    //
// map to an instance id -> {body,geom,actuator,sensor}_plugin. nstate and (for   //
// sensor plugins) nsensordata are queried from the plugin callbacks after the    //
// model is allocated (CopyPlugins, user_model.cc:3122).                          //
// --------------------------------------------------------------------------- //
struct CPlugin {
  int slot = -1;
  std::string name;              // instance name ("" for an implicit instance)
  std::vector<char> attr;        // flattened_attributes
  int needstage = mjSTAGE_POS;   // from the plugin decl (sensor plugins)
  int capabilityflags = 0;
};

struct PluginCollection {
  std::vector<CPlugin> plugins;
  std::unordered_map<const void*, int> elem_of;  // element src ptr -> plugin id
};

// Pack a plugin instance's config into flattened_attributes: the plugin's
// declared attributes in order, each value NUL-terminated, an empty NUL for an
// absent one (mjCPlugin::Compile). nattribute==0 emits a single NUL.
std::vector<char> PackPluginAttr(
    const mjpPlugin* p, const std::map<std::string, std::string>& cfg) {
  std::vector<char> out;
  for (int i = 0; i < p->nattribute; ++i) {
    auto it = cfg.find(p->attributes[i]);
    if (it == cfg.end()) {
      out.push_back('\0');
    } else {
      out.insert(out.end(), it->second.begin(), it->second.end());
      out.push_back('\0');
    }
  }
  if (p->nattribute == 0) out.push_back('\0');
  return out;
}

// Collect plugin instances (explicit <extension><plugin><instance> first, then
// implicit inline-config instances in element order) and map every plugin-
// bearing element to its instance id. Returns false + a diagnostic on an
// unresolved plugin name.
bool CollectPlugins(const Model& m, PluginCollection& pc,
                    std::vector<ps::mjcf::Diagnostic>& diags) {
  auto slot_of = [&](const std::string& name, const ps::SourceLoc&) -> int {
    int slot = -1;
    if (!mjp_getPlugin(name.c_str(), &slot)) return -1;
    return slot;
  };

  // Referenced explicit-instance names drive RemovePlugins: an explicit instance
  // never referenced by name is dropped (user_model.cc:566).
  std::set<std::string> referenced;
  for (const auto& ac : m.actuators)
    if (ac) for (const auto& any : ac->actuators)
      if (any.kind() == ActuatorAny::Kind::ActuatorPlugin) {
        const auto& e = *std::get<std::unique_ptr<ActuatorPlugin>>(any.node);
        if (e.instance && !e.instance->name.empty()) referenced.insert(e.instance->name);
      }
  for (const auto& sn : m.sensors)
    if (sn) for (const auto& any : sn->sensors)
      if (any.kind() == SensorAny::Kind::SensorPlugin) {
        const auto& e = *std::get<std::unique_ptr<SensorPlugin>>(any.node);
        if (e.instance && !e.instance->name.empty()) referenced.insert(e.instance->name);
      }

  // Geom plugins (SDF) live in the body tree; each references an explicit
  // instance (kept by RemovePlugins) and fills geom_plugin. Collect them and
  // their instance names.
  std::vector<const Geom*> plugin_geoms;
  {
    std::function<void(const std::vector<BodyChildAny>&)> walk =
        [&](const std::vector<BodyChildAny>& sub) {
          for (const auto& c : sub) {
            switch (c.kind()) {
              case BodyChildAny::Kind::Geom: {
                const auto& g = std::get<std::unique_ptr<Geom>>(c.node);
                if (g && !g->plugin.empty() && g->plugin.front()) {
                  plugin_geoms.push_back(g.get());
                  const auto& pr = *g->plugin.front();
                  if (pr.instance && !pr.instance->name.empty())
                    referenced.insert(pr.instance->name);
                }
                break;
              }
              case BodyChildAny::Kind::Body:
                walk(std::get<std::unique_ptr<Body>>(c.node)->subtree); break;
              case BodyChildAny::Kind::Frame:
                walk(std::get<std::unique_ptr<Frame>>(c.node)->subtree); break;
              case BodyChildAny::Kind::Replicate:
                walk(std::get<std::unique_ptr<Replicate>>(c.node)->subtree); break;
              default: break;
            }
          }
        };
    for (const auto& wb : m.worldbody)
      if (wb) walk(wb->subtree);
  }
  // Mesh (SDF) plugins reference explicit instances too (kept by RemovePlugins,
  // even if no geom also references them).
  for (const auto& a : m.assets)
    if (a) for (const auto& mesh : a->meshs)
      if (mesh && !mesh->plugin.empty() && mesh->plugin.front() &&
          mesh->plugin.front()->instance &&
          !mesh->plugin.front()->instance->name.empty())
        referenced.insert(mesh->plugin.front()->instance->name);

  // Explicit instances, in extension document order.
  std::unordered_map<std::string, int> id_of_name;
  for (const auto& ext : m.extensions) {
    if (!ext) continue;
    for (const auto& pd : ext->pluginDefs) {
      if (!pd || !pd->plugin) continue;
      const int slot = slot_of(*pd->plugin, pd->loc);
      if (slot < 0) {
        diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "plugin",
                         "native: unresolved plugin '" + *pd->plugin + "'", {}});
        return false;
      }
      const mjpPlugin* p = mjp_getPluginAtSlot(slot);
      for (const auto& inst : pd->pluginInstances) {
        if (!inst) continue;
        const std::string name = inst->name ? *inst->name : std::string();
        if (!name.empty() && !referenced.count(name)) continue;  // RemovePlugins
        CPlugin cp;
        cp.slot = slot;
        cp.name = name;
        cp.attr = PackPluginAttr(p, ConfigMap(inst->config));
        cp.needstage = p->needstage;
        cp.capabilityflags = p->capabilityflags;
        const int id = static_cast<int>(pc.plugins.size());
        pc.plugins.push_back(std::move(cp));
        if (!name.empty()) id_of_name[name] = id;
      }
    }
  }

  // Implicit instances + element->instance mapping. An element referencing an
  // explicit instance by name maps to it; one with an inline plugin name + config
  // creates a fresh implicit instance (empty name). Element order: actuators then
  // sensors (bodies/geoms/meshes plugins are gated before reaching here).
  auto bind = [&](const void* src, const ps::opt<std::string>& plugin_name,
                  const ps::opt<ps::Ref<PluginInstance>>& instance,
                  const std::vector<std::unique_ptr<Config>>& config,
                  const ps::SourceLoc& loc) -> bool {
    if (instance && !instance->name.empty()) {
      auto it = id_of_name.find(instance->name);
      if (it == id_of_name.end()) {
        diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "plugin",
                         "native: unresolved plugin instance '" + instance->name + "'", {}});
        return false;
      }
      pc.elem_of[src] = it->second;
      return true;
    }
    const std::string pname = plugin_name ? *plugin_name : std::string();
    const int slot = slot_of(pname, loc);
    if (slot < 0) {
      diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "plugin",
                       "native: unresolved plugin '" + pname + "'", {}});
      return false;
    }
    const mjpPlugin* p = mjp_getPluginAtSlot(slot);
    CPlugin cp;
    cp.slot = slot;
    cp.attr = PackPluginAttr(p, ConfigMap(config));
    cp.needstage = p->needstage;
    cp.capabilityflags = p->capabilityflags;
    const int id = static_cast<int>(pc.plugins.size());
    pc.plugins.push_back(std::move(cp));
    pc.elem_of[src] = id;
    return true;
  };

  // Geom plugins first (worldbody precedes the actuator/sensor sections in
  // document order, so any implicit instances they create are ordered before).
  for (const Geom* g : plugin_geoms) {
    const auto& pr = *g->plugin.front();
    if (!bind(g, pr.plugin, pr.instance, pr.config, g->loc)) return false;
  }
  for (const auto& ac : m.actuators)
    if (ac) for (const auto& any : ac->actuators)
      if (any.kind() == ActuatorAny::Kind::ActuatorPlugin) {
        const auto& e = *std::get<std::unique_ptr<ActuatorPlugin>>(any.node);
        if (!bind(&e, e.plugin, e.instance, e.config, e.loc)) return false;
      }
  for (const auto& sn : m.sensors)
    if (sn) for (const auto& any : sn->sensors)
      if (any.kind() == SensorAny::Kind::SensorPlugin) {
        const auto& e = *std::get<std::unique_ptr<SensorPlugin>>(any.node);
        if (!bind(&e, e.plugin, e.instance, e.config, e.loc)) return false;
      }
  return true;
}

// Fill the mjModel plugin arrays + query the plugin callbacks (nstate; sensor
// dim/needstage). Runs after every element fill so the *_plugin arrays and the
// plugin_attr blob are in place before the callbacks read them.
void FillPlugins(mjModel* m, const PluginCollection& pc,
                 const std::vector<CActuator>& actuators,
                 const std::vector<CSensor>& sensors,
                 const std::vector<CGeom>& geoms) {
  const int nplugin = static_cast<int>(pc.plugins.size());

  // Slot + packed attributes.
  int adr = 0;
  for (int i = 0; i < nplugin; ++i) {
    m->plugin[i] = pc.plugins[i].slot;
    const int size = static_cast<int>(pc.plugins[i].attr.size());
    std::memcpy(m->plugin_attr + adr, pc.plugins[i].attr.data(), size);
    m->plugin_attradr[i] = adr;
    adr += size;
  }

  // Element -> instance id (actuator/sensor plugins in the achievable scope).
  auto lookup = [&](const void* src) -> int {
    auto it = pc.elem_of.find(src);
    return it == pc.elem_of.end() ? -1 : it->second;
  };
  for (int i = 0; i < static_cast<int>(actuators.size()); ++i)
    if (actuators[i].is_plugin) m->actuator_plugin[i] = lookup(actuators[i].src);
  // Geom plugins (SDF): geom_plugin[i] = instance id, -1 otherwise.
  for (int i = 0; i < static_cast<int>(geoms.size()); ++i) {
    const int pid = lookup(geoms[i].src);
    if (pid >= 0) m->geom_plugin[i] = pid;
  }
  std::vector<std::vector<int>> plugin_to_sensors(nplugin);
  for (int i = 0; i < static_cast<int>(sensors.size()); ++i)
    if (sensors[i].type == mjSENS_PLUGIN) {
      const int pid = lookup(sensors[i].src);
      m->sensor_plugin[i] = pid;
      if (pid >= 0) plugin_to_sensors[pid].push_back(i);
    }

  // nstate + sensor dim/needstage (CopyPlugins, user_model.cc:3177). The state
  // and sensordata callbacks read plugin_attr, so they run after the fill above.
  int stateadr = 0;
  for (int i = 0; i < nplugin; ++i) {
    const mjpPlugin* p = mjp_getPluginAtSlot(m->plugin[i]);
    const int nstate = p->nstate ? p->nstate(m, i) : 0;
    m->plugin_stateadr[i] = stateadr;
    m->plugin_statenum[i] = nstate;
    stateadr += nstate;
    if (p->capabilityflags & mjPLUGIN_SENSOR)
      for (int sid : plugin_to_sensors[i]) {
        const int dim = p->nsensordata ? p->nsensordata(m, i, sid) : 0;
        m->sensor_dim[sid] = dim;
        m->sensor_needstage[sid] = pc.plugins[i].needstage;
      }
  }
  m->npluginstate = stateadr;

  // Plugin sensor dims were 0 at the size census; recompute sensor_adr and the
  // total nsensordata now that the plugin dims are set.
  int sadr = 0;
  for (int i = 0; i < m->nsensor; ++i) {
    m->sensor_adr[i] = sadr;
    sadr += m->sensor_dim[i];
  }
  m->nsensordata = sadr;
}

mjModel* BuildNativeModel(const Model& m, const ps::mjcf::CompileOptions& opts,
                          std::vector<ps::mjcf::Diagnostic>& diags) {
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
      if (!HfieldCompile(*hf, cs, opts.base_dir, ch, diags)) return nullptr;
      hfields.push_back(std::move(ch));
    }
  }
  AssetBinds asset_binds;
  for (int i = 0; i < static_cast<int>(hfields.size()); ++i) {
    if (hfields[i].listname.empty()) continue;
    HfieldBind hb;
    hb.id = i;
    for (int k = 0; k < 4; ++k) hb.size[k] = hfields[i].size[k];
    asset_binds.hfield[hfields[i].listname] = hb;
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
  std::vector<CLight>& lights = collector.lights();

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

  // Flex name -> id map, in the order flexes are compiled below (authored flexes
  // across <deformable> sections, then flexcomp-synthesized flexes in collect
  // order). Built before equalities so flex-equality refs resolve by name.
  std::unordered_map<std::string, int> flexid_of;
  {
    int fid = 0;
    for (const auto& df : m.deformables) {
      if (!df) continue;
      for (const auto& fl : df->flexs)
        if (fl) { if (fl->name) flexid_of[*fl->name] = fid; ++fid; }
    }
    for (const auto& fl : collector.synth_flexes())
      if (fl) { if (fl->name) flexid_of[*fl->name] = fid; ++fid; }
  }

  // Equality constraints (S8): document order across all <equality> sections,
  // then flexcomp-synthesized flex (edge) equalities in collect order.
  std::vector<CEquality> equalities;
  for (const auto& eq : m.equalitys) {
    if (!eq) continue;
    for (const auto& any : eq->equalities)
      equalities.push_back(EqualityCompile(default_idx, any, bodyid_of,
                                           siteid_of, jointid_of, tendonid_of));
  }
  for (const auto& se : collector.synth_equalities())
    if (se.eq)
      equalities.push_back(FlexEqualityCompile(
          default_idx, *se.eq, flexid_of, se.type,
          se.has_data ? se.data : nullptr));

  // Actuators (S8): document order across all <actuator> sections. Transmission
  // targets resolve against joint/tendon/body id maps built above.
  RangeLookup rlook{&joints, &tendons, &jointid_of, &tendonid_of};
  std::vector<CActuator> actuators;
  for (const auto& ac : m.actuators) {
    if (!ac) continue;
    for (const auto& any : ac->actuators)
      actuators.push_back(ActuatorCompile(m, any, jointid_of, tendonid_of,
                                          bodyid_of, siteid_of, rlook));
  }

  // Actuator / camera name -> id maps for sensor targets.
  std::unordered_map<std::string, int> actuatorid_of, cameraid_of;
  for (int i = 0; i < static_cast<int>(actuators.size()); ++i)
    if (actuators[i].name) actuatorid_of[*actuators[i].name] = i;
  for (int i = 0; i < static_cast<int>(cameras.size()); ++i)
    if (cameras[i].src && cameras[i].src->name) cameraid_of[*cameras[i].src->name] = i;

  // Mesh name -> id / nvert maps (tactile sensor obj + dim). Mesh id is the fill
  // index; the key is the authored name (else the file stem, as elsewhere).
  std::unordered_map<std::string, int> meshid_of, meshnvert_of;
  for (int i = 0; i < static_cast<int>(meshes.size()); ++i) {
    if (!meshes[i].listname.empty()) {
      meshid_of[meshes[i].listname] = i;
      meshnvert_of[meshes[i].listname] = meshes[i].r.nvert();
    }
  }

  // Sensors (S8): document order across all <sensor> sections.
  SensorMaps smaps{&bodyid_of, &geomid_of, &siteid_of, &cameraid_of,
                   &jointid_of, &tendonid_of, &actuatorid_of,
                   &meshid_of, &meshnvert_of};
  std::vector<CSensor> sensors;
  for (const auto& sn : m.sensors) {
    if (!sn) continue;
    for (const auto& any : sn->sensors)
      sensors.push_back(SensorCompile(m, any, smaps));
  }

  // Plugins (extension/instance resolution + element->instance mapping).
  PluginCollection plugins;
  if (!CollectPlugins(m, plugins, diags)) return nullptr;

  // Keyframes (S12): flattened key list, document order.
  std::vector<const Key*> keys;
  for (const auto& kf : m.keyframes) {
    if (!kf) continue;
    for (const auto& k : kf->keys)
      if (k) keys.push_back(k.get());
  }
  // <size nkey="N"> pre-allocates N keyframe slots (ReadAttrInt, last block
  // wins); the compiler pads to max(authored, N) with fully-default keyframes
  // (AddKey: empty name, time 0, qpos->qpos0, qvel/act/ctrl 0, mocap ref pose --
  // user_model.cc:5028). The padded slots carry no name (unlike an authored
  // unnamed key, which the XML path auto-names symmetrically).
  int size_nkey = 0;
  for (const auto& sz : m.sizes)
    if (sz && sz->nkey) size_nkey = *sz->nkey;
  const int total_nkey =
      std::max(static_cast<int>(keys.size()), size_nkey);

  // Custom fields (S8): numeric / text / tuple, document order across all
  // <custom> sections. Tuple obj refs resolve against the id maps built above.
  std::vector<CNumeric> numerics;
  std::vector<CText> texts;
  std::vector<CTuple> tuples;
  {
    auto tuple_id = [&](int objtype, const std::string& nm) -> int {
      const NameIdMap* map = nullptr;
      switch (objtype) {
        case mjOBJ_BODY: case mjOBJ_XBODY: map = &bodyid_of; break;
        case mjOBJ_GEOM: map = &geomid_of; break;
        case mjOBJ_SITE: map = &siteid_of; break;
        case mjOBJ_JOINT: map = &jointid_of; break;
        case mjOBJ_CAMERA: map = &cameraid_of; break;
        case mjOBJ_TENDON: map = &tendonid_of; break;
        case mjOBJ_ACTUATOR: map = &actuatorid_of; break;
        default: return -1;
      }
      auto it = map->find(nm);
      return it == map->end() ? -1 : it->second;
    };
    for (const auto& cu : m.customs) {
      if (!cu) continue;
      for (const auto& n : cu->numerics)
        if (n) numerics.push_back(NumericCompile(*n, opts));
      for (const auto& t : cu->texts)
        if (t) texts.push_back(TextCompile(*t, opts));
      for (const auto& tp : cu->tuples) {
        if (!tp) continue;
        CTuple ct;
        ct.name = EffectiveName(*tp, opts);
        for (const auto& el : tp->tupleElements) {
          if (!el) continue;
          CTupleEntry e;
          e.objtype = el->objtype ? mju_str2Type(el->objtype->c_str()) : mjOBJ_UNKNOWN;
          e.objid = tuple_id(e.objtype, el->objname ? *el->objname : std::string());
          e.prm = el->prm ? *el->prm : 0.0;
          ct.entries.push_back(e);
        }
        tuples.push_back(std::move(ct));
      }
    }
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
      if (!TextureCompile(m, *tx, cs, opts.base_dir, ct, diags)) return nullptr;
      textures.push_back(std::move(ct));
    }
  }
  for (int i = 0; i < static_cast<int>(textures.size()); ++i)
    if (!textures[i].listname.empty()) texid_of[textures[i].listname] = i;

  std::vector<CMaterial> materials;
  for (const auto& as : m.assets) {
    if (!as) continue;
    for (const auto& mat : as->materials)
      if (mat) materials.push_back(MaterialCompile(m, *mat, texid_of));
  }
  for (int i = 0; i < static_cast<int>(materials.size()); ++i)
    if (materials[i].name) matid_of[*materials[i].name] = i;

  // Skins (document order across <asset> then <deformable>; compiled after the
  // body walk + materials for bone body-id and material-id resolution).
  std::vector<CSkin> skins;
  auto compile_skins = [&](const auto& list) -> bool {
    for (const auto& sk : list) {
      if (!sk) continue;
      CSkin cs2;
      if (!SkinCompile(m, *sk, cs, opts.base_dir, bodyid_of, matid_of, cs2, diags))
        return false;
      skins.push_back(std::move(cs2));
    }
    return true;
  };
  for (const auto& as : m.assets)
    if (as && !compile_skins(as->skins)) return nullptr;
  for (const auto& df : m.deformables)
    if (df && !compile_skins(df->skins)) return nullptr;

  // Resolve material refs to ids (geoms/sites/tendons; CopyObjects/IndexAssets).
  auto resolve_mat = [&](const std::string& nm) -> int {
    if (nm.empty()) return -1;
    auto it = matid_of.find(nm);
    return it == matid_of.end() ? -1 : it->second;
  };
  for (CGeom& cg : geoms) cg.matid = resolve_mat(cg.material_name);
  for (CSite& cs2 : sites) cs2.matid = resolve_mat(cs2.material_name);
  for (CTendon& t : tendons) t.matid = resolve_mat(t.material_name);
  // An image light's texture ref -> light_texid (mjCLight::Compile).
  for (CLight& cl : lights) {
    if (cl.texture_name.empty()) continue;
    auto it = texid_of.find(cl.texture_name);
    cl.texid = it == texid_of.end() ? -1 : it->second;
  }

  // Flexes (NC5): document order across every <deformable> section. Compiled
  // after bodies (needs xpos0/xquat0/weldid) and materials (matid).
  // Flex names carrying a strain equality: drives the eigendecomposed stiffness
  // (interpolated). Scanned from the compiled equalities (built above).
  std::unordered_set<std::string> strain_flexes;
  for (const CEquality& ce : equalities)
    if (ce.type == mjEQ_FLEXSTRAIN) strain_flexes.insert(ce.flex_name);

  std::vector<CFlex> flexes;
  for (const auto& df : m.deformables) {
    if (!df) continue;
    for (const auto& fl : df->flexs) {
      if (!fl) continue;
      CFlex cf;
      const bool hs = fl->name && strain_flexes.count(*fl->name);
      if (!FlexCompile(*fl, cbs, bodyid_of, matid_of, cf, diags, {}, hs))
        return nullptr;
      flexes.push_back(std::move(cf));
    }
  }
  // Flexcomp-synthesized flexes (NC5 Wave 2/6): appended after authored flexes,
  // in collect (document) order. Interpolated flexes carry node local offsets
  // and the empty-cell mask from the flexcomp expansion.
  {
    const auto& node_locals = collector.synth_node_local();
    const auto& cell_empties = collector.synth_cell_empty();
    const auto& synth = collector.synth_flexes();
    for (int si = 0; si < static_cast<int>(synth.size()); ++si) {
      if (!synth[si]) continue;
      CFlex cf;
      const bool hs = synth[si]->name && strain_flexes.count(*synth[si]->name);
      if (!FlexCompile(*synth[si], cbs, bodyid_of, matid_of, cf, diags,
                       node_locals[si], hs, cell_empties[si]))
        return nullptr;
      flexes.push_back(std::move(cf));
    }
  }
  // flex_edgeequality (user_model.cc:3532-3549): a flex referenced by a flex
  // equality records the constraint kind (edge=1, vert=2, strain=3). Scanned
  // over the compiled equalities by flex name, matching upstream's first-hit.
  for (CFlex& cf : flexes) {
    for (const CEquality& ce : equalities) {
      if (ce.flex_name != cf.name) continue;
      if (ce.type == mjEQ_FLEX) { cf.edgeequality = 1; break; }
      if (ce.type == mjEQ_FLEXVERT) { cf.edgeequality = 2; break; }
      if (ce.type == mjEQ_FLEXSTRAIN) { cf.edgeequality = 3; break; }
    }
  }

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

  // Bodies demoted from "simple" by an inertia-bearing tendon (armature>0): its
  // site wrap's body, or its cylinder/sphere wrap's geom body.
  std::vector<int> tendon_demote;
  for (const CTendon& t : tendons) {
    if (t.armature <= 0) continue;
    for (const CWrap& w : t.path) {
      if (w.type == mjWRAP_SITE && w.objid >= 0 &&
          w.objid < static_cast<int>(sites.size()))
        tendon_demote.push_back(sites[w.objid].bodyid);
      else if ((w.type == mjWRAP_SPHERE || w.type == mjWRAP_CYLINDER) &&
               w.objid >= 0 && w.objid < static_cast<int>(geoms.size()))
        tendon_demote.push_back(geoms[w.objid].bodyid);
    }
  }

  // Dof tree + sparse sizes (the crown-jewel pass).
  DofTree dt = ComputeDofTree(cbs, joints, nv, tendon_demote);

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
  // Flex is not auto-named (bridge AutoNameableFamily); authored name only.
  for (const CFlex& cf : flexes) nl.flex.push_back(cf.name);
  // Assets (texture/material/mesh/hfield/skin) are NOT auto-named on the XML
  // path (bridge AutoNameableFamily), so leg C uses the authored name only --
  // an unnamed asset keeps an empty name, exactly like leg B.
  for (const CMesh& cm : meshes) nl.mesh.push_back(cm.listname);
  for (const CHField& hf : hfields)
    nl.hfield.push_back(hf.listname);
  for (const CTexture& ct : textures)
    nl.tex.push_back(ct.listname);
  for (const CSkin& sk : skins)
    nl.skin.push_back(sk.listname);
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
  // numeric/text/tuple carry a mandatory authored name (never auto-named).
  for (const CNumeric& n : numerics) nl.numeric.push_back(n.name);
  for (const CText& t : texts) nl.text.push_back(t.name);
  for (const CTuple& t : tuples) nl.tuple.push_back(t.name);
  for (const Key* k : keys) nl.key.push_back(EffectiveName(*k, opts));
  for (int i = static_cast<int>(keys.size()); i < total_nkey; ++i)
    nl.key.push_back("");  // padded keyframe: empty name (AddKey)
  // Plugin instance names (implicit instances keep an empty name), CopyNames-last.
  for (const CPlugin& p : plugins.plugins) nl.plugin.push_back(p.name);

  // nbvh census (SetSizes): static BVH nodes over all bodies, then meshes. Mesh
  // face-BVH nodes are laid out after every body node (CopyObjects bvh_adr).
  int nbvh_body = 0;
  for (const CBody& cb : cbs) nbvh_body += static_cast<int>(cb.bvh_child.size() / 2);
  int nbvh_mesh = 0;
  for (const CMesh& cm : meshes) nbvh_mesh += cm.nbvh();
  int nbvhstatic = nbvh_body + nbvh_mesh;
  // Flex BVH nodes are dynamic (aabb recomputed each step); they follow every
  // static (body+mesh) node in the shared bvh_* arrays (user_model.cc:2176).
  int nbvhdynamic = 0;
  for (const CFlex& f : flexes) nbvhdynamic += f.nbvh();

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
  // Skins (SetSizes user_model.cc:2257-2264).
  sizes.nskin = static_cast<int>(skins.size());
  int nskinvert = 0, nskintexvert = 0, nskinface = 0, nskinbone = 0,
      nskinbonevert = 0;
  for (const CSkin& sk : skins) {
    nskinvert += static_cast<int>(sk.vert.size() / 3);
    nskintexvert += static_cast<int>(sk.texcoord.size() / 2);
    nskinface += static_cast<int>(sk.face.size() / 3);
    nskinbone += static_cast<int>(sk.bodyid.size());
    for (const auto& vid : sk.vertid) nskinbonevert += static_cast<int>(vid.size());
  }
  sizes.nskinvert = nskinvert;
  sizes.nskintexvert = nskintexvert;
  sizes.nskinface = nskinface;
  sizes.nskinbone = nskinbone;
  sizes.nskinbonevert = nskinbonevert;
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
  sizes.nJmom = ComputeNJmom(actuators, joints, tendons, cbs, geoms, sites,
                             dt.dof_parentid, nv);
  sizes.nuser_actuator = nuser_actuator;
  sizes.nsensor = static_cast<int>(sensors.size());
  int nsensordata = 0;
  for (const CSensor& s : sensors) nsensordata += s.dim;
  sizes.nsensordata = nsensordata;
  sizes.nuser_sensor = nuser_sensor;
  // nhistory: delay/interp history buffer, actuators then sensors (user_model.cc
  // :2288). Actuator ring = [user, cursor, times(n), values(n)] = 2 + 2n; sensor
  // ring = [user, cursor, times(n), values(n*dim)] = 2 + n + n*dim.
  int nhistory = 0;
  for (const CActuator& a : actuators)
    if (a.nsample > 0) nhistory += 2 + 2 * a.nsample;
  for (const CSensor& s : sensors)
    if (s.nsample > 0) nhistory += 2 + s.nsample + s.nsample * s.dim;
  sizes.nhistory = nhistory;
  // Plugins (SetSizes user_model.cc:2132/2317). npluginstate is queried post-alloc
  // (nstate callback) since it needs the allocated model; it sizes only mjData.
  sizes.nplugin = static_cast<int>(plugins.plugins.size());
  int npluginattr = 0;
  for (const CPlugin& p : plugins.plugins) npluginattr += static_cast<int>(p.attr.size());
  sizes.npluginattr = npluginattr;
  // Custom fields (SetSizes user_model.cc:2302-2314).
  sizes.nnumeric = static_cast<int>(numerics.size());
  int nnumericdata = 0;
  for (const CNumeric& n : numerics) nnumericdata += n.size;
  sizes.nnumericdata = nnumericdata;
  sizes.ntext = static_cast<int>(texts.size());
  int ntextdata = 0;
  for (const CText& t : texts) ntextdata += static_cast<int>(t.data.size()) + 1;
  sizes.ntextdata = ntextdata;
  sizes.ntuple = static_cast<int>(tuples.size());
  int ntupledata = 0;
  for (const CTuple& t : tuples) ntupledata += static_cast<int>(t.entries.size());
  sizes.ntupledata = ntupledata;
  sizes.nkey = total_nkey;
  sizes.nbvh = nbvhstatic + nbvhdynamic;
  sizes.nbvhstatic = nbvhstatic;
  sizes.nbvhdynamic = nbvhdynamic;
  // Flex census (SetSizes user_model.cc:2181-2242).
  sizes.nflex = static_cast<int>(flexes.size());
  int nflexnode = 0, nflexvert = 0, nflexedge = 0, nflexelem = 0,
      nflexelemdata = 0, nflexelemedge = 0, nflexshelldata = 0, nflexevpair = 0,
      nflextexcoord = 0, nflexstiffness = 0, nflexbending = 0, nJfe = 0, nJfv = 0;
  for (const CFlex& f : flexes) {
    nflexnode += f.nnode;
    nflexvert += f.nvert;
    nflexedge += f.nedge;
    nflexelem += f.nelem;
    nflexelemdata += f.nelem * (f.dim + 1);
    nflexelemedge += f.nelem * kFlexNumEdges[f.dim - 1];
    nflexshelldata += static_cast<int>(f.shell.size());
    nflexevpair += static_cast<int>(f.evpair.size()) / 2;
    nflextexcoord += f.HasTexcoord() ? static_cast<int>(f.texcoord.size()) / 2 : 0;
    nflexstiffness += static_cast<int>(f.stiffness.size());
    nflexbending += static_cast<int>(f.bending.size());
    FlexJacobianCounts(f, cbs, nJfe, nJfv);
  }
  sizes.nflexnode = nflexnode;
  sizes.nflexvert = nflexvert;
  sizes.nflexedge = nflexedge;
  sizes.nflexelem = nflexelem;
  sizes.nflexelemdata = nflexelemdata;
  sizes.nflexelemedge = nflexelemedge;
  sizes.nflexshelldata = nflexshelldata;
  sizes.nflexevpair = nflexevpair;
  sizes.nflextexcoord = nflextexcoord;
  sizes.nflexstiffness = nflexstiffness;
  sizes.nflexbending = nflexbending;
  sizes.nJfe = nJfe;
  sizes.nJfv = nJfv;
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
  for (const CHField& ch : hfields)
    if (!ch.file.empty()) npaths += static_cast<int>(ch.file.size()) + 1;
  for (const CMesh& cm : meshes)
    if (!cm.file.empty()) npaths += static_cast<int>(cm.file.size()) + 1;
  for (const CSkin& sk : skins)
    if (!sk.file.empty()) npaths += static_cast<int>(sk.file.size()) + 1;
  for (const CTexture& ct : textures)
    if (!ct.file.empty()) npaths += static_cast<int>(ct.file.size()) + 1;
  sizes.npaths = npaths == 0 ? 1 : npaths;

  // S11 Allocate.
  mjModel* out = lift::MakeModel(sizes);
  if (!out) {
    diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "alloc",
                     "native: mj_makeModel lift failed", {}});
    return nullptr;
  }

  // Deprecated constraint-size + arena fields (<size njmax/nconmax/nstack/
  // memory>). njmax/nconmax become mjModel members (user_model.cc:3334-3335,
  // CopyObjects:5539); nstack/memory are not stored, only feeding SetNarena.
  // Last-writer-wins across <size> blocks, matching the reader.
  int size_njmax = -1, size_nconmax = -1, size_nstack = -1;
  long long size_memory = -1;
  for (const auto& sz : m.sizes) {
    if (!sz) continue;
    if (sz->njmax) size_njmax = *sz->njmax;
    if (sz->nconmax) size_nconmax = *sz->nconmax;
    if (sz->nstack) size_nstack = *sz->nstack;
    if (sz->memory) {
      bool ok = true;
      size_memory = ParseSizeMemoryBytes(*sz->memory, &ok);
      // Admitted models have a parseable memory (the gate rejects the rest).
    }
  }
  out->nconmax = size_nconmax;
  out->njmax = size_njmax;

  // S11 Fill.
  FillNames(out, nl);
  FillTextures(out, textures);
  FillMaterials(out, materials);
  FillHfields(out, hfields);
  FillMeshes(out, meshes, nbvh_body);
  FillSkins(out, skins);
  FillMeshPaths(out, hfields, meshes, skins, textures);
  FillFlexes(out, flexes, cbs, nbvhstatic);
  FillTree(out, cbs, geoms, joints, dt);
  for (int i = 0; i < static_cast<int>(cbs.size()); ++i)
    out->body_mocapid[i] = mocapid[i];
  FillVisual(out, sites, cameras, lights, bodyid_of);
  FillPairs(out, pairs);
  FillExcludes(out, excludes);
  FillEqualities(out, equalities);
  FillTendons(out, tendons);
  int delay_adr = 0;  // shared history cursor: actuators then sensors (nhistory)
  FillActuators(out, actuators, delay_adr);
  FillSensors(out, sensors, delay_adr);
  FillPlugins(out, plugins, actuators, sensors, geoms);
  FillNumerics(out, numerics);
  FillTexts(out, texts);
  FillTuples(out, tuples);
  // Keyframe padding uses qpos0 / body_pos / body_quat / body_mocapid, all set
  // by FillTree above (mjCKey::Compile).
  FillKeyframes(out, keys);

  // Arena size heuristic (ledger 3.7, user_model.cc:5219-5252). Authored
  // <size memory|nstack> select the explicit-bytes / legacy-stack branches;
  // njmax/nconmax feed both the pre-arena footprint and the heuristic fallback.
  SetNarena(out, size_memory, size_nstack);

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
    diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "finalize",
                     "native: mj_makeData failed during finalize", {}});
    mj_deleteModel(out);
    return nullptr;
  }
  mj_setConst(out, d);

  // Automatic spring-damper (mjCModel::AutoSpringDamper, user_model.cc:2373,
  // between mj_setConst and LengthRange): a joint authoring springdamper=
  // "timeconst dampratio" (both >0) derives jnt_stiffness + dof_damping from the
  // effective inertia recovered from dof_invweight0 (which mj_setConst filled).
  for (int n = 0; n < out->njnt; ++n) {
    const double timeconst = joints[n].springdamper[0];
    const double dampratio = joints[n].springdamper[1];
    if (timeconst <= 0 || dampratio <= 0) continue;
    const int adr = out->jnt_dofadr[n];
    const int ndim = joints[n].nv();
    double inv = 0;
    for (int i = 0; i < ndim; ++i) inv += out->dof_invweight0[adr + i];
    const double inertia = static_cast<double>(ndim) / std::max(mjMINVAL, inv);
    const double stiffness =
        inertia / std::max(mjMINVAL, timeconst * timeconst * dampratio * dampratio);
    const double damping = 2 * inertia / std::max(mjMINVAL, timeconst);
    out->jnt_stiffness[n] = stiffness;
    for (int i = 0; i < ndim; ++i) out->dof_damping[adr + i] = damping;
  }

  // Actuator length-range pass (mjCModel::LengthRange, user_model.cc:2444, after
  // mj_setConst): mj_setLengthRange (public) forward-simulates each muscle/user
  // actuator with physics disabled to fill actuator_lengthrange. Run for every
  // model; the default mode (mjLRMODE_MUSCLE) + useexisting skip make it a no-op
  // for non-muscle / already-ranged actuators. LROpt from mj_defaultLROpt,
  // overridden by the <compiler><lengthrange> block.
  mjLROpt lropt;
  mj_defaultLROpt(&lropt);
  for (const auto& c : m.compilers) {
    if (!c) continue;
    for (const auto& lr : c->lengthRanges) {
      if (!lr) continue;
      if (lr->mode) lropt.mode = static_cast<int>(*lr->mode);
      if (lr->useexisting) lropt.useexisting = *lr->useexisting ? 1 : 0;
      if (lr->uselimit) lropt.uselimit = *lr->uselimit ? 1 : 0;
      if (lr->accel) lropt.accel = *lr->accel;
      if (lr->maxforce) lropt.maxforce = *lr->maxforce;
      if (lr->timeconst) lropt.timeconst = *lr->timeconst;
      if (lr->timestep) lropt.timestep = *lr->timestep;
      if (lr->inttotal) lropt.inttotal = *lr->inttotal;
      if (lr->interval) lropt.interval = *lr->interval;
      if (lr->tolrange) lropt.tolrange = *lr->tolrange;
    }
  }
  const mjOption saveopt = out->opt;
  out->opt.disableflags = mjDSBL_FRICTIONLOSS | mjDSBL_CONTACT | mjDSBL_SPRING |
                          mjDSBL_DAMPER | mjDSBL_GRAVITY | mjDSBL_ACTUATION;
  if (lropt.timestep > 0) out->opt.timestep = lropt.timestep;
  char lrerr[200] = {0};
  bool lr_ok = true;
  for (int i = 0; i < out->nu; ++i)
    if (!mj_setLengthRange(out, d, i, &lropt, lrerr, sizeof(lrerr))) {
      lr_ok = false;
      break;
    }
  out->opt = saveopt;
  if (!lr_ok) {
    diags.push_back({ps::mjcf::Diagnostic::Severity::Error, "finalize",
                     std::string("native: mj_setLengthRange failed: ") + lrerr,
                     {}});
    mj_deleteData(d);
    mj_deleteModel(out);
    return nullptr;
  }
  mj_deleteData(d);

  return out;
}

}  // namespace ps::mjcf::compile
