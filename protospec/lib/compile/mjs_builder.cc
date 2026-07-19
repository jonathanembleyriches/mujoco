// ProtoSpec -> mjSpec builder (Wave 2). See mjs_builder.h for the contract.
//
// The tree walk mirrors MuJoCo's own xml_native_reader.cc element-for-element so
// that mj_compile sees byte-identical mjsSpec state to what mj_parseXML would
// have produced: same childclass threading (mjs_setDefault of `childdef?:def`),
// same actuator shortcut argument derivation (read the current mjs field as the
// class default, override with the authored value, call mjs_setTo*), same
// transmission-target priority, same equality data packing, same wrap order.
// Where the generated ApplyMjs already writes an authored field faithfully it is
// reused; the per-family waivers it leaves open (targets, typed refs, data
// packing, shortcut params, engine-struct blocks) are filled here.

#include "mjs_builder.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <mujoco/mujoco.h>

#include "keywords.h"   // ToMjcf(enum) -> MJCF keyword (flexcomp type/dof strings)
#include "mjcf.h"        // io::WriteMjcf / io::ParseMjcfString (composite route-D)
#include "mjs_binding.h"
#include "mjs_convert.h"

// mjs_addSpec registers a parsed child spec on a parent for <attach>. It is an
// exported (MJAPI, C-linkage) symbol of libmujoco but lives in the engine's
// internal user_api.h, not the public <mujoco/mujoco.h>; declare it here.
extern "C" void mjs_addSpec(mjSpec* s, mjSpec* child);

namespace ps::mjcf::compile {
namespace {

// --- small opt/ref accessors --------------------------------------------- //
template <class T>
const char* RefCStr(const ps::opt<ps::Ref<T>>& r) {
  return (r.has_value() && !r->name.empty()) ? r->name.c_str() : nullptr;
}

// --- enum -> engine constant maps (elements whole-waived by the emitter) --- //
int IntegratorToMjt(Integrator v) {
  switch (v) {
    case Integrator::Euler: return mjINT_EULER;
    case Integrator::RK4: return mjINT_RK4;
    case Integrator::implicit: return mjINT_IMPLICIT;
    case Integrator::implicitfast: return mjINT_IMPLICITFAST;
  }
  return mjINT_EULER;
}
int ConeToMjt(Cone v) {
  return v == Cone::elliptic ? mjCONE_ELLIPTIC : mjCONE_PYRAMIDAL;
}
int JacToMjt(JacobianType v) {
  switch (v) {
    case JacobianType::dense: return mjJAC_DENSE;
    case JacobianType::sparse: return mjJAC_SPARSE;
    case JacobianType::auto_: return mjJAC_AUTO;
  }
  return mjJAC_AUTO;
}
int SolverToMjt(SolverType v) {
  switch (v) {
    case SolverType::PGS: return mjSOL_PGS;
    case SolverType::CG: return mjSOL_CG;
    case SolverType::Newton: return mjSOL_NEWTON;
  }
  return mjSOL_NEWTON;
}
int TexRoleToMjt(TexRole v) {
  switch (v) {
    case TexRole::rgb: return mjTEXROLE_RGB;
    case TexRole::occlusion: return mjTEXROLE_OCCLUSION;
    case TexRole::roughness: return mjTEXROLE_ROUGHNESS;
    case TexRole::metallic: return mjTEXROLE_METALLIC;
    case TexRole::normal: return mjTEXROLE_NORMAL;
    case TexRole::opacity: return mjTEXROLE_OPACITY;
    case TexRole::emissive: return mjTEXROLE_EMISSIVE;
    case TexRole::rgba: return mjTEXROLE_RGBA;
    case TexRole::orm: return mjTEXROLE_ORM;
  }
  return mjTEXROLE_RGB;
}

// Exotic-sensor data-field enums -> engine bit indices (mirror the reader's
// raydata_map / condata_map, whose enum order the schema enums match). The
// keyword set is OR'd into a single-bit-per-field mask in mjsSensor.intprm[0].
int RayDataToMjt(RayData v) {
  switch (v) {
    case RayData::dist: return mjRAYDATA_DIST;
    case RayData::dir: return mjRAYDATA_DIR;
    case RayData::origin: return mjRAYDATA_ORIGIN;
    case RayData::point: return mjRAYDATA_POINT;
    case RayData::normal: return mjRAYDATA_NORMAL;
    case RayData::depth: return mjRAYDATA_DEPTH;
  }
  return mjRAYDATA_DIST;
}
int ContactDataToMjt(ContactData v) {
  switch (v) {
    case ContactData::found: return mjCONDATA_FOUND;
    case ContactData::force: return mjCONDATA_FORCE;
    case ContactData::torque: return mjCONDATA_TORQUE;
    case ContactData::dist: return mjCONDATA_DIST;
    case ContactData::pos: return mjCONDATA_POS;
    case ContactData::normal: return mjCONDATA_NORMAL;
    case ContactData::tangent: return mjCONDATA_TANGENT;
  }
  return mjCONDATA_FOUND;
}
int ContactReduceToInt(ContactReduce v) {
  switch (v) {
    case ContactReduce::none: return 0;
    case ContactReduce::mindist: return 1;
    case ContactReduce::maxforce: return 2;
    case ContactReduce::netforce: return 3;
  }
  return 0;
}

// Flex sub-block enums -> engine ints (mirror flexself_map / elastic2d_map).
int FlexSelfToInt(FlexSelfCollide v) {
  switch (v) {
    case FlexSelfCollide::none: return mjFLEXSELF_NONE;
    case FlexSelfCollide::narrow: return mjFLEXSELF_NARROW;
    case FlexSelfCollide::bvh: return mjFLEXSELF_BVH;
    case FlexSelfCollide::sap: return mjFLEXSELF_SAP;
    case FlexSelfCollide::auto_: return mjFLEXSELF_AUTO;
  }
  return mjFLEXSELF_AUTO;
}
int Elastic2DToInt(Elastic2D v) {
  switch (v) {
    case Elastic2D::none: return 0;
    case Elastic2D::bend: return 1;
    case Elastic2D::stretch: return 2;
    case Elastic2D::both: return 3;
  }
  return 0;
}

// ProtoSpec MeshBuiltin enum order (none,sphere,hemisphere,cone,supertorus,
// supersphere,wedge,plate) does NOT match mjtMeshBuiltin (...,SUPERSPHERE,
// SUPERTORUS,...) -- the two supersphere/supertorus slots are swapped -- so this
// must be an explicit switch, never a static_cast.
mjtMeshBuiltin MeshBuiltinToMjt(MeshBuiltin v) {
  switch (v) {
    case MeshBuiltin::none: return mjMESH_BUILTIN_NONE;
    case MeshBuiltin::sphere: return mjMESH_BUILTIN_SPHERE;
    case MeshBuiltin::hemisphere: return mjMESH_BUILTIN_HEMISPHERE;
    case MeshBuiltin::cone: return mjMESH_BUILTIN_CONE;
    case MeshBuiltin::supertorus: return mjMESH_BUILTIN_SUPERTORUS;
    case MeshBuiltin::supersphere: return mjMESH_BUILTIN_SUPERSPHERE;
    case MeshBuiltin::wedge: return mjMESH_BUILTIN_WEDGE;
    case MeshBuiltin::plate: return mjMESH_BUILTIN_PLATE;
  }
  return mjMESH_BUILTIN_NONE;
}

// --- <replicate> expansion helpers (mirror the reader exactly) ------------- //
// mjuu_frameaccum: pos += R(quat)*offset; quat = quat*rot. Reimplemented from
// mju_rotVecQuat + mju_mulQuat (user_util.cc mjuu_frameaccum). ps_path_diff is
// the drift net for this reimplementation.
void FrameAccum(double pos[3], double quat[4], const double offset[3],
                const double rot[4]) {
  double r[3];
  mju_rotVecQuat(r, offset, quat);
  pos[0] += r[0]; pos[1] += r[1]; pos[2] += r[2];
  double q[4];
  mju_mulQuat(q, quat, rot);
  mju_copy4(quat, q);
}

// UpdateString mirror (xml_native_reader.cc): the per-copy suffix is the `sep`
// string followed by the index zero-padded to the digit-width of `count`.
std::string ReplicateSuffix(const std::string& sep, int count, int i) {
  int ndigits = static_cast<int>(std::to_string(count).length());
  std::string is = std::to_string(i), pad;
  while (ndigits-- > static_cast<int>(is.length())) pad += '0';
  return sep + pad + is;
}

// --- the builder --------------------------------------------------------- //
class Builder {
 public:
  Builder(mjSpec* spec, const io::AutoNames& names) : spec_(spec), names_(names) {}

  void Build(const Model& m, const CompileOptions& opts);

  // First fatal error raised during Build (macro expansion), empty if none.
  const std::string& error() const { return err_; }

 private:
  // If `serial` was assigned a reserved auto-name, stamp it (the authored name,
  // if any, was already set by ApplyMjs -- the two are mutually exclusive so
  // this never overwrites an authored name).
  void AutoName(mjsElement* el, std::uint64_t serial) {
    auto it = names_.find(serial);
    if (it != names_.end()) mjs_setName(el, it->second.c_str());
  }

  const mjsDefault* FindClass(const ps::opt<ps::Ref<Default>>& dclass,
                              const mjsDefault* fallback) {
    if (dclass.has_value() && !dclass->name.empty())
      return mjs_findDefault(spec_, dclass->name.c_str());
    return fallback;
  }

  void BuildCompiler(const Model& m, const CompileOptions& opts);
  void BuildOption(const Option& o);
  void BuildFlag(const Flag& f, mjOption& opt);
  void BuildSize(const Size& s);
  void BuildStatistic(const Statistic& s);
  void BuildVisual(const Visual& v);
  void BuildDefaults(const Model& m);
  void BuildDefaultNode(const Default& d, mjsDefault* into);
  void BuildAssets(const Model& m);
  void BuildWorldbody(const Model& m);
  void BuildSubtree(mjsBody* body, const std::vector<BodyChildAny>& subtree,
                    mjsFrame* frame, const mjsDefault* inherited);
  void BuildContact(const Model& m);
  void BuildEquality(const Model& m);
  void BuildTendons(const Model& m);
  void BuildActuators(const Model& m);
  void BuildSensors(const Model& m);
  void BuildCustom(const Model& m);
  void BuildKeyframes(const Model& m);
  void BuildDeformables(const Model& m);
  void BuildSkin(const Skin& sk);
  void BuildExtensions(const Model& m);

  // Plugin helpers (mirror xml_native_reader.cc OnePlugin / ReadPluginConfigs).
  void ApplyPluginConfigs(const std::vector<std::unique_ptr<Config>>& cfgs,
                          mjsPlugin* plugin);
  // Fills an element-embedded mjsPlugin: active=1, plugin_name (type) + name
  // (instance). Empty instance name => an inline anonymous instance is created
  // (mjs_addPlugin) and the configs applied to it; otherwise it references a
  // predefined instance by name (no new instance, configs must be empty).
  void SetPluginFields(const char* plugin_name, const char* instance_name,
                       const std::vector<std::unique_ptr<Config>>& cfgs,
                       mjsPlugin* plugin);
  void ApplyPluginRef(const PluginRef& pr, mjsPlugin* plugin);
  void BuildModelAssets(const Model& m);

  template <class A>
  void ApplyActuatorTail(const A& e, mjsActuator* out);
  template <class T>
  void ApplyActuatorShortcut(const T& e, mjsActuator* a);

  // Macro expansion (replicate/composite) via mjs_attach requires deep copy so a
  // self-contained copy is grafted; the reader holds it true across the whole
  // worldbody parse. Set/restored by BuildWorldbody.
  void ExpandReplicate(mjsBody* body, const Replicate& e, mjsFrame* frame,
                       const mjsDefault* inherited);
  void GraftComposite(mjsBody* body, const Composite& e, mjsFrame* frame);
  void BuildFlexcomp(mjsBody* body, const Flexcomp& e);

  mjSpec* spec_;
  const io::AutoNames& names_;
  const Model* model_ = nullptr;  // whole model (composite route-D needs context)
  std::string err_;               // first fatal build error (aborts BuildSpec)
  std::string base_dir_;   // model directory for resolving <model file=...>
};

// --- compiler / spec-level blocks ---------------------------------------- //
void Builder::BuildCompiler(const Model& m, const CompileOptions& opts) {
  // ProtoSpec stores angle-unit scalars verbatim in their authored unit and
  // carries the compiler `angle` selector; MuJoCo applies the degree->radian
  // conversion at compile, per consuming element -- identical on both paths.
  bool degree = true;          // MJCF default is degrees
  for (const auto& c : m.compilers) {
    if (!c) continue;
    ApplyMjs(*c, &spec_->compiler);
    if (c->angle.has_value())
      degree = (*c->angle == AngleUnit::degree);
    if (c->eulerseq.has_value()) {
      const std::string& s = *c->eulerseq;
      for (int i = 0; i < 3 && i < static_cast<int>(s.size()); ++i)
        spec_->compiler.eulerseq[i] = s[i];
    }
    if (c->strippath.has_value()) spec_->strippath = *c->strippath;
    if (c->alignfree.has_value())
      spec_->compiler.alignfree = *c->alignfree ? mjALIGNFREE_TRUE
                                                : mjALIGNFREE_FALSE;
    // assetdir seeds mesh/texture dirs that an explicit dir overrides.
    if (c->assetdir.has_value()) {
      if (!c->meshdir.has_value())
        mjs_setString(spec_->compiler.meshdir, c->assetdir->c_str());
      if (!c->texturedir.has_value())
        mjs_setString(spec_->compiler.texturedir, c->assetdir->c_str());
    }
  }
  spec_->compiler.degree = degree ? 1 : 0;

  // Asset resolution: point the spec at the model directory so on-disk
  // meshdir/texturedir resolve exactly as the XML path's VFS base_dir does.
  if (!opts.base_dir.empty())
    mjs_setString(spec_->modelfiledir, opts.base_dir.c_str());
  if (m.model.has_value())
    mjs_setString(spec_->modelname, m.model->c_str());
}

void Builder::BuildFlag(const Flag& f, mjOption& opt) {
  // <flag> booleans: true == "enable", false == "disable". Disable-family flags
  // set their bit when disabled; enable-family flags set their bit when enabled.
  auto dis = [&](const ps::opt<bool>& v, int bit) {
    if (v.has_value() && !*v) opt.disableflags |= bit;
  };
  auto ena = [&](const ps::opt<bool>& v, int bit) {
    if (v.has_value() && *v) opt.enableflags |= bit;
  };
  dis(f.constraint, mjDSBL_CONSTRAINT);
  dis(f.equality, mjDSBL_EQUALITY);
  dis(f.frictionloss, mjDSBL_FRICTIONLOSS);
  dis(f.limit, mjDSBL_LIMIT);
  dis(f.contact, mjDSBL_CONTACT);
  dis(f.spring, mjDSBL_SPRING);
  dis(f.damper, mjDSBL_DAMPER);
  dis(f.gravity, mjDSBL_GRAVITY);
  dis(f.clampctrl, mjDSBL_CLAMPCTRL);
  dis(f.warmstart, mjDSBL_WARMSTART);
  dis(f.filterparent, mjDSBL_FILTERPARENT);
  dis(f.actuation, mjDSBL_ACTUATION);
  dis(f.refsafe, mjDSBL_REFSAFE);
  dis(f.sensor, mjDSBL_SENSOR);
  dis(f.midphase, mjDSBL_MIDPHASE);
  dis(f.eulerdamp, mjDSBL_EULERDAMP);
  dis(f.autoreset, mjDSBL_AUTORESET);
  dis(f.nativeccd, mjDSBL_NATIVECCD);
  dis(f.island, mjDSBL_ISLAND);
  dis(f.multiccd, mjDSBL_MULTICCD);
  ena(f.override_, mjENBL_OVERRIDE);
  ena(f.energy, mjENBL_ENERGY);
  ena(f.fwdinv, mjENBL_FWDINV);
  ena(f.invdiscrete, mjENBL_INVDISCRETE);
  ena(f.sleep, mjENBL_SLEEP);
  ena(f.diagexact, mjENBL_DIAGEXACT);
}

void Builder::BuildOption(const Option& o) {
  mjOption& p = spec_->option;
#define PS_SCALAR(field) if (o.field.has_value()) p.field = *o.field;
#define PS_VEC(field, n) \
  if (o.field.has_value()) for (int i = 0; i < n; ++i) p.field[i] = (*o.field)[i];
  PS_SCALAR(timestep) PS_SCALAR(impratio) PS_SCALAR(tolerance)
  PS_SCALAR(ls_tolerance) PS_SCALAR(noslip_tolerance) PS_SCALAR(ccd_tolerance)
  PS_SCALAR(sleep_tolerance) PS_SCALAR(density) PS_SCALAR(viscosity)
  PS_SCALAR(o_margin) PS_SCALAR(iterations) PS_SCALAR(ls_iterations)
  PS_SCALAR(noslip_iterations) PS_SCALAR(ccd_iterations) PS_SCALAR(sdf_iterations)
  PS_SCALAR(sdf_initpoints)
  PS_VEC(gravity, 3) PS_VEC(wind, 3) PS_VEC(magnetic, 3)
  PS_VEC(o_solref, 2) PS_VEC(o_solimp, 5) PS_VEC(o_friction, 5)
#undef PS_SCALAR
#undef PS_VEC
  if (o.integrator.has_value()) p.integrator = IntegratorToMjt(*o.integrator);
  if (o.cone.has_value()) p.cone = ConeToMjt(*o.cone);
  if (o.jacobian.has_value()) p.jacobian = JacToMjt(*o.jacobian);
  if (o.solver.has_value()) p.solver = SolverToMjt(*o.solver);
  if (o.actuatorgroupdisable.has_value()) {
    int mask = 0;
    for (int g : *o.actuatorgroupdisable) mask |= (1 << g);
    p.disableactuator = mask;
  }
  for (const auto& f : o.flags) if (f) BuildFlag(*f, p);
}

void Builder::BuildSize(const Size& s) {
  if (s.nkey.has_value()) spec_->nkey = *s.nkey;
  if (s.nuserdata.has_value()) spec_->nuserdata = *s.nuserdata;
  if (s.nconmax.has_value()) spec_->nconmax = *s.nconmax;
  if (s.nstack.has_value()) spec_->nstack = *s.nstack;
  if (s.njmax.has_value()) spec_->njmax = *s.njmax;
  if (s.nuser_body.has_value()) spec_->nuser_body = *s.nuser_body;
  if (s.nuser_jnt.has_value()) spec_->nuser_jnt = *s.nuser_jnt;
  if (s.nuser_geom.has_value()) spec_->nuser_geom = *s.nuser_geom;
  if (s.nuser_site.has_value()) spec_->nuser_site = *s.nuser_site;
  if (s.nuser_cam.has_value()) spec_->nuser_cam = *s.nuser_cam;
  if (s.nuser_tendon.has_value()) spec_->nuser_tendon = *s.nuser_tendon;
  if (s.nuser_actuator.has_value()) spec_->nuser_actuator = *s.nuser_actuator;
  if (s.nuser_sensor.has_value()) spec_->nuser_sensor = *s.nuser_sensor;
  // memory string is scanned by the fallback (unsupported unit suffixes route
  // to XML); a plain integer is parsed here.
  if (s.memory.has_value()) {
    try { spec_->memory = std::stoull(*s.memory); } catch (...) {}
  }
}

void Builder::BuildStatistic(const Statistic& s) {
  mjStatistic& st = spec_->stat;
  if (s.meaninertia.has_value()) st.meaninertia = *s.meaninertia;
  if (s.meanmass.has_value()) st.meanmass = *s.meanmass;
  if (s.meansize.has_value()) st.meansize = *s.meansize;
  if (s.extent.has_value()) st.extent = *s.extent;
  if (s.center.has_value()) for (int i = 0; i < 3; ++i) st.center[i] = (*s.center)[i];
}

void Builder::BuildVisual(const Visual& v) {
  mjVisual& mv = spec_->visual;
#define PS_S(src, field, dst) if ((src).field.has_value()) dst = *(src).field;
#define PS_A(src, field, dst, n) \
  if ((src).field.has_value()) for (int i = 0; i < n; ++i) dst[i] = (*(src).field)[i];
  for (const auto& g : v.visualGlobals) if (g) {
    PS_S(*g, cameraid, mv.global.cameraid) PS_S(*g, orthographic, mv.global.orthographic)
    PS_S(*g, fovy, mv.global.fovy) PS_S(*g, ipd, mv.global.ipd)
    PS_S(*g, azimuth, mv.global.azimuth) PS_S(*g, elevation, mv.global.elevation)
    PS_S(*g, linewidth, mv.global.linewidth) PS_S(*g, glow, mv.global.glow)
    PS_S(*g, offwidth, mv.global.offwidth) PS_S(*g, offheight, mv.global.offheight)
    PS_S(*g, realtime, mv.global.realtime)
    PS_S(*g, ellipsoidinertia, mv.global.ellipsoidinertia)
    PS_S(*g, bvactive, mv.global.bvactive)
  }
  for (const auto& q : v.visualQualitys) if (q) {
    PS_S(*q, shadowsize, mv.quality.shadowsize) PS_S(*q, offsamples, mv.quality.offsamples)
    PS_S(*q, numslices, mv.quality.numslices) PS_S(*q, numstacks, mv.quality.numstacks)
    PS_S(*q, numquads, mv.quality.numquads)
  }
  for (const auto& h : v.visualHeadlights) if (h) {
    PS_A(*h, ambient, mv.headlight.ambient, 3) PS_A(*h, diffuse, mv.headlight.diffuse, 3)
    PS_A(*h, specular, mv.headlight.specular, 3) PS_S(*h, active, mv.headlight.active)
  }
  for (const auto& mp : v.visualMaps) if (mp) {
    PS_S(*mp, stiffness, mv.map.stiffness) PS_S(*mp, stiffnessrot, mv.map.stiffnessrot)
    PS_S(*mp, force, mv.map.force) PS_S(*mp, torque, mv.map.torque)
    PS_S(*mp, alpha, mv.map.alpha) PS_S(*mp, fogstart, mv.map.fogstart)
    PS_S(*mp, fogend, mv.map.fogend) PS_S(*mp, znear, mv.map.znear)
    PS_S(*mp, zfar, mv.map.zfar) PS_S(*mp, haze, mv.map.haze)
    PS_S(*mp, shadowclip, mv.map.shadowclip) PS_S(*mp, shadowscale, mv.map.shadowscale)
    PS_S(*mp, actuatortendon, mv.map.actuatortendon)
  }
  for (const auto& sc : v.visualScales) if (sc) {
    PS_S(*sc, forcewidth, mv.scale.forcewidth) PS_S(*sc, contactwidth, mv.scale.contactwidth)
    PS_S(*sc, contactheight, mv.scale.contactheight) PS_S(*sc, connect, mv.scale.connect)
    PS_S(*sc, com, mv.scale.com) PS_S(*sc, camera, mv.scale.camera)
    PS_S(*sc, light, mv.scale.light) PS_S(*sc, selectpoint, mv.scale.selectpoint)
    PS_S(*sc, jointlength, mv.scale.jointlength) PS_S(*sc, jointwidth, mv.scale.jointwidth)
    PS_S(*sc, actuatorlength, mv.scale.actuatorlength)
    PS_S(*sc, actuatorwidth, mv.scale.actuatorwidth)
    PS_S(*sc, framelength, mv.scale.framelength) PS_S(*sc, framewidth, mv.scale.framewidth)
    PS_S(*sc, constraint, mv.scale.constraint) PS_S(*sc, slidercrank, mv.scale.slidercrank)
    PS_S(*sc, frustum, mv.scale.frustum)
  }
  for (const auto& r : v.visualRgbas) if (r) {
    PS_A(*r, fog, mv.rgba.fog, 4) PS_A(*r, haze, mv.rgba.haze, 4)
    PS_A(*r, force, mv.rgba.force, 4) PS_A(*r, inertia, mv.rgba.inertia, 4)
    PS_A(*r, joint, mv.rgba.joint, 4) PS_A(*r, actuator, mv.rgba.actuator, 4)
    PS_A(*r, actuatornegative, mv.rgba.actuatornegative, 4)
    PS_A(*r, actuatorpositive, mv.rgba.actuatorpositive, 4)
    PS_A(*r, com, mv.rgba.com, 4) PS_A(*r, camera, mv.rgba.camera, 4)
    PS_A(*r, light, mv.rgba.light, 4) PS_A(*r, selectpoint, mv.rgba.selectpoint, 4)
    PS_A(*r, connect, mv.rgba.connect, 4) PS_A(*r, contactpoint, mv.rgba.contactpoint, 4)
    PS_A(*r, contactforce, mv.rgba.contactforce, 4)
    PS_A(*r, contactfriction, mv.rgba.contactfriction, 4)
    PS_A(*r, contacttorque, mv.rgba.contacttorque, 4)
    PS_A(*r, contactgap, mv.rgba.contactgap, 4)
    PS_A(*r, rangefinder, mv.rgba.rangefinder, 4)
    PS_A(*r, constraint, mv.rgba.constraint, 4)
    PS_A(*r, slidercrank, mv.rgba.slidercrank, 4)
    PS_A(*r, crankbroken, mv.rgba.crankbroken, 4) PS_A(*r, frustum, mv.rgba.frustum, 4)
    PS_A(*r, bv, mv.rgba.bv, 4) PS_A(*r, bvactive, mv.rgba.bvactive, 4)
  }
#undef PS_S
#undef PS_A
}

// --- defaults tree ------------------------------------------------------- //
void Builder::BuildDefaultNode(const Default& d, mjsDefault* into) {
  // Apply each authored per-family template onto the matching mjsDefault slot.
  for (const auto& x : d.joint) if (x) ApplyMjs(*x, into->joint);
  for (const auto& x : d.geom) {
    if (!x) continue;
    ApplyMjs(*x, into->geom);
    // shellinertia/fluidshape are emitter-waived keyword->enum conversions
    // applied to real geoms in BuildSubtree; OneGeom applies them to default
    // templates too, so a class-level fluidshape="ellipsoid" (balloons/cards)
    // propagates to inheriting geoms.
    if (x->shellinertia.has_value())
      into->geom->typeinertia = *x->shellinertia ? mjINERTIA_SHELL : mjINERTIA_VOLUME;
    if (x->fluidshape.has_value())
      into->geom->fluid_ellipsoid = (*x->fluidshape == FluidShape::ellipsoid) ? 1 : 0;
  }
  for (const auto& x : d.site) if (x) ApplyMjs(*x, into->site);
  for (const auto& x : d.camera) if (x) ApplyMjs(*x, into->camera);
  for (const auto& x : d.light) if (x) ApplyMjs(*x, into->light);
  for (const auto& x : d.material) if (x) ApplyMjs(*x, into->material);
  for (const auto& x : d.mesh) if (x) ApplyMjs(*x, into->mesh);
  for (const auto& x : d.pair) if (x) ApplyMjs(*x, into->pair);
  for (const auto& x : d.equality) if (x) ApplyMjs(*x, into->equality);
  for (const auto& x : d.tendon) if (x) ApplyMjs(*x, into->tendon);
  // Actuator-family templates all fold onto the single mjsDefault::actuator.
  // Each runs ApplyMjs then the gain/bias shortcut, mirroring OneActuator on a
  // <default> entry, so class-level shortcut params (kp/kv/...) propagate.
  for (const auto& x : d.general) if (x) ApplyActuatorTail(*x, into->actuator);
  auto def_act = [&](const auto& list) {
    for (const auto& x : list) if (x) { ApplyMjs(*x, into->actuator);
                                        ApplyActuatorShortcut(*x, into->actuator); }
  };
  def_act(d.motor);
  def_act(d.position);
  def_act(d.velocity);
  def_act(d.intvelocity);
  def_act(d.damper);
  def_act(d.cylinder);
  def_act(d.muscle);
  def_act(d.adhesion);
  def_act(d.dcmotor);

  for (const auto& sub : d.subclasses) {
    if (!sub) continue;
    const char* cn = sub->dclass.has_value() ? sub->dclass->c_str() : "";
    mjsDefault* child = mjs_addDefault(spec_, cn, into);
    BuildDefaultNode(*sub, child);
  }
}

void Builder::BuildDefaults(const Model& m) {
  mjsDefault* root = mjs_getSpecDefault(spec_);
  for (const auto& d : m.defaults) if (d) BuildDefaultNode(*d, root);
}

// --- plugins ------------------------------------------------------------- //
void Builder::ApplyPluginConfigs(
    const std::vector<std::unique_ptr<Config>>& cfgs, mjsPlugin* plugin) {
  if (cfgs.empty()) return;
  // mjs_setPluginAttributes reinterprets the pointer as a
  // std::map<string,string,std::less<>> and move-assigns it (user_api.cc).
  std::map<std::string, std::string, std::less<>> attribs;
  for (const auto& c : cfgs)
    if (c && c->key.has_value()) attribs[*c->key] = c->value.value_or("");
  mjs_setPluginAttributes(plugin, &attribs);
}

void Builder::SetPluginFields(
    const char* plugin_name, const char* instance_name,
    const std::vector<std::unique_ptr<Config>>& cfgs, mjsPlugin* plugin) {
  plugin->active = 1;
  mjs_setString(plugin->plugin_name, plugin_name ? plugin_name : "");
  const bool has_instance = instance_name && instance_name[0];
  mjs_setString(plugin->name, has_instance ? instance_name : "");
  if (!has_instance) {
    plugin->element = mjs_addPlugin(spec_)->element;
    ApplyPluginConfigs(cfgs, plugin);
  }
}

void Builder::ApplyPluginRef(const PluginRef& pr, mjsPlugin* plugin) {
  SetPluginFields(pr.plugin.has_value() ? pr.plugin->c_str() : nullptr,
                  (pr.instance.has_value() && !pr.instance->name.empty())
                      ? pr.instance->name.c_str() : nullptr,
                  pr.config, plugin);
}

// Top-level <extension><plugin><instance><config>: activate each plugin type
// and create one mjsPlugin per predefined instance. Runs before any element
// that carries an inline plugin (ordering: explicit instances must precede
// implicit ones), so this is invoked early in Build().
void Builder::BuildExtensions(const Model& m) {
  for (const auto& ext : m.extensions) {
    if (!ext) continue;
    for (const auto& def : ext->pluginDefs) {
      if (!def) continue;
      const std::string pname = def->plugin.value_or("");
      if (!pname.empty()) mjs_activatePlugin(spec_, pname.c_str());
      for (const auto& inst : def->pluginInstances) {
        if (!inst) continue;
        mjsPlugin* p = mjs_addPlugin(spec_);
        mjs_setString(p->plugin_name, pname.c_str());
        if (inst->name.has_value()) mjs_setString(p->name, inst->name->c_str());
        ApplyPluginConfigs(inst->config, p);
      }
    }
  }
}

// --- attach (model assets) ----------------------------------------------- //
// Parse each <asset><model file=...> into a child spec and register it under
// its (optionally overridden) modelname, then enable deep copy so <attach>
// grafts a self-contained copy of the child subtree into the main spec. The
// child specs are owned by the main spec after mjs_addSpec and freed with it.
void Builder::BuildModelAssets(const Model& m) {
  bool any = false;
  for (const auto& a : m.assets) {
    if (!a) continue;
    for (const auto& ma : a->modelAssets) {
      if (!ma || !ma->file.has_value()) continue;
      const std::string path =
          base_dir_.empty() ? *ma->file : base_dir_ + "/" + *ma->file;
      char err[1024] = {0};
      // mj_parse dispatches by content-type (XML/URDF/MJB); a null content-type
      // dispatches by file extension, matching the XML reader's <model> handling.
      const char* ctype =
          ma->content_type.has_value() ? ma->content_type->c_str() : nullptr;
      mjSpec* child = mj_parse(path.c_str(), ctype, nullptr, err, sizeof(err));
      if (!child) continue;  // a bad child file fails identically on the XML leg
      if (ma->name.has_value())
        mjs_setString(child->modelname, ma->name->c_str());
      mjs_addSpec(spec_, child);
      any = true;
    }
  }
  if (any) mjs_setDeepCopy(spec_, 1);
}

// --- assets -------------------------------------------------------------- //
void Builder::BuildAssets(const Model& m) {
  for (const auto& a : m.assets) {
    if (!a) continue;
    for (const auto& tex : a->textures) {
      if (!tex) continue;
      mjsTexture* t = mjs_addTexture(spec_);
      ApplyMjs(*tex, t);
      // Cube-face separate files + gridlayout are emitter-waived (a char[12]
      // array and a per-face string vector). Mirror xml_native_reader.cc
      // OneTexture: when any face file is authored, set all six cubefiles slots
      // in reader order (right,left,up,down,front,back), empty for unauthored;
      // memcpy the gridlayout string into the char[12] field.
      const ps::opt<std::string>* faces[6] = {
          &tex->fileright, &tex->fileleft, &tex->fileup,
          &tex->filedown,  &tex->filefront, &tex->fileback};
      bool any_face = false;
      for (const auto* f : faces) if (f->has_value()) any_face = true;
      if (any_face) {
        for (int i = 0; i < 6; ++i)
          mjs_setInStringVec(t->cubefiles, i,
                             faces[i]->has_value() ? faces[i]->value().c_str()
                                                   : "");
      }
      if (tex->gridlayout.has_value()) {
        const std::string& gl = *tex->gridlayout;
        for (int i = 0; i < 12; ++i)
          t->gridlayout[i] =
              i < static_cast<int>(gl.size()) ? gl[i] : '\0';
      }
    }
    for (const auto& mat : a->materials) {
      if (!mat) continue;
      mjsMaterial* mm = mjs_addMaterial(spec_, nullptr);
      ApplyMjs(*mat, mm);
      for (const auto& layer : mat->layers) {
        if (layer && layer->texture.has_value() && layer->role.has_value())
          mjs_setInStringVec(mm->textures, TexRoleToMjt(*layer->role),
                             layer->texture->name.c_str());
      }
    }
    for (const auto& mesh : a->meshs) {
      if (!mesh) continue;
      mjsMesh* mo = mjs_addMesh(spec_, nullptr);
      ApplyMjs(*mesh, mo);
      // Builtin procedural mesh: no mjsMesh struct field exists, so the geometry
      // is generated by mjs_makeMesh from (builtin, params) -- mirrors the XML
      // reader's OneMesh. builtin/params stay emitter-waived for this reason.
      if (mesh->builtin.has_value()) {
        std::vector<double> params;
        if (mesh->params.has_value())
          params.assign(mesh->params->begin(), mesh->params->end());
        mjs_makeMesh(mo, MeshBuiltinToMjt(*mesh->builtin),
                     params.empty() ? nullptr : params.data(),
                     static_cast<int>(params.size()));
      }
      if (!mesh->plugin.empty() && mesh->plugin.front())
        ApplyPluginRef(*mesh->plugin.front(), &mo->plugin);
    }
    for (const auto& hf : a->hfields) {
      if (!hf) continue;
      mjsHField* h = mjs_addHField(spec_);
      // Hfield.elevation aliases userdata (float); ApplyMjs handles name/file/
      // nrow/ncol/size and copies elevation verbatim.
      ApplyMjs(*hf, h);
      // OneHField: for a procedural hfield (no file, nrow/ncol > 0) stock builds
      // userdata itself -- elevation copied in REVERSE row order (so the XML
      // string reads top-to-bottom), or zero-filled when no elevation is given.
      // ApplyMjs's verbatim copy matches neither; redo it to match stock.
      const bool has_file = hf->file.has_value() && !hf->file->empty();
      const int nrow = hf->nrow.value_or(0);
      const int ncol = hf->ncol.value_or(0);
      if (!has_file && nrow > 0 && ncol > 0) {
        std::vector<float> buf(static_cast<std::size_t>(nrow) * ncol, 0.0f);
        if (hf->elevation.has_value() &&
            static_cast<int>(hf->elevation->size()) == nrow * ncol) {
          const auto& src = *hf->elevation;
          for (int i = 0; i < nrow; ++i)
            for (int j = 0; j < ncol; ++j)
              buf[(nrow - 1 - i) * ncol + j] =
                  static_cast<float>(src[i * ncol + j]);
        }
        mjs_setFloat(h->userdata, buf.data(), static_cast<int>(buf.size()));
      }
    }
    // <skin> is also accepted under <asset> (legacy location); same builder as
    // the <deformable> one.
    for (const auto& sk : a->skins) {
      if (sk) BuildSkin(*sk);
    }
  }
}

// --- worldbody ----------------------------------------------------------- //
void Builder::BuildWorldbody(const Model& m) {
  mjsBody* world = mjs_findBody(spec_, "world");
  const mjsDefault* root = mjs_getSpecDefault(spec_);
  // Deep copy must be on while the worldbody is built so every mjs_attach (model
  // assets, replicate expansion, composite graft) grafts a self-contained copy;
  // the reader holds it true across the whole worldbody parse and restores it
  // before compile.
  mjs_setDeepCopy(spec_, 1);
  for (const auto& wb : m.worldbody) {
    if (wb) BuildSubtree(world, wb->subtree, nullptr, root);
  }
  mjs_setDeepCopy(spec_, 0);
}

void Builder::BuildSubtree(mjsBody* body,
                           const std::vector<BodyChildAny>& subtree,
                           mjsFrame* frame, const mjsDefault* inherited) {
  for (const auto& item : subtree) {
    std::visit([&](const auto& p) {
      if (!p) return;
      using T = std::decay_t<decltype(*p)>;
      const T& e = *p;
      if constexpr (std::is_same_v<T, Geom>) {
        const mjsDefault* def = FindClass(e.dclass, inherited);
        mjsGeom* g = mjs_addGeom(body, def);
        ApplyMjs(e, g);
        if (e.shellinertia.has_value())
          g->typeinertia = *e.shellinertia ? mjINERTIA_SHELL : mjINERTIA_VOLUME;
        if (e.fluidshape.has_value())
          g->fluid_ellipsoid = (*e.fluidshape == FluidShape::ellipsoid) ? 1 : 0;
        if (frame) mjs_setFrame(g->element, frame);
        if (!e.plugin.empty() && e.plugin.front())
          ApplyPluginRef(*e.plugin.front(), &g->plugin);
        AutoName(g->element, e.serial);
      } else if constexpr (std::is_same_v<T, Joint>) {
        const mjsDefault* def = FindClass(e.dclass, inherited);
        mjsJoint* j = mjs_addJoint(body, def);
        ApplyMjs(e, j);
        if (frame) mjs_setFrame(j->element, frame);
        AutoName(j->element, e.serial);
      } else if constexpr (std::is_same_v<T, FreeJoint>) {
        mjsJoint* j = mjs_addFreeJoint(body);
        if (frame) mjs_setFrame(j->element, frame);
        mjs_setDefault(j->element, inherited);
        ApplyMjs(e, j);
        AutoName(j->element, e.serial);
      } else if constexpr (std::is_same_v<T, Site>) {
        const mjsDefault* def = FindClass(e.dclass, inherited);
        mjsSite* s = mjs_addSite(body, def);
        ApplyMjs(e, s);
        if (frame) mjs_setFrame(s->element, frame);
        AutoName(s->element, e.serial);
      } else if constexpr (std::is_same_v<T, Camera>) {
        const mjsDefault* def = FindClass(e.dclass, inherited);
        mjsCamera* c = mjs_addCamera(body, def);
        ApplyMjs(e, c);
        if (frame) mjs_setFrame(c->element, frame);
        AutoName(c->element, e.serial);
      } else if constexpr (std::is_same_v<T, Light>) {
        const mjsDefault* def = FindClass(e.dclass, inherited);
        mjsLight* l = mjs_addLight(body, def);
        ApplyMjs(e, l);
        if (frame) mjs_setFrame(l->element, frame);
        AutoName(l->element, e.serial);
      } else if constexpr (std::is_same_v<T, Body>) {
        const mjsDefault* childdef =
            (e.childclass.has_value() && !e.childclass->name.empty())
                ? mjs_findDefault(spec_, e.childclass->name.c_str())
                : nullptr;
        mjsBody* child = mjs_addBody(body, childdef);
        ApplyMjs(e, child);
        const mjsDefault* bodydef = childdef ? childdef : inherited;
        mjs_setDefault(child->element, bodydef);
        if (frame) mjs_setFrame(child->element, frame);
        AutoName(child->element, e.serial);
        for (const auto& in : e.inertial) {
          if (in) { child->explicitinertial = 1; ApplyMjs(*in, child); }
        }
        BuildSubtree(child, e.subtree, nullptr, bodydef);
      } else if constexpr (std::is_same_v<T, Frame>) {
        const mjsDefault* childdef =
            (e.dclass.has_value() && !e.dclass->name.empty())
                ? mjs_findDefault(spec_, e.dclass->name.c_str())
                : nullptr;
        mjsFrame* pframe = mjs_addFrame(body, frame);
        ApplyMjs(e, pframe);
        const mjsDefault* framedef = childdef ? childdef : inherited;
        mjs_setDefault(pframe->element, framedef);
        BuildSubtree(body, e.subtree, pframe, framedef);
      } else if constexpr (std::is_same_v<T, PluginRef>) {
        // A <plugin> directly under a body sets that body's plugin sub-struct.
        ApplyPluginRef(e, &body->plugin);
      } else if constexpr (std::is_same_v<T, Attach>) {
        // Native subtree attach (mirror the reader): parent is the enclosing
        // frame, or a fresh frame on this body; child is the named body in the
        // registered child spec (or the whole child model when body is empty).
        // Deep copy (enabled in BuildModelAssets) grafts a self-contained copy.
        if (e.model.has_value()) {
          if (mjSpec* child = mjs_findSpec(spec_, e.model->name.c_str())) {
            mjsFrame* pframe = frame ? frame : mjs_addFrame(body, nullptr);
            mjsElement* childel = nullptr;
            if (e.body.has_value() && !e.body->empty()) {
              if (mjsBody* cb = mjs_findBody(child, e.body->c_str()))
                childel = cb->element;
            } else {
              childel = child->element;
            }
            if (childel)
              mjs_attach(pframe->element, childel,
                         e.prefix.has_value() ? e.prefix->c_str() : "", "");
          }
        }
      } else if constexpr (std::is_same_v<T, Replicate>) {
        ExpandReplicate(body, e, frame, inherited);
      } else if constexpr (std::is_same_v<T, Composite>) {
        GraftComposite(body, e, frame);
      } else if constexpr (std::is_same_v<T, Flexcomp>) {
        BuildFlexcomp(body, e);
      }
    }, item.node);
  }
}

// --- <replicate> --------------------------------------------------------- //
// Reader-mirror expansion (xml_native_reader.cc <replicate> branch). Build the
// child subtree once under a temp body/frame, then attach `count` deep copies
// with an accumulated frame pose and a zero-padded suffix, then delete the temp.
// Free-jointed bodies MUST expand this way -- grafting through an extra attach
// frame (route-D) diverges xpos because the frame is not folded into qpos0.
// ps_path_diff is the drift net for this reimplementation.
void Builder::ExpandReplicate(mjsBody* body, const Replicate& e, mjsFrame* frame,
                              const mjsDefault* inherited) {
  const int count = e.count;
  double offset[3] = {0, 0, 0}, euler[3] = {0, 0, 0};
  if (e.offset.has_value()) for (int i = 0; i < 3; ++i) offset[i] = (*e.offset)[i];
  if (e.euler.has_value()) for (int i = 0; i < 3; ++i) euler[i] = (*e.euler)[i];
  const std::string sep = e.sep.value_or("");

  // per-step rotation, resolved in the spec's compiler angle/eulerseq context
  mjsOrientation alt;
  mjs_defaultOrientation(&alt);
  alt.type = mjORIENTATION_EULER;
  mju_copy3(alt.euler, euler);
  double rotation[4] = {1, 0, 0, 0};
  mjs_resolveOrientation(rotation, spec_->compiler.degree,
                         spec_->compiler.eulerseq, &alt);

  const mjsDefault* childdef =
      (e.childclass.has_value() && !e.childclass->name.empty())
          ? mjs_findDefault(spec_, e.childclass->name.c_str())
          : nullptr;
  const mjsDefault* framedef = childdef ? childdef : inherited;

  // temp subtree body + frame carrying the replicated children
  mjsBody* subtree = mjs_addBody(body, childdef);
  mjsFrame* pframe = mjs_addFrame(subtree, frame);
  mjs_setDefault(pframe->element, framedef);
  for (const auto& in : e.inertial) {
    if (in) { subtree->explicitinertial = 1; ApplyMjs(*in, subtree); }
  }
  BuildSubtree(subtree, e.subtree, pframe, framedef);

  double pos[3] = {0, 0, 0}, quat[4] = {1, 0, 0, 0};
  for (int i = 0; i < count; ++i) {
    // recompute the absolute rotation at full precision from i*euler
    alt.euler[0] = i * euler[0];
    alt.euler[1] = i * euler[1];
    alt.euler[2] = i * euler[2];
    mjs_resolveOrientation(quat, spec_->compiler.degree,
                           spec_->compiler.eulerseq, &alt);
    mju_copy3(pframe->pos, pos);
    mju_copy4(pframe->quat, quat);
    FrameAccum(pos, quat, offset, rotation);  // advance pos for the next copy
    std::string suffix = ReplicateSuffix(sep, count, i);
    if (!mjs_attach(body->element, pframe->element, "", suffix.c_str())) {
      if (err_.empty()) err_ = mjs_getError(spec_);
      break;
    }
  }
  mjs_delete(spec_, subtree->element);
}

// --- <composite> (route-D graft) ----------------------------------------- //
// Composites have no mjsComposite; the parser expands them into plain bodies
// during mj_parseXMLString (which also surfaces the deprecated-type / non-1D
// errors from user_composite.cc -- only "cable" survives). Emit a minimal
// complete fragment (compiler + defaults + extensions + a holder body carrying
// this one composite), parse it, register it, and attach the expanded chain
// under a frame in `body`. The holder sits at the origin; `body` carries the
// world placement, exactly as the inline composite would. ps_path_diff is the
// drift net. Assets are NOT copied into the fragment: mjs_addSpec merges the
// fragment into spec_ (which already holds every asset), and a duplicated
// asset name (e.g. a shared "texplane" from an <include>d scene) would collide
// at compile ("repeated name"); the composite geometry resolves any material
// against the parent spec after the merge.
void Builder::GraftComposite(mjsBody* body, const Composite& e, mjsFrame* frame) {
  Model frag;
  if (model_) {
    for (const auto& c : model_->compilers) if (c) frag.compilers.push_back(Clone(*c));
    for (const auto& d : model_->defaults) if (d) frag.defaults.push_back(Clone(*d));
    for (const auto& x : model_->extensions) if (x) frag.extensions.push_back(Clone(*x));
  }
  auto holder = std::make_unique<Body>();
  holder->name = "holder";
  BodyChildAny comp_child;
  comp_child.node = Clone(e);
  holder->subtree.push_back(std::move(comp_child));
  auto world = std::make_unique<Body>();  // the <worldbody> wrapper
  BodyChildAny holder_child;
  holder_child.node = std::move(holder);
  world->subtree.push_back(std::move(holder_child));
  frag.worldbody.push_back(std::move(world));

  std::vector<ps::Diagnostic> werr;
  const std::string xml = io::WriteMjcf(frag, &werr);
  if (xml.empty()) {
    if (err_.empty()) err_ = "composite: failed to serialize fragment";
    return;
  }
  char perr[1024] = {0};
  mjSpec* F = mj_parseXMLString(xml.c_str(), nullptr, perr, sizeof(perr));
  if (!F) {  // deprecated composite type / dim error surfaces here, on all paths
    if (err_.empty()) err_ = perr;
    return;
  }
  mjs_addSpec(spec_, F);  // owned by spec_, freed with it
  mjsBody* fholder = mjs_findBody(F, "holder");
  if (!fholder) { if (err_.empty()) err_ = "composite: holder body vanished"; return; }
  mjsFrame* fr = frame ? frame : mjs_addFrame(body, nullptr);
  std::vector<mjsElement*> kids;
  for (mjsElement* c = mjs_firstChild(fholder, mjOBJ_BODY, 0); c;
       c = mjs_nextChild(fholder, c, 0))
    kids.push_back(c);
  for (mjsElement* k : kids)
    if (!mjs_attach(fr->element, k, "", "")) {
      if (err_.empty()) err_ = mjs_getError(spec_);
      return;
    }
}

// --- <flexcomp> ---------------------------------------------------------- //
// Primary route: mjs_makeFlex generates the flex geometry from the flexcomp
// parameters (mirrors mjCFlexcomp::Make), then post-set the mjsFlex fields the
// reader's OneFlexcomp writes after generation. Unauthored args take the
// mjCFlexcomp / mjs_defaultFlex defaults. Pins and flexcomp plugins have no
// mjs_makeFlex / mjsFlex surface and are not reproduced on this route.
void Builder::BuildFlexcomp(mjsBody* body, const Flexcomp& e) {
  const std::string type_s =
      e.type.has_value() ? std::string(ToMjcf(*e.type)) : std::string("grid");
  std::string dof_str;
  const char* dof_s = nullptr;
  if (e.dof.has_value()) { dof_str = std::string(ToMjcf(*e.dof)); dof_s = dof_str.c_str(); }
  const int dim = e.dim.value_or(2);
  int count[3] = {10, 10, 10};
  if (e.count.has_value()) for (int i = 0; i < 3; ++i) count[i] = (*e.count)[i];
  int cellcount[3] = {-1, -1, -1};
  if (e.cellcount.has_value()) for (int i = 0; i < 3; ++i) cellcount[i] = (*e.cellcount)[i];
  double spacing[3] = {0.02, 0.02, 0.02};
  if (e.spacing.has_value()) for (int i = 0; i < 3; ++i) spacing[i] = (*e.spacing)[i];
  double scale[3] = {1, 1, 1};
  if (e.scale.has_value()) for (int i = 0; i < 3; ++i) scale[i] = (*e.scale)[i];
  double pos[3] = {0, 0, 0}, quat[4] = {1, 0, 0, 0}, origin[3] = {0, 0, 0};
  if (e.pos.has_value()) for (int i = 0; i < 3; ++i) pos[i] = (*e.pos)[i];
  if (e.quat.has_value()) for (int i = 0; i < 4; ++i) quat[i] = (*e.quat)[i];
  if (e.origin.has_value()) for (int i = 0; i < 3; ++i) origin[i] = (*e.origin)[i];
  const double radius = e.radius.value_or(0.005);
  const double mass = e.mass.value_or(1.0);
  const double inertiabox = e.inertiabox.value_or(0.005);
  const int rigid = e.rigid.value_or(false) ? 1 : 0;
  const int flatskin = e.flatskin.value_or(false) ? 1 : 0;

  // equality (from <edge equality=...>) and elastic2d (from <elasticity ...>)
  // are makeFlex arguments; the rest post-set below.
  int equality = 0;
  for (const auto& ed : e.flexcompEdges)
    if (ed && ed->equality.has_value()) equality = static_cast<int>(*ed->equality);
  int elastic2d = 0;
  for (const auto& el : e.flexElasticitys)
    if (el && el->elastic2d.has_value()) elastic2d = Elastic2DToInt(*el->elastic2d);

  const std::string name = e.name.value_or("");
  const std::string file = e.file.value_or("");
  mjsFlex* f = mjs_makeFlex(
      body, name.c_str(), type_s.c_str(), dim, dof_s, count, cellcount, spacing,
      scale, radius, mass, inertiabox, equality, rigid, flatskin, elastic2d, pos,
      quat, origin, file.empty() ? nullptr : file.c_str(), nullptr);
  if (!f) { if (err_.empty()) err_ = mjs_getError(spec_); return; }

  // root-attribute mjsFlex fields (OneFlexcomp)
  if (e.material.has_value()) mjs_setString(f->material, e.material->name.c_str());
  if (e.rgba.has_value()) for (int i = 0; i < 4; ++i) f->rgba[i] = (*e.rgba)[i];
  if (e.group.has_value()) f->group = *e.group;
  AutoName(f->element, e.serial);

  // <edge>: stiffness/damping onto the flex. equality feeds makeFlex; solref/
  // solimp cannot (mjs_makeFlex has no edge-solref surface), so post-set them on
  // the equalities makeFlex just generated for THIS flex (matched by name1 ==
  // flex name; flex names are unique). OneFlexcomp routes edge solref/solimp to
  // fcomp.def.spec.equality, from which Make seeds every generated equality.
  for (const auto& ed : e.flexcompEdges) {
    if (!ed) continue;
    if (ed->stiffness.has_value()) f->edgestiffness = *ed->stiffness;
    if (ed->damping.has_value()) f->edgedamping = *ed->damping;
    if (!ed->solref.has_value() && !ed->solimp.has_value()) continue;
    for (mjsElement* el = mjs_firstElement(spec_, mjOBJ_EQUALITY); el;
         el = mjs_nextElement(spec_, el)) {
      mjsEquality* q = mjs_asEquality(el);
      if (!q) continue;
      if (q->type != mjEQ_FLEX && q->type != mjEQ_FLEXVERT &&
          q->type != mjEQ_FLEXSTRAIN)
        continue;
      if (name != mjs_getString(q->name1)) continue;
      if (ed->solref.has_value())
        for (std::size_t i = 0; i < ed->solref->size() && i < mjNREF; ++i)
          q->solref[i] = (*ed->solref)[i];
      if (ed->solimp.has_value())
        for (std::size_t i = 0; i < ed->solimp->size() && i < mjNIMP; ++i)
          q->solimp[i] = (*ed->solimp)[i];
    }
  }
  // <elasticity>
  for (const auto& el : e.flexElasticitys) {
    if (!el) continue;
    if (el->young.has_value()) f->young = *el->young;
    if (el->poisson.has_value()) f->poisson = *el->poisson;
    if (el->damping.has_value()) f->damping = *el->damping;
    if (el->thickness.has_value()) f->thickness = *el->thickness;
  }
  // <contact>
  for (const auto& c : e.flexContacts) {
    if (!c) continue;
    if (c->contype.has_value()) f->contype = *c->contype;
    if (c->conaffinity.has_value()) f->conaffinity = *c->conaffinity;
    if (c->condim.has_value()) f->condim = *c->condim;
    if (c->priority.has_value()) f->priority = *c->priority;
    if (c->friction.has_value())
      for (std::size_t i = 0; i < c->friction->size(); ++i) f->friction[i] = (*c->friction)[i];
    if (c->solmix.has_value()) f->solmix = *c->solmix;
    if (c->solref.has_value())
      for (std::size_t i = 0; i < c->solref->size(); ++i) f->solref[i] = (*c->solref)[i];
    if (c->solimp.has_value())
      for (std::size_t i = 0; i < c->solimp->size(); ++i) f->solimp[i] = (*c->solimp)[i];
    if (c->margin.has_value()) f->margin = *c->margin;
    if (c->gap.has_value()) f->gap = *c->gap;
    if (c->internal.has_value()) f->internal = *c->internal ? 1 : 0;
    if (c->selfcollide.has_value()) f->selfcollide = FlexSelfToInt(*c->selfcollide);
    if (c->passive.has_value()) f->passive = *c->passive ? 1 : 0;
    if (c->activelayers.has_value()) f->activelayers = *c->activelayers;
  }
}

// --- contact ------------------------------------------------------------- //
void Builder::BuildContact(const Model& m) {
  for (const auto& c : m.contacts) {
    if (!c) continue;
    for (const auto& pr : c->pairs) {
      if (!pr) continue;
      const mjsDefault* def = FindClass(pr->dclass, mjs_getSpecDefault(spec_));
      mjsPair* p = mjs_addPair(spec_, def);
      ApplyMjs(*pr, p);
      AutoName(p->element, pr->serial);
    }
    for (const auto& ex : c->excludes) {
      if (!ex) continue;
      mjsExclude* x = mjs_addExclude(spec_);
      ApplyMjs(*ex, x);
      AutoName(x->element, ex->serial);
    }
  }
}

// --- equality ------------------------------------------------------------ //
void Builder::BuildEquality(const Model& m) {
  for (const auto& eq : m.equalitys) {
    if (!eq) continue;
    for (const auto& any : eq->equalities) {
      std::visit([&](const auto& p) {
        if (!p) return;
        using T = std::decay_t<decltype(*p)>;
        const T& e = *p;
        const mjsDefault* def = mjs_getSpecDefault(spec_);
        if constexpr (!std::is_same_v<T, EqualityDefault>) {
          if constexpr (requires { e.dclass; })
            def = FindClass(e.dclass, def);
        }
        mjsEquality* q = mjs_addEquality(spec_, def);
        if constexpr (std::is_same_v<T, Connect>) {
          q->type = mjEQ_CONNECT;
          ApplyMjs(e, q);
          if (e.body1.has_value()) {
            q->objtype = mjOBJ_BODY;
            mjs_setString(q->name1, e.body1->name.c_str());
            if (e.body2.has_value()) mjs_setString(q->name2, e.body2->name.c_str());
            if (e.anchor.has_value())
              for (int i = 0; i < 3; ++i) q->data[i] = (*e.anchor)[i];
          } else {
            q->objtype = mjOBJ_SITE;
            if (e.site1.has_value()) mjs_setString(q->name1, e.site1->name.c_str());
            if (e.site2.has_value()) mjs_setString(q->name2, e.site2->name.c_str());
          }
        } else if constexpr (std::is_same_v<T, Weld>) {
          q->type = mjEQ_WELD;
          ApplyMjs(e, q);
          if (e.body1.has_value()) {
            q->objtype = mjOBJ_BODY;
            mjs_setString(q->name1, e.body1->name.c_str());
            if (e.body2.has_value()) mjs_setString(q->name2, e.body2->name.c_str());
            for (int i = 0; i < 3; ++i)
              q->data[i] = e.anchor.has_value() ? (*e.anchor)[i] : 0.0;
          } else {
            q->objtype = mjOBJ_SITE;
            if (e.site1.has_value()) mjs_setString(q->name1, e.site1->name.c_str());
            if (e.site2.has_value()) mjs_setString(q->name2, e.site2->name.c_str());
          }
          if (e.relpose.has_value())
            for (int i = 0; i < 7; ++i) q->data[3 + i] = (*e.relpose)[i];
          if (e.torquescale.has_value()) q->data[10] = *e.torquescale;
        } else if constexpr (std::is_same_v<T, EqualityJoint>) {
          q->type = mjEQ_JOINT;
          ApplyMjs(e, q);
          if (e.joint1.has_value()) mjs_setString(q->name1, e.joint1->name.c_str());
          if (e.joint2.has_value()) mjs_setString(q->name2, e.joint2->name.c_str());
          if (e.polycoef.has_value())
            for (std::size_t i = 0; i < e.polycoef->size() && i < 5; ++i)
              q->data[i] = (*e.polycoef)[i];
        } else if constexpr (std::is_same_v<T, EqualityTendon>) {
          q->type = mjEQ_TENDON;
          ApplyMjs(e, q);
          if (e.tendon1.has_value()) mjs_setString(q->name1, e.tendon1->name.c_str());
          if (e.tendon2.has_value()) mjs_setString(q->name2, e.tendon2->name.c_str());
          if (e.polycoef.has_value())
            for (std::size_t i = 0; i < e.polycoef->size() && i < 5; ++i)
              q->data[i] = (*e.polycoef)[i];
        } else if constexpr (std::is_same_v<T, EqualityFlex>) {
          q->type = mjEQ_FLEX;
          ApplyMjs(e, q);
          if (e.flex.has_value()) mjs_setString(q->name1, e.flex->name.c_str());
        } else if constexpr (std::is_same_v<T, Flexvert>) {
          q->type = mjEQ_FLEXVERT;
          ApplyMjs(e, q);
          if (e.flex.has_value()) mjs_setString(q->name1, e.flex->name.c_str());
        } else if constexpr (std::is_same_v<T, Flexstrain>) {
          q->type = mjEQ_FLEXSTRAIN;
          ApplyMjs(e, q);
          if (e.flex.has_value()) mjs_setString(q->name1, e.flex->name.c_str());
          if (e.cell.has_value())
            for (int i = 0; i < 3; ++i) q->data[i] = (*e.cell)[i];
        }
        AutoName(q->element, e.serial);
      }, any.node);
    }
  }
}

// --- tendons ------------------------------------------------------------- //
void Builder::BuildTendons(const Model& m) {
  for (const auto& td : m.tendons) {
    if (!td) continue;
    for (const auto& any : td->tendons) {
      std::visit([&](const auto& p) {
        if (!p) return;
        using T = std::decay_t<decltype(*p)>;
        const T& e = *p;
        const mjsDefault* def = FindClass(e.dclass, mjs_getSpecDefault(spec_));
        mjsTendon* t = mjs_addTendon(spec_, def);
        ApplyMjs(e, t);
        AutoName(t->element, e.serial);
        if constexpr (std::is_same_v<T, Spatial>) {
          for (const auto& wp : e.path) {
            std::visit([&](const auto& wq) {
              if (!wq) return;
              using W = std::decay_t<decltype(*wq)>;
              const W& w = *wq;
              if constexpr (std::is_same_v<W, SpatialSite>) {
                if (w.site.has_value()) mjs_wrapSite(t, w.site->name.c_str());
              } else if constexpr (std::is_same_v<W, SpatialGeom>) {
                const char* ss = RefCStr(w.sidesite);
                mjs_wrapGeom(t, w.geom.has_value() ? w.geom->name.c_str() : "",
                             ss ? ss : "");
              } else if constexpr (std::is_same_v<W, Pulley>) {
                mjs_wrapPulley(t, w.divisor.value_or(0.0));
              }
            }, wp.node);
          }
        } else if constexpr (std::is_same_v<T, Fixed>) {
          for (const auto& fj : e.fixedJoints) {
            if (fj && fj->joint.has_value())
              mjs_wrapJoint(t, fj->joint->name.c_str(), fj->coef.value_or(0.0));
          }
        }
      }, any.node);
    }
  }
}

// --- actuators ----------------------------------------------------------- //
// Sets transmission type + primary target in MuJoCo's reader priority order.
// slidersite/refsite/cranklength are written by the generated ApplyMjs.
void SetTransmission(mjsActuator* out, const char* joint, const char* jip,
                     const char* tendon, const char* cranksite,
                     const char* site, const char* body) {
  if (joint) { mjs_setString(out->target, joint); out->trntype = mjTRN_JOINT; }
  else if (jip) { mjs_setString(out->target, jip); out->trntype = mjTRN_JOINTINPARENT; }
  else if (tendon) { mjs_setString(out->target, tendon); out->trntype = mjTRN_TENDON; }
  else if (cranksite) { mjs_setString(out->target, cranksite); out->trntype = mjTRN_SLIDERCRANK; }
  else if (site) { mjs_setString(out->target, site); out->trntype = mjTRN_SITE; }
  else if (body) { mjs_setString(out->target, body); out->trntype = mjTRN_BODY; }
}

// ApplyMjs for the actuator common tail without name (used for the
// general-family default template, whose name must not be set).
template <class A>
void Builder::ApplyActuatorTail(const A& e, mjsActuator* out) {
  ApplyMjs(e, out);
}

void Builder::BuildActuators(const Model& m) {
  for (const auto& act : m.actuators) {
    if (!act) continue;
    for (const auto& any : act->actuators) {
      std::visit([&](const auto& p) {
        if (!p) return;
        using T = std::decay_t<decltype(*p)>;
        const T& e = *p;
        const mjsDefault* def = FindClass(e.dclass, mjs_getSpecDefault(spec_));
        mjsActuator* a = mjs_addActuator(spec_, def);
        ApplyMjs(e, a);

        // transmission (the field set varies per actuator kind -- Muscle has no
        // site/body, Adhesion has only body; guard each by presence).
        {
          const char* joint = nullptr;
          const char* jip = nullptr;
          const char* tendon = nullptr;
          const char* cranksite = nullptr;
          const char* site = nullptr;
          const char* body = nullptr;
          if constexpr (requires { e.joint; }) joint = RefCStr(e.joint);
          if constexpr (requires { e.jointinparent; }) jip = RefCStr(e.jointinparent);
          if constexpr (requires { e.tendon; }) tendon = RefCStr(e.tendon);
          if constexpr (requires { e.cranksite; }) cranksite = RefCStr(e.cranksite);
          if constexpr (requires { e.site; }) site = RefCStr(e.site);
          if constexpr (requires { e.body; }) body = RefCStr(e.body);
          SetTransmission(a, joint, jip, tendon, cranksite, site, body);
        }

        // gain/bias shortcut (mjs_setTo*): shared with the default-template path.
        ApplyActuatorShortcut(e, a);
        // ActuatorGeneral: gaintype/biastype/dyntype/gainprm/biasprm/dynprm are
        // written directly by ApplyMjs -- no shortcut needed.
        if constexpr (std::is_same_v<T, ActuatorPlugin>) {
          // Plugin actuator: standard transmission (set above) + dyntype/actdim/
          // dynprm/actearly (ApplyMjs) + the plugin sub-struct.
          SetPluginFields(e.plugin.has_value() ? e.plugin->c_str() : nullptr,
                          (e.instance.has_value() && !e.instance->name.empty())
                              ? e.instance->name.c_str() : nullptr,
                          e.config, &a->plugin);
        }
        AutoName(a->element, e.serial);
      }, any.node);
    }
  }
}

// gain/bias shortcut chain (xml_native_reader.cc OneActuator): read the current
// mjs field as the class-default base, override with the authored value, then
// invoke the matching mjs_setTo*. Shared by real actuators and default templates
// (OneActuator runs it for <default> actuator entries too), so a class-level
// kp/kv (e.g. slider_crank's <position kp="30"/>) reaches inheriting actuators.
template <class T>
void Builder::ApplyActuatorShortcut(const T& e, mjsActuator* a) {
        if constexpr (std::is_same_v<T, Motor>) {
          mjs_setToMotor(a);
        } else if constexpr (std::is_same_v<T, Position> ||
                             std::is_same_v<T, IntVelocity>) {
          double kp = a->gainprm[0];
          if (e.kp.has_value()) kp = *e.kp;
          double kvd, *kv = nullptr;
          if (e.kv.has_value()) { kvd = *e.kv; kv = &kvd; }
          double drd, *dr = nullptr;
          if (e.dampratio.has_value()) { drd = *e.dampratio; dr = &drd; }
          double tcd, *tc = nullptr;
          if constexpr (requires { e.timeconst; }) {  // Position only; IntVelocity has none
            if (e.timeconst.has_value()) { tcd = *e.timeconst; tc = &tcd; }
          }
          double ir = a->inheritrange;
          if (e.inheritrange.has_value()) ir = *e.inheritrange;
          if constexpr (std::is_same_v<T, Position>)
            mjs_setToPosition(a, kp, kv, dr, tc, ir);
          else
            mjs_setToIntVelocity(a, kp, kv, dr, tc, ir);
        } else if constexpr (std::is_same_v<T, Velocity>) {
          double kv = a->gainprm[0];
          if (e.kv.has_value()) kv = *e.kv;
          mjs_setToVelocity(a, kv);
        } else if constexpr (std::is_same_v<T, Damper>) {
          double kv = 0;
          if (e.kv.has_value()) kv = *e.kv;
          mjs_setToDamper(a, kv);
        } else if constexpr (std::is_same_v<T, Cylinder>) {
          double timeconst = a->dynprm[0];
          if (e.timeconst.has_value()) timeconst = *e.timeconst;
          double bias = a->biasprm[0];
          if (e.bias.has_value()) bias = (*e.bias)[0];
          double area = a->gainprm[0];
          if (e.area.has_value()) area = *e.area;
          mjs_setToCylinder(a, timeconst, bias, area, /*diameter=*/-1);
        } else if constexpr (std::is_same_v<T, Muscle>) {
          double timeconst[2] = {-1, -1};
          if (e.timeconst.has_value())
            for (int i = 0; i < 2; ++i) timeconst[i] = (*e.timeconst)[i];
          double tausmooth = a->dynprm[2];
          if (e.tausmooth.has_value()) tausmooth = *e.tausmooth;
          double range[2] = {-1, -1};
          if (e.range.has_value())
            for (int i = 0; i < 2; ++i) range[i] = (*e.range)[i];
          double force = e.force.value_or(-1), scale = e.scale.value_or(-1);
          double lmin = e.lmin.value_or(-1), lmax = e.lmax.value_or(-1);
          double vmax = e.vmax.value_or(-1), fpmax = e.fpmax.value_or(-1);
          double fvmax = e.fvmax.value_or(-1);
          mjs_setToMuscle(a, timeconst, tausmooth, range, force, scale, lmin,
                          lmax, vmax, fpmax, fvmax);
        } else if constexpr (std::is_same_v<T, Adhesion>) {
          double gain = a->gainprm[0];
          if (e.gain.has_value()) gain = *e.gain;
          mjs_setToAdhesion(a, gain);
        } else if constexpr (std::is_same_v<T, DcMotor>) {
          bool inh = (a->gaintype == mjGAIN_DCMOTOR);
          double motorconst[2] = {inh ? a->gainprm[1] : 0, 0};
          double resistance = inh ? a->gainprm[0] : 0;
          double nominal[3] = {0, 0, 0};
          double saturation[3] = {0, 0, inh ? a->dynprm[1] : 0};
          double controller[6] = {
              inh ? a->gainprm[4] : 0, inh ? a->gainprm[5] : 0,
              inh ? a->gainprm[6] : 0, inh ? a->dynprm[7] : 0,
              inh ? a->dynprm[8] : 0, inh ? a->gainprm[7] : 0};
          double inductance[2] = {0, inh ? a->dynprm[0] : 0};
          double cogging[3] = {inh ? a->biasprm[0] : 0, inh ? a->biasprm[1] : 0,
                               inh ? a->biasprm[2] : 0};
          double thermal[6] = {inh ? a->dynprm[2] : 0, inh ? a->dynprm[3] : 0, 0,
                               inh ? a->gainprm[2] : 0, inh ? a->gainprm[3] : 0,
                               inh ? a->dynprm[4] : 0};
          double lugre[5] = {inh ? a->dynprm[5] : 0, inh ? a->dynprm[6] : 0,
                             inh ? a->biasprm[3] : 0, inh ? a->biasprm[4] : 0,
                             inh ? a->biasprm[5] : 0};
          int input_mode = inh ? static_cast<int>(a->gainprm[8]) : 0;
          auto ov2 = [](const ps::opt<ps::InlineVec<double, 2>>& s, double* d) {
            if (s.has_value()) for (std::size_t i = 0; i < s->size(); ++i) d[i] = (*s)[i];
          };
          auto ov3 = [](const ps::opt<ps::InlineVec<double, 3>>& s, double* d) {
            if (s.has_value()) for (std::size_t i = 0; i < s->size(); ++i) d[i] = (*s)[i];
          };
          auto ov6 = [](const ps::opt<ps::InlineVec<double, 6>>& s, double* d) {
            if (s.has_value()) for (std::size_t i = 0; i < s->size(); ++i) d[i] = (*s)[i];
          };
          auto ov5 = [](const ps::opt<ps::InlineVec<double, 5>>& s, double* d) {
            if (s.has_value()) for (std::size_t i = 0; i < s->size(); ++i) d[i] = (*s)[i];
          };
          ov2(e.motorconst, motorconst);
          if (e.resistance.has_value()) resistance = *e.resistance;
          ov3(e.nominal, nominal);
          ov3(e.saturation, saturation);
          ov2(e.inductance, inductance);
          ov3(e.cogging, cogging);
          ov6(e.controller, controller);
          ov6(e.thermal, thermal);
          ov5(e.lugre, lugre);
          if (e.input.has_value())
            input_mode = static_cast<int>(*e.input);  // enum order matches
          mjs_setToDCMotor(a, motorconst, resistance, nominal, saturation,
                           inductance, cogging, controller, thermal, lugre,
                           input_mode);
        }
}

// --- sensors ------------------------------------------------------------- //
void Builder::BuildSensors(const Model& m) {
  for (const auto& sen : m.sensors) {
    if (!sen) continue;
    for (const auto& any : sen->sensors) {
      std::visit([&](const auto& p) {
        if (!p) return;
        using T = std::decay_t<decltype(*p)>;
        const T& e = *p;
        mjsSensor* s = mjs_addSensor(spec_);
        ApplyMjs(e, s);

        auto site_obj = [&](mjtSensor type, const auto& ref) {
          s->type = type; s->objtype = mjOBJ_SITE;
          if (const char* n = RefCStr(ref)) mjs_setString(s->objname, n);
        };
        auto joint_obj = [&](mjtSensor type, const auto& ref) {
          s->type = type; s->objtype = mjOBJ_JOINT;
          if (const char* n = RefCStr(ref)) mjs_setString(s->objname, n);
        };
        auto tendon_obj = [&](mjtSensor type, const auto& ref) {
          s->type = type; s->objtype = mjOBJ_TENDON;
          if (const char* n = RefCStr(ref)) mjs_setString(s->objname, n);
        };
        auto act_obj = [&](mjtSensor type, const auto& ref) {
          s->type = type; s->objtype = mjOBJ_ACTUATOR;
          if (const char* n = RefCStr(ref)) mjs_setString(s->objname, n);
        };
        auto body_obj = [&](mjtSensor type, const auto& ref) {
          s->type = type; s->objtype = mjOBJ_BODY;
          if (const char* n = RefCStr(ref)) mjs_setString(s->objname, n);
        };
        // objtype/objname (+ optional reftype/refname) from string keywords;
        // ApplyMjs already set the objname/refname strings for these.
        auto frame_obj = [&](mjtSensor type) {
          s->type = type;
          if constexpr (requires { e.objtype; }) {
            if (e.objtype.has_value())
              s->objtype = static_cast<mjtObj>(mju_str2Type(e.objtype->c_str()));
          }
          if constexpr (requires { e.reftype; }) {
            if (e.reftype.has_value())
              s->reftype = static_cast<mjtObj>(mju_str2Type(e.reftype->c_str()));
          }
        };

        if constexpr (std::is_same_v<T, Touch>) site_obj(mjSENS_TOUCH, e.site);
        else if constexpr (std::is_same_v<T, Accelerometer>) site_obj(mjSENS_ACCELEROMETER, e.site);
        else if constexpr (std::is_same_v<T, Velocimeter>) site_obj(mjSENS_VELOCIMETER, e.site);
        else if constexpr (std::is_same_v<T, Gyro>) site_obj(mjSENS_GYRO, e.site);
        else if constexpr (std::is_same_v<T, Force>) site_obj(mjSENS_FORCE, e.site);
        else if constexpr (std::is_same_v<T, Torque>) site_obj(mjSENS_TORQUE, e.site);
        else if constexpr (std::is_same_v<T, Magnetometer>) site_obj(mjSENS_MAGNETOMETER, e.site);
        else if constexpr (std::is_same_v<T, Camprojection>) {
          site_obj(mjSENS_CAMPROJECTION, e.site);
          s->reftype = mjOBJ_CAMERA;
          if (const char* n = RefCStr(e.camera)) mjs_setString(s->refname, n);
        }
        else if constexpr (std::is_same_v<T, Jointpos>) joint_obj(mjSENS_JOINTPOS, e.joint);
        else if constexpr (std::is_same_v<T, Jointvel>) joint_obj(mjSENS_JOINTVEL, e.joint);
        else if constexpr (std::is_same_v<T, Jointactuatorfrc>) joint_obj(mjSENS_JOINTACTFRC, e.joint);
        else if constexpr (std::is_same_v<T, Ballquat>) joint_obj(mjSENS_BALLQUAT, e.joint);
        else if constexpr (std::is_same_v<T, Ballangvel>) joint_obj(mjSENS_BALLANGVEL, e.joint);
        else if constexpr (std::is_same_v<T, Jointlimitpos>) joint_obj(mjSENS_JOINTLIMITPOS, e.joint);
        else if constexpr (std::is_same_v<T, Jointlimitvel>) joint_obj(mjSENS_JOINTLIMITVEL, e.joint);
        else if constexpr (std::is_same_v<T, Jointlimitfrc>) joint_obj(mjSENS_JOINTLIMITFRC, e.joint);
        else if constexpr (std::is_same_v<T, Tendonpos>) tendon_obj(mjSENS_TENDONPOS, e.tendon);
        else if constexpr (std::is_same_v<T, Tendonvel>) tendon_obj(mjSENS_TENDONVEL, e.tendon);
        else if constexpr (std::is_same_v<T, Tendonactuatorfrc>) tendon_obj(mjSENS_TENDONACTFRC, e.tendon);
        else if constexpr (std::is_same_v<T, Tendonlimitpos>) tendon_obj(mjSENS_TENDONLIMITPOS, e.tendon);
        else if constexpr (std::is_same_v<T, Tendonlimitvel>) tendon_obj(mjSENS_TENDONLIMITVEL, e.tendon);
        else if constexpr (std::is_same_v<T, Tendonlimitfrc>) tendon_obj(mjSENS_TENDONLIMITFRC, e.tendon);
        else if constexpr (std::is_same_v<T, Actuatorpos>) act_obj(mjSENS_ACTUATORPOS, e.actuator);
        else if constexpr (std::is_same_v<T, Actuatorvel>) act_obj(mjSENS_ACTUATORVEL, e.actuator);
        else if constexpr (std::is_same_v<T, Actuatorfrc>) act_obj(mjSENS_ACTUATORFRC, e.actuator);
        else if constexpr (std::is_same_v<T, Subtreecom>) body_obj(mjSENS_SUBTREECOM, e.body);
        else if constexpr (std::is_same_v<T, Subtreelinvel>) body_obj(mjSENS_SUBTREELINVEL, e.body);
        else if constexpr (std::is_same_v<T, Subtreeangmom>) body_obj(mjSENS_SUBTREEANGMOM, e.body);
        else if constexpr (std::is_same_v<T, Framepos>) frame_obj(mjSENS_FRAMEPOS);
        else if constexpr (std::is_same_v<T, Framequat>) frame_obj(mjSENS_FRAMEQUAT);
        else if constexpr (std::is_same_v<T, Framexaxis>) frame_obj(mjSENS_FRAMEXAXIS);
        else if constexpr (std::is_same_v<T, Frameyaxis>) frame_obj(mjSENS_FRAMEYAXIS);
        else if constexpr (std::is_same_v<T, Framezaxis>) frame_obj(mjSENS_FRAMEZAXIS);
        else if constexpr (std::is_same_v<T, Framelinvel>) frame_obj(mjSENS_FRAMELINVEL);
        else if constexpr (std::is_same_v<T, Frameangvel>) frame_obj(mjSENS_FRAMEANGVEL);
        else if constexpr (std::is_same_v<T, Framelinacc>) frame_obj(mjSENS_FRAMELINACC);
        else if constexpr (std::is_same_v<T, Frameangacc>) frame_obj(mjSENS_FRAMEANGACC);
        else if constexpr (std::is_same_v<T, EPotential>) s->type = mjSENS_E_POTENTIAL;
        else if constexpr (std::is_same_v<T, EKinetic>) s->type = mjSENS_E_KINETIC;
        else if constexpr (std::is_same_v<T, Clock>) s->type = mjSENS_CLOCK;
        else if constexpr (std::is_same_v<T, Rangefinder>) {
          // Site OR camera is the sensorized object (never a ref). data keyword
          // set -> intprm[0] bitmask; default 1<<mjRAYDATA_DIST. dim/datatype/
          // needstage are derived by the compiler (mjs_sensorDim).
          s->type = mjSENS_RANGEFINDER;
          if (const char* n = RefCStr(e.site)) {
            s->objtype = mjOBJ_SITE; mjs_setString(s->objname, n);
          } else if (const char* n = RefCStr(e.camera)) {
            s->objtype = mjOBJ_CAMERA; mjs_setString(s->objname, n);
          }
          int spec = 1 << mjRAYDATA_DIST;
          if (e.data.has_value() && !e.data->empty()) {
            spec = 0;
            for (RayData d : *e.data) spec |= (1 << RayDataToMjt(d));
          }
          s->intprm[0] = spec;
        }
        else if constexpr (std::is_same_v<T, SensorContact>) {
          // First present of site/body1/subtree1/geom1 is the object; second of
          // body2/subtree2/geom2 is the ref (subtree maps to XBODY). intprm:
          // [0]=data bitmask (default 1<<mjCONDATA_FOUND), [1]=reduce, [2]=num.
          s->type = mjSENS_CONTACT;
          if (const char* n = RefCStr(e.site)) {
            s->objtype = mjOBJ_SITE; mjs_setString(s->objname, n);
          } else if (const char* n = RefCStr(e.body1)) {
            s->objtype = mjOBJ_BODY; mjs_setString(s->objname, n);
          } else if (const char* n = RefCStr(e.subtree1)) {
            s->objtype = mjOBJ_XBODY; mjs_setString(s->objname, n);
          } else if (const char* n = RefCStr(e.geom1)) {
            s->objtype = mjOBJ_GEOM; mjs_setString(s->objname, n);
          }
          if (const char* n = RefCStr(e.body2)) {
            s->reftype = mjOBJ_BODY; mjs_setString(s->refname, n);
          } else if (const char* n = RefCStr(e.subtree2)) {
            s->reftype = mjOBJ_XBODY; mjs_setString(s->refname, n);
          } else if (const char* n = RefCStr(e.geom2)) {
            s->reftype = mjOBJ_GEOM; mjs_setString(s->refname, n);
          }
          int spec = 1 << mjCONDATA_FOUND;
          if (e.data.has_value() && !e.data->empty()) {
            spec = 0;
            for (ContactData d : *e.data) spec |= (1 << ContactDataToMjt(d));
          }
          s->intprm[0] = spec;
          s->intprm[1] = e.reduce.has_value() ? ContactReduceToInt(*e.reduce) : 0;
          s->intprm[2] = e.num.value_or(1);
        }
        else if constexpr (std::is_same_v<T, Tactile>) {
          // Inverted: the MESH is the object, the GEOM is the ref.
          s->type = mjSENS_TACTILE;
          s->objtype = mjOBJ_MESH;
          if (const char* n = RefCStr(e.mesh)) mjs_setString(s->objname, n);
          s->reftype = mjOBJ_GEOM;
          if (const char* n = RefCStr(e.geom)) mjs_setString(s->refname, n);
        }
        else if constexpr (std::is_same_v<T, SensorUser>) {
          // datatype/needstage/dim/objname are exact-mapped by ApplyMjs and NOT
          // recomputed by the compiler for user sensors; only objtype (a string
          // keyword) needs the mju_str2Type conversion the emitter waives.
          s->type = mjSENS_USER;
          if (e.objtype.has_value())
            s->objtype = static_cast<mjtObj>(mju_str2Type(e.objtype->c_str()));
        }
        else if constexpr (std::is_same_v<T, SensorPlugin>) {
          // Plugin sensor: type + plugin sub-struct + objtype/reftype keywords
          // (objname/refname are exact-mapped by ApplyMjs).
          s->type = mjSENS_PLUGIN;
          SetPluginFields(e.plugin.has_value() ? e.plugin->c_str() : nullptr,
                          (e.instance.has_value() && !e.instance->name.empty())
                              ? e.instance->name.c_str() : nullptr,
                          e.config, &s->plugin);
          if (e.objtype.has_value())
            s->objtype = static_cast<mjtObj>(mju_str2Type(e.objtype->c_str()));
          if (e.reftype.has_value())
            s->reftype = static_cast<mjtObj>(mju_str2Type(e.reftype->c_str()));
        }
        else if constexpr (std::is_same_v<T, Insidesite>) {
          // The sensorized object is (objtype,objname); the site is the ref.
          s->type = mjSENS_INSIDESITE;
          s->reftype = mjOBJ_SITE;
          if (const char* n = RefCStr(e.site)) mjs_setString(s->refname, n);
          if (e.objtype.has_value())
            s->objtype = static_cast<mjtObj>(mju_str2Type(e.objtype->c_str()));
          // objname exact-mapped by ApplyMjs.
        }
        else if constexpr (std::is_same_v<T, Distance> ||
                           std::is_same_v<T, Normal> ||
                           std::is_same_v<T, Fromto>) {
          // Geometric-distance family: exactly one of (geom1,body1) is the
          // object, one of (geom2,body2) the ref (validated on both paths).
          if constexpr (std::is_same_v<T, Distance>) s->type = mjSENS_GEOMDIST;
          else if constexpr (std::is_same_v<T, Normal>) s->type = mjSENS_GEOMNORMAL;
          else s->type = mjSENS_GEOMFROMTO;
          if (const char* n = RefCStr(e.body1)) {
            s->objtype = mjOBJ_BODY; mjs_setString(s->objname, n);
          } else if (const char* n = RefCStr(e.geom1)) {
            s->objtype = mjOBJ_GEOM; mjs_setString(s->objname, n);
          }
          if (const char* n = RefCStr(e.body2)) {
            s->reftype = mjOBJ_BODY; mjs_setString(s->refname, n);
          } else if (const char* n = RefCStr(e.geom2)) {
            s->reftype = mjOBJ_GEOM; mjs_setString(s->refname, n);
          }
        }
        AutoName(s->element, e.serial);
      }, any.node);
    }
  }
}

// --- custom -------------------------------------------------------------- //
void Builder::BuildCustom(const Model& m) {
  for (const auto& c : m.customs) {
    if (!c) continue;
    for (const auto& n : c->numerics) {
      if (!n) continue;
      mjsNumeric* mn = mjs_addNumeric(spec_);
      ApplyMjs(*n, mn);
      // ProtoSpec materializes numeric data to its authored size (zero-padded /
      // truncated), so the declared field size equals the data length.
      if (n->data.has_value()) mn->size = static_cast<int>(n->data->size());
      AutoName(mn->element, n->serial);
    }
    for (const auto& t : c->texts) {
      if (!t) continue;
      mjsText* mt = mjs_addText(spec_);
      ApplyMjs(*t, mt);
      AutoName(mt->element, t->serial);
    }
    for (const auto& tp : c->tuples) {
      if (!tp) continue;
      mjsTuple* mtp = mjs_addTuple(spec_);
      ApplyMjs(*tp, mtp);
      std::vector<int> objtype;
      std::string objname;
      std::vector<double> objprm;
      for (const auto& el : tp->tupleElements) {
        if (!el) continue;
        objtype.push_back(el->objtype.has_value()
                              ? mju_str2Type(el->objtype->c_str())
                              : mjOBJ_UNKNOWN);
        objname += " ";
        if (el->objname.has_value()) objname += *el->objname;
        objprm.push_back(el->prm.value_or(0.0));
      }
      mjs_setInt(mtp->objtype, objtype.data(), static_cast<int>(objtype.size()));
      mjs_setStringVec(mtp->objname, objname.c_str());
      mjs_setDouble(mtp->objprm, objprm.data(), static_cast<int>(objprm.size()));
      AutoName(mtp->element, tp->serial);
    }
  }
}

// --- keyframes ----------------------------------------------------------- //
void Builder::BuildKeyframes(const Model& m) {
  for (const auto& kf : m.keyframes) {
    if (!kf) continue;
    for (const auto& k : kf->keys) {
      if (!k) continue;
      mjsKey* mk = mjs_addKey(spec_);
      ApplyMjs(*k, mk);
      AutoName(mk->element, k->serial);
    }
  }
}

// --- deformables (flex / skin) ------------------------------------------- //
void Builder::BuildDeformables(const Model& m) {
  for (const auto& d : m.deformables) {
    if (!d) continue;
    for (const auto& fx : d->flexs) {
      if (!fx) continue;
      mjsFlex* f = mjs_addFlex(spec_);
      ApplyMjs(*fx, f);  // flat fields + vertbody/nodebody/vert/elem/texcoord
      AutoName(f->element, fx->serial);
      // dof -> order (mirror OneFlex: quadratic=2, trilinear=1, else 0; the
      // reader always writes order, so set it unconditionally here too).
      int order = 0;
      if (fx->dof.has_value()) {
        if (*fx->dof == FlexDof::quadratic) order = 2;
        else if (*fx->dof == FlexDof::trilinear) order = 1;
      }
      f->order = order;
      // <contact> sub-block folds into the flat contact fields.
      for (const auto& c : fx->flexContacts) {
        if (!c) continue;
        if (c->contype.has_value()) f->contype = *c->contype;
        if (c->conaffinity.has_value()) f->conaffinity = *c->conaffinity;
        if (c->condim.has_value()) f->condim = *c->condim;
        if (c->priority.has_value()) f->priority = *c->priority;
        if (c->friction.has_value())
          for (std::size_t i = 0; i < c->friction->size(); ++i)
            f->friction[i] = (*c->friction)[i];
        if (c->solmix.has_value()) f->solmix = *c->solmix;
        if (c->solref.has_value())
          for (std::size_t i = 0; i < c->solref->size(); ++i)
            f->solref[i] = (*c->solref)[i];
        if (c->solimp.has_value())
          for (std::size_t i = 0; i < c->solimp->size(); ++i)
            f->solimp[i] = (*c->solimp)[i];
        if (c->margin.has_value()) f->margin = *c->margin;
        if (c->gap.has_value()) f->gap = *c->gap;
        if (c->internal.has_value()) f->internal = *c->internal ? 1 : 0;
        if (c->selfcollide.has_value())
          f->selfcollide = FlexSelfToInt(*c->selfcollide);
        if (c->passive.has_value()) f->passive = *c->passive ? 1 : 0;
        if (c->activelayers.has_value()) f->activelayers = *c->activelayers;
      }
      // <edge> sub-block.
      for (const auto& ed : fx->flexEdges) {
        if (!ed) continue;
        if (ed->stiffness.has_value()) f->edgestiffness = *ed->stiffness;
        if (ed->damping.has_value()) f->edgedamping = *ed->damping;
      }
      // <elasticity> sub-block (its `damping` is the flex damping field, not
      // edgedamping).
      for (const auto& el : fx->flexElasticitys) {
        if (!el) continue;
        if (el->young.has_value()) f->young = *el->young;
        if (el->poisson.has_value()) f->poisson = *el->poisson;
        if (el->damping.has_value()) f->damping = *el->damping;
        if (el->thickness.has_value()) f->thickness = *el->thickness;
        if (el->elastic2d.has_value()) f->elastic2d = Elastic2DToInt(*el->elastic2d);
      }
    }
    for (const auto& sk : d->skins) {
      if (sk) BuildSkin(*sk);
    }
  }
}

// A <skin> (valid under both <asset> and <deformable>). ApplyMjs sets the flat
// fields; <bone> rows fold into the parallel arrays: bodyname/vertid/vertweight
// append one entry per bone; bindpos/bindquat are single flattened float arrays
// set once (mirror OneSkin).
void Builder::BuildSkin(const Skin& sk) {
  mjsSkin* s = mjs_addSkin(spec_);
  ApplyMjs(sk, s);  // file/material/rgba/inflate/vert/texcoord/face/group
  AutoName(s->element, sk.serial);
  std::vector<float> bindpos, bindquat;
  for (const auto& b : sk.bones) {
    if (!b) continue;
    mjs_appendString(s->bodyname,
                     b->body.has_value() ? b->body->name.c_str() : "");
    for (int i = 0; i < 3; ++i)
      bindpos.push_back(b->bindpos.has_value()
                            ? static_cast<float>((*b->bindpos)[i]) : 0.0f);
    for (int i = 0; i < 4; ++i)
      bindquat.push_back(b->bindquat.has_value()
                             ? static_cast<float>((*b->bindquat)[i]) : 0.0f);
    std::vector<int> vid;
    if (b->vertid.has_value()) vid.assign(b->vertid->begin(), b->vertid->end());
    mjs_appendIntVec(s->vertid, vid.data(), static_cast<int>(vid.size()));
    std::vector<float> vw;
    if (b->vertweight.has_value())
      vw.assign(b->vertweight->begin(), b->vertweight->end());
    mjs_appendFloatVec(s->vertweight, vw.data(), static_cast<int>(vw.size()));
  }
  if (!bindpos.empty())
    mjs_setFloat(s->bindpos, bindpos.data(), static_cast<int>(bindpos.size()));
  if (!bindquat.empty())
    mjs_setFloat(s->bindquat, bindquat.data(), static_cast<int>(bindquat.size()));
}

void Builder::Build(const Model& m, const CompileOptions& opts) {
  base_dir_ = opts.base_dir;
  model_ = &m;
  BuildCompiler(m, opts);
  for (const auto& o : m.options) if (o) BuildOption(*o);
  for (const auto& s : m.sizes) if (s) BuildSize(*s);
  for (const auto& s : m.statistics) if (s) BuildStatistic(*s);
  for (const auto& v : m.visuals) if (v) BuildVisual(*v);
  // Section order mirrors mjXReader::Parse EXACTLY (xml_native_reader.cc):
  // every standalone section is parsed BEFORE <worldbody>, which comes last.
  // This is load-bearing for element-array ordering: worldbody macros generate
  // section elements (flexcomp/composite append <equality> entries; replicate's
  // mjs_attach rewrites references in already-built <tendon> wraps), and stock
  // emits those AFTER the user's own section entries because worldbody runs
  // last. Building worldbody first (as before) inverted that order.
  BuildDefaults(m);
  BuildExtensions(m);
  BuildCustom(m);
  BuildAssets(m);
  BuildContact(m);
  BuildDeformables(m);
  BuildEquality(m);
  BuildTendons(m);
  BuildActuators(m);
  BuildSensors(m);
  BuildKeyframes(m);
  BuildModelAssets(m);
  BuildWorldbody(m);
}

// --- fallback scan ------------------------------------------------------- //
void AddReason(std::vector<FallbackReason>& out, const char* feature) {
  for (auto& r : out) {
    if (r.feature == feature) { r.count++; return; }
  }
  FallbackReason r;
  r.feature = feature;
  r.count = 1;
  out.push_back(std::move(r));
}

// Walk a body subtree (recursing through nested Body/Frame/Replicate) flagging
// any Flexcomp the mjs_makeFlex API cannot reproduce: (a) <pin> subelements --
// mjs_makeFlex takes no pinid/pinrange/pingrid arguments, so the generated flex
// keeps every vertex a free dof (a SILENT wrong model); (b) type="direct" (and
// mesh/gmsh without a file) -- the inline point/element topology also has no
// makeFlex surface ("Point and element required" at build time).
void ScanFlexcompSubtree(const std::vector<BodyChildAny>& subtree,
                         std::vector<FallbackReason>& out) {
  for (const auto& item : subtree) {
    std::visit([&](const auto& p) {
      if (!p) return;
      using T = std::decay_t<decltype(*p)>;
      const T& e = *p;
      if constexpr (std::is_same_v<T, Flexcomp>) {
        if (!e.flexcompPins.empty())
          AddReason(out, "mjs.flexcomp_pin");
        const bool has_file = e.file.has_value() && !e.file->empty();
        if (e.type.has_value() &&
            (*e.type == FlexcompType::direct ||
             ((*e.type == FlexcompType::mesh || *e.type == FlexcompType::gmsh) &&
              !has_file)))
          AddReason(out, "mjs.flexcomp_direct");
        // A material with no inline texcoord (and no file supplying UVs) makes
        // mjCFlexcomp::Make auto-generate texcoord (needtex), which needs the
        // material on the flex def BEFORE Make -- mjs_makeFlex has no such hook.
        const bool has_texcoord = e.texcoord.has_value() && !e.texcoord->empty();
        if (e.material.has_value() && !e.material->name.empty() &&
            !has_texcoord && !has_file)
          AddReason(out, "mjs.flexcomp_material_texcoord");
      } else if constexpr (std::is_same_v<T, Body> ||
                           std::is_same_v<T, Frame> ||
                           std::is_same_v<T, Replicate>) {
        ScanFlexcompSubtree(e.subtree, out);
      }
    }, item.node);
  }
}

}  // namespace

// The scan holds guards for content that either is invalid on BOTH compile paths
// (always-error, surfaced as a clean model error) or that the mjs_makeFlex/mjSpec
// API cannot reproduce (fallbackable: Auto routes those to XmlPath, forced
// MjsPath errors loudly). Every other former fallback family -- replicate/
// composite macros, builtin meshes, URDF/MJB child models, plugins, deformables,
// exotic sensors/textures -- is now built directly with full parity (ps_path_diff
// is the net), so none appear here.
//   * coordinate="global" [always-error]: removed from MuJoCo (2.3.3+);
//     mjsCompiler has no global field, so the mjs build would silently
//     reinterpret it as local. Deprecated composite types are NOT guarded here --
//     they surface their own reader error when GraftComposite parses the fragment.
//   * flexcomp <pin> / type=direct [fallbackable]: no mjs_makeFlex surface (see
//     ScanFlexcompSubtree). XmlPath reproduces them exactly.
std::vector<FallbackReason> MjsFallbackScan(const Model& m) {
  std::vector<FallbackReason> out;
  for (const auto& c : m.compilers) {
    if (c && c->coordinate.has_value() && *c->coordinate == Coordinate::global)
      AddReason(out, "mjs.global_coordinates");
  }
  for (const auto& wb : m.worldbody)
    if (wb) ScanFlexcompSubtree(wb->subtree, out);
  return out;
}

mjSpec* BuildSpec(const Model& model, const CompileOptions& opts,
                  const io::AutoNames& auto_names, std::string& err) {
  mjSpec* spec = mj_makeSpec();
  if (!spec) { err = "mj_makeSpec failed"; return nullptr; }
  Builder b(spec, auto_names);
  b.Build(model, opts);
  if (!b.error().empty()) {  // macro expansion failed (e.g. deprecated composite)
    err = b.error();
    mj_deleteSpec(spec);
    return nullptr;
  }
  return spec;
}

}  // namespace ps::mjcf::compile
