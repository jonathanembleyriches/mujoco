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
#include <string>
#include <variant>
#include <vector>

#include <mujoco/mujoco.h>

#include "mjs_binding.h"
#include "mjs_convert.h"

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

// --- the builder --------------------------------------------------------- //
class Builder {
 public:
  Builder(mjSpec* spec, const io::AutoNames& names) : spec_(spec), names_(names) {}

  void Build(const Model& m, const CompileOptions& opts);

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

  template <class A>
  void ApplyActuatorTail(const A& e, mjsActuator* out);

  mjSpec* spec_;
  const io::AutoNames& names_;
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
  for (const auto& x : d.geom) if (x) ApplyMjs(*x, into->geom);
  for (const auto& x : d.site) if (x) ApplyMjs(*x, into->site);
  for (const auto& x : d.camera) if (x) ApplyMjs(*x, into->camera);
  for (const auto& x : d.light) if (x) ApplyMjs(*x, into->light);
  for (const auto& x : d.material) if (x) ApplyMjs(*x, into->material);
  for (const auto& x : d.mesh) if (x) ApplyMjs(*x, into->mesh);
  for (const auto& x : d.pair) if (x) ApplyMjs(*x, into->pair);
  for (const auto& x : d.equality) if (x) ApplyMjs(*x, into->equality);
  for (const auto& x : d.tendon) if (x) ApplyMjs(*x, into->tendon);
  // Actuator-family templates all fold onto the single mjsDefault::actuator.
  for (const auto& x : d.general) if (x) ApplyActuatorTail(*x, into->actuator);
  for (const auto& x : d.motor) if (x) ApplyMjs(*x, into->actuator);
  for (const auto& x : d.position) if (x) ApplyMjs(*x, into->actuator);
  for (const auto& x : d.velocity) if (x) ApplyMjs(*x, into->actuator);
  for (const auto& x : d.intvelocity) if (x) ApplyMjs(*x, into->actuator);
  for (const auto& x : d.damper) if (x) ApplyMjs(*x, into->actuator);
  for (const auto& x : d.cylinder) if (x) ApplyMjs(*x, into->actuator);
  for (const auto& x : d.muscle) if (x) ApplyMjs(*x, into->actuator);
  for (const auto& x : d.adhesion) if (x) ApplyMjs(*x, into->actuator);
  for (const auto& x : d.dcmotor) if (x) ApplyMjs(*x, into->actuator);

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

// --- assets -------------------------------------------------------------- //
void Builder::BuildAssets(const Model& m) {
  for (const auto& a : m.assets) {
    if (!a) continue;
    for (const auto& tex : a->textures) {
      if (!tex) continue;
      mjsTexture* t = mjs_addTexture(spec_);
      ApplyMjs(*tex, t);
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
    }
    for (const auto& hf : a->hfields) {
      if (!hf) continue;
      mjsHField* h = mjs_addHField(spec_);
      // Hfield.elevation aliases userdata (float); ApplyMjs handles name/file/
      // nrow/ncol/size and the elevation buffer.
      ApplyMjs(*hf, h);
    }
  }
}

// --- worldbody ----------------------------------------------------------- //
void Builder::BuildWorldbody(const Model& m) {
  mjsBody* world = mjs_findBody(spec_, "world");
  const mjsDefault* root = mjs_getSpecDefault(spec_);
  for (const auto& wb : m.worldbody) {
    if (wb) BuildSubtree(world, wb->subtree, nullptr, root);
  }
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
      }
      // Composite/Flexcomp/Replicate/Attach/PluginRef are excluded by
      // MjsFallbackScan, so they never reach here.
    }, item.node);
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

        // gain/bias shortcut: read the current mjs field as the class default,
        // override with the authored value, then invoke mjs_setTo* -- exactly
        // as MuJoCo's XML reader does (xml_native_reader.cc OneActuator).
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
        // ActuatorGeneral: gaintype/biastype/dyntype/gainprm/biasprm/dynprm are
        // written directly by ApplyMjs -- no shortcut needed.
        AutoName(a->element, e.serial);
      }, any.node);
    }
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

void Builder::Build(const Model& m, const CompileOptions& opts) {
  BuildCompiler(m, opts);
  for (const auto& o : m.options) if (o) BuildOption(*o);
  for (const auto& s : m.sizes) if (s) BuildSize(*s);
  for (const auto& s : m.statistics) if (s) BuildStatistic(*s);
  for (const auto& v : m.visuals) if (v) BuildVisual(*v);
  BuildDefaults(m);
  BuildAssets(m);
  BuildWorldbody(m);
  BuildContact(m);
  BuildEquality(m);
  BuildTendons(m);
  BuildActuators(m);
  BuildSensors(m);
  BuildCustom(m);
  BuildKeyframes(m);
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

void ScanSubtree(const std::vector<BodyChildAny>& subtree,
                 std::vector<FallbackReason>& out);

void ScanBodyChild(const BodyChildAny& item, std::vector<FallbackReason>& out) {
  std::visit([&](const auto& p) {
    if (!p) return;
    using T = std::decay_t<decltype(*p)>;
    const T& e = *p;
    if constexpr (std::is_same_v<T, Composite>) AddReason(out, "mjs.composite");
    else if constexpr (std::is_same_v<T, Flexcomp>) AddReason(out, "mjs.flexcomp");
    else if constexpr (std::is_same_v<T, Replicate>) AddReason(out, "mjs.replicate");
    else if constexpr (std::is_same_v<T, Attach>) AddReason(out, "mjs.attach");
    else if constexpr (std::is_same_v<T, PluginRef>) AddReason(out, "mjs.plugin");
    else if constexpr (std::is_same_v<T, Geom>) {
      if (!e.plugin.empty()) AddReason(out, "mjs.plugin");
    }
    else if constexpr (std::is_same_v<T, Body>) ScanSubtree(e.subtree, out);
    else if constexpr (std::is_same_v<T, Frame>) ScanSubtree(e.subtree, out);
  }, item.node);
}

void ScanSubtree(const std::vector<BodyChildAny>& subtree,
                 std::vector<FallbackReason>& out) {
  for (const auto& item : subtree) ScanBodyChild(item, out);
}

}  // namespace

std::vector<FallbackReason> MjsFallbackScan(const Model& m) {
  std::vector<FallbackReason> out;
  for (const auto& c : m.compilers) {
    if (c && c->coordinate.has_value() && *c->coordinate == Coordinate::global)
      AddReason(out, "mjs.global_coordinates");
  }
  for (const auto& e : m.extensions) {
    if (e && !e->pluginDefs.empty()) AddReason(out, "mjs.plugin");
  }
  for (const auto& a : m.assets) {
    if (!a) continue;
    if (!a->skins.empty()) AddReason(out, "mjs.skin");
    if (!a->modelAssets.empty()) AddReason(out, "mjs.model_asset");
    for (const auto& mesh : a->meshs)
      if (mesh && (mesh->builtin.has_value() || !mesh->plugin.empty()))
        AddReason(out, "mjs.mesh_builtin_or_plugin");
    for (const auto& tex : a->textures)
      if (tex && (tex->fileright.has_value() || tex->fileleft.has_value() ||
                  tex->fileup.has_value() || tex->filedown.has_value() ||
                  tex->filefront.has_value() || tex->fileback.has_value() ||
                  tex->gridlayout.has_value()))
        AddReason(out, "mjs.texture_cube_faces");
  }
  for (const auto& d : m.deformables) {
    if (d && (!d->flexs.empty() || !d->skins.empty()))
      AddReason(out, "mjs.deformable");
  }
  for (const auto& wb : m.worldbody)
    if (wb) ScanSubtree(wb->subtree, out);
  for (const auto& act : m.actuators) {
    if (!act) continue;
    for (const auto& any : act->actuators)
      if (any.kind() == ActuatorAny::Kind::ActuatorPlugin)
        AddReason(out, "mjs.plugin");
  }
  for (const auto& sen : m.sensors) {
    if (!sen) continue;
    for (const auto& any : sen->sensors) {
      switch (any.kind()) {
        case SensorAny::Kind::Rangefinder:
        case SensorAny::Kind::SensorContact:
        case SensorAny::Kind::Tactile:
        case SensorAny::Kind::SensorUser:
        case SensorAny::Kind::SensorPlugin:
          AddReason(out, "mjs.sensor_unsupported");
          break;
        default: break;
      }
    }
  }
  return out;
}

mjSpec* BuildSpec(const Model& model, const CompileOptions& opts,
                  const io::AutoNames& auto_names, std::string& err) {
  mjSpec* spec = mj_makeSpec();
  if (!spec) { err = "mj_makeSpec failed"; return nullptr; }
  Builder b(spec, auto_names);
  b.Build(model, opts);
  return spec;
}

}  // namespace ps::mjcf::compile
