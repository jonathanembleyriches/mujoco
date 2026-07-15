// Native compiler driver (impl-plan T0.3). Inside the MuJoCo quarantine zone
// (compile/ + bridge/ jointly): this TU includes mujoco.h.
//
// Today the driver is scaffolding: the CDR-2 gate walks the model, finds that
// no feature is natively supported (native_supported.h admits nothing), and
// NativeCompile returns null with fallback reasons. The stage pipeline seam
// (S1 Collect -> ... -> S13 smoke) is marked but empty; NC1 fills it and grows
// native_supported.h, at which point the same gate starts admitting models.
//
// --- Open Q2 (allocator spike, T0.5): RESOLVED -> LIFT mj_makeModel. -------- //
// The design left open whether to allocate the mjModel by lifting MuJoCo's
// mj_makeModel or by writing our own mjxmacro-driven allocator. Finding from the
// spike (cpp/compile/lifted/make_model.*, round-trip test in test_native.cc):
//
//   * mj_makeModel is NOT public. It is declared without MJAPI in the engine-
//     internal header src/engine/engine_io.h (line 50) and is absent from
//     include/mujoco/mujoco.h, so it is not exported from mujoco.dll and cannot
//     be linked. (The design's premise "mj_makeModel IS public" is incorrect for
//     this pin, 3.10.0 / mjVERSION_HEADER 3010000.) The only exported model
//     constructors -- mj_copyModel and mj_loadModel -- both need an existing
//     model, so neither builds an mjModel from raw sizes without the compiler.
//   * Therefore the allocator must be LIFTED, not linked. The lift is small and
//     tracks upstream structurally because the buffer layout is driven entirely
//     by the PUBLIC include/mujoco/mjxmacro.h X-macros (MJMODEL_SIZES /
//     MJMODEL_POINTERS): mj_makeModel, its buffer-offset helper, and the pointer
//     setup (mj_setPtrModel) are ~120 lines of xmacro plumbing over mju_malloc /
//     mju_free (both MJAPI). New mjModel fields are picked up automatically when
//     mjxmacro.h changes -- exactly the structural robustness the design wanted.
//   * The spike proves layout compatibility the decisive way: a model allocated
//     by our lifted ps::MakeModel round-trips through the ENGINE's own
//     mj_copyModel and mj_deleteModel (which walk the same layout) without
//     corruption. If our offsets disagreed with mj_setPtrModel's, mj_copyModel's
//     buffer copy or mj_deleteModel's frees would fault; they do not.
//
// Decision: lift (make_model.cc, registry entry make_model). "Own allocator" is
// rejected -- it would duplicate the exact same xmacro walk with no upside and a
// second thing to keep layout-compatible across bumps.

#include "native.h"

#include <algorithm>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <mujoco/mujoco.h>

#include "attach.h"    // ps::sdk::detail::RefPrefixer (cross-model NameSpace)
#include "build.h"
#include "classes.h"   // ps::sdk defaults resolution (partial-array detector)
#include "context.h"
#include "mjcf.h"      // io::ParseMjcfFile (recursive child-model parse)
#include "native_supported.h"
#include "reflect.h"
#include "visit.h"

namespace ps::mjcf::compile {
namespace {

// MuJoCo shares one actuator default per class across every spelling; ProtoSpec
// keeps a per-spelling partial and ps::sdk::Effective merges only the matching
// one. So when an actuator's governing class chain also carries an actuator
// partial of a DIFFERENT spelling (e.g. a root <general ctrllimited="true">
// inherited by a <velocity>), MuJoCo's shared default applies fields the native
// per-spelling merge misses. Detect a foreign actuator partial and fall back.
template <class T>
bool ForeignActuatorDefault(const ps::sdk::detail::DefaultIndex& idx,
                            const ps::sdk::ParentMap& pm, const T& e) {
  std::string cls = ps::sdk::detail::ResolveClassName(pm, ps::sdk::detail::OwnClass(e), &e);
  for (const ps::mjcf::Default* d = idx.ByNameOrRoot(cls); d; d = idx.ParentOf(d)) {
    auto has = [&](const auto& vec) { return !vec.empty() && vec.front(); };
    // every actuator family except the element's own T
    if (!std::is_same_v<T, ActuatorGeneral> && has(d->general)) return true;
    if (!std::is_same_v<T, Motor> && has(d->motor)) return true;
    if (!std::is_same_v<T, Position> && has(d->position)) return true;
    if (!std::is_same_v<T, Velocity> && has(d->velocity)) return true;
    if (!std::is_same_v<T, IntVelocity> && has(d->intvelocity)) return true;
    if (!std::is_same_v<T, Damper> && has(d->damper)) return true;
    if (!std::is_same_v<T, Cylinder> && has(d->cylinder)) return true;
    if (!std::is_same_v<T, Muscle> && has(d->muscle)) return true;
    if (!std::is_same_v<T, Adhesion> && has(d->adhesion)) return true;
    if (has(d->dcmotor)) return true;  // dcmotor is always gated anyway
  }
  return false;
}

// The feature key a used element contributes to the gate. Today this is the
// element's MJCF tag (falling back to its IR name for tagless families) -- a
// stable, human-legible key that can later be refined below family granularity
// (e.g. "geom.mesh_ref") without changing the FallbackReason ABI.
std::string FeatureKey(ElementType et) {
  const reflect::ElementDescriptor& d = reflect::Describe(et);
  std::string_view key = d.xml.empty() ? d.name : d.xml;
  return std::string(key);
}

// One used feature: how many times it appears and where it first appears.
struct FeatureUse {
  int count = 0;
  ps::SourceLoc first;
};

// Walks the tree once, recording every element family it uses. Mirrors the
// bridge Collector's traversal: the Model root is entered via its section child
// lists, Default subtrees are pruned (their contents are class templates, not
// compiled elements), and every other element contributes its family key.
class FeatureCollector {
 public:
  explicit FeatureCollector(std::unordered_map<ElementType, FeatureUse>& used)
      : used_(used) {}

  void operator()(const Model& m) {
    Recurse rec{this};
    Visit(m, rec);
  }

  template <class E>
  void operator()(const E& e) {
    if constexpr (std::is_same_v<E, Default>) {
      (void)e;
      return;  // prune default-class template subtrees
    } else {
      const ElementType et = element_type_of<E>::value;
      FeatureUse& u = used_[et];
      if (u.count == 0) u.first = e.loc;
      ++u.count;
      Recurse rec{this};
      Visit(e, rec);
    }
  }

 private:
  struct Recurse {
    FeatureCollector* c;
    template <class T>
    void field(int, const char*, const T&) {}
    template <class T>
    void child(int, const char*, const std::vector<std::unique_ptr<T>>& l) {
      for (const auto& p : l)
        if (p) (*c)(*p);
    }
    template <class U>
    void union_child(int, const char*, const std::vector<U>& l) {
      for (const auto& item : l)
        std::visit([&](const auto& p) { if (p) (*c)(*p); }, item.node);
    }
  };

  std::unordered_map<ElementType, FeatureUse>& used_;
};

}  // namespace

// Finer-than-family sub-feature scan: some models use an admitted family in a
// way the native path cannot yet compile -- a mesh/asset geom, a site material
// or light texture (uncompiled assets), or free-joint alignment (align="true"
// rewrites frames). Each routes the whole model to the XML fallback with an
// explicit sub-feature key (so the fallback reason names what is missing).
class SubFeatureScanner {
 public:
  SubFeatureScanner(const Model& m, std::unordered_map<std::string, FeatureUse>& out)
      : m_(m), out_(out) {}

  void operator()(const Model& m) { Recurse rec{this}; Visit(m, rec); }

  template <class E>
  void operator()(const E& e) {
    if constexpr (std::is_same_v<E, Default>) {
      (void)e;
      return;  // class templates are not compiled objects
    } else {
      if constexpr (std::is_same_v<E, Geom>) Check(e);
      else if constexpr (std::is_same_v<E, Mesh>) CheckMesh(e);
      else if constexpr (std::is_same_v<E, Texture>) CheckTexture(e);
      else if constexpr (std::is_same_v<E, Material>) CheckMaterial(e);
      else if constexpr (std::is_same_v<E, Hfield>) CheckHfield(e);
      else if constexpr (std::is_same_v<E, FreeJoint>) CheckFreeJoint(e);
      else if constexpr (std::is_same_v<E, ActuatorGeneral> ||
                         std::is_same_v<E, Motor> ||
                         std::is_same_v<E, Position> ||
                         std::is_same_v<E, Velocity> ||
                         std::is_same_v<E, IntVelocity> ||
                         std::is_same_v<E, Damper> ||
                         std::is_same_v<E, Cylinder> ||
                         std::is_same_v<E, Adhesion>)
        CheckActuator(e);
      else if constexpr (std::is_same_v<E, Flex>) CheckFlex(e);
      else if constexpr (std::is_same_v<E, Flexcomp>) CheckFlexcomp(e);
      else if constexpr (std::is_same_v<E, EqualityFlex>) CheckEqualityFlex(e);
      else if constexpr (std::is_same_v<E, SensorContact>) CheckSensorContact(e);
      else if constexpr (std::is_same_v<E, Rangefinder>) CheckRangefinder(e);
      else if constexpr (std::is_same_v<E, Camprojection> ||
                         std::is_same_v<E, Insidesite> ||
                         std::is_same_v<E, Distance> ||
                         std::is_same_v<E, Normal> ||
                         std::is_same_v<E, Fromto>)
        CheckSensorDelay(e);
      else if constexpr (std::is_same_v<E, Compiler>) CheckCompiler(e);
      else if constexpr (std::is_same_v<E, Size>) CheckSize(e);
      else if constexpr (std::is_same_v<E, TupleElement>) CheckTupleElement(e);
      else if constexpr (std::is_same_v<E, Replicate>) CheckReplicate(e);
      Recurse rec{this};
      Visit(e, rec);
    }
  }

 private:
  void Note(const char* key, const ps::SourceLoc& loc) {
    FeatureUse& u = out_[key];
    if (u.count == 0) u.first = loc;
    ++u.count;
  }
  // Native textures: builtin procedural (gradient/checker/flat) and single-file
  // 2D/cube (PNG/KTX/custom, gridsize composition, hflip/vflip). Cube-from-
  // separate-files (LoadCubeSeparate), an authored content_type (needs the
  // MIME-attr parser), and a placeholder with neither builtin nor file route the
  // whole model to the XML fallback.
  void CheckTexture(const Texture& t) {
    std::unique_ptr<Texture> eff = ps::sdk::Effective(m_, t);
    const bool has_cube = eff->fileright || eff->fileleft || eff->fileup ||
                          eff->filedown || eff->filefront || eff->fileback;
    const TexFile* tf =
        eff->source ? std::get_if<TexFile>(&*eff->source) : nullptr;
    const TextureBuiltin* b =
        eff->source ? std::get_if<TextureBuiltin>(&*eff->source) : nullptr;
    const bool is_builtin = b && *b != TextureBuiltin::none;
    const bool is_file = tf && !tf->file.empty();
    if (has_cube || eff->content_type || (!is_builtin && !is_file))
      Note("texture.file", t.loc);
  }
  // The native material compile inherits scalar fields from the class chain but
  // not the <layer> child list (child-list class inheritance is unmodeled), so a
  // material whose effective layer set is larger than its authored one (layers
  // inherited from a class default) routes to the XML fallback.
  void CheckMaterial(const Material& mat) {
    if (!mat.layers.empty()) return;  // authored its own layers: handled
    ps::sdk::ParentMap pm(m_);
    ps::sdk::detail::DefaultIndex idx(m_);
    const std::string cls =
        ps::sdk::detail::ResolveClassName(pm, ps::sdk::detail::OwnClass(mat), &mat);
    for (const ps::mjcf::Default* d = idx.ByNameOrRoot(cls); d;
         d = idx.ParentOf(d)) {
      const auto* vec = ps::sdk::DefaultVec<Material>(*d);
      if (vec && !vec->empty() && vec->front() && !vec->front()->layers.empty()) {
        Note("material.class_layers", mat.loc);
        return;
      }
    }
  }
  // Native hfields: inline elevation (user-data) and single-file (PNG grey via
  // lifted DecodePNG, or custom binary). An authored content_type (needs the
  // MIME-attr parser) routes to the XML fallback.
  void CheckHfield(const Hfield& h) {
    if (h.content_type) Note("hfield.file", h.loc);
  }
  // Free-joint alignment (align="true", or align="auto" with compiler
  // alignfree) rewrites the body/inertial frames and child-geom poses -- out of
  // the NC1b rigid-body slice, so route such a model to the XML fallback.
  void CheckFreeJoint(const FreeJoint& fj) {
    if (fj.align && *fj.align == TriState::true_) Note("freejoint.align", fj.loc);
  }
  // The native path resolves joint/jointinparent/tendon/body transmission only,
  // has no history/delay buffer, and does not run mj_setLengthRange, so site /
  // refsite / slidercrank transmission, a delay buffer, or a length-range-needing
  // gain/bias type (muscle/user) route to the XML fallback.
  template <class E>
  void CheckActuator(const E& a) {
    if constexpr (requires { a.nsample; })
      if (a.nsample) Note("actuator.delay", a.loc);
    if constexpr (requires { a.delay; })
      if (a.delay) Note("actuator.delay", a.loc);
    // MuJoCo's cylinder reader reads the 3-vector `bias` into a single double
    // (ReadAttr("bias", 3, &bias)), overflowing the stack and corrupting the
    // adjacent `timeconst` with bias[1] -- deterministic upstream UB that leg B
    // reproduces but the native path cannot. Gate any cylinder authoring bias.
    if constexpr (std::is_same_v<E, Cylinder>) {
      if (a.bias) Note("cylinder.bias", a.loc);
    }
    if constexpr (std::is_same_v<E, ActuatorGeneral>) {
      if (a.gaintype && (*a.gaintype == GainType::user ||
                         *a.gaintype == GainType::dcmotor))
        Note("actuator.gaintype", a.loc);
      if (a.biastype && (*a.biastype == BiasType::user ||
                         *a.biastype == BiasType::dcmotor))
        Note("actuator.biastype", a.loc);
      if (a.dyntype && (*a.dyntype == DynType::user ||
                        *a.dyntype == DynType::dcmotor))
        Note("actuator.dyntype", a.loc);
    }
  }
  // The native flex path (NC5 Wave 1) compiles only the young=0, non-interpolated
  // edge-only geometry. Node/dof interpolation, young>0 linear elasticity, and
  // elastic2d bending route to the XML fallback; a dim>1 edge stiffness is an
  // upstream hard error (both paths), gated here to keep the native path off it.
  void CheckFlex(const Flex& f) {
    if (f.node || f.dof) Note("flex.interpolated", f.loc);
    const int dim = f.dim ? *f.dim : 2;
    if (!f.flexElasticitys.empty() && f.flexElasticitys.front()) {
      const FlexElasticity& e = *f.flexElasticitys.front();
      if (e.young && *e.young > 0) Note("flex.elasticity", f.loc);
      if (e.elastic2d && *e.elastic2d != Elastic2D::none)
        Note("flex.elastic2d", f.loc);
    }
    if (!f.flexEdges.empty() && f.flexEdges.front()) {
      const FlexEdge& e = *f.flexEdges.front();
      if (e.stiffness && *e.stiffness > 0 && dim > 1)
        Note("flex.edgestiffness", f.loc);
    }
  }
  // EqualityFlex shares the "flex" tag with the Flex element (admitted), but the
  // native equality path does not compile flex equalities yet.
  void CheckEqualityFlex(const EqualityFlex& e) { Note("equality.flex", e.loc); }
  // The native flexcomp path expands the procedural grid/box/square family plus
  // the direct-point family -- direct (inline points, Wave 4), gmsh (.msh file,
  // Wave 5), and mesh (OBJ/STL file, Wave 5b) -- at dof=full, non-interpolated,
  // with edge/vert/none equality (Wave 6b), young>0 linear elasticity (Stencil2D/
  // 3D stiffness + elastic2d bending, Wave 3), and radial/2d reduced dof. Only
  // interpolated dof (trilinear/quadratic) and strain equality route to fallback.
  void CheckFlexcomp(const Flexcomp& fc) {
    const FlexDof dof = fc.dof ? *fc.dof : FlexDof::full;
    const bool interpolated =
        dof == FlexDof::trilinear || dof == FlexDof::quadratic;
    // interpolated dof (trilinear/quadratic) now expands the nodal finite-element
    // mesh + FE stiffness/bending native (NC5 Wave 6); strain (mjEQ_FLEXSTRAIN)
    // emits one constraint per FE cell / boundary face.
    const FlexElasticity* el =
        (!fc.flexElasticitys.empty() && fc.flexElasticitys.front())
            ? fc.flexElasticitys.front().get()
            : nullptr;
    const double young = el && el->young ? *el->young : 0;
    const bool shell = el && el->elastic2d && *el->elastic2d != Elastic2D::none;
    if (!fc.flexcompEdges.empty() && fc.flexcompEdges.front()) {
      const FlexcompEdge& fe = *fc.flexcompEdges.front();
      const bool has_equality = fe.equality.has_value();
      // reader hard error (xml_native_reader.cc:2942): flex constraints and
      // elasticity (young>0) cannot coexist unless elastic2d==bend. Both paths
      // fail the same way, so route to the oracle.
      if (has_equality && young > 0 && !shell)
        Note("flexcomp.constraint_elasticity", fc.loc);
      // interpolated pin grid(range) on a non-grid type uses an adjusted count
      // check (Make :364) not reproduced natively.
      if (interpolated && fc.type && *fc.type != FlexcompType::grid) {
        for (const auto& p : fc.flexcompPins)
          if (p && (p->grid || p->gridrange))
            Note("flexcomp.interpolated_pingrid", fc.loc);
      }
    }
    if (!fc.plugin.empty()) Note("flexcomp.plugin", fc.loc);
  }
  // The contact sensor shares the "contact" tag with the <contact> section
  // (admitted for pairs/excludes), so it slips past the family gate; its
  // variable dim + intprm bitmask are out of NC2 scope. Force the fallback.
  // The contact sensor (dataspec/reduce/num intprm, variable dim) is native; only
  // its delay buffer routes to the fallback (shared sensor delay gate).
  void CheckSensorContact(const SensorContact& s) { CheckSensorDelay(s); }
  // A sensor delay buffer (nsample>0) or fixed delay drives the sensor_history
  // ring the native path does not emit; route such a sensor to the fallback.
  template <class E>
  void CheckSensorDelay(const E& e) {
    if ((e.nsample && *e.nsample > 0) || e.delay) Note("sensor.delay", e.loc);
  }
  // A camera-target rangefinder's dim scales with the camera resolution (one ray
  // per pixel); only the single-ray site rangefinder is native. Plus the shared
  // delay gate.
  void CheckRangefinder(const Rangefinder& r) {
    if (r.camera) Note("rangefinder.camera", r.loc);
    CheckSensorDelay(r);
  }
  void CheckCompiler(const Compiler& c) {
    if (c.alignfree && *c.alignfree) Note("compiler.alignfree", c.loc);
    // discardvisual="true" drops visual-only geoms in the parser (renumbering
    // geoms/bvh and dropping their inertia); the native path keeps every geom,
    // so route such a model to the XML fallback.
    if (c.discardvisual && *c.discardvisual) Note("compiler.discardvisual", c.loc);
  }
  // A tuple resolves obj refs only for the object families the native compiler
  // builds an id map for (body/xbody/geom/site/joint/camera/tendon/actuator);
  // any other objtype (mesh/material/etc.) routes to the XML fallback.
  // The native replicate expansion clones the subtree with an accumulating pose
  // and name suffix, but does not yet propagate a childclass into the clones
  // (the reader builds the subtree under the childdef default); gate those.
  void CheckReplicate(const Replicate& r) {
    if (!r.inertial.empty()) Note("replicate.inertial", r.loc);
  }
  void CheckTupleElement(const TupleElement& e) {
    if (!e.objtype) return;
    switch (mju_str2Type(e.objtype->c_str())) {
      case mjOBJ_BODY: case mjOBJ_XBODY: case mjOBJ_GEOM: case mjOBJ_SITE:
      case mjOBJ_JOINT: case mjOBJ_CAMERA: case mjOBJ_TENDON: case mjOBJ_ACTUATOR:
        return;
      default:
        Note("tuple.objtype", e.loc);
    }
  }
  // <size memory|nstack|njmax|nconmax> now drive SetNarena's explicit-bytes /
  // legacy-stack / deprecated-constraint branches natively. Only a malformed
  // <size memory> string (which mj_loadXML itself rejects) routes to fallback.
  void CheckSize(const Size& s) {
    if (s.memory) {
      bool ok = true;
      ParseSizeMemoryBytes(*s.memory, &ok);
      if (!ok) Note("size.memory", s.loc);
    }
  }
  // A compiled (non-plugin) mesh geom is native; an SDF geom, a fitgeom (a
  // primitive type that references a mesh, whose size FitGeom overwrites), and a
  // plugin/fluidshape geom are not. Effective resolves type/mesh through the
  // class chain so a mesh/sdf type or mesh ref authored in a default is seen.
  void Check(const Geom& g) {
    std::unique_ptr<Geom> eff = ps::sdk::Effective(m_, g);
    const int type = eff->type ? static_cast<int>(*eff->type)
                               : static_cast<int>(GeomType::sphere);
    if (type == static_cast<int>(GeomType::sdf)) Note("geom.sdf", g.loc);
    if (!g.plugin.empty()) Note("geom.plugin", g.loc);
  }
  // Plugin / SDF meshes need marching cubes + octree (not native). A builtin
  // (procedural sphere/cone/...) mesh needs the generator the native pipeline
  // does not run, so route it to the XML fallback.
  void CheckMesh(const Mesh& mesh) {
    if (!mesh.plugin.empty()) Note("mesh.plugin", mesh.loc);
    if (mesh.builtin && *mesh.builtin != MeshBuiltin::none)
      Note("mesh.builtin", mesh.loc);
  }
  struct Recurse {
    SubFeatureScanner* c;
    template <class T> void field(int, const char*, const T&) {}
    template <class T>
    void child(int, const char*, const std::vector<std::unique_ptr<T>>& l) {
      for (const auto& p : l) if (p) (*c)(*p);
    }
    template <class U>
    void union_child(int, const char*, const std::vector<U>& l) {
      for (const auto& item : l)
        std::visit([&](const auto& p) { if (p) (*c)(*p); }, item.node);
    }
  };
  const Model& m_;
  std::unordered_map<std::string, FeatureUse>& out_;
};

// Walks geoms/sites and flags the eager-copy partial-size-default case.
class PartialArrayScanner {
 public:
  PartialArrayScanner(const Model& m, std::unordered_map<std::string, FeatureUse>& out)
      : m_(m), idx_(m), pm_(m), out_(out) {}

  void operator()(const Model& m) { Recurse rec{this}; Visit(m, rec); }

  template <class E>
  void operator()(const E& e) {
    if constexpr (std::is_same_v<E, Default>) { (void)e; return; }
    else {
      if constexpr (std::is_same_v<E, ActuatorGeneral> ||
                           std::is_same_v<E, Motor> || std::is_same_v<E, Position> ||
                           std::is_same_v<E, Velocity> || std::is_same_v<E, IntVelocity> ||
                           std::is_same_v<E, Damper> || std::is_same_v<E, Cylinder> ||
                           std::is_same_v<E, Muscle> || std::is_same_v<E, Adhesion>) {
        if (ForeignActuatorDefault(idx_, pm_, e))
          Note("actuator.cross_spelling_default", e.loc);
      }
      Recurse rec{this};
      Visit(e, rec);
    }
  }

 private:
  void Note(const char* key, const ps::SourceLoc& loc) {
    FeatureUse& u = out_[key];
    if (u.count == 0) u.first = loc;
    ++u.count;
  }
  struct Recurse {
    PartialArrayScanner* c;
    template <class T> void field(int, const char*, const T&) {}
    template <class T>
    void child(int, const char*, const std::vector<std::unique_ptr<T>>& l) {
      for (const auto& p : l) if (p) (*c)(*p);
    }
    template <class U>
    void union_child(int, const char*, const std::vector<U>& l) {
      for (const auto& it : l)
        std::visit([&](const auto& p) { if (p) (*c)(*p); }, it.node);
    }
  };
  const Model& m_;
  ps::sdk::detail::DefaultIndex idx_;
  ps::sdk::ParentMap pm_;
  std::unordered_map<std::string, FeatureUse>& out_;
};

// Names of referenceable elements (body/geom/site/joint) authored inside a
// <replicate> subtree. After native expansion each carries an index suffix, so a
// model-level element that references one of these ORIGINAL names is one the XML
// reader would replicate too (mjs_attach's referencing-element cloning), which
// the native expansion does not reproduce.
void CollectReplicateNames(const std::vector<BodyChildAny>& subtree, bool in_rep,
                           std::set<std::string>& out) {
  auto add = [&](const ps::opt<std::string>& nm) {
    if (in_rep && nm && !nm->empty()) out.insert(*nm);
  };
  for (const BodyChildAny& c : subtree) {
    switch (c.kind()) {
      case BodyChildAny::Kind::Geom:
        add(std::get<std::unique_ptr<Geom>>(c.node)->name); break;
      case BodyChildAny::Kind::Site:
        add(std::get<std::unique_ptr<Site>>(c.node)->name); break;
      case BodyChildAny::Kind::Joint:
        add(std::get<std::unique_ptr<Joint>>(c.node)->name); break;
      case BodyChildAny::Kind::FreeJoint:
        add(std::get<std::unique_ptr<FreeJoint>>(c.node)->name); break;
      case BodyChildAny::Kind::Body: {
        const auto& b = std::get<std::unique_ptr<Body>>(c.node);
        add(b->name); CollectReplicateNames(b->subtree, in_rep, out); break;
      }
      case BodyChildAny::Kind::Frame:
        CollectReplicateNames(std::get<std::unique_ptr<Frame>>(c.node)->subtree,
                              in_rep, out);
        break;
      case BodyChildAny::Kind::Replicate:
        CollectReplicateNames(std::get<std::unique_ptr<Replicate>>(c.node)->subtree,
                              true, out);
        break;
      default:
        break;
    }
  }
}

// Collect the target names of every typed cross-reference (opt<Ref<>>/Ref<>) in
// a model section subtree.
class RefNameCollector {
 public:
  explicit RefNameCollector(std::set<std::string>& out) : out_(out) {}
  template <class E>
  void operator()(const E& e) { Recurse rec{this}; Visit(e, rec); }

 private:
  struct Recurse {
    RefNameCollector* c;
    template <class T>
    void field(int, const char*, const T& v) {
      using DT = std::decay_t<T>;
      if constexpr (ps::sdk::detail::opt_ref<DT>::value) {
        if (v && !v->name.empty()) c->out_.insert(v->name);
      } else if constexpr (ps::sdk::detail::is_ref<DT>::value) {
        if (!v.name.empty()) c->out_.insert(v.name);
      }
    }
    template <class T>
    void child(int, const char*, const std::vector<std::unique_ptr<T>>& l) {
      for (const auto& p : l) if (p) (*c)(*p);
    }
    template <class U>
    void union_child(int, const char*, const std::vector<U>& l) {
      for (const auto& it : l)
        std::visit([&](const auto& p) { if (p) (*c)(*p); }, it.node);
    }
  };
  std::set<std::string>& out_;
};

// Gate a <replicate> whose (unclassed) clones would need a childclass the native
// ParentMap cannot supply (clones live outside the source tree): a childclass in
// scope -- the replicate's own, an ancestor body's, or one inside its subtree --
// means an unclassed descendant silently loses its class in the clone.
bool SubtreeHasChildclass(const std::vector<BodyChildAny>& subtree) {
  for (const BodyChildAny& c : subtree) {
    switch (c.kind()) {
      case BodyChildAny::Kind::Body: {
        const auto& b = std::get<std::unique_ptr<Body>>(c.node);
        if (b->childclass) return true;
        if (SubtreeHasChildclass(b->subtree)) return true;
        break;
      }
      case BodyChildAny::Kind::Frame:
        if (SubtreeHasChildclass(
                std::get<std::unique_ptr<Frame>>(c.node)->subtree)) return true;
        break;
      case BodyChildAny::Kind::Replicate: {
        const auto& r = std::get<std::unique_ptr<Replicate>>(c.node);
        if (r->childclass) return true;
        if (SubtreeHasChildclass(r->subtree)) return true;
        break;
      }
      default:
        break;
    }
  }
  return false;
}

void ScanReplicateChildclass(
    const std::vector<BodyChildAny>& subtree, bool cc_scope,
    const std::function<void(const char*, const ps::SourceLoc&)>& note) {
  for (const BodyChildAny& c : subtree) {
    switch (c.kind()) {
      case BodyChildAny::Kind::Body: {
        const auto& b = std::get<std::unique_ptr<Body>>(c.node);
        ScanReplicateChildclass(b->subtree, cc_scope || b->childclass.has_value(),
                                note);
        break;
      }
      case BodyChildAny::Kind::Frame:
        ScanReplicateChildclass(
            std::get<std::unique_ptr<Frame>>(c.node)->subtree, cc_scope, note);
        break;
      case BodyChildAny::Kind::Replicate: {
        const auto& r = std::get<std::unique_ptr<Replicate>>(c.node);
        if (cc_scope || r->childclass || SubtreeHasChildclass(r->subtree))
          note("replicate.childclass", r->loc);
        ScanReplicateChildclass(r->subtree, cc_scope || r->childclass.has_value(),
                                note);
        break;
      }
      default:
        break;
    }
  }
}

// True if `subtree` contains a flexcomp anywhere (frames/replicate/bodies
// recursed). Used by the document-order hazard scan below.
bool SubtreeHasFlexcomp(const std::vector<BodyChildAny>& subtree) {
  for (const BodyChildAny& c : subtree) {
    switch (c.kind()) {
      case BodyChildAny::Kind::Flexcomp:
        return true;
      case BodyChildAny::Kind::Body: {
        const auto& b = std::get<std::unique_ptr<Body>>(c.node);
        if (b && SubtreeHasFlexcomp(b->subtree)) return true;
        break;
      }
      case BodyChildAny::Kind::Frame: {
        const auto& f = std::get<std::unique_ptr<Frame>>(c.node);
        if (f && SubtreeHasFlexcomp(f->subtree)) return true;
        break;
      }
      case BodyChildAny::Kind::Replicate: {
        const auto& r = std::get<std::unique_ptr<Replicate>>(c.node);
        if (r && SubtreeHasFlexcomp(r->subtree)) return true;
        break;
      }
      default:
        break;
    }
  }
  return false;
}

// The native collector expands a body's own flexcomps (assigning flex ids)
// before descending into child bodies, whereas mjCFlexcomp::Make runs at parse
// time in document order (a flexcomp inside an earlier child body precedes a
// later flexcomp sibling). The two orders diverge exactly when, within one
// (frame/replicate-flattened) body level, a child body transitively holding a
// flexcomp appears before a flexcomp sibling. `seen_body_fc` threads that state
// across the flattened level.
bool FlexcompOrderHazard(const std::vector<BodyChildAny>& subtree,
                         bool& seen_body_fc) {
  for (const BodyChildAny& c : subtree) {
    switch (c.kind()) {
      case BodyChildAny::Kind::Flexcomp:
        if (seen_body_fc) return true;
        break;
      case BodyChildAny::Kind::Body: {
        const auto& b = std::get<std::unique_ptr<Body>>(c.node);
        if (b) {
          if (SubtreeHasFlexcomp(b->subtree)) seen_body_fc = true;
          bool child_seen = false;
          if (FlexcompOrderHazard(b->subtree, child_seen)) return true;
        }
        break;
      }
      case BodyChildAny::Kind::Frame: {  // flattened inline: shares the flag
        const auto& f = std::get<std::unique_ptr<Frame>>(c.node);
        if (f && FlexcompOrderHazard(f->subtree, seen_body_fc)) return true;
        break;
      }
      case BodyChildAny::Kind::Replicate: {
        const auto& r = std::get<std::unique_ptr<Replicate>>(c.node);
        if (r && FlexcompOrderHazard(r->subtree, seen_body_fc)) return true;
        break;
      }
      default:
        break;
    }
  }
  return false;
}

// --------------------------------------------------------------------------- //
// NC6/NC6c: native <attach>/<model> expansion (full mjs_attach import).         //
//                                                                              //
// ProtoSpec stores <model file=...> as a ModelAsset (a file ref, NOT a parsed  //
// child) and <attach model=.. body=.. prefix=..> as an Attach body-child; the  //
// reader passes both through unexpanded (DR-7), so leg B relies on mj_loadXML   //
// to recursively parse the child file and run mjs_attach. The native path       //
// reproduces mjs_attach (user_api.cc:386-459 -> mjCFrame::operator+= :2939 ->   //
// mjCModel::operator+= :452-503) into a synthetic model: parse the child        //
// (io::ParseMjcfFile; orientation is already quat-resolved in the child's own   //
// compiler context, so the graft is context-safe), deep-clone the named body    //
// (or fabricate an identity frame over the whole child worldbody for an empty-  //
// body whole-model attach, :397-412), and prefix-namespace it. Since attach is  //
// always cross-model, mjCBase::NameSpace's `m != model` guards are always true, //
// so every name, classname, and typed reference is prefixed uniformly           //
// (user_objects.cc:1490-1497 + the per-family NameSpace overrides). The child's //
// <default> tree grafts as a prefix-named subclass under the parent root, its   //
// assets and referencing sections (tendons/equalities/actuators/sensors/        //
// contacts) are appended prefixed, and its keyframes are placed at the graft's  //
// qpos offset -- all merged into the synth so the existing build pipeline        //
// compiles the graft in context. Clone regenerates serials, so parent unnamed   //
// elements' original serials are copied back (their _ps auto-name must match the //
// oracle) and imported child unnamed elements are given an authored empty name   //
// (the oracle pulls the child raw, so they stay unnamed).                        //
// --------------------------------------------------------------------------- //
namespace {

// --- NC6c: full mjs_attach import ----------------------------------------- //
//
// mjs_attach copies far more than the graft body: the child model's default
// tree, its referenced assets, its referencing (non-tree) elements, and its
// keyframes, all NameSpace-prefixed into the parent (user_api.cc:386-459 ->
// mjCFrame::operator+= :2939-2985 -> mjCModel::operator+= :452-503). Since attach
// is always cross-model, mjCBase::NameSpace's `m != model` guards are always
// true, so every name, classname, and every typed reference is prefixed
// uniformly (user_objects.cc:1490-1497 + the per-family NameSpace overrides).
// ProtoSpec reproduces this by MERGING the child sections into the synth Model
// so the existing build pipeline compiles the graft in context.

// Set an element's own class to `cls` when it authored none (opt<Ref<Default>>
// dclass field). No-op for elements without a dclass field.
template <class E>
void TagClass(E& e, const std::string& cls) {
  if constexpr (requires { e.dclass; }) {
    using DT = std::decay_t<decltype(e.dclass)>;
    if constexpr (ps::sdk::detail::is_opt<DT>::value) {
      using Inner = typename ps::sdk::detail::is_opt<DT>::inner;
      if constexpr (ps::sdk::detail::is_ref<Inner>::value) {
        if (!e.dclass || e.dclass->name.empty()) {
          Inner r;
          r.name = cls;
          e.dclass = r;
        }
      }
    }
  }
}

// Prefix every name and every typed reference (target refs + dclass + childclass)
// throughout an element subtree -- the uniform cross-model NameSpace.
template <class E>
void PrefixTree(E& root, const std::string& prefix) {
  ps::sdk::detail::WalkTree(root, [&](auto& e) {
    if (const std::string* nm = ps::sdk::detail::NameOf(e))
      ps::sdk::detail::SetName(e, prefix + *nm);
    ps::sdk::detail::RefPrefixer p{&prefix};
    ps::mjcf::Visit(e, p);
  });
}

// A class-free child element resolves to the child's root default ("main"). After
// merge that root is a subclass named `prefix` under the parent root, so a
// class-free imported element must carry `prefix` as its class; one that inherits
// a childclass already resolves correctly and is left alone.
void TagGraftSubtree(std::vector<BodyChildAny>& subtree, const std::string& prefix,
                     bool cc_scope);

void TagGraftBody(Body& b, const std::string& prefix, bool cc_scope) {
  bool scope = cc_scope || (b.childclass && !b.childclass->name.empty());
  if (!scope) {
    ps::Ref<Default> r;
    r.name = prefix;
    b.childclass = r;  // shields descendants from any host childclass
    scope = true;
  }
  TagGraftSubtree(b.subtree, prefix, scope);
}

void TagGraftSubtree(std::vector<BodyChildAny>& subtree, const std::string& prefix,
                     bool cc_scope) {
  for (BodyChildAny& c : subtree) {
    switch (c.kind()) {
      case BodyChildAny::Kind::Body:
        if (auto& b = std::get<std::unique_ptr<Body>>(c.node))
          TagGraftBody(*b, prefix, cc_scope);
        break;
      case BodyChildAny::Kind::Frame:
        if (auto& f = std::get<std::unique_ptr<Frame>>(c.node))
          TagGraftSubtree(f->subtree, prefix, cc_scope);
        break;
      case BodyChildAny::Kind::Replicate:
        if (auto& r = std::get<std::unique_ptr<Replicate>>(c.node))
          TagGraftSubtree(r->subtree, prefix, cc_scope);
        break;
      case BodyChildAny::Kind::Geom:
        if (!cc_scope) TagClass(*std::get<std::unique_ptr<Geom>>(c.node), prefix);
        break;
      case BodyChildAny::Kind::Site:
        if (!cc_scope) TagClass(*std::get<std::unique_ptr<Site>>(c.node), prefix);
        break;
      case BodyChildAny::Kind::Joint:
        if (!cc_scope) TagClass(*std::get<std::unique_ptr<Joint>>(c.node), prefix);
        break;
      case BodyChildAny::Kind::Camera:
        if (!cc_scope) TagClass(*std::get<std::unique_ptr<Camera>>(c.node), prefix);
        break;
      case BodyChildAny::Kind::Light:
        if (!cc_scope) TagClass(*std::get<std::unique_ptr<Light>>(c.node), prefix);
        break;
      default:
        break;
    }
  }
}

// Tag class-free referencing elements inside one imported section with `prefix`
// (only referencing families carry a dclass; wrap/anchor sub-items do not).
template <class E>
void TagSectionClasses(E& section, const std::string& prefix) {
  ps::sdk::detail::WalkTree(section, [&](auto& e) { TagClass(e, prefix); });
}

// Give every still-unnamed nameable element in an imported subtree an authored
// EMPTY name. mjs_attach pulls the child straight from disk in the XML oracle,
// so its unnamed elements stay unnamed; the native build otherwise injects the
// `_ps:<family>:<serial>` auto-name (which uses the clone's regenerated serial
// and would never match). A present-but-empty name suppresses that auto-name and
// reproduces the empty mjModel name entry. (Default's dclass identity is skipped:
// imported default subclasses are always renamed, never left blank.)
template <class E>
void EmptyUnnamed(E& root) {
  ps::sdk::detail::WalkTree(root, [&](auto& e) {
    using X = std::decay_t<decltype(e)>;
    if constexpr (!std::is_same_v<X, Default>) {
      if (ps::sdk::detail::HasNameField<X>() && !ps::sdk::detail::NameOf(e))
        ps::sdk::detail::SetName(e, "");
    }
  });
}

// Merge one parsed child's default tree, assets, and referencing elements into
// the synth model (prefix-namespaced). `root` is synth's root default; the child
// default tree grafts under it as a subclass named `prefix` so a class-free
// child element resolves prefix -> root (child main -> parent main), and a named
// child class resolves child_class -> prefix -> root (mjCModel::operator+= :644 +
// the def merge at user_objects.cc:2969-2974).
void ImportChildSections(const Model& child, const std::string& prefix,
                         Default* root, Model& synth) {
  // Defaults: exactly one child block for the tractable slice (see gate).
  for (const auto& d : child.defaults) {
    if (!d) continue;
    std::unique_ptr<Default> cd = Clone(*d);
    PrefixTree(*cd, prefix);   // prefixes nested class names + template refs
    cd->dclass = prefix;       // child root ("" -> prefix), nested under parent root
    root->subclasses.push_back(std::move(cd));
  }

  // Assets (meshes/skins/hfields/textures/materials): appended prefixed. Nested
  // <model> assets are gated out before this runs.
  for (const auto& a : child.assets) {
    if (!a) continue;
    std::unique_ptr<Asset> ca = Clone(*a);
    PrefixTree(*ca, prefix);
    EmptyUnnamed(*ca);
    synth.assets.push_back(std::move(ca));
  }

  // Referencing sections (tendons, equalities, actuators, sensors, contacts):
  // prefixed and tagged, appended after the parent's own (operator+= push_back
  // order). Refs that do not resolve into the graft would be dropped by
  // mjs_attach; the corpus grafts resolve every ref, so the whole child section
  // is imported and any unresolved ref surfaces as a build failure -> fallback.
  auto import = [&](const auto& src, auto& dst) {
    for (const auto& s : src) {
      if (!s) continue;
      auto cs = Clone(*s);
      PrefixTree(*cs, prefix);
      TagSectionClasses(*cs, prefix);
      EmptyUnnamed(*cs);
      dst.push_back(std::move(cs));
    }
  };
  import(child.tendons, synth.tendons);
  import(child.equalitys, synth.equalitys);
  import(child.actuators, synth.actuators);
  import(child.sensors, synth.sensors);
  import(child.contacts, synth.contacts);
}

// One imported child keyframe awaiting offset resolution: the grafted node it
// belongs to determines where its qpos slice lands in the parent layout.
struct KeyImport {
  Key* key;
  const void* graft;
};

// nq contributed by one joint.
int JointNq(const Joint& j) {
  switch (j.type ? *j.type : JointType::hinge) {
    case JointType::free: return 7;
    case JointType::ball: return 4;
    default:              return 1;  // slide, hinge
  }
}

// Sum the nq of every joint in a subtree; set `macro` if a joint-bearing macro
// (replicate/flexcomp/composite) is present (its dof count is not known until the
// build expands it, so a keyframe offset past it cannot be computed here).
void CountSubtreeNq(const std::vector<BodyChildAny>& sub, int& nq, bool& macro) {
  for (const BodyChildAny& c : sub) {
    switch (c.kind()) {
      case BodyChildAny::Kind::Joint:
        nq += JointNq(*std::get<std::unique_ptr<Joint>>(c.node)); break;
      case BodyChildAny::Kind::FreeJoint: nq += 7; break;
      case BodyChildAny::Kind::Body:
        CountSubtreeNq(std::get<std::unique_ptr<Body>>(c.node)->subtree, nq, macro);
        break;
      case BodyChildAny::Kind::Frame:
        CountSubtreeNq(std::get<std::unique_ptr<Frame>>(c.node)->subtree, nq, macro);
        break;
      case BodyChildAny::Kind::Replicate: {
        // A <replicate count=N> emits N copies of its subtree (native replicate
        // pass), contributing N x subtree-nq dofs.
        const auto& r = std::get<std::unique_ptr<Replicate>>(c.node);
        int sub_nq = 0;
        CountSubtreeNq(r->subtree, sub_nq, macro);
        nq += r->count * sub_nq;
        break;
      }
      case BodyChildAny::Kind::Flexcomp:
      case BodyChildAny::Kind::Composite:
        macro = true; break;
      default: break;
    }
  }
}

// Depth-first (== mjModel qposadr) walk accumulating the nq of every joint BEFORE
// `graft` into `nq`; sets `found` when the graft node is reached and `macro` if a
// joint-bearing macro precedes it (offset not computable). Grafts are complete
// bodies/frames, so joints are counted whole.
void WalkToGraftNq(const std::vector<BodyChildAny>& sub, const void* graft,
                   int& nq, bool& found, bool& macro) {
  for (const BodyChildAny& c : sub) {
    if (found || macro) return;
    switch (c.kind()) {
      case BodyChildAny::Kind::Joint:
        nq += JointNq(*std::get<std::unique_ptr<Joint>>(c.node)); break;
      case BodyChildAny::Kind::FreeJoint: nq += 7; break;
      case BodyChildAny::Kind::Body: {
        const auto& b = std::get<std::unique_ptr<Body>>(c.node);
        if (b.get() == graft) { found = true; return; }
        WalkToGraftNq(b->subtree, graft, nq, found, macro);
        break;
      }
      case BodyChildAny::Kind::Frame: {
        const auto& f = std::get<std::unique_ptr<Frame>>(c.node);
        if (f.get() == graft) { found = true; return; }
        WalkToGraftNq(f->subtree, graft, nq, found, macro);
        break;
      }
      case BodyChildAny::Kind::Replicate: {
        // A <replicate> BEFORE the graft (an attach is never nested inside one
        // that reaches here -- that is gated) contributes count x subtree-nq.
        const auto& r = std::get<std::unique_ptr<Replicate>>(c.node);
        int sub_nq = 0;
        CountSubtreeNq(r->subtree, sub_nq, macro);
        nq += r->count * sub_nq;
        break;
      }
      case BodyChildAny::Kind::Flexcomp:
      case BodyChildAny::Kind::Composite:
        macro = true; return;
      default: break;
    }
  }
}

// Import a child's keyframes: prefix names, and (for the tractable slice) record
// each for offset resolution. Only qpos keys are handled; a key carrying qvel/act/
// ctrl/mpos/mquat routes to the fallback (the graft's dof/act/mocap offsets are a
// separate placement not yet reproduced). Returns false + reason on such a key.
bool ImportChildKeyframes(const Model& child, const std::string& prefix,
                          const void* graft, Model& synth,
                          std::vector<KeyImport>& keyimports,
                          std::vector<bridge::FallbackReason>& reasons,
                          const ps::SourceLoc& loc) {
  for (const auto& kf : child.keyframes) {
    if (!kf) continue;
    for (const auto& k : kf->keys) {
      if (k && (k->qvel || k->act || k->ctrl || k->mpos || k->mquat)) {
        bridge::FallbackReason r;
        r.feature = "attach.keyframe_state";
        r.count = 1;
        r.first = loc;
        reasons.push_back(std::move(r));
        return false;
      }
    }
    std::unique_ptr<Keyframe> ckf = Clone(*kf);
    PrefixTree(*ckf, prefix);
    EmptyUnnamed(*ckf);
    for (const auto& k : ckf->keys)
      if (k) keyimports.push_back({k.get(), graft});
    synth.keyframes.push_back(std::move(ckf));
  }
  return true;
}

// A child section this wave does not import bit-exactly yet routes the whole
// model to the XML fallback with a precise reason.
bool ChildGate(const Model& child, const char*& key) {
  for (const auto& a : child.assets)
    if (a && !a->modelAssets.empty()) { key = "attach.child_nested_model"; return true; }
  if (child.defaults.size() > 1) { key = "attach.child_multidefault"; return true; }
  if (!child.deformables.empty()){ key = "attach.child_deformable"; return true; }
  if (!child.customs.empty())    { key = "attach.child_custom";     return true; }
  if (!child.extensions.empty()) { key = "attach.child_plugin";     return true; }
  return false;
}

// Find a top-level worldbody body by name in a parsed child model.
const Body* FindWorldbodyBody(const Model& child, const std::string& name) {
  for (const auto& wb : child.worldbody) {
    if (!wb) continue;
    for (const BodyChildAny& c : wb->subtree) {
      if (c.kind() == BodyChildAny::Kind::Body) {
        const auto& b = std::get<std::unique_ptr<Body>>(c.node);
        if (b && b->name && *b->name == name) return b.get();
      }
    }
  }
  return nullptr;
}

// Parse + resolve the child referenced by an <attach>, returning the parsed
// model (owned by `store`) or nullptr with a written reason.
const Model* LoadChild(
    const Attach& at, const std::string& base_dir,
    const std::unordered_map<std::string, const ModelAsset*>& modelassets,
    std::vector<std::unique_ptr<ps::mjcf::Model>>& store,
    std::vector<bridge::FallbackReason>& reasons, const char*& why) {
  auto fail = [&](const char* feature) -> const Model* {
    why = feature;
    bridge::FallbackReason r;
    r.feature = feature;
    r.count = 1;
    r.first = at.loc;
    reasons.push_back(std::move(r));
    return nullptr;
  };
  if (!at.prefix) return fail("attach.no_prefix");
  if (!at.model || at.model->name.empty()) return fail("attach.no_model_ref");
  auto it = modelassets.find(at.model->name);
  if (it == modelassets.end() || !it->second->file || it->second->file->empty())
    return fail("attach.model_not_found");
  std::filesystem::path child_path =
      (std::filesystem::path(base_dir) / *it->second->file).lexically_normal();
  ps::mjcf::io::ParseResult pr = ps::mjcf::io::ParseMjcfFile(child_path.string());
  if (!pr.ok() || !pr.model) return fail("attach.child_parse_failed");
  const char* gate = nullptr;
  if (ChildGate(*pr.model, gate)) return fail(gate);
  store.push_back(std::move(pr.model));
  return store.back().get();
}

// Resolve one <attach> into the graft node (a prefixed body clone, or a fabricated
// identity frame over the whole child worldbody for a whole-model attach) AND
// import the child's defaults/assets/referencing elements into `synth`. Returns
// false (with a written reason) to route the model to the fallback. `in_replicate`
// gates an attach whose referencing elements would need per-clone replication.
bool ExpandOneAttach(
    const Attach& at, BodyChildAny& slot, bool in_replicate,
    const std::string& base_dir, Default* root, Model& synth,
    const std::unordered_map<std::string, const ModelAsset*>& modelassets,
    std::vector<std::unique_ptr<ps::mjcf::Model>>& store,
    std::vector<KeyImport>& keyimports,
    std::vector<bridge::FallbackReason>& reasons) {
  auto fail = [&](const char* feature) {
    bridge::FallbackReason r;
    r.feature = feature;
    r.count = 1;
    r.first = at.loc;
    reasons.push_back(std::move(r));
    return false;
  };

  const char* why = nullptr;
  const Model* childp = LoadChild(at, base_dir, modelassets, store, reasons, why);
  if (!childp) return false;
  const Model& child = *childp;
  const std::string& prefix = *at.prefix;

  const bool whole = !at.body || at.body->empty();
  const bool has_refs =
      !child.tendons.empty() || !child.equalitys.empty() ||
      !child.actuators.empty() || !child.sensors.empty() || !child.contacts.empty();
  // An attach inside a <replicate> whose child carries referencing elements or
  // keyframes would need those replicated per clone (mjs_attach's per-clone
  // referencing/keyframe cloning); the native replicate pass does not do that.
  if (in_replicate && (has_refs || !child.keyframes.empty()))
    return fail("attach.replicate_referencing");

  if (whole) {
    // Whole-model attach: fabricate an identity frame over every top-level child
    // worldbody element (user_api.cc:397-412), namespace + tag it, splice it in.
    auto frame = std::make_unique<Frame>();
    for (const auto& wb : child.worldbody) {
      if (!wb) continue;
      for (const BodyChildAny& c : wb->subtree)
        frame->subtree.push_back(Clone(c));
    }
    PrefixTree(*frame, prefix);
    TagGraftSubtree(frame->subtree, prefix, false);
    EmptyUnnamed(*frame);
    const void* graft = frame.get();
    ImportChildSections(child, prefix, root, synth);
    if (!ImportChildKeyframes(child, prefix, graft, synth, keyimports, reasons,
                              at.loc))
      return false;
    slot = BodyChildAny{std::move(frame)};
    return true;
  }

  const Body* target = FindWorldbodyBody(child, *at.body);
  if (!target) return fail("attach.body_not_found");
  std::unique_ptr<Body> clone = Clone(*target);
  PrefixTree(*clone, prefix);
  TagGraftBody(*clone, prefix, false);
  EmptyUnnamed(*clone);
  const void* graft = clone.get();
  ImportChildSections(child, prefix, root, synth);
  if (!ImportChildKeyframes(child, prefix, graft, synth, keyimports, reasons,
                            at.loc))
    return false;
  slot = BodyChildAny{std::move(clone)};
  return true;
}

// Replace every <attach> in a body subtree (and nested bodies/frames/replicates)
// with its resolved graft, importing each child's sections into `synth`. On any
// non-tractable attach a reason is pushed and the Attach is left in place (the
// caller falls back before building).
void ExpandAttachInSubtree(
    std::vector<BodyChildAny>& subtree, bool in_replicate,
    const std::string& base_dir, Default* root, Model& synth,
    const std::unordered_map<std::string, const ModelAsset*>& modelassets,
    std::vector<std::unique_ptr<ps::mjcf::Model>>& store,
    std::vector<KeyImport>& keyimports,
    std::vector<bridge::FallbackReason>& reasons) {
  for (BodyChildAny& c : subtree) {
    switch (c.kind()) {
      case BodyChildAny::Kind::Body: {
        auto& b = std::get<std::unique_ptr<Body>>(c.node);
        if (b) ExpandAttachInSubtree(b->subtree, in_replicate, base_dir, root,
                                     synth, modelassets, store, keyimports, reasons);
        break;
      }
      case BodyChildAny::Kind::Frame: {
        auto& f = std::get<std::unique_ptr<Frame>>(c.node);
        if (f) ExpandAttachInSubtree(f->subtree, in_replicate, base_dir, root,
                                     synth, modelassets, store, keyimports, reasons);
        break;
      }
      case BodyChildAny::Kind::Replicate: {
        auto& r = std::get<std::unique_ptr<Replicate>>(c.node);
        if (r) ExpandAttachInSubtree(r->subtree, /*in_replicate=*/true, base_dir,
                                     root, synth, modelassets, store, keyimports,
                                     reasons);
        break;
      }
      case BodyChildAny::Kind::Attach: {
        const auto& at = std::get<std::unique_ptr<Attach>>(c.node);
        if (!at) break;
        ExpandOneAttach(*at, c, in_replicate, base_dir, root, synth, modelassets,
                        store, keyimports, reasons);
        break;
      }
      default:
        break;
    }
  }
}

}  // namespace

// True if the model authors any <attach>. Attach models are rare; the 259
// existing native files author none, so this keeps them on the zero-copy path.
bool ModelUsesAttach(const Model& m) {
  bool found = false;
  ps::sdk::detail::WalkModelLive(const_cast<Model&>(m), [&](auto& e) {
    if constexpr (std::is_same_v<std::decay_t<decltype(e)>, Attach>) found = true;
  });
  return found;
}

// Build a synthetic model with every <attach> expanded in place, or nullptr when
// the model authors no attach (caller compiles the original). Non-tractable
// attaches populate `reasons`; the caller falls back without building.
std::unique_ptr<Model> ExpandAttaches(const Model& m,
                                      const bridge::CompileOptions& opts,
                                      std::vector<bridge::FallbackReason>& reasons) {
  if (!ModelUsesAttach(m)) return nullptr;

  std::unique_ptr<Model> synth = Clone(m);

  // Clone regenerates every serial, but a parent element's `_ps:<family>:<serial>`
  // auto-name (used when it is unnamed) must match the XML oracle, which derives
  // it from the ORIGINAL parse. Copy the original serials back onto the synth in
  // lockstep (both walk the same structure in the same order) before any attach
  // expands and adds graft elements. Graft elements are emptied (EmptyUnnamed), so
  // their regenerated serials never surface.
  {
    std::vector<std::uint64_t> serials;
    ps::sdk::detail::WalkModelLive(const_cast<Model&>(m), [&](auto& e) {
      if constexpr (requires { e.serial; }) serials.push_back(e.serial);
    });
    std::size_t i = 0;
    ps::sdk::detail::WalkModelLive(*synth, [&](auto& e) {
      if constexpr (requires { e.serial; })
        if (i < serials.size()) e.serial = serials[i++];
    });
  }

  // ModelAsset name -> file lookup (the Attach.model ref resolves by name). An
  // <model> with an explicit name uses it; a nameless <model file=child.xml> is
  // referenced by the CHILD's own <mujoco model=..> name (the reader overwrites
  // it only when a name attr is given, xml_native_reader.cc:3624-3628), so parse
  // the nameless ones once to index them by that name.
  std::unordered_map<std::string, const ModelAsset*> modelassets;
  for (const auto& a : synth->assets) {
    if (!a) continue;
    for (const auto& ma : a->modelAssets) {
      if (!ma) continue;
      if (ma->name) {
        modelassets[*ma->name] = ma.get();
      } else if (ma->file && !ma->file->empty()) {
        std::filesystem::path cp =
            (std::filesystem::path(opts.base_dir) / *ma->file).lexically_normal();
        ps::mjcf::io::ParseResult pr = ps::mjcf::io::ParseMjcfFile(cp.string());
        if (pr.ok() && pr.model && pr.model->model && !pr.model->model->empty())
          modelassets[*pr.model->model] = ma.get();
      }
    }
  }

  // The child default trees graft as subclasses under synth's root default;
  // ensure it exists (a parent with no <default> gets an empty "main").
  Default* root = ps::sdk::detail::EnsureRoot(*synth);

  // Parsed children live for the whole expansion (grafts reference nothing from
  // them after cloning, but keep them alive through the walk).
  std::vector<std::unique_ptr<ps::mjcf::Model>> store;
  std::vector<KeyImport> keyimports;
  for (auto& wb : synth->worldbody)
    if (wb) ExpandAttachInSubtree(wb->subtree, /*in_replicate=*/false,
                                  opts.base_dir, root, *synth, modelassets, store,
                                  keyimports, reasons);

  // Resolve each imported keyframe's qpos placement now that every graft is
  // spliced. A child keyframe covers the graft's own dofs; it lands at the graft's
  // qposadr in the parent layout (the joints before it, depth-first) with the
  // surrounding dofs left at qpos0 (NaN gap -> qpos0 in FillKeyframes), mirroring
  // mjCModel::ResolveKeyframes/RestoreState. A macro before the graft (offset not
  // computable) or a key wider than the graft (child has non-grafted bodies)
  // routes the model to the XML fallback.
  for (const KeyImport& ki : keyimports) {
    // Locate the graft (depth-first) to get the qpos offset before it.
    int offset = 0;
    bool found = false, macro = false;
    for (auto& wb : synth->worldbody)
      if (wb && !found)
        WalkToGraftNq(wb->subtree, ki.graft, offset, found, macro);
    // graft nq via a direct subtree scan of the located node handled below
    auto push_gate = [&](const char* feat) {
      bridge::FallbackReason r;
      r.feature = feat;
      r.count = 1;
      r.first = ki.key->loc;
      reasons.push_back(std::move(r));
    };
    if (macro || !found) { push_gate("attach.keyframe_macro_offset"); continue; }

    // graft width: recompute over the graft node's own subtree.
    int gwidth = 0;
    bool wmacro = false;
    // The graft node is a Body or a Frame; scan whichever holds ki.graft.
    std::function<bool(const std::vector<BodyChildAny>&)> scan =
        [&](const std::vector<BodyChildAny>& sub) -> bool {
      for (const BodyChildAny& c : sub) {
        const void* p = nullptr;
        const std::vector<BodyChildAny>* inner = nullptr;
        if (c.kind() == BodyChildAny::Kind::Body) {
          const auto& b = std::get<std::unique_ptr<Body>>(c.node);
          p = b.get(); inner = &b->subtree;
        } else if (c.kind() == BodyChildAny::Kind::Frame) {
          const auto& f = std::get<std::unique_ptr<Frame>>(c.node);
          p = f.get(); inner = &f->subtree;
        }
        if (p == ki.graft) {
          if (c.kind() == BodyChildAny::Kind::Body)
            CountSubtreeNq(std::get<std::unique_ptr<Body>>(c.node)->subtree, gwidth, wmacro);
          else
            CountSubtreeNq(*inner, gwidth, wmacro);
          return true;
        }
        if (inner && scan(*inner)) return true;
      }
      return false;
    };
    for (auto& wb : synth->worldbody)
      if (wb) if (scan(wb->subtree)) break;

    std::vector<double>& q = *ki.key->qpos;
    if (static_cast<int>(q.size()) > gwidth) { push_gate("attach.keyframe_partial"); continue; }
    // Prepend `offset` NaN gap slots (default to qpos0 at fill time).
    if (offset > 0)
      q.insert(q.begin(), static_cast<std::size_t>(offset),
               std::numeric_limits<double>::quiet_NaN());
  }

  return synth;
}

std::vector<bridge::FallbackReason> CollectUnsupportedFeatures(const Model& m) {
  std::unordered_map<ElementType, FeatureUse> used;
  FeatureCollector collect(used);
  collect(m);

  // Collapse per-ElementType uses into per-feature-key reasons (distinct element
  // types can share a tag, e.g. structural vs sensor "site"): sum the counts,
  // keep the earliest source location.
  std::unordered_map<std::string, FeatureUse> by_key;
  for (const auto& [et, use] : used) {
    const std::string key = FeatureKey(et);
    if (IsFeatureSupported(key)) continue;
    FeatureUse& agg = by_key[key];
    if (agg.count == 0 || (use.first.line && !agg.first.line)) agg.first = use.first;
    agg.count += use.count;
  }

  // Finer-than-family sub-feature keys (e.g. geoms referencing meshes/materials).
  // These are never "supported" -- their presence forces the XML fallback.
  std::unordered_map<std::string, FeatureUse> sub;
  SubFeatureScanner gs(m, sub);
  gs(m);
  PartialArrayScanner pas(m, sub);
  pas(m);

  // Default-class <geom> subtrees are pruned by the scanners above (they are
  // templates, not compiled geoms), but a gated geom sub-feature authored in a
  // default (e.g. <default><geom fluidshape="ellipsoid">) still applies to every
  // geom that inherits it. Walk the default tree and gate those the same way.
  auto note_sub = [&](const char* key, const ps::SourceLoc& loc) {
    FeatureUse& u = sub[key];
    if (u.count == 0) u.first = loc;
    ++u.count;
  };
  std::function<void(const Default&)> scan_defaults = [&](const Default& d) {
    for (const auto& g : d.geom) {
      if (!g) continue;
      if (!g->plugin.empty()) note_sub("geom.plugin", g->loc);
    }
    for (const auto& sc : d.subclasses)
      if (sc) scan_defaults(*sc);
  };
  for (const auto& d : m.defaults)
    if (d) scan_defaults(*d);

  // Replicate gates that need whole-model context: a childclass in scope (the
  // native ParentMap can't reach the clones) and a model-level element that
  // references a name authored inside a replicate subtree (mjs_attach would
  // replicate the referencing element, which the native expansion does not).
  {
    std::set<std::string> rep_names;
    for (const auto& b : m.worldbody)
      if (b) CollectReplicateNames(b->subtree, false, rep_names);
    if (!rep_names.empty()) {
      std::set<std::string> section_refs;
      RefNameCollector rc(section_refs);
      for (const auto& t : m.tendons) if (t) rc(*t);
      for (const auto& a : m.actuators) if (a) rc(*a);
      for (const auto& s : m.sensors) if (s) rc(*s);
      for (const auto& e : m.equalitys) if (e) rc(*e);
      for (const auto& ct : m.contacts) if (ct) rc(*ct);
      bool refs_replicated = false;
      ps::SourceLoc ref_loc;
      for (const auto& nm : rep_names)
        if (section_refs.count(nm)) { refs_replicated = true; break; }
      if (refs_replicated) note_sub("replicate.referencing_element", ref_loc);
    }
    for (const auto& b : m.worldbody)
      if (b) ScanReplicateChildclass(b->subtree, false, note_sub);
  }

  // Flexcomp document-order hazard: a flexcomp inside an earlier child body must
  // get a lower flex id than a later flexcomp sibling, but the native collector
  // expands a body's own flexcomps first. All <worldbody> blocks merge into one
  // body, so the flag threads across them.
  {
    bool seen_body_fc = false;
    ps::SourceLoc loc;
    for (const auto& b : m.worldbody)
      if (b && FlexcompOrderHazard(b->subtree, seen_body_fc)) {
        note_sub("flexcomp.document_order", loc);
        break;
      }
  }
  for (const auto& [key, use] : sub) {
    FeatureUse& agg = by_key[key];
    if (agg.count == 0 || (use.first.line && !agg.first.line)) agg.first = use.first;
    agg.count += use.count;
  }

  std::vector<bridge::FallbackReason> out;
  for (const auto& [key, use] : by_key) {
    bridge::FallbackReason r;
    r.feature = key;
    r.count = use.count;
    r.first = use.first;
    out.push_back(std::move(r));
  }
  // Deterministic order (map iteration is unspecified): sort by feature key.
  std::sort(out.begin(), out.end(),
            [](const bridge::FallbackReason& a, const bridge::FallbackReason& b) {
              return a.feature < b.feature;
            });
  return out;
}

mjModel* NativeCompile(const Model& m, const bridge::CompileOptions& opts,
                       bridge::CompileReport& report) {
  (void)opts;
  report.taken = bridge::CompilePath::NativePath;

  // S0 gate: route or record fallback (CDR-2).
  std::vector<bridge::FallbackReason> unsupported = CollectUnsupportedFeatures(m);

  // Every family admitted: expand <attach> (NC6b) then run the build pipeline.
  if (unsupported.empty()) {
    // NC6b: splice each tractable <attach> into a synthetic model. A non-
    // tractable attach (child needing asset import / default merge / keyframe
    // resize) routes here with an explicit attach.* reason; a model with no
    // attach yields synth == nullptr and compiles unchanged.
    std::vector<bridge::FallbackReason> attach_reasons;
    std::unique_ptr<Model> synth = ExpandAttaches(m, opts, attach_reasons);
    if (!attach_reasons.empty()) {
      unsupported = std::move(attach_reasons);
    } else {
      const Model& build_m = synth ? *synth : m;
      // The grafted child body may use a feature the gate on the original tree
      // could not see (the child was an unparsed file ref); re-gate the synth.
      if (synth) unsupported = CollectUnsupportedFeatures(*synth);

      if (unsupported.empty()) {
        std::vector<bridge::Diagnostic> diags;
        mjModel* built = BuildNativeModel(build_m, opts, diags);
        if (built) {
          report.fallback_reasons.clear();
          return built;
        }
        // A build failure on an admitted model is a native-compiler bug:
        // surface the diagnostics and fall back (Auto) / hard-error (forced).
        for (auto& d : diags) report.errors.push_back(std::move(d));
        bridge::FallbackReason r;
        r.feature = "native.build_failed";
        r.count = 1;
        unsupported.push_back(std::move(r));
      }
    }
  }

  report.fallback_reasons = unsupported;

  // Build the UnsupportedNatively error: name the offending features so a forced
  // NativePath caller (and the harness) sees exactly what is missing.
  std::string msg = "native compile unsupported (UnsupportedNatively); features: ";
  for (std::size_t i = 0; i < unsupported.size(); ++i) {
    if (i) msg += ", ";
    msg += unsupported[i].feature;
    if (unsupported[i].count > 0)
      msg += "(x" + std::to_string(unsupported[i].count) + ")";
  }
  bridge::Diagnostic d;
  d.severity = bridge::Diagnostic::Severity::Error;
  d.pass = "gate";
  d.message = std::move(msg);
  report.errors.push_back(std::move(d));

  // NC1 seam: when `unsupported.empty()` and the pipeline exists, construct a
  //   CompileContext ctx{&m};
  // and run stages S1..S13, returning the built mjModel here.
  return nullptr;
}

}  // namespace ps::mjcf::compile
