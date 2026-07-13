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
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <mujoco/mujoco.h>

#include "build.h"
#include "classes.h"   // ps::sdk defaults resolution (partial-array detector)
#include "context.h"
#include "native_supported.h"
#include "reflect.h"
#include "visit.h"

namespace ps::mjcf::compile {
namespace {

// The nearest class in a geom/site's governing chain that authors `size`, and
// how many elements it authors. MuJoCo eager-copies a class's full size array
// then overwrites only the authored prefix of the element (ReadAttr copies just
// the present values), so a geom/site that authors FEWER size values than its
// class default provides inherits the class's tail -- a semantic ProtoSpec's
// whole-field presence merge (CDR-5) does not reproduce. Detect and fall back.
template <class T>
int ClassSizeLen(const Model& m, const ps::sdk::detail::DefaultIndex& idx,
                 const ps::sdk::ParentMap& pm, const T& e) {
  std::string cls = ps::sdk::detail::ResolveClassName(pm, ps::sdk::detail::OwnClass(e), &e);
  for (const ps::mjcf::Default* d = idx.ByNameOrRoot(cls); d; d = idx.ParentOf(d)) {
    const auto* vec = ps::sdk::DefaultVec<T>(*d);
    if (vec && !vec->empty() && vec->front() && vec->front()->size)
      return static_cast<int>(vec->front()->size->size());
  }
  return 0;
}

template <class T>
bool PartialSizeInherit(const Model& m, const ps::sdk::detail::DefaultIndex& idx,
                        const ps::sdk::ParentMap& pm, const T& e) {
  const int authored = e.size ? static_cast<int>(e.size->size()) : -1;
  if (authored < 0 || authored >= 3) return false;  // full or unauthored: no tail
  return authored < ClassSizeLen(m, idx, pm, e);
}

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
      else if constexpr (std::is_same_v<E, Body>) CheckBody(e);
      else if constexpr (std::is_same_v<E, Texture>) CheckTexture(e);
      else if constexpr (std::is_same_v<E, Hfield>) CheckHfield(e);
      else if constexpr (std::is_same_v<E, Light>) CheckLight(e);
      else if constexpr (std::is_same_v<E, Joint>) CheckJoint(e);
      else if constexpr (std::is_same_v<E, FreeJoint>) CheckFreeJoint(e);
      else if constexpr (std::is_same_v<E, Spatial>) CheckSpatial(e);
      else if constexpr (std::is_same_v<E, Fixed>) CheckFixed(e);
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
      else if constexpr (std::is_same_v<E, EqualityFlex>) CheckEqualityFlex(e);
      else if constexpr (std::is_same_v<E, SensorContact>) CheckSensorContact(e);
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
  // Only builtin procedural textures (gradient/checker/flat, no cube-face files)
  // are native. A file texture (source=TexFile), cube-separate faces, or a
  // builtin="none" placeholder route the whole model to the XML fallback until
  // file-texture loading lands.
  void CheckTexture(const Texture& t) {
    const bool has_cube = t.fileright || t.fileleft || t.fileup || t.filedown ||
                          t.filefront || t.fileback;
    const TextureBuiltin* b =
        t.source ? std::get_if<TextureBuiltin>(&*t.source) : nullptr;
    const bool is_builtin = b && *b != TextureBuiltin::none;
    if (!is_builtin || has_cube) Note("texture.file", t.loc);
  }
  // Only user-data (inline elevation) hfields are native; a file (PNG/custom)
  // hfield needs image loading, a later wave -> XML fallback.
  void CheckHfield(const Hfield& h) {
    if (h.file) Note("hfield.file", h.loc);
  }
  // A light's texture would need a compiled asset (not native yet).
  void CheckLight(const Light& l) {
    if (l.texture) Note("light.texture", l.loc);
  }
  // Free-joint alignment (align="true", or align="auto" with compiler
  // alignfree) rewrites the body/inertial frames and child-geom poses -- out of
  // the NC1b rigid-body slice, so route such a model to the XML fallback.
  void CheckFreeJoint(const FreeJoint& fj) {
    if (fj.align && *fj.align == TriState::true_) Note("freejoint.align", fj.loc);
  }
  // springdamper drives the AutoSpringDamper post-build pass (stiffness/damping
  // from dof_invweight0), which the native path does not yet run: route to the
  // XML fallback until it lands.
  void CheckJoint(const Joint& j) {
    if (j.springdamper) Note("joint.springdamper", j.loc);
  }
  // A non-default per-body sleep policy sets tree_sleep_policy, which the native
  // path does not yet emit (it writes mjSLEEP_AUTO for every tree).
  void CheckBody(const Body& b) {
    if (b.sleep) Note("body.sleep", b.loc);
  }
  // Tendon armature drives a sparse-size (nC/body_simple) demotion the native
  // path does not yet model; route such tendons to the XML fallback.
  void CheckSpatial(const Spatial& s) {
    if (s.armature && *s.armature != 0) Note("tendon.armature", s.loc);
  }
  void CheckFixed(const Fixed& f) {
    if (f.armature && *f.armature != 0) Note("tendon.armature", f.loc);
  }
  // The native path resolves joint/jointinparent/tendon/body transmission only,
  // has no history/delay buffer, and does not run mj_setLengthRange, so site /
  // refsite / slidercrank transmission, a delay buffer, or a length-range-needing
  // gain/bias type (muscle/user) route to the XML fallback.
  template <class E>
  void CheckActuator(const E& a) {
    if constexpr (requires { a.site; })
      if (a.site) Note("actuator.site_transmission", a.loc);
    if constexpr (requires { a.refsite; })
      if (a.refsite) Note("actuator.site_transmission", a.loc);
    if constexpr (requires { a.cranksite; })
      if (a.cranksite) Note("actuator.slidercrank", a.loc);
    if constexpr (requires { a.slidersite; })
      if (a.slidersite) Note("actuator.slidercrank", a.loc);
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
      if (a.gaintype && (*a.gaintype == GainType::muscle ||
                         *a.gaintype == GainType::user ||
                         *a.gaintype == GainType::dcmotor))
        Note("actuator.gaintype", a.loc);
      if (a.biastype && (*a.biastype == BiasType::muscle ||
                         *a.biastype == BiasType::user ||
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
  // The contact sensor shares the "contact" tag with the <contact> section
  // (admitted for pairs/excludes), so it slips past the family gate; its
  // variable dim + intprm bitmask are out of NC2 scope. Force the fallback.
  void CheckSensorContact(const SensorContact& s) {
    Note("sensor.contact", s.loc);
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
    if (eff->mesh && type != static_cast<int>(GeomType::mesh) &&
        type != static_cast<int>(GeomType::sdf))
      Note("geom.mesh_fit", g.loc);
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
      if constexpr (std::is_same_v<E, Geom>) {
        if (PartialSizeInherit(m_, idx_, pm_, e)) Note("geom.partial_size_default", e.loc);
      } else if constexpr (std::is_same_v<E, Site>) {
        if (PartialSizeInherit(m_, idx_, pm_, e)) Note("site.partial_size_default", e.loc);
      } else if constexpr (std::is_same_v<E, ActuatorGeneral> ||
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

  // Every feature admitted: run the S1..S13 build pipeline. On success return
  // the built model with report.taken == NativePath.
  if (unsupported.empty()) {
    std::vector<bridge::Diagnostic> diags;
    mjModel* built = BuildNativeModel(m, opts, diags);
    if (built) {
      report.fallback_reasons.clear();
      return built;
    }
    // A build failure on an admitted model is a native-compiler bug: surface the
    // diagnostics and fall back (Auto) / hard-error (forced NativePath).
    for (auto& d : diags) report.errors.push_back(std::move(d));
    bridge::FallbackReason r;
    r.feature = "native.build_failed";
    r.count = 1;
    unsupported.push_back(std::move(r));
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
