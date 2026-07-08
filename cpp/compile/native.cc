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
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <mujoco/mujoco.h>

#include "context.h"
#include "native_supported.h"
#include "reflect.h"
#include "visit.h"

namespace ps::mjcf::compile {
namespace {

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

  // Until NC1 lands the stage pipeline, even a model whose every feature were
  // admitted cannot be built (no stages exist). Record that honestly so a
  // hypothetical fully-admitted model still falls back rather than falsely
  // claiming a native compile. Today the supported set is empty, so `unsupported`
  // is non-empty for any non-trivial model and this branch adds a pipeline note
  // only for the degenerate empty model.
  if (unsupported.empty()) {
    bridge::FallbackReason r;
    r.feature = "native.pipeline_unimplemented";
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
