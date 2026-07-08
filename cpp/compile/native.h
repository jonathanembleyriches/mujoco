// Native compiler entry point (impl-plan T0.3 / Section 1). NativeCompile walks
// a ProtoSpec Model and -- when the CDR-2 feature gate admits every feature it
// uses -- builds an mjModel directly, no XML, no mjSpec. Today the gate admits
// nothing (native_supported.h is empty), so NativeCompile always reports the
// model UnsupportedNatively and returns null; the bridge dispatch falls back to
// the XML oracle path. The scaffolding (gate walk, CompileContext side tables,
// stage pipeline seam) is in place for NC1 to grow the supported set.
//
// This lives in namespace ps::mjcf::compile and plugs in BEHIND the M5 bridge
// surface (ps::mjcf::bridge). It reuses the bridge's CompileOptions and
// CompileReport types so a caller sees one contract regardless of path taken.
//
// Purity (CDR-14): NativeCompile takes `const Model&` end to end and never
// mutates the tree. All intermediate state lives in a stack-owned
// CompileContext (context.h).
//
// This header forward-declares mjModel and never includes mujoco.h.
#ifndef PROTOSPEC_COMPILE_NATIVE_H
#define PROTOSPEC_COMPILE_NATIVE_H

#include <vector>

#include "compile.h"   // bridge::CompileOptions
#include "report.h"    // bridge::CompileReport, FallbackReason, CompilePath
#include "types.h"     // ps::mjcf::Model

struct mjModel_;
typedef struct mjModel_ mjModel;

namespace ps::mjcf::compile {

// The CDR-2 gate: walk `m` and return one FallbackReason per feature it uses
// that the native path cannot yet compile (deduplicated by feature key, with a
// use count and the first SourceLoc). Empty result == the whole model is
// natively supported. Exposed so the differential harness and the purity/gate
// tests can query support without running a compile.
std::vector<bridge::FallbackReason> CollectUnsupportedFeatures(const Model& m);

// Compile `m` to an mjModel via the native path. On success returns a freshly
// allocated mjModel (caller owns it; free with mj_deleteModel) and sets
// `report.taken == NativePath`. On an unsupported feature (all models today)
// returns nullptr, sets `report.taken == NativePath`, fills
// `report.fallback_reasons` with the gate's findings, and adds one
// UnsupportedNatively error diagnostic. Never throws.
//
// This is the forced-native entry: it does NOT fall back to XML. The bridge's
// Auto dispatch is what turns a null-with-fallback-reasons result into an XML
// compile (compile.cc).
mjModel* NativeCompile(const Model& m, const bridge::CompileOptions& opts,
                       bridge::CompileReport& report);

}  // namespace ps::mjcf::compile

#endif  // PROTOSPEC_COMPILE_NATIVE_H
