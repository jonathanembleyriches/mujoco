// ProtoSpec compile report: the structured outcome of a Compile / Recompile.
//
// One Diagnostic type carries every error and warning with SourceLoc
// provenance; FallbackReason records why a requested path was not taken; the
// CompileReport aggregates them. This header is MuJoCo-free (it is included by
// the public compile.h and pulls in nothing from mujoco.h), so downstream code
// can inspect results without linking the engine.
//
// Stability: CompileReport is append-only (native-compiler impl-plan risk I3).
// Fields are never removed or renamed, enum values never renumbered; Render()
// text is non-contractual.
#ifndef PROTOSPEC_BRIDGE_REPORT_H
#define PROTOSPEC_BRIDGE_REPORT_H

#include <string>
#include <vector>

#include "protospec/core.h"
#include "protospec/diag.h"

namespace ps::mjcf {

// Compile diagnostics are the shared ps::Diagnostic (protospec/diag.h): `source`
// is the stage that raised it ("serialize", "load", "bind", "validate", ...).

// Why a requested compile path was not taken (impl-plan CDR-2).
struct FallbackReason {
  std::string feature;   // manifest feature key, e.g. "native.unimplemented"
  int count = 0;
  ps::SourceLoc first;
};

// Which compile path was requested / actually taken. NativePath exists so the
// contract is stable, but the native implementation lands later: requesting it
// today yields an UnsupportedNatively error (no silent fallback). MjsPath builds
// a throwaway mjSpec and calls mj_compile. Auto prefers MjsPath but falls back to
// XmlPath for a model the mjSpec API cannot reproduce (the fallback scan's
// "fallbackable" guards: flexcomp pin/direct/material-texcoord) -- report.taken
// then reads XmlPath and fallback_reasons carries the why. An "always-error"
// guard (coordinate=global) surfaces a clean model error instead. XmlPath stays
// the oracle, reachable when explicitly forced. Auto never routes to NativePath.
enum class CompilePath { Auto, XmlPath, NativePath, MjsPath };

struct CompileReport {
  CompilePath requested = CompilePath::Auto;
  CompilePath taken = CompilePath::XmlPath;         // the ratchet keys off this
  std::vector<FallbackReason> fallback_reasons;     // empty when taken == requested
  std::vector<ps::Diagnostic> errors;
  std::vector<ps::Diagnostic> warnings;             // surfaced, never swallowed

  bool ok() const { return errors.empty(); }
};

}  // namespace ps::mjcf

#endif  // PROTOSPEC_BRIDGE_REPORT_H
