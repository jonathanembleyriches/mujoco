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

namespace ps::mjcf {

namespace detail {
inline std::string RenderLoc(const ps::SourceLoc& loc) {
  if (loc.file.empty()) return "";
  std::string s = loc.file;
  if (loc.line > 0) {
    s += ':';
    s += std::to_string(loc.line);
  }
  s += ": ";
  return s;
}
}  // namespace detail

// A single diagnostic. Mirrors the impl-plan's CDR-9 one-type-for-all shape so
// the eventual native path and this XML path report identically.
struct Diagnostic {
  enum class Severity { Error, Warning };

  Severity severity = Severity::Error;
  std::string pass;      // stage that raised it ("serialize", "load", "bind", ...)
  std::string message;
  ps::SourceLoc loc;     // empty file for programmatic elements

  // "file:line: [pass] message" (file/line omitted when unknown).
  std::string Render() const {
    std::string s = detail::RenderLoc(loc);
    if (!pass.empty()) {
      s += '[';
      s += pass;
      s += "] ";
    }
    s += message;
    return s;
  }
};

// Why a requested compile path was not taken (impl-plan CDR-2).
struct FallbackReason {
  std::string feature;   // manifest feature key, e.g. "native.unimplemented"
  int count = 0;
  ps::SourceLoc first;
};

// Which compile path was requested / actually taken. NativePath exists so the
// contract is stable, but the native implementation lands later: requesting it
// today yields an UnsupportedNatively error (no silent fallback).
enum class CompilePath { Auto, XmlPath, NativePath };

struct CompileReport {
  CompilePath requested = CompilePath::Auto;
  CompilePath taken = CompilePath::XmlPath;         // the ratchet keys off this
  std::vector<FallbackReason> fallback_reasons;     // empty when taken == requested
  std::vector<Diagnostic> errors;
  std::vector<Diagnostic> warnings;                 // surfaced, never swallowed

  bool ok() const { return errors.empty(); }
};

}  // namespace ps::mjcf

#endif  // PROTOSPEC_BRIDGE_REPORT_H
