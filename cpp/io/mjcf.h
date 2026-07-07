// ProtoSpec MJCF IO: the public read/write surface.
//
// ParseMjcf turns an MJCF document (tinyxml2 under the hood) into a ps::Model;
// WriteMjcf serializes a Model back to a canonical MJCF string. Only the Model-
// level blocks (compiler/option/size/statistic/visual) and the body tree
// (worldbody/body/frame/inertial/joint/freejoint/geom/site/camera/light) are
// understood so far; every other element is reported as a structured
// "unsupported element" diagnostic (Diagnostic::Kind::UnsupportedElement) that a
// harness can use to skip a file, kept distinct from malformed-input errors.
//
// This is the one module that links tinyxml2; the core ps library stays
// dependency-free.
#ifndef PROTOSPEC_IO_MJCF_H
#define PROTOSPEC_IO_MJCF_H

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "protospec/core.h"
#include "types.h"

namespace ps::mjcf::io {

// Serial-keyed auto-name overrides. An element whose creation serial is a key
// here is emitted with the mapped `name="..."` attribute even if its own name
// field is empty; the tree is never mutated. The compile bridge uses this to
// give unnamed elements deterministic reserved names for binding (DR-10), so
// the serialization-time naming lives here (in the serializer) with no tree
// edit. Elements already carrying an authored name are unaffected (the bridge
// only enters unnamed elements).
using AutoNames = std::unordered_map<std::uint64_t, std::string>;

// A single reader diagnostic, always carrying file:line provenance.
struct Diagnostic {
  enum class Kind {
    MalformedInput,     // syntax / unknown attribute / unknown element / bad value
    UnsupportedElement  // a well-formed element outside the supported families
  };

  Kind kind;
  std::string message;   // human-readable, prefixed with file:line on render
  ps::SourceLoc loc;     // {file, line}
  std::string element;   // for UnsupportedElement: the offending lowercase tag

  // "file:line: message" (line omitted when unknown).
  std::string Render() const;
};

// The outcome of a parse: a Model on success, else a list of diagnostics. NaN
// values raise non-fatal warnings (MuJoCo mju_warning parity) that do not fail
// the parse.
struct ParseResult {
  std::unique_ptr<Model> model;
  std::vector<Diagnostic> errors;
  std::vector<Diagnostic> warnings;

  bool ok() const { return errors.empty() && model != nullptr; }

  // True when at least one error is an unsupported-element report (and there are
  // no malformed-input errors): the document is well-formed but uses elements
  // outside the supported families. The harness skips such files.
  bool unsupported_only() const;
};

// Parse MJCF from an in-memory string (filename is used only for diagnostics).
ParseResult ParseMjcfString(const std::string& xml,
                            const std::string& filename = "<string>");

// Parse MJCF from a file on disk.
ParseResult ParseMjcfFile(const std::string& path);

// Serialize a Model to a deterministic MJCF string (2-space indent, radians,
// shortest round-trip numeric formatting). Emits exactly the authored fields.
std::string WriteMjcf(const Model& model);

// As above, but unnamed elements whose serial appears in `auto_names` are
// emitted with the mapped reserved name. Used by the compile bridge.
std::string WriteMjcf(const Model& model, const AutoNames& auto_names);

}  // namespace ps::mjcf::io

#endif  // PROTOSPEC_IO_MJCF_H
