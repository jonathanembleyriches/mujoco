// ProtoSpec -> mjSpec builder (Wave 2 of the mjSpec shim). Constructs a
// throwaway mjSpec from an authored Model tree and hands it to mj_compile,
// bypassing the XML serialize/parse round trip. MuJoCo-linked: compiled only by
// the compile target, never the Tier-0 protospec library.
//
// The builder is stateless per compile: every call allocates a fresh mjSpec via
// mj_makeSpec, populates it, and the caller (compile.cc) owns it and deletes it
// with mj_deleteSpec after mj_compile. The tree is read const end to end.
//
// Parity contract: for any valid model the mjModel that mj_compile produces from
// the built spec is bit-identical to the one the XML path produces (ps_path_diff
// --path-a XmlPath --path-b MjsPath). The builder reproduces every family in
// process, including the macros (replicate expands via a reader-mirror; composite
// grafts a re-parsed fragment; flexcomp via mjs_makeFlex) and builtin meshes /
// URDF-MJB children.
#ifndef PROTOSPEC_COMPILE_MJS_BUILDER_H
#define PROTOSPEC_COMPILE_MJS_BUILDER_H

#include <string>
#include <vector>

#include "compile.h"   // CompileOptions
#include "mjcf.h"       // io::AutoNames
#include "report.h"     // FallbackReason
#include "types.h"

struct mjSpec_;
typedef struct mjSpec_ mjSpec;

namespace ps::mjcf::compile {

// Walks the tree for always-error guards: content that is invalid on BOTH
// compile paths and that the mjs build would otherwise drop silently (currently
// only coordinate="global"). Every former fallback family is built directly with
// full parity, so it no longer appears here. A non-empty result is a clean model
// error, not a route-to-XML signal.
std::vector<FallbackReason> MjsFallbackScan(const Model& model);

// Builds a fresh mjSpec from `model`. `auto_names` is the serial-keyed reserved
// name map the XML path injects (compile.cc builds it once, shared here so the
// Binding sees identical names on both paths). On success returns a spec the
// caller must mj_deleteSpec; on failure returns nullptr and fills `err`.
mjSpec* BuildSpec(const Model& model, const CompileOptions& opts,
                  const io::AutoNames& auto_names, std::string& err);

}  // namespace ps::mjcf::compile

#endif  // PROTOSPEC_COMPILE_MJS_BUILDER_H
