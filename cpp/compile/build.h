// Native build pipeline (impl-plan T1.x / stages S1..S13). Given a const Model
// whose every feature the CDR-2 gate has already admitted, BuildNativeModel runs
// the compile stages against a CompileContext and returns a freshly allocated
// mjModel (caller owns it; free with mj_deleteModel) that is bit-identical to the
// XML oracle path, or nullptr with error diagnostics appended to `diags`.
//
// This header forward-declares mjModel and never includes mujoco.h; the
// implementation (build.cc) is inside the MuJoCo quarantine zone.
#ifndef PROTOSPEC_COMPILE_BUILD_H
#define PROTOSPEC_COMPILE_BUILD_H

#include <vector>

#include "compile.h"  // bridge::CompileOptions
#include "report.h"   // bridge::Diagnostic
#include "types.h"    // ps::mjcf::Model

struct mjModel_;
typedef struct mjModel_ mjModel;

namespace ps::mjcf::compile {

// Run S1..S13 over `m`. `opts` supplies the auto-naming policy (mirrored from
// the XML path so name tables and mj_name2id stay bit-identical). On success
// returns the built mjModel; on an internal error (which for an admitted model
// is a native-compiler bug) returns nullptr and appends an Error diagnostic.
mjModel* BuildNativeModel(const Model& m, const bridge::CompileOptions& opts,
                          std::vector<bridge::Diagnostic>& diags);

}  // namespace ps::mjcf::compile

#endif  // PROTOSPEC_COMPILE_BUILD_H
