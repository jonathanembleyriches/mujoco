// Lifted from MuJoCo: mjModel allocation (impl-plan T0.5 / ledger 3.7 alloc).
//
//   Upstream: src/engine/engine_io.c  (mj_makeModel, mj_setPtrModel,
//             safeAddToBufferSize, SKIP)
//   Pin:      mjVERSION_HEADER 3010000 (MuJoCo 3.10.0)
//   License:  Apache-2.0, (c) DeepMind Technologies Limited. See NOTICE.
//   Registry: snapshots/lifted_code.json entry "make_model".
//
// Why lifted, not linked: mj_makeModel is declared without MJAPI in the engine-
// internal engine_io.h and is not exported from mujoco.dll (Open Q2, resolved in
// native.cc). The lift is layout-faithful because the buffer is laid out purely
// by the PUBLIC include/mujoco/mjxmacro.h X-macros, so it picks up new mjModel
// fields structurally across bumps.
//
// Plumbing deviation from upstream (noted in the registry): the ~80 positional
// size parameters of mj_makeModel are replaced by reading the n* fields from a
// caller-provided mjModel; the allocate-into-existing-dest (reuse) branch and
// its freeModelBuffers path are dropped (we always allocate fresh). The buffer-
// size accumulation, alignment padding, and pointer setup are verbatim.
#ifndef PROTOSPEC_COMPILE_LIFTED_MAKE_MODEL_H
#define PROTOSPEC_COMPILE_LIFTED_MAKE_MODEL_H

struct mjModel_;
typedef struct mjModel_ mjModel;

namespace ps::mjcf::compile::lifted {

// Allocate an mjModel whose n* size fields equal those of `sizes` (only size
// fields are read). Returns a freshly allocated, zero-filled, pointer-wired
// mjModel with default option/visual/statistic, or nullptr if the sizes are
// invalid (negative, overflow, or nbody == 0) or allocation fails. The result
// is layout-identical to one from the engine's own mj_makeModel: it can be
// mj_copyModel'd and must be freed with mj_deleteModel.
mjModel* MakeModel(const mjModel& sizes);

}  // namespace ps::mjcf::compile::lifted

#endif  // PROTOSPEC_COMPILE_LIFTED_MAKE_MODEL_H
