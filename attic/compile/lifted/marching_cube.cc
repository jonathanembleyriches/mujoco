// Upstream: MarchingCubeCpp (MC.h), the third-party single-header lib MuJoCo
// fetches for mjCMesh::LoadSDF. Registry id marching_cube (snapshots/lifted_code.json).
// Single instantiation of the vendored MarchingCubeCpp implementation. MC.h is a
// single-header lib whose function bodies are guarded by MC_IMPLEM_ENABLE; this
// TU is the one place that macro is defined, so the impl is compiled exactly
// once (every other includer of marching_cube.h sees declarations only). Kept as
// a thin wrapper so the vendored header stays byte-identical to upstream for the
// drift gate. Provenance: snapshots/lifted_code.json (MC::marching_cube).
// MC.h's impl body names uint64_t without including <cstdint>; MSVC's STL pulls it
// in transitively, libstdc++ does not. Included here rather than in the header,
// which must stay byte-identical to upstream.
#include <cstdint>

#define MC_IMPLEM_ENABLE
#include "marching_cube.h"
