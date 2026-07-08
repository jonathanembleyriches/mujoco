// Lifted mjModel allocator. See make_model.h for provenance and the plumbing
// deviations. Inside the MuJoCo quarantine zone (includes mujoco headers).

#include "make_model.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>
#include <mujoco/mjxmacro.h>

namespace ps::mjcf::compile::lifted {
namespace {

// Engine-internal constants (engine_io.h / engine_io.c), reproduced here.
constexpr int kLoadMultiple = 2;                 // mjLOAD_MULTIPLE
constexpr std::int64_t kMaxArraySize = INT_MAX;  // MAX_ARRAY_SIZE

// number of bytes to be skipped to achieve 64-byte alignment (verbatim: SKIP).
inline unsigned int Skip(std::intptr_t offset) {
  const unsigned int align = 64;
  return static_cast<unsigned int>((align - (offset % align)) % align);
}

// increases buffer size without causing integer overflow, returns 0 if the
// operation would overflow (verbatim from safeAddToBufferSize, MSVC branch --
// the portable overflow checks, no __builtin_*_overflow).
int SafeAddToBufferSize(std::intptr_t* offset, std::size_t* nbuffer,
                        std::size_t type_size, std::int64_t nr, std::int64_t nc) {
  if (nr < 0 || nc < 0) {
    return 0;
  }
  std::size_t product;
  std::size_t to_add;
  std::size_t skip = Skip(*offset);

  // nc * nr
  if (nr > 0 && static_cast<std::size_t>(nc) > SIZE_MAX / static_cast<std::size_t>(nr)) return 0;
  product = static_cast<std::size_t>(nc) * static_cast<std::size_t>(nr);

  // product * type_size
  if (type_size > 0 && product > SIZE_MAX / type_size) return 0;
  product *= type_size;

  // product + Skip(*offset)
  if (product > SIZE_MAX - skip) return 0;
  to_add = product + skip;

  // *nbuffer + to_add
  if (*nbuffer > SIZE_MAX - to_add) return 0;
  *nbuffer += to_add;

  // *offset + to_add
  if (*offset > 0 && to_add > static_cast<std::size_t>(INTPTR_MAX - *offset)) return 0;
  *offset += static_cast<std::intptr_t>(to_add);

  return 1;
}

// set pointers into the mjModel buffer (verbatim mj_setPtrModel).
void SetPtrModel(mjModel* m) {
  char* ptr = static_cast<char*>(m->buffer);

  // prepare symbols needed by xmacro (nc columns reference these locals).
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4244)  // mjtSize -> int narrowing in the preamble
#endif
  MJMODEL_POINTERS_PREAMBLE(m)
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
  (void)nq; (void)nv; (void)na; (void)nu; (void)nmocap;

  // assign pointers with padding.
#define X(type, name, nr, nc)                                            \
  m->name = (type*)(ptr + Skip((std::intptr_t)ptr));                     \
  ptr += Skip((std::intptr_t)ptr) + sizeof(type) * (m->nr) * (nc);

  MJMODEL_POINTERS
#undef X
  (void)ptr;
}

}  // namespace

mjModel* MakeModel(const mjModel& sizes) {
  mjModel* m = static_cast<mjModel*>(mju_malloc(sizeof(mjModel)));
  if (!m) {
    return nullptr;
  }
  std::memset(m, 0, sizeof(mjModel));

  // Copy every input size field. MJMODEL_SIZES enumerates all size ints; the
  // derived ones (nnames_map, nbuffer, and the post-build census: nJmom, narena,
  // ...) are zero in a freshly-sized model and are (re)computed here or later.
#define X(name) m->name = sizes.name;
  MJMODEL_SIZES
#undef X

  // validate: sizes non-negative and < INT_MAX, except the byte arrays.
#define X(name)                                                     \
  if (m->name < 0) { mju_free(m); return nullptr; }                 \
  if ((std::int64_t)m->name >= kMaxArraySize &&                     \
      std::strcmp(#name, "ntexdata") != 0 &&                        \
      std::strcmp(#name, "ntextdata") != 0) {                       \
    mju_free(m); return nullptr;                                    \
  }
  MJMODEL_SIZES
#undef X

  // nbody must be positive.
  if (m->nbody == 0) {
    mju_free(m);
    return nullptr;
  }

  // names hash-map size (verbatim term list).
  long nnames_map = (long)m->nbody + m->njnt + m->ngeom + m->nsite + m->ncam +
                    m->nlight + m->nflex + m->nmesh + m->nskin + m->nhfield +
                    m->ntex + m->nmat + m->npair + m->nexclude + m->neq +
                    m->ntendon + m->nu + m->nsensor + m->nnumeric + m->ntext +
                    m->ntuple + m->nkey + m->nplugin;
  if (nnames_map >= INT_MAX / kLoadMultiple) {
    mju_free(m);
    return nullptr;
  }
  m->nnames_map = kLoadMultiple * nnames_map;

  // compute buffer size (needs the preamble locals for the nc columns).
  std::intptr_t offset = 0;
  std::size_t nbuffer = 0;
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4244)
#endif
  MJMODEL_POINTERS_PREAMBLE(m)
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
  (void)nq; (void)nv; (void)na; (void)nu; (void)nmocap;
#define X(type, name, nr, nc)                                                    \
  if (!SafeAddToBufferSize(&offset, &nbuffer, sizeof(type), m->nr, nc)) {        \
    mju_free(m);                                                                 \
    return nullptr;                                                             \
  }
  MJMODEL_POINTERS
#undef X
  m->nbuffer = nbuffer;

  // allocate and clear the buffer, then wire the pointers.
  m->buffer = mju_malloc(m->nbuffer);
  if (!m->buffer) {
    mju_free(m);
    return nullptr;
  }
  std::memset(m->buffer, 0, m->nbuffer);
  SetPtrModel(m);

  // default sub-structs. mj_defaultOption/Visual are MJAPI; mj_defaultStatistic
  // is engine-internal (not exported), so its trivial body is inlined here.
  mj_defaultOption(&m->opt);
  mj_defaultVisual(&m->vis);
  m->stat.center[0] = m->stat.center[1] = m->stat.center[2] = 0;
  m->stat.extent = 2;
  m->stat.meaninertia = 1;
  m->stat.meanmass = 1;
  m->stat.meansize = 0.2;

  return m;
}

}  // namespace ps::mjcf::compile::lifted
