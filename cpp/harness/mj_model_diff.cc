// mj_model_diff: structurally compare two mjModel structs compiled from MJCF.
//
// Usage: mj_model_diff a.xml b.xml [--json] [--tol T] [--examples N]
//
// Exit codes:
//   0  models are identical within tolerance
//   2  models differ
//   1  a load error occurred (stderr names the file and MuJoCo's message)
//
// The comparison is driven entirely by MuJoCo's own mjxmacro tables so that
// coverage is total and survives MuJoCo version bumps: MJMODEL_SIZES enumerates
// every size int, MJMODEL_POINTERS enumerates every array field. A handful of
// mjData-derived invariants (forward kinematics at qpos0) catch semantic
// divergences that raw size/array equality can miss.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include <mujoco/mujoco.h>
#include <mujoco/mjxmacro.h>

namespace {

struct Tol {
  double rtol = 1e-9;
  double atol = 1e-9;
};

// A single field that differed, with a bounded sample of offending indices.
struct FieldDiff {
  std::string field;
  std::string note;              // e.g. size mismatch, otherwise empty
  int64_t count_a = 0;
  int64_t count_b = 0;
  int64_t num_diff = 0;
  struct Example {
    int64_t index;
    double a;
    double b;
  };
  std::vector<Example> examples;
};

struct SizeDiff {
  std::string name;
  int64_t a;
  int64_t b;
};

struct NameDiff {
  std::string objtype;
  int id;
  std::string a;
  std::string b;
};

struct Report {
  std::vector<SizeDiff> sizes;
  std::vector<NameDiff> names;
  std::vector<FieldDiff> fields;
  std::vector<FieldDiff> invariants;  // mjData-derived (xpos/xquat)
  bool sizes_equal = true;

  bool Differs() const {
    return !sizes.empty() || !names.empty() || !fields.empty() ||
           !invariants.empty();
  }
};

bool NearlyEqual(double a, double b, const Tol& tol) {
  if (a == b) return true;
  if (std::isnan(a) && std::isnan(b)) return true;  // NaN sentinels match
  if (std::isnan(a) || std::isnan(b)) return false;
  double diff = std::fabs(a - b);
  double scale = std::fmax(std::fabs(a), std::fabs(b));
  return diff <= tol.atol + tol.rtol * scale;
}

// Element-wise comparison of one array field. Floating types use the rel+abs
// tolerance; integral/byte types compare exactly. Assumes count_a == count_b
// (the caller only reaches array fields when all sizes matched).
template <typename T>
void CompareField(const char* name, const T* a, const T* b, int64_t count,
                  const Tol& tol, int max_examples, Report& report) {
  if (count <= 0 || a == nullptr || b == nullptr) return;
  FieldDiff fd;
  fd.field = name;
  fd.count_a = count;
  fd.count_b = count;
  for (int64_t i = 0; i < count; ++i) {
    bool eq;
    if constexpr (std::is_floating_point_v<T>) {
      eq = NearlyEqual(static_cast<double>(a[i]), static_cast<double>(b[i]),
                       tol);
    } else {
      eq = (a[i] == b[i]);
    }
    if (!eq) {
      ++fd.num_diff;
      if (static_cast<int>(fd.examples.size()) < max_examples) {
        fd.examples.push_back({i, static_cast<double>(a[i]),
                               static_cast<double>(b[i])});
      }
    }
  }
  if (fd.num_diff > 0) report.fields.push_back(std::move(fd));
}

struct ObjClass {
  const char* label;
  int type;
  int64_t count;
};

void CompareNames(const mjModel* a, const mjModel* b, Report& report) {
  const ObjClass classes[] = {
      {"body", mjOBJ_BODY, a->nbody},        {"joint", mjOBJ_JOINT, a->njnt},
      {"geom", mjOBJ_GEOM, a->ngeom},        {"site", mjOBJ_SITE, a->nsite},
      {"camera", mjOBJ_CAMERA, a->ncam},     {"light", mjOBJ_LIGHT, a->nlight},
      {"flex", mjOBJ_FLEX, a->nflex},        {"mesh", mjOBJ_MESH, a->nmesh},
      {"skin", mjOBJ_SKIN, a->nskin},        {"hfield", mjOBJ_HFIELD, a->nhfield},
      {"texture", mjOBJ_TEXTURE, a->ntex},   {"material", mjOBJ_MATERIAL, a->nmat},
      {"pair", mjOBJ_PAIR, a->npair},        {"exclude", mjOBJ_EXCLUDE, a->nexclude},
      {"equality", mjOBJ_EQUALITY, a->neq},  {"tendon", mjOBJ_TENDON, a->ntendon},
      {"actuator", mjOBJ_ACTUATOR, a->nu},   {"sensor", mjOBJ_SENSOR, a->nsensor},
      {"numeric", mjOBJ_NUMERIC, a->nnumeric}, {"text", mjOBJ_TEXT, a->ntext},
      {"tuple", mjOBJ_TUPLE, a->ntuple},     {"key", mjOBJ_KEY, a->nkey},
      {"plugin", mjOBJ_PLUGIN, a->nplugin},
  };
  for (const auto& c : classes) {
    for (int64_t id = 0; id < c.count; ++id) {
      const char* na = mj_id2name(a, c.type, static_cast<int>(id));
      const char* nb = mj_id2name(b, c.type, static_cast<int>(id));
      std::string sa = na ? na : "";
      std::string sb = nb ? nb : "";
      if (sa != sb)
        report.names.push_back({c.label, static_cast<int>(id), sa, sb});
    }
  }
}

void CompareSizes(const mjModel* a, const mjModel* b, Report& report) {
#define X(name)                                                    \
  if (static_cast<int64_t>(a->name) != static_cast<int64_t>(b->name)) \
    report.sizes.push_back(                                        \
        {#name, static_cast<int64_t>(a->name), static_cast<int64_t>(b->name)});
  MJMODEL_SIZES
#undef X
  report.sizes_equal = report.sizes.empty();
}

void CompareFields(const mjModel* a, const mjModel* b, const Tol& tol,
                   int max_examples, Report& report) {
  const mjModel* m = a;  // sizes are equal here; row counts read from a
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4244)  // MuJoCo's preamble narrows mjtSize -> int
#endif
  MJMODEL_POINTERS_PREAMBLE(m)
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
  (void)nq;
  (void)nv;
  (void)na;
  (void)nu;
  (void)nmocap;
#define X(type, name, nr, nc)                                             \
  CompareField<type>(#name, a->name, b->name,                            \
                     static_cast<int64_t>(m->nr) * static_cast<int64_t>(nc), \
                     tol, max_examples, report);
  MJMODEL_POINTERS
#undef X
}

// Forward-kinematics invariant: mj_makeData initializes qpos to qpos0, and
// mj_forward then fills xpos/xquat. Comparing those exposes orientation, unit,
// and frame bugs that identical array sizes would hide.
bool CompareInvariants(const mjModel* a, const mjModel* b, const Tol& tol,
                       int max_examples, Report& report, std::string& err) {
  mjData* da = mj_makeData(a);
  mjData* db = mj_makeData(b);
  if (!da || !db) {
    if (da) mj_deleteData(da);
    if (db) mj_deleteData(db);
    err = "mj_makeData failed";
    return false;
  }
  mj_forward(a, da);
  mj_forward(b, db);
  if (a->nbody == b->nbody) {
    CompareField<mjtNum>("xpos", da->xpos, db->xpos,
                         static_cast<int64_t>(a->nbody) * 3, tol, max_examples,
                         report);
    CompareField<mjtNum>("xquat", da->xquat, db->xquat,
                         static_cast<int64_t>(a->nbody) * 4, tol, max_examples,
                         report);
    // Move any xpos/xquat findings from fields into invariants.
    for (auto it = report.fields.begin(); it != report.fields.end();) {
      if (it->field == "xpos" || it->field == "xquat") {
        report.invariants.push_back(std::move(*it));
        it = report.fields.erase(it);
      } else {
        ++it;
      }
    }
  }
  mj_deleteData(da);
  mj_deleteData(db);
  return true;
}

std::string JsonEscape(const std::string& s) {
  std::string out;
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\t': out += "\\t"; break;
      default: out += c;
    }
  }
  return out;
}

void PrintFieldExamples(const FieldDiff& fd) {
  std::printf("  %-28s %lld/%lld differ", fd.field.c_str(),
              static_cast<long long>(fd.num_diff),
              static_cast<long long>(fd.count_a));
  if (!fd.note.empty()) std::printf(" (%s)", fd.note.c_str());
  std::printf("\n");
  for (const auto& e : fd.examples) {
    std::printf("      [%lld]  a=%.17g  b=%.17g\n",
                static_cast<long long>(e.index), e.a, e.b);
  }
}

void PrintText(const mjModel*, const mjModel*, const Report& report,
               const char* fa, const char* fb) {
  if (!report.Differs()) {
    std::printf("IDENTICAL\n  %s\n  %s\n", fa, fb);
    return;
  }
  std::printf("DIFFER\n  a: %s\n  b: %s\n", fa, fb);
  if (!report.sizes.empty()) {
    std::printf("\nSIZE INTS (%zu differ):\n", report.sizes.size());
    for (const auto& s : report.sizes) {
      std::printf("  %-20s a=%lld  b=%lld\n", s.name.c_str(),
                  static_cast<long long>(s.a), static_cast<long long>(s.b));
    }
  }
  if (!report.names.empty()) {
    std::printf("\nNAME TABLES (%zu differ):\n", report.names.size());
    for (const auto& n : report.names) {
      std::printf("  %s[%d]  a=\"%s\"  b=\"%s\"\n", n.objtype.c_str(), n.id,
                  n.a.c_str(), n.b.c_str());
    }
  }
  if (!report.invariants.empty()) {
    std::printf("\nFORWARD-KINEMATICS INVARIANTS (%zu differ):\n",
                report.invariants.size());
    for (const auto& fd : report.invariants) PrintFieldExamples(fd);
  }
  if (!report.sizes_equal) {
    std::printf(
        "\nARRAY FIELDS: skipped (size ints differ; align sizes first)\n");
  } else if (!report.fields.empty()) {
    std::printf("\nARRAY FIELDS (%zu differ):\n", report.fields.size());
    for (const auto& fd : report.fields) PrintFieldExamples(fd);
  }
}

void PrintFieldJson(const FieldDiff& fd, bool& first) {
  if (!first) std::printf(",");
  first = false;
  std::printf(
      "\n    {\"field\":\"%s\",\"num_diff\":%lld,\"count\":%lld,\"examples\":[",
      JsonEscape(fd.field).c_str(), static_cast<long long>(fd.num_diff),
      static_cast<long long>(fd.count_a));
  bool ef = true;
  for (const auto& e : fd.examples) {
    if (!ef) std::printf(",");
    ef = false;
    std::printf("{\"index\":%lld,\"a\":%.17g,\"b\":%.17g}",
                static_cast<long long>(e.index), e.a, e.b);
  }
  std::printf("]}");
}

void PrintJson(const Report& report, const char* fa, const char* fb) {
  std::printf("{\n");
  std::printf("  \"a\":\"%s\",\n  \"b\":\"%s\",\n", JsonEscape(fa).c_str(),
              JsonEscape(fb).c_str());
  std::printf("  \"identical\":%s,\n", report.Differs() ? "false" : "true");
  std::printf("  \"sizes\":[");
  bool first = true;
  for (const auto& s : report.sizes) {
    if (!first) std::printf(",");
    first = false;
    std::printf("\n    {\"name\":\"%s\",\"a\":%lld,\"b\":%lld}", s.name.c_str(),
                static_cast<long long>(s.a), static_cast<long long>(s.b));
  }
  std::printf("],\n  \"names\":[");
  first = true;
  for (const auto& n : report.names) {
    if (!first) std::printf(",");
    first = false;
    std::printf(
        "\n    {\"objtype\":\"%s\",\"id\":%d,\"a\":\"%s\",\"b\":\"%s\"}",
        n.objtype.c_str(), n.id, JsonEscape(n.a).c_str(),
        JsonEscape(n.b).c_str());
  }
  std::printf("],\n  \"invariants\":[");
  first = true;
  for (const auto& fd : report.invariants) PrintFieldJson(fd, first);
  std::printf("],\n  \"fields\":[");
  first = true;
  for (const auto& fd : report.fields) PrintFieldJson(fd, first);
  std::printf("]\n}\n");
}

mjModel* Load(const char* path, std::string& err) {
  char errbuf[1024] = {0};
  mjModel* m = mj_loadXML(path, nullptr, errbuf, sizeof(errbuf));
  if (!m) err = errbuf[0] ? errbuf : "unknown load error";
  return m;
}

}  // namespace

int main(int argc, char** argv) {
  const char* fa = nullptr;
  const char* fb = nullptr;
  bool json = false;
  Tol tol;
  int max_examples = 8;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--json") {
      json = true;
    } else if (arg == "--tol" && i + 1 < argc) {
      double t = std::atof(argv[++i]);
      tol.rtol = t;
      tol.atol = t;
    } else if (arg == "--atol" && i + 1 < argc) {
      tol.atol = std::atof(argv[++i]);
    } else if (arg == "--rtol" && i + 1 < argc) {
      tol.rtol = std::atof(argv[++i]);
    } else if (arg == "--examples" && i + 1 < argc) {
      max_examples = std::atoi(argv[++i]);
    } else if (!fa) {
      fa = argv[i];
    } else if (!fb) {
      fb = argv[i];
    } else {
      std::fprintf(stderr, "unexpected argument: %s\n", arg.c_str());
      return 1;
    }
  }

  if (!fa || !fb) {
    std::fprintf(stderr,
                 "usage: mj_model_diff a.xml b.xml [--json] [--tol T] "
                 "[--atol A] [--rtol R] [--examples N]\n");
    return 1;
  }

  std::string err;
  mjModel* a = Load(fa, err);
  if (!a) {
    std::fprintf(stderr, "load error in a (%s): %s\n", fa, err.c_str());
    return 1;
  }
  mjModel* b = Load(fb, err);
  if (!b) {
    std::fprintf(stderr, "load error in b (%s): %s\n", fb, err.c_str());
    mj_deleteModel(a);
    return 1;
  }

  Report report;
  CompareSizes(a, b, report);
  CompareNames(a, b, report);
  if (report.sizes_equal) {
    CompareFields(a, b, tol, max_examples, report);
  }
  std::string inv_err;
  if (!CompareInvariants(a, b, tol, max_examples, report, inv_err)) {
    std::fprintf(stderr, "invariant check failed: %s\n", inv_err.c_str());
    mj_deleteModel(a);
    mj_deleteModel(b);
    return 1;
  }

  if (json) {
    PrintJson(report, fa, fb);
  } else {
    PrintText(a, b, report, fa, fb);
  }

  int exit_code = report.Differs() ? 2 : 0;
  mj_deleteModel(a);
  mj_deleteModel(b);
  return exit_code;
}
