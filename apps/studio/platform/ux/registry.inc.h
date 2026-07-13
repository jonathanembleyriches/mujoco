// ProtoSpec Studio plugin registry implementation (ps::studio, ours).
//
// Shared template bodies for RegisterPlugin/ForEachPlugin plus an explicit-
// instantiation macro. This replaces MuJoCo Studio's engine_global_table.h
// backing (see plugin.h "Adaptation"): a per-type function-local table, ordered
// by registration, with case-insensitive name uniqueness. Included by
// registry.cc (core plugin types) and ps_plugin_ext.cc (extension types) so the
// bodies are defined once and instantiated per translation unit.

#ifndef PS_STUDIO_PLATFORM_UX_REGISTRY_INC_H_
#define PS_STUDIO_PLATFORM_UX_REGISTRY_INC_H_

#include <cctype>
#include <cstring>
#include <functional>
#include <vector>

namespace ps::studio {
namespace detail {

template <typename T>
std::vector<T>& PluginTable() {
  static std::vector<T> table;
  return table;
}

inline bool CaseInsensitiveEqual(const char* a, const char* b) {
  if (a == nullptr || b == nullptr) {
    return a == b;
  }
  while (*a && *b) {
    if (std::tolower((unsigned char)*a) != std::tolower((unsigned char)*b)) {
      return false;
    }
    ++a;
    ++b;
  }
  return *a == *b;
}

}  // namespace detail

template <typename T>
void RegisterPlugin(T plugin) {
  auto& table = detail::PluginTable<T>();
  for (const T& existing : table) {
    if (detail::CaseInsensitiveEqual(existing.name, plugin.name)) {
      return;  // AppendIfUnique semantics: first registration wins.
    }
  }
  table.push_back(plugin);
}

template <typename T>
void ForEachPlugin(const std::function<void(T*)>& fn) {
  auto& table = detail::PluginTable<T>();
  for (T& plugin : table) {
    fn(&plugin);
  }
}

}  // namespace ps::studio

#define PS_STUDIO_INSTANTIATE_PLUGIN(T)                                     \
  template void ps::studio::RegisterPlugin<T>(T);                           \
  template void ps::studio::ForEachPlugin<T>(                              \
      const std::function<void(T*)>&)

#endif  // PS_STUDIO_PLATFORM_UX_REGISTRY_INC_H_
