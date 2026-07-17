// ProtoSpec Studio: layer composition (ps::studio, ours). See layers.h.

#include "editor/layers.h"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "editor/undo.h"
#include "mjcf.h"
#include "protospec/detail.h"
#include "reflect.h"
#include "protospec/traversal.h"
#include "types.h"

namespace ps::studio {
namespace {

namespace mj = ps::mjcf;
namespace io = ps::mjcf::io;
namespace sdkd = ps::sdk::detail;

// Move every element out of `src` onto the end of `dst`. The sections are flat
// vectors of owned elements, so a merge is a concatenation per section -- MJCF
// permits repeated <asset>/<worldbody>/<default> blocks, and the writer and the
// native compiler both take them as one.
template <class T>
void AppendAll(std::vector<std::unique_ptr<T>>& dst,
               std::vector<std::unique_ptr<T>>& src) {
  for (std::unique_ptr<T>& p : src) dst.push_back(std::move(p));
  src.clear();
}

void MergeInto(mj::Model& dst, mj::Model& src) {
  AppendAll(dst.compilers, src.compilers);
  AppendAll(dst.options, src.options);
  AppendAll(dst.sizes, src.sizes);
  AppendAll(dst.visuals, src.visuals);
  AppendAll(dst.statistics, src.statistics);
  AppendAll(dst.defaults, src.defaults);
  AppendAll(dst.extensions, src.extensions);
  AppendAll(dst.customs, src.customs);
  AppendAll(dst.assets, src.assets);
  AppendAll(dst.worldbody, src.worldbody);
  AppendAll(dst.deformables, src.deformables);
  AppendAll(dst.contacts, src.contacts);
  AppendAll(dst.equalitys, src.equalitys);
  AppendAll(dst.tendons, src.tendons);
  AppendAll(dst.actuators, src.actuators);
  AppendAll(dst.sensors, src.sensors);
  AppendAll(dst.keyframes, src.keyframes);
}

// Every (type, name) a layer's tree claims, with a readable label per key.
void CollectNames(mj::Model& m,
                  std::unordered_map<std::string, std::string>& out) {
  sdkd::WalkModelAll(m, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if constexpr (!std::is_same_v<E, mj::Model>) {
      const std::string* nm = sdkd::NameOf(e);
      if (!nm || nm->empty()) return;
      constexpr mj::ElementType kType = mj::element_type_of<E>::value;
      out.emplace(std::to_string(static_cast<int>(kType)) + ':' + *nm,
                  std::string(mj::reflect::Describe(kType).name) + " '" + *nm +
                      "'");
    }
  });
}

std::string StemOf(const std::string& path) {
  std::error_code ec;
  std::filesystem::path p(path);
  const std::string stem = p.stem().string();
  return stem.empty() ? path : stem;
}

}  // namespace

ComposeResult ComposeLayers(EditorContext& ctx) {
  ComposeResult out;
  out.model = std::make_unique<mj::Model>();

  // key -> the layer that claimed it first, for conflict reporting.
  std::unordered_map<std::string, std::string> claimed_by;
  std::unordered_map<std::string, std::string> labels;

  for (int i = 0; i < static_cast<int>(ctx.layers.size()); ++i) {
    if (!ctx.layers[i].enabled) continue;
    mj::Model* src = ctx.LayerTree(i);
    if (!src) continue;

    std::unordered_map<std::string, std::string> names;
    CollectNames(*src, names);
    for (const auto& [key, label] : names) {
      auto it = claimed_by.find(key);
      if (it != claimed_by.end()) {
        out.conflicts.push_back({label, it->second, ctx.layers[i].name});
      } else {
        claimed_by.emplace(key, ctx.layers[i].name);
      }
    }

    // Serials must survive: a pick on the composed model reports the serial of
    // the element it hit, and that has to name the authored element.
    std::unique_ptr<mj::Model> clone = CloneWithSerials(*src);
    if (!out.model->model && clone->model) out.model->model = clone->model;
    MergeInto(*out.model, *clone);
    ++out.layers_used;
  }
  return out;
}

ps::mjcf::Model* BuildComposed(EditorContext& ctx) {
  if (ctx.layers.empty()) return ctx.tree.get();

  // A lone enabled layer is its own composition; skip the clone so the common
  // single-layer case compiles the authored tree directly (and a gizmo drag
  // does not pay for a whole-model clone every frame).
  int enabled = 0, only = -1;
  for (int i = 0; i < static_cast<int>(ctx.layers.size()); ++i) {
    if (ctx.layers[i].enabled) {
      ++enabled;
      only = i;
    }
  }
  if (enabled == 1) {
    ctx.composed.reset();
    return ctx.LayerTree(only);
  }
  if (enabled == 0) {
    ctx.Diagnose(DiagEntry{DiagEntry::Severity::Warning,
                           "[layers] every layer is disabled -- nothing to compile",
                           {},
                           {}});
    ctx.composed.reset();
    return nullptr;
  }

  ComposeResult r = ComposeLayers(ctx);
  for (const LayerConflict& c : r.conflicts) {
    ctx.Diagnose(DiagEntry{DiagEntry::Severity::Error,
                           "[layers] " + c.name + " is defined by both '" +
                               c.first + "' and '" + c.second + "'",
                           {},
                           {}});
  }
  ctx.composed = std::move(r.model);
  return ctx.composed.get();
}

int LayerOwningSerial(EditorContext& ctx, std::uint64_t serial) {
  if (serial == 0) return -1;
  for (int i = 0; i < static_cast<int>(ctx.layers.size()); ++i) {
    mj::Model* t = ctx.LayerTree(i);
    if (!t) continue;
    bool found = false;
    sdkd::WalkModelAll(*t, [&](auto& e) {
      using E = std::decay_t<decltype(e)>;
      if constexpr (!std::is_same_v<E, mj::Model> && requires { e.serial; }) {
        if (!found && e.serial == serial) found = true;
      }
    });
    if (found) return i;
  }
  return -1;
}

void ResetLayers(EditorContext& ctx, const std::string& name,
                 const std::string& path) {
  ctx.layers.clear();
  ctx.active_layer = 0;
  Layer base;
  base.name = name.empty() ? "base" : name;
  base.path = path;
  base.enabled = true;
  base.tree = nullptr;  // the active layer's tree lives in ctx.tree
  ctx.layers.push_back(std::move(base));
}

void AddEmptyLayer(EditorContext& ctx, const std::string& name) {
  Layer l;
  l.name = name.empty() ? ("layer " + std::to_string(ctx.layers.size() + 1))
                        : name;
  l.enabled = true;
  l.tree = std::make_unique<mj::Model>();
  ctx.layers.push_back(std::move(l));
  ctx.SetActiveLayer(static_cast<int>(ctx.layers.size()) - 1);
  ctx.RequestRecompile();
}

bool AddLayerFromFile(EditorContext& ctx, const std::string& path,
                      std::string* error) {
  io::ParseResult parsed = io::ParseMjcfFile(path);
  for (const io::Diagnostic& w : parsed.warnings) {
    DiagEntry e{DiagEntry::Severity::Warning, "[layer parse] " + w.message, {}, {}};
    if (!w.loc.file.empty()) e.loc = w.loc;
    ctx.Diagnose(std::move(e));
  }
  if (!parsed.ok()) {
    std::string msg = parsed.errors.empty() ? "parse failed"
                                            : parsed.errors.front().message;
    for (const io::Diagnostic& d : parsed.errors) {
      DiagEntry e{DiagEntry::Severity::Error, "[layer parse] " + d.message, {}, {}};
      if (!d.loc.file.empty()) e.loc = d.loc;
      ctx.Diagnose(std::move(e));
    }
    if (error) *error = msg;
    return false;
  }

  Layer l;
  l.name = StemOf(path);
  l.path = path;
  l.enabled = true;
  l.tree = std::move(parsed.model);
  ctx.layers.push_back(std::move(l));
  ctx.RequestRecompile();
  return true;
}

void RemoveLayer(EditorContext& ctx, int index) {
  const int n = static_cast<int>(ctx.layers.size());
  if (index < 0 || index >= n || n <= 1) return;  // never drop the last layer

  // Removing the edit target hands `tree` to whichever layer becomes active,
  // so the invariant survives.
  if (index == ctx.active_layer) {
    const int next = index == 0 ? 1 : index - 1;
    ctx.SetActiveLayer(next);
  }
  ctx.layers.erase(ctx.layers.begin() + index);
  if (ctx.active_layer > index) --ctx.active_layer;
  ctx.RequestRecompile();
}

namespace {

// A filename-safe token from a layer name ("my layer!" -> "my_layer").
std::string Slug(const std::string& s, int fallback_index) {
  std::string out;
  for (char ch : s) {
    if (std::isalnum(static_cast<unsigned char>(ch))) {
      out += static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    } else if (!out.empty() && out.back() != '_') {
      out += '_';
    }
  }
  while (!out.empty() && out.back() == '_') out.pop_back();
  return out.empty() ? ("layer" + std::to_string(fallback_index)) : out;
}

bool WriteFile(const std::string& path, const std::string& text,
               std::string* error) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    if (error) *error = "cannot open " + path;
    return false;
  }
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
  out.close();
  if (!out) {
    if (error) *error = "write error " + path;
    return false;
  }
  return true;
}

}  // namespace

bool ExportLayeredMjcf(EditorContext& ctx, const std::string& root_path,
                       std::string* error) {
  std::filesystem::path root(root_path);
  const std::string stem = root.stem().string();
  const std::filesystem::path dir = root.parent_path();

  // Collect first: a failure to serialize must not leave a half-written stack.
  struct Part {
    std::string file;  // basename, as it appears in the <include>
    std::string mjcf;
  };
  std::vector<Part> parts;
  for (int i = 0; i < static_cast<int>(ctx.layers.size()); ++i) {
    if (!ctx.layers[i].enabled) continue;
    mj::Model* t = ctx.LayerTree(i);
    if (!t) continue;
    parts.push_back({stem + "." + Slug(ctx.layers[i].name, i) + ".xml",
                     io::WriteMjcf(*t)});
  }
  if (parts.empty()) {
    if (error) *error = "no enabled layer to export";
    return false;
  }

  std::string root_doc = "<mujoco model=\"" + stem + "\">\n";
  for (const Part& p : parts) {
    root_doc += "  <include file=\"" + p.file + "\"/>\n";
  }
  root_doc += "</mujoco>\n";

  for (const Part& p : parts) {
    if (!WriteFile((dir / p.file).string(), p.mjcf, error)) return false;
  }
  if (!WriteFile(root_path, root_doc, error)) return false;

  ctx.Log("exported layered stack: " + root_path + " (+" +
          std::to_string(parts.size()) + " layer files)");
  ctx.status_line = "exported " + std::to_string(parts.size()) +
                    " layers -> " + root_path;
  return true;
}

void MoveLayer(EditorContext& ctx, int index, int delta) {
  const int n = static_cast<int>(ctx.layers.size());
  const int to = index + delta;
  if (index < 0 || index >= n || to < 0 || to >= n) return;
  std::swap(ctx.layers[index], ctx.layers[to]);
  if (ctx.active_layer == index) {
    ctx.active_layer = to;
  } else if (ctx.active_layer == to) {
    ctx.active_layer = index;
  }
  ctx.RequestRecompile();
}

}  // namespace ps::studio
