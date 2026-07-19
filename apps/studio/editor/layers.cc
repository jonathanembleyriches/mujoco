// ProtoSpec Studio: layers as provenance tags (ps::studio, ours). See layers.h.

#include "editor/layers.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string_view>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tinyxml2.h"

#include "editor/editor_ops.h"
#include "editor/undo.h"
#include "mjcf.h"
#include "protospec/refs.h"
#include "protospec/traversal.h"
#include "reflect.h"
#include "types.h"
#include "visit.h"

namespace ps::studio {
namespace {

namespace mj = ps::mjcf;
namespace io = ps::mjcf::io;
namespace reflect = ps::mjcf::reflect;
namespace sdk = ps::sdk;
namespace xt = tinyxml2;

// Visit every element of the tree EXCEPT the Model root (the root is the
// document, not a layer member: its loc names whichever file happened to open
// the stack and must never partition, prune, or gate anything).
template <class Fn>
void ForEachElement(mj::Model& m, Fn&& fn) {
  sdk::WalkModel(m, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if constexpr (!std::is_same_v<E, mj::Model>) {
      fn(e);
    }
  });
}

std::string StemOf(const std::string& path) {
  std::filesystem::path p(path);
  const std::string stem = p.stem().string();
  return stem.empty() ? path : stem;
}

// Display name for a key: "layer://x" -> "x", a path -> its stem.
std::string DisplayNameForKey(const std::string& key) {
  constexpr std::string_view kScheme = "layer://";
  if (key.rfind(kScheme, 0) == 0) return key.substr(kScheme.size());
  return StemOf(key);
}

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

// Stamp every element with an empty loc.file to `key`. This is how in-editor
// authored elements join a layer: the compile seam runs it with the ACTIVE
// layer's key right after every CommitEdit (BuildCompileModel), so no
// authoring op needs to know layers exist.
void StampEmptyLoc(mj::Model& m, const std::string& key) {
  ForEachElement(m, [&](auto& e) {
    if (e.loc.file.empty()) e.loc.file = key;
  });
}

// Remove every element whose loc.file is in `drop`, subtrees included (removing
// a body removes everything under it -- correct and expected). sdk::PruneSubtrees
// is the generalization of the editor's original LayerPruner: same erase-then-
// descend order, keyed on a predicate rather than the loc.file set directly. It
// deliberately does NOT scan/clear references to pruned elements -- a disabled
// layer's dangling refs are meant to surface as honest compile/validate errors.
void PruneKeys(mj::Model& m, const std::unordered_set<std::string>& drop) {
  sdk::PruneSubtrees(m, [&](const auto& e) { return drop.count(e.loc.file) > 0; });
}

// Move every top-level element of `src` onto the end of `dst`, per section
// (MJCF permits repeated <asset>/<worldbody>/... blocks). Used by the
// add-layer-from-file graft; the moved elements keep their parse provenance,
// which is exactly what makes them a layer.
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

}  // namespace

// --- Partition / lookup ---------------------------------------------------- //

int LayerIndexForKey(const EditorContext& ctx, const std::string& key) {
  for (int i = 0; i < static_cast<int>(ctx.layers.size()); ++i) {
    if (ctx.layers[i].key == key) return i;
  }
  return -1;
}

void SplitLayersFromTree(EditorContext& ctx, const std::string& root_key,
                         const std::string& root_name) {
  ctx.layers.clear();
  ctx.layer_graph = LayerGraph{};
  ctx.active_layer = 0;
  if (!ctx.tree) return;

  // Programmatically built elements (no provenance) belong to the root layer.
  StampEmptyLoc(*ctx.tree, root_key);

  // Distinct keys in first-appearance (document) order.
  std::vector<std::string> keys;
  ForEachElement(*ctx.tree, [&](auto& e) {
    const std::string& k = e.loc.file;
    for (const std::string& seen : keys) {
      if (seen == k) return;
    }
    keys.push_back(k);
  });
  if (keys.empty()) keys.push_back(root_key);  // empty model: one root layer

  for (const std::string& k : keys) {
    Layer l;
    l.key = k;
    l.name = (k == root_key && !root_name.empty()) ? root_name
                                                   : DisplayNameForKey(k);
    l.enabled = true;
    ctx.layers.push_back(std::move(l));
  }
  const int root_idx = LayerIndexForKey(ctx, root_key);
  ctx.active_layer = root_idx >= 0 ? root_idx : 0;
  RecomputeLayerGraph(ctx);
}

void ReconcileLayers(EditorContext& ctx) {
  if (!ctx.tree) return;
  bool changed = false;
  ForEachElement(*ctx.tree, [&](auto& e) {
    const std::string& k = e.loc.file;
    if (k.empty() || LayerIndexForKey(ctx, k) >= 0) return;
    Layer l;
    l.key = k;
    l.name = DisplayNameForKey(k);
    l.enabled = true;
    ctx.layers.push_back(std::move(l));
    changed = true;
  });
  if (changed) RecomputeLayerGraph(ctx);
}

int LayerOfSerial(EditorContext& ctx, std::uint64_t serial) {
  if (!ctx.tree || serial == 0) return -1;
  std::string key;
  bool found = false;
  ForEachElement(*ctx.tree, [&](auto& e) {
    if (!found && e.serial == serial) {
      key = e.loc.file;
      found = true;
    }
  });
  if (!found) return -1;
  return LayerIndexForKey(ctx, key);
}

bool SerialInActiveLayer(EditorContext& ctx, std::uint64_t serial) {
  if (ctx.layers.size() <= 1) return true;  // degenerate case: no gating
  const int li = LayerOfSerial(ctx, serial);
  return li < 0 || li == ctx.active_layer;  // fail open on bookkeeping gaps
}

int CountLayerElements(EditorContext& ctx, int index) {
  if (!ctx.tree || index < 0 || index >= static_cast<int>(ctx.layers.size())) {
    return 0;
  }
  const std::string& key = ctx.layers[index].key;
  int count = 0;
  ForEachElement(*ctx.tree, [&](auto& e) {
    if (e.loc.file == key) ++count;
  });
  return count;
}

// --- Compile seam ---------------------------------------------------------- //

ps::mjcf::Model* BuildCompileModel(
    EditorContext& ctx, std::unique_ptr<ps::mjcf::Model>* out_pruned) {
  out_pruned->reset();
  if (!ctx.tree) return nullptr;

  // Housekeeping at the one seam every tree change passes through: new
  // elements join the active layer, unseen keys get rows, and the dependency
  // graph refreshes (per recompile, never per frame).
  if (const std::string* key = ctx.ActiveLayerKey()) {
    StampEmptyLoc(*ctx.tree, *key);
  }
  ReconcileLayers(ctx);
  RecomputeLayerGraph(ctx);

  std::unordered_set<std::string> drop;
  for (const Layer& l : ctx.layers) {
    if (!l.enabled) drop.insert(l.key);
  }
  if (drop.empty()) return ctx.tree.get();  // common case: no clone

  std::unique_ptr<mj::Model> clone = CloneWithSerials(*ctx.tree);
  PruneKeys(*clone, drop);
  *out_pruned = std::move(clone);
  return out_pruned->get();
}

void RecomputeLayerGraph(EditorContext& ctx) {
  LayerGraph g;
  const int n = static_cast<int>(ctx.layers.size());
  g.group.assign(n, 0);
  for (int i = 0; i < n; ++i) g.group[i] = i;

  if (ctx.tree && n > 1) {
    // Index every named element: name -> [(type, layer)].
    struct Named {
      mj::ElementType type;
      int layer;
    };
    std::unordered_map<std::string, std::vector<Named>> named;
    ForEachElement(*ctx.tree, [&](auto& e) {
      using E = std::decay_t<decltype(e)>;
      const std::string* nm = sdk::Name(e);
      if (nm && !nm->empty()) {
        named[*nm].push_back({mj::element_type_of<E>::value,
                              LayerIndexForKey(ctx, e.loc.file)});
      }
    });

    // Every authored reference (typed scalar, ref<T>[] entry, dynamic
    // target_from) that resolves across layers is an edge; one example kept
    // per (from, to) pair.
    std::set<std::pair<int, int>> seen;
    ForEachElement(*ctx.tree, [&](auto& e) {
      using E = std::decay_t<decltype(e)>;
      const int a = LayerIndexForKey(ctx, e.loc.file);
      if (a < 0) return;
      sdk::ScanRefs(
          e, [&](int, const char* fname, const std::string& rn,
                 const std::vector<mj::ElementType>& tgts) {
            auto it = named.find(rn);
            if (it == named.end()) return;
            for (const Named& cand : it->second) {
              if (cand.layer < 0 || cand.layer == a) continue;
              if (std::find(tgts.begin(), tgts.end(), cand.type) == tgts.end())
                continue;
              if (!seen.insert({a, cand.layer}).second) continue;
              std::string who = std::string(
                  reflect::Describe(mj::element_type_of<E>::value).name);
              if (const std::string* enm = sdk::Name(e)) {
                if (!enm->empty()) who += " '" + *enm + "'";
              }
              g.edges.push_back(
                  {a, cand.layer, who + "." + fname + " -> '" + rn + "'"});
            }
          });
    });

    // Connected components over the undirected edges (union-find).
    std::vector<int> parent(n);
    for (int i = 0; i < n; ++i) parent[i] = i;
    std::function<int(int)> find = [&](int x) {
      while (parent[x] != x) x = parent[x] = parent[parent[x]];
      return x;
    };
    for (const LayerEdge& e : g.edges) {
      const int ra = find(e.from), rb = find(e.to);
      if (ra != rb) parent[ra] = rb;
    }
    for (int i = 0; i < n; ++i) g.group[i] = find(i);
  }

  // Structural containment: layer B's element sitting inside layer A's element
  // (an <include> spliced into another file's body). The NEAREST differing
  // ancestor is the container; the first found (walk order) wins for a
  // multi-region layer. Distinct from reference edges -- see LayerGraph.
  g.inside.assign(n, -1);
  g.inside_example.assign(n, std::string());
  if (ctx.tree && n > 1) {
    std::unordered_map<const void*, int> layer_of;
    ForEachElement(*ctx.tree, [&](auto& e) {
      layer_of[static_cast<const void*>(&e)] =
          LayerIndexForKey(ctx, e.loc.file);
    });
    ps::sdk::ParentMap pm(*ctx.tree);
    ForEachElement(*ctx.tree, [&](auto& e) {
      const int b = LayerIndexForKey(ctx, e.loc.file);
      if (b < 0 || g.inside[b] >= 0) return;
      const void* p = pm.ParentOf(e);
      while (p) {
        auto it = layer_of.find(p);
        if (it == layer_of.end()) break;  // reached the Model root
        if (it->second >= 0 && it->second != b) {
          g.inside[b] = it->second;
          std::string who = std::string(
              reflect::Describe(mj::element_type_of<
                                    std::decay_t<decltype(e)>>::value)
                  .name);
          if (const std::string* enm = sdk::Name(e)) {
            if (!enm->empty()) who += " '" + *enm + "'";
          }
          const ps::sdk::ParentMap::Node* pn = pm.Lookup(p);
          std::string owner =
              pn ? std::string(reflect::Describe(pn->type).name) +
                       (pn->name.empty() ? "" : " '" + pn->name + "'")
                 : std::string("an element");
          g.inside_example[b] = who + " sits inside " + owner;
          return;
        }
        const ps::sdk::ParentMap::Node* pn = pm.Lookup(p);
        p = pn ? pn->parent : nullptr;
      }
    });
  }

  ctx.layer_graph = std::move(g);
}

LayerLock LayerLockInfo(const EditorContext& ctx, int index) {
  LayerLock out;
  for (const LayerEdge& e : ctx.layer_graph.edges) {
    if (e.to != index || e.from == index) continue;
    if (e.from < 0 || e.from >= static_cast<int>(ctx.layers.size())) continue;
    if (!ctx.layers[e.from].enabled) continue;  // a disabled dependent unlocks
    out.locked = true;
    if (!out.dependents.empty()) out.dependents += ", ";
    out.dependents += "'" + ctx.layers[e.from].name + "'";
    if (out.example.empty()) out.example = e.example;
  }
  return out;
}

// --- Layer list ops -------------------------------------------------------- //

void AddEmptyLayer(EditorContext& ctx, const std::string& name) {
  const std::string base =
      name.empty() ? ("layer " + std::to_string(ctx.layers.size() + 1)) : name;
  std::string key = "layer://" + Slug(base, static_cast<int>(ctx.layers.size()));
  for (int k = 2; LayerIndexForKey(ctx, key) >= 0; ++k) {
    key = "layer://" + Slug(base, static_cast<int>(ctx.layers.size())) + "_" +
          std::to_string(k);
  }
  Layer l;
  l.name = base;
  l.key = key;
  l.enabled = true;
  ctx.layers.push_back(std::move(l));
  ctx.SetActiveLayer(static_cast<int>(ctx.layers.size()) - 1);
  RecomputeLayerGraph(ctx);
}

bool AddLayerFromFile(EditorContext& ctx, const std::string& path,
                      std::string* error) {
  if (!ctx.tree) {
    if (error) *error = "no model loaded";
    return false;
  }
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

  // Graft: the parsed elements keep their provenance (the file itself, plus
  // one key per file its own <include>s spliced in), so ReconcileLayers turns
  // them into layer rows with no re-tagging. One undo step.
  ctx.BeginEdit();
  MergeInto(*ctx.tree, *parsed.model);
  ctx.CommitEdit("Add layer '" + StemOf(path) + "'");
  ReconcileLayers(ctx);
  const int idx = LayerIndexForKey(ctx, path);
  if (idx >= 0) ctx.SetActiveLayer(idx);
  ctx.Log("added layer from file: " + path);
  return true;
}

bool RemoveLayer(EditorContext& ctx, int index, bool delete_elements) {
  const int n = static_cast<int>(ctx.layers.size());
  if (index < 0 || index >= n || n <= 1) return false;  // keep the last layer

  const int count = CountLayerElements(ctx, index);
  if (count > 0) {
    if (!delete_elements) return false;
    ctx.BeginEdit();
    PruneKeys(*ctx.tree, {ctx.layers[index].key});
    ctx.CommitEdit("Remove layer '" + ctx.layers[index].name + "'");
    // The selection may have gone with the layer; re-resolve (clears if gone).
    SelectBySerial(ctx, ctx.selected_serial);
  }

  ctx.layers.erase(ctx.layers.begin() + index);
  if (ctx.active_layer >= static_cast<int>(ctx.layers.size())) {
    ctx.active_layer = static_cast<int>(ctx.layers.size()) - 1;
  } else if (ctx.active_layer > index) {
    --ctx.active_layer;
  }
  RecomputeLayerGraph(ctx);
  ctx.RequestRecompile();
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
  RecomputeLayerGraph(ctx);  // graph indexes layers by position
}

// --- Export ---------------------------------------------------------------- //
//
// Mechanism: serialize the (pruned) tree with WriteMjcf, reparse that
// deterministic output with tinyxml2, then walk the ProtoSpec tree and the DOM
// in lockstep (the writer emits exactly one XML element per tree element, in
// Visit order) tagging every DOM node with its element's layer key. DOM
// surgery then extracts foreign-tagged subtrees into fragment documents,
// replacing them with <include> elements.

namespace {

// Children of `e` in writer emission order (the same Visit order the writer's
// child/union_child hooks append them in).
template <class Fn>
struct EmitChildVisitor {
  Fn* fn;
  template <class U>
  void field(int, const char*, U&) {}
  template <class T>
  void child(int, const char*, std::vector<std::unique_ptr<T>>& list) {
    for (auto& p : list) {
      if (p) (*fn)(*p);
    }
  }
  template <class U>
  void union_child(int, const char*, std::vector<U>& list) {
    for (auto& item : list) {
      std::visit([&](auto& p) { if (p) (*fn)(*p); }, item.node);
    }
  }
};

using DomTags = std::unordered_map<const xt::XMLElement*, std::string>;

// Tag every DOM element with its tree counterpart's layer key. Returns false
// on any structural mismatch (defensive: WriteMjcf and this walk share the
// Visit order, so a mismatch means a writer change broke the invariant).
template <class E>
bool TagDomTree(E& e, xt::XMLElement* dom, DomTags& tags) {
  if (!dom) return false;
  tags[dom] = e.loc.file;
  bool ok = true;
  xt::XMLElement* c = dom->FirstChildElement();
  auto rec = [&](auto& child) {
    if (!ok) return;
    if (!TagDomTree(child, c, tags)) {
      ok = false;
      return;
    }
    c = c->NextSiblingElement();
  };
  EmitChildVisitor<decltype(rec)> v{&rec};
  mj::Visit(e, v);
  return ok && c == nullptr;
}

// Deep-clone `src` into `doc`, copying the layer tags onto the clones.
void CopyTags(const xt::XMLElement* src, const xt::XMLElement* clone,
              DomTags& tags) {
  auto it = tags.find(src);
  if (it != tags.end()) tags[clone] = it->second;
  const xt::XMLElement* sc = src->FirstChildElement();
  const xt::XMLElement* cc = clone->FirstChildElement();
  while (sc && cc) {
    CopyTags(sc, cc, tags);
    sc = sc->NextSiblingElement();
    cc = cc->NextSiblingElement();
  }
}

xt::XMLElement* CloneWithTags(const xt::XMLElement* src, xt::XMLDocument& doc,
                              DomTags& tags) {
  xt::XMLElement* clone = src->DeepClone(&doc)->ToElement();
  if (clone) CopyTags(src, clone, tags);
  return clone;
}

// One output file of the export: a fragment document rooted <mujocoinclude>.
struct ExportFile {
  std::string key;       // owning layer key
  std::string filename;  // basename (export naming; kept for logs)
  std::string abs_path;      // where the file is written
  std::string include_attr;  // what the <include file="..."> says
  bool top_level = false;  // referenced from the root document
  std::unique_ptr<xt::XMLDocument> doc;
  xt::XMLElement* root = nullptr;
};

struct ExportState {
  EditorContext* ctx;
  DomTags* tags;
  std::vector<std::unique_ptr<ExportFile>> files;
  std::unordered_map<std::string, int> regions;  // key -> files created so far
  std::unordered_set<std::string> used_names;    // filename collision guard

  // Save-in-place mode: a file-backed layer's primary file is ITS OWN path
  // (the layer key), not a slug beside the root -- saving writes each layer
  // back to the file it came from. An authored (layer://) layer is assigned
  // a slug file beside the root on first save and `assigned` records it so the
  // caller can retag the layer as file-backed. Export mode keeps slug naming.
  bool in_place = false;
  std::string root_dir;
  std::unordered_map<std::string, std::string> assigned;  // layer:// -> abs path

  const std::string& TagOf(const xt::XMLElement* e) const {
    static const std::string kEmpty;
    auto it = tags->find(e);
    return it == tags->end() ? kEmpty : it->second;
  }

  ExportFile& NewFileFor(const std::string& key) {
    const int idx = LayerIndexForKey(*ctx, key);
    const std::string base =
        Slug(idx >= 0 ? ctx->layers[idx].name : DisplayNameForKey(key),
             static_cast<int>(files.size()));
    const int region = ++regions[key];
    std::string fn =
        region == 1 ? base + ".xml" : base + "." + std::to_string(region) + ".xml";
    for (int k = 2; used_names.count(fn); ++k) {
      fn = base + "_" + std::to_string(k) + ".xml";
    }
    used_names.insert(fn);

    auto f = std::make_unique<ExportFile>();
    f->key = key;
    f->filename = fn;
    if (in_place) {
      constexpr std::string_view kScheme = "layer://";
      const bool file_backed = key.rfind(kScheme, 0) != 0;
      if (file_backed && region == 1) {
        f->abs_path = key;  // the layer's own file: write back where it came from
      } else if (file_backed) {
        // Extra disjoint region of a file-backed layer: beside its primary.
        std::filesystem::path k(key);
        f->abs_path = (k.parent_path() /
                       (k.stem().string() + "." + std::to_string(region) + ".xml"))
                          .string();
      } else {
        // Authored layer: assign a real file beside the root on first save.
        f->abs_path = (std::filesystem::path(root_dir) / fn).string();
        if (region == 1) assigned[key] = f->abs_path;
      }
      std::error_code ec;
      const std::filesystem::path rel =
          std::filesystem::relative(f->abs_path, root_dir, ec);
      f->include_attr = (ec || rel.empty()) ? f->abs_path : rel.string();
    } else {
      f->abs_path = (std::filesystem::path(root_dir) / fn).string();
      f->include_attr = fn;
    }
    f->doc = std::make_unique<xt::XMLDocument>();
    f->root = f->doc->NewElement("mujocoinclude");
    f->doc->InsertEndChild(f->root);
    files.push_back(std::move(f));
    return *files.back();
  }

  // The layer's primary file if one exists yet, else a fresh one.
  ExportFile& FileFor(const std::string& key) {
    for (auto& f : files) {
      if (f->key == key) return *f;
    }
    return NewFileFor(key);
  }
};

// Insert `node` into `parent` immediately before `before`.
void InsertBefore(xt::XMLElement* parent, xt::XMLNode* node,
                  xt::XMLElement* before) {
  xt::XMLNode* prev = before->PreviousSibling();
  if (prev) {
    parent->InsertAfterChild(prev, node);
  } else {
    parent->InsertFirstChild(node);
  }
}

// Extract foreign-tagged subtrees below `el` (owned by layer `owner`):
// contiguous runs of children carrying another layer's key move into that
// layer's file (a fresh fragment per disjoint region), replaced in place by an
// <include> of it. Recurses into both the kept and the moved children.
void ExtractForeign(xt::XMLElement* el, const std::string& owner,
                    ExportState& st) {
  std::vector<xt::XMLElement*> kids;
  for (xt::XMLElement* c = el->FirstChildElement(); c;
       c = c->NextSiblingElement()) {
    kids.push_back(c);
  }
  std::size_t i = 0;
  while (i < kids.size()) {
    const std::string& k = st.TagOf(kids[i]);
    if (k.empty() || k == owner) {
      ExtractForeign(kids[i], owner, st);
      ++i;
      continue;
    }
    // A maximal run of same-foreign-key siblings becomes ONE fragment region.
    std::size_t j = i;
    while (j < kids.size() && st.TagOf(kids[j]) == k) ++j;

    ExportFile& frag = st.NewFileFor(k);
    std::vector<xt::XMLElement*> moved;
    for (std::size_t t = i; t < j; ++t) {
      xt::XMLElement* clone = CloneWithTags(kids[t], *frag.doc, *st.tags);
      frag.root->InsertEndChild(clone);
      moved.push_back(clone);
    }
    xt::XMLElement* inc = el->GetDocument()->NewElement("include");
    inc->SetAttribute("file", frag.filename.c_str());
    InsertBefore(el, inc, kids[i]);
    for (std::size_t t = i; t < j; ++t) el->DeleteChild(kids[t]);
    for (xt::XMLElement* m : moved) ExtractForeign(m, k, st);
    i = j;
  }
}

bool WriteTextFile(const std::string& path, const std::string& text,
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

// Shared core of Export (slug files beside the root, disabled layers pruned)
// and Save-in-place (each layer back to its own file, EVERYTHING written --
// enable/disable is compile state, not file state; a save must never drop an
// authored layer's file because it happened to be toggled off).
static bool WriteStack(EditorContext& ctx, const std::string& root_path,
                       bool in_place, std::string* error) {
  if (!ctx.tree || ctx.layers.empty()) {
    if (error) *error = "no model loaded";
    return false;
  }

  mj::Model* src = ctx.tree.get();
  std::unique_ptr<mj::Model> pruned;
  if (!in_place) {
    // Disabled layers are not exported: prune them from a serial-preserving
    // clone (the authored tree is never touched).
    std::unordered_set<std::string> drop;
    for (const Layer& l : ctx.layers) {
      if (!l.enabled) drop.insert(l.key);
    }
    if (!drop.empty()) {
      pruned = CloneWithSerials(*ctx.tree);
      PruneKeys(*pruned, drop);
      src = pruned.get();
    }
  }

  // Serialize -> reparse -> tag the DOM by layer.
  const std::string xml = io::WriteMjcf(*src);
  xt::XMLDocument main;
  if (main.Parse(xml.c_str(), xml.size()) != xt::XML_SUCCESS) {
    if (error) *error = "internal: reparse of WriteMjcf output failed";
    return false;
  }
  xt::XMLElement* mroot = main.RootElement();
  DomTags tags;
  if (!mroot || !TagDomTree(*src, mroot, tags)) {
    if (error) *error = "internal: writer/DOM structure mismatch";
    return false;
  }

  const std::filesystem::path root_fs(root_path);
  const std::filesystem::path dir = root_fs.parent_path();
  const std::string root_base = root_fs.filename().string();

  ExportState st;
  st.ctx = &ctx;
  st.tags = &tags;
  st.in_place = in_place;
  st.root_dir = dir.string();
  st.used_names.insert(root_base);

  // Top-level sections: each goes to its own layer's primary file (grouped by
  // layer, in tree order within a layer). The root document will hold one
  // <include> per top-level file, in layer display order.
  std::vector<xt::XMLElement*> top;
  for (xt::XMLElement* c = mroot->FirstChildElement(); c;
       c = c->NextSiblingElement()) {
    top.push_back(c);
  }
  const std::string* active_key = ctx.ActiveLayerKey();
  for (xt::XMLElement* c : top) {
    std::string k = st.TagOf(c);
    if (k.empty() && active_key) k = *active_key;  // defensive; stamped normally
    ExportFile& f = st.FileFor(k);
    f.top_level = true;
    xt::XMLElement* clone = CloneWithTags(c, *f.doc, tags);
    f.root->InsertEndChild(clone);
    mroot->DeleteChild(c);
  }
  if (st.files.empty()) {
    if (error) *error = "no enabled layer content to export";
    return false;
  }

  // Nested extraction inside each top-level file (fragments append to
  // st.files as they are discovered; index loop tolerates growth).
  for (std::size_t fi = 0; fi < st.files.size(); ++fi) {
    ExportFile& f = *st.files[fi];
    std::vector<xt::XMLElement*> kids;
    for (xt::XMLElement* c = f.root->FirstChildElement(); c;
         c = c->NextSiblingElement()) {
      kids.push_back(c);
    }
    for (xt::XMLElement* c : kids) ExtractForeign(c, f.key, st);
  }

  // Root document: the <mujoco> attrs plus the include lines, ordered by the
  // layer display order (unknown keys, if any, trail in creation order).
  for (const Layer& l : ctx.layers) {
    for (auto& f : st.files) {
      if (f->top_level && f->key == l.key) {
        xt::XMLElement* inc = main.NewElement("include");
        inc->SetAttribute("file", f->include_attr.c_str());
        mroot->InsertEndChild(inc);
        f->top_level = false;  // consumed
      }
    }
  }
  for (auto& f : st.files) {
    if (f->top_level) {
      xt::XMLElement* inc = main.NewElement("include");
      inc->SetAttribute("file", f->include_attr.c_str());
      mroot->InsertEndChild(inc);
      f->top_level = false;
    }
  }

  // Print everything first; write only when every document serialized.
  struct Out {
    std::string path;
    std::string text;
  };
  std::vector<Out> outs;
  {
    xt::XMLPrinter pr;
    main.Print(&pr);
    outs.push_back({root_path, pr.CStr()});
  }
  for (auto& f : st.files) {
    xt::XMLPrinter pr;
    f->doc->Print(&pr);
    outs.push_back({f->abs_path, pr.CStr()});
  }
  for (const Out& o : outs) {
    if (!WriteTextFile(o.path, o.text, error)) return false;
  }

  if (in_place) {
    // First save of an authored layer assigns its file: retag its elements and
    // the layer itself so it is file-backed from here on (and future saves go
    // to the same place). Keys changed, so the graph recomputes.
    if (!st.assigned.empty()) {
      for (const auto& [old_key, new_path] : st.assigned) {
        ForEachElement(*ctx.tree, [&](auto& e) {
          if (e.loc.file == old_key) e.loc.file = new_path;
        });
        const int li = LayerIndexForKey(ctx, old_key);
        if (li >= 0) ctx.layers[li].key = new_path;
      }
      RecomputeLayerGraph(ctx);
    }
    ctx.Log("saved layered stack: " + root_path + " (+" +
            std::to_string(st.files.size()) + " layer files, in place)");
    ctx.status_line = "saved " + std::to_string(st.files.size()) +
                      " layer files (in place)";
  } else {
    ctx.Log("exported layered stack: " + root_path + " (+" +
            std::to_string(st.files.size()) + " layer files)");
    ctx.status_line = "exported " + std::to_string(st.files.size()) +
                      " layer files -> " + root_path;
  }
  return true;
}

bool ExportLayeredMjcf(EditorContext& ctx, const std::string& root_path,
                       std::string* error) {
  return WriteStack(ctx, root_path, /*in_place=*/false, error);
}

bool SaveLayeredMjcf(EditorContext& ctx, const std::string& root_path,
                     std::string* error) {
  return WriteStack(ctx, root_path, /*in_place=*/true, error);
}

}  // namespace ps::studio
