// ProtoSpec Studio: the generated Details panel renderer (ps::studio, ours).
//
// A single reflection- and type-driven visitor draws any element's fields. The
// widget for a field is chosen from the C++ storage type (bool / integral /
// real / string / enum / Ref / std::array / InlineVec / std::vector / variant),
// so no element or field is special-cased. Presence layering (authored /
// class-inherited / IDL-default) is read off sdk::Effective(): an unset field
// shows its effective value grayed with a badge, editing materialises it, and a
// per-field revert clears it again. Every mapping decision lives in
// details_panel.h and is unit-tested windowless; this file is only the draw.
#include "editor/details_panel.h"

#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <imgui.h>
#include <imgui_stdlib.h>

#include "binding.h"
#include "editor/editor_ops.h"
#include "editor/layers.h"
#include "platform/ux/plugin.h"
#include "protospec/classes.h"

namespace ps::studio::details {
namespace {

namespace sdk = ps::sdk;

// --- Edit lifecycle bridge to the SE1a EditorContext ----------------------- //
// The BeginEdit/CommitEdit/CancelEdit/RequestRecompile contract is frozen but
// owned by the concurrent SE1a cluster. Calling through these `requires`-guarded
// shims (templated so the discarded branch is never instantiated) keeps this TU
// green whether or not those methods have landed yet, and binds to them the
// moment they do. INTEGRATION: once the context exposes the four methods these
// become direct calls; the shims can be dropped.
template <class C>
void EditBegin(C& c) {
  if constexpr (requires { c.BeginEdit(); }) c.BeginEdit();
}
template <class C>
void EditCommit(C& c, std::string_view label) {
  if constexpr (requires { c.CommitEdit(label); }) c.CommitEdit(label);
}
template <class C>
void EditCancel(C& c) {
  if constexpr (requires { c.CancelEdit(); }) c.CancelEdit();
}

// A gesture widget (drag/text) begins on activation, cancels on escape (deactivate
// without an edit), and commits on deactivate-after-edit. Call right after the
// widget. Returns true on the frame the gesture commits.
bool GestureShouldCommit(EditorContext& ctx) {
  if (ImGui::IsItemActivated()) EditBegin(ctx);
  const bool after_edit = ImGui::IsItemDeactivatedAfterEdit();
  if (ImGui::IsItemDeactivated() && !after_edit) EditCancel(ctx);
  return after_edit;
}

// An instant widget (checkbox/combo) begins and commits in the same frame.
bool InstantShouldCommit(EditorContext& ctx, bool changed) {
  if (changed) EditBegin(ctx);
  return changed;
}

// --- Variant arm labels ---------------------------------------------------- //
// The surviving variant families (GeomShape, TextureSource) are a fixed schema
// feature; their POD arms carry no reflection entry, so their selector labels
// are named here. Orientation, inertia and camera intrinsics were canonicalized
// to plain fields (Q-ORIENT/Q-INERTIA/R1) and render through the generic field
// path -- quat/iquat/diaginertia/fovy/focal are ordinary array/scalar rows now.
template <class A>
const char* ArmLabel() {
  if constexpr (std::is_same_v<A, mj::Explicit>) return "size";
  else if constexpr (std::is_same_v<A, mj::FromTo>) return "fromto";
  else if constexpr (std::is_same_v<A, mj::TexFile>) return "file";
  else if constexpr (std::is_same_v<A, mj::TextureBuiltin>) return "builtin";
  else return "option";
}

template <class V, std::size_t... I>
void ArmLabelsImpl(std::vector<const char*>& out, std::index_sequence<I...>) {
  (out.push_back(ArmLabel<std::variant_alternative_t<I, V>>()), ...);
}
template <class V>
std::vector<const char*> ArmLabels() {
  std::vector<const char*> out;
  ArmLabelsImpl<V>(out, std::make_index_sequence<std::variant_size_v<V>>{});
  return out;
}

template <class V, std::size_t... I>
void SetArmImpl(V& v, std::size_t idx, std::index_sequence<I...>) {
  ((I == idx ? (void)(v = std::variant_alternative_t<I, V>{}) : (void)0), ...);
}
template <class V>
void SetArmByIndex(V& v, std::size_t idx) {
  SetArmImpl<V>(v, idx, std::make_index_sequence<std::variant_size_v<V>>{});
}

// --- Grayed style for inherited/default values ----------------------------- //
struct Grayed {
  bool on;
  explicit Grayed(bool g) : on(g) {
    if (on) {
      ImGui::PushStyleColor(ImGuiCol_Text,
                            ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
    }
  }
  ~Grayed() {
    if (on) ImGui::PopStyleColor(1);
  }
};

constexpr float kFieldWidth = 90.0f;

// The single mapping from the window-free NumericWidget (chosen from the storage
// type in details_panel.h) to ImGui's data-type constant. Kept here so imgui.h
// stays out of the unit-tested header; the width invariant is proven there.
constexpr ImGuiDataType ImGuiTypeOf(NumericWidget w) {
  switch (w) {
    case NumericWidget::S8:  return ImGuiDataType_S8;
    case NumericWidget::S16: return ImGuiDataType_S16;
    case NumericWidget::S32: return ImGuiDataType_S32;
    case NumericWidget::S64: return ImGuiDataType_S64;
    case NumericWidget::U8:  return ImGuiDataType_U8;
    case NumericWidget::U16: return ImGuiDataType_U16;
    case NumericWidget::U32: return ImGuiDataType_U32;
    case NumericWidget::U64: return ImGuiDataType_U64;
    case NumericWidget::F32: return ImGuiDataType_Float;
    case NumericWidget::F64: return ImGuiDataType_Double;
  }
  return ImGuiDataType_S32;
}

// --- Scalar / composite value widgets -------------------------------------- //
// DrawValue draws editing widgets bound to `work` and returns true on the frame
// the user commits an edit. It never touches the model; the caller writes `work`
// back into the field (materialising an unset one) on commit. Dispatch is purely
// on the C++ type, so it serves element fields and variant-arm fields alike.

// Forward declaration for the variant recursion.
template <class Inner>
bool DrawValue(EditorContext& ctx, const char* label, Inner& work, int arity_min);

bool DrawScalarReal(EditorContext& ctx, const char* label, double& v) {
  ImGui::SetNextItemWidth(kFieldWidth);
  ImGui::DragScalar(label, ImGuiDataType_Double, &v, 0.01f);
  return GestureShouldCommit(ctx);
}
bool DrawScalarReal(EditorContext& ctx, const char* label, float& v) {
  ImGui::SetNextItemWidth(kFieldWidth);
  ImGui::DragScalar(label, ImGuiDataType_Float, &v, 0.01f);
  return GestureShouldCommit(ctx);
}
template <class I>
bool DrawScalarInt(EditorContext& ctx, const char* label, I& v) {
  ImGui::SetNextItemWidth(kFieldWidth);
  ImGui::DragScalar(label, ImGuiTypeOf(NumericWidgetOf<I>()), &v, 0.2f);
  return GestureShouldCommit(ctx);
}

// A fixed-count numeric row (std::array<N> or the filled span of an InlineVec).
template <class X>
bool DrawNumericRow(EditorContext& ctx, X* data, std::size_t count) {
  bool commit = false;
  for (std::size_t i = 0; i < count; ++i) {
    if (i) ImGui::SameLine();
    ImGui::PushID(static_cast<int>(i));
    ImGui::SetNextItemWidth(kFieldWidth * 0.7f);
    ImGui::DragScalar("##c", ImGuiTypeOf(NumericWidgetOf<X>()), &data[i],
                      std::is_floating_point_v<X> ? 0.01f : 0.2f);
    commit = GestureShouldCommit(ctx) || commit;
    ImGui::PopID();
  }
  return commit;
}

template <class E>
bool DrawEnum(EditorContext& ctx, const char* label, E& v) {
  const std::vector<std::string_view> labels = EnumLabels<E>();
  int cur = EnumIndexOf(v);
  std::string preview =
      (cur >= 0 && cur < static_cast<int>(labels.size()))
          ? std::string(labels[cur])
          : std::string("<") + std::to_string(cur) + ">";
  bool changed = false;
  ImGui::SetNextItemWidth(kFieldWidth * 1.6f);
  if (ImGui::BeginCombo(label, preview.c_str())) {
    for (std::size_t i = 0; i < labels.size(); ++i) {
      const std::string item(labels[i]);
      const bool sel = static_cast<int>(i) == cur;
      if (ImGui::Selectable(item.c_str(), sel)) {
        v = static_cast<E>(i);
        changed = true;
      }
    }
    ImGui::EndCombo();
  }
  return InstantShouldCommit(ctx, changed);
}

// A keyword set (std::vector<Enum>): a checkbox per enum value; checked == the
// value is present in the vector.
template <class E>
bool DrawEnumSet(EditorContext& ctx, std::vector<E>& v) {
  const std::vector<std::string_view> labels = EnumLabels<E>();
  bool changed = false;
  for (std::size_t i = 0; i < labels.size(); ++i) {
    const E val = static_cast<E>(i);
    bool on = false;
    for (E e : v) {
      if (e == val) { on = true; break; }
    }
    const std::string item(labels[i]);
    if (i) ImGui::SameLine();
    if (ImGui::Checkbox(item.c_str(), &on)) {
      if (on) {
        v.push_back(val);
      } else {
        for (auto it = v.begin(); it != v.end(); ++it) {
          if (*it == val) { v.erase(it); break; }
        }
      }
      changed = true;
    }
  }
  return InstantShouldCommit(ctx, changed);
}

template <class Target>
bool DrawRef(EditorContext& ctx, const char* label, ps::Ref<Target>& ref) {
  const std::vector<mj::ElementType> targets = sdkd::RefTargetTypes<Target>();
  const std::vector<std::string> cands =
      RefCandidates(*ctx.tree, targets);
  const bool dangling = RefIsDangling(cands, ref.name);
  bool changed = false;
  ImGui::SetNextItemWidth(kFieldWidth * 1.8f);
  std::string preview = ref.name.empty() ? "(none)" : ref.name;
  if (dangling) preview += "  [!]";
  if (ImGui::BeginCombo(label, preview.c_str())) {
    if (ImGui::Selectable("(none)", ref.name.empty())) {
      ref.name.clear();
      changed = true;
    }
    for (const std::string& c : cands) {
      if (ImGui::Selectable(c.c_str(), c == ref.name)) {
        ref.name = c;
        changed = true;
      }
    }
    // A dangling value is kept selectable so an edit is deliberate, not forced.
    if (dangling) {
      const std::string keep = ref.name + "  (missing)";
      ImGui::Selectable(keep.c_str(), true);
    }
    ImGui::EndCombo();
  }
  if (dangling) {
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.95f, 0.6f, 0.2f, 1.0f), "dangling");
  }
  return InstantShouldCommit(ctx, changed);
}

// Renders each field of a variant's active arm (arm fields are plain, non-opt);
// hoisted to namespace scope because a function-local class cannot carry the
// template member functions the Visit hook requires.
struct ArmFieldVisitor {
  EditorContext& ctx;
  bool* commit;
  template <class U>
  void field(int, const char* name, U& value) {
    if (DrawValue(ctx, name, value, 0)) *commit = true;
  }
  template <class C>
  void child(int, const char*, C&) {}
  template <class C>
  void union_child(int, const char*, C&) {}
};

// Renders a variant's arm selector plus the active arm's fields.
template <class V>
bool DrawVariant(EditorContext& ctx, const char* label, V& work) {
  bool commit = false;
  (void)label;
  const std::vector<const char*> labels = ArmLabels<V>();
  const std::size_t cur = work.index();
  ImGui::SetNextItemWidth(kFieldWidth * 1.4f);
  if (ImGui::BeginCombo("##arm", labels[cur])) {
    for (std::size_t i = 0; i < labels.size(); ++i) {
      if (ImGui::Selectable(labels[i], i == cur) && i != cur) {
        SetArmByIndex(work, i);
        commit = true;
      }
    }
    ImGui::EndCombo();
  }
  commit = InstantShouldCommit(ctx, commit) || commit;

  ImGui::Indent();
  ArmFieldVisitor av{ctx, &commit};
  std::visit(
      [&](auto& arm) {
        // A struct arm exposes fields through Visit; a scalar arm (e.g. the
        // enum arm of TextureSource) is edited directly.
        if constexpr (requires { mj::Visit(arm, av); }) {
          mj::Visit(arm, av);
        } else {
          if (DrawValue(ctx, "##armval", arm, 0)) commit = true;
        }
      },
      work);
  ImGui::Unindent();
  return commit;
}

template <class Inner>
bool DrawValue(EditorContext& ctx, const char* label, Inner& work,
               int arity_min) {
  if constexpr (std::is_same_v<Inner, bool>) {
    const bool changed = ImGui::Checkbox(label, &work);
    return InstantShouldCommit(ctx, changed);
  } else if constexpr (std::is_same_v<Inner, std::string>) {
    ImGui::SetNextItemWidth(kFieldWidth * 2.2f);
    ImGui::InputText(label, &work);
    return GestureShouldCommit(ctx);
  } else if constexpr (std::is_enum_v<Inner>) {
    return DrawEnum(ctx, label, work);
  } else if constexpr (std::is_floating_point_v<Inner>) {
    return DrawScalarReal(ctx, label, work);
  } else if constexpr (std::is_integral_v<Inner>) {
    return DrawScalarInt(ctx, label, work);
  } else if constexpr (sdkd::is_ref<Inner>::value) {
    return DrawRef(ctx, label, work);
  } else if constexpr (is_std_array<Inner>::value) {
    return DrawNumericRow(ctx, work.data(), is_std_array<Inner>::size);
  } else if constexpr (is_inline_vec<Inner>::value) {
    bool commit = false;
    std::size_t n = work.size();
    if (ImGui::SmallButton("-") && InlineVecCanShrink(n, arity_min)) {
      work.resize(n - 1);
      commit = InstantShouldCommit(ctx, true) || commit;
      n = work.size();
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("+") &&
        InlineVecCanGrow(n, is_inline_vec<Inner>::capacity)) {
      work.resize(n + 1);
      commit = InstantShouldCommit(ctx, true) || commit;
      n = work.size();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(%zu/%zu)", n, is_inline_vec<Inner>::capacity);
    if (n) {
      commit = DrawNumericRow(ctx, &work[0], n) || commit;
    }
    return commit;
  } else if constexpr (is_std_vector<Inner>::value) {
    using X = typename is_std_vector<Inner>::elem;
    if constexpr (std::is_enum_v<X>) {
      return DrawEnumSet(ctx, work);
    } else if constexpr (sdkd::is_ref<X>::value) {
      // ref<T>[] (e.g. a flex's body list): one ref combo per entry, +/- to
      // grow/shrink -- the vector row's shape with DrawRef as the cell.
      bool commit = false;
      if (ImGui::SmallButton("-") && !work.empty()) {
        work.pop_back();
        commit = InstantShouldCommit(ctx, true) || commit;
      }
      ImGui::SameLine();
      if (ImGui::SmallButton("+")) {
        work.push_back(X{});
        commit = InstantShouldCommit(ctx, true) || commit;
      }
      ImGui::SameLine();
      ImGui::TextDisabled("(%zu)", work.size());
      for (std::size_t i = 0; i < work.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        commit = DrawRef(ctx, "##entry", work[i]) || commit;
        ImGui::PopID();
      }
      return commit;
    } else {
      bool commit = false;
      if (ImGui::SmallButton("-") && !work.empty()) {
        work.pop_back();
        commit = InstantShouldCommit(ctx, true) || commit;
      }
      ImGui::SameLine();
      if (ImGui::SmallButton("+")) {
        work.push_back(X{});
        commit = InstantShouldCommit(ctx, true) || commit;
      }
      ImGui::SameLine();
      ImGui::TextDisabled("(%zu)", work.size());
      if (!work.empty()) {
        commit = DrawNumericRow(ctx, work.data(), work.size()) || commit;
      }
      return commit;
    }
  } else if constexpr (is_variant<Inner>::value) {
    return DrawVariant(ctx, label, work);
  } else {
    ImGui::TextDisabled("%s: unsupported", label);
    return false;
  }
}

// The outcome of a field edit widget: `changed` is true on any frame the value
// moved (so the caller can materialise it into the field live), `commit` is true
// on the frame the gesture ends (one undo entry + one recompile). For plain drag/
// text widgets the two coincide (ImGui holds the in-progress value); a ColorEdit
// picker separates them -- see DrawColorArray.
struct FieldEdit {
  bool changed = false;
  bool commit = false;
  explicit operator bool() const { return changed || commit; }
};

// --- Colour swatch/picker -------------------------------------------------- //
// A ColorEdit widget bound to a fixed 3/4-wide float or double array. ImGui only
// edits floats, so a double-backed field round-trips through a float scratch.
template <class X, std::size_t N>
FieldEdit DrawColorArray(EditorContext& ctx, const char* label,
                         std::array<X, N>& a) {
  static_assert(N == 3 || N == 4, "color array is rgb(3) or rgba(4)");
  float tmp[4] = {0, 0, 0, 1};
  for (std::size_t i = 0; i < N; ++i) tmp[i] = static_cast<float>(a[i]);
  ImGui::SetNextItemWidth(kFieldWidth * 2.2f);
  const ImGuiColorEditFlags flags =
      ImGuiColorEditFlags_AlphaBar | ImGuiColorEditFlags_AlphaPreviewHalf;
  const bool changed =
      (N == 4) ? ImGui::ColorEdit4(label, tmp, flags)
               : ImGui::ColorEdit3(label, tmp, ImGuiColorEditFlags_None);
  // ColorEdit's popup does not re-apply its pick on the deactivation (commit)
  // frame, and the caller re-seeds `work` from the field every frame, so
  // committing on release alone writes the ORIGINAL value back ("snap back").
  // Report `changed` so the caller materialises the picked value live (surviving
  // the re-seed); still `commit` once, on release, for the undo entry + recompile.
  if (ImGui::IsItemActivated()) EditBegin(ctx);
  const bool commit = ImGui::IsItemDeactivatedAfterEdit();
  if (ImGui::IsItemDeactivated() && !commit) EditCancel(ctx);
  if (changed || commit) {
    for (std::size_t i = 0; i < N; ++i) a[i] = static_cast<X>(tmp[i]);
  }
  return {changed, commit};
}

// Draw a field's value, preferring a colour picker for recognised rgb(a) fields
// (falls back to the generic type-driven DrawValue for everything else). For the
// non-colour widgets `changed` and `commit` coincide (the value is only read back
// on commit), so their edit reports {c, c}.
template <class Inner>
FieldEdit DrawFieldValue(EditorContext& ctx, const reflect::FieldDescriptor& fd,
                         const char* label, Inner& work, int arity_min) {
  if constexpr (is_std_array<Inner>::value) {
    using X = typename is_std_array<Inner>::elem;
    if constexpr (std::is_floating_point_v<X> &&
                  (is_std_array<Inner>::size == 3 ||
                   is_std_array<Inner>::size == 4)) {
      const ColorKind ck = ColorKindOf(fd);
      if ((ck == ColorKind::Rgba4 && is_std_array<Inner>::size == 4) ||
          (ck == ColorKind::Rgb3 && is_std_array<Inner>::size == 3)) {
        return DrawColorArray(ctx, label, work);
      }
    }
  }
  const bool c = DrawValue(ctx, label, work, arity_min);
  return {c, c};
}

// --- Field rows (presence-aware) ------------------------------------------- //

std::string RowLabel(const reflect::FieldDescriptor& fd) {
  return std::string("##") + std::string(fd.name);
}

// The pixel column where every field's value widget begins. One constant keeps
// names and value widgets aligned across every row of the panel.
constexpr float kLabelColumn = 180.0f;

// Draws a field's name label and, when the schema carries a description for it,
// attaches that description as a hover tooltip. The doc text is generated into
// each reflect::FieldDescriptor from the schema comment, so every field the
// model can hold documents itself here at zero per-field cost.
void FieldLabel(std::string_view name, std::string_view doc) {
  ImGui::TextUnformatted(name.data(), name.data() + name.size());
  if (!doc.empty()) {
    ImGui::SetItemTooltip("%.*s", static_cast<int>(doc.size()), doc.data());
  }
}

void DrawBadge(Presence pres) {
  std::string_view badge = PresenceBadge(pres);
  if (badge.empty()) return;
  ImGui::SameLine();
  const ImVec4 col = pres == Presence::Inherited
                         ? ImVec4(0.5f, 0.7f, 0.95f, 1.0f)
                         : ImVec4(0.6f, 0.6f, 0.6f, 1.0f);
  ImGui::TextColored(col, "[%.*s]", static_cast<int>(badge.size()),
                     badge.data());
}

// One optional (presence-tracked) field: name, badge, value widget, revert.
template <class E, class Inner>
void RowOptional(EditorContext& ctx, const reflect::FieldDescriptor& fd, int id,
                 ps::opt<Inner>& slot, const E* eff_class, const E* eff_full) {
  const bool authored = slot.has_value();
  const ps::opt<Inner>* clsF =
      eff_class ? sdkd::FieldAt<E, ps::opt<Inner>>(*eff_class, id) : nullptr;
  const ps::opt<Inner>* fullF =
      eff_full ? sdkd::FieldAt<E, ps::opt<Inner>>(*eff_full, id) : nullptr;
  const bool inherited = clsF && clsF->has_value();
  const bool has_default = fullF && fullF->has_value();
  const Presence pres =
      PresenceFromLayers(fd.optional, authored, inherited, has_default);

  ImGui::PushID(id);
  FieldLabel(fd.name, fd.doc);
  DrawBadge(pres);
  ImGui::SameLine(kLabelColumn);

  // Always edit a temp: BeginEdit (fired inside DrawValue on activation) snapshots
  // the still-pristine tree, and the field is written only on commit -- so undo
  // captures the pre-edit state and an unset field materialises exactly on edit.
  const std::string label = RowLabel(fd);
  Inner work = SeedValue<Inner>(authored, slot, clsF, fullF);
  {
    Grayed g(!authored);
    const FieldEdit e = DrawFieldValue(ctx, fd, label.c_str(), work, fd.arity_min);
    // Materialise on any change so a multi-frame gesture (e.g. a colour picker,
    // whose value the caller would otherwise re-seed away each frame) is not lost;
    // record the undo entry + recompile once, on commit.
    if (e.changed) slot = work;
    if (e.commit) {
      slot = std::move(work);
      EditCommit(ctx, std::string("edit ") + std::string(fd.name));
    }
  }
  if (authored) {
    ImGui::SameLine();
    if (ImGui::SmallButton("x")) {  // revert: clear presence -> inherited/default
      EditBegin(ctx);
      slot.reset();
      EditCommit(ctx, std::string("revert ") + std::string(fd.name));
    }
  }
  ImGui::PopID();
}

// One required (non-optional) field: plain, no presence.
template <class Inner>
void RowRequired(EditorContext& ctx, const reflect::FieldDescriptor& fd, int id,
                 Inner& slot) {
  ImGui::PushID(id);
  FieldLabel(fd.name, fd.doc);
  ImGui::SameLine(kLabelColumn);
  Inner work = slot;
  const std::string label = RowLabel(fd);
  const FieldEdit e = DrawFieldValue(ctx, fd, label.c_str(), work, fd.arity_min);
  if (e.changed) slot = work;
  if (e.commit) {
    slot = std::move(work);
    EditCommit(ctx, std::string("edit ") + std::string(fd.name));
  }
  ImGui::PopID();
}

// --- The element visitor --------------------------------------------------- //
template <class E>
struct RowVisitor {
  EditorContext& ctx;
  const reflect::ElementDescriptor& desc;
  const E* eff_class;
  const E* eff_full;
  bool transform_phase;

  template <class T>
  void field(int id, const char*, T& value) {
    if (id < 0 || id >= static_cast<int>(desc.field_count)) return;
    const reflect::FieldDescriptor& fd = desc.fields[id];
    // The name field is edited through the rename op (referrers rewritten
    // atomically), NOT written in place -- so it is drawn by RenderNameRow and
    // skipped here (SE1b reconciliation).
    if (fd.xml == "name") return;
    if (IsTransformField(fd) != transform_phase) return;
    if constexpr (sdkd::is_opt<T>::value) {
      using Inner = typename sdkd::is_opt<T>::inner;
      RowOptional<E, Inner>(ctx, fd, id, value, eff_class, eff_full);
    } else {
      RowRequired<T>(ctx, fd, id, value);
    }
  }
  template <class C>
  void child(int, const char*, C&) {}
  template <class C>
  void union_child(int, const char*, C&) {}
};

template <class E>
bool AnyTransformField(const reflect::ElementDescriptor& desc) {
  for (std::size_t i = 0; i < desc.field_count; ++i) {
    if (IsTransformField(desc.fields[i])) return true;
  }
  return false;
}

// True for an element carrying an `opt<string> name` field (excludes Default,
// whose identity is `dclass` and whose rename is owned by the Hierarchy).
template <class E>
constexpr bool ElemHasName() {
  if constexpr (std::is_same_v<E, mj::Default>) {
    return false;
  } else if constexpr (requires(E& x) { x.name; }) {
    using NT = std::decay_t<decltype(std::declval<E&>().name)>;
    return sdkd::is_opt<NT>::value &&
           std::is_same_v<typename sdkd::is_opt<NT>::inner, std::string>;
  } else {
    return false;
  }
}

// The element's name field, routed through the rename op so every typed referrer
// is rewritten atomically (a direct write would leave danglers). Classes keep the
// existing in-place editor (the Hierarchy owns class rename).
template <class E>
void RenderNameRow(EditorContext& ctx, E& e) {
  if constexpr (ElemHasName<E>()) {
    const std::string* nm = sdkd::NameOf(e);
    const std::string cur = nm ? *nm : "";
    std::string work = cur;
    ImGui::PushID("name_field");
    FieldLabel("name", "element name (unique; referrers update on rename)");
    ImGui::SameLine(kLabelColumn);
    ImGui::SetNextItemWidth(kFieldWidth * 2.2f);
    ImGui::InputText("##name", &work);
    if (GestureShouldCommit(ctx)) {
      if (!work.empty() && work != cur &&
          ps::studio::RenameBySerial(ctx, e.serial, work) >= 0) {
        EditCommit(ctx, "rename");
        ps::studio::SelectBySerial(ctx, e.serial);
      } else {
        // Rejected (empty / reserved prefix / name in use) or no effective
        // change: nothing was applied, drop the pending snapshot. The field
        // snaps back to the current name on refresh.
        if (!work.empty() && work != cur) {
          ctx.Log("rename rejected: '" + work +
                  "' (reserved prefix or already in use)");
        }
        EditCancel(ctx);
      }
    }
    ImGui::PopID();
  }
}

// A Material's texture <layer>s are an owned child list, so the generic field
// visitor (which only walks scalar/array fields) never reaches them. This is the
// one element whose appearance authoring needs its child list edited in place:
// per layer a texture reference + a role, with add / remove. Each change is its
// own undo entry, and every mutation routes through the windowless-tested helpers
// in details_panel.h. Rows follow the panel's label-column layout so the layer
// combos line up with every other field's value column. SE5 "full material and
// texturing editing".
inline void RenderMaterialTextureLayers(EditorContext& ctx, mj::Material& mat) {
  if (!ImGui::CollapsingHeader("Texture Layers", ImGuiTreeNodeFlags_DefaultOpen))
    return;
  const std::vector<std::string> cands =
      RefCandidates(*ctx.tree, {mj::ElementType::Texture});
  const std::vector<std::string_view> roles = EnumLabels<mj::TexRole>();

  if (mat.layers.empty()) {
    ImGui::TextDisabled("No texture layers.");
  }

  int remove_idx = -1;
  for (std::size_t i = 0; i < mat.layers.size(); ++i) {
    mj::MaterialLayer* layer = mat.layers[i].get();
    if (!layer) continue;
    ImGui::PushID(static_cast<int>(i));

    const std::string row = "layer " + std::to_string(i);
    FieldLabel(row, "one texture channel: the source texture and the role it feeds");
    ImGui::SameLine(kLabelColumn);

    // Texture reference combo.
    const std::string cur = LayerTextureName(*layer);
    ImGui::SetNextItemWidth(kFieldWidth * 1.8f);
    if (ImGui::BeginCombo("##tex", cur.empty() ? "(none)" : cur.c_str())) {
      if (ImGui::Selectable("(none)", cur.empty())) {
        EditBegin(ctx);
        SetLayerTexture(*layer, "");
        EditCommit(ctx, "edit layer texture");
      }
      for (const std::string& c : cands) {
        if (ImGui::Selectable(c.c_str(), c == cur)) {
          EditBegin(ctx);
          SetLayerTexture(*layer, c);
          EditCommit(ctx, "edit layer texture");
        }
      }
      ImGui::EndCombo();
    }
    ImGui::SameLine();

    // Role combo.
    const int role_cur = LayerRoleIndex(*layer);
    const std::string role_prev =
        (role_cur >= 0 && role_cur < static_cast<int>(roles.size()))
            ? std::string(roles[role_cur])
            : std::string("rgb");
    ImGui::SetNextItemWidth(kFieldWidth * 1.4f);
    if (ImGui::BeginCombo("##role", role_prev.c_str())) {
      for (std::size_t r = 0; r < roles.size(); ++r) {
        if (ImGui::Selectable(std::string(roles[r]).c_str(),
                              static_cast<int>(r) == role_cur)) {
          EditBegin(ctx);
          SetLayerRole(*layer, static_cast<mj::TexRole>(r));
          EditCommit(ctx, "edit layer role");
        }
      }
      ImGui::EndCombo();
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("x")) remove_idx = static_cast<int>(i);
    ImGui::PopID();
  }

  if (remove_idx >= 0) {
    EditBegin(ctx);
    RemoveMaterialLayer(mat, static_cast<std::size_t>(remove_idx));
    EditCommit(ctx, "remove texture layer");
    return;  // mat.layers mutated; stop iterating it this frame
  }
  if (ImGui::SmallButton("+ Add layer")) {
    EditBegin(ctx);
    AddMaterialLayer(mat, cands.empty() ? std::string_view{} : cands.front());
    EditCommit(ctx, "add texture layer");
  }
}

template <class E>
void RenderElement(EditorContext& ctx, E& e) {
  const reflect::ElementDescriptor& desc =
      reflect::Describe(mj::element_type_of<E>::value);

  // Read-only identity surface.
  ImGui::TextDisabled("%s", std::string(desc.name).c_str());
  ImGui::Separator();
  ImGui::TextDisabled("serial %llu",
                      static_cast<unsigned long long>(e.serial));
  if (std::optional<int> bid = ctx.compiled.binding.Id(e)) {
    ImGui::SameLine();
    ImGui::TextDisabled("| binding id %d", *bid);
  }
  if (!e.loc.file.empty()) {
    ImGui::TextDisabled("%s:%d", e.loc.file.c_str(), e.loc.line);
  }
  ImGui::Separator();

  RenderNameRow(ctx, e);

  const sdk::EffectiveContext ectx(*ctx.tree);  // one lookup build, two merges
  std::unique_ptr<E> eff_class = sdk::Effective(ectx, e, false);
  std::unique_ptr<E> eff_full = sdk::Effective(ectx, e, true);

  if (AnyTransformField<E>(desc)) {
    if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
      RowVisitor<E> v{ctx, desc, eff_class.get(), eff_full.get(), true};
      mj::Visit(e, v);
    }
  }
  if (ImGui::CollapsingHeader("Properties", ImGuiTreeNodeFlags_DefaultOpen)) {
    RowVisitor<E> v{ctx, desc, eff_class.get(), eff_full.get(), false};
    mj::Visit(e, v);
  }
  if constexpr (std::is_same_v<E, mj::Material>) {
    RenderMaterialTextureLayers(ctx, e);
  }
}

void DetailsUpdate(GuiPlugin* self) {
  EditorContext* c = static_cast<EditorContext*>(self->data);
  if (!c->tree) {
    ImGui::TextUnformatted("No model loaded.");
    return;
  }
  if (c->selected_serial == 0) {
    ImGui::TextUnformatted("Nothing selected.");
    ImGui::TextDisabled("Click an element in the viewport or Hierarchy.");
    return;
  }
  // Rendering an element edits the tree the walk is iterating: a cancelled
  // gesture restores a snapshot over `tree`, and the layer rows add/remove
  // children. Both free memory the walk still holds. Locate the element here
  // and render it once the traversal has finished.
  std::function<void()> render;
  std::string elem_layer_key;
  sdkd::WalkModelAll(*c->tree, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if constexpr (!std::is_same_v<E, mj::Model> && requires { e.serial; }) {
      if (!render && e.serial == c->selected_serial) {
        render = [c, &e] { RenderElement(*c, e); };
        elem_layer_key = e.loc.file;
      }
    }
  });
  if (render) {
    // Two gates, both visible rather than silently inert: the reset-pose rule
    // (CanEdit) and the layer edit scope -- an element owned by an inactive
    // layer stays readable, with a one-click affordance to switch scope.
    const int owner = LayerIndexForKey(*c, elem_layer_key);
    const bool foreign = c->layers.size() > 1 && owner >= 0 &&
                         owner != c->active_layer;
    if (foreign) {
      ImGui::TextDisabled("Owned by layer '%s' (not the active layer).",
                          c->layers[owner].name.c_str());
      ImGui::SameLine();
      if (ImGui::SmallButton(
              ("Edit layer '" + c->layers[owner].name + "'").c_str())) {
        c->SetActiveLayer(owner);
      }
      ImGui::Separator();
    }
    const bool editable = c->CanEdit() && !foreign;
    if (!c->CanEdit()) {
      ImGui::TextDisabled("Simulation running or advanced -- press Stop to edit.");
      ImGui::Separator();
    }
    ImGui::BeginDisabled(!editable);
    render();
    ImGui::EndDisabled();
  } else {
    ImGui::TextDisabled("Selected element is not in the current model.");
  }
}

}  // namespace

void RegisterDetailsPanel(EditorContext& ctx) {
  GuiPlugin plugin;
  plugin.name = "Details";
  plugin.update = DetailsUpdate;
  plugin.active = true;
  plugin.data = &ctx;
  RegisterPlugin<GuiPlugin>(plugin);
}

}  // namespace ps::studio::details
