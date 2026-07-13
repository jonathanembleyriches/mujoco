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
#include <string>
#include <vector>

#include <imgui.h>
#include <imgui_stdlib.h>

#include "binding.h"
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
// The five variant families are a fixed schema feature; their POD arms carry no
// reflection entry, so their selector labels are named here (the only
// schema-shaped table in the renderer).
template <class A>
const char* ArmLabel() {
  if constexpr (std::is_same_v<A, mj::Quat>) return "quat";
  else if constexpr (std::is_same_v<A, mj::AxisAngle>) return "axisangle";
  else if constexpr (std::is_same_v<A, mj::XYAxes>) return "xyaxes";
  else if constexpr (std::is_same_v<A, mj::ZAxis>) return "zaxis";
  else if constexpr (std::is_same_v<A, mj::Euler>) return "euler";
  else if constexpr (std::is_same_v<A, mj::Explicit>) return "size";
  else if constexpr (std::is_same_v<A, mj::FromTo>) return "fromto";
  else if constexpr (std::is_same_v<A, mj::DiagInertia>) return "diaginertia";
  else if constexpr (std::is_same_v<A, mj::FullInertia>) return "fullinertia";
  else if constexpr (std::is_same_v<A, mj::Fovy>) return "fovy";
  else if constexpr (std::is_same_v<A, mj::Focal>) return "focal";
  else if constexpr (std::is_same_v<A, mj::TexFile>) return "file";
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
  ImGui::DragScalar(label, ImGuiDataType_S64, &v, 0.2f);
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
    if constexpr (std::is_floating_point_v<X>) {
      ImGui::DragScalar("##c",
                        std::is_same_v<X, float> ? ImGuiDataType_Float
                                                 : ImGuiDataType_Double,
                        &data[i], 0.01f);
    } else {
      ImGui::DragScalar("##c", ImGuiDataType_S64, &data[i], 0.2f);
    }
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

// --- Field rows (presence-aware) ------------------------------------------- //

std::string RowLabel(const reflect::FieldDescriptor& fd) {
  return std::string("##") + std::string(fd.name);
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
  ImGui::TextUnformatted(std::string(fd.name).c_str());
  DrawBadge(pres);
  ImGui::SameLine(180.0f);

  // Always edit a temp: BeginEdit (fired inside DrawValue on activation) snapshots
  // the still-pristine tree, and the field is written only on commit -- so undo
  // captures the pre-edit state and an unset field materialises exactly on edit.
  const std::string label = RowLabel(fd);
  Inner work = authored ? *slot
                        : (inherited ? **clsF : (has_default ? **fullF : Inner{}));
  {
    Grayed g(!authored);
    if (DrawValue(ctx, label.c_str(), work, fd.arity_min)) {
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
  ImGui::TextUnformatted(std::string(fd.name).c_str());
  ImGui::SameLine(180.0f);
  Inner work = slot;
  const std::string label = RowLabel(fd);
  if (DrawValue(ctx, label.c_str(), work, fd.arity_min)) {
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

  std::unique_ptr<E> eff_class = sdk::Effective(*ctx.tree, e, false);
  std::unique_ptr<E> eff_full = sdk::Effective(*ctx.tree, e, true);

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
}

void DetailsUpdate(GuiPlugin* self) {
  EditorContext* c = static_cast<EditorContext*>(self->data);
  if (!c->tree) {
    ImGui::TextUnformatted("No model loaded.");
    return;
  }
  if (c->selected_serial == 0) {
    ImGui::TextUnformatted("No selection.");
    ImGui::TextDisabled("Select an element in the Hierarchy or viewport.");
    return;
  }
  bool found = false;
  sdkd::WalkModelAll(*c->tree, [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if constexpr (!std::is_same_v<E, mj::Model> && requires { e.serial; }) {
      if (!found && e.serial == c->selected_serial) {
        found = true;
        RenderElement(*c, e);
      }
    }
  });
  if (!found) {
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
