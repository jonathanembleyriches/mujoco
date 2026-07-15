// ProtoSpec Studio: windowless tests for the generated Details panel (ps::studio).
//
// No SDL/ImGui: everything the panel decides -- which widget a field maps to,
// how presence layers resolve, how a Ref combo is populated, how enum tables and
// InlineVec bounds behave -- lives in details_panel.h as pure functions and is
// exercised here directly. The headline is the reflection-coverage test: every
// field of every one of the schema's element descriptors must classify to a real
// widget, so a schema change that introduces an unhandled shape fails loudly.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "editor/details_panel.h"
#include "keywords.h"
#include "mjcf.h"
#include "protospec/classes.h"
#include "reflect.h"
#include "types.h"

namespace det = ps::studio::details;
namespace mj = ps::mjcf;
namespace reflect = ps::mjcf::reflect;
namespace sdk = ps::sdk;
namespace sdkd = ps::sdk::detail;

static int g_failed = 0;
static int g_checks = 0;

#define CHECK(cond)                                                \
  do {                                                             \
    ++g_checks;                                                    \
    if (!(cond)) {                                                 \
      ++g_failed;                                                  \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
    }                                                              \
  } while (0)

// --- 1. Reflection coverage ------------------------------------------------ //
// Every field of every descriptor maps to a concrete widget. Any Unhandled is a
// schema shape the renderer would silently drop, so it is a hard failure that
// names the offending element/field.
static void TestWidgetCoverage() {
  int fields_seen = 0;
  int unhandled = 0;
  for (std::size_t i = 0; i < reflect::ElementCount(); ++i) {
    const reflect::ElementDescriptor& e = reflect::ElementAt(i);
    for (std::size_t f = 0; f < e.field_count; ++f) {
      const reflect::FieldDescriptor& fd = e.fields[f];
      ++fields_seen;
      const det::WidgetKind w = det::ClassifyField(fd);
      if (w == det::WidgetKind::Unhandled) {
        ++unhandled;
        std::printf("  UNHANDLED %.*s.%.*s\n",
                    static_cast<int>(e.name.size()), e.name.data(),
                    static_cast<int>(fd.name.size()), fd.name.data());
      }
    }
  }
  CHECK(fields_seen > 1000);  // the full schema, not an empty walk
  CHECK(unhandled == 0);
  std::printf("  coverage: %d fields across %zu descriptors, %d unhandled\n",
              fields_seen, reflect::ElementCount(), unhandled);
}

// The specific mappings the design calls out by name.
static void TestWidgetMappingSpecifics() {
  using det::ClassifyField;
  using det::WidgetKind;
  const reflect::ElementDescriptor& geom =
      reflect::Describe(mj::ElementType::Geom);
  auto field = [&](const reflect::ElementDescriptor& d, std::string_view name)
      -> const reflect::FieldDescriptor* {
    for (std::size_t i = 0; i < d.field_count; ++i)
      if (d.fields[i].name == name) return &d.fields[i];
    return nullptr;
  };

  CHECK(ClassifyField(*field(geom, "shellinertia")) == WidgetKind::Checkbox);
  CHECK(ClassifyField(*field(geom, "condim")) == WidgetKind::IntScalar);
  CHECK(ClassifyField(*field(geom, "mass")) == WidgetKind::RealScalar);
  CHECK(ClassifyField(*field(geom, "name")) == WidgetKind::Text);
  CHECK(ClassifyField(*field(geom, "type")) == WidgetKind::EnumCombo);
  CHECK(ClassifyField(*field(geom, "material")) == WidgetKind::RefCombo);
  // Q-ORIENT: orientation is now the canonical `quat` (a fixed double[4]) ->
  // RealRow; the surviving GeomShape (`shape`) is the Variant widget.
  CHECK(ClassifyField(*field(geom, "quat")) == WidgetKind::RealRow);   // fixed[4]
  CHECK(ClassifyField(*field(geom, "shape")) == WidgetKind::Variant);
  CHECK(ClassifyField(*field(geom, "size")) == WidgetKind::RealRow);   // range
  CHECK(ClassifyField(*field(geom, "rgba")) == WidgetKind::RealRow);   // fixed
  CHECK(ClassifyField(*field(geom, "user")) == WidgetKind::RealRow);   // unbounded

  // The keyword-set: Camera.output is std::vector<Enum> -> EnumSet.
  const reflect::ElementDescriptor& cam =
      reflect::Describe(mj::ElementType::Camera);
  CHECK(ClassifyField(*field(cam, "output")) == WidgetKind::EnumSet);
}

// --- 2. Presence layering -------------------------------------------------- //
static const reflect::FieldDescriptor* FindField(
    const reflect::ElementDescriptor& d, std::string_view name, int& id_out) {
  for (std::size_t i = 0; i < d.field_count; ++i) {
    if (d.fields[i].name == name) {
      id_out = static_cast<int>(i);
      return &d.fields[i];
    }
  }
  id_out = -1;
  return nullptr;
}

// A geom that inherits `size` from its class shows the inherited value when
// unset; authoring overrides; reverting (reset) restores inheritance.
static void TestPresenceInherited() {
  mj::Model m;
  auto def = std::make_unique<mj::Default>();
  def->dclass = "wheel";
  auto part = std::make_unique<mj::Geom>();
  part->size = ps::InlineVec<double, 3>{0.3, 0.1, 0.0};
  def->geom.push_back(std::move(part));
  m.defaults.push_back(std::move(def));

  mj::Geom g;
  g.dclass = ps::Ref<mj::Default>("wheel");
  g.name = "w1";

  const reflect::ElementDescriptor& gd =
      reflect::Describe(mj::ElementType::Geom);
  int size_id = -1;
  const reflect::FieldDescriptor* fd = FindField(gd, "size", size_id);
  CHECK(fd != nullptr);

  using OptVec = ps::opt<ps::InlineVec<double, 3>>;

  auto classify = [&](const mj::Geom& e) {
    const bool authored =
        sdkd::FieldAt<mj::Geom, OptVec>(e, size_id)->has_value();
    std::unique_ptr<mj::Geom> ec = sdk::Effective(m, e, false);
    std::unique_ptr<mj::Geom> ef = sdk::Effective(m, e, true);
    const bool inherited =
        sdkd::FieldAt<mj::Geom, OptVec>(*ec, size_id)->has_value();
    const bool has_default =
        sdkd::FieldAt<mj::Geom, OptVec>(*ef, size_id)->has_value();
    return det::PresenceFromLayers(fd->optional, authored, inherited,
                                   has_default);
  };

  // Unset on the element, present on the class chain -> inherited.
  CHECK(classify(g) == det::Presence::Inherited);
  {
    std::unique_ptr<mj::Geom> ec = sdk::Effective(m, g, false);
    CHECK(ec->size.has_value());
    CHECK((*ec->size)[0] == 0.3);  // the class value flows through
  }

  // Author it -> materialised, the panel would drop the badge.
  g.size = ps::InlineVec<double, 3>{1.0, 1.0, 1.0};
  CHECK(classify(g) == det::Presence::Authored);

  // Revert (what the per-field clear does) -> inheritance restored.
  g.size.reset();
  CHECK(classify(g) == det::Presence::Inherited);
}

// A defaultable field with no class involvement but an IDL default shows the
// default, not "unset".
static void TestPresenceIdlDefault() {
  mj::Model m;
  mj::ActuatorGeneral a;  // no class, nothing authored

  const reflect::ElementDescriptor& ad =
      reflect::Describe(mj::ElementType::ActuatorGeneral);
  int dyn_id = -1;
  const reflect::FieldDescriptor* fd = FindField(ad, "dyntype", dyn_id);
  CHECK(fd != nullptr);
  CHECK(fd->has_default);  // schema says dyntype has an IDL default

  using OptDyn = ps::opt<mj::DynType>;
  const bool authored =
      sdkd::FieldAt<mj::ActuatorGeneral, OptDyn>(a, dyn_id)->has_value();
  std::unique_ptr<mj::ActuatorGeneral> ec = sdk::Effective(m, a, false);
  std::unique_ptr<mj::ActuatorGeneral> ef = sdk::Effective(m, a, true);
  const bool inherited =
      sdkd::FieldAt<mj::ActuatorGeneral, OptDyn>(*ec, dyn_id)->has_value();
  const bool has_default =
      sdkd::FieldAt<mj::ActuatorGeneral, OptDyn>(*ef, dyn_id)->has_value();

  CHECK(!authored);
  CHECK(!inherited);
  CHECK(has_default);
  CHECK(det::PresenceFromLayers(fd->optional, authored, inherited,
                               has_default) == det::Presence::DefaultIdl);
}

static void TestPresenceRequiredAndUnset() {
  // Required (non-optional) -> Required regardless of the other layers.
  CHECK(det::PresenceFromLayers(false, false, false, false) ==
        det::Presence::Required);
  // Optional, nothing anywhere -> Unset.
  CHECK(det::PresenceFromLayers(true, false, false, false) ==
        det::Presence::Unset);
}

// --- 3. Reference combo population ----------------------------------------- //
static mj::BodyChildAny SiteChild(const std::string& name) {
  auto s = std::make_unique<mj::Site>();
  s->name = name;
  mj::BodyChildAny item;
  item.node = std::move(s);
  return item;
}

static void TestRefTargetsAndCandidates() {
  // Single-element target.
  const std::vector<mj::ElementType> site_t = det::RefTargets("Site");
  CHECK(site_t.size() == 1);
  CHECK(!site_t.empty() && site_t[0] == mj::ElementType::Site);

  // Union target expands to its members (Ref<TendonAny> -> Spatial, Fixed).
  const std::vector<mj::ElementType> tendon_t = det::RefTargets("TendonAny");
  CHECK(tendon_t.size() == 2);
  CHECK(sdkd::Contains(tendon_t, mj::ElementType::Spatial));
  CHECK(sdkd::Contains(tendon_t, mj::ElementType::Fixed));

  // Population walks the whole tree.
  mj::Model m;
  auto body = std::make_unique<mj::Body>();
  body->name = "b";
  body->subtree.push_back(SiteChild("s1"));
  body->subtree.push_back(SiteChild("s2"));
  m.worldbody.push_back(std::move(body));

  const std::vector<std::string> cands = det::RefCandidates(m, site_t);
  CHECK(cands.size() == 2);
  bool has_s1 = false, has_s2 = false;
  for (const std::string& c : cands) {
    has_s1 = has_s1 || c == "s1";
    has_s2 = has_s2 || c == "s2";
  }
  CHECK(has_s1 && has_s2);

  // Dangling detection: a name not among candidates flags; a live one does not;
  // empty is never dangling.
  CHECK(det::RefIsDangling(cands, "s3"));
  CHECK(!det::RefIsDangling(cands, "s1"));
  CHECK(!det::RefIsDangling(cands, ""));
}

// --- 4. Enum keyword round-trip -------------------------------------------- //
static void TestEnumLabels() {
  const std::vector<std::string_view> labels = det::EnumLabels<mj::GeomType>();
  CHECK(labels.size() == 9);  // plane..sdf
  bool has_box = false;
  for (std::size_t i = 0; i < labels.size(); ++i) {
    // Table round-trip: every enumerated label parses back to its own index.
    mj::GeomType parsed{};
    CHECK(mj::FromMjcf(labels[i], parsed));
    CHECK(static_cast<int>(parsed) == static_cast<int>(i));
    CHECK(mj::ToMjcf(parsed) == labels[i]);
    has_box = has_box || labels[i] == "box";
  }
  CHECK(has_box);

  // A TriState (auto_ spelled "auto") also enumerates cleanly.
  const std::vector<std::string_view> tri = det::EnumLabels<mj::TriState>();
  CHECK(tri.size() == 3);
}

// --- 5. InlineVec grow / shrink bounds ------------------------------------- //
static void TestInlineVecBounds() {
  CHECK(det::InlineVecCanGrow(0, 3));
  CHECK(det::InlineVecCanGrow(2, 3));
  CHECK(!det::InlineVecCanGrow(3, 3));

  CHECK(det::InlineVecCanShrink(3, 1));
  CHECK(det::InlineVecCanShrink(2, 1));
  CHECK(!det::InlineVecCanShrink(1, 1));
  CHECK(!det::InlineVecCanShrink(0, 0));
}

// --- 6. Numeric widget width (display fidelity) ---------------------------- //
// The drag widget for a numeric field must read and write exactly the storage
// type's bytes. The original defect drove every integral field with a 64-bit
// drag, so an int32 field's widget read the adjacent stack word as the value's
// high bits and displayed a huge uninitialised number ("super big numbers").
// The width now derives from the storage type via NumericWidgetOf; these assert
// the mapping is width-exact for every arithmetic type the schema uses.
static void TestNumericWidgetWidths() {
  using det::NumericWidget;
  using det::NumericWidgetBytes;
  using det::NumericWidgetOf;
  // The exact regression: an int32 must map to a 4-byte drag, never 8.
  CHECK(NumericWidgetOf<int32_t>() == NumericWidget::S32);
  CHECK(NumericWidgetBytes(NumericWidgetOf<int32_t>()) == 4);
  CHECK(NumericWidgetBytes(NumericWidgetOf<int32_t>()) != 8);
  // The mapping is width- and sign-exact across the arithmetic types.
  CHECK(NumericWidgetBytes(NumericWidgetOf<double>()) == sizeof(double));
  CHECK(NumericWidgetBytes(NumericWidgetOf<float>()) == sizeof(float));
  CHECK(NumericWidgetBytes(NumericWidgetOf<std::int64_t>()) == 8);
  CHECK(NumericWidgetBytes(NumericWidgetOf<std::uint64_t>()) == 8);
  CHECK(NumericWidgetBytes(NumericWidgetOf<std::int16_t>()) == 2);
  CHECK(NumericWidgetBytes(NumericWidgetOf<std::uint32_t>()) == 4);
  CHECK(NumericWidgetOf<std::uint32_t>() == NumericWidget::U32);
  CHECK(NumericWidgetOf<double>() == NumericWidget::F64);
}

// --- 7. Display-value fidelity sweep --------------------------------------- //
// Walk the panel's model windowlessly and assert that every value the Details
// panel would display is drawn from a real layer -- the authored value, the
// class/IDL effective value, or the type's defined zero -- and never left
// uninitialised, and that every numeric field's drag widget is exactly as wide
// as its storage. This is the automated coverage the certification previously
// lacked: widget-kind mapping was checked, value display fidelity was not.

// Assert the drag widget for one numeric leaf type reads exactly its bytes.
template <class X>
static void CheckNumericWidth() {
  if constexpr (std::is_arithmetic_v<X> && !std::is_same_v<X, bool>) {
    CHECK(det::NumericWidgetBytes(det::NumericWidgetOf<X>()) == sizeof(X));
  }
}

// Run the width check on whatever numeric leaf a field storage type bottoms out
// in (scalar, std::array, InlineVec, or std::vector element).
template <class Inner>
static void AuditNumericWidth() {
  if constexpr (det::is_std_array<Inner>::value) {
    CheckNumericWidth<typename det::is_std_array<Inner>::elem>();
  } else if constexpr (det::is_inline_vec<Inner>::value) {
    CheckNumericWidth<typename det::is_inline_vec<Inner>::elem>();
  } else if constexpr (det::is_std_vector<Inner>::value) {
    CheckNumericWidth<typename det::is_std_vector<Inner>::elem>();
  } else {
    CheckNumericWidth<Inner>();
  }
}

// True when a storage type bottoms out in a non-bool arithmetic leaf -- the
// shapes whose widget can misread its width and show garbage.
template <class Inner>
static constexpr bool IsNumericBearing() {
  if constexpr (det::is_std_array<Inner>::value) {
    using X = typename det::is_std_array<Inner>::elem;
    return std::is_arithmetic_v<X> && !std::is_same_v<X, bool>;
  } else if constexpr (det::is_inline_vec<Inner>::value) {
    using X = typename det::is_inline_vec<Inner>::elem;
    return std::is_arithmetic_v<X> && !std::is_same_v<X, bool>;
  } else if constexpr (det::is_std_vector<Inner>::value) {
    using X = typename det::is_std_vector<Inner>::elem;
    return std::is_arithmetic_v<X> && !std::is_same_v<X, bool>;
  } else {
    return std::is_arithmetic_v<Inner> && !std::is_same_v<Inner, bool>;
  }
}

// The per-element visitor: audits every field's displayed seed and widget width.
template <class E>
struct DisplayAudit {
  const E* eff_class;
  const E* eff_full;
  template <class T>
  void field(int id, const char*, T& value) {
    if constexpr (sdkd::is_opt<T>::value) {
      using Inner = typename sdkd::is_opt<T>::inner;
      AuditNumericWidth<Inner>();
      if constexpr (IsNumericBearing<Inner>()) {
        const bool authored = value.has_value();
        const ps::opt<Inner>* clsF =
            eff_class ? sdkd::FieldAt<E, ps::opt<Inner>>(*eff_class, id)
                      : nullptr;
        const ps::opt<Inner>* fullF =
            eff_full ? sdkd::FieldAt<E, ps::opt<Inner>>(*eff_full, id) : nullptr;
        // The exact value the row would seed its edit temp -- and display -- with.
        const Inner seed = det::SeedValue<Inner>(authored, value, clsF, fullF);
        // Independently: the highest-priority layer that carries a value (the
        // full Effective folds element + class + IDL), else the defined zero.
        const bool has_full = fullF && fullF->has_value();
        const Inner expected = authored ? *value : (has_full ? **fullF : Inner{});
        CHECK(seed == expected);
        // Unset with no inherited/default value -> the widget shows the type's
        // defined zero, never uninitialised stack garbage.
        if (!authored && !has_full) {
          CHECK(seed == Inner{});
        }
      }
    } else {
      AuditNumericWidth<T>();
    }
  }
  template <class C>
  void child(int, const char*, C&) {}
  template <class C>
  void union_child(int, const char*, C&) {}
};

template <class E>
static void AuditElement(const mj::Model& m, E& e) {
  std::unique_ptr<E> ec = sdk::Effective(m, e, false);
  std::unique_ptr<E> ef = sdk::Effective(m, e, true);
  DisplayAudit<E> a{ec.get(), ef.get()};
  mj::Visit(e, a);
}

static std::string CorpusRoot() {
  if (const char* env = std::getenv("PROTOSPEC_CORPUS")) return env;
  return "C:/Users/jonat/Documents/Unreal Projects/url_proj/Plugins/"
         "UnrealRoboticsLab/third_party/MuJoCo/src";
}

static void TestDisplayFidelityFresh() {
  mj::Model m;  // empty: no classes, IDL defaults only
  // Cover the group-bearing families and the numeric-array fields.
  mj::Geom g;             AuditElement(m, g);
  mj::Joint j;            AuditElement(m, j);
  mj::Site s;             AuditElement(m, s);
  mj::Camera c;           AuditElement(m, c);
  mj::Light l;            AuditElement(m, l);
  mj::Body b;             AuditElement(m, b);
  mj::Mesh mesh;          AuditElement(m, mesh);
  mj::Material mat;       AuditElement(m, mat);
  mj::Pair p;             AuditElement(m, p);
  mj::ActuatorGeneral a;  AuditElement(m, a);

  // The reported field, checked explicitly: an unset int32 `group` seeds to 0
  // (not garbage) and its widget is a 4-byte drag, not the old 8-byte one.
  int gid = -1;
  const reflect::ElementDescriptor& gd =
      reflect::Describe(mj::ElementType::Geom);
  const reflect::FieldDescriptor* fd = FindField(gd, "group", gid);
  CHECK(fd != nullptr);
  CHECK(fd->kind == reflect::FieldKind::Int32);
  CHECK(!g.group.has_value());  // humanoid geoms rarely author group
  std::unique_ptr<mj::Geom> ec = sdk::Effective(m, g, false);
  std::unique_ptr<mj::Geom> ef = sdk::Effective(m, g, true);
  const int32_t seed = det::SeedValue<int32_t>(
      g.group.has_value(), g.group,
      sdkd::FieldAt<mj::Geom, ps::opt<int32_t>>(*ec, gid),
      sdkd::FieldAt<mj::Geom, ps::opt<int32_t>>(*ef, gid));
  CHECK(seed == 0);
  CHECK(det::NumericWidgetBytes(det::NumericWidgetOf<int32_t>()) == 4);
}

static void TestDisplayFidelityHumanoid() {
  const std::string path =
      (std::filesystem::path(CorpusRoot()) / "model" / "humanoid" /
       "humanoid.xml")
          .string();
  if (!std::filesystem::exists(path)) {
    std::printf("SKIP humanoid display sweep (corpus not found at %s)\n",
                path.c_str());
    return;
  }
  ps::mjcf::io::ParseResult r = ps::mjcf::io::ParseMjcfFile(path);
  CHECK(r.model != nullptr);
  if (!r.model) return;
  const mj::Model& m = *r.model;
  int elems = 0;
  sdkd::WalkModelAll(const_cast<mj::Model&>(m), [&](auto& e) {
    using E = std::decay_t<decltype(e)>;
    if constexpr (!std::is_same_v<E, mj::Model> && requires { e.serial; }) {
      ++elems;
      AuditElement(m, e);
    }
  });
  CHECK(elems > 0);
  std::printf("  display sweep: audited %d humanoid elements\n", elems);
}

int main() {
  TestWidgetCoverage();
  TestWidgetMappingSpecifics();
  TestPresenceInherited();
  TestPresenceIdlDefault();
  TestPresenceRequiredAndUnset();
  TestRefTargetsAndCandidates();
  TestEnumLabels();
  TestInlineVecBounds();
  TestNumericWidgetWidths();
  TestDisplayFidelityFresh();
  TestDisplayFidelityHumanoid();

  std::printf("%d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
