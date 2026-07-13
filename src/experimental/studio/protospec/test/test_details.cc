// ProtoSpec Studio: windowless tests for the generated Details panel (ps::studio).
//
// No SDL/ImGui: everything the panel decides -- which widget a field maps to,
// how presence layers resolve, how a Ref combo is populated, how enum tables and
// InlineVec bounds behave -- lives in details_panel.h as pure functions and is
// exercised here directly. The headline is the reflection-coverage test: every
// field of every one of the schema's element descriptors must classify to a real
// widget, so a schema change that introduces an unhandled shape fails loudly.

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "editor/details_panel.h"
#include "keywords.h"
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
  CHECK(ClassifyField(*field(geom, "orient")) == WidgetKind::Variant);
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

int main() {
  TestWidgetCoverage();
  TestWidgetMappingSpecifics();
  TestPresenceInherited();
  TestPresenceIdlDefault();
  TestPresenceRequiredAndUnset();
  TestRefTargetsAndCandidates();
  TestEnumLabels();
  TestInlineVecBounds();

  std::printf("%d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
