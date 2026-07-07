// MJCF IO tests: fixpoint round-trips plus a quirk-by-quirk battery over
// embedded snippets and the matching negative cases. The reader/writer are
// exercised end to end; the differential-against-MuJoCo dimension belongs to the
// harness (cpp/harness/, separate owner).

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <variant>

#include "keywords.h"
#include "mjcf.h"
#include "numeric.h"
#include "types.h"

using namespace ps::mjcf;
using namespace ps::mjcf::io;

static int g_failed = 0;
static int g_checks = 0;

#define CHECK(cond)                                                    \
  do {                                                                 \
    ++g_checks;                                                        \
    if (!(cond)) {                                                     \
      ++g_failed;                                                      \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);      \
    }                                                                  \
  } while (0)

static bool Near(double a, double b) { return std::fabs(a - b) < 1e-9; }

static ParseResult Parse(const std::string& xml) {
  return ParseMjcfString(xml, "<test>");
}

// The world Body of a parsed model (worldbody[0]).
static const Body* World(const Model& m) {
  return m.worldbody.empty() ? nullptr : m.worldbody.front().get();
}

// --- fixpoint: Read(Write(Read(x))) == Read(x); Write is deterministic ----- //
static void TestFixpoint() {
  const std::string xml = R"(<mujoco model="rig">
  <compiler angle="degree" eulerseq="xyz" meshdir="meshes"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast">
    <flag contact="disable" energy="enable"/>
  </option>
  <size memory="1M" nkey="2"/>
  <visual>
    <global fovy="50" offwidth="640"/>
    <map force="0.01"/>
    <rgba haze="1 1 1 1"/>
  </visual>
  <statistic meansize="0.05" extent="2"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="torso" pos="0 0 1" euler="90 0 0">
      <inertial pos="0 0 0" mass="5" diaginertia="0.1 0.1 0.1"/>
      <joint name="hinge" type="hinge" axis="0 1 0" range="-90 90" ref="10"/>
      <geom name="cap" type="capsule" fromto="0 0 0 0 0 0.5" size="0.05"
            friction="1 0.5"/>
      <site name="s0" pos="0 0 0.5" quat="1 0 0 0"/>
      <camera name="eye" pos="0 -1 0" fovy="45"/>
      <body name="link" pos="0 0 0.5">
        <joint name="slide" type="slide" axis="0 0 1" range="0 0.3"/>
        <geom name="g2" type="sphere" size="0.03"/>
        <frame name="f0" pos="0 0 0.1">
          <geom name="g3" type="box" size="0.02 0.02 0.02"/>
        </frame>
      </body>
    </body>
  </worldbody>
</mujoco>)";

  ParseResult a = Parse(xml);
  CHECK(a.ok());
  if (!a.ok()) {
    for (const auto& e : a.errors) std::printf("  err: %s\n", e.Render().c_str());
    return;
  }
  std::string out1 = WriteMjcf(*a.model);
  ParseResult b = ParseMjcfString(out1, "<rt>");
  CHECK(b.ok());
  if (!b.ok()) {
    for (const auto& e : b.errors) std::printf("  err: %s\n", e.Render().c_str());
    return;
  }
  // Read(Write(Read(x))) == Read(x) under generated deep equality.
  CHECK(*a.model == *b.model);
  // Write is deterministic: Write(Read(Write(Read(x)))) is byte-identical.
  std::string out2 = WriteMjcf(*b.model);
  CHECK(out1 == out2);
}

// --- Q-ANGLE: values and compiler unit are preserved verbatim -------------- //
// MuJoCo converts joint range only for LIMITED hinge/ball and ref/springref
// only for hinge, resolving those conditions per consuming element at compile
// (user_objects.cc:3207-3282); a shared default class can feed elements that
// resolve them differently, so no read-time conversion is correct. ProtoSpec
// therefore stores every angle exactly as authored and round trips the compiler
// unit, deferring the conversion to MuJoCo.
static void TestAngle() {
  auto r = Parse(R"(<mujoco>
    <compiler angle="degree"/>
    <worldbody>
      <body name="b" euler="90 0 0">
        <joint name="h" type="hinge" range="-90 180" ref="90" springref="45"/>
        <joint name="s" type="slide" range="0 2" ref="1"/>
      </body>
    </worldbody>
  </mujoco>)");
  CHECK(r.ok());
  const Body* b = World(*r.model)->bodies.front().get();
  // The body carries the euler orientation exactly as authored (degrees here).
  CHECK(b->orient.has_value());
  const auto& eu = std::get<Euler>(*b->orient);
  CHECK(Near(eu.angles[0], 90.0));
  CHECK(Near(eu.angles[1], 0.0) && Near(eu.angles[2], 0.0));

  const Joint& h = *b->joints[0];
  CHECK(Near((*h.range)[0], -90.0) && Near((*h.range)[1], 180.0));
  CHECK(Near(*h.ref, 90.0));
  CHECK(Near(*h.springref, 45.0));
  const Joint& s = *b->joints[1];
  CHECK(Near((*s.range)[0], 0.0) && Near((*s.range)[1], 2.0));
  CHECK(Near(*s.ref, 1.0));

  // The authored compiler unit round trips verbatim: no pinning to radian.
  std::string out = WriteMjcf(*r.model);
  CHECK(out.find("angle=\"degree\"") != std::string::npos);
  CHECK(out.find("angle=\"radian\"") == std::string::npos);

  // A radian-authored model likewise keeps its unit and values verbatim.
  auto rad = Parse(R"(<mujoco>
    <compiler angle="radian"/>
    <worldbody><body euler="1.5707963 0 0"/></worldbody>
  </mujoco>)");
  CHECK(rad.ok());
  const auto& eu2 = std::get<Euler>(*World(*rad.model)->bodies.front()->orient);
  CHECK(Near(eu2.angles[0], 1.5707963));
  std::string rad_out = WriteMjcf(*rad.model);
  CHECK(rad_out.find("angle=\"radian\"") != std::string::npos);
}

// --- Q-ORIENT: each encoding + multiple-specifier and zero-quat errors ----- //
static void TestOrient() {
  auto ok = Parse(R"(<mujoco><worldbody>
    <geom name="g" type="sphere" size="1" axisangle="0 0 1 90"/>
  </worldbody></mujoco>)");
  CHECK(ok.ok());
  const auto& g = *World(*ok.model)->geoms.front();
  CHECK(g.orient.has_value());
  const auto& aa = std::get<AxisAngle>(*g.orient);
  CHECK(Near(aa.axis[2], 1.0) && Near(aa.angle, 90.0));  // authored, verbatim

  auto multi = Parse(R"(<mujoco><worldbody>
    <geom type="sphere" size="1" quat="1 0 0 0" euler="0 0 0"/>
  </worldbody></mujoco>)");
  CHECK(!multi.ok());
  CHECK(multi.errors[0].message.find("multiple orientation") != std::string::npos);
  CHECK(multi.errors[0].loc.line > 0);

  auto zero = Parse(R"(<mujoco><worldbody>
    <geom type="sphere" size="1" quat="0 0 0 0"/>
  </worldbody></mujoco>)");
  CHECK(!zero.ok());
  CHECK(zero.errors[0].message.find("zero quaternion") != std::string::npos);
}

// --- Q-FROMTO: routes into GeomShape; size stays a plain field ------------- //
static void TestFromto() {
  auto r = Parse(R"(<mujoco><worldbody>
    <geom name="c" type="capsule" fromto="0 0 0 0 0 1" size="0.1"/>
    <geom name="s" type="sphere" size="0.2"/>
  </worldbody></mujoco>)");
  CHECK(r.ok());
  const auto& c = *World(*r.model)->geoms[0];
  CHECK(c.shape.has_value() && std::holds_alternative<FromTo>(*c.shape));
  CHECK(Near(std::get<FromTo>(*c.shape).fromto[5], 1.0));
  CHECK(c.size.has_value() && Near((*c.size)[0], 0.1));
  // No fromto -> shape unset (implicit explicit form), size present.
  const auto& s = *World(*r.model)->geoms[1];
  CHECK(!s.shape.has_value());
  CHECK(s.size.has_value() && Near((*s.size)[0], 0.2));
}

// --- Q-INERTIA: full vs diagonal; fullinertia + orientation exclusion ------ //
static void TestInertia() {
  auto full = Parse(R"(<mujoco><worldbody><body>
    <inertial pos="0 0 0" mass="1" fullinertia="1 2 3 0.1 0.2 0.3"/>
  </body></worldbody></mujoco>)");
  CHECK(full.ok());
  const auto& body = *World(*full.model)->bodies.front();
  CHECK(!body.inertial.empty());
  const auto& in = *body.inertial.front();
  CHECK(in.inertia.has_value() && std::holds_alternative<FullInertia>(*in.inertia));

  auto both = Parse(R"(<mujoco><worldbody><body>
    <inertial pos="0 0 0" mass="1" fullinertia="1 2 3 0 0 0" euler="0 0 0"/>
  </body></worldbody></mujoco>)");
  CHECK(!both.ok());
  CHECK(both.errors[0].message.find("fullinertia and inertial orientation") !=
        std::string::npos);
}

// --- Q-ARITY: fewer-than-max OK, more-than-max errors ---------------------- //
static void TestArity() {
  auto r = Parse(R"(<mujoco><worldbody>
    <geom name="g" type="sphere" size="1" friction="1.5"/>
  </worldbody></mujoco>)");
  CHECK(r.ok());
  const auto& g = *World(*r.model)->geoms.front();
  CHECK(g.friction.has_value() && g.friction->size() == 1);  // filled count = 1
  CHECK(Near((*g.friction)[0], 1.5));

  auto too_many = Parse(R"(<mujoco><worldbody>
    <geom type="sphere" size="1" friction="1 2 3 4"/>
  </worldbody></mujoco>)");
  CHECK(!too_many.ok());
  CHECK(too_many.errors[0].message.find("too much data") != std::string::npos);
}

// --- Q-NUM: inf, memory suffix, overflow, NaN warning ---------------------- //
static void TestNumeric() {
  auto r = Parse(R"(<mujoco>
    <size memory="2M"/>
    <worldbody><body>
      <joint name="j" type="hinge" armature="inf"/>
    </body></worldbody>
  </mujoco>)");
  CHECK(r.ok());
  CHECK(r.model->sizes.front()->memory.has_value());
  CHECK(*r.model->sizes.front()->memory == std::to_string(2ull << 20));
  const auto& j = *World(*r.model)->bodies.front()->joints.front();
  CHECK(j.armature.has_value() && std::isinf(*j.armature));

  // Integer overflow is an error.
  auto over = Parse(R"(<mujoco><size nkey="99999999999999999999"/></mujoco>)");
  CHECK(!over.ok());
  CHECK(over.errors[0].message.find("too large") != std::string::npos);

  // NaN parses (MuJoCo parity) but raises a non-fatal warning.
  auto nan = Parse(R"(<mujoco><worldbody><body>
    <joint type="hinge" armature="nan"/>
  </body></worldbody></mujoco>)");
  CHECK(nan.ok());
  CHECK(!nan.warnings.empty());

  // memory canonical-byte parsing in isolation.
  std::uint64_t bytes = 0;
  CHECK(num::ParseMemory("1K", bytes) == num::MemStatus::Ok && bytes == 1024);
  CHECK(num::ParseMemory("-1", bytes) == num::MemStatus::Unset);
  CHECK(num::ParseMemory("1Q", bytes) == num::MemStatus::Bad);
}

// --- presence (DR-1): only authored attributes set fields ------------------ //
static void TestPresence() {
  auto r = Parse(R"(<mujoco><worldbody>
    <geom name="g" type="box" size="1 1 1"/>
  </worldbody></mujoco>)");
  CHECK(r.ok());
  const auto& g = *World(*r.model)->geoms.front();
  CHECK(g.type.has_value() && *g.type == GeomType::box);
  CHECK(g.size.has_value());
  CHECK(!g.mass.has_value());       // unauthored -> unset (no default written)
  CHECK(!g.density.has_value());
  CHECK(!g.rgba.has_value());
  CHECK(!g.contype.has_value());
  // The writer emits only authored attributes.
  std::string out = WriteMjcf(*r.model);
  CHECK(out.find("mass=") == std::string::npos);
  CHECK(out.find("density=") == std::string::npos);
  CHECK(out.find("type=\"box\"") != std::string::npos);
}

// --- unknown attribute / unknown element: malformed (not unsupported) ------ //
static void TestMalformed() {
  auto attr = Parse(R"(<mujoco><worldbody>
    <geom type="sphere" size="1" bogus="3"/>
  </worldbody></mujoco>)");
  CHECK(!attr.ok());
  CHECK(!attr.unsupported_only());
  CHECK(attr.errors[0].kind == Diagnostic::Kind::MalformedInput);
  CHECK(attr.errors[0].message.find("unknown attribute 'bogus'") !=
        std::string::npos);
  CHECK(attr.errors[0].loc.line > 0);

  auto elem = Parse(R"(<mujoco><worldbody>
    <notathing/>
  </worldbody></mujoco>)");
  CHECK(!elem.ok());
  CHECK(!elem.unsupported_only());
  CHECK(elem.errors[0].message.find("unknown element 'notathing'") !=
        std::string::npos);
}

// --- unsupported element: distinct, machine-detectable skip signal --------- //
static void TestUnsupported() {
  auto r = Parse(R"(<mujoco>
    <worldbody><geom type="sphere" size="1"/></worldbody>
    <actuator/>
  </mujoco>)");
  CHECK(!r.ok());
  CHECK(r.unsupported_only());
  bool found = false;
  for (const auto& e : r.errors) {
    if (e.kind == Diagnostic::Kind::UnsupportedElement && e.element == "actuator")
      found = true;
  }
  CHECK(found);

  // An unsupported child of a supported element also signals skip (geom plugin).
  auto plug = Parse(R"(<mujoco><worldbody>
    <geom type="sphere" size="1"><plugin plugin="x"/></geom>
  </worldbody></mujoco>)");
  CHECK(!plug.ok());
  CHECK(plug.unsupported_only());
}

// --- root must be <mujoco> ------------------------------------------------- //
static void TestRoot() {
  auto r = Parse("<notmujoco/>");
  CHECK(!r.ok());
  CHECK(r.errors[0].message.find("root element must be <mujoco>") !=
        std::string::npos);

  auto bad = Parse("<mujoco><worldbody></mujoco>");  // malformed XML
  CHECK(!bad.ok());
}

// --- Defaults (family c): classes are DATA, read verbatim, resolve nothing - //
static void TestDefaults() {
  const std::string xml = R"(<mujoco>
  <default>
    <geom rgba="1 0 0 1" friction="1 0.1 0.1"/>
    <joint damping="0.5"/>
    <position kp="100" ctrlrange="-1 1"/>
    <pair friction="1 1 0.01 0.01 0.01"/>
    <equality solref="0.02 1"/>
    <mesh scale="2 2 2"/>
    <material specular="0.5"/>
    <default class="sub">
      <geom rgba="0 1 0 1"/>
      <default class="leaf">
        <joint damping="2"/>
      </default>
    </default>
  </default>
  <worldbody>
    <geom type="sphere" size="1" class="sub"/>
  </worldbody>
</mujoco>)";
  ParseResult r = Parse(xml);
  CHECK(r.ok());
  if (!r.ok()) {
    for (const auto& e : r.errors) std::printf("  err: %s\n", e.Render().c_str());
    return;
  }
  CHECK(r.model->defaults.size() == 1);
  const Default& root = *r.model->defaults.front();
  // The root class is unnamed (implicitly "main"); sub-elements are stored as
  // partial specs with only their authored fields present.
  CHECK(!root.dclass.has_value());
  CHECK(root.geom.size() == 1 && root.geom.front()->rgba.has_value());
  CHECK(root.joint.size() == 1 &&
        Near((*root.joint.front()->damping)[0], 0.5));
  CHECK(root.position.size() == 1 && Near(*root.position.front()->kp, 100));
  CHECK(root.pair.size() == 1);
  CHECK(root.equality.size() == 1);
  CHECK(root.mesh.size() == 1);
  CHECK(root.material.size() == 1);
  // Nested class tree preserved verbatim; nothing resolved into the geom.
  CHECK(root.subclasses.size() == 1);
  const Default& sub = *root.subclasses.front();
  CHECK(sub.dclass.has_value() && *sub.dclass == "sub");
  CHECK(sub.subclasses.size() == 1 && *sub.subclasses.front()->dclass == "leaf");
  // The worldbody geom keeps its class ref by name; no default is applied.
  const Geom& g = *World(*r.model)->geoms.front();
  CHECK(g.dclass.has_value() && g.dclass->name == "sub");
  CHECK(!g.rgba.has_value());  // class value is NOT merged into the element

  // Fixpoint: the whole default tree round trips under deep equality.
  std::string out1 = WriteMjcf(*r.model);
  ParseResult b = ParseMjcfString(out1, "<rt>");
  CHECK(b.ok());
  CHECK(*r.model == *b.model);
  CHECK(out1 == WriteMjcf(*b.model));
}

// --- Q-AUTO/defaults: root class name rules (xml_native_reader.cc:3041-3055) //
static void TestDefaultClassNames() {
  // Top level may be unnamed or exactly "main".
  CHECK(Parse(R"(<mujoco><default><geom size="1"/></default></mujoco>)").ok());
  CHECK(Parse(R"(<mujoco><default class="main"/></mujoco>)").ok());

  auto renamed = Parse(R"(<mujoco><default class="root"/></mujoco>)");
  CHECK(!renamed.ok());
  CHECK(renamed.errors[0].message.find(
            "top-level default class 'main' cannot be renamed") !=
        std::string::npos);
  CHECK(renamed.errors[0].loc.line > 0);

  // A nested default must name a non-empty class.
  auto empty = Parse(R"(<mujoco><default>
    <default><geom size="1"/></default>
  </default></mujoco>)");
  CHECK(!empty.ok());
  CHECK(empty.errors[0].message.find("empty class name") != std::string::npos);
}

// --- Unknown class reference: NOT a read-time error (deferred to tier 2) ---- //
// MuJoCo resolves classes during parse and errors immediately on a dangling
// name (xml_native_reader.cc:4705-4719). ProtoSpec instead stores every ref by
// name and resolves none at read (DR-8, plan Section 9 tier 2), exactly as it
// treats mesh/material/target refs -- referential validation, not IO, reports
// dangling names with provenance later. This is harness-neutral: a genuinely
// dangling class makes MuJoCo reject the model at compile on both sides.
static void TestUnknownClassRef() {
  auto r = Parse(R"(<mujoco><worldbody>
    <geom type="sphere" size="1" class="does_not_exist"/>
  </worldbody></mujoco>)");
  CHECK(r.ok());
  const Geom& g = *World(*r.model)->geoms.front();
  CHECK(g.dclass.has_value() && g.dclass->name == "does_not_exist");
}

// --- Assets (family d): mesh/hfield/material/skin read as data -------------- //
static void TestAssets() {
  const std::string xml = R"(<mujoco>
  <asset>
    <mesh name="m" file="parts/arm.obj" scale="0.1 0.1 0.1" refpos="0 0 1"/>
    <mesh name="verts" vertex="0 0 0 1 0 0 0 1 0" face="0 1 2"/>
    <hfield name="h" file="terrain.png" nrow="4" ncol="4" size="1 1 1 0.1"/>
    <material name="mat" specular="0.3" rgba="1 1 1 1">
      <layer texture="t_rough" role="rgb"/>
    </material>
    <skin name="s" file="skin.skn" inflate="0.01"/>
    <model name="nested" file="submodel.xml"/>
  </asset>
  <worldbody>
    <geom type="mesh" mesh="m"/>
  </worldbody>
</mujoco>)";
  ParseResult r = Parse(xml);
  CHECK(r.ok());
  if (!r.ok()) {
    for (const auto& e : r.errors) std::printf("  err: %s\n", e.Render().c_str());
    return;
  }
  CHECK(r.model->assets.size() == 1);
  const Asset& a = *r.model->assets.front();
  CHECK(a.meshs.size() == 2);
  const Mesh& m0 = *a.meshs.front();
  // Asset file paths are DATA: stored verbatim, contents never loaded (DR-7).
  CHECK(m0.file.has_value() && *m0.file == "parts/arm.obj");
  CHECK(m0.scale.has_value() && Near((*m0.scale)[0], 0.1));
  CHECK(a.meshs[1]->vertex.has_value() && a.meshs[1]->vertex->size() == 9);
  CHECK(a.hfields.size() == 1 && *a.hfields.front()->nrow == 4);
  CHECK(a.materials.size() == 1);
  const Material& mat = *a.materials.front();
  CHECK(mat.layers.size() == 1 && mat.layers.front()->role.has_value());
  CHECK(a.skins.size() == 1 && *a.skins.front()->file == "skin.skn");
  CHECK(a.modelAssets.size() == 1 && *a.modelAssets.front()->file == "submodel.xml");

  std::string out1 = WriteMjcf(*r.model);
  ParseResult b = ParseMjcfString(out1, "<rt>");
  CHECK(b.ok());
  CHECK(*r.model == *b.model);
  CHECK(out1 == WriteMjcf(*b.model));
}

// --- Q-TEX: the texture source variant (builtin vs file) ------------------- //
static void TestTextureSource() {
  auto tex = Parse(R"(<mujoco><asset>
    <texture name="grid" type="2d" builtin="checker" width="8" height="8"/>
    <texture name="img" type="2d" file="wood.png"/>
  </asset></mujoco>)");
  CHECK(tex.ok());
  const Asset& a = *tex.model->assets.front();
  CHECK(a.textures.size() == 2);
  const Texture& t0 = *a.textures[0];
  CHECK(t0.source.has_value() &&
        std::holds_alternative<TextureBuiltin>(*t0.source));
  CHECK(std::get<TextureBuiltin>(*t0.source) == TextureBuiltin::checker);
  const Texture& t1 = *a.textures[1];
  CHECK(t1.source.has_value() && std::holds_alternative<TexFile>(*t1.source));
  CHECK(std::get<TexFile>(*t1.source).file == "wood.png");

  std::string out = WriteMjcf(*tex.model);
  CHECK(out.find("builtin=\"checker\"") != std::string::npos);
  CHECK(out.find("file=\"wood.png\"") != std::string::npos);
  // A file wins over a redundant builtin="none" (single-variant modeling).
  auto both = Parse(R"(<mujoco><asset>
    <texture name="x" type="2d" builtin="none" file="a.png"/>
  </asset></mujoco>)");
  CHECK(both.ok());
  CHECK(std::holds_alternative<TexFile>(*both.model->assets.front()
                                             ->textures.front()->source));
}

// --- Include (Q-INC): splice-in-place, provenance, once-per-file globally --- //
namespace {
int g_tmp_counter = 0;
std::filesystem::path TempDir() {
  auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  std::filesystem::path d =
      std::filesystem::temp_directory_path() /
      ("ps_io_inc_" + std::to_string(stamp) + "_" +
       std::to_string(++g_tmp_counter));
  std::filesystem::remove_all(d);
  std::filesystem::create_directories(d);
  return d;
}
void WriteText(const std::filesystem::path& p, const std::string& s) {
  std::filesystem::create_directories(p.parent_path());
  std::ofstream(p) << s;
}
}  // namespace

static void TestInclude() {
  std::filesystem::path dir = TempDir();
  WriteText(dir / "main.xml", R"(<mujoco model="m">
  <include file="defs.xml"/>
  <worldbody>
    <include file="sub/body.xml"/>
    <geom name="floor" type="plane" size="1 1 1"/>
  </worldbody>
</mujoco>)");
  WriteText(dir / "defs.xml", R"(<mujoco>
  <default><default class="vis"><geom rgba="1 0 0 1"/></default></default>
</mujoco>)");
  WriteText(dir / "sub" / "body.xml", R"(<mujocoinclude>
  <body name="b" pos="0 0 1">
    <geom name="g" type="sphere" size="0.1" class="vis"/>
  </body>
</mujocoinclude>)");

  ParseResult r = ParseMjcfFile((dir / "main.xml").string());
  CHECK(r.ok());
  if (!r.ok()) {
    for (const auto& e : r.errors) std::printf("  err: %s\n", e.Render().c_str());
  } else {
    // The <default> from defs.xml and the <body> from sub/body.xml are spliced
    // in place; the include elements vanish (flattened tree).
    CHECK(r.model->defaults.size() == 1);
    CHECK(r.model->defaults.front()->subclasses.size() == 1);
    const Body* w = World(*r.model);
    CHECK(w->geoms.size() == 1 && *w->geoms.front()->name == "floor");
    CHECK(w->bodies.size() == 1 && *w->bodies.front()->name == "b");
    // Provenance (DR-9): the spliced body's SourceLoc names the INCLUDED file.
    const Body& b = *w->bodies.front();
    CHECK(b.loc.file.find("body.xml") != std::string::npos);
    CHECK(b.loc.line == 2);
  }
  std::filesystem::remove_all(dir);
}

static void TestIncludeErrors() {
  // Self-include is caught by the once-per-file rule (seeded with the top file).
  {
    std::filesystem::path dir = TempDir();
    WriteText(dir / "c.xml", R"(<mujoco><include file="c.xml"/></mujoco>)");
    ParseResult r = ParseMjcfFile((dir / "c.xml").string());
    CHECK(!r.ok());
    CHECK(r.errors[0].message.find("already included") != std::string::npos);
    std::filesystem::remove_all(dir);
  }
  // A duplicate include of the same file anywhere is rejected globally.
  {
    std::filesystem::path dir = TempDir();
    WriteText(dir / "m.xml", R"(<mujoco>
      <include file="a.xml"/><include file="a.xml"/>
    </mujoco>)");
    WriteText(dir / "a.xml", R"(<mujoco><compiler/></mujoco>)");
    ParseResult r = ParseMjcfFile((dir / "m.xml").string());
    CHECK(!r.ok());
    CHECK(r.errors[0].message.find("already included") != std::string::npos);
    std::filesystem::remove_all(dir);
  }
  // Missing file attribute.
  {
    std::filesystem::path dir = TempDir();
    WriteText(dir / "m.xml", R"(<mujoco><include/></mujoco>)");
    ParseResult r = ParseMjcfFile((dir / "m.xml").string());
    CHECK(!r.ok());
    CHECK(r.errors[0].message.find("missing file attribute") !=
          std::string::npos);
    std::filesystem::remove_all(dir);
  }
  // Include element with children.
  {
    std::filesystem::path dir = TempDir();
    WriteText(dir / "m.xml",
              R"(<mujoco><include file="a.xml"><geom/></include></mujoco>)");
    WriteText(dir / "a.xml", R"(<mujoco><compiler/></mujoco>)");
    ParseResult r = ParseMjcfFile((dir / "m.xml").string());
    CHECK(!r.ok());
    CHECK(r.errors[0].message.find("cannot have children") != std::string::npos);
    std::filesystem::remove_all(dir);
  }
  // Unresolvable file.
  {
    std::filesystem::path dir = TempDir();
    WriteText(dir / "m.xml", R"(<mujoco><include file="nope.xml"/></mujoco>)");
    ParseResult r = ParseMjcfFile((dir / "m.xml").string());
    CHECK(!r.ok());
    CHECK(r.errors[0].message.find("cannot open") != std::string::npos);
    std::filesystem::remove_all(dir);
  }
}

int main() {
  TestFixpoint();
  TestAngle();
  TestOrient();
  TestFromto();
  TestInertia();
  TestArity();
  TestNumeric();
  TestPresence();
  TestMalformed();
  TestUnsupported();
  TestRoot();
  TestDefaults();
  TestDefaultClassNames();
  TestUnknownClassRef();
  TestAssets();
  TestTextureSource();
  TestInclude();
  TestIncludeErrors();

  std::printf("%d checks, %d failures\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
