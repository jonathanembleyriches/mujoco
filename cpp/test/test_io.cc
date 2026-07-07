// MJCF IO tests: fixpoint round-trips plus a quirk-by-quirk battery over
// embedded snippets and the matching negative cases. The reader/writer are
// exercised end to end; the differential-against-MuJoCo dimension belongs to the
// harness (cpp/harness/, separate owner).

#include <cmath>
#include <cstdio>
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

static constexpr double kPi = 3.141592653589793;

static bool Near(double a, double b) { return std::fabs(a - b) < 1e-12; }

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

// --- Q-ANGLE: degree conversion, joint type dependence, radian on write ---- //
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
  // The body carries the euler orientation converted to radians (hand-checked).
  CHECK(b->orient.has_value());
  const auto& eu = std::get<Euler>(*b->orient);
  CHECK(Near(eu.angles[0], kPi / 2));
  CHECK(Near(eu.angles[1], 0.0) && Near(eu.angles[2], 0.0));

  const Joint& h = *b->joints[0];
  CHECK(Near((*h.range)[0], -kPi / 2) && Near((*h.range)[1], kPi));
  CHECK(Near(*h.ref, kPi / 2));
  CHECK(Near(*h.springref, kPi / 4));
  // A slide joint's range/ref are lengths, NOT angles: no conversion.
  const Joint& s = *b->joints[1];
  CHECK(Near((*s.range)[0], 0.0) && Near((*s.range)[1], 2.0));
  CHECK(Near(*s.ref, 1.0));

  // The reader normalizes to radians and the writer pins compiler angle=radian.
  std::string out = WriteMjcf(*r.model);
  CHECK(out.find("angle=\"radian\"") != std::string::npos);
  CHECK(out.find("angle=\"degree\"") == std::string::npos);
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
  CHECK(Near(aa.axis[2], 1.0) && Near(aa.angle, kPi / 2));  // deg->rad

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

  std::printf("%d checks, %d failures\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
