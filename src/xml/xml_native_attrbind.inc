// Generated from protospec/schema/mujoco.spec + snapshots/mjspec_fields.json by
// protospec_gen.emit_native -- do not edit.
//
// Per-element AttrBind tables: the mechanical attr->mjs-field reads of each
// One<Elem> parser, driven by mjXReader::ReadAttrBinds. Fields needing more
// than a copy (name, class, orientation, vectors, bool flags, variant shape,
// per-element interplay quirks) stay hand-written in xml_native_reader.cc.
// Regenerate with: uv run python -m protospec_gen.emit_native --write

// clang-format off

const AttrBind kSiteBinds[] = {
  {"pos",      kBindDouble, 3, offsetof(mjsSite, pos), true, nullptr, 0},
  {"type",     kBindEnum,   0, offsetof(mjsSite, type), true, geom_map, mjNGEOMTYPES},
  {"group",    kBindInt,    1, offsetof(mjsSite, group), true, nullptr, 0},
  {"material", kBindString, 0, offsetof(mjsSite, material), true, nullptr, 0},
  {"size",     kBindDouble, 3, offsetof(mjsSite, size), false, nullptr, 0},
  {"rgba",     kBindFloat,  4, offsetof(mjsSite, rgba), true, nullptr, 0},
};
const int kSiteBindsN = 6;

const AttrBind kCameraBinds[] = {
  {"pos",            kBindDouble, 3, offsetof(mjsCamera, pos), true, nullptr, 0},
  {"projection",     kBindEnum,   0, offsetof(mjsCamera, proj), true, projection_map, projection_sz},
  {"ipd",            kBindDouble, 1, offsetof(mjsCamera, ipd), true, nullptr, 0},
  {"resolution",     kBindIntArr, 2, offsetof(mjsCamera, resolution), true, nullptr, 0},
  {"mode",           kBindEnum,   0, offsetof(mjsCamera, mode), true, camlight_map, camlight_sz},
  {"target",         kBindString, 0, offsetof(mjsCamera, targetbody), true, nullptr, 0},
  {"focalpixel",     kBindFloat,  2, offsetof(mjsCamera, focal_pixel), true, nullptr, 0},
  {"principal",      kBindFloat,  2, offsetof(mjsCamera, principal_length), true, nullptr, 0},
  {"principalpixel", kBindFloat,  2, offsetof(mjsCamera, principal_pixel), true, nullptr, 0},
  {"focal",          kBindFloat,  2, offsetof(mjsCamera, focal_length), true, nullptr, 0},
};
const int kCameraBindsN = 10;

const AttrBind kLightBinds[] = {
  {"pos",         kBindDouble, 3, offsetof(mjsLight, pos), true, nullptr, 0},
  {"dir",         kBindDouble, 3, offsetof(mjsLight, dir), true, nullptr, 0},
  {"bulbradius",  kBindFloat,  1, offsetof(mjsLight, bulbradius), true, nullptr, 0},
  {"intensity",   kBindFloat,  1, offsetof(mjsLight, intensity), true, nullptr, 0},
  {"range",       kBindFloat,  1, offsetof(mjsLight, range), true, nullptr, 0},
  {"attenuation", kBindFloat,  3, offsetof(mjsLight, attenuation), true, nullptr, 0},
  {"cutoff",      kBindFloat,  1, offsetof(mjsLight, cutoff), true, nullptr, 0},
  {"exponent",    kBindFloat,  1, offsetof(mjsLight, exponent), true, nullptr, 0},
  {"ambient",     kBindFloat,  3, offsetof(mjsLight, ambient), true, nullptr, 0},
  {"diffuse",     kBindFloat,  3, offsetof(mjsLight, diffuse), true, nullptr, 0},
  {"specular",    kBindFloat,  3, offsetof(mjsLight, specular), true, nullptr, 0},
  {"mode",        kBindEnum,   0, offsetof(mjsLight, mode), true, camlight_map, camlight_sz},
  {"target",      kBindString, 0, offsetof(mjsLight, targetbody), true, nullptr, 0},
  {"texture",     kBindString, 0, offsetof(mjsLight, texture), true, nullptr, 0},
};
const int kLightBindsN = 14;

const AttrBind kMaterialBinds[] = {
  {"texrepeat",   kBindFloat, 2, offsetof(mjsMaterial, texrepeat), true, nullptr, 0},
  {"emission",    kBindFloat, 1, offsetof(mjsMaterial, emission), true, nullptr, 0},
  {"specular",    kBindFloat, 1, offsetof(mjsMaterial, specular), true, nullptr, 0},
  {"shininess",   kBindFloat, 1, offsetof(mjsMaterial, shininess), true, nullptr, 0},
  {"reflectance", kBindFloat, 1, offsetof(mjsMaterial, reflectance), true, nullptr, 0},
  {"metallic",    kBindFloat, 1, offsetof(mjsMaterial, metallic), true, nullptr, 0},
  {"roughness",   kBindFloat, 1, offsetof(mjsMaterial, roughness), true, nullptr, 0},
  {"rgba",        kBindFloat, 4, offsetof(mjsMaterial, rgba), true, nullptr, 0},
};
const int kMaterialBindsN = 8;

const AttrBind kPairBinds[] = {
  {"condim",         kBindInt,    1, offsetof(mjsPair, condim), true, nullptr, 0},
  {"friction",       kBindDouble, 5, offsetof(mjsPair, friction), false, nullptr, 0},
  {"solref",         kBindDouble, 2, offsetof(mjsPair, solref), false, nullptr, 0},
  {"solreffriction", kBindDouble, 2, offsetof(mjsPair, solreffriction), false, nullptr, 0},
  {"solimp",         kBindDouble, 5, offsetof(mjsPair, solimp), false, nullptr, 0},
  {"gap",            kBindDouble, 1, offsetof(mjsPair, gap), true, nullptr, 0},
  {"margin",         kBindDouble, 1, offsetof(mjsPair, margin), true, nullptr, 0},
  {"adhesion",       kBindDouble, 1, offsetof(mjsPair, adhesion), true, nullptr, 0},
};
const int kPairBindsN = 8;
// clang-format on
