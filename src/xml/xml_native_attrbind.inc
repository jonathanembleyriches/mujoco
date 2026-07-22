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

const AttrBind kGeomBinds[] = {
  {"pos",         kBindDouble, 3, offsetof(mjsGeom, pos), true, nullptr, 0},
  {"type",        kBindEnum,   0, offsetof(mjsGeom, type), true, geom_map, mjNGEOMTYPES},
  {"contype",     kBindInt,    1, offsetof(mjsGeom, contype), true, nullptr, 0},
  {"conaffinity", kBindInt,    1, offsetof(mjsGeom, conaffinity), true, nullptr, 0},
  {"condim",      kBindInt,    1, offsetof(mjsGeom, condim), true, nullptr, 0},
  {"group",       kBindInt,    1, offsetof(mjsGeom, group), true, nullptr, 0},
  {"priority",    kBindInt,    1, offsetof(mjsGeom, priority), true, nullptr, 0},
  {"size",        kBindDouble, 3, offsetof(mjsGeom, size), false, nullptr, 0},
  {"material",    kBindString, 0, offsetof(mjsGeom, material), true, nullptr, 0},
  {"friction",    kBindDouble, 3, offsetof(mjsGeom, friction), false, nullptr, 0},
  {"mass",        kBindDouble, 1, offsetof(mjsGeom, mass), true, nullptr, 0},
  {"density",     kBindDouble, 1, offsetof(mjsGeom, density), true, nullptr, 0},
  {"solmix",      kBindDouble, 1, offsetof(mjsGeom, solmix), true, nullptr, 0},
  {"solref",      kBindDouble, 2, offsetof(mjsGeom, solref), false, nullptr, 0},
  {"solimp",      kBindDouble, 5, offsetof(mjsGeom, solimp), false, nullptr, 0},
  {"margin",      kBindDouble, 1, offsetof(mjsGeom, margin), true, nullptr, 0},
  {"gap",         kBindDouble, 1, offsetof(mjsGeom, gap), true, nullptr, 0},
  {"surfacevel",  kBindDouble, 6, offsetof(mjsGeom, surfacevel), false, nullptr, 0},
  {"adhesion",    kBindDouble, 1, offsetof(mjsGeom, adhesion), true, nullptr, 0},
  {"hfield",      kBindString, 0, offsetof(mjsGeom, hfieldname), true, nullptr, 0},
  {"mesh",        kBindString, 0, offsetof(mjsGeom, meshname), true, nullptr, 0},
  {"fitscale",    kBindDouble, 1, offsetof(mjsGeom, fitscale), true, nullptr, 0},
  {"rgba",        kBindFloat,  4, offsetof(mjsGeom, rgba), true, nullptr, 0},
  {"fluidcoef",   kBindDouble, 5, offsetof(mjsGeom, fluid_coefs), false, nullptr, 0},
};
const int kGeomBindsN = 24;

const AttrBind kJointBinds[] = {
  {"type",               kBindEnum,   0, offsetof(mjsJoint, type), true, joint_map, joint_sz},
  {"group",              kBindInt,    1, offsetof(mjsJoint, group), true, nullptr, 0},
  {"pos",                kBindDouble, 3, offsetof(mjsJoint, pos), true, nullptr, 0},
  {"axis",               kBindDouble, 3, offsetof(mjsJoint, axis), true, nullptr, 0},
  {"springdamper",       kBindDouble, 2, offsetof(mjsJoint, springdamper), true, nullptr, 0},
  {"limited",            kBindEnum,   0, offsetof(mjsJoint, limited), true, TFAuto_map, 3},
  {"actuatorfrclimited", kBindEnum,   0, offsetof(mjsJoint, actfrclimited), true, TFAuto_map, 3},
  {"solreflimit",        kBindDouble, 2, offsetof(mjsJoint, solref_limit), false, nullptr, 0},
  {"solimplimit",        kBindDouble, 5, offsetof(mjsJoint, solimp_limit), false, nullptr, 0},
  {"solreffriction",     kBindDouble, 2, offsetof(mjsJoint, solref_friction), false, nullptr, 0},
  {"solimpfriction",     kBindDouble, 5, offsetof(mjsJoint, solimp_friction), false, nullptr, 0},
  {"stiffness",          kBindDouble, 3, offsetof(mjsJoint, stiffness), false, nullptr, 0},
  {"range",              kBindDouble, 2, offsetof(mjsJoint, range), true, nullptr, 0},
  {"actuatorfrcrange",   kBindDouble, 2, offsetof(mjsJoint, actfrcrange), true, nullptr, 0},
  {"margin",             kBindDouble, 1, offsetof(mjsJoint, margin), true, nullptr, 0},
  {"ref",                kBindDouble, 1, offsetof(mjsJoint, ref), true, nullptr, 0},
  {"springref",          kBindDouble, 1, offsetof(mjsJoint, springref), true, nullptr, 0},
  {"armature",           kBindDouble, 1, offsetof(mjsJoint, armature), true, nullptr, 0},
  {"damping",            kBindDouble, 3, offsetof(mjsJoint, damping), false, nullptr, 0},
  {"frictionloss",       kBindDouble, 1, offsetof(mjsJoint, frictionloss), true, nullptr, 0},
};
const int kJointBindsN = 20;

const AttrBind kActuatorGeneralBinds[] = {
  {"group",        kBindInt,    1, offsetof(mjsActuator, group), true, nullptr, 0},
  {"nsample",      kBindInt,    1, offsetof(mjsActuator, nsample), true, nullptr, 0},
  {"interp",       kBindEnum,   0, offsetof(mjsActuator, interp), true, interp_map, interp_sz},
  {"delay",        kBindDouble, 1, offsetof(mjsActuator, delay), true, nullptr, 0},
  {"ctrllimited",  kBindEnum,   0, offsetof(mjsActuator, ctrllimited), true, TFAuto_map, 3},
  {"forcelimited", kBindEnum,   0, offsetof(mjsActuator, forcelimited), true, TFAuto_map, 3},
  {"actlimited",   kBindEnum,   0, offsetof(mjsActuator, actlimited), true, TFAuto_map, 3},
  {"ctrlrange",    kBindDouble, 2, offsetof(mjsActuator, ctrlrange), true, nullptr, 0},
  {"forcerange",   kBindDouble, 2, offsetof(mjsActuator, forcerange), true, nullptr, 0},
  {"actrange",     kBindDouble, 2, offsetof(mjsActuator, actrange), true, nullptr, 0},
  {"lengthrange",  kBindDouble, 2, offsetof(mjsActuator, lengthrange), true, nullptr, 0},
  {"gear",         kBindDouble, 6, offsetof(mjsActuator, gear), false, nullptr, 0},
  {"damping",      kBindDouble, 3, offsetof(mjsActuator, damping), false, nullptr, 0},
  {"armature",     kBindDouble, 1, offsetof(mjsActuator, armature), false, nullptr, 0},
};
const int kActuatorGeneralBindsN = 14;

const AttrBind kSpatialBinds[] = {
  {"group",              kBindInt,    1, offsetof(mjsTendon, group), true, nullptr, 0},
  {"limited",            kBindEnum,   0, offsetof(mjsTendon, limited), true, TFAuto_map, 3},
  {"actuatorfrclimited", kBindEnum,   0, offsetof(mjsTendon, actfrclimited), true, TFAuto_map, 3},
  {"range",              kBindDouble, 2, offsetof(mjsTendon, range), true, nullptr, 0},
  {"actuatorfrcrange",   kBindDouble, 2, offsetof(mjsTendon, actfrcrange), true, nullptr, 0},
  {"solreflimit",        kBindDouble, 2, offsetof(mjsTendon, solref_limit), false, nullptr, 0},
  {"solimplimit",        kBindDouble, 5, offsetof(mjsTendon, solimp_limit), false, nullptr, 0},
  {"solreffriction",     kBindDouble, 2, offsetof(mjsTendon, solref_friction), false, nullptr, 0},
  {"solimpfriction",     kBindDouble, 5, offsetof(mjsTendon, solimp_friction), false, nullptr, 0},
  {"frictionloss",       kBindDouble, 1, offsetof(mjsTendon, frictionloss), true, nullptr, 0},
  {"width",              kBindDouble, 1, offsetof(mjsTendon, width), true, nullptr, 0},
  {"material",           kBindString, 0, offsetof(mjsTendon, material), true, nullptr, 0},
  {"margin",             kBindDouble, 1, offsetof(mjsTendon, margin), true, nullptr, 0},
  {"stiffness",          kBindDouble, 3, offsetof(mjsTendon, stiffness), false, nullptr, 0},
  {"damping",            kBindDouble, 3, offsetof(mjsTendon, damping), false, nullptr, 0},
  {"armature",           kBindDouble, 1, offsetof(mjsTendon, armature), true, nullptr, 0},
  {"rgba",               kBindFloat,  4, offsetof(mjsTendon, rgba), true, nullptr, 0},
};
const int kSpatialBindsN = 17;

const AttrBind kConnectBinds[] = {
  {"solref", kBindDouble, 2, offsetof(mjsEquality, solref), false, nullptr, 0},
  {"solimp", kBindDouble, 5, offsetof(mjsEquality, solimp), false, nullptr, 0},
};
const int kConnectBindsN = 2;

const AttrBind kMeshBinds[] = {
  {"refpos",   kBindDouble, 3, offsetof(mjsMesh, refpos), true, nullptr, 0},
  {"refquat",  kBindDouble, 4, offsetof(mjsMesh, refquat), true, nullptr, 0},
  {"scale",    kBindDouble, 3, offsetof(mjsMesh, scale), true, nullptr, 0},
  {"inertia",  kBindEnum,   0, offsetof(mjsMesh, inertia), true, meshinertia_map, 4},
  {"material", kBindString, 0, offsetof(mjsMesh, material), true, nullptr, 0},
};
const int kMeshBindsN = 5;

const AttrBind kTouchBinds[] = {
  {"nsample",  kBindInt,    1, offsetof(mjsSensor, nsample), true, nullptr, 0},
  {"interp",   kBindEnum,   0, offsetof(mjsSensor, interp), true, interp_map, interp_sz},
  {"delay",    kBindDouble, 1, offsetof(mjsSensor, delay), true, nullptr, 0},
  {"interval", kBindDouble, 2, offsetof(mjsSensor, interval), false, nullptr, 0},
  {"cutoff",   kBindDouble, 1, offsetof(mjsSensor, cutoff), true, nullptr, 0},
  {"noise",    kBindDouble, 1, offsetof(mjsSensor, noise), true, nullptr, 0},
};
const int kTouchBindsN = 6;

const AttrBind kTextureBinds[] = {
  {"type",         kBindEnum,   0, offsetof(mjsTexture, type), true, texture_map, texture_sz},
  {"colorspace",   kBindEnum,   0, offsetof(mjsTexture, colorspace), true, colorspace_map, colorspace_sz},
  {"content_type", kBindString, 0, offsetof(mjsTexture, content_type), true, nullptr, 0},
  {"gridsize",     kBindIntArr, 2, offsetof(mjsTexture, gridsize), true, nullptr, 0},
  {"rgb1",         kBindDouble, 3, offsetof(mjsTexture, rgb1), true, nullptr, 0},
  {"rgb2",         kBindDouble, 3, offsetof(mjsTexture, rgb2), true, nullptr, 0},
  {"mark",         kBindEnum,   0, offsetof(mjsTexture, mark), true, mark_map, mark_sz},
  {"markrgb",      kBindDouble, 3, offsetof(mjsTexture, markrgb), true, nullptr, 0},
  {"random",       kBindDouble, 1, offsetof(mjsTexture, random), true, nullptr, 0},
  {"width",        kBindInt,    1, offsetof(mjsTexture, width), true, nullptr, 0},
  {"height",       kBindInt,    1, offsetof(mjsTexture, height), true, nullptr, 0},
  {"nchannel",     kBindInt,    1, offsetof(mjsTexture, nchannel), true, nullptr, 0},
};
const int kTextureBindsN = 12;

const AttrBind kHfieldBinds[] = {
  {"content_type", kBindString, 0, offsetof(mjsHField, content_type), true, nullptr, 0},
  {"nrow",         kBindInt,    1, offsetof(mjsHField, nrow), true, nullptr, 0},
  {"ncol",         kBindInt,    1, offsetof(mjsHField, ncol), true, nullptr, 0},
};
const int kHfieldBindsN = 3;
// clang-format on
