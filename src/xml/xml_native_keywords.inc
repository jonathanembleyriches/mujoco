// Generated from protospec/schema/mujoco.spec by protospec_gen.emit_native -- do not edit.
//
// The reader's schema-backed `const mjMap NAME[]` keyword tables. Each map's
// keyword spellings and order are an IDL enum's member values; the value
// column (C constants / integers) and array-size tokens are the MuJoCo-side
// mapping the emitter owns (KEYWORD_MAPS in emit_native.py). Maps that are not
// schema-backed 1:1 (bool_map, enable_map, TFAuto_map, equality_map,
// jkind_map, shape_map, meshtype_map) stay hand-written in xml_native_reader.cc.
// Regenerate with: uv run python -m protospec_gen.emit_native --write

// clang-format off

// Coordinate -> coordinate_map
const mjMap coordinate_map[2] = {
  {"local",  0},
  {"global", 1},
};

// AngleUnit -> angle_map
const mjMap angle_map[2] = {
  {"radian", 0},
  {"degree", 1},
};

// FluidShape -> fluid_map
const mjMap fluid_map[2] = {
  {"none",      0},
  {"ellipsoid", 1},
};

// SimpleMode -> FAuto_map
const mjMap FAuto_map[2] = {
  {"false", 0},
  {"auto",  1},
};

// BodySleep -> bodysleep_map
const int bodysleep_sz = 4;
const mjMap bodysleep_map[bodysleep_sz] = {
  {"auto",    mjSLEEP_AUTO},
  {"never",   mjSLEEP_NEVER},
  {"allowed", mjSLEEP_ALLOWED},
  {"init",    mjSLEEP_INIT},
};

// JointType -> joint_map
const int joint_sz = 4;
const mjMap joint_map[joint_sz] = {
  {"free",  mjJNT_FREE},
  {"ball",  mjJNT_BALL},
  {"slide", mjJNT_SLIDE},
  {"hinge", mjJNT_HINGE},
};

// GeomType -> geom_map
const mjMap geom_map[mjNGEOMTYPES] = {
  {"plane",     mjGEOM_PLANE},
  {"hfield",    mjGEOM_HFIELD},
  {"sphere",    mjGEOM_SPHERE},
  {"capsule",   mjGEOM_CAPSULE},
  {"ellipsoid", mjGEOM_ELLIPSOID},
  {"cylinder",  mjGEOM_CYLINDER},
  {"box",       mjGEOM_BOX},
  {"mesh",      mjGEOM_MESH},
  {"sdf",       mjGEOM_SDF},
};

// CameraProjection -> projection_map
const int projection_sz = 2;
const mjMap projection_map[projection_sz] = {
  {"perspective",  mjPROJ_PERSPECTIVE},
  {"orthographic", mjPROJ_ORTHOGRAPHIC},
};

// CamLightMode -> camlight_map
const int camlight_sz = 5;
const mjMap camlight_map[camlight_sz] = {
  {"fixed",         mjCAMLIGHT_FIXED},
  {"track",         mjCAMLIGHT_TRACK},
  {"trackcom",      mjCAMLIGHT_TRACKCOM},
  {"targetbody",    mjCAMLIGHT_TARGETBODY},
  {"targetbodycom", mjCAMLIGHT_TARGETBODYCOM},
};

// LightType -> lighttype_map
const int lighttype_sz = 4;
const mjMap lighttype_map[lighttype_sz] = {
  {"spot",        mjLIGHT_SPOT},
  {"directional", mjLIGHT_DIRECTIONAL},
  {"point",       mjLIGHT_POINT},
  {"image",       mjLIGHT_IMAGE},
};

// TexRole -> texrole_map
const int texrole_sz = mjNTEXROLE - 1;
const mjMap texrole_map[texrole_sz] = {
  {"rgb",       mjTEXROLE_RGB},
  {"occlusion", mjTEXROLE_OCCLUSION},
  {"roughness", mjTEXROLE_ROUGHNESS},
  {"metallic",  mjTEXROLE_METALLIC},
  {"normal",    mjTEXROLE_NORMAL},
  {"opacity",   mjTEXROLE_OPACITY},
  {"emissive",  mjTEXROLE_EMISSIVE},
  {"rgba",      mjTEXROLE_RGBA},
  {"orm",       mjTEXROLE_ORM},
};

// Integrator -> integrator_map
const int integrator_sz = 4;
const mjMap integrator_map[integrator_sz] = {
  {"Euler",        mjINT_EULER},
  {"RK4",          mjINT_RK4},
  {"implicit",     mjINT_IMPLICIT},
  {"implicitfast", mjINT_IMPLICITFAST},
};

// Cone -> cone_map
const int cone_sz = 2;
const mjMap cone_map[cone_sz] = {
  {"pyramidal", mjCONE_PYRAMIDAL},
  {"elliptic",  mjCONE_ELLIPTIC},
};

// JacobianType -> jac_map
const int jac_sz = 3;
const mjMap jac_map[jac_sz] = {
  {"dense",  mjJAC_DENSE},
  {"sparse", mjJAC_SPARSE},
  {"auto",   mjJAC_AUTO},
};

// SolverType -> solver_map
const int solver_sz = 3;
const mjMap solver_map[solver_sz] = {
  {"PGS",    mjSOL_PGS},
  {"CG",     mjSOL_CG},
  {"Newton", mjSOL_NEWTON},
};

// TextureType -> texture_map
const int texture_sz = 3;
const mjMap texture_map[texture_sz] = {
  {"2d",     mjTEXTURE_2D},
  {"cube",   mjTEXTURE_CUBE},
  {"skybox", mjTEXTURE_SKYBOX},
};

// ColorSpace -> colorspace_map
const int colorspace_sz = 3;
const mjMap colorspace_map[colorspace_sz] = {
  {"auto",   mjCOLORSPACE_AUTO},
  {"linear", mjCOLORSPACE_LINEAR},
  {"sRGB",   mjCOLORSPACE_SRGB},
};

// TextureBuiltin -> builtin_map
const int builtin_sz = 4;
const mjMap builtin_map[builtin_sz] = {
  {"none",     mjBUILTIN_NONE},
  {"gradient", mjBUILTIN_GRADIENT},
  {"checker",  mjBUILTIN_CHECKER},
  {"flat",     mjBUILTIN_FLAT},
};

// TextureMark -> mark_map
const int mark_sz = 4;
const mjMap mark_map[mark_sz] = {
  {"none",   mjMARK_NONE},
  {"edge",   mjMARK_EDGE},
  {"cross",  mjMARK_CROSS},
  {"random", mjMARK_RANDOM},
};

// DynType -> dyn_map
const int dyn_sz = 7;
const mjMap dyn_map[dyn_sz] = {
  {"none",        mjDYN_NONE},
  {"integrator",  mjDYN_INTEGRATOR},
  {"filter",      mjDYN_FILTER},
  {"filterexact", mjDYN_FILTEREXACT},
  {"muscle",      mjDYN_MUSCLE},
  {"dcmotor",     mjDYN_DCMOTOR},
  {"user",        mjDYN_USER},
};

// DcMotorInput -> dcmotorinput_map
const int dcmotorinput_sz = 3;
const mjMap dcmotorinput_map[dcmotorinput_sz] = {
  {"voltage",  0},
  {"position", 1},
  {"velocity", 2},
};

// GainType -> gain_map
const int gain_sz = 6;
const mjMap gain_map[gain_sz] = {
  {"fixed",   mjGAIN_FIXED},
  {"affine",  mjGAIN_AFFINE},
  {"muscle",  mjGAIN_MUSCLE},
  {"dcmotor", mjGAIN_DCMOTOR},
  {"so3",     mjGAIN_SO3},
  {"user",    mjGAIN_USER},
};

// SO3Input -> input_map
const int input_sz = 2;
const mjMap input_map[input_sz] = {
  {"expmap", mjCHART_EXPMAP},
  {"quat",   mjCHART_QUAT},
};

// BiasType -> bias_map
const int bias_sz = 6;
const mjMap bias_map[bias_sz] = {
  {"none",    mjBIAS_NONE},
  {"affine",  mjBIAS_AFFINE},
  {"muscle",  mjBIAS_MUSCLE},
  {"dcmotor", mjBIAS_DCMOTOR},
  {"so3",     mjBIAS_SO3},
  {"user",    mjBIAS_USER},
};

// InterpType -> interp_map
const int interp_sz = 3;
const mjMap interp_map[interp_sz] = {
  {"zoh",    0},
  {"linear", 1},
  {"cubic",  2},
};

// NeedStage -> stage_map
const int stage_sz = 4;
const mjMap stage_map[stage_sz] = {
  {"none", mjSTAGE_NONE},
  {"pos",  mjSTAGE_POS},
  {"vel",  mjSTAGE_VEL},
  {"acc",  mjSTAGE_ACC},
};

// DataType -> datatype_map
const int datatype_sz = 4;
const mjMap datatype_map[datatype_sz] = {
  {"real",       mjDATATYPE_REAL},
  {"positive",   mjDATATYPE_POSITIVE},
  {"axis",       mjDATATYPE_AXIS},
  {"quaternion", mjDATATYPE_QUATERNION},
};

// ContactData -> condata_map
const mjMap condata_map[mjNCONDATA] = {
  {"found",   mjCONDATA_FOUND},
  {"force",   mjCONDATA_FORCE},
  {"torque",  mjCONDATA_TORQUE},
  {"dist",    mjCONDATA_DIST},
  {"pos",     mjCONDATA_POS},
  {"normal",  mjCONDATA_NORMAL},
  {"tangent", mjCONDATA_TANGENT},
};

// RayData -> raydata_map
const mjMap raydata_map[mjNRAYDATA] = {
  {"dist",   mjRAYDATA_DIST},
  {"dir",    mjRAYDATA_DIR},
  {"origin", mjRAYDATA_ORIGIN},
  {"point",  mjRAYDATA_POINT},
  {"normal", mjRAYDATA_NORMAL},
  {"depth",  mjRAYDATA_DEPTH},
};

// CameraOutput -> camout_map
const int camout_sz = mjNCAMOUT;
const mjMap camout_map[mjNCAMOUT] = {
  {"rgb",          mjCAMOUT_RGB},
  {"depth",        mjCAMOUT_DEPTH},
  {"distance",     mjCAMOUT_DIST},
  {"normal",       mjCAMOUT_NORMAL},
  {"segmentation", mjCAMOUT_SEG},
};

// ContactReduce -> reduce_map
const int reduce_sz = 4;
const mjMap reduce_map[reduce_sz] = {
  {"none",     0},
  {"mindist",  1},
  {"maxforce", 2},
  {"netforce", 3},
};

// Conflict -> conflict_map
const int conflict_sz = 3;
const mjMap conflict_map[conflict_sz] = {
  {"warning", mjCONFLICT_WARNING},
  {"merge",   mjCONFLICT_MERGE},
  {"error",   mjCONFLICT_ERROR},
};

// LRMode -> lrmode_map
const int lrmode_sz = 4;
const mjMap lrmode_map[lrmode_sz] = {
  {"none",       mjLRMODE_NONE},
  {"muscle",     mjLRMODE_MUSCLE},
  {"muscleuser", mjLRMODE_MUSCLEUSER},
  {"all",        mjLRMODE_ALL},
};

// CompositeType -> comp_map
const mjMap comp_map[mjNCOMPTYPES] = {
  {"particle", mjCOMPTYPE_PARTICLE},
  {"grid",     mjCOMPTYPE_GRID},
  {"rope",     mjCOMPTYPE_ROPE},
  {"loop",     mjCOMPTYPE_LOOP},
  {"cable",    mjCOMPTYPE_CABLE},
  {"cloth",    mjCOMPTYPE_CLOTH},
};

// MeshInertia -> meshinertia_map
const mjMap meshinertia_map[4] = {
  {"convex", mjMESH_INERTIA_CONVEX},
  {"legacy", mjMESH_INERTIA_LEGACY},
  {"exact",  mjMESH_INERTIA_EXACT},
  {"shell",  mjMESH_INERTIA_SHELL},
};

// MeshBuiltin -> meshbuiltin_map
const int meshbuiltin_sz = 8;
const mjMap meshbuiltin_map[meshbuiltin_sz] = {
  {"none",        mjMESH_BUILTIN_NONE},
  {"sphere",      mjMESH_BUILTIN_SPHERE},
  {"hemisphere",  mjMESH_BUILTIN_HEMISPHERE},
  {"cone",        mjMESH_BUILTIN_CONE},
  {"supertorus",  mjMESH_BUILTIN_SUPERTORUS},
  {"supersphere", mjMESH_BUILTIN_SUPERSPHERE},
  {"wedge",       mjMESH_BUILTIN_WEDGE},
  {"plate",       mjMESH_BUILTIN_PLATE},
};

// FlexcompType -> fcomp_map
const mjMap fcomp_map[mjNFCOMPTYPES] = {
  {"grid",      mjFCOMPTYPE_GRID},
  {"box",       mjFCOMPTYPE_BOX},
  {"cylinder",  mjFCOMPTYPE_CYLINDER},
  {"ellipsoid", mjFCOMPTYPE_ELLIPSOID},
  {"square",    mjFCOMPTYPE_SQUARE},
  {"disc",      mjFCOMPTYPE_DISC},
  {"circle",    mjFCOMPTYPE_CIRCLE},
  {"mesh",      mjFCOMPTYPE_MESH},
  {"gmsh",      mjFCOMPTYPE_GMSH},
  {"direct",    mjFCOMPTYPE_DIRECT},
};

// FlexDof -> fdof_map
const mjMap fdof_map[mjNFCOMPDOFS] = {
  {"full",      mjFCOMPDOF_FULL},
  {"radial",    mjFCOMPDOF_RADIAL},
  {"trilinear", mjFCOMPDOF_TRILINEAR},
  {"quadratic", mjFCOMPDOF_QUADRATIC},
  {"2d",        mjFCOMPDOF_2D},
};

// FlexSelfCollide -> flexself_map
const mjMap flexself_map[5] = {
  {"none",   mjFLEXSELF_NONE},
  {"narrow", mjFLEXSELF_NARROW},
  {"bvh",    mjFLEXSELF_BVH},
  {"sap",    mjFLEXSELF_SAP},
  {"auto",   mjFLEXSELF_AUTO},
};

// Elastic2D -> elastic2d_map
const mjMap elastic2d_map[5] = {
  {"none",    0},
  {"bend",    1},
  {"stretch", 2},
  {"both",    3},
};

// FlexEquality -> flexeq_map
const mjMap flexeq_map[4] = {
  {"false",  0},
  {"true",   1},
  {"vert",   2},
  {"strain", 3},
};
// clang-format on
