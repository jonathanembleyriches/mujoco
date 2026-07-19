// Lifted from MuJoCo src/user/user_mesh.cc (the mjCMesh::Make* builtin mesh
// generators) + src/user/user_api.cc (mjs_makeMesh dispatch/validation): the
// procedural mesh generators (sphere/hemisphere/cone/supersphere/supertorus/
// wedge/plate) as free functions producing plain uservert/usernormal/userface
// arrays. Provenance + registry: snapshots/lifted_code.json. The generated
// arrays feed the ordinary mesh pipeline (mesh_pipeline.h) exactly as an
// authored user-vertex mesh does.
#ifndef PROTOSPEC_COMPILE_LIFTED_BUILTIN_MESH_H
#define PROTOSPEC_COMPILE_LIFTED_BUILTIN_MESH_H

#include <string>
#include <vector>

namespace ps::mjcf::compile::lifted {

// mjtMeshBuiltin (mjmodel.h) -- the numbering the generators dispatch on.
enum class MeshBuiltinKind {
  Sphere,
  Hemisphere,
  Cone,
  Supertorus,
  Supersphere,
  Wedge,
  Plate,
};

struct BuiltinMeshResult {
  std::vector<float> uservert;
  std::vector<float> usernormal;
  std::vector<int> userface;
  bool inertia_shell = false;  // plate sets mjMESH_INERTIA_SHELL
};

// Generate a builtin mesh's raw geometry (mjs_makeMesh + mjCMesh::Make*).
// `params` is the authored <mesh params="..."> vector; returns false with `err`
// set on an invalid parameter count/range (mirroring the mjs_makeMesh checks).
bool MakeBuiltinMesh(MeshBuiltinKind kind, const std::vector<double>& params,
                     BuiltinMeshResult& out, std::string& err);

}  // namespace ps::mjcf::compile::lifted

#endif  // PROTOSPEC_COMPILE_LIFTED_BUILTIN_MESH_H
