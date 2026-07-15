// Lifted from MuJoCo src/user/user_mesh.cc + src/plugin/{obj,stl}_decoder: the
// mesh compile pipeline (parsers, vertex processing, normals, convex-hull graph
// via qhull, polygon merge, volume/COM/inertia, face bounding volumes) as free
// functions over plain arrays. Provenance + registry: snapshots/lifted_code.json.
// Symbols live in ps::mjcf::compile::lifted so the native compiler (build.cc)
// can drive them without touching mjC* internals or the qhull/tinyobj headers.
#ifndef PROTOSPEC_COMPILE_LIFTED_MESH_PIPELINE_H
#define PROTOSPEC_COMPILE_LIFTED_MESH_PIPELINE_H

#include <cstdint>
#include <string>
#include <vector>

namespace ps::mjcf::compile::lifted {

// How the raw geometry for a mesh is sourced. UserVertex meshes carry authored
// vert/normal/texcoord/face arrays directly; the file formats are parsed from
// bytes (mirroring MuJoCo's MSH reader and the OBJ/STL decoder plugins).
enum class MeshFormat { UserVertex, Msh, Obj, Stl };

// Everything the pipeline needs to compile one mesh: the raw source plus the
// authored transforms/options. `inertia` uses the mjtMeshInertia numbering
// (convex=0, exact=1, legacy=2, shell=3); `needhull` is the caller's verdict
// (a collision/pair/convex-inertia mesh geom forces the convex-hull graph).
struct MeshInput {
  MeshFormat format = MeshFormat::UserVertex;

  // authored user data (UserVertex); for files the parser fills these.
  std::vector<float> uservert;
  std::vector<float> usernormal;
  std::vector<float> usertexcoord;
  std::vector<int> userface;
  std::vector<int> userfacenormal;
  std::vector<int> userfacetexcoord;

  // file bytes (Msh/Obj/Stl) and the resolved content type (e.g. "model/obj").
  std::vector<char> filebytes;
  std::string content_type;

  double scale[3] = {1, 1, 1};
  double refpos[3] = {0, 0, 0};
  double refquat[4] = {1, 0, 0, 0};
  bool smoothnormal = false;
  int maxhullvert = -1;
  int inertia = 2;  // mjMESH_INERTIA_LEGACY
  bool needhull = false;
};

// The compiled mesh, in the exact array layout mjModel expects. `graph` is the
// convex-hull graph blob (empty when szgraph()==0); `center`/`face_aabb` are the
// per-face circumcenters/AABBs the caller feeds to the shared BVH builder.
struct MeshResult {
  std::vector<float> vert, normal, texcoord;
  std::vector<int> face, facenormal, facetexcoord;
  std::vector<int> graph;
  std::vector<std::vector<int>> polygons;
  std::vector<double> polygon_normals;
  std::vector<std::vector<int>> polygon_map;
  std::vector<double> center;     // 3*nface
  std::vector<double> face_aabb;  // 6*nface, (center,half) per face
  double boxsz[3] = {0, 0, 0};
  double aamm[6] = {0, 0, 0, 0, 0, 0};
  double pos[3] = {0, 0, 0};
  double quat[4] = {1, 0, 0, 0};
  double scale[3] = {1, 1, 1};
  double volume = 0;
  double surface = 0;
  double volume_ref = 0;  // GetVolumeRef(): surface for shell inertia, else volume

  int nvert() const { return static_cast<int>(vert.size() / 3); }
  int nnormal() const { return static_cast<int>(normal.size() / 3); }
  int ntexcoord() const { return static_cast<int>(texcoord.size() / 2); }
  int nface() const { return static_cast<int>(face.size() / 3); }
  bool has_texcoord() const { return !texcoord.empty(); }
  int szgraph() const { return static_cast<int>(graph.size()); }
  int npolygon() const { return static_cast<int>(polygons.size()); }
  int npolygonvert() const {
    int a = 0;
    for (const auto& p : polygons) a += static_cast<int>(p.size());
    return a;
  }
  int npolygonmap() const {
    int a = 0;
    for (const auto& p : polygon_map) a += static_cast<int>(p.size());
    return a;
  }
};

// Compile one mesh: parse (if a file), weld/finite-check vertices, orient, build
// the convex hull graph, normals, polygons, principal-axis inertia frame, face
// circumcenters and AABBs. Returns false with `err` set on any invalid input
// (mirroring the mjCError throws in mjCMesh::Compile).
bool CompileMesh(const MeshInput& in, MeshResult& out, std::string& err);

// The raw parsed geometry of a mesh file: vertices/faces exactly as the loader
// (LoadOBJ/LoadSTL) produces them, before any CoM/principal-frame transform.
// This is what mjCFlexcomp::MakeMesh reads via mjCMesh::Vert()/Face() -- distinct
// from CompileMesh, which bakes the mesh-GEOM inertia pipeline.
struct MeshRawResult {
  std::vector<float> vert;
  std::vector<int> face;
  std::vector<float> texcoord;
  std::vector<int> facetexcoord;
  bool has_texcoord() const { return !texcoord.empty(); }
};

// Load-only mesh entry (mjCMesh::LoadFromResource without Process): parse the
// file bytes and return raw vert/face/texcoord. Only the file formats are
// supported (Obj/Stl); Msh is rejected upstream in the flexcomp mesh path.
// Returns false with `err` set on a parse error.
bool LoadMeshRaw(const MeshInput& in, MeshRawResult& out, std::string& err);

}  // namespace ps::mjcf::compile::lifted

#endif  // PROTOSPEC_COMPILE_LIFTED_MESH_PIPELINE_H
