// Lifted from MuJoCo src/user/user_mesh.cc + src/plugin/obj_decoder/obj_decoder.cc
// + src/plugin/stl_decoder/stl_decoder.cc (the mesh compile pipeline). The mjC*
// member reads become plain-array inputs and mjCError throws become a local
// MeshError caught at the CompileMesh boundary; the algorithms are verbatim.
// Provenance tracked in snapshots/lifted_code.json. Symbols live in
// ps::mjcf::compile::lifted.
#include "mesh_pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mujoco/mujoco.h>
#include <libqhull_r/qhull_ra.h>
#include <tiny_obj_loader.h>

#include "mjuu_util.h"

namespace ps::mjcf::compile::lifted {
namespace {

using std::abs;
using std::max;
using std::min;
using std::sqrt;

// mjCError stand-in: thrown on invalid mesh data, caught in CompileMesh.
struct MeshError {
  std::string msg;
};

// compute triangle area, surface normal, center (user_mesh.cc triangle()).
double triangle(double* normal, double* center,
                const double* v1, const double* v2, const double* v3) {
  double normal_local[3];  // if normal is nullptr
  double* normal_ptr = (normal) ? normal : normal_local;
  // center
  if (center) {
    center[0] = (v1[0] + v2[0] + v3[0])/3;
    center[1] = (v1[1] + v2[1] + v3[1])/3;
    center[2] = (v1[2] + v2[2] + v3[2])/3;
  }

  // normal = (v2-v1) cross (v3-v1)
  double b[3] = { v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2] };
  double c[3] = { v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2] };
  mjuu_crossvec(normal_ptr, b, c);

  // get length
  double len = sqrt(mjuu_dot3(normal_ptr, normal_ptr));

  // ignore small faces
  if (len < mjMINVAL) {
    return 0;
  }

  // normalize
  if (normal) {
    normal_ptr[0] /= len;
    normal_ptr[1] /= len;
    normal_ptr[2] /= len;
  }

  // return area
  return 0.5 * len;
}

// Read data of type T from a potentially unaligned buffer pointer.
template <typename T>
void ReadFromBuffer(T* dst, const char* src) {
  std::memcpy(dst, src, sizeof(T));
}

// ---- polygon merge helpers (user_mesh.cc MeshPolygon / MakePolygons) ------- //

class MeshPolygon {
 public:
  MeshPolygon(const double v1[3], const double v2[3], const double v3[3],
              int v1i, int v2i, int v3i, double theta, double phi);
  MeshPolygon() = delete;
  MeshPolygon(const MeshPolygon&) = delete;
  MeshPolygon& operator=(const MeshPolygon&) = delete;
  MeshPolygon(MeshPolygon&&) = default;
  MeshPolygon& operator=(MeshPolygon&&) = default;

  void InsertFace(int v1, int v2, int v3);
  std::vector<std::vector<int>> Paths() const;
  const double* Normal() const { return normal_; }
  double Normal(int i) const { return normal_[i]; }

 private:
  std::vector<std::pair<int, int>> edges_;
  std::vector<int> islands_;
  int nisland_ = 0;
  double normal_[3] = {0.0, 0.0, 0.0};
  void CombineIslands(int& island1, int& island2);
};

bool MeshPolygonKey(std::pair<double, double>& angles, const double v1[3],
                    const double v2[3], const double v3[3], double angle_tol) {
  double diff12[3] = {v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]};
  double diff13[3] = {v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]};
  double normal[3], norm;

  mjuu_crossvec(normal, diff12, diff13);
  if ((norm = std::sqrt(mjuu_dot3(normal, normal))) < mjMINVAL) {
    return false;
  }

  // atan2 is sensitive to sign of 0.0, adding 0.0 to enforcing only positive 0.0
  normal[0] = (normal[0] / norm) + 0.0;
  normal[1] = (normal[1] / norm) + 0.0;
  normal[2] = (normal[2] / norm) + 0.0;
  double rtheta = 0.0, rphi = 0.0;

  // clamp normal to be in valid range for acos
  if (std::abs(normal[2]) > 1.0 - 1e-7) {
    if (normal[2] < 0) rphi = std::round(mjPI / angle_tol);
    angles = std::make_pair(rtheta, rphi);
    return true;
  }
  // rounded azimuthal and polar angles
  rtheta = std::round(std::atan2(normal[1], normal[0]) / angle_tol);
  rphi = std::round(std::acos(normal[2]) / angle_tol);
  angles = std::make_pair(rtheta, rphi);
  return true;
}

MeshPolygon::MeshPolygon(const double v1[3], const double v2[3],
                         const double v3[3], int v1i, int v2i, int v3i,
                         double theta, double phi) {
  (void)v1; (void)v2; (void)v3;
  normal_[0] = std::cos(theta) * std::sin(phi);
  normal_[1] = std::sin(theta) * std::sin(phi);
  normal_[2] = std::cos(phi);

  edges_ = {{v1i, v2i}, {v2i, v3i}, {v3i, v1i}};
  nisland_ = 1;
  islands_ = {0, 0, 0};
}

void MeshPolygon::CombineIslands(int& island1, int& island2) {
  if (island2 < island1) {
    int tmp = island1;
    island1 = island2;
    island2 = tmp;
  }
  for (int k = 0; k < static_cast<int>(islands_.size()); ++k) {
    if (islands_[k] == island2) {
      islands_[k] = island1;
    } else if (islands_[k] > island2) {
      islands_[k]--;
    }
  }
}

void MeshPolygon::InsertFace(int v1, int v2, int v3) {
  int add1 = 1, add2 = 1, add3 = 1;
  int island = -1;

  for (int i = 0; i < static_cast<int>(edges_.size()); ++i) {
    if (edges_[i].first == v2 && edges_[i].second == v1) {
      add1 = 0;
      island = islands_[i];
      edges_.erase(edges_.begin() + i);
      islands_.erase(islands_.begin() + i);
      break;
    }
  }

  for (int i = 0; i < static_cast<int>(edges_.size()); ++i) {
    if (edges_[i].first == v3 && edges_[i].second == v2) {
      int island2 = islands_[i];
      if (island == -1) {
        island = island2;
      } else if (island2 != island) {
        nisland_--;
        CombineIslands(island, island2);
      }
      add2 = 0;
      edges_.erase(edges_.begin() + i);
      islands_.erase(islands_.begin() + i);
      break;
    }
  }

  for (int i = 0; i < static_cast<int>(edges_.size()); ++i) {
    if (edges_[i].first == v1 && edges_[i].second == v3) {
      int island3 = islands_[i];
      if (island == -1) {
        island = island3;
      } else if (island3 != island) {
        nisland_--;
        CombineIslands(island, island3);
      }
      add3 = 0;
      edges_.erase(edges_.begin() + i);
      islands_.erase(islands_.begin() + i);
      break;
    }
  }

  if (island == -1) {
    island = nisland_++;
  }

  if (add1) {
    edges_.push_back({v1, v2});
    islands_.push_back(island);
  }
  if (add2) {
    edges_.push_back({v2, v3});
    islands_.push_back(island);
  }
  if (add3) {
    edges_.push_back({v3, v1});
    islands_.push_back(island);
  }
}

std::vector<std::vector<int>> MeshPolygon::Paths() const {
  std::vector<std::vector<int>> paths;
  if (edges_.size() == 3) {
    return {{edges_[0].first, edges_[1].first, edges_[2].first}};
  }

  for (int i = 0; i < nisland_; ++i) {
    std::vector<int> path;

    for (int j = 0; j < static_cast<int>(edges_.size()); ++j) {
      if (islands_[j] == i) {
        path.push_back(edges_[j].first);
        path.push_back(edges_[j].second);
        break;
      }
    }

    if (path.empty()) {
      continue;
    }

    int next = path.back();
    for (int l = 0; l < static_cast<int>(edges_.size()); ++l) {
      int finished = 0;
      for (int k = 1; k < static_cast<int>(edges_.size()); ++k) {
        if (islands_[k] == i && edges_[k].first == next) {
          next = edges_[k].second;
          if (next == path[0]) {
            paths.push_back(path);
            finished = 1;
            break;
          }
          path.push_back(next);
          break;
        }
      }
      if (finished) {
        break;
      }
    }
  }
  return paths;
}

struct PairHash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& pair) const {
    return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
  }
};

// --------------------------------------------------------------------------- //
// Mesh: mirrors the mjCMesh state the pipeline touches (member reads retargeted //
// to plain arrays). All methods are verbatim ports of user_mesh.cc.            //
// --------------------------------------------------------------------------- //
struct Mesh {
  // inputs
  double scale[3] = {1, 1, 1};
  double refpos[3] = {0, 0, 0};
  double refquat[4] = {1, 0, 0, 0};
  bool smoothnormal = false;
  int maxhullvert_ = -1;
  int inertia = mjMESH_INERTIA_LEGACY;
  bool needhull_ = false;
  bool needreorient_ = true;
  std::string content_type_;

  // geometry
  std::vector<float> vert_;
  std::vector<float> normal_;
  std::vector<float> texcoord_;
  std::vector<int> face_;
  std::vector<int> facenormal_;
  std::vector<int> facetexcoord_;
  std::vector<std::pair<int, int>> halfedge_;

  // derived
  std::vector<int> graph_;
  int szgraph_ = 0;
  std::vector<double> center_;
  std::vector<double> face_aabb_;
  std::vector<std::vector<int>> polygons_;
  std::vector<double> polygon_normals_;
  std::vector<std::vector<int>> polygon_map_;
  double boxsz_[3] = {0, 0, 0};
  double aamm_[6] = {1e10, 1e10, 1e10, -1e10, -1e10, -1e10};
  double pos_[3] = {0, 0, 0};
  double quat_[4] = {1, 0, 0, 0};
  double volume_ = 0;
  double surface_ = 0;

  int nvert() const { return static_cast<int>(vert_.size() / 3); }
  int nnormal() const { return static_cast<int>(normal_.size() / 3); }
  int ntexcoord() const { return static_cast<int>(texcoord_.size() / 2); }
  int nface() const { return static_cast<int>(face_.size() / 3); }
  int* GraphFaces() { return graph_.data() + 2 + 3 * (graph_[0] + graph_[1]); }
  double* GetInertiaBoxPtr() { return boxsz_; }
  double GetVolumeRef() const {
    return (inertia == mjMESH_INERTIA_SHELL) ? surface_ : volume_;
  }

  void ProcessVertices(const std::vector<float>& vert);
  void LoadMSH(const std::vector<char>& bytes);
  void LoadOBJ(const std::vector<char>& bytes);
  void LoadSTL(const std::vector<char>& bytes);
  void CheckInitialMesh() const;
  void Process();
  void MakeGraph(const double* dvert);
  void CopyGraph();
  void MakeNormal(const double* dvert);
  void MakeCenter(const double* dvert);
  void MakePolygons(const double* dvert);
  void MakePolygonNormals(const double* dvert);
  void ApplyTransformations(double* dvert);
  double ComputeFaceCentroid(double facecen[3], const double* dvert) const;
  double ComputeVolume(double CoM[3], const double facecen[3], const double* dvert) const;
  double ComputeSurfaceArea(double CoM[3], const double facecen[3], const double* dvert) const;
  double ComputeInertia(double inert[6], const double CoM[3], const double* dvert) const;
  void Rotate(double quat[4], double* dvert);
  void SetBoundingVolume(int faceid, const double* dvert);
};

// process (finite-check + float copy; the XML/file path never dedups).
void Mesh::ProcessVertices(const std::vector<float>& vert) {
  vert_.clear();
  int nvert = static_cast<int>(vert.size());

  if (nvert % 3) {
    throw MeshError{"vertex data must be a multiple of 3"};
  }
  if (face_.size() % 3) {
    throw MeshError{"face data must be a multiple of 3"};
  }

  vert_.reserve(nvert);
  for (int i = 0; i < nvert / 3; ++i) {
    const float* v = &vert[3 * i];
    if (!std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2])) {
      throw MeshError{"vertex coordinate is not finite"};
    }
    vert_.push_back(v[0]);
    vert_.push_back(v[1]);
    vert_.push_back(v[2]);
  }
}

// load MSH binary mesh (user_mesh.cc LoadMSH).
void Mesh::LoadMSH(const std::vector<char>& bytes) {
  bool righthand = scale[0] * scale[1] * scale[2] > 0;
  const char* buffer = bytes.data();
  int buffer_sz = static_cast<int>(bytes.size());

  if (buffer_sz < 0) {
    throw MeshError{"could not read MSH file"};
  } else if (!buffer_sz) {
    throw MeshError{"MSH file is empty"};
  }
  if (static_cast<size_t>(buffer_sz) < 4 * sizeof(int)) {
    throw MeshError{"missing header in MSH file"};
  }

  int nvbuf = 0, nfbuf = 0, nnbuf = 0, ntbuf = 0;
  ReadFromBuffer(&nvbuf, buffer);
  ReadFromBuffer(&nnbuf, buffer + sizeof(int));
  ReadFromBuffer(&ntbuf, buffer + 2 * sizeof(int));
  ReadFromBuffer(&nfbuf, buffer + 3 * sizeof(int));

  if (nvbuf < 4 || nfbuf < 0 || nnbuf < 0 || ntbuf < 0 ||
      (nnbuf > 0 && nnbuf != nvbuf) ||
      (ntbuf > 0 && ntbuf != nvbuf)) {
    throw MeshError{"invalid sizes in MSH file"};
  }

  if (nvbuf >= INT_MAX / static_cast<int>(sizeof(float)) / 3 ||
      nnbuf >= INT_MAX / static_cast<int>(sizeof(float)) / 3 ||
      ntbuf >= INT_MAX / static_cast<int>(sizeof(float)) / 2 ||
      nfbuf >= INT_MAX / static_cast<int>(sizeof(int)) / 3) {
    throw MeshError{"too large sizes in MSH file"};
  }
  if (static_cast<size_t>(buffer_sz) !=
      4 * sizeof(int) + 3 * nvbuf * sizeof(float) + 3 * nnbuf * sizeof(float) +
      2 * ntbuf * sizeof(float) + 3 * nfbuf * sizeof(int)) {
    throw MeshError{"unexpected file size in MSH file"};
  }

  using UnalignedFloat = char[sizeof(float)];
  auto fdata = reinterpret_cast<const UnalignedFloat*>(buffer + 4 * sizeof(int));
  std::vector<float> vert;
  int nvert = 0;
  if (nvbuf) {
    vert.assign(3 * nvbuf, 0);
    nvert = 3 * nvbuf;
    memcpy(vert.data(), fdata, nvert * sizeof(float));
    fdata += nvert;
  }
  if (nnbuf) {
    normal_.assign(nvert, 0);
    memcpy(normal_.data(), fdata, nvert * sizeof(float));
    fdata += nvert;
  }
  if (ntbuf) {
    texcoord_.assign(2 * (nvert / 3), 0);
    memcpy(texcoord_.data(), fdata, 2 * (nvert / 3) * sizeof(float));
    fdata += 2 * (nvert / 3);
  }
  if (nfbuf) {
    face_.assign(3 * nfbuf, 0);
    facenormal_.assign(3 * nfbuf, 0);
    memcpy(face_.data(), fdata, 3 * nfbuf * sizeof(int));
    memcpy(facenormal_.data(), fdata, 3 * nfbuf * sizeof(int));
  }
  if (nfbuf && !texcoord_.empty()) {
    facetexcoord_.assign(3 * nfbuf, 0);
    memcpy(facetexcoord_.data(), fdata, 3 * nfbuf * sizeof(int));
  }

  if (nfbuf && !righthand) {
    for (int i = 0; i < nfbuf; i++) {
      int tmp = face_[3 * i + 1];
      face_[3 * i + 1] = face_[3 * i + 2];
      face_[3 * i + 2] = tmp;
    }
  }
  ProcessVertices(vert);
}

// load OBJ via tinyobjloader (obj_decoder.cc Decode + LoadFromDecoder).
void Mesh::LoadOBJ(const std::vector<char>& bytes) {
  tinyobj::ObjReader obj_reader;
  obj_reader.ParseFromString(std::string(bytes.data(), bytes.size()),
                             std::string());
  if (!obj_reader.Valid()) {
    throw MeshError{"could not parse OBJ file"};
  }

  const auto& attrib = obj_reader.GetAttrib();
  std::vector<float> usernormal = attrib.normals;
  std::vector<float> usertexcoord = attrib.texcoords;
  std::vector<int> userface;
  std::vector<int> userfacenormal;
  std::vector<int> userfacetexcoord;

  if (!obj_reader.GetShapes().empty()) {
    const auto& obj_shape = obj_reader.GetShapes()[0];
    if (!obj_shape.mesh.indices.empty()) {
      const auto& obj_mesh = obj_shape.mesh;

      size_t num_face_indices = 0;
      for (size_t face = 0; face < obj_mesh.num_face_vertices.size(); ++face) {
        int nfacevert = obj_mesh.num_face_vertices[face];
        if (nfacevert == 3) {
          num_face_indices += 3;
        } else if (nfacevert == 4) {
          num_face_indices += 6;
        }
      }

      std::vector<tinyobj::index_t> face_indices;
      face_indices.reserve(num_face_indices);
      userface.reserve(num_face_indices);
      if (!usernormal.empty()) userfacenormal.reserve(num_face_indices);
      if (!usertexcoord.empty()) userfacetexcoord.reserve(num_face_indices);

      for (size_t face = 0, idx = 0; idx < obj_mesh.indices.size();) {
        int nfacevert = obj_mesh.num_face_vertices[face];
        if (nfacevert < 3 || nfacevert > 4) {
          throw MeshError{"only tri or quad meshes are supported in OBJ file"};
        }
        face_indices.push_back(obj_mesh.indices[idx]);
        face_indices.push_back(obj_mesh.indices[idx + 1]);
        face_indices.push_back(obj_mesh.indices[idx + 2]);
        if (nfacevert == 4) {
          face_indices.push_back(obj_mesh.indices[idx]);
          face_indices.push_back(obj_mesh.indices[idx + 2]);
          face_indices.push_back(obj_mesh.indices[idx + 3]);
        }
        idx += nfacevert;
        ++face;
      }

      for (const auto& mesh_index : face_indices) {
        userface.push_back(mesh_index.vertex_index);
        if (!usernormal.empty()) userfacenormal.push_back(mesh_index.normal_index);
        if (!usertexcoord.empty()) userfacetexcoord.push_back(mesh_index.texcoord_index);
      }
    } else if (!obj_shape.lines.indices.empty()) {
      const auto& obj_lines = obj_shape.lines;
      userface.reserve(3 * (obj_lines.indices.size() -
                            obj_lines.num_line_vertices.size()));
      size_t idx = 0;
      for (size_t l = 0; l < obj_lines.num_line_vertices.size(); l++) {
        int nlinevert = obj_lines.num_line_vertices[l];
        for (int v = 0; v < nlinevert - 1; v++) {
          int i1 = obj_lines.indices[idx + v].vertex_index;
          int i2 = obj_lines.indices[idx + v + 1].vertex_index;
          userface.push_back(i1);
          userface.push_back(i2);
          userface.push_back(i2);
        }
        idx += nlinevert;
      }
    }
  }

  for (size_t i = 0; i < usertexcoord.size() / 2; i++) {
    usertexcoord[2 * i + 1] = 1 - usertexcoord[2 * i + 1];
  }

  // LoadFromDecoder: decoded arrays become the mesh arrays; verts get processed.
  normal_.assign(usernormal.begin(), usernormal.end());
  texcoord_.assign(usertexcoord.begin(), usertexcoord.end());
  face_.assign(userface.begin(), userface.end());
  facenormal_.assign(userfacenormal.begin(), userfacenormal.end());
  facetexcoord_.assign(userfacetexcoord.begin(), userfacetexcoord.end());
  std::vector<float> vert(attrib.vertices.begin(), attrib.vertices.end());
  ProcessVertices(vert);
}

// binary STL with bit-exact vertex weld (stl_decoder.cc Decode).
void Mesh::LoadSTL(const std::vector<char>& bytes) {
  struct Vec3Key {
    int x, y, z;
    bool operator<(const Vec3Key& o) const {
      if (x != o.x) return x < o.x;
      if (y != o.y) return y < o.y;
      return z < o.z;
    }
  };
  auto FloatToKey = [](const float v[3]) {
    int x, y, z;
    std::memcpy(&x, &v[0], sizeof(int));
    std::memcpy(&y, &v[1], sizeof(int));
    std::memcpy(&z, &v[2], sizeof(int));
    return Vec3Key{x, y, z};
  };

  int buffer_sz = static_cast<int>(bytes.size());
  if (!buffer_sz) {
    throw MeshError{"STL file is empty"};
  }
  const char* buffer = bytes.data();
  if (buffer_sz < 84) {
    throw MeshError{"invalid header in STL file"};
  }

  int nfaces = 0;
  ReadFromBuffer(&nfaces, buffer + 80);
  if (nfaces < 1 || nfaces > 200000) {
    throw MeshError{"number of faces should be between 1 and 200000 in STL file"};
  }
  if (nfaces * 50 != buffer_sz - 84) {
    throw MeshError{"STL file has wrong size; perhaps this is an ASCII file?"};
  }

  const char* stl = buffer + 84;
  std::vector<float> uservert;
  std::vector<int> userface(3 * nfaces, 0);
  std::map<Vec3Key, int> vertmap;

  for (int i = 0; i < nfaces; i++) {
    for (int j = 0; j < 3; j++) {
      float v[3];
      ReadFromBuffer(&v, stl + 50 * i + 12 * (j + 1));
      if (std::fabs(v[0]) > std::pow(2, 30) ||
          std::fabs(v[1]) > std::pow(2, 30) ||
          std::fabs(v[2]) > std::pow(2, 30)) {
        throw MeshError{"vertex in STL file exceeds maximum bounds"};
      }
      Vec3Key key = FloatToKey(v);
      auto [it, inserted] =
          vertmap.emplace(key, static_cast<int>(uservert.size() / 3));
      if (inserted) {
        uservert.push_back(v[0]);
        uservert.push_back(v[1]);
        uservert.push_back(v[2]);
      }
      userface[3 * i + j] = it->second;
    }
  }

  face_.assign(userface.begin(), userface.end());
  ProcessVertices(uservert);
}

void Mesh::CheckInitialMesh() const {
  if (vert_.size() < 12) {
    throw MeshError{"at least 4 vertices required"};
  }
  if (vert_.size() % 3) {
    throw MeshError{"vertex data must be a multiple of 3"};
  }
  if (normal_.size() % 3) {
    throw MeshError{"normal data must be a multiple of 3"};
  }
  if (texcoord_.size() % 2) {
    throw MeshError{"texcoord must be a multiple of 2"};
  }
  if (face_.size() % 3) {
    throw MeshError{"face data must be a multiple of 3"};
  }
  if (!texcoord_.empty() &&
      texcoord_.size() != 2 * static_cast<size_t>(nvert()) &&
      facetexcoord_.empty() && content_type_ != "model/obj") {
    throw MeshError{
        "texcoord must be 2*nv if face texcoord indices are not provided in an "
        "OBJ file"};
  }
  if (vert_.empty()) {
    throw MeshError{"no vertices"};
  }
  int nv = nvert(), nf = static_cast<int>(face_.size());
  for (int i = 0; i < nf; i++) {
    if (face_[i] >= nv || face_[i] < 0) {
      throw MeshError{"face vertex index does not exist"};
    }
  }
}

void Mesh::MakeGraph(const double* dvert) {
  int adr, ok, curlong, totlong, exitcode;
  facetT* facet, **facetp;
  vertexT* vertex, *vertex1, **vertex1p;

  std::string qhopt = "qhull Qt";
  if (maxhullvert_ > -1) {
    qhopt += " Q9 TA" + std::to_string(maxhullvert_ - 4);
  }

  if (nvert() < 4) {
    return;
  }

  {
    int v1 = -1;
    double len1 = 0;
    for (int i = 1; i < nvert(); i++) {
      len1 = mjuu_dist3(dvert + 3 * i, dvert);
      if (len1 > mjMINVAL) {
        v1 = i;
        break;
      }
    }
    if (v1 < 0) {
      throw MeshError{"mesh has colocated vertices, cannot compute convex hull"};
    }

    double edge1[3] = {dvert[3 * v1 + 0] - dvert[0],
                       dvert[3 * v1 + 1] - dvert[1],
                       dvert[3 * v1 + 2] - dvert[2]};
    double normal[3] = {0, 0, 0};
    bool collinear = true;
    for (int i = 1; i < nvert(); i++) {
      if (i == v1) continue;
      double edge2[3] = {dvert[3 * i + 0] - dvert[0],
                         dvert[3 * i + 1] - dvert[1],
                         dvert[3 * i + 2] - dvert[2]};
      double len2 = sqrt(mjuu_dot3(edge2, edge2));
      if (len2 < mjMINVAL) continue;
      mjuu_crossvec(normal, edge1, edge2);
      double norm = sqrt(mjuu_dot3(normal, normal));
      if (norm > mjMINVAL * len1 * len2) {
        normal[0] /= norm;
        normal[1] /= norm;
        normal[2] /= norm;
        collinear = false;
        break;
      }
    }
    if (collinear) {
      throw MeshError{"mesh has collinear vertices, cannot compute convex hull"};
    }

    double d = mjuu_dot3(normal, dvert);
    bool coplanar = true;
    for (int i = 0; i < nvert(); i++) {
      if (fabs(mjuu_dot3(normal, dvert + 3 * i) - d) > mjMINVAL * len1) {
        coplanar = false;
        break;
      }
    }
    if (coplanar) {
      throw MeshError{"mesh has coplanar vertices, cannot compute convex hull"};
    }
  }

  qhT qh_qh;
  qhT* qh = &qh_qh;
  qh_zero(qh, stderr);
  qh_init_A(qh, stdin, stdout, stderr, 0, nullptr);

  exitcode = setjmp(qh->errexit);
  qh->NOerrexit = false;
  if (!exitcode) {
    qh_initflags(qh, const_cast<char*>(qhopt.c_str()));
    qh_init_B(qh, const_cast<double*>(dvert), nvert(), 3, qh_False);
    qh_qhull(qh);
    qh_triangulate(qh);
    qh_vertexneighbors(qh);

    int numvert = qh->num_vertices;
    int numface = qh->num_facets;
    szgraph_ = 2 + 3 * numvert + 6 * numface;
    graph_.assign(szgraph_, 0);
    graph_[0] = numvert;
    graph_[1] = numface;

    int* vert_edgeadr = graph_.data() + 2;
    int* vert_globalid = graph_.data() + 2 + numvert;
    int* edge_localid = graph_.data() + 2 + 2 * numvert;
    int* face_globalid = graph_.data() + 2 + 3 * numvert + 3 * numface;

    int i = adr = 0;
    ok = 1;
    FORALLvertices {
      int pid = qh_pointid(qh, vertex->point);
      if (pid < 0 || pid >= nvert()) {
        ok = 0;
        break;
      }
      vert_edgeadr[i] = adr;
      vert_globalid[i] = pid;

      int start = adr;
      FOREACHsetelement_(facetT, vertex->neighbors, facet) {
        int cnt = 0;
        FOREACHsetelement_(vertexT, facet->vertices, vertex1) {
          cnt++;
          int pid1 = qh_pointid(qh, vertex1->point);
          if (pid1 < 0 || pid1 >= nvert()) {
            ok = 0;
            break;
          }
          if (pid != pid1) {
            int j;
            for (j = start; j < adr; j++)
              if (pid1 == edge_localid[j]) {
                break;
              }
            if (j >= adr) {
              edge_localid[adr++] = pid1;
            }
          }
        }
        if (cnt != 3) {
          throw MeshError{"Qhull did not return triangle"};
        }
      }
      edge_localid[adr++] = -1;
      i++;
    }

    if (adr != numvert + 3 * numface) {
      throw MeshError{"Wrong size in convex hull graph"};
    }

    adr = 0;
    FORALLfacets {
      int ii = 0;
      int ind[3] = {0, 1, 2};
      if (facet->toporient) {
        ind[0] = 1;
        ind[1] = 0;
      }
      FOREACHsetelement_(vertexT, facet->vertices, vertex1) {
        if (ii >= 3) {
          throw MeshError{"Qhull did not return triangle"};
        }
        face_globalid[adr + ind[ii++]] = qh_pointid(qh, vertex1->point);
      }
      adr += 3;
    }

    qh_freeqhull(qh, !qh_ALL);
    qh_memfreeshort(qh, &curlong, &totlong);

    if (!ok) {
      szgraph_ = 0;
      graph_.clear();
      throw MeshError{"Could not construct convex hull graph"};
    }

    for (int k = 0; k < numvert + 3 * numface; k++) {
      if (edge_localid[k] >= 0) {
        int a;
        for (a = 0; a < numvert; a++) {
          if (vert_globalid[a] == edge_localid[k]) {
            edge_localid[k] = a;
            break;
          }
        }
        if (a >= numvert) {
          throw MeshError{"Vertex id not found in convex hull"};
        }
      }
    }
  } else {
    qh_freeqhull(qh, !qh_ALL);
    qh_memfreeshort(qh, &curlong, &totlong);
    graph_.clear();
    szgraph_ = 0;
    throw MeshError{"qhull error"};
  }
}

void Mesh::CopyGraph() {
  if (!face_.empty()) {
    return;
  }
  int numvert = graph_[0];
  face_.assign(3 * graph_[1], 0);
  for (int i = 0; i < nface(); i++) {
    int j = 2 + 3 * numvert + 3 * nface() + 3 * i;
    face_[3 * i + 0] = graph_[j + 0];
    face_[3 * i + 1] = graph_[j + 1];
    face_[3 * i + 2] = graph_[j + 2];
  }
}

void Mesh::MakeNormal(const double* dvert) {
  if (!normal_.empty()) {
    return;
  }
  normal_.assign(3 * nvert(), 0);
  if (facenormal_.empty()) {
    facenormal_.assign(3 * nface(), 0);
  }

  for (int i = 0; i < nface(); i++) {
    int vertid[3];
    for (int j = 0; j < 3; j++) vertid[j] = face_[3 * i + j];
    double vec01[3], vec02[3];
    for (int j = 0; j < 3; j++) {
      vec01[j] = dvert[3 * vertid[1] + j] - dvert[3 * vertid[0] + j];
      vec02[j] = dvert[3 * vertid[2] + j] - dvert[3 * vertid[0] + j];
    }
    double nrm[3];
    mjuu_crossvec(nrm, vec01, vec02);
    double area = mjuu_normvec(nrm, 3);
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        normal_[3 * vertid[j] + k] += nrm[k] * area;
      }
      facenormal_[3 * i + j] = vertid[j];
    }
  }

  if (!smoothnormal) {
    std::vector<float> nremove(3 * nnormal(), 0.0f);
    for (int i = 0; i < nface(); i++) {
      int vertid[3];
      for (int j = 0; j < 3; j++) vertid[j] = face_[3 * i + j];
      double vec01[3], vec02[3];
      for (int j = 0; j < 3; j++) {
        vec01[j] = dvert[3 * vertid[1] + j] - dvert[3 * vertid[0] + j];
        vec02[j] = dvert[3 * vertid[2] + j] - dvert[3 * vertid[0] + j];
      }
      double nrm[3];
      mjuu_crossvec(nrm, vec01, vec02);
      double area = mjuu_normvec(nrm, 3);
      for (int j = 0; j < 3; j++) {
        double vnrm[3] = {normal_[3 * vertid[j]], normal_[3 * vertid[j] + 1],
                          normal_[3 * vertid[j] + 2]};
        mjuu_normvec(vnrm, 3);
        if (mjuu_dot3(nrm, vnrm) < 0.8) {
          for (int k = 0; k < 3; k++) {
            nremove[3 * vertid[j] + k] += nrm[k] * area;
          }
        }
      }
    }
    for (int i = 0; i < 3 * nnormal(); i++) {
      normal_[i] -= nremove[i];
    }
  }

  for (int i = 0; i < nnormal(); i++) {
    float len = sqrtf(normal_[3 * i] * normal_[3 * i] +
                      normal_[3 * i + 1] * normal_[3 * i + 1] +
                      normal_[3 * i + 2] * normal_[3 * i + 2]);
    if (len > mjMINVAL) {
      for (int j = 0; j < 3; j++) normal_[3 * i + j] /= len;
    } else {
      normal_[3 * i] = normal_[3 * i + 1] = 0;
      normal_[3 * i + 2] = 1;
    }
  }
}

void Mesh::MakeCenter(const double* dvert) {
  if (!center_.empty()) {
    return;
  }
  center_.assign(3 * nface(), 0);
  for (int i = 0; i < nface(); i++) {
    const int* vertid = face_.data() + 3 * i;
    double a[3], b[3];
    for (int j = 0; j < 3; j++) {
      a[j] = dvert[3 * vertid[0] + j] - dvert[3 * vertid[2] + j];
      b[j] = dvert[3 * vertid[1] + j] - dvert[3 * vertid[2] + j];
    }
    double nrm[3];
    mjuu_crossvec(nrm, a, b);
    double norm_a_2 = mjuu_dot3(a, a);
    double norm_b_2 = mjuu_dot3(b, b);
    double area = sqrt(mjuu_dot3(nrm, nrm));
    double res[3], vec[3] = {
      norm_a_2 * b[0] - norm_b_2 * a[0],
      norm_a_2 * b[1] - norm_b_2 * a[1],
      norm_a_2 * b[2] - norm_b_2 * a[2]
    };
    mjuu_crossvec(res, vec, nrm);
    center_[3 * i + 0] = res[0] / (2 * area * area) + dvert[3 * vertid[2] + 0];
    center_[3 * i + 1] = res[1] / (2 * area * area) + dvert[3 * vertid[2] + 1];
    center_[3 * i + 2] = res[2] / (2 * area * area) + dvert[3 * vertid[2] + 2];
  }
}

void Mesh::MakePolygonNormals(const double* dvert) {
  for (int i = 0; i < static_cast<int>(polygons_.size()); ++i) {
    double n[3];
    mjuu_makenormal(n, &dvert[3 * polygons_[i][0]], &dvert[3 * polygons_[i][1]],
                    &dvert[3 * polygons_[i][2]]);
    polygon_normals_[3 * i + 0] = n[0];
    polygon_normals_[3 * i + 1] = n[1];
    polygon_normals_[3 * i + 2] = n[2];
  }
}

void Mesh::MakePolygons(const double* dvert) {
  constexpr double kAngleTol = 0.01;
  std::unordered_map<std::pair<double, double>, MeshPolygon, PairHash> mesh_polygons;
  polygons_.clear();
  polygon_normals_.clear();
  polygon_map_.clear();
  polygon_map_.resize(nvert());

  int* faces = GraphFaces();
  int nfaces = graph_[1];

  for (int i = 0; i < nfaces; i++) {
    int vi1 = faces[3 * i + 0];
    int vi2 = faces[3 * i + 1];
    int vi3 = faces[3 * i + 2];
    const double* v1 = &dvert[3 * vi1];
    const double* v2 = &dvert[3 * vi2];
    const double* v3 = &dvert[3 * vi3];

    std::pair<double, double> key;
    if (!MeshPolygonKey(key, v1, v2, v3, kAngleTol)) {
      continue;
    }
    auto it = mesh_polygons.find(key);
    if (it == mesh_polygons.end()) {
      double theta = kAngleTol * key.first;
      double phi = kAngleTol * key.second;
      mesh_polygons.emplace(key, MeshPolygon(v1, v2, v3, vi1, vi2, vi3, theta, phi));
    } else {
      it->second.InsertFace(vi1, vi2, vi3);
    }
  }

  for (const auto& pair : mesh_polygons) {
    const MeshPolygon& polygon = pair.second;
    std::vector<std::vector<int>> paths = polygon.Paths();
    for (const auto& path : paths) {
      if (path.size() < 3) continue;
      polygons_.push_back(path);
      polygon_normals_.push_back(polygon.Normal(0));
      polygon_normals_.push_back(polygon.Normal(1));
      polygon_normals_.push_back(polygon.Normal(2));
    }
  }

  for (int i = 0; i < static_cast<int>(polygons_.size()); i++) {
    for (int j = 0; j < static_cast<int>(polygons_[i].size()); ++j) {
      polygon_map_[polygons_[i][j]].push_back(i);
    }
  }
}

void Mesh::ApplyTransformations(double* dvert) {
  if (refpos[0] != 0 || refpos[1] != 0 || refpos[2] != 0) {
    int nv = nvert();
    for (int i = 0; i < nv; i++) {
      dvert[3 * i + 0] -= refpos[0];
      dvert[3 * i + 1] -= refpos[1];
      dvert[3 * i + 2] -= refpos[2];
    }
  }

  if (refquat[0] != 1 || refquat[1] != 0 || refquat[2] != 0 || refquat[3] != 0) {
    double quat[4] = {refquat[0], refquat[1], refquat[2], refquat[3]};
    double mat[9];
    mjuu_normvec(quat, 4);
    mjuu_quat2mat(mat, quat);
    for (int i = 0; i < nvert(); i++) {
      mjuu_mulvecmatT(&dvert[3 * i], &dvert[3 * i], mat);
    }
    for (int i = 0; i < nnormal(); i++) {
      double n1[3], n0[3] = {normal_[3 * i], normal_[3 * i + 1], normal_[3 * i + 2]};
      mjuu_mulvecmatT(n1, n0, mat);
      normal_[3 * i] = (float)n1[0];
      normal_[3 * i + 1] = (float)n1[1];
      normal_[3 * i + 2] = (float)n1[2];
    }
  }

  if (scale[0] != 1 || scale[1] != 1 || scale[2] != 1) {
    for (int i = 0; i < nvert(); i++) {
      dvert[3 * i + 0] *= scale[0];
      dvert[3 * i + 1] *= scale[1];
      dvert[3 * i + 2] *= scale[2];
    }
    for (int i = 0; i < nnormal(); i++) {
      normal_[3 * i + 0] *= scale[0];
      normal_[3 * i + 1] *= scale[1];
      normal_[3 * i + 2] *= scale[2];
    }
  }

  for (int i = 0; i < nnormal(); i++) {
    float len = normal_[3 * i] * normal_[3 * i] +
                normal_[3 * i + 1] * normal_[3 * i + 1] +
                normal_[3 * i + 2] * normal_[3 * i + 2];
    if (len > mjMINVAL) {
      float scl = 1 / sqrtf(len);
      normal_[3 * i + 0] *= scl;
      normal_[3 * i + 1] *= scl;
      normal_[3 * i + 2] *= scl;
    } else {
      normal_[3 * i + 0] = 0;
      normal_[3 * i + 1] = 0;
      normal_[3 * i + 2] = 1;
    }
  }
}

double Mesh::ComputeFaceCentroid(double facecen[3], const double* dvert) const {
  double total_area = 0;
  for (int i = 0; i < nface(); i++) {
    double area, center[3];
    area = triangle(nullptr, center, &dvert[3 * face_[3 * i]],
                    &dvert[3 * face_[3 * i + 1]], &dvert[3 * face_[3 * i + 2]]);
    facecen[0] += area * center[0];
    facecen[1] += area * center[1];
    facecen[2] += area * center[2];
    total_area += area;
  }
  if (total_area >= mjMINVAL) {
    facecen[0] /= total_area;
    facecen[1] /= total_area;
    facecen[2] /= total_area;
  }
  return total_area;
}

double Mesh::ComputeVolume(double CoM[3], const double facecen[3],
                           const double* dvert) const {
  double normal[3], center[3], total_volume = 0;
  CoM[0] = CoM[1] = CoM[2] = 0;
  int nf = (inertia == mjMESH_INERTIA_CONVEX) ? graph_[1] : nface();
  const int* f = (inertia == mjMESH_INERTIA_CONVEX)
                     ? const_cast<Mesh*>(this)->GraphFaces()
                     : face_.data();

  for (int i = 0; i < nf; i++) {
    double area = triangle(normal, center, &dvert[3 * f[3 * i]],
                           &dvert[3 * f[3 * i + 1]], &dvert[3 * f[3 * i + 2]]);
    double vec[3] = {center[0] - facecen[0], center[1] - facecen[1],
                     center[2] - facecen[2]};
    double volume = mjuu_dot3(vec, normal) * area / 3;
    if (inertia == mjMESH_INERTIA_LEGACY) {
      volume = std::abs(volume);
    }
    total_volume += volume;
    CoM[0] += volume * (center[0] * 3.0 / 4.0 + facecen[0] / 4.0);
    CoM[1] += volume * (center[1] * 3.0 / 4.0 + facecen[1] / 4.0);
    CoM[2] += volume * (center[2] * 3.0 / 4.0 + facecen[2] / 4.0);
  }
  if (total_volume >= mjMINVAL) {
    CoM[0] /= total_volume;
    CoM[1] /= total_volume;
    CoM[2] /= total_volume;
  }
  return total_volume;
}

double Mesh::ComputeSurfaceArea(double CoM[3], const double facecen[3],
                                const double* dvert) const {
  double surface = 0;
  CoM[0] = CoM[1] = CoM[2] = 0;
  for (int i = 0; i < nface(); i++) {
    double area, center[3];
    area = triangle(nullptr, center, &dvert[3 * face_[3 * i]],
                    &dvert[3 * face_[3 * i + 1]], &dvert[3 * face_[3 * i + 2]]);
    surface += area;
    CoM[0] += area * (center[0] * 3.0 / 4.0 + facecen[0] / 4.0);
    CoM[1] += area * (center[1] * 3.0 / 4.0 + facecen[1] / 4.0);
    CoM[2] += area * (center[2] * 3.0 / 4.0 + facecen[2] / 4.0);
  }
  if (surface >= mjMINVAL) {
    CoM[0] /= surface;
    CoM[1] /= surface;
    CoM[2] /= surface;
  }
  return surface;
}

double Mesh::ComputeInertia(double inert[6], const double CoM[3],
                            const double* dvert) const {
  double total_volume = 0;
  std::vector<double> vert_centered;
  vert_centered.reserve(3 * nvert());
  for (int i = 0; i < nvert(); i++) {
    vert_centered.push_back(dvert[3 * i + 0] - CoM[0]);
    vert_centered.push_back(dvert[3 * i + 1] - CoM[1]);
    vert_centered.push_back(dvert[3 * i + 2] - CoM[2]);
  }

  const int k[6][2] = {{0, 0}, {1, 1}, {2, 2}, {0, 1}, {0, 2}, {1, 2}};
  double P[6] = {0, 0, 0, 0, 0, 0};
  int nf = (inertia == mjMESH_INERTIA_CONVEX) ? graph_[1] : nface();
  const int* f = (inertia == mjMESH_INERTIA_CONVEX)
                     ? const_cast<Mesh*>(this)->GraphFaces()
                     : face_.data();
  for (int i = 0; i < nf; i++) {
    const double* D = &vert_centered[3 * f[3 * i + 0]];
    const double* E = &vert_centered[3 * f[3 * i + 1]];
    const double* F = &vert_centered[3 * f[3 * i + 2]];
    double normal[3], center[3];
    double volume, area = triangle(normal, center, D, E, F);
    if (inertia == mjMESH_INERTIA_SHELL) {
      volume = area;
    } else {
      volume = mjuu_dot3(center, normal) * area / 3;
    }
    if (inertia == mjMESH_INERTIA_LEGACY) {
      volume = abs(volume);
    }
    total_volume += volume;
    int C = (inertia == mjMESH_INERTIA_SHELL) ? 12 : 20;
    for (int j = 0; j < 6; j++) {
      P[j] += volume / C * (
        2 * (D[k[j][0]] * D[k[j][1]] +
             E[k[j][0]] * E[k[j][1]] +
             F[k[j][0]] * F[k[j][1]]) +
        D[k[j][0]] * E[k[j][1]] + D[k[j][1]] * E[k[j][0]] +
        D[k[j][0]] * F[k[j][1]] + D[k[j][1]] * F[k[j][0]] +
        E[k[j][0]] * F[k[j][1]] + E[k[j][1]] * F[k[j][0]]);
    }
  }

  inert[0] = P[1] + P[2];
  inert[1] = P[0] + P[2];
  inert[2] = P[0] + P[1];
  inert[3] = -P[3];
  inert[4] = -P[4];
  inert[5] = -P[5];
  return total_volume;
}

void Mesh::Rotate(double quat[4], double* dvert) {
  double neg[4] = {quat[0], -quat[1], -quat[2], -quat[3]};
  double mat[9];
  mjuu_quat2mat(mat, neg);
  int nv = nvert();
  for (int i = 0; i < nv; i++) {
    mjuu_mulvecmat(&dvert[3 * i], &dvert[3 * i], mat);
    aamm_[0] = min(aamm_[0], dvert[3 * i + 0]);
    aamm_[3] = max(aamm_[3], dvert[3 * i + 0]);
    aamm_[1] = min(aamm_[1], dvert[3 * i + 1]);
    aamm_[4] = max(aamm_[4], dvert[3 * i + 1]);
    aamm_[2] = min(aamm_[2], dvert[3 * i + 2]);
    aamm_[5] = max(aamm_[5], dvert[3 * i + 2]);
  }
  for (int i = 0; i < nnormal(); i++) {
    const double nrm[3] = {normal_[3 * i], normal_[3 * i + 1], normal_[3 * i + 2]};
    double res[3];
    mjuu_mulvecmat(res, nrm, mat);
    for (int j = 0; j < 3; j++) {
      normal_[3 * i + j] = (float)res[j];
    }
  }
}

void Mesh::SetBoundingVolume(int faceid, const double* dvert) {
  constexpr double kMaxVal = std::numeric_limits<double>::max();
  double face_aamm[6] = {kMaxVal, kMaxVal, kMaxVal, -kMaxVal, -kMaxVal, -kMaxVal};
  for (int j = 0; j < 3; j++) {
    int vertid = face_[3 * faceid + j];
    face_aamm[0] = min(face_aamm[0], dvert[3 * vertid + 0]);
    face_aamm[1] = min(face_aamm[1], dvert[3 * vertid + 1]);
    face_aamm[2] = min(face_aamm[2], dvert[3 * vertid + 2]);
    face_aamm[3] = max(face_aamm[3], dvert[3 * vertid + 0]);
    face_aamm[4] = max(face_aamm[4], dvert[3 * vertid + 1]);
    face_aamm[5] = max(face_aamm[5], dvert[3 * vertid + 2]);
  }
  face_aabb_.push_back(.5 * (face_aamm[0] + face_aamm[3]));
  face_aabb_.push_back(.5 * (face_aamm[1] + face_aamm[4]));
  face_aabb_.push_back(.5 * (face_aamm[2] + face_aamm[5]));
  face_aabb_.push_back(.5 * (face_aamm[3] - face_aamm[0]));
  face_aabb_.push_back(.5 * (face_aamm[4] - face_aamm[1]));
  face_aabb_.push_back(.5 * (face_aamm[5] - face_aamm[2]));
}

// user_mesh.cc Process(): the ordered driver. Octree (SDF) is gated out.
void Mesh::Process() {
  std::vector<double> dvert(vert_.begin(), vert_.end());

  if (halfedge_.empty()) {
    for (int i = 0; i < nface(); i++) {
      int v0 = face_[3 * i + 0];
      int v1 = face_[3 * i + 1];
      int v2 = face_[3 * i + 2];
      if (triangle(nullptr, nullptr, &dvert[3 * v0], &dvert[3 * v1],
                   &dvert[3 * v2]) > sqrt(mjMINVAL)) {
        halfedge_.push_back({v0, v1});
        halfedge_.push_back({v1, v2});
        halfedge_.push_back({v2, v0});
      }
    }
  }

  if (!halfedge_.empty()) {
    std::stable_sort(halfedge_.begin(), halfedge_.end());
    auto iterator = std::adjacent_find(halfedge_.begin(), halfedge_.end());
    if (iterator != halfedge_.end() && inertia == mjMESH_INERTIA_EXACT) {
      throw MeshError{"faces of mesh have inconsistent orientation"};
    }
  }

  if (needhull_ || face_.empty()) {
    MakeGraph(dvert.data());
  }
  if (face_.empty()) {
    CopyGraph();
  }

  if (normal_.empty()) {
    MakeNormal(dvert.data());
  }
  if (!facenormal_.empty() && facenormal_.size() != face_.size()) {
    throw MeshError{"face data must have the same size as face normal data"};
  }
  if (facetexcoord_.empty() && !texcoord_.empty()) {
    facetexcoord_ = face_;
  }
  if (facenormal_.empty()) {
    int normal_per_vertex =
        static_cast<int>(normal_.size() / vert_.size());
    facenormal_.assign(face_.size(), 0);
    for (int i = 0; i < static_cast<int>(face_.size()); i++) {
      facenormal_[i] = normal_per_vertex * face_[i];
    }
  }

  if (szgraph_) {
    MakePolygons(dvert.data());
  } else {
    polygon_map_.resize(nvert());
  }

  bool righthand = scale[0] * scale[1] * scale[2] > 0;
  if (!righthand) {
    for (size_t i = 0; i < face_.size(); i += 3) {
      std::swap(face_[i + 1], face_[i + 2]);
    }
    for (size_t i = 0; i < facenormal_.size(); i += 3) {
      std::swap(facenormal_[i + 1], facenormal_[i + 2]);
    }
    for (size_t i = 0; i < facetexcoord_.size(); i += 3) {
      std::swap(facetexcoord_[i + 1], facetexcoord_[i + 2]);
    }
  }

  ApplyTransformations(dvert.data());

  double facecen[3] = {0, 0, 0};
  if (ComputeFaceCentroid(facecen, dvert.data()) < mjMINVAL) {
    throw MeshError{"mesh surface area is too small"};
  }

  double CoM[3] = {0, 0, 0};
  double inert[6] = {0, 0, 0, 0, 0, 0};

  if (inertia == mjMESH_INERTIA_SHELL) {
    surface_ = ComputeSurfaceArea(CoM, facecen, dvert.data());
    if (surface_ < mjMINVAL) {
      throw MeshError{"mesh surface area is too small"};
    }
  } else {
    if ((volume_ = ComputeVolume(CoM, facecen, dvert.data())) < mjMINVAL) {
      if (volume_ < 0) {
        throw MeshError{"mesh volume is negative (misoriented triangles)"};
      } else {
        throw MeshError{"mesh volume is too small; try setting inertia to shell"};
      }
    }
  }

  double total_volume = ComputeInertia(inert, CoM, dvert.data());
  if (inertia == mjMESH_INERTIA_SHELL) {
    surface_ = total_volume;
  } else {
    volume_ = total_volume;
  }

  double eigval[3], eigvec[9], quattmp[4];
  double full[9] = {
    inert[0], inert[3], inert[4],
    inert[3], inert[1], inert[5],
    inert[4], inert[5], inert[2]
  };
  mjuu_eig3(eigval, eigvec, quattmp, full);

  constexpr double inequality_atol = 1e-9;
  constexpr double inequality_rtol = 1e-6;
  if (eigval[2] <= 0) {
    throw MeshError{"eigenvalue of mesh inertia must be positive"};
  }
  if (eigval[0] + eigval[1] < eigval[2] * (1.0 - inequality_rtol) - inequality_atol ||
      eigval[0] + eigval[2] < eigval[1] * (1.0 - inequality_rtol) - inequality_atol ||
      eigval[1] + eigval[2] < eigval[0] * (1.0 - inequality_rtol) - inequality_atol) {
    throw MeshError{"eigenvalues of mesh inertia violate A + B >= C"};
  }

  double volume = GetVolumeRef();
  boxsz_[0] = 0.5 * std::sqrt(6 * (eigval[1] + eigval[2] - eigval[0]) / volume);
  boxsz_[1] = 0.5 * std::sqrt(6 * (eigval[0] + eigval[2] - eigval[1]) / volume);
  boxsz_[2] = 0.5 * std::sqrt(6 * (eigval[0] + eigval[1] - eigval[2]) / volume);

  // Marching-cubes SDF meshes keep their generated frame (mjCMesh::Process,
  // user_mesh.cc:1511-1515): null CoM/quat so no reorientation is applied.
  if (!needreorient_) {
    mjuu_setvec(CoM, 0, 0, 0);
    mjuu_setvec(quattmp, 1, 0, 0, 0);
  }
  for (int i = 0; i < nvert(); i++) {
    dvert[3 * i + 0] -= CoM[0];
    dvert[3 * i + 1] -= CoM[1];
    dvert[3 * i + 2] -= CoM[2];
  }
  Rotate(quattmp, dvert.data());

  mjuu_copyvec(pos_, CoM, 3);
  mjuu_copyvec(quat_, quattmp, 4);

  if (center_.empty()) {
    MakeCenter(dvert.data());
  }
  MakePolygonNormals(dvert.data());

  face_aabb_.clear();
  face_aabb_.reserve(6 * face_.size());
  for (int i = 0; i < nface(); i++) {
    SetBoundingVolume(i, dvert.data());
  }

  for (int i = 0; i < static_cast<int>(dvert.size()); i++) {
    vert_[i] = (float)dvert[i];
  }
}

}  // namespace

bool CompileMesh(const MeshInput& in, MeshResult& out, std::string& err) {
  Mesh m;
  mjuu_copyvec(m.scale, in.scale, 3);
  mjuu_copyvec(m.refpos, in.refpos, 3);
  mjuu_copyvec(m.refquat, in.refquat, 4);
  m.smoothnormal = in.smoothnormal;
  m.maxhullvert_ = in.maxhullvert;
  m.inertia = in.inertia;
  m.needhull_ = in.needhull;
  m.needreorient_ = in.needreorient;
  m.content_type_ = in.content_type;

  try {
    switch (in.format) {
      case MeshFormat::UserVertex:
        m.normal_.assign(in.usernormal.begin(), in.usernormal.end());
        m.texcoord_.assign(in.usertexcoord.begin(), in.usertexcoord.end());
        m.face_.assign(in.userface.begin(), in.userface.end());
        m.facenormal_.assign(in.userfacenormal.begin(), in.userfacenormal.end());
        m.facetexcoord_.assign(in.userfacetexcoord.begin(),
                               in.userfacetexcoord.end());
        m.ProcessVertices(in.uservert);
        break;
      case MeshFormat::Msh:
        m.LoadMSH(in.filebytes);
        break;
      case MeshFormat::Obj:
        m.LoadOBJ(in.filebytes);
        break;
      case MeshFormat::Stl:
        m.LoadSTL(in.filebytes);
        break;
    }
    m.CheckInitialMesh();
    m.Process();
  } catch (const MeshError& e) {
    err = e.msg;
    return false;
  }

  out.vert = std::move(m.vert_);
  out.normal = std::move(m.normal_);
  out.texcoord = std::move(m.texcoord_);
  out.face = std::move(m.face_);
  out.facenormal = std::move(m.facenormal_);
  out.facetexcoord = std::move(m.facetexcoord_);
  out.graph = std::move(m.graph_);
  out.polygons = std::move(m.polygons_);
  out.polygon_normals = std::move(m.polygon_normals_);
  out.polygon_map = std::move(m.polygon_map_);
  out.center = std::move(m.center_);
  out.face_aabb = std::move(m.face_aabb_);
  mjuu_copyvec(out.boxsz, m.boxsz_, 3);
  mjuu_copyvec(out.aamm, m.aamm_, 6);
  mjuu_copyvec(out.pos, m.pos_, 3);
  mjuu_copyvec(out.quat, m.quat_, 4);
  mjuu_copyvec(out.scale, m.scale, 3);
  out.volume = m.volume_;
  out.surface = m.surface_;
  out.volume_ref = m.GetVolumeRef();
  return true;
}

bool LoadMeshRaw(const MeshInput& in, MeshRawResult& out, std::string& err) {
  Mesh m;
  // scale sign drives the MSH righthand face flip; refpos/refquat and the
  // inertia pipeline are irrelevant to the raw load, so leave them defaulted.
  mjuu_copyvec(m.scale, in.scale, 3);
  m.content_type_ = in.content_type;

  try {
    switch (in.format) {
      case MeshFormat::Obj:
        m.LoadOBJ(in.filebytes);
        break;
      case MeshFormat::Stl:
        m.LoadSTL(in.filebytes);
        break;
      case MeshFormat::Msh:
      case MeshFormat::UserVertex:
        err = "unsupported mesh format for flexcomp raw load";
        return false;
    }
  } catch (const MeshError& e) {
    err = e.msg;
    return false;
  }

  out.vert = std::move(m.vert_);
  out.face = std::move(m.face_);
  out.texcoord = std::move(m.texcoord_);
  out.facetexcoord = std::move(m.facetexcoord_);
  return true;
}

}  // namespace ps::mjcf::compile::lifted
