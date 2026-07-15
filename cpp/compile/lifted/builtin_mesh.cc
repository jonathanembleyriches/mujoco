// Lifted from MuJoCo src/user/user_mesh.cc (mjCMesh::MakeHemisphere/MakeSphere/
// MakeSupersphere/MakeSupertorus/MakeWedge/MakeRect/MakeCone + the file-local
// Fovea/LinSpace/BinEdges/SphericalToCartesian/TangentFrame/aux_c/aux_s helpers)
// and src/user/user_api.cc (mjs_makeMesh). The generators are copied verbatim,
// retargeting mjs_setFloat/mjs_setInt writes to plain output vectors and the
// mjCError throws to a returned error string. Provenance + registry:
// snapshots/lifted_code.json. Upstream: mjCMesh::MakeSphere et al.
#include "builtin_mesh.h"

#include <cmath>
#include <map>
#include <utility>

#include "mjuu_util.h"

namespace ps::mjcf::compile::lifted {
namespace {

constexpr double mjPI = 3.14159265358979323846;

inline double mjMAX(double a, double b) { return a > b ? a : b; }
inline double mjMIN(double a, double b) { return a < b ? a : b; }

// --- file-local helpers (user_mesh.cc:83-150), verbatim ------------------- //

double Fovea(double x, double gamma) {
  if (!gamma) return x;
  double g = mjMAX(0, mjMIN(1, gamma));
  return g * std::pow(x, 5) + (1 - g) * x;
}

void LinSpace(double lower, double upper, int n, double array[]) {
  double increment = n > 1 ? (upper - lower) / (n - 1) : 0;
  for (int i = 0; i < n; ++i) {
    *array = lower;
    ++array;
    lower += increment;
  }
}

void BinEdges(double* x_edges, double* y_edges, int size[2], double fov[2],
              double gamma) {
  LinSpace(-1, 1, size[0] + 1, x_edges);
  LinSpace(-1, 1, size[1] + 1, y_edges);
  for (int i = 0; i < size[0] + 1; i++) x_edges[i] = Fovea(x_edges[i], gamma);
  for (int i = 0; i < size[1] + 1; i++) y_edges[i] = Fovea(y_edges[i], gamma);
  mjuu_scalevec(x_edges, x_edges, fov[0] * mjPI / 180, size[0] + 1);
  mjuu_scalevec(y_edges, y_edges, fov[1] * mjPI / 180, size[1] + 1);
}

void SphericalToCartesian(const double aer[3], float xyz[3]) {
  double a = aer[0], e = aer[1], r = aer[2];
  xyz[0] = r * std::cos(e) * std::sin(a);
  xyz[1] = r * std::sin(e);
  xyz[2] = -r * std::cos(e) * std::cos(a);
}

void TangentFrame(const double aer[3], float mat[9]) {
  double a = aer[0], e = aer[1], r = aer[2];
  double ta[3] = {r * std::cos(e) * std::cos(a), 0, r * std::cos(e) * std::sin(a)};
  double te[3] = {-r * std::sin(e) * std::sin(a), r * std::cos(e),
                  r * std::sin(e) * std::cos(a)};
  double n[3];
  mjuu_normvec(ta, 3);
  mjuu_normvec(te, 3);
  mjuu_copyvec(mat + 3, ta, 3);
  mjuu_copyvec(mat + 6, te, 3);
  mjuu_crossvec(n, te, ta);
  mjuu_copyvec(mat, n, 3);
}

double aux_c(double omega, double m) {
  return std::copysign(std::pow(std::abs(std::cos(omega)), m), std::cos(omega));
}
double aux_s(double omega, double m) {
  return std::copysign(std::pow(std::abs(std::sin(omega)), m), std::sin(omega));
}

// --- generators (user_mesh.cc:2010-2498), verbatim ------------------------ //

void MakeHemisphere(int res, bool make_faces, bool make_cap,
                    BuiltinMeshResult& o) {
  constexpr double kNorthPole[3] = {0, 0, 1};
  constexpr double kEquator[4][3] = {
      {1, 0, 0}, {0, 1, 0}, {-1, 0, 0}, {0, -1, 0},
  };
  int nvert = 1 + 2 * (res + 1) * (res + 2);
  nvert += make_cap && make_faces;
  std::vector<float> vert(3 * nvert);
  vert[0] = kNorthPole[0];
  vert[1] = kNorthPole[1];
  vert[2] = kNorthPole[2];
  int v = 1;
  for (int row = 0; row <= res; row++) {
    for (int side = 0; side < 4; side++) {
      double factor = static_cast<double>(row + 1) / (res + 1);
      double start[3], end[3];
      for (int i = 0; i < 3; i++) {
        start[i] = kNorthPole[i] + factor * (kEquator[side][i] - kNorthPole[i]);
        end[i] = kNorthPole[i] + factor * (kEquator[(side + 1) % 4][i] - kNorthPole[i]);
      }
      double delta[3];
      for (int i = 0; i < 3; i++) delta[i] = (end[i] - start[i]) / (row + 1);
      for (int i = 0; i < row + 1; i++) {
        double p[3];
        for (int j = 0; j < 3; j++) p[j] = start[j] + i * delta[j];
        double norm = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
        vert[3 * v + 0] = p[0] / norm;
        vert[3 * v + 1] = p[1] / norm;
        vert[3 * v + 2] = p[2] / norm;
        v++;
      }
    }
  }
  if (make_faces && make_cap) {
    vert[3 * (nvert - 1) + 0] = 0;
    vert[3 * (nvert - 1) + 1] = 0;
    vert[3 * (nvert - 1) + 2] = 0;
  }
  o.uservert = std::move(vert);
  if (make_faces) {
    int nface = 4 * (res + 1) * (res + 1);
    nface += make_cap * (4 * (res + 1));
    std::vector<int> face(3 * nface);
    int f = 0;
    face[f++] = 0; face[f++] = 1; face[f++] = 2;
    face[f++] = 0; face[f++] = 2; face[f++] = 3;
    face[f++] = 0; face[f++] = 3; face[f++] = 4;
    face[f++] = 0; face[f++] = 4; face[f++] = 1;
    for (int row = 0; row < res; row++) {
      const int start_curr = 2 * row * (row + 1) + 1;
      const int count_curr = 4 * (row + 1);
      const int start_next = start_curr + count_curr;
      const int count_next = 4 * (row + 2);
      for (int side = 0; side < 4; side++) {
        for (int i = 0; i < row + 2; i++) {
          const int v_curr = i + (row + 1) * side;
          const int v_next = i + (row + 2) * side;
          face[f++] = start_curr + v_curr % count_curr;
          face[f++] = start_next + v_next % count_next;
          face[f++] = start_next + (v_next + 1) % count_next;
          if (i < row + 1) {
            face[f++] = start_curr + v_curr % count_curr;
            face[f++] = start_next + (v_next + 1) % count_next;
            face[f++] = start_curr + (v_curr + 1) % count_curr;
          }
        }
      }
    }
    if (make_cap) {
      const int start = 2 * res * (res + 1) + 1;
      const int count = 4 * (res + 1);
      for (int i = 0; i < count; i++) {
        face[f++] = start + i;
        face[f++] = nvert - 1;
        face[f++] = start + (i + 1) % count;
      }
    }
    o.userface = std::move(face);
  }
}

void MakeSphere(int subdiv, bool make_faces, BuiltinMeshResult& o) {
  const float phi = (1.0 + std::sqrt(5.0)) / 2.0;
  std::vector<float> vert = {
      -1.0f,  phi, 0.0f,   1.0f,  phi, 0.0f,  -1.0f, -phi, 0.0f,   1.0f, -phi, 0.0f,
       0.0f, -1.0f,  phi,   0.0f,  1.0f,  phi,   0.0f, -1.0f, -phi,   0.0f,  1.0f, -phi,
       phi,  0.0f, -1.0f,   phi,  0.0f,  1.0f,  -phi,  0.0f, -1.0f,  -phi,  0.0f,  1.0f,
  };
  const double norm = std::sqrt(1.0 + phi * phi);
  for (float& x : vert) x /= norm;
  std::vector<int> face = {
    0,  11, 5,    0,  5,  1,    0,  1,  7,    0,  7,  10,   0,  10, 11,
    1,  5,  9,    5,  11, 4,    11, 10, 2,    10, 7,  6,    7,  1,  8,
    3,  9,  4,    3,  4,  2,    3,  2,  6,    3,  6,  8,    3,  8,  9,
    4,  9,  5,    2,  4,  11,   6,  2,  10,   8,  6,  7,    9,  8,  1
  };
  if (subdiv > 0) {
    auto get_midpoint = [&vert](int v1_idx, int v2_idx,
                                std::map<std::pair<int, int>, int>& cache) -> int {
      std::pair<int, int> key = std::minmax(v1_idx, v2_idx);
      auto it = cache.find(key);
      if (it != cache.end()) return it->second;
      const float* v1 = &vert[v1_idx * 3];
      const float* v2 = &vert[v2_idx * 3];
      float mid_x = (v1[0] + v2[0]) / 2.0f;
      float mid_y = (v1[1] + v2[1]) / 2.0f;
      float mid_z = (v1[2] + v2[2]) / 2.0f;
      float mid_norm = std::sqrt(mid_x * mid_x + mid_y * mid_y + mid_z * mid_z);
      mid_x /= mid_norm;
      mid_y /= mid_norm;
      mid_z /= mid_norm;
      int new_idx = vert.size() / 3;
      vert.push_back(mid_x);
      vert.push_back(mid_y);
      vert.push_back(mid_z);
      cache[key] = new_idx;
      return new_idx;
    };
    for (int i = 0; i < subdiv; ++i) {
      std::map<std::pair<int, int>, int> midpoint_cache;
      std::vector<int> new_face;
      new_face.reserve(face.size() * 4);
      for (size_t j = 0; j < face.size(); j += 3) {
        int v1 = face[j], v2 = face[j + 1], v3 = face[j + 2];
        int m12 = get_midpoint(v1, v2, midpoint_cache);
        int m23 = get_midpoint(v2, v3, midpoint_cache);
        int m31 = get_midpoint(v3, v1, midpoint_cache);
        new_face.insert(new_face.end(), {v1, m12, m31});
        new_face.insert(new_face.end(), {v2, m23, m12});
        new_face.insert(new_face.end(), {v3, m31, m23});
        new_face.insert(new_face.end(), {m12, m23, m31});
      }
      face = std::move(new_face);
    }
  }
  o.uservert = std::move(vert);
  if (make_faces) o.userface = std::move(face);
}

void MakeSupersphere(int res, double e, double n, BuiltinMeshResult& o) {
  int nvert = (res - 1) * res + 2;
  int nface = 2 * res * (res - 1);
  std::vector<float> vert;
  vert.reserve(3 * nvert);
  std::vector<int> face;
  face.reserve(3 * nface);
  vert.insert(vert.end(), {0.0f, 0.0f, -1.0f});
  for (int i = 1; i < res; i++) {
    double vv = -mjPI / 2 + i * mjPI / res;
    for (int j = 0; j < res; j++) {
      double u = -mjPI + j * 2 * mjPI / res;
      vert.push_back(aux_c(vv, n) * aux_c(u, e));
      vert.push_back(aux_c(vv, n) * aux_s(u, e));
      vert.push_back(aux_s(vv, n));
    }
  }
  vert.insert(vert.end(), {0.0f, 0.0f, 1.0f});
  for (int j = 0; j < res; j++) {
    int v2 = 1 + j;
    int v3 = 1 + (j + 1) % res;
    face.insert(face.end(), {0, v3, v2});
  }
  for (int i = 0; i < res - 2; i++) {
    for (int j = 0; j < res; j++) {
      int v1 = 1 + i * res + j;
      int v2 = 1 + i * res + (j + 1) % res;
      int v4 = 1 + (i + 1) * res + j;
      int v3 = 1 + (i + 1) * res + (j + 1) % res;
      face.insert(face.end(), {v1, v2, v4});
      face.insert(face.end(), {v2, v3, v4});
    }
  }
  int north_pole_idx = nvert - 1;
  int last_ring_start_idx = 1 + (res - 2) * res;
  for (int j = 0; j < res; j++) {
    int v1 = last_ring_start_idx + j;
    int v2 = last_ring_start_idx + (j + 1) % res;
    face.insert(face.end(), {v1, v2, north_pole_idx});
  }
  o.uservert = std::move(vert);
  o.userface = std::move(face);
}

void MakeSupertorus(int res, double radius, double s, double t,
                    BuiltinMeshResult& o) {
  int nvert = res * res;
  int nface = res * res * 2;
  std::vector<float> vert(3 * nvert);
  std::vector<int> face(3 * nface);
  for (int i = 0; i < res; ++i) {
    for (int j = 0; j < res; ++j) {
      double u = 2 * mjPI * i / res;
      double vv = 2 * mjPI * j / res;
      int vidx = i * res + j;
      vert[3 * vidx + 0] = (1 + radius * aux_c(vv, s)) * aux_c(u, t);
      vert[3 * vidx + 1] = (1 + radius * aux_c(vv, s)) * aux_s(u, t);
      vert[3 * vidx + 2] = radius * aux_s(vv, s);
    }
  }
  int fidx = 0;
  for (int i = 0; i < res; ++i) {
    for (int j = 0; j < res; ++j) {
      int i_next = (i + 1) % res;
      int j_next = (j + 1) % res;
      int v1 = i * res + j;
      int v2 = i_next * res + j;
      int v3 = i_next * res + j_next;
      int v4 = i * res + j_next;
      face[3 * fidx + 0] = v1;
      face[3 * fidx + 1] = v2;
      face[3 * fidx + 2] = v4;
      fidx++;
      face[3 * fidx + 0] = v2;
      face[3 * fidx + 1] = v3;
      face[3 * fidx + 2] = v4;
      fidx++;
    }
  }
  o.uservert = std::move(vert);
  o.userface = std::move(face);
}

void MakeWedge(int resolution[2], double fov[2], double gamma,
               BuiltinMeshResult& o) {
  std::vector<double> x_edges(resolution[0] + 1, 0);
  std::vector<double> y_edges(resolution[1] + 1, 0);
  BinEdges(x_edges.data(), y_edges.data(), resolution, fov, gamma);
  std::vector<float> uservert(3 * resolution[0] * resolution[1], 0);
  std::vector<float> usernormal(9 * resolution[0] * resolution[1], 0);
  for (int i = 0; i < resolution[0]; i++) {
    for (int j = 0; j < resolution[1]; j++) {
      double aer[3];
      aer[0] = 0.5 * (x_edges[i + 1] + x_edges[i]);
      aer[1] = 0.5 * (y_edges[j + 1] + y_edges[j]);
      aer[2] = 1;
      SphericalToCartesian(aer, uservert.data() + 3 * (i * resolution[1] + j));
      TangentFrame(aer, usernormal.data() + 9 * (i * resolution[1] + j));
    }
  }
  o.uservert = std::move(uservert);
  o.usernormal = std::move(usernormal);
}

void MakeRect(int resolution[2], BuiltinMeshResult& o) {
  std::vector<float> uservert(3 * resolution[0] * resolution[1], 0);
  std::vector<float> usernormal(9 * resolution[0] * resolution[1], 0);
  std::vector<int> userface(6 * (resolution[0] - 1) * (resolution[1] - 1), 0);
  o.inertia_shell = true;
  for (int i = 0; i < resolution[0]; i++) {
    for (int j = 0; j < resolution[1]; j++) {
      int vert = i * resolution[1] + j;
      double dx = 2. / resolution[0];
      double dy = 2. / resolution[1];
      uservert[3 * vert + 0] = -1 + (i + 0.5) * dx;
      uservert[3 * vert + 1] = -1 + (j + 0.5) * dy;
      uservert[3 * vert + 2] = -1;
      usernormal[9 * vert + 0] = 1;
      usernormal[9 * vert + 4] = 1;
      usernormal[9 * vert + 8] = 1;
      if (i > 0 && j > 0) {
        int cell = (i - 1) * (resolution[1] - 1) + j - 1;
        userface[6 * cell + 0] = (i - 1) * resolution[1] + j - 1;
        userface[6 * cell + 1] = (i - 0) * resolution[1] + j - 1;
        userface[6 * cell + 2] = (i - 1) * resolution[1] + j - 0;
        userface[6 * cell + 3] = (i - 0) * resolution[1] + j - 0;
        userface[6 * cell + 4] = (i - 1) * resolution[1] + j - 0;
        userface[6 * cell + 5] = (i - 0) * resolution[1] + j - 1;
      }
    }
  }
  o.uservert = std::move(uservert);
  o.usernormal = std::move(usernormal);
  o.userface = std::move(userface);
}

void MakeCone(int nedge, double radius, BuiltinMeshResult& o) {
  int n = 3 * (nedge + (radius > 0 ? nedge : 1));
  std::vector<float> uservert(n, 0);
  for (int i = 0; i < nedge; i++) {
    uservert[3 * i + 0] = std::cos(2 * i * mjPI / nedge);
    uservert[3 * i + 1] = std::sin(2 * i * mjPI / nedge);
    uservert[3 * i + 2] = -1;
  }
  if (radius > 0) {
    for (int i = nedge; i < 2 * nedge; i++) {
      uservert[3 * i + 0] = radius * std::cos(2 * i * mjPI / nedge);
      uservert[3 * i + 1] = radius * std::sin(2 * i * mjPI / nedge);
      uservert[3 * i + 2] = 1;
    }
  } else {
    uservert[3 * nedge + 2] = 1;
  }
  o.uservert = std::move(uservert);
}

}  // namespace

bool MakeBuiltinMesh(MeshBuiltinKind kind, const std::vector<double>& params,
                     BuiltinMeshResult& out, std::string& err) {
  const int np = static_cast<int>(params.size());
  auto p = [&](int i) { return params[i]; };
  switch (kind) {
    case MeshBuiltinKind::Hemisphere: {
      if (np != 1) { err = "Hemisphere mesh type requires 1 parameter"; return false; }
      int subdiv = static_cast<int>(p(0));
      if (subdiv < 0) { err = "Hemisphere resolution cannot be negative"; return false; }
      if (subdiv > 10) { err = "Hemisphere resolution cannot be greater than 10"; return false; }
      MakeHemisphere(subdiv, true, true, out);
      return true;
    }
    case MeshBuiltinKind::Sphere: {
      if (np != 1) { err = "Sphere mesh type requires 1 parameter"; return false; }
      int subdiv = static_cast<int>(p(0));
      if (subdiv < 0) { err = "Sphere subdivision cannot be negative"; return false; }
      if (subdiv > 4) { err = "Sphere subdivision cannot be greater than 4"; return false; }
      MakeSphere(subdiv, true, out);
      return true;
    }
    case MeshBuiltinKind::Supersphere: {
      if (np != 3) { err = "Supersphere mesh type requires 3 parameters"; return false; }
      int res = static_cast<int>(p(0));
      if (res < 3) { err = "Supersphere resolution must be greater than 2"; return false; }
      double e = p(1);
      if (e < 0) { err = "Supersphere 'e' cannot be negative"; return false; }
      double n = p(2);
      if (n < 0) { err = "Supersphere 'n' cannot be negative"; return false; }
      MakeSupersphere(res, e, n, out);
      return true;
    }
    case MeshBuiltinKind::Supertorus: {
      if (np != 4) { err = "Supertorus mesh type requires 4 parameters"; return false; }
      int res = static_cast<int>(p(0));
      if (res < 3) { err = "Supertorus resolution must be greater than 3"; return false; }
      double radius = p(1);
      if (radius <= 0 || radius > 1) { err = "Supertorus radius must be in (0, 1]"; return false; }
      double s = p(2);
      if (s <= 0) { err = "Supertorus 's' must be greater than 0"; return false; }
      double t = p(3);
      if (t <= 0) { err = "Supertorus 't' must be greater than 0"; return false; }
      MakeSupertorus(res, radius, s, t, out);
      return true;
    }
    case MeshBuiltinKind::Wedge: {
      if (np != 5) { err = "Wedge builtin mesh types require 5 parameters"; return false; }
      int resolution[2] = {static_cast<int>(p(0)), static_cast<int>(p(1))};
      double fov[2] = {p(2), p(3)};
      double gamma = p(4);
      if (fov[0] <= 0 || fov[0] > 180) { err = "fov[0] must be a float between (0, 180] degrees"; return false; }
      if (fov[1] <= 0 || fov[1] > 90) { err = "`fov[1]` must be a float between (0, 90] degrees"; return false; }
      if (resolution[0] <= 0 || resolution[1] <= 0) { err = "Horizontal and vertical resolutions must be positive"; return false; }
      if (gamma < 0 || gamma > 1) { err = "`gamma` must be a nonnegative float between [0, 1]"; return false; }
      MakeWedge(resolution, fov, gamma, out);
      return true;
    }
    case MeshBuiltinKind::Plate: {
      if (np != 2) { err = "Plate builtin mesh type requires 2 parameters"; return false; }
      int resolution[2] = {static_cast<int>(p(0)), static_cast<int>(p(1))};
      if (resolution[0] <= 0 || resolution[1] <= 0) { err = "Horizontal and vertical resolutions must be positive"; return false; }
      MakeRect(resolution, out);
      return true;
    }
    case MeshBuiltinKind::Cone: {
      if (np != 2) { err = "Cone mesh type requires 2 parameters"; return false; }
      int nedge = static_cast<int>(p(0));
      MakeCone(nedge, p(1), out);
      return true;
    }
  }
  err = "Unsupported mesh type";
  return false;
}

}  // namespace ps::mjcf::compile::lifted
