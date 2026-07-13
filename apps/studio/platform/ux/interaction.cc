// Vendored from MuJoCo Studio, adapted for ProtoSpec Studio (ps::studio).
//
// Upstream: src/experimental/platform/ux/interaction.cc @ mujoco 67a1ea6d
// Adaptation: kept the pick-ray math (MakePickRay/PickGeom/PickFlex/PickSkin/
// Pick), MoveCamera and SetCamera. Dropped InitPerturb/MovePerturb (perturb
// wiring is not part of the Edit-mode tree-edit model; see the header). Includes
// private engine visualization headers (engine_vis_*), same as cpp/harness: the
// vendored src/ dir is on the include path.

#include "platform/ux/interaction.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <mujoco/mujoco.h>
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_misc.h"
#include "engine/engine_vis_interact.h"
#include "engine/engine_vis_visualize.h"

namespace ps::studio {

static mjtNum CalculateMovementScale(const mjModel* m, const mjvCamera* cam) {
  float zclip[2] = {0, 0}, zver[2] = {0, 0};
  mjv_cameraFrustum(zver, nullptr, zclip, m, cam);
  if (cam->orthographic) {
    return (zver[1] + zver[0]) * 0.15;
  } else if (zclip[0] >= mjMINVAL) {
    return (zver[1] + zver[0]) / zclip[0];
  } else {
    mjERROR("mjvScene frustum_near too small");
    return 0;
  }
}

static void AlignToCamera(mjtNum res[3], mjtMouse action, mjtNum dx, mjtNum dy,
                          const mjtNum forward[3]) {
  mjtNum vec[3];
  switch (action) {
    case mjMOUSE_MOVE_V:
    case mjMOUSE_MOVE_V_REL:
      vec[0] = dx;
      vec[1] = 0;
      vec[2] = -dy;
      break;
    case mjMOUSE_MOVE_H:
    case mjMOUSE_MOVE_H_REL:
      vec[0] = dx;
      vec[1] = -dy;
      vec[2] = 0;
      break;
    default:
      vec[0] = vec[1] = vec[2] = 0;
      break;
  }
  mjv_alignToCamera(res, vec, forward);
}

void MoveCamera(const mjModel* m, const mjData* d, mjvCamera* cam,
                CameraMotion motion, mjtNum dx, mjtNum dy) {
  if (cam->type == mjCAMERA_FIXED) {
    return;
  }

  mjtNum headpos[3], forward[3], up[3], right[3];
  mjtNum vec[3], dif[3], scl;

  switch (motion) {
    case CameraMotion::ZOOM:
      cam->distance -= mju_log(1 + cam->distance / m->stat.extent / 3) * dy *
                       9 * m->stat.extent;
      break;

    case CameraMotion::ORBIT:
      cam->azimuth -= dx * 180.0;
      cam->elevation -= dy * 180.0;
      break;

    case CameraMotion::TRUCK_PEDESTAL:
    case CameraMotion::TRUCK_DOLLY:
      if (cam->type == mjCAMERA_TRACKING) {
        return;
      }
      mjv_cameraFrame(headpos, forward, up, nullptr, d, cam);
      mju_cross(right, forward, up);
      mju_addToScl3(cam->lookat,
                    (motion == CameraMotion::TRUCK_PEDESTAL) ? up : forward, dy);
      mju_addToScl3(cam->lookat, right, dx);
      break;

    case CameraMotion::PAN_TILT:
      if (cam->type == mjCAMERA_TRACKING) {
        return;
      }
      mjv_cameraFrame(headpos, forward, nullptr, nullptr, d, cam);
      cam->azimuth -= dx * 180.0;
      cam->elevation -= dy * 180.0;
      mjv_cameraFrame(nullptr, forward, nullptr, nullptr, d, cam);
      mju_addScl3(cam->lookat, headpos, forward, cam->distance);
      break;

    case CameraMotion::PLANAR_MOVE_V:
    case CameraMotion::PLANAR_MOVE_H:
      if (cam->type == mjCAMERA_TRACKING) {
        return;
      }
      mjv_cameraFrame(headpos, forward, nullptr, nullptr, d, cam);
      AlignToCamera(vec,
                    (motion == CameraMotion::PLANAR_MOVE_V) ? mjMOUSE_MOVE_V
                                                            : mjMOUSE_MOVE_H,
                    dx, dy, forward);
      mju_sub3(dif, cam->lookat, headpos);
      scl = CalculateMovementScale(m, cam) * mju_dot3(dif, forward);
      mju_addToScl3(cam->lookat, vec, -scl);
      break;
  }

  if (cam->azimuth > 180) cam->azimuth -= 360;
  if (cam->azimuth < -180) cam->azimuth += 360;
  if (cam->elevation > 89) cam->elevation = 89;
  if (cam->elevation < -89) cam->elevation = -89;
  if (cam->distance < 0.01 * m->stat.extent) {
    cam->distance = 0.01 * m->stat.extent;
  }
  if (cam->distance > 100 * m->stat.extent) {
    cam->distance = 100 * m->stat.extent;
  }
}

static void MakePickRay(mjtNum pos[3], mjtNum ray[3], const mjModel* m,
                        const mjData* d, const mjvCamera* camera, float relx,
                        float rely, float aspect_ratio) {
  mjtNum forward[3], up[3], right[3];
  mjv_cameraFrame(pos, forward, up, right, d, camera);

  float zver[2], zhor[2], zclip[2] = {0, 0};
  mjv_cameraFrustum(zver, zhor, zclip, m, camera);

  mjtNum half_width = 0.5 * aspect_ratio * (zver[0] + zver[1]);
  mjtNum frustum_center = (zhor[1] - zhor[0]) / 2;

  mjtNum d_up = -zver[0] + rely * (zver[0] + zver[1]);
  mjtNum d_right = frustum_center + (2 * relx - 1) * half_width;

  if (camera->orthographic) {
    mju_copy3(ray, forward);
    mju_addToScl3(pos, up, d_up);
    mju_addToScl3(pos, right, d_right);
  } else {
    mju_scl3(ray, forward, zclip[0]);
    mju_addToScl3(ray, up, d_up);
    mju_addToScl3(ray, right, d_right);
    mju_normalize3(ray);
  }
}

static PickResult PickGeom(const mjModel* m, const mjData* d,
                           const mjtNum ray_pos[3], const mjtNum ray_dir[3],
                           const mjvOption* vis_options) {
  PickResult result;
  result.dist =
      mj_ray(m, d, ray_pos, ray_dir, vis_options->geomgroup,
             vis_options->flags[mjVIS_STATIC], -1, &result.geom, nullptr);
  if (result.geom >= 0) {
    mju_addScl3(result.point, ray_pos, ray_dir, result.dist);
    result.body = m->geom_bodyid[result.geom];
  }
  return result;
}

static PickResult PickFlex(const mjModel* m, const mjData* d,
                           const mjtNum ray_pos[3], const mjtNum ray_dir[3],
                           const mjvOption* vis_options) {
  const mjtByte flag_vert = vis_options->flags[mjVIS_FLEXVERT];
  const mjtByte flag_edge = vis_options->flags[mjVIS_FLEXEDGE];
  const mjtByte flag_face = vis_options->flags[mjVIS_FLEXFACE];
  const mjtByte flag_skin = vis_options->flags[mjVIS_FLEXSKIN];

  PickResult result;
  if (!flag_vert && !flag_edge && !flag_face && !flag_skin) {
    return result;
  }

  for (int i = 0; i < m->nflex; i++) {
    int vertid;
    const mjtNum test_dist =
        mj_rayFlex(m, d, vis_options->flex_layer, flag_vert, flag_edge,
                   flag_face, flag_skin, i, ray_pos, ray_dir, &vertid, nullptr);
    if (test_dist < 0) {
      continue;
    } else if (result.dist >= 0 && test_dist >= result.dist) {
      continue;
    }
    result.dist = test_dist;
    result.flex = i;
    result.body = mjv_flexBodyId(m, d, i, vertid, result.point);
  }
  return result;
}

static void MakeSkin(const mjModel* m, const mjData* d, const mjvOption* opt,
                     int i, float* skinnormal, float* skinvert) {
  int vertadr = m->skin_vertadr[i];
  int vertnum = m->skin_vertnum[i];
  int faceadr = m->skin_faceadr[i];
  int facenum = m->skin_facenum[i];

  for (int j = m->skin_boneadr[i]; j < m->skin_boneadr[i] + m->skin_bonenum[i];
       j++) {
    mjtNum bindpos[3] = {(mjtNum)m->skin_bonebindpos[3 * j + 0],
                         (mjtNum)m->skin_bonebindpos[3 * j + 1],
                         (mjtNum)m->skin_bonebindpos[3 * j + 2]};
    mjtNum bindquat[4] = {(mjtNum)m->skin_bonebindquat[4 * j + 0],
                          (mjtNum)m->skin_bonebindquat[4 * j + 1],
                          (mjtNum)m->skin_bonebindquat[4 * j + 2],
                          (mjtNum)m->skin_bonebindquat[4 * j + 3]};

    int bodyid = m->skin_bonebodyid[j];
    mjtNum quat[4], quatneg[4], rotate[9];
    mju_negQuat(quatneg, bindquat);
    mju_mulQuat(quat, d->xquat + 4 * bodyid, quatneg);
    mju_quat2Mat(rotate, quat);

    mjtNum translate[3];
    mju_mulMatVec3(translate, rotate, bindpos);
    mju_sub3(translate, d->xpos + 3 * bodyid, translate);

    for (int k = m->skin_bonevertadr[j];
         k < m->skin_bonevertadr[j] + m->skin_bonevertnum[j]; k++) {
      int vid = m->skin_bonevertid[k];
      float vweight = m->skin_bonevertweight[k];

      mjtNum pos[3] = {
          (mjtNum)m->skin_vert[3 * (vertadr + vid)],
          (mjtNum)m->skin_vert[3 * (vertadr + vid) + 1],
          (mjtNum)m->skin_vert[3 * (vertadr + vid) + 2],
      };

      mjtNum pos1[3];
      mju_mulMatVec3(pos1, rotate, pos);
      mju_addTo3(pos1, translate);

      skinvert[(3 * vid)] += vweight * (float)pos1[0];
      skinvert[(3 * vid) + 1] += vweight * (float)pos1[1];
      skinvert[(3 * vid) + 2] += vweight * (float)pos1[2];
    }
  }

  if (m->skin_inflate[i] && skinnormal != nullptr) {
    for (int k = faceadr; k < faceadr + facenum; k++) {
      int vid[3] = {m->skin_face[3 * k], m->skin_face[3 * k + 1],
                    m->skin_face[3 * k + 2]};
      mjtNum vec01[3], vec02[3];
      for (int r = 0; r < 3; r++) {
        vec01[r] = skinvert[3 * (vid[1]) + r] - skinvert[3 * (vid[0]) + r];
        vec02[r] = skinvert[3 * (vid[2]) + r] - skinvert[3 * (vid[0]) + r];
      }
      mjtNum nrm[3];
      mju_cross(nrm, vec01, vec02);
      for (int r = 0; r < 3; r++) {
        for (int t = 0; t < 3; t++) {
          skinnormal[3 * (vid[r]) + t] += nrm[t];
        }
      }
    }
    for (int k = 0; k < vertnum; k++) {
      float s = sqrtf(skinnormal[3 * (k) + 0] * skinnormal[3 * k + 0] +
                      skinnormal[3 * (k) + 1] * skinnormal[3 * k + 1] +
                      skinnormal[3 * (k) + 2] * skinnormal[3 * k + 2]);
      float scl = 1 / mjMAX(mjMINVAL, s);
      skinnormal[3 * k] *= scl;
      skinnormal[3 * k + 1] *= scl;
      skinnormal[3 * k + 2] *= scl;
    }
    float inflate = m->skin_inflate[i];
    for (int k = 0; k < vertnum; k++) {
      skinvert[3 * k] += inflate * skinnormal[3 * k];
      skinvert[3 * k + 1] += inflate * skinnormal[3 * k + 1];
      skinvert[3 * k + 2] += inflate * skinnormal[3 * k + 2];
    }
  }
}

static PickResult PickSkin(const mjModel* m, const mjData* d,
                           const mjtNum ray_pos[3], const mjtNum ray_dir[3],
                           const mjvOption* vis_options) {
  PickResult result;
  if (!vis_options->flags[mjVIS_SKIN]) {
    return result;
  }

  std::vector<float> vertex_buffer;
  std::vector<float> normal_buffer;

  for (int i = 0; i < m->nskin; i++) {
    const int skin_group = mjMAX(0, mjMIN(mjNGROUP - 1, m->skin_group[i]));
    if (!vis_options->skingroup[skin_group]) {
      continue;
    }

    vertex_buffer.assign(3 * m->skin_vertnum[i], 0.0f);
    if (m->skin_inflate[i]) {
      normal_buffer.assign(3 * m->skin_vertnum[i], 0.0f);
    }

    float* skinvert = vertex_buffer.data();
    float* skinnormal = m->skin_inflate[i] ? normal_buffer.data() : nullptr;
    MakeSkin(m, d, vis_options, i, skinnormal, skinvert);

    int vertid;
    mjtNum test_dist = mju_raySkin(m->skin_facenum[i], m->skin_vertnum[i],
                                   m->skin_face + 3 * m->skin_faceadr[i],
                                   skinvert, ray_pos, ray_dir, &vertid);
    if (test_dist < 0) {
      continue;
    } else if (result.dist >= 0 && test_dist >= result.dist) {
      continue;
    }

    result.dist = test_dist;

    float best_weight = -1;
    for (int j = m->skin_boneadr[i];
         j < m->skin_boneadr[i] + m->skin_bonenum[i]; j++) {
      for (int k = m->skin_bonevertadr[j];
           k < m->skin_bonevertadr[j] + m->skin_bonevertnum[j]; k++) {
        const int vertex_id = m->skin_bonevertid[k];
        const float vertex_weight = m->skin_bonevertweight[k];
        if (vertex_id == vertid && vertex_weight > best_weight) {
          best_weight = vertex_weight;
          result.body = m->skin_bonebodyid[j];
          result.skin = i;
          mju_f2n(result.point, skinvert + 3 * vertid, 3);
        }
      }
    }
  }
  return result;
}

PickResult Pick(const mjModel* m, const mjData* d, const mjvCamera* camera,
                float x, float y, float aspect_ratio,
                const mjvOption* vis_options) {
  mjtNum ray_pos[3];
  mjtNum ray_dir[3];
  MakePickRay(ray_pos, ray_dir, m, d, camera, x, 1.0 - y, aspect_ratio);

  PickResult results[3];
  results[0] = PickGeom(m, d, ray_pos, ray_dir, vis_options);
  results[1] = PickFlex(m, d, ray_pos, ray_dir, vis_options);
  results[2] = PickSkin(m, d, ray_pos, ray_dir, vis_options);

  PickResult best_result;
  for (int i = 0; i < 3; i++) {
    if (results[i].dist < 0) {
      continue;
    }
    if (best_result.dist < 0 || results[i].dist < best_result.dist) {
      best_result = results[i];
    }
  }
  return best_result;
}

int SetCamera(const mjModel* m, mjvCamera* camera, int request_idx) {
  const int ncam = m ? m->ncam : 0;
  const int camera_idx = std::clamp(request_idx, kTumbleCameraIdx, ncam - 1);

  if (camera_idx == kTumbleCameraIdx) {
    camera->type = mjCAMERA_FREE;
    camera->fixedcamid = -1;
  } else if (camera_idx == kFreeCameraIdx) {
    camera->type = mjCAMERA_FREE;
    camera->distance = 2.0f;
    camera->fixedcamid = -1;
  } else if (camera_idx == kTrackingCameraIdx) {
    camera->type = (camera->trackbodyid >= 0) ? mjCAMERA_TRACKING
                                              : mjCAMERA_FREE;
    camera->fixedcamid = -1;
  } else {
    camera->type = mjCAMERA_FIXED;
    camera->fixedcamid = camera_idx;
  }
  return camera_idx;
}

}  // namespace ps::studio
