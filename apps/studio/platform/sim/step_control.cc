// Vendored from MuJoCo Studio, adapted for ProtoSpec Studio (ps::studio).
//
// Upstream: src/experimental/platform/sim/step_control.cc @ mujoco 67a1ea6d
// Adaptation: namespace only (mujoco::platform -> ps::studio). Logic unchanged.

#include "platform/sim/step_control.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <optional>
#include <ratio>

#include <mujoco/mujoco.h>

namespace ps::studio {

static mjtNum Timer() {
  using TimerClock = std::chrono::steady_clock;
  using Milliseconds = std::chrono::duration<double, std::milli>;
  static TimerClock::time_point start = TimerClock::now();
  return Milliseconds(TimerClock::now() - start).count();
}

// Sets viscous-pause parameters and restores them when done.
struct ViscousPauseState {
  explicit ViscousPauseState(mjModel* model) : model(model) {
    if (model) {
      mju_copy3(gravity, model->opt.gravity);
      viscosity = model->opt.viscosity;
      disableflags = model->opt.disableflags;
      mju_zero3(model->opt.gravity);
      model->opt.viscosity = 10;
      model->opt.disableflags |= mjDSBL_SPRING;
    }
  }
  ~ViscousPauseState() {
    if (model) {
      mju_copy3(model->opt.gravity, gravity);
      model->opt.viscosity = viscosity;
      model->opt.disableflags = disableflags;
    }
  }
  mjModel* model;
  mjtNum gravity[3];
  mjtNum viscosity;
  int disableflags;
};

StepControl::StepControl() { mjcb_time = Timer; }

float StepControl::GetSpeedMeasured() const { return speed_measured_; }
float StepControl::GetSpeed() const { return speed_; }

void StepControl::SetSpeed(float speed_percent_real_time) {
  speed_ = std::clamp(speed_percent_real_time, .1f, 100.f);
  ForceSync();
}

void StepControl::ForceSync() { force_sync_ = true; }

void StepControl::GetNoiseParameters(float& ctrl_noise_scale,
                                     float& ctrl_noise_rate) const {
  ctrl_noise_scale = ctrl_noise_std_;
  ctrl_noise_rate = ctrl_noise_rate_;
}

void StepControl::SetNoiseParameters(float ctrl_noise_scale,
                                     float ctrl_noise_rate) {
  ctrl_noise_std_ = ctrl_noise_scale;
  ctrl_noise_rate_ = ctrl_noise_rate;
}

void StepControl::SetPauseState(PauseState state) { pause_state_ = state; }
StepControl::PauseState StepControl::GetPauseState() const {
  return pause_state_;
}

StepControl::Status StepControl::Advance(mjModel* m, mjData* d,
                                         StepFn step_fn) {
  if (!m) {
    return Status::kOk;
  }

  std::optional<ViscousPauseState> viscous_pause_state;
  if (pause_state_ == PauseState::kViscousPaused) {
    viscous_pause_state.emplace(m);
  }

  if (pause_state_ == PauseState::kNormalPaused) {
    force_sync_ = true;
    if (!single_step_) {
      mj_forward(m, d);
      if (pause_update_) {
        mju_copy(d->qacc_warmstart, d->qacc, m->nv);
      }
      return Status::kPaused;
    }
    single_step_ = false;
  }

  const Clock::time_point start_cpu = Clock::now();
  const double slowdown = 100. / std::clamp<double>(speed_, 0.001, 100.);
  double elapsed_cpu = Seconds(start_cpu - sync_cpu_).count();
  double elapsed_sim = d->time - sync_sim_;

  bool resync = false;
  if (force_sync_) {
    force_sync_ = false;
    resync = true;
  }
  if (sync_cpu_.time_since_epoch().count() == 0) {
    resync = true;
  }
  if (elapsed_cpu < 0 || elapsed_sim < 0) {
    resync = true;
  }
  if (std::abs(elapsed_cpu / slowdown - elapsed_sim) > sync_misalign_) {
    resync = true;
  }
  if (resync) {
    sync_cpu_ = start_cpu;
    sync_sim_ = d->time;
  }

  while (true) {
    const Clock::time_point now_cpu = Clock::now();
    elapsed_cpu = Seconds(now_cpu - sync_cpu_).count();
    elapsed_sim = d->time - sync_sim_;

    if (elapsed_sim * slowdown >= elapsed_cpu) {
      return Status::kOk;
    }

    constexpr Clock::duration kMaxCpuTimeForSim = std::chrono::milliseconds(12);
    if (now_cpu - start_cpu >= kMaxCpuTimeForSim) {
      return Status::kOk;
    }

    if (elapsed_sim > 0) {
      double measured_slowdown = elapsed_cpu / elapsed_sim;
      speed_measured_ = 100. / measured_slowdown;
    }

    mjtNum prev_time = d->time;
    InjectNoise(m, d);
    if (step_fn) {
      step_fn(m, d);
    } else {
      mj_step(m, d);
    }

    if (mjDISABLED(mjDSBL_AUTORESET)) {
      for (mjtWarning w : kDivergedWarnings) {
        if (d->warning[w].number > 0) {
          SetPauseState(PauseState::kNormalPaused);
          return Status::kDiverged;
        }
      }
    } else {
      if (d->time < prev_time) {
        return Status::kAutoReset;
      }
    }

    if (resync) {
      return Status::kOk;
    }
  }
}

void StepControl::InjectNoise(const mjModel* m, mjData* d) {
  if (ctrl_noise_std_ <= 0) {
    return;
  }

  mjtNum rate = mju_exp(-m->opt.timestep / ctrl_noise_rate_);
  mjtNum scale = ctrl_noise_std_ * mju_sqrt(1 - rate * rate);

  for (int i = 0; i < m->nu; i++) {
    mjtNum bottom = 0;
    mjtNum top = 0;
    mjtNum midpoint = 0;
    mjtNum halfrange = 1;
    if (m->actuator_ctrllimited[i]) {
      bottom = m->actuator_ctrlrange[2 * i];
      top = m->actuator_ctrlrange[2 * i + 1];
      midpoint = 0.5 * (top + bottom);
      halfrange = 0.5 * (top - bottom);
    }
    d->ctrl[i] = rate * d->ctrl[i] + (1 - rate) * midpoint;
    d->ctrl[i] += scale * halfrange * mju_standardNormal(nullptr);
    if (m->actuator_ctrllimited[i]) {
      d->ctrl[i] = mju_clip(d->ctrl[i], bottom, top);
    }
  }
}

}  // namespace ps::studio
