// Vendored from MuJoCo Studio, adapted for ProtoSpec Studio (ps::studio).
//
// Upstream: src/experimental/platform/sim/step_control.h @ mujoco 67a1ea6d
// Adaptation: namespace only (mujoco::platform -> ps::studio). Logic unchanged.
//
// State and logic for physics synchronization and stepping (pause/run/speed/
// single-step), respecting a real-time budget.

#ifndef PS_STUDIO_PLATFORM_SIM_STEP_CONTROL_H_
#define PS_STUDIO_PLATFORM_SIM_STEP_CONTROL_H_

#include <chrono>
#include <functional>
#include <string>

#include <mujoco/mujoco.h>

namespace ps::studio {

using Seconds = std::chrono::duration<double>;
using Clock = std::chrono::steady_clock;
using StepFn = std::function<void(mjModel*, mjData*)>;

class StepControl {
 public:
  StepControl();

  enum class Status {
    kOk,
    kPaused,
    kViscousPaused,
    kAutoReset,
    kDiverged,
  };

  static constexpr mjtWarning kDivergedWarnings[] = {
      mjWARN_BADQACC, mjWARN_BADQVEL, mjWARN_BADQPOS};

  // Steps physics forward, respecting speed settings and refresh budget.
  Status Advance(mjModel* m, mjData* d, StepFn step_fn = nullptr);

  // Ensures the next Advance() re-synchronizes time and steps once.
  void ForceSync();

  float GetSpeed() const;
  float GetSpeedMeasured() const;
  void SetSpeed(float speed);  // clamped to [0.1%, 100%]

  void GetNoiseParameters(float& noise_scale, float& noise_rate) const;
  void SetNoiseParameters(float noise_scale, float noise_rate);

  enum class PauseState { kUnpaused, kNormalPaused, kViscousPaused };

  void SetPauseState(PauseState state);
  PauseState GetPauseState() const;

  // Performs a single step on the next Advance() if paused.
  void RequestSingleStep() { single_step_ = true; }

 private:
  void InjectNoise(const mjModel* m, mjData* d);

  double ctrl_noise_std_ = 0;
  double ctrl_noise_rate_ = 0;
  float speed_ = 100;
  float speed_measured_ = -1;
  bool force_sync_ = true;
  std::chrono::time_point<Clock> sync_cpu_;
  mjtNum sync_sim_ = 0;
  double sync_misalign_ = .1;
  PauseState pause_state_ = PauseState::kUnpaused;
  bool single_step_ = false;
  bool pause_update_ = false;
};

}  // namespace ps::studio

#endif  // PS_STUDIO_PLATFORM_SIM_STEP_CONTROL_H_
