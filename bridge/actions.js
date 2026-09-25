// Real-time continuous motor controls and camera look application.
// Fast path: <0.2ms execution without blocking.

const MAX_YAW_RATE = 3.5;    // rad/s (~200 deg/s max turn speed)
const MAX_PITCH_RATE = 2.0;  // rad/s (~115 deg/s max pitch speed)
const DEADZONE = 0.04;       // normalized rate deadzone to prevent camera drift
const TAU = 0.04;            // 40ms low-pass filter time constant

function applyContinuousControls(bot, motor) {
  if (!bot || !bot.entity || !motor) return;

  const isForward = motor.move_z > 0.1;
  bot.setControlState('forward', isForward);
  bot.setControlState('back', motor.move_z < -0.2);
  bot.setControlState('left', motor.move_x < -0.2);
  bot.setControlState('right', motor.move_x > 0.2);

  bot.setControlState('jump', Boolean(motor.jump));
  bot.setControlState('sneak', Boolean(motor.sneak));
  bot.setControlState('sprint', Boolean(motor.sprint));

  if (!bot._motorState) {
    bot._motorState = {
      lastTime: Date.now() / 1000.0,
      smoothedYawRate: 0.0,
      smoothedPitchRate: 0.0,
    };
  }

  const state = bot._motorState;
  const now = Date.now() / 1000.0;
  const dt = Math.max(0.002, Math.min(0.1, now - state.lastTime));
  state.lastTime = now;

  // Extract normalized angular rates (pure motor_rate_v2 action semantics)
  const rawYaw = Number.isFinite(motor.yaw_rate) ? motor.yaw_rate : 0.0;
  const rawPitch = Number.isFinite(motor.pitch_rate) ? motor.pitch_rate : 0.0;

  // Frequency-invariant exponential moving average (low-pass filter)
  const alpha = 1.0 - Math.exp(-dt / TAU);
  state.smoothedYawRate += alpha * (Math.max(-1.0, Math.min(1.0, rawYaw)) - state.smoothedYawRate);
  state.smoothedPitchRate += alpha * (Math.max(-1.0, Math.min(1.0, rawPitch)) - state.smoothedPitchRate);

  // Apply deadzone to eliminate minor stochastic noise
  const effYawRate = Math.abs(state.smoothedYawRate) < DEADZONE ? 0.0 : state.smoothedYawRate;
  const effPitchRate = Math.abs(state.smoothedPitchRate) < DEADZONE ? 0.0 : state.smoothedPitchRate;

  // Integrate over dt to get physical angular increment
  const deltaYaw = effYawRate * MAX_YAW_RATE * dt;
  const deltaPitch = effPitchRate * MAX_PITCH_RATE * dt;

  if (Math.abs(deltaYaw) > 1e-4 || Math.abs(deltaPitch) > 1e-4) {
    const newYaw = bot.entity.yaw + deltaYaw;
    const newPitch = Math.max(-1.3, Math.min(1.3, bot.entity.pitch + deltaPitch));
    bot.look(newYaw, newPitch, true).catch(() => {});
  }
}

module.exports = { applyContinuousControls };

