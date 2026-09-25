// Real-time continuous motor controls and camera look application.
// Fast path: <0.2ms execution without blocking.

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

  if (Number.isFinite(motor.yaw_delta) && Number.isFinite(motor.pitch_delta)) {
    const newYaw = bot.entity.yaw + motor.yaw_delta * 0.12;
    const newPitch = Math.max(-0.55, Math.min(0.55, bot.entity.pitch + motor.pitch_delta * 0.08));
    bot.look(newYaw, newPitch, true).catch(() => {});
  }
}

module.exports = { applyContinuousControls };
