const { executeInventoryAction } = require('./inventory');
const { executeCraftingAction } = require('./crafting');
const { executeContainerAction } = require('./containers');
const { executeWorldMechanicsAction } = require('./mechanics');

async function executeHierarchicalAction(bot, action) {
  if (!bot || !bot.entity || !action) {
    return {
      action_primitive: 'noop',
      success: false,
      failure_reason: 'BOT_UNAVAILABLE',
      elapsed_ticks: 0,
      state_delta: {},
    };
  }

  const motor = action.motor || {};
  const cmd = action.command || {};

  // 1. Continuous Locomotion & Camera Orientation
  bot.setControlState('forward', motor.move_z > 0.2);
  bot.setControlState('back', motor.move_z < -0.2);
  bot.setControlState('left', motor.move_x < -0.2);
  bot.setControlState('right', motor.move_x > 0.2);
  bot.setControlState('jump', Boolean(motor.jump));
  bot.setControlState('sneak', Boolean(motor.sneak));
  bot.setControlState('sprint', Boolean(motor.sprint));

  if (Number.isFinite(motor.yaw_delta) && Number.isFinite(motor.pitch_delta)) {
    const newYaw = bot.entity.yaw + motor.yaw_delta * 0.15;
    const newPitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, bot.entity.pitch + motor.pitch_delta * 0.1));
    await bot.look(newYaw, newPitch, true).catch(() => {});
  }

  // 2. Discrete Primitive Dispatch
  const primitive = cmd.primitive || 'noop';
  const startHealth = bot.health;
  let result = { success: true, reason: 'NONE', delta: {} };

  if (['hotbar_select', 'equip', 'unequip', 'swap_slots', 'drop_held', 'drop_stack', 'offhand_swap'].includes(primitive)) {
    result = await executeInventoryAction(bot, cmd);
  } else if (['craft', 'craft_batch'].includes(primitive)) {
    result = await executeCraftingAction(bot, cmd);
  } else if (['open_container', 'close_container', 'container_move', 'smelt'].includes(primitive)) {
    result = await executeContainerAction(bot, cmd);
  } else if (['dig', 'place', 'activate_block', 'toggle_switch', 'use_item', 'attack_entity', 'interact_entity', 'sleep', 'mount', 'dismount'].includes(primitive)) {
    result = await executeWorldMechanicsAction(bot, cmd);
  } else {
    // noop or move_camera
    result = { success: true, reason: 'NONE', delta: {} };
  }

  const healthDelta = Number((bot.health - startHealth).toFixed(2));
  result.delta.health_delta = healthDelta;

  return {
    action_primitive: primitive,
    success: result.success,
    failure_reason: result.reason,
    elapsed_ticks: Number(cmd.duration_ticks || 1),
    state_delta: result.delta,
  };
}

module.exports = { executeHierarchicalAction };
