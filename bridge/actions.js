// Action executor translating HierarchicalAction into Mineflayer motor controls and tracking ActionResult

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

  // -------------------------------------------------------------
  // 1. Continuous Locomotion & Camera Control
  // -------------------------------------------------------------
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

  // -------------------------------------------------------------
  // 2. Discrete Action Primitive Execution
  // -------------------------------------------------------------
  const primitive = cmd.primitive || 'noop';
  let success = true;
  let failureReason = 'NONE';
  const startHealth = bot.health;

  try {
    switch (primitive) {
      case 'noop':
      case 'move_camera':
        // Motor controls already applied
        break;

      case 'attack_entity': {
        const entities = Object.values(bot.entities || {}).filter(e => e !== bot.entity && e.position);
        const target = cmd.target_entity_idx < entities.length ? entities[cmd.target_entity_idx] : null;
        if (target && bot.entity.position.distanceTo(target.position) <= 3.8) {
          bot.attack(target);
        } else {
          success = false;
          failureReason = 'OUT_OF_RANGE';
        }
        break;
      }

      case 'interact_entity': {
        const entities = Object.values(bot.entities || {}).filter(e => e !== bot.entity && e.position);
        const target = cmd.target_entity_idx < entities.length ? entities[cmd.target_entity_idx] : null;
        if (target && bot.entity.position.distanceTo(target.position) <= 3.8) {
          bot.useOn(target);
        } else {
          success = false;
          failureReason = 'OUT_OF_RANGE';
        }
        break;
      }

      case 'dig': {
        const targetBlock = bot.blockAtCursor ? bot.blockAtCursor(4.5) : null;
        if (targetBlock) {
          if (bot.canDigBlock && bot.canDigBlock(targetBlock)) {
            bot.dig(targetBlock).catch(() => {});
          } else {
            success = false;
            failureReason = 'CANNOT_DIG';
          }
        } else {
          success = false;
          failureReason = 'NO_TARGET_BLOCK';
        }
        break;
      }

      case 'place': {
        const targetBlock = bot.blockAtCursor ? bot.blockAtCursor(4.5) : null;
        if (targetBlock && bot.heldItem) {
          const faces = [
            { x: 0, y: -1, z: 0 }, { x: 0, y: 1, z: 0 },
            { x: 0, y: 0, z: -1 }, { x: 0, y: 0, z: 1 },
            { x: -1, y: 0, z: 0 }, { x: 1, y: 0, z: 0 },
          ];
          const face = faces[cmd.target_block_face % 6] || { x: 0, y: 1, z: 0 };
          await bot.placeBlock(targetBlock, face).catch(err => {
            success = false;
            failureReason = err.message || 'PLACE_FAILED';
          });
        } else {
          success = false;
          failureReason = targetBlock ? 'NO_HELD_ITEM' : 'NO_TARGET_BLOCK';
        }
        break;
      }

      case 'use_item': {
        if (bot.heldItem) {
          bot.activateItem();
        } else {
          success = false;
          failureReason = 'NO_HELD_ITEM';
        }
        break;
      }

      case 'hotbar_select': {
        const slot = Math.max(0, Math.min(8, Number(cmd.target_slot || 0)));
        bot.setQuickBarSlot(slot);
        break;
      }

      case 'swap_slots': {
        const src = Number(cmd.target_slot || 0);
        const dest = Number(cmd.target_slot_dest || 0);
        if (bot.inventory && src >= 0 && dest >= 0) {
          await bot.moveSlotItem(src, dest).catch(err => {
            success = false;
            failureReason = err.message || 'SWAP_FAILED';
          });
        }
        break;
      }

      case 'drop_held': {
        if (bot.heldItem) {
          await bot.toss(bot.heldItem.type, null, 1).catch(() => {});
        } else {
          success = false;
          failureReason = 'NO_HELD_ITEM';
        }
        break;
      }

      case 'drop_stack': {
        if (bot.heldItem) {
          await bot.tossStack(bot.heldItem).catch(() => {});
        } else {
          success = false;
          failureReason = 'NO_HELD_ITEM';
        }
        break;
      }

      case 'craft': {
        if (cmd.recipe_name) {
          // Look up recipes for this item
          const item = bot.registry.itemsByName[cmd.recipe_name];
          if (item) {
            const recipes = bot.recipesFor(item.id, null, 1, null);
            if (recipes && recipes.length > 0) {
              await bot.craft(recipes[0], 1, null).catch(err => {
                success = false;
                failureReason = err.message || 'CRAFT_FAILED';
              });
            } else {
              success = false;
              failureReason = 'NO_VALID_RECIPE_OR_MATERIALS';
            }
          } else {
            success = false;
            failureReason = 'UNKNOWN_RECIPE_ITEM';
          }
        }
        break;
      }

      case 'sleep': {
        const targetBlock = bot.blockAtCursor ? bot.blockAtCursor(3.0) : null;
        if (targetBlock && bot.isABed(targetBlock)) {
          await bot.sleep(targetBlock).catch(err => {
            success = false;
            failureReason = err.message || 'CANNOT_SLEEP';
          });
        } else {
          success = false;
          failureReason = 'NO_BED_IN_REACH';
        }
        break;
      }

      default:
        break;
    }
  } catch (err) {
    success = false;
    failureReason = err.message || 'EXECUTION_EXCEPTION';
  }

  return {
    action_primitive: primitive,
    success: success,
    failure_reason: failureReason,
    elapsed_ticks: Number(cmd.duration_ticks || 1),
    state_delta: {
      health_delta: Number((bot.health - startHealth).toFixed(2)),
    },
  };
}

module.exports = { executeHierarchicalAction };
