// Tier 2, 3, 8, 9: World manipulation, interaction, vehicle, and environmental mechanics
async function executeWorldMechanicsAction(bot, cmd) {
  const primitive = cmd.primitive;

  try {
    switch (primitive) {
      case 'dig': {
        const targetBlock = bot.blockAtCursor ? bot.blockAtCursor(4.5) : null;
        if (!targetBlock) {
          return { success: false, reason: 'NO_TARGET_BLOCK_IN_REACH', delta: {} };
        }
        if (!bot.canDigBlock(targetBlock)) {
          return { success: false, reason: 'BLOCK_NOT_DIGGABLE', delta: {} };
        }
        await bot.dig(targetBlock);
        return { success: true, reason: 'NONE', delta: { block_dug: targetBlock.name } };
      }

      case 'place': {
        const targetBlock = bot.blockAtCursor ? bot.blockAtCursor(4.5) : null;
        if (!targetBlock) {
          return { success: false, reason: 'NO_TARGET_BLOCK_IN_REACH', delta: {} };
        }
        if (!bot.heldItem) {
          return { success: false, reason: 'NO_HELD_ITEM_TO_PLACE', delta: {} };
        }
        const faces = [
          { x: 0, y: -1, z: 0 }, { x: 0, y: 1, z: 0 },
          { x: 0, y: 0, z: -1 }, { x: 0, y: 0, z: 1 },
          { x: -1, y: 0, z: 0 }, { x: 1, y: 0, z: 0 },
        ];
        const face = faces[cmd.target_block_face % 6] || { x: 0, y: 1, z: 0 };
        await bot.placeBlock(targetBlock, face);
        return { success: true, reason: 'NONE', delta: { block_placed: bot.heldItem.name } };
      }

      case 'activate_block':
      case 'toggle_switch': {
        const targetBlock = bot.blockAtCursor ? bot.blockAtCursor(4.5) : null;
        if (!targetBlock) {
          return { success: false, reason: 'NO_TARGET_BLOCK_IN_REACH', delta: {} };
        }
        await bot.activateBlock(targetBlock);
        return { success: true, reason: 'NONE', delta: { block_activated: targetBlock.name } };
      }

      case 'use_item': {
        if (!bot.heldItem) {
          return { success: false, reason: 'NO_HELD_ITEM_TO_USE', delta: {} };
        }
        bot.activateItem();
        return { success: true, reason: 'NONE', delta: { item_used: bot.heldItem.name } };
      }

      case 'attack_entity': {
        const entities = Object.values(bot.entities || {}).filter(e => e !== bot.entity && e.position);
        const target = cmd.target_entity_idx < entities.length ? entities[cmd.target_entity_idx] : null;
        if (!target) {
          return { success: false, reason: 'NO_TARGET_ENTITY', delta: {} };
        }
        if (bot.entity.position.distanceTo(target.position) > 3.8) {
          return { success: false, reason: 'ENTITY_OUT_OF_RANGE', delta: {} };
        }
        bot.attack(target);
        return { success: true, reason: 'NONE', delta: { attacked_entity: target.name || 'unknown' } };
      }

      case 'interact_entity': {
        const entities = Object.values(bot.entities || {}).filter(e => e !== bot.entity && e.position);
        const target = cmd.target_entity_idx < entities.length ? entities[cmd.target_entity_idx] : null;
        if (!target) {
          return { success: false, reason: 'NO_TARGET_ENTITY', delta: {} };
        }
        if (bot.entity.position.distanceTo(target.position) > 3.8) {
          return { success: false, reason: 'ENTITY_OUT_OF_RANGE', delta: {} };
        }
        bot.useOn(target);
        return { success: true, reason: 'NONE', delta: { interacted_entity: target.name || 'unknown' } };
      }

      case 'sleep': {
        const targetBlock = bot.blockAtCursor ? bot.blockAtCursor(3.5) : null;
        if (!targetBlock || !bot.isABed(targetBlock)) {
          return { success: false, reason: 'NO_BED_IN_REACH', delta: {} };
        }
        await bot.sleep(targetBlock);
        return { success: true, reason: 'NONE', delta: { sleeping: true } };
      }

      case 'mount': {
        const vehicle = bot.nearestEntity(e => ['boat', 'minecart', 'horse', 'donkey'].includes(e.name) && bot.entity.position.distanceTo(e.position) <= 3.5);
        if (!vehicle) {
          return { success: false, reason: 'NO_VEHICLE_NEARBY', delta: {} };
        }
        bot.mount(vehicle);
        return { success: true, reason: 'NONE', delta: { mounted: vehicle.name } };
      }

      case 'dismount': {
        if (!bot.vehicle) {
          return { success: false, reason: 'NOT_MOUNTED', delta: {} };
        }
        bot.dismount();
        return { success: true, reason: 'NONE', delta: { dismounted: true } };
      }

      default:
        return { success: false, reason: 'UNSUPPORTED_WORLD_PRIMITIVE', delta: {} };
    }
  } catch (err) {
    return { success: false, reason: err.message || 'WORLD_MECHANICS_EXCEPTION', delta: {} };
  }
}

module.exports = { executeWorldMechanicsAction };
