// Tier 6 & Tier 7: Container and processing operations for Mineflayer
async function executeContainerAction(bot, cmd) {
  const primitive = cmd.primitive;

  try {
    switch (primitive) {
      case 'open_container': {
        const targetBlock = bot.blockAtCursor ? bot.blockAtCursor(4.5) : null;
        if (!targetBlock) {
          return { success: false, reason: 'NO_TARGET_BLOCK_IN_RANGE', delta: {} };
        }
        const containerBlocks = ['chest', 'trapped_chest', 'barrel', 'furnace', 'blast_furnace', 'smoker', 'shulker_box'];
        const isContainer = containerBlocks.some(c => targetBlock.name.includes(c));
        if (!isContainer) {
          return { success: false, reason: 'TARGET_NOT_A_CONTAINER', delta: {} };
        }
        await bot.openContainer(targetBlock);
        return { success: true, reason: 'NONE', delta: { container_opened: targetBlock.name } };
      }

      case 'close_container': {
        if (!bot.currentWindow) {
          return { success: false, reason: 'NO_OPEN_CONTAINER', delta: {} };
        }
        bot.closeWindow(bot.currentWindow);
        return { success: true, reason: 'NONE', delta: { container_closed: true } };
      }

      case 'container_move': {
        if (!bot.currentWindow) {
          return { success: false, reason: 'NO_OPEN_CONTAINER', delta: {} };
        }
        const src = Number(cmd.target_slot || 0);
        const dest = Number(cmd.target_slot_dest || 0);
        await bot.moveSlotItem(src, dest);
        return { success: true, reason: 'NONE', delta: { container_slot_moved: [src, dest] } };
      }

      case 'smelt': {
        if (!bot.currentWindow || !bot.currentWindow.type.includes('furnace')) {
          return { success: false, reason: 'FURNACE_NOT_OPEN', delta: {} };
        }
        // Slot 0 is input, slot 1 is fuel
        const held = bot.heldItem;
        if (!held) {
          return { success: false, reason: 'NO_HELD_ITEM_FOR_SMELTING', delta: {} };
        }
        await bot.putItemStack(0, held);
        return { success: true, reason: 'NONE', delta: { input_deposited: held.name } };
      }

      default:
        return { success: false, reason: 'UNSUPPORTED_CONTAINER_PRIMITIVE', delta: {} };
    }
  } catch (err) {
    return { success: false, reason: err.message || 'CONTAINER_EXCEPTION', delta: {} };
  }
}

module.exports = { executeContainerAction };
