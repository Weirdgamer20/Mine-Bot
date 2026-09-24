// Tier 4: Inventory management operations for Mineflayer
async function executeInventoryAction(bot, cmd) {
  const primitive = cmd.primitive;
  const srcSlot = Number(cmd.target_slot || 0);
  const destSlot = Number(cmd.target_slot_dest || 0);

  if (!bot || !bot.inventory) {
    return { success: false, reason: 'INVENTORY_UNAVAILABLE', delta: {} };
  }

  const initialHeld = bot.heldItem ? { type: bot.heldItem.type, count: bot.heldItem.count } : null;

  try {
    switch (primitive) {
      case 'hotbar_select': {
        const slot = Math.max(0, Math.min(8, srcSlot));
        bot.setQuickBarSlot(slot);
        return { success: true, reason: 'NONE', delta: { selected_slot: slot } };
      }

      case 'swap_slots': {
        if (srcSlot < 0 || srcSlot >= bot.inventory.slots.length ||
            destSlot < 0 || destSlot >= bot.inventory.slots.length) {
          return { success: false, reason: 'SLOT_INDEX_OUT_OF_BOUNDS', delta: {} };
        }
        await bot.moveSlotItem(srcSlot, destSlot);
        return { success: true, reason: 'NONE', delta: { swapped: [srcSlot, destSlot] } };
      }

      case 'drop_held': {
        if (!bot.heldItem) {
          return { success: false, reason: 'NO_HELD_ITEM', delta: {} };
        }
        await bot.toss(bot.heldItem.type, null, 1);
        return { success: true, reason: 'NONE', delta: { dropped: initialHeld } };
      }

      case 'drop_stack': {
        if (!bot.heldItem) {
          return { success: false, reason: 'NO_HELD_ITEM', delta: {} };
        }
        await bot.tossStack(bot.heldItem);
        return { success: true, reason: 'NONE', delta: { dropped_stack: initialHeld } };
      }

      case 'offhand_swap': {
        // In modern Minecraft, offhand is slot 45
        const offhandSlot = 45;
        const currentHeldSlot = bot.quickBarSlot + 36;
        await bot.moveSlotItem(currentHeldSlot, offhandSlot);
        return { success: true, reason: 'NONE', delta: { offhand_swapped: true } };
      }

      case 'equip': {
        const item = bot.inventory.slots[srcSlot];
        if (!item) {
          return { success: false, reason: 'SOURCE_SLOT_EMPTY', delta: {} };
        }
        const destinations = ['hand', 'head', 'torso', 'legs', 'feet', 'off-hand'];
        const dest = destinations[destSlot % destinations.length] || 'hand';
        await bot.equip(item, dest);
        return { success: true, reason: 'NONE', delta: { equipped_to: dest } };
      }

      case 'unequip': {
        const destinations = ['head', 'torso', 'legs', 'feet', 'off-hand'];
        const dest = destinations[destSlot % destinations.length] || 'head';
        await bot.unequip(dest);
        return { success: true, reason: 'NONE', delta: { unequipped_from: dest } };
      }

      default:
        return { success: false, reason: 'UNSUPPORTED_INVENTORY_PRIMITIVE', delta: {} };
    }
  } catch (err) {
    return { success: false, reason: err.message || 'INVENTORY_EXCEPTION', delta: {} };
  }
}

module.exports = { executeInventoryAction };
