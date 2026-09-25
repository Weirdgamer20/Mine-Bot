// Tier 5: Pure learned crafting operations for Mineflayer
// Executes exact requested recipe parameter without hardcoded heuristic guessing.

async function executeCraftingAction(bot, cmd) {
  const primitive = cmd.primitive;
  const count = Math.max(1, Math.min(64, Number(cmd.duration_ticks || 1)));

  if (!bot || !bot.recipesFor) {
    return { success: false, reason: 'CRAFTING_UNAVAILABLE', delta: {} };
  }

  // Find nearby crafting table (required for 3x3 recipes)
  const craftingTable = bot.findBlock({
    matching: bot.registry.blocksByName.crafting_table ? bot.registry.blocksByName.crafting_table.id : -1,
    maxDistance: 4.0,
  });

  let recipeTargetName = cmd.recipe_name;
  if (!recipeTargetName && cmd.target_slot != null) {
    // If agent specified a hotbar or inventory slot as the target for crafting
    const slotItem = bot.inventory.slots[cmd.target_slot];
    if (slotItem) {
      recipeTargetName = slotItem.name;
    }
  }

  if (!recipeTargetName) {
    return { success: false, reason: 'NO_RECIPE_SPECIFIED', delta: {} };
  }

  const item = bot.registry.itemsByName[recipeTargetName] || bot.registry.blocksByName[recipeTargetName];
  if (!item) {
    return { success: false, reason: 'UNKNOWN_RECIPE_TARGET', delta: {} };
  }

  const recipes = bot.recipesFor(item.id, null, 1, craftingTable);
  if (!recipes || recipes.length === 0) {
    return { success: false, reason: 'NO_VALID_RECIPE_OR_MATERIALS', delta: {} };
  }

  const recipe = recipes[0];
  try {
    const craftCount = primitive === 'craft_batch' ? Math.min(count, 16) : 1;
    const craftPromise = bot.craft(recipe, craftCount, craftingTable);
    const craftTimeout = new Promise((_, reject) =>
      setTimeout(() => reject(new Error('CRAFT_TIMEOUT')), 1500)
    );
    await Promise.race([craftPromise, craftTimeout]);
    return {
      success: true,
      reason: 'NONE',
      delta: { crafted: item.name, count: craftCount * (recipe.result ? (recipe.result.count || 1) : 1) },
    };
  } catch (err) {
    return { success: false, reason: err.message || 'CRAFT_EXECUTION_FAILED', delta: {} };
  }
}

module.exports = { executeCraftingAction };
