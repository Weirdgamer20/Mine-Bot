// Tier 5: Crafting operations for Mineflayer
async function executeCraftingAction(bot, cmd) {
  const primitive = cmd.primitive;
  const recipeTargetName = cmd.recipe_name;
  const count = Math.max(1, Math.min(64, Number(cmd.duration_ticks || 1)));

  if (!bot || !bot.recipesFor) {
    return { success: false, reason: 'CRAFTING_UNAVAILABLE', delta: {} };
  }

  if (!recipeTargetName) {
    return { success: false, reason: 'NO_RECIPE_NAME_SPECIFIED', delta: {} };
  }

  // Look up item in bot registry
  const item = bot.registry.itemsByName[recipeTargetName] || bot.registry.blocksByName[recipeTargetName];
  if (!item) {
    return { success: false, reason: 'UNKNOWN_RECIPE_TARGET', delta: {} };
  }

  // Find nearby crafting table if needed
  const craftingTable = bot.findBlock({
    matching: bot.registry.blocksByName.crafting_table ? bot.registry.blocksByName.crafting_table.id : -1,
    maxDistance: 4.0,
  });

  const recipes = bot.recipesFor(item.id, null, 1, craftingTable);
  if (!recipes || recipes.length === 0) {
    return { success: false, reason: 'NO_VALID_RECIPE_OR_MATERIALS', delta: {} };
  }

  const recipe = recipes[0];
  try {
    const craftCount = primitive === 'craft_batch' ? Math.min(count, 16) : 1;
    await bot.craft(recipe, craftCount, craftingTable);
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
