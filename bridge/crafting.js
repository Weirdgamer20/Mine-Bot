// Tier 5: Crafting operations for Mineflayer
async function executeCraftingAction(bot, cmd) {
  const primitive = cmd.primitive;
  let recipeTargetName = cmd.recipe_name;
  const count = Math.max(1, Math.min(64, Number(cmd.duration_ticks || 1)));

  if (!bot || !bot.recipesFor) {
    return { success: false, reason: 'CRAFTING_UNAVAILABLE', delta: {} };
  }

  // Find nearby crafting table (required for 3x3 recipes)
  const craftingTable = bot.findBlock({
    matching: bot.registry.blocksByName.crafting_table ? bot.registry.blocksByName.crafting_table.id : -1,
    maxDistance: 4.0,
  });

  // Auto-discover a craftable recipe when the agent hasn't specified one.
  // The agent currently selects the 'craft' primitive but cannot yet output
  // a recipe name — so we scan the bot's inventory for the first item that
  // has a valid recipe given the current materials.
  if (!recipeTargetName) {
    const allItems = Object.values(bot.registry.itemsByName);
    for (const item of allItems) {
      const recipes = bot.recipesFor(item.id, null, 1, craftingTable);
      if (recipes && recipes.length > 0) {
        recipeTargetName = item.name;
        console.log(`[Craft] Auto-selected craftable recipe: ${recipeTargetName}`);
        break;
      }
    }
    if (!recipeTargetName) {
      return { success: false, reason: 'NO_CRAFTABLE_RECIPE_WITH_CURRENT_MATERIALS', delta: {} };
    }
  }

  // Look up item in bot registry
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
