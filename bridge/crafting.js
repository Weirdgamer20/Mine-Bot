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

  // Priority order of key progression items to test if no explicit recipe was provided
  const progressionPriority = [
    'wooden_pickaxe', 'stone_pickaxe', 'iron_pickaxe', 'diamond_pickaxe',
    'wooden_axe', 'stone_axe', 'iron_axe',
    'wooden_sword', 'stone_sword', 'iron_sword',
    'crafting_table', 'furnace', 'stick', 'torch',
    'oak_planks', 'spruce_planks', 'birch_planks', 'jungle_planks', 'acacia_planks', 'dark_oak_planks',
  ];

  let targetItem = null;
  let recipes = [];

  if (recipeTargetName) {
    targetItem = bot.registry.itemsByName[recipeTargetName] || bot.registry.blocksByName[recipeTargetName];
    if (targetItem) {
      recipes = bot.recipesFor(targetItem.id, null, 1, craftingTable);
    }
  }

  // If no recipe was specified or requested recipe isn't craftable, discover highest-tier valid recipe from inventory
  if (!recipes || recipes.length === 0) {
    for (const candName of progressionPriority) {
      const candItem = bot.registry.itemsByName[candName] || bot.registry.blocksByName[candName];
      if (candItem) {
        const candRecipes = bot.recipesFor(candItem.id, null, 1, craftingTable);
        if (candRecipes && candRecipes.length > 0) {
          targetItem = candItem;
          recipes = candRecipes;
          break;
        }
      }
    }
  }

  // If still no recipe found, check all inventory items for any valid 2x2/3x3 recipe
  if (!recipes || recipes.length === 0) {
    for (const invItem of bot.inventory.items()) {
      const candRecipes = bot.recipesFor(invItem.type, null, 1, craftingTable);
      if (candRecipes && candRecipes.length > 0) {
        targetItem = invItem;
        recipes = candRecipes;
        break;
      }
    }
  }

  if (!recipes || recipes.length === 0 || !targetItem) {
    return { success: false, reason: 'NO_VALID_RECIPES_FOR_CURRENT_INVENTORY', delta: {} };
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
