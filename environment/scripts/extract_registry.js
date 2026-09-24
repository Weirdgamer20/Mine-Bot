const fs = require('fs');
const path = require('path');
const mcData = require('minecraft-data');

// Use 1.20.4 as reference version for canonical registry generation
const REF_VERSION = '1.20.4';
const data = mcData(REF_VERSION);

if (!data) {
  console.error(`Failed to load minecraft-data for version ${REF_VERSION}`);
  process.exit(1);
}

const outDir = path.join(__dirname, '..', 'registry');
fs.mkdirSync(outDir, { recursive: true });

console.log(`Extracting canonical registry from minecraft-data for version ${REF_VERSION}...`);

// 1. Blocks
const blocks = {};
for (const b of data.blocksArray) {
  blocks[b.name] = {
    name: b.name,
    displayName: b.displayName,
    hardness: b.hardness !== undefined ? b.hardness : 0.0,
    diggable: Boolean(b.diggable),
    transparent: Boolean(b.transparent),
    boundingBox: b.boundingBox || 'block',
    material: b.material || 'default',
    harvestTools: b.harvestTools ? Object.keys(b.harvestTools) : [],
  };
}
fs.writeFileSync(path.join(outDir, 'blocks.json'), JSON.stringify(blocks, null, 2));
console.log(`Extracted ${Object.keys(blocks).length} canonical blocks.`);

// 2. Items
const items = {};
for (const it of data.itemsArray) {
  items[it.name] = {
    name: it.name,
    displayName: it.displayName,
    stackSize: it.stackSize || 64,
    maxDurability: it.maxDurability || 0,
  };
}
fs.writeFileSync(path.join(outDir, 'items.json'), JSON.stringify(items, null, 2));
console.log(`Extracted ${Object.keys(items).length} canonical items.`);

// 3. Entities
const entities = {};
for (const e of data.entitiesArray) {
  let category = 'misc';
  if (e.type === 'hostile' || ['zombie', 'skeleton', 'creeper', 'spider', 'enderman', 'witch', 'slime', 'drowned'].includes(e.name)) {
    category = 'hostile';
  } else if (e.type === 'passive' || e.type === 'animal') {
    category = 'passive';
  } else if (e.type === 'player') {
    category = 'player';
  } else if (e.type === 'projectile') {
    category = 'projectile';
  }

  entities[e.name] = {
    name: e.name,
    displayName: e.displayName,
    type: e.type || 'mob',
    category: category,
    height: e.height || 1.8,
    width: e.width || 0.6,
  };
}
fs.writeFileSync(path.join(outDir, 'entities.json'), JSON.stringify(entities, null, 2));
console.log(`Extracted ${Object.keys(entities).length} canonical entities.`);

// 4. Biomes
const biomes = {};
for (const bio of (data.biomesArray || [])) {
  biomes[bio.name] = {
    name: bio.name,
    displayName: bio.displayName,
    temperature: bio.temperature || 0.5,
    rainfall: bio.rainfall || 0.5,
    category: bio.category || 'none',
  };
}
fs.writeFileSync(path.join(outDir, 'biomes.json'), JSON.stringify(biomes, null, 2));
console.log(`Extracted ${Object.keys(biomes).length} canonical biomes.`);

// 5. Recipes
const recipes = [];
for (const [resultId, recipeList] of Object.entries(data.recipes)) {
  const resultItem = data.items[resultId];
  if (!resultItem) continue;

  for (const r of recipeList) {
    const ingredients = [];
    if (r.ingredients) {
      for (const ing of r.ingredients) {
        if (ing === null) continue;
        const ingId = typeof ing === 'object' ? ing.id : ing;
        const itemObj = data.items[ingId];
        if (itemObj) ingredients.push(itemObj.name);
      }
    } else if (r.inShape) {
      for (const row of r.inShape) {
        for (const ing of row) {
          if (ing === null) continue;
          const ingId = typeof ing === 'object' ? ing.id : ing;
          const itemObj = data.items[ingId];
          if (itemObj) ingredients.push(itemObj.name);
        }
      }
    }

    recipes.push({
      result: resultItem.name,
      resultCount: r.result ? (r.result.count || 1) : 1,
      requiresCraftingTable: Boolean(r.inShape && (r.inShape.length > 2 || (r.inShape[0] && r.inShape[0].length > 2))),
      ingredients: ingredients,
    });
  }
}
fs.writeFileSync(path.join(outDir, 'recipes.json'), JSON.stringify(recipes, null, 2));
console.log(`Extracted ${recipes.length} canonical recipes.`);

// 6. Build Universal Dictionary (stable integer mapping for embeddings)
// Index 0 is always <UNKNOWN>/<PAD>
const universalDict = {
  vocab_version: 1,
  reference_version: REF_VERSION,
  namespaces: {
    blocks: { '<UNKNOWN>': 0 },
    items: { '<UNKNOWN>': 0 },
    entities: { '<UNKNOWN>': 0 },
    biomes: { '<UNKNOWN>': 0 },
  }
};

let bIdx = 1;
for (const name of Object.keys(blocks).sort()) {
  universalDict.namespaces.blocks[name] = bIdx++;
}

let iIdx = 1;
for (const name of Object.keys(items).sort()) {
  universalDict.namespaces.items[name] = iIdx++;
}

let eIdx = 1;
for (const name of Object.keys(entities).sort()) {
  universalDict.namespaces.entities[name] = eIdx++;
}

let bioIdx = 1;
for (const name of Object.keys(biomes).sort()) {
  universalDict.namespaces.biomes[name] = bioIdx++;
}

fs.writeFileSync(path.join(outDir, 'universal_dictionary.json'), JSON.stringify(universalDict, null, 2));
console.log(`Universal dictionary built: ${bIdx} blocks, ${iIdx} items, ${eIdx} entities, ${bioIdx} biomes.`);
console.log('Registry extraction completed successfully.');
