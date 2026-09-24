const fs = require('fs');
const path = require('path');
const mcData = require('minecraft-data');

class CanonicalBridgeRegistry {
  constructor(registryDir) {
    const baseDir = registryDir || path.join(__dirname, '..', 'environment', 'registry');
    const dictPath = path.join(baseDir, 'universal_dictionary.json');

    if (!fs.existsSync(dictPath)) {
      throw new Error(`Universal dictionary not found at ${dictPath}. Run extract_registry.js first.`);
    }

    this.dict = JSON.parse(fs.readFileSync(dictPath, 'utf-8'));
    this.blockMap = this.dict.namespaces.blocks;
    this.itemMap = this.dict.namespaces.items;
    this.entityMap = this.dict.namespaces.entities;
    this.biomeMap = this.dict.namespaces.biomes;
    this.mcDataInstance = null;
  }

  initForBot(bot) {
    const version = bot.version || '1.20.4';
    try {
      this.mcDataInstance = mcData(version);
    } catch {
      this.mcDataInstance = mcData('1.20.4');
    }
  }

  // Maps a Mineflayer block to stable canonical ID
  getBlockCanonicalId(block) {
    if (!block || !block.name) return 0;
    const name = block.name.replace(/^minecraft:/, '');
    return this.blockMap[name] !== undefined ? this.blockMap[name] : 0;
  }

  // Maps an item to stable canonical ID
  getItemCanonicalId(item) {
    if (!item || !item.name) return 0;
    const name = item.name.replace(/^minecraft:/, '');
    return this.itemMap[name] !== undefined ? this.itemMap[name] : 0;
  }

  // Maps an entity to stable canonical ID
  getEntityCanonicalId(entity) {
    if (!entity || !entity.name) return 0;
    const name = entity.name.replace(/^minecraft:/, '');
    return this.entityMap[name] !== undefined ? this.entityMap[name] : 0;
  }

  // Maps biome ID/name to stable canonical ID
  getBiomeCanonicalId(biomeId) {
    if (!this.mcDataInstance || biomeId === undefined) return 0;
    const biomeObj = this.mcDataInstance.biomes ? this.mcDataInstance.biomes[biomeId] : null;
    if (!biomeObj || !biomeObj.name) return 0;
    const name = biomeObj.name.replace(/^minecraft:/, '');
    return this.biomeMap[name] !== undefined ? this.biomeMap[name] : 0;
  }

  getVersionManifest(bot) {
    return {
      minecraft_version: bot.version || '1.20.4',
      protocol_version: bot.protocolVersion || 765,
      mineflayer_version: '4.25.0',
      minecraft_data_version: '3.78.0',
      environment_schema_version: 1,
      registry_vocab_version: this.dict.vocab_version || 1,
      timestamp: Date.now() / 1000.0,
    };
  }
}

module.exports = { CanonicalBridgeRegistry };
