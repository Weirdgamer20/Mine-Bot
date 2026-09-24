// Pure mechanical observation extractor from Mineflayer state
function buildFullObservation(bot, registry, lastActionResult, episodeId, stepId) {
  if (!bot || !bot.entity || !bot.entity.position) {
    return null;
  }

  const p = bot.entity.position;

  // 1. 11x11x11 Local Terrain Voxel Grid (1331 canonical IDs)
  const r = 5;
  const voxels = [];
  for (let dy = -r; dy <= r; dy++) {
    for (let dz = -r; dz <= r; dz++) {
      for (let dx = -r; dx <= r; dx++) {
        const b = bot.blockAt(p.offset(dx, dy, dz));
        voxels.push(registry.getBlockCanonicalId(b));
      }
    }
  }

  // 2. Kinematic Player State Vector & Directional / Environmental Awareness (28 Features)
  const yaw = Number(bot.entity.yaw || 0.0);
  const pitch = Number(bot.entity.pitch || 0.0);
  const sinYaw = -Math.sin(yaw);
  const cosYaw = Math.cos(yaw);
  const sinPitch = Math.sin(pitch);
  const cosPitch = Math.cos(pitch);

  // 3D Look unit vector (Direction Awareness)
  const lookX = sinYaw * cosPitch;
  const lookY = sinPitch;
  const lookZ = cosYaw * cosPitch;

  // Day / Night cycle (Environmental Awareness)
  const timeOfDay = bot.time ? Number(bot.time.timeOfDay % 24000) / 24000.0 : 0.0;
  const isDay = bot.time ? (bot.time.isDay ? 1.0 : 0.0) : 1.0;

  // Cliff & Fall Hazard Ahead of Feet
  let dropDepth = 0;
  if (bot.entity && bot.entity.onGround) {
    const frontX = sinYaw * 0.9;
    const frontZ = cosYaw * 0.9;
    for (let dy = 1; dy <= 8; dy++) {
      const b = bot.blockAt(p.offset(frontX, -dy, frontZ));
      if (!b || b.boundingBox !== 'block') {
        dropDepth++;
      } else {
        break;
      }
    }
  }

  // Threat awareness: proximity to nearest hostile mob
  const hostileMobs = Object.values(bot.entities || {})
    .filter(e => e !== bot.entity && e.position && (e.type === 'mob' || ['zombie', 'skeleton', 'creeper', 'spider', 'witch', 'slime'].some(m => (e.name || '').includes(m))))
    .map(e => p.distanceTo(e.position))
    .sort((a, b) => a - b);
  const threatDist = hostileMobs.length > 0 ? Math.min(32.0, hostileMobs[0]) / 32.0 : 1.0;

  const player_state = [
    Number(bot.health || 0.0),          // 0: Health (0..20)
    Number(bot.food || 0.0),            // 1: Food (0..20)
    Number(bot.foodSaturation || 0.0),  // 2: Food Saturation
    Number(bot.oxygenLevel || 20.0),    // 3: Oxygen Level
    Number(p.x || 0.0),                 // 4: X
    Number(p.y || 0.0),                 // 5: Y
    Number(p.z || 0.0),                 // 6: Z
    Number(bot.entity.velocity?.x || 0.0), // 7: Vx
    Number(bot.entity.velocity?.y || 0.0), // 8: Vy
    Number(bot.entity.velocity?.z || 0.0), // 9: Vz
    pitch,                              // 10: Pitch radians
    yaw,                                // 11: Yaw radians
    sinYaw,                             // 12: Cardinal East/West direction [-1, 1]
    cosYaw,                             // 13: Cardinal North/South direction [-1, 1]
    sinPitch,                           // 14: Vertical gaze elevation [-1, 1]
    cosPitch,                           // 15: Horizontal gaze magnitude [0, 1]
    lookX,                              // 16: Look vector X
    lookY,                              // 17: Look vector Y
    lookZ,                              // 18: Look vector Z
    bot.entity.onGround ? 1.0 : 0.0,    // 19: onGround
    bot.controlState?.sneak ? 1.0 : 0.0,// 20: isSneaking
    bot.controlState?.sprint ? 1.0 : 0.0,// 21: isSprinting
    bot.entity.isInWater ? 1.0 : 0.0,   // 22: isInWater
    bot.entity.isInLava ? 1.0 : 0.0,    // 23: isInLava
    timeOfDay,                          // 24: Day cycle [0.0, 1.0]
    isDay,                              // 25: 1.0 = Day, 0.0 = Night
    dropDepth / 8.0,                    // 26: Cliff drop depth ahead [0.0, 1.0]
    threatDist,                         // 27: Hostile threat proximity [0.0 = close, 1.0 = safe]
  ];

  // 3. Complete Inventory & Equipment (36 slots = 0..8 Hotbar + 9..35 Main Inventory)
  const slots = [];
  const rawSlots = bot.inventory ? bot.inventory.slots : [];
  for (let i = 0; i < 36; i++) {
    // Logical slot 0..8 -> physical Hotbar (slots 36..44)
    // Logical slot 9..35 -> physical Main Inventory (slots 9..35)
    const physSlot = (i < 9) ? (36 + i) : i;
    const item = rawSlots[physSlot];
    slots.push({
      slot_index: i,
      item_canonical_id: registry.getItemCanonicalId(item),
      count: item ? Number(item.count || 0) : 0,
      durability: item?.durabilityUsed ? Number(item.durabilityUsed) : 0.0,
    });
  }

  const getEquipment = (slotIdx) => {
    const item = rawSlots[slotIdx];
    return {
      slot_index: slotIdx,
      item_canonical_id: registry.getItemCanonicalId(item),
      count: item ? Number(item.count || 0) : 0,
      durability: item?.durabilityUsed ? Number(item.durabilityUsed) : 0.0,
    };
  };

  const inventory = {
    slots: slots,
    armor_head: getEquipment(5),
    armor_chest: getEquipment(6),
    armor_legs: getEquipment(7),
    armor_feet: getEquipment(8),
    offhand: getEquipment(45),
    selected_hotbar_slot: Number(bot.quickBarSlot || 0),
  };

  // 4. Perceived Nearby Entities (top 16 closest within 32 blocks)
  const entities = Object.values(bot.entities || {})
    .filter(e => e !== bot.entity && e.position)
    .map(e => {
      const dist = p.distanceTo(e.position);
      return {
        entity_id: Number(e.id || 0),
        canonical_type_id: registry.getEntityCanonicalId(e),
        category: e.type || 'mob',
        dx: Number(e.position.x - p.x),
        dy: Number(e.position.y - p.y),
        dz: Number(e.position.z - p.z),
        vx: Number(e.velocity?.x || 0.0),
        vy: Number(e.velocity?.y || 0.0),
        vz: Number(e.velocity?.z || 0.0),
        health: Number(e.health || 0.0),
        distance: Number(dist),
        is_alive: Boolean(e.isValid),
      };
    })
    .sort((a, b) => a.distance - b.distance)
    .slice(0, 16);

  // 5. Mechanical Affordances (Raycasts & Local Facts)
  const targetBlock = bot.blockAtCursor ? bot.blockAtCursor(4.5) : null;
  const currentBlock = bot.blockAt ? bot.blockAt(p) : null;
  const nearestTargetEntity = entities.length > 0 && entities[0].distance <= 3.5 ? 0 : -1;

  const affordances = {
    targeted_block_canonical_id: registry.getBlockCanonicalId(targetBlock),
    targeted_block_distance: targetBlock ? p.distanceTo(targetBlock.position) : 0.0,
    targeted_block_face: 1,
    can_mine_target: Boolean(targetBlock && bot.canDigBlock && bot.canDigBlock(targetBlock)),
    targeted_entity_idx: nearestTargetEntity,
    light_level: currentBlock ? Number(currentBlock.light || 0) : 0,
    sky_light: currentBlock ? Number(currentBlock.skyLight || 0) : 0,
    open_container_type: bot.currentWindow ? bot.currentWindow.type || 'container' : 'none',
  };

  // 6. Action Validity Mask (Mechanical Feasibility)
  const validity_mask = {
    can_jump: Boolean(bot.entity.onGround || bot.entity.isInWater),
    can_sprint: Boolean(bot.food > 6),
    can_sneak: true,
    can_attack_entity: nearestTargetEntity >= 0,
    can_dig_block: affordances.can_mine_target,
    can_place_block: Boolean(targetBlock && bot.heldItem),
    can_use_item: Boolean(bot.heldItem),
    can_open_container: Boolean(bot.currentWindow !== null),
    can_sleep: Boolean(currentBlock && currentBlock.name && currentBlock.name.includes('bed')),
  };

  return {
    voxels,
    voxel_shape: [11, 11, 11],
    player_state,
    inventory,
    entities,
    affordances,
    last_action_result: lastActionResult || {
      action_primitive: 'noop',
      success: true,
      failure_reason: 'NONE',
      elapsed_ticks: 1,
      state_delta: {},
    },
    validity_mask,
    done: !bot.isAlive,
    episode_id: episodeId,
    step_id: stepId,
    timestamp: Date.now() / 1000.0,
    info: {
      dimension: bot.game?.dimension || 'overworld',
      time_of_day: bot.time?.timeOfDay ? bot.time.timeOfDay / 24000.0 : 0.0,
    },
  };
}

module.exports = { buildFullObservation };
