const net = require('net');
const mineflayer = require('mineflayer');

const CONFIG = {
  minecraft: {
    host: process.env.MC_HOST || '127.0.0.1',
    port: Number(process.env.MC_PORT || 25565),
    username: process.env.MC_USERNAME || 'LearningAgent',
    version: process.env.MC_VERSION || false,
    auth: process.env.MC_AUTH || 'offline',
  },
  agentStream: {
    host: process.env.AGENT_HOST || '127.0.0.1',
    port: Number(process.env.AGENT_PORT || 9099),
  },
  tickRateMs: Number(process.env.TICK_RATE_MS || 100),
};

console.log('[Bridge] Starting Mineflayer Direct-Stream Bridge...');
console.log(`[Bridge] Target Minecraft: ${CONFIG.minecraft.host}:${CONFIG.minecraft.port}`);
console.log(`[Bridge] Target Agent Stream: ${CONFIG.agentStream.host}:${CONFIG.agentStream.port}`);

// Persistent TCP Stream Client
class StreamClient {
  constructor(host, port, onMessage) {
    this.host = host;
    this.port = port;
    this.onMessage = onMessage;
    this.socket = null;
    this.buffer = Buffer.alloc(0);
    this.connected = false;
    this.pendingResolvers = [];
  }

  connect() {
    this.socket = new net.Socket();
    this.socket.setNoDelay(true);

    this.socket.connect(this.port, this.host, () => {
      console.log(`[Stream] Connected persistent TCP stream to WSL agent at ${this.host}:${this.port}`);
      this.connected = true;
    });

    this.socket.on('data', chunk => {
      this.buffer = Buffer.concat([this.buffer, chunk]);
      while (this.buffer.length >= 4) {
        const payloadLength = this.buffer.readUInt32BE(0);
        if (this.buffer.length < 4 + payloadLength) {
          break; // wait for full frame
        }
        const payload = this.buffer.slice(4, 4 + payloadLength);
        this.buffer = this.buffer.slice(4 + payloadLength);
        try {
          const msg = JSON.parse(payload.toString('utf-8'));
          if (this.pendingResolvers.length > 0) {
            const resolver = this.pendingResolvers.shift();
            resolver(msg);
          } else if (this.onMessage) {
            this.onMessage(msg);
          }
        } catch (err) {
          console.error('[Stream] Frame JSON parse error:', err.message);
        }
      }
    });

    this.socket.on('error', err => {
      console.error(`[Stream] TCP error: ${err.message}`);
    });

    this.socket.on('close', () => {
      this.connected = false;
      console.log('[Stream] TCP stream disconnected. Reconnecting in 2 seconds...');
      setTimeout(() => this.connect(), 2000);
    });
  }

  send(msg) {
    if (!this.connected || !this.socket) {
      return Promise.reject(new Error('Stream not connected'));
    }
    const payload = Buffer.from(JSON.stringify(msg), 'utf-8');
    const header = Buffer.alloc(4);
    header.writeUInt32BE(payload.length, 0);

    return new Promise(resolve => {
      this.pendingResolvers.push(resolve);
      this.socket.write(Buffer.concat([header, payload]));
    });
  }
}

// -------------------------------------------------------------
// Minecraft Bot Initializer & Mechanical Observation Collector
// -------------------------------------------------------------
let bot = null;
let streamClient = null;
let isStepActive = false;

function initBot() {
  bot = mineflayer.createBot(CONFIG.minecraft);

  bot.once('spawn', () => {
    console.log('[Minecraft] Bot spawned into world. Starting real-time stream loop.');
    startStreamLoop();
  });

  bot.on('death', () => {
    console.log('[Minecraft] Bot died. Notifying learning agent of episode termination.');
    if (streamClient && streamClient.connected) {
      streamClient.send({
        type: 'OBSERVE',
        observation: { ...buildObservation(), done: true },
      }).catch(() => {});
    }
  });

  bot.on('error', err => console.error('[Minecraft] Error:', err.message));
  bot.on('kicked', reason => console.warn('[Minecraft] Kicked:', reason));
  bot.on('end', () => {
    console.log('[Minecraft] Disconnected. Reconnecting in 5 seconds...');
    setTimeout(initBot, 5000);
  });
}

function numericId(block) {
  return block ? Number(block.type || 0) : 0;
}

function buildObservation() {
  if (!bot || !bot.entity || !bot.entity.position) {
    return { voxels: [], player_state: [], inventory: [], entities: [], done: false };
  }

  const p = bot.entity.position;

  // 1. 11x11x11 Local Voxel Grid (radius 5)
  const r = 5;
  const voxels = [];
  for (let dy = -r; dy <= r; dy++) {
    for (let dz = -r; dz <= r; dz++) {
      for (let dx = -r; dx <= r; dx++) {
        const b = bot.blockAt(p.offset(dx, dy, dz));
        voxels.push(numericId(b));
      }
    }
  }

  // 2. Physical Player State Vector
  const player_state = [
    bot.health || 0.0,
    bot.food || 0.0,
    bot.foodSaturation || 0.0,
    bot.oxygenLevel || 20.0,
    p.x, p.y, p.z,
    bot.entity.velocity?.x || 0.0,
    bot.entity.velocity?.y || 0.0,
    bot.entity.velocity?.z || 0.0,
    bot.entity.pitch || 0.0,
    bot.entity.yaw || 0.0,
    bot.entity.onGround ? 1.0 : 0.0,
    bot.controlState?.sneak ? 1.0 : 0.0,
    bot.controlState?.sprint ? 1.0 : 0.0,
    bot.entity.isInWater ? 1.0 : 0.0,
    bot.isAlive ? 1.0 : 0.0,
  ];

  // 3. Inventory Slots (36 regular inventory slots)
  const inventory = [];
  const slots = bot.inventory ? bot.inventory.slots : [];
  for (let i = 0; i < 36; i++) {
    const item = slots[i];
    inventory.push({
      slot_index: i,
      item_id: item ? Number(item.type || 0) : 0,
      count: item ? Number(item.count || 0) : 0,
      durability: item?.durabilityUsed ? Number(item.durabilityUsed) : 0.0,
    });
  }

  // 4. Nearby Entities (top 16 nearest)
  const entities = Object.values(bot.entities || {})
    .filter(e => e !== bot.entity && e.position)
    .sort((a, b) => p.distanceSquared(a.position) - p.distanceSquared(b.position))
    .slice(0, 16)
    .map(e => ({
      entity_id: Number(e.id || 0),
      type_id: Number(e.entityType || 0),
      dx: e.position.x - p.x,
      dy: e.position.y - p.y,
      dz: e.position.z - p.z,
      vx: e.velocity?.x || 0.0,
      vy: e.velocity?.y || 0.0,
      vz: e.velocity?.z || 0.0,
      health: Number(e.health || 0.0),
      is_alive: e.isValid || true,
    }));

  // 5. Mechanical Affordances (pure game facts, no strategy)
  const targetBlock = bot.blockAtCursor ? bot.blockAtCursor(4.5) : null;
  const currentBlock = bot.blockAt ? bot.blockAt(p) : null;
  const affordances = {
    target_block_id: numericId(targetBlock),
    target_block_distance: targetBlock ? p.distanceTo(targetBlock.position) : 0.0,
    can_mine: targetBlock ? Boolean(bot.canDigBlock && bot.canDigBlock(targetBlock)) : false,
    light_level: currentBlock ? Number(currentBlock.light || 0) : 0,
    time_of_day: bot.time?.timeOfDay ? bot.time.timeOfDay / 24000.0 : 0.0,
    is_raining: Boolean(bot.isRaining),
    is_sleeping: Boolean(bot.isSleeping),
    is_swimming: Boolean(bot.entity?.isInWater),
    equipped_item_id: bot.heldItem ? Number(bot.heldItem.type || 0) : 0,
  };

  return {
    voxels,
    voxel_shape: [11, 11, 11],
    player_state,
    inventory,
    entities,
    affordances,
    done: !bot.isAlive,
  };
}

// -------------------------------------------------------------
// Action Actuators (Mechanical execution of neural commands)
// -------------------------------------------------------------
async function executeAction(action) {
  if (!bot || !bot.entity || !action) return;

  // 1. Controls
  bot.setControlState('forward', action.move_z > 0.2);
  bot.setControlState('back', action.move_z < -0.2);
  bot.setControlState('left', action.move_x < -0.2);
  bot.setControlState('right', action.move_x > 0.2);
  bot.setControlState('jump', action.jump > 0.5);
  bot.setControlState('sneak', action.sneak > 0.5);
  bot.setControlState('sprint', action.sprint > 0.5);

  // 2. View Angles (Yaw & Pitch)
  if (Number.isFinite(action.yaw_delta) && Number.isFinite(action.pitch_delta)) {
    const newYaw = bot.entity.yaw + action.yaw_delta * 0.15;
    const newPitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, bot.entity.pitch + action.pitch_delta * 0.1));
    await bot.look(newYaw, newPitch, true).catch(() => {});
  }

  // 3. Hotbar selection
  if (action.slot >= 0 && action.slot < 9) {
    bot.setQuickBarSlot(action.slot);
  }

  // 4. Attack / Mine
  if (action.attack > 0.5) {
    const targetEntity = bot.nearestEntity(e => e !== bot.entity && bot.entity.position.distanceTo(e.position) <= 3.5);
    if (targetEntity) {
      bot.attack(targetEntity);
    } else {
      const targetBlock = bot.blockAtCursor ? bot.blockAtCursor(4.0) : null;
      if (targetBlock && bot.canDigBlock && bot.canDigBlock(targetBlock)) {
        bot.dig(targetBlock).catch(() => {});
      } else {
        bot.swingArm('right');
      }
    }
  }

  // 5. Use / Place
  if (action.use > 0.5) {
    const targetBlock = bot.blockAtCursor ? bot.blockAtCursor(4.0) : null;
    if (targetBlock) {
      bot.placeBlock(targetBlock, { x: 0, y: 1, z: 0 }).catch(() => {
        bot.activateItem();
      });
    } else {
      try { bot.activateItem(); } catch {}
    }
  }

  // 6. Drop item
  if (action.drop > 0.5 && bot.heldItem) {
    bot.tossStack(bot.heldItem).catch(() => {});
  }
}

// -------------------------------------------------------------
// Real-time Persistent Streaming Loop
// -------------------------------------------------------------
async function stepOnce() {
  if (isStepActive || !bot || !bot.entity || !streamClient || !streamClient.connected) {
    return;
  }
  isStepActive = true;

  try {
    const obs = buildObservation();
    const response = await streamClient.send({
      type: 'OBSERVE',
      observation: obs,
    });

    if (response && response.type === 'ACTION') {
      await executeAction(response.action);
    }
  } catch (err) {
    // Expected during reconnection/transitions
  } finally {
    isStepActive = false;
  }
}

function startStreamLoop() {
  setInterval(stepOnce, CONFIG.tickRateMs);
}

// Initialize persistent stream connection and bot
streamClient = new StreamClient(CONFIG.agentStream.host, CONFIG.agentStream.port);
streamClient.connect();
initBot();
