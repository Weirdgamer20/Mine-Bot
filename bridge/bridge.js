const net = require('net');
const mineflayer = require('mineflayer');
const { MessageType, encodeFrame, StreamParser } = require('./protocol');
const { CanonicalBridgeRegistry } = require('./registry');
const { buildFullObservation } = require('./observation');
const { applyContinuousControls } = require('./actions');
const { ActionExecutor } = require('./action_executor');
const { BridgeStateCache } = require('./state_cache');
const { discoverLanWorld, resolveMinecraftHost } = require('./minecraft');

const cliPort = process.argv[2] && !isNaN(Number(process.argv[2])) ? Number(process.argv[2]) : null;
const AGENT_ID = String(process.env.MC_AGENT_ID || 'LB-01').trim().toUpperCase();
const DEFAULT_MINECRAFT_NAMES = { 'LB-01': 'LB01', 'LB-02': 'LB02', 'LB-03': 'LB03', 'LB-04': 'LB04' };

const CONFIG = {
  agentId: AGENT_ID,
  minecraft: {
    host: resolveMinecraftHost(),
    port: cliPort || Number(process.env.MC_PORT || 25565),
    username: process.env.MC_USERNAME || DEFAULT_MINECRAFT_NAMES[AGENT_ID] || AGENT_ID.replace(/[^A-Za-z0-9_]/g, '').slice(0, 16),
    version: process.env.MC_VERSION || '1.20.4',
    auth: process.env.MC_AUTH || 'offline',
    viewDistance: Number(process.env.MC_VIEW_DISTANCE || 16),
  },
  agentStream: {
    host: process.env.AGENT_HOST || '127.0.0.1',
    port: Number(process.env.AGENT_PORT || 9099),
  },
  tickRateMs: Number(process.env.TICK_RATE_MS || 50), // 20 Hz observation rate
};

console.log('='.repeat(60));
console.log('MINECRAFT LEARNING BOT — ENVIRONMENT CONTRACT BRIDGE v1');
console.log('='.repeat(60));
console.log(`Agent ID: ${CONFIG.agentId}`);
console.log(`Minecraft Target: ${CONFIG.minecraft.host}:${CONFIG.minecraft.port}`);
console.log(`Persistent Stream Target: ${CONFIG.agentStream.host}:${CONFIG.agentStream.port}`);

let bot = null;
let streamSocket = null;
let streamParser = null;
let registry = null;
let stateCache = new BridgeStateCache(CONFIG.agentId);
let actionExecutor = null;
let sequenceId = 1;
let episodeId = 1;
let stepId = 0;
let lastActionResult = null;
let handshakeComplete = false;

// -------------------------------------------------------------
// 1. Initialize Canonical Registry
// -------------------------------------------------------------
try {
  registry = new CanonicalBridgeRegistry();
  console.log(`[Registry] Loaded Universal Dictionary (${Object.keys(registry.blockMap).length} blocks, ${Object.keys(registry.itemMap).length} items).`);
} catch (err) {
  console.error('[Registry] Failed to initialize canonical registry:', err.message);
  process.exit(1);
}

// -------------------------------------------------------------
// 2. Persistent TCP Socket Stream
// -------------------------------------------------------------
function sendFrame(msgType, payload) {
  if (!streamSocket || streamSocket.destroyed || !streamSocket.writable) {
    return false;
  }
  const frameBuf = encodeFrame(msgType, sequenceId++, payload);
  streamSocket.write(frameBuf);
  return true;
}

function handleActionResult(resultEnvelope) {
  lastActionResult = {
    action_primitive: resultEnvelope.action_primitive || 'noop',
    success: resultEnvelope.status === 'SUCCESS',
    failure_reason: resultEnvelope.failure_reason || 'NONE',
    elapsed_ticks: Math.max(1, (resultEnvelope.world_tick_end - resultEnvelope.world_tick_start) || 1),
    state_delta: resultEnvelope.state_delta || {},
  };
  sendFrame(MessageType.ACTION_RESULT, resultEnvelope);
}

function connectToAgentStream() {
  console.log(`[Stream] Connecting to WSL Learning Agent at ${CONFIG.agentStream.host}:${CONFIG.agentStream.port}...`);
  streamSocket = new net.Socket();
  streamSocket.setNoDelay(true);

  streamParser = new StreamParser(handleAgentMessage);

  streamSocket.connect(CONFIG.agentStream.port, CONFIG.agentStream.host, () => {
    console.log('[Stream] Persistent direct TCP connection established.');
    if (bot && bot.entity) {
      performHandshake();
    }
  });

  streamSocket.on('data', chunk => {
    try {
      streamParser.push(chunk);
    } catch (err) {
      console.error('[Stream] Protocol parsing error:', err.message);
    }
  });

  streamSocket.on('error', err => {
    console.warn(`[Stream] Socket error: ${err.message}`);
  });

  streamSocket.on('close', () => {
    handshakeComplete = false;
    console.log('[Stream] Connection closed. Reconnecting in 2 seconds...');
    setTimeout(connectToAgentStream, 2000);
  });
}

function performHandshake() {
  if (!bot) return;
  const manifest = registry.getVersionManifest(bot);
  console.log('[Stream] Sending HELLO handshake with Environment Manifest...');
  sendFrame(MessageType.HELLO, { agent_id: CONFIG.agentId, manifest });
}

async function handleAgentMessage(msg) {
  const { type, seqId, payload } = msg;

  if (type === MessageType.WELCOME) {
    console.log('[Stream] Handshake WELCOME received from WSL agent. Environment Contract verified.');
    console.log(`[Agent] ${CONFIG.agentId} registered as ${payload.personality || 'UNKNOWN'} peer.`);
    handshakeComplete = true;
    startObservationLoop();
    return;
  }

  if (type === MessageType.ACTION) {
    if (!bot || !bot.isAlive) return;

    const action = payload.action;
    if (!action) return;

    // 1. Continuous Controls: applied immediately (<0.1ms) without blocking
    if (action.motor) {
      applyContinuousControls(bot, action.motor);
    }

    // 2. Discrete Actions: queued for serialized executor with lifecycle tracking
    const cmd = action.command;
    if (cmd && cmd.primitive && cmd.primitive !== 'noop' && actionExecutor) {
      actionExecutor.enqueue({
        action_id: payload.action_id || seqId,
        agent_id: CONFIG.agentId,
        observation_seq: payload.observation_seq || 0,
        world_tick: payload.world_tick || (bot.time ? bot.time.age : 0),
        created_ns: payload.created_ns || Date.now() * 1000000,
        model_version: payload.model_version || 1,
        action: action,
      });
    }
    return;
  }

  if (type === MessageType.PONG) {
    // Heartbeat acknowledged
    return;
  }
}

// -------------------------------------------------------------
// 3. Minecraft Bot Initializer & Ticking Loop
// -------------------------------------------------------------
function initBot() {
  try {
    bot = mineflayer.createBot(CONFIG.minecraft);
    actionExecutor = new ActionExecutor(bot, handleActionResult);
  } catch (err) {
    console.error('[Minecraft] Failed to create bot:', err.message);
    setTimeout(initBot, 5000);
    return;
  }

  bot.once('spawn', () => {
    console.log(`[Minecraft] [${CONFIG.agentId}] Bot spawned as '${bot.username}' (MC Version: ${bot.version}).`);
    console.log(`[Minecraft] Render distance active: ${CONFIG.minecraft.viewDistance} chunks (${CONFIG.minecraft.viewDistance * 16} blocks radius).`);
    registry.initForBot(bot);

    if (streamSocket && !streamSocket.destroyed && !handshakeComplete) {
      performHandshake();
    }
  });

  bot.on('death', () => {
    console.log(`[Minecraft] [${CONFIG.agentId}] Bot died! Cancelling pending actions and sending DEATH notification.`);
    if (actionExecutor) {
      actionExecutor.cancelAll();
    }
    if (handshakeComplete) {
      try {
        const obs = buildFullObservation(bot, registry, lastActionResult, episodeId, stepId);
        if (obs) {
          obs.done = true;
          const envelope = stateCache.createEnvelope(obs, bot);
          sendFrame(MessageType.DEATH, envelope);
        }
      } catch (_) {}
    }
    episodeId++;
    stepId = 0;
    lastActionResult = null;
  });

  bot.on('error', err => console.error('[Minecraft] Error:', err.message));
  bot.on('kicked', reason => console.warn('[Minecraft] Kicked from server:', reason));
  bot.on('end', reason => {
    console.log(`[Minecraft] Server disconnected (reason: ${reason}). Reconnecting in 5 seconds...`);
    handshakeComplete = false;
    if (actionExecutor) {
      actionExecutor.cancelAll();
    }
    setTimeout(initBot, 5000);
  });
}

let tickTimer = null;
function startObservationLoop() {
  if (tickTimer) clearInterval(tickTimer);

  // Decoupled observation loop: ticks at simulation rate (20 Hz = 50ms)
  // NEVER blocked by action execution.
  tickTimer = setInterval(() => {
    if (!handshakeComplete || !bot || !bot.isAlive || !bot.entity) {
      return;
    }

    stepId++;
    try {
      const obs = buildFullObservation(bot, registry, lastActionResult, episodeId, stepId);
      if (obs) {
        const envelope = stateCache.createEnvelope(obs, bot);
        sendFrame(MessageType.OBSERVATION, envelope);
      }
    } catch (err) {
      console.error(`[Bridge] [${CONFIG.agentId}] Error building/sending observation:`, err.message);
    }
  }, CONFIG.tickRateMs);
}

async function start() {
  if (cliPort) {
    console.log(`[Minecraft] Using command-line specified port: ${cliPort}`);
  } else if (!process.env.MC_PORT) {
    console.log('[Minecraft] Scanning for TLauncher LAN worlds on local network (2s timeout)...');
    const lan = await discoverLanWorld(2000);
    if (lan) {
      console.log(`[Minecraft] Discovered TLauncher LAN world: "${lan.motd}" on port ${lan.port}!`);
      CONFIG.minecraft.port = lan.port;
    } else {
      console.log(`[Minecraft] No UDP LAN broadcast detected (WSL NAT boundary).`);
      console.log(`[Minecraft] Defaulting to ${CONFIG.minecraft.host}:${CONFIG.minecraft.port}`);
      console.log(`[Tip] If TLauncher opened to a LAN port (e.g. 54321), run: node bridge.js 54321`);
    }
  }

  connectToAgentStream();
  initBot();
}

start();
