const net = require('net');
const mineflayer = require('mineflayer');
const { MessageType, encodeFrame, StreamParser } = require('./protocol');
const { CanonicalBridgeRegistry } = require('./registry');
const { buildFullObservation } = require('./observation');
const { executeHierarchicalAction } = require('./actions');
const { discoverLanWorld } = require('./minecraft');

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

console.log('='.repeat(60));
console.log('MINECRAFT LEARNING BOT — ENVIRONMENT CONTRACT BRIDGE v1');
console.log('='.repeat(60));
console.log(`Minecraft Target: ${CONFIG.minecraft.host}:${CONFIG.minecraft.port}`);
console.log(`Persistent Stream Target: ${CONFIG.agentStream.host}:${CONFIG.agentStream.port}`);

let bot = null;
let streamSocket = null;
let streamParser = null;
let registry = null;
let sequenceId = 1;
let episodeId = 1;
let stepId = 0;
let isStepInProgress = false;
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
  sendFrame(MessageType.HELLO, { manifest });
}

async function handleAgentMessage(msg) {
  const { type, seqId, payload } = msg;

  if (type === MessageType.WELCOME) {
    console.log('[Stream] Handshake WELCOME received from WSL agent. Environment Contract verified.');
    handshakeComplete = true;
    startObservationLoop();
    return;
  }

  if (type === MessageType.ACTION) {
    if (bot && bot.isAlive) {
      lastActionResult = await executeHierarchicalAction(bot, payload.action);
    }
    isStepInProgress = false;
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
  bot = mineflayer.createBot(CONFIG.minecraft);

  bot.once('spawn', () => {
    console.log(`[Minecraft] Bot spawned into world as '${bot.username}' (MC Version: ${bot.version}).`);
    registry.initForBot(bot);

    if (streamSocket && !streamSocket.destroyed && !handshakeComplete) {
      performHandshake();
    }
  });

  bot.on('death', () => {
    console.log('[Minecraft] Bot died! Sending DEATH notification to agent (episode resets, intelligence persists).');
    if (handshakeComplete) {
      const obs = buildFullObservation(bot, registry, lastActionResult, episodeId, stepId);
      if (obs) {
        obs.done = true;
        sendFrame(MessageType.DEATH, { observation: obs });
      }
    }
    episodeId++;
    stepId = 0;
    lastActionResult = null;
  });

  bot.on('error', err => console.error('[Minecraft] Error:', err.message));
  bot.on('kicked', reason => console.warn('[Minecraft] Kicked from server:', reason));
  bot.on('end', () => {
    console.log('[Minecraft] Server disconnected. Reconnecting in 5 seconds...');
    handshakeComplete = false;
    setTimeout(initBot, 5000);
  });
}

let tickTimer = null;
function startObservationLoop() {
  if (tickTimer) clearInterval(tickTimer);

  tickTimer = setInterval(async () => {
    if (!handshakeComplete || isStepInProgress || !bot || !bot.isAlive || !bot.entity) {
      return;
    }

    isStepInProgress = true;
    stepId++;

    const obs = buildFullObservation(bot, registry, lastActionResult, episodeId, stepId);
    if (obs) {
      sendFrame(MessageType.OBSERVATION, { observation: obs });
    } else {
      isStepInProgress = false;
    }
  }, CONFIG.tickRateMs);
}

async function start() {
  if (!process.env.MC_PORT) {
    console.log('[Minecraft] Scanning for TLauncher LAN worlds on local network (2s timeout)...');
    const lan = await discoverLanWorld(2000);
    if (lan) {
      console.log(`[Minecraft] Discovered TLauncher LAN world: "${lan.motd}" on port ${lan.port}!`);
      CONFIG.minecraft.port = lan.port;
    } else {
      console.log(`[Minecraft] No LAN broadcast found; defaulting to port ${CONFIG.minecraft.port}.`);
    }
  }

  connectToAgentStream();
  initBot();
}

start();
