// TLauncher & Minecraft Connection Helper
const dgram = require('dgram');
const fs = require('fs');
const os = require('os');
const { execSync } = require('child_process');

/**
 * Resolves the IP address to connect to Minecraft.
 * When running inside WSL2 while Minecraft is running on Windows (TLauncher),
 * connections to 127.0.0.1 stay inside WSL. To reach the Windows host,
 * we automatically detect the Windows default gateway IP.
 */
function resolveMinecraftHost() {
  if (process.env.MC_HOST && process.env.MC_HOST.trim().length > 0) {
    return process.env.MC_HOST.trim();
  }

  // Detect WSL2 environment
  let isWsl = false;
  try {
    if (os.release().toLowerCase().includes('microsoft')) {
      isWsl = true;
    } else if (fs.existsSync('/proc/version')) {
      const ver = fs.readFileSync('/proc/version', 'utf8').toLowerCase();
      if (ver.includes('microsoft') || ver.includes('wsl')) {
        isWsl = true;
      }
    }
  } catch {}

  if (isWsl) {
    try {
      const routeOutput = execSync('ip route show default', { encoding: 'utf8', timeout: 1000 });
      const match = routeOutput.match(/default via ([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)/);
      if (match && match[1]) {
        console.log(`[WSL Host Detection] Detected Windows host gateway IP: ${match[1]}`);
        return match[1];
      }
    } catch (e) {
      console.warn('[WSL Host Detection] Could not query ip route:', e.message);
    }
  }

  return '127.0.0.1';
}

/**
 * Listens for Minecraft LAN broadcast packets on UDP 224.0.2.0:4445
 * to automatically discover open LAN worlds launched from TLauncher.
 */
function discoverLanWorld(timeoutMs = 2000) {
  return new Promise(resolve => {
    const socket = dgram.createSocket({ type: 'udp4', reuseAddr: true });
    let resolved = false;

    const timer = setTimeout(() => {
      if (!resolved) {
        resolved = true;
        try { socket.close(); } catch {}
        resolve(null);
      }
    }, timeoutMs);

    socket.on('message', (msg) => {
      const text = msg.toString();
      // Minecraft LAN format: [MOTD]World Name[/MOTD][AD]Port[/AD]
      const portMatch = text.match(/\[AD\](\d+)\[\/AD\]/);
      const motdMatch = text.match(/\[MOTD\](.*?)\[\/MOTD\]/);
      if (portMatch && !resolved) {
        resolved = true;
        clearTimeout(timer);
        const port = Number(portMatch[1]);
        const motd = motdMatch ? motdMatch[1] : 'Minecraft LAN World';
        try { socket.close(); } catch {}
        resolve({ host: resolveMinecraftHost(), port, motd });
      }
    });

    socket.on('error', () => {
      if (!resolved) {
        resolved = true;
        clearTimeout(timer);
        try { socket.close(); } catch {}
        resolve(null);
      }
    });

    try {
      socket.bind(4445, () => {
        try {
          socket.addMembership('224.0.2.0');
        } catch {}
      });
    } catch {
      resolve(null);
    }
  });
}

module.exports = { resolveMinecraftHost, discoverLanWorld };
