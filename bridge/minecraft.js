// TLauncher & Minecraft Connection Helper
const dgram = require('dgram');

/**
 * Listens for Minecraft LAN broadcast packets on UDP 224.0.2.0:4445
 * to automatically discover open LAN worlds launched from TLauncher.
 */
function discoverLanWorld(timeoutMs = 3000) {
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
        resolve({ host: '127.0.0.1', port, motd });
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

module.exports = { discoverLanWorld };
