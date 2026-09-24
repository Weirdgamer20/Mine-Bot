// 16-byte binary framing protocol with CRC32 verification for Mineflayer bridge
const MAGIC = Buffer.from([0x4D, 0x42]); // 'MB'
const PROTOCOL_VERSION = 1;
const HEADER_SIZE = 16;

const MessageType = {
  HELLO: 1,
  WELCOME: 2,
  OBSERVATION: 3,
  ACTION: 4,
  ACTION_RESULT: 5,
  RESET: 6,
  DEATH: 7,
  PING: 8,
  PONG: 9,
  ERROR: 10,
  SHUTDOWN: 11,
};

// Fast CRC32 table generator (standard IEEE 802.3 polynomial 0xEDB88320)
const crcTable = new Uint32Array(256);
for (let i = 0; i < 256; i++) {
  let c = i;
  for (let j = 0; j < 8; j++) {
    c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
  }
  crcTable[i] = c >>> 0;
}

function computeCRC32(buf) {
  let crc = 0xFFFFFFFF;
  for (let i = 0; i < buf.length; i++) {
    crc = crcTable[(crc ^ buf[i]) & 0xFF] ^ (crc >>> 8);
  }
  return (crc ^ 0xFFFFFFFF) >>> 0;
}

function encodeFrame(msgType, seqId, payloadObj) {
  const payloadBytes = Buffer.from(JSON.stringify(payloadObj), 'utf-8');
  const length = payloadBytes.length;
  const crc = computeCRC32(payloadBytes);

  const header = Buffer.alloc(HEADER_SIZE);
  header[0] = 0x4D; // 'M'
  header[1] = 0x42; // 'B'
  header[2] = PROTOCOL_VERSION;
  header[3] = Number(msgType);
  header.writeUInt32BE(seqId >>> 0, 4);
  header.writeUInt32BE(length >>> 0, 8);
  header.writeUInt32BE(crc >>> 0, 12);

  return Buffer.concat([header, payloadBytes]);
}

class StreamParser {
  constructor(onMessage) {
    this.onMessage = onMessage;
    this.buffer = Buffer.alloc(0);
  }

  push(chunk) {
    this.buffer = Buffer.concat([this.buffer, chunk]);

    while (this.buffer.length >= HEADER_SIZE) {
      if (this.buffer[0] !== 0x4D || this.buffer[1] !== 0x42) {
        throw new Error(`Invalid protocol magic bytes: ${this.buffer[0]}, ${this.buffer[1]}`);
      }

      const version = this.buffer[2];
      if (version !== PROTOCOL_VERSION) {
        throw new Error(`Unsupported protocol version: ${version}`);
      }

      const msgType = this.buffer[3];
      const seqId = this.buffer.readUInt32BE(4);
      const payloadLength = this.buffer.readUInt32BE(8);
      const expectedCrc = this.buffer.readUInt32BE(12);

      if (this.buffer.length < HEADER_SIZE + payloadLength) {
        break; // Wait for full payload frame
      }

      const payloadBytes = this.buffer.slice(HEADER_SIZE, HEADER_SIZE + payloadLength);
      this.buffer = this.buffer.slice(HEADER_SIZE + payloadLength);

      const actualCrc = computeCRC32(payloadBytes);
      if (actualCrc !== expectedCrc) {
        throw new Error(`CRC32 mismatch on seq ${seqId}: expected ${expectedCrc}, got ${actualCrc}`);
      }

      const payload = JSON.parse(payloadBytes.toString('utf-8'));
      if (this.onMessage) {
        this.onMessage({ type: msgType, seqId, payload });
      }
    }
  }
}

module.exports = {
  MAGIC,
  PROTOCOL_VERSION,
  HEADER_SIZE,
  MessageType,
  computeCRC32,
  encodeFrame,
  StreamParser,
};
