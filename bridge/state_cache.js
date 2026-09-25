// Latest-value state cache for Mineflayer bot observations.
// Ensures fresh state delivery and tracking of world ticks.

class BridgeStateCache {
  constructor(agentId) {
    this.agentId = agentId;
    this.sequence = 0;
    this.lastWorldTick = -1;
    this.latestEnvelope = null;
  }

  createEnvelope(observation, bot) {
    this.sequence += 1;
    const nowMs = Date.now();
    const timestampNs = nowMs * 1000000;
    const worldTick = bot && bot.time ? Number(bot.time.age) : 0;

    const envelope = {
      sequence: this.sequence,
      world_tick: worldTick,
      timestamp_ns: timestampNs,
      timestamp: nowMs / 1000.0,
      agent_id: this.agentId,
      observation: observation,
    };

    this.latestEnvelope = envelope;
    this.lastWorldTick = worldTick;
    return envelope;
  }

  getLatest() {
    return this.latestEnvelope;
  }
}

module.exports = { BridgeStateCache };
