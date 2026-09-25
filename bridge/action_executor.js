// Single serialized FIFO discrete action executor per bot peer.
// Decouples slow Minecraft operations (dig, place, attack, craft, container)
// from the 100 Hz real-time observation and motor control stream.

class ActionExecutor {
  constructor(bot, onActionResult) {
    this.bot = bot;
    this.onActionResult = onActionResult;
    this.queue = [];
    this.current = null;
    this.isExecuting = false;
    this.cancelledIds = new Set();
  }

  enqueue(actionEnvelope) {
    if (!actionEnvelope || !actionEnvelope.action_id) return;
    this.queue.push(actionEnvelope);
    this.processQueue();
  }

  cancel(actionId) {
    if (actionId) {
      this.cancelledIds.add(actionId);
    }
    if (this.current && this.current.action_id === actionId) {
      this.abortCurrentAction('CANCELLED');
    }
  }

  cancelAll() {
    this.queue = [];
    this.cancelledIds.clear();
    if (this.current) {
      this.abortCurrentAction('CANCELLED');
    }
  }

  abortCurrentAction(status = 'CANCELLED') {
    if (this.bot) {
      if (this.bot.targetDigBlock) {
        try { this.bot.stopDigging(); } catch (_) {}
      }
      if (typeof this.bot.deactivateItem === 'function') {
        try { this.bot.deactivateItem(); } catch (_) {}
      }
    }
  }

  async processQueue() {
    if (this.isExecuting || this.queue.length === 0) {
      return;
    }

    this.isExecuting = true;

    while (this.queue.length > 0) {
      const item = this.queue.shift();
      const actionId = item.action_id;

      if (this.cancelledIds.has(actionId)) {
        this.cancelledIds.delete(actionId);
        this.emitResult(item, 'CANCELLED', 0, {}, 'ACTION_CANCELLED_BEFORE_EXECUTION');
        continue;
      }

      this.current = item;
      const startedNs = Date.now() * 1000000;
      const worldTickStart = this.bot && this.bot.time ? this.bot.time.age : 0;
      const startHealth = this.bot && this.bot.health ? this.bot.health : 20;

      let status = 'SUCCESS';
      let failureReason = 'NONE';
      let stateDelta = {};

      const timeoutMs = 2500;
      let timeoutHandle = null;

      try {
        const timeoutPromise = new Promise((_, reject) => {
          timeoutHandle = setTimeout(() => reject(new Error('ACTION_TIMEOUT')), timeoutMs);
        });

        const executionPromise = this.executeDiscretePrimitive(item);
        const result = await Promise.race([executionPromise, timeoutPromise]);

        status = result.success ? 'SUCCESS' : 'FAILURE';
        failureReason = result.reason || 'NONE';
        stateDelta = result.delta || {};
      } catch (err) {
        if (err.message === 'ACTION_TIMEOUT') {
          // Action timed out. Per architecture rule 11: Timeout != Minecraft failure.
          // Mark outcome as UNKNOWN so learner does not treat it as a hard negative.
          status = 'UNKNOWN';
          failureReason = 'ACTION_TIMEOUT';
        } else {
          status = 'UNKNOWN';
          failureReason = err.message || 'EXECUTION_ERROR';
        }
        this.abortCurrentAction(status);
      } finally {
        if (timeoutHandle) clearTimeout(timeoutHandle);
      }

      const completedNs = Date.now() * 1000000;
      const worldTickEnd = this.bot && this.bot.time ? this.bot.time.age : worldTickStart;
      const endHealth = this.bot && this.bot.health ? this.bot.health : startHealth;
      stateDelta.health_delta = Number((endHealth - startHealth).toFixed(2));

      const elapsedMs = (completedNs - startedNs) / 1000000.0;
      this.emitResult(item, status, elapsedMs, stateDelta, failureReason, startedNs, completedNs, worldTickStart, worldTickEnd);
      this.current = null;
    }

    this.isExecuting = false;
  }

  async executeDiscretePrimitive(item) {
    const { executeInventoryAction } = require('./inventory');
    const { executeCraftingAction } = require('./crafting');
    const { executeContainerAction } = require('./containers');
    const { executeWorldMechanicsAction } = require('./mechanics');

    const cmd = item.action?.command || {};
    const primitive = cmd.primitive || 'noop';

    if (primitive === 'noop' || primitive === 'move_camera') {
      return { success: true, reason: 'NONE', delta: {} };
    }

    if (['hotbar_select', 'equip', 'unequip', 'swap_slots', 'drop_held', 'drop_stack', 'offhand_swap'].includes(primitive)) {
      return await executeInventoryAction(this.bot, cmd);
    }
    if (['craft', 'craft_batch'].includes(primitive)) {
      return await executeCraftingAction(this.bot, cmd);
    }
    if (['open_container', 'close_container', 'container_move', 'smelt'].includes(primitive)) {
      return await executeContainerAction(this.bot, cmd);
    }
    if (['dig', 'place', 'activate_block', 'toggle_switch', 'use_item', 'attack_entity', 'interact_entity', 'sleep', 'mount', 'dismount'].includes(primitive)) {
      return await executeWorldMechanicsAction(this.bot, cmd);
    }

    return { success: true, reason: 'NONE', delta: {} };
  }

  emitResult(item, status, elapsedMs, stateDelta, failureReason, startedNs, completedNs, worldTickStart, worldTickEnd) {
    if (this.onActionResult) {
      this.onActionResult({
        action_id: item.action_id,
        agent_id: item.agent_id,
        action_primitive: item.action?.command?.primitive || 'noop',
        started_ns: startedNs || Date.now() * 1000000,
        completed_ns: completedNs || Date.now() * 1000000,
        world_tick_start: worldTickStart || 0,
        world_tick_end: worldTickEnd || 0,
        status: status,
        elapsed_ms: elapsedMs,
        state_delta: stateDelta || {},
        failure_reason: failureReason || 'NONE',
      });
    }
  }
}

module.exports = { ActionExecutor };
