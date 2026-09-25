from __future__ import annotations

import queue
import threading
import time
import logging
from typing import Optional, Dict, Any

from ..experience import TemporalTransition, ValidationStatus

logger = logging.getLogger("ReplayWorker")


class ReplayWorker(threading.Thread):
    """
    Dedicated worker thread for processing incoming transitions into Prioritized Experience Replay.
    Decouples trajectory collation, spatial hashing, and graph updates from the 100 Hz control loop.
    """

    def __init__(
        self,
        experience_queue: queue.Queue,
        memory,
        spatial_memory=None,
        experience_graph=None,
    ):
        super().__init__(daemon=True, name="ReplayWorker")
        self.experience_queue = experience_queue
        self.memory = memory
        self.spatial_memory = spatial_memory
        self.experience_graph = experience_graph
        self.running = False

        # Telemetry metrics
        self.experience_processed = 0
        self.experience_dropped = 0
        self.duplicate_rejected = 0

    def run(self):
        self.running = True
        logger.info("[ReplayWorker] Started async replay ingestion thread.")

        while self.running:
            try:
                transition: TemporalTransition = self.experience_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._ingest_transition(transition)
                self.experience_processed += 1
            except Exception as e:
                self.experience_dropped += 1
                logger.error("[ReplayWorker] Error ingesting transition: %s", e)
            finally:
                self.experience_queue.task_done()

    def _ingest_transition(self, t: TemporalTransition):
        if t.validation_status != ValidationStatus.VALID:
            self.duplicate_rejected += 1
            return

        obs = t.state
        act = t.action
        next_obs = t.next_state

        # Flatten arrays for replay buffer
        shape = tuple(obs.voxel_shape) if len(obs.voxel_shape) == 3 else (11, 11, 11)
        expected_size = shape[0] * shape[1] * shape[2]
        vox = obs.voxels if len(obs.voxels) == expected_size else [0] * expected_size

        import numpy as np
        import torch
        import torch.nn.functional as F

        vox_arr = np.array(vox, dtype=np.int64).reshape(*shape)
        player_arr = np.array(obs.player_state, dtype=np.float32)
        if len(player_arr) < 18:
            padded = np.zeros(18, dtype=np.float32)
            padded[: len(player_arr)] = player_arr
            player_arr = padded
        else:
            player_arr = player_arr[:18]

        inv_data = []
        raw_slots = obs.inventory.slots if obs.inventory else []
        for i in range(36):
            if i < len(raw_slots):
                s = raw_slots[i]
                inv_data.append([s.item_canonical_id, s.count, s.durability])
            else:
                inv_data.append([0, 0, 0.0])
        for a in [obs.inventory.armor_head, obs.inventory.armor_chest, obs.inventory.armor_legs, obs.inventory.armor_feet]:
            inv_data.append([a.item_canonical_id, a.count, a.durability])
        oh = obs.inventory.offhand
        inv_data.append([oh.item_canonical_id, oh.count, oh.durability])
        inv_arr = np.array(inv_data, dtype=np.float32)

        ent_data = []
        for i in range(16):
            if i < len(obs.entities):
                e = obs.entities[i]
                ent_data.append([e.canonical_type_id, e.dx, e.dy, e.dz, e.vx, e.vy, e.vz, e.health, 1.0 if e.is_alive else 0.0])
            else:
                ent_data.append([0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ent_arr = np.array(ent_data, dtype=np.float32)

        aff_arr = np.array([
            obs.affordances.can_jump,
            obs.affordances.can_sprint,
            obs.affordances.can_sneak,
            obs.affordances.can_dig,
            obs.affordances.can_place,
            obs.affordances.can_attack,
            obs.affordances.can_interact,
            obs.affordances.in_water,
        ], dtype=np.float32)

        val_arr = np.array(obs.validity_mask.valid_primitives_mask[:9], dtype=np.float32)

        # Flat action array: 7 motor + 27 primitive one-hot = 34
        motor_arr = np.array([
            act.motor.move_x,
            act.motor.move_z,
            act.motor.yaw_delta,
            act.motor.pitch_delta,
            1.0 if act.motor.jump else 0.0,
            1.0 if act.motor.sprint else 0.0,
            1.0 if act.motor.sneak else 0.0,
        ], dtype=np.float32)

        from ..schemas import PRIMITIVE_TO_IDX
        p_idx = PRIMITIVE_TO_IDX.get(act.command.primitive.value, 0)
        prim_arr = np.zeros(27, dtype=np.float32)
        prim_arr[min(p_idx, 26)] = 1.0
        full_act_arr = np.concatenate([motor_arr, prim_arr], axis=-1)

        priority = max(abs(t.reward.total) + 0.1, 0.1)

        self.memory.add(
            voxels=vox_arr,
            player_state=player_arr,
            inventory=inv_arr,
            entities=ent_arr,
            affordances=aff_arr,
            validity_mask=val_arr,
            action=full_act_arr,
            reward=t.reward.total,
            continuation=t.continuation,
            done=t.done,
            priority=priority,
            success=t.result.success,
            failure_reason=t.result.failure_reason,
            consequence_delta=float(t.result.state_delta.get("health_delta", 0.0)),
        )

        if self.spatial_memory:
            px = obs.player_state[4] if len(obs.player_state) > 4 else 0.0
            py = obs.player_state[5] if len(obs.player_state) > 5 else 64.0
            pz = obs.player_state[6] if len(obs.player_state) > 6 else 0.0
            yaw = obs.player_state[11] if len(obs.player_state) > 11 else 0.0
            self.spatial_memory.record_visit(px, py, pz, yaw, None, consequence_delta=0.0)

    def stop(self):
        self.running = False
