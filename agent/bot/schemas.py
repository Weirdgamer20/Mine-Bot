from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

@dataclass
class EntityObservation:
    entity_id: int = 0
    type_id: int = 0
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    health: float = 0.0
    is_alive: bool = True

@dataclass
class InventorySlot:
    slot_index: int = 0
    item_id: int = 0
    count: int = 0
    durability: float = 0.0

@dataclass
class Affordances:
    """Mechanical facts provided by Minecraft (the rulebook, not strategy)."""
    target_block_id: int = 0
    target_block_distance: float = 0.0
    can_mine: bool = False
    light_level: int = 0
    time_of_day: float = 0.0
    is_raining: bool = False
    is_sleeping: bool = False
    is_swimming: bool = False
    equipped_item_id: int = 0

@dataclass
class Observation:
    voxels: List[int] = field(default_factory=list)
    voxel_shape: List[int] = field(default_factory=lambda: [11, 11, 11])
    player_state: List[float] = field(default_factory=list)
    inventory: List[InventorySlot] = field(default_factory=list)
    entities: List[EntityObservation] = field(default_factory=list)
    affordances: Affordances = field(default_factory=Affordances)
    delta_state: List[float] = field(default_factory=list)
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        aff_data = data.get("affordances", {})
        aff = Affordances(
            target_block_id=int(aff_data.get("target_block_id", 0)),
            target_block_distance=float(aff_data.get("target_block_distance", 0.0)),
            can_mine=bool(aff_data.get("can_mine", False)),
            light_level=int(aff_data.get("light_level", 0)),
            time_of_day=float(aff_data.get("time_of_day", 0.0)),
            is_raining=bool(aff_data.get("is_raining", False)),
            is_sleeping=bool(aff_data.get("is_sleeping", False)),
            is_swimming=bool(aff_data.get("is_swimming", False)),
            equipped_item_id=int(aff_data.get("equipped_item_id", 0)),
        )

        inv = [
            InventorySlot(
                slot_index=int(s.get("slot_index", idx)),
                item_id=int(s.get("item_id", 0)),
                count=int(s.get("count", s.get("quantity", 0))),
                durability=float(s.get("durability", 0.0)),
            )
            for idx, s in enumerate(data.get("inventory", []))
        ]

        ents = [
            EntityObservation(
                entity_id=int(e.get("entity_id", 0)),
                type_id=int(e.get("type_id", 0)),
                dx=float(e.get("dx", 0.0)),
                dy=float(e.get("dy", 0.0)),
                dz=float(e.get("dz", 0.0)),
                vx=float(e.get("vx", 0.0)),
                vy=float(e.get("vy", 0.0)),
                vz=float(e.get("vz", 0.0)),
                health=float(e.get("health", 0.0)),
                is_alive=bool(e.get("is_alive", True)),
            )
            for e in data.get("entities", [])
        ]

        return cls(
            voxels=data.get("voxels", []),
            voxel_shape=data.get("voxel_shape", [11, 11, 11]),
            player_state=[float(x) for x in data.get("player_state", [])],
            inventory=inv,
            entities=ents,
            affordances=aff,
            delta_state=[float(x) for x in data.get("delta_state", [])],
            done=bool(data.get("done", False)),
            info=data.get("info", {}),
        )

@dataclass
class Action:
    move_x: float = 0.0       # -1.0 (left) to +1.0 (right)
    move_z: float = 0.0       # -1.0 (back) to +1.0 (forward)
    jump: float = 0.0         # > 0.5 activates jump
    sneak: float = 0.0        # > 0.5 activates sneak
    sprint: float = 0.0       # > 0.5 activates sprint
    yaw_delta: float = 0.0    # rotation delta horizontal
    pitch_delta: float = 0.0  # rotation delta vertical
    attack: float = 0.0       # > 0.5 swings/attacks/mines
    use: float = 0.0          # > 0.5 uses/places held item
    slot: int = 0             # 0 to 8: selected hotbar slot
    interact: float = 0.0     # > 0.5 interact with targeted block/entity
    drop: float = 0.0         # > 0.5 drop held item

    def to_dict(self) -> Dict[str, Any]:
        return {
            "move_x": float(self.move_x),
            "move_z": float(self.move_z),
            "jump": float(self.jump),
            "sneak": float(self.sneak),
            "sprint": float(self.sprint),
            "yaw_delta": float(self.yaw_delta),
            "pitch_delta": float(self.pitch_delta),
            "attack": float(self.attack),
            "use": float(self.use),
            "slot": int(self.slot),
            "interact": float(self.interact),
            "drop": float(self.drop),
        }

    @classmethod
    def from_vector(cls, vec: List[float]) -> "Action":
        """Decodes raw neural policy outputs into physical mechanical actions."""
        # Ensure 12 dimensions
        padded = list(vec) + [0.0] * max(0, 12 - len(vec))
        # Continuous controls
        mx = float(np.clip(padded[0], -1.0, 1.0))
        mz = float(np.clip(padded[1], -1.0, 1.0))
        # Binary buttons (sigmoid / threshold)
        jump = float(1.0 / (1.0 + np.exp(-padded[2])))
        sneak = float(1.0 / (1.0 + np.exp(-padded[3])))
        sprint = float(1.0 / (1.0 + np.exp(-padded[4])))
        # Angle deltas
        yaw_delta = float(np.clip(padded[5], -1.0, 1.0))
        pitch_delta = float(np.clip(padded[6], -1.0, 1.0))
        # Interactions
        attack = float(1.0 / (1.0 + np.exp(-padded[7])))
        use = float(1.0 / (1.0 + np.exp(-padded[8])))
        # Hotbar slot (discrete 0-8)
        slot = int(np.clip(np.floor((padded[9] + 1.0) * 4.5), 0, 8))
        interact = float(1.0 / (1.0 + np.exp(-padded[10])))
        drop = float(1.0 / (1.0 + np.exp(-padded[11])))

        return cls(
            move_x=mx,
            move_z=mz,
            jump=jump,
            sneak=sneak,
            sprint=sprint,
            yaw_delta=yaw_delta,
            pitch_delta=pitch_delta,
            attack=attack,
            use=use,
            slot=slot,
            interact=interact,
            drop=drop,
        )
