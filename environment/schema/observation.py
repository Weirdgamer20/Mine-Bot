from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from .action import ActionResult, ActionValidityMask

@dataclass
class ItemSlotData:
    slot_index: int = 0
    item_canonical_id: int = 0  # Canonical stable index from Universal Dictionary
    count: int = 0
    durability: float = 0.0

@dataclass
class CompleteInventoryState:
    slots: List[ItemSlotData] = field(default_factory=list) # 36 main inventory slots
    armor_head: ItemSlotData = field(default_factory=ItemSlotData)
    armor_chest: ItemSlotData = field(default_factory=ItemSlotData)
    armor_legs: ItemSlotData = field(default_factory=ItemSlotData)
    armor_feet: ItemSlotData = field(default_factory=ItemSlotData)
    offhand: ItemSlotData = field(default_factory=ItemSlotData)
    selected_hotbar_slot: int = 0

@dataclass
class PerceivedEntityData:
    entity_id: int = 0
    canonical_type_id: int = 0 # Canonical stable index from Universal Dictionary
    category: str = "misc"     # "hostile", "passive", "neutral", "player", "projectile", "misc"
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    health: float = 0.0
    distance: float = 0.0
    is_alive: bool = True

@dataclass
class MechanicalAffordanceState:
    targeted_block_canonical_id: int = 0
    targeted_block_distance: float = 0.0
    targeted_block_face: int = 1
    can_mine_target: bool = False
    targeted_entity_idx: int = -1
    light_level: int = 0
    sky_light: int = 0
    open_container_type: str = "none"

@dataclass
class FullObservation:
    # 1. 11x11x11 Voxel terrain (1331 canonical block IDs)
    voxels: List[int] = field(default_factory=list)
    voxel_shape: List[int] = field(default_factory=lambda: [11, 11, 11])

    # 2. Kinematic physical player state vector (health, food, coords, velocity, status)
    player_state: List[float] = field(default_factory=list)

    # 3. Complete inventory (36 main slots + 4 armor + 1 offhand + selected hotbar slot)
    inventory: CompleteInventoryState = field(default_factory=CompleteInventoryState)

    # 4. Nearby entities (up to 16 closest within perception radius)
    entities: List[PerceivedEntityData] = field(default_factory=list)

    # 5. Mechanical affordances (raycast targeted block, distance, light, container)
    affordances: MechanicalAffordanceState = field(default_factory=MechanicalAffordanceState)

    # 6. Action result feedback from the previous action
    last_action_result: ActionResult = field(default_factory=ActionResult)

    # 7. Mechanical validity mask (VALIDITY != STRATEGY)
    validity_mask: ActionValidityMask = field(default_factory=ActionValidityMask)

    # 8. Termination and context
    done: bool = False
    episode_id: int = 0
    step_id: int = 0
    timestamp: float = 0.0
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "voxels": self.voxels,
            "voxel_shape": self.voxel_shape,
            "player_state": self.player_state,
            "inventory": {
                "slots": [asdict(s) for s in self.inventory.slots],
                "armor_head": asdict(self.inventory.armor_head),
                "armor_chest": asdict(self.inventory.armor_chest),
                "armor_legs": asdict(self.inventory.armor_legs),
                "armor_feet": asdict(self.inventory.armor_feet),
                "offhand": asdict(self.inventory.offhand),
                "selected_hotbar_slot": int(self.inventory.selected_hotbar_slot),
            },
            "entities": [asdict(e) for e in self.entities],
            "affordances": asdict(self.affordances),
            "last_action_result": self.last_action_result.to_dict(),
            "validity_mask": self.validity_mask.to_dict(),
            "done": bool(self.done),
            "episode_id": int(self.episode_id),
            "step_id": int(self.step_id),
            "timestamp": float(self.timestamp),
            "info": self.info,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FullObservation":
        inv_data = data.get("inventory", {})
        slots = [
            ItemSlotData(
                slot_index=int(s.get("slot_index", idx)),
                item_canonical_id=int(s.get("item_canonical_id", s.get("item_id", 0))),
                count=int(s.get("count", s.get("quantity", 0))),
                durability=float(s.get("durability", 0.0)),
            )
            for idx, s in enumerate(inv_data.get("slots", []))
        ]
        
        inv = CompleteInventoryState(
            slots=slots,
            armor_head=ItemSlotData(**inv_data.get("armor_head", {})),
            armor_chest=ItemSlotData(**inv_data.get("armor_chest", {})),
            armor_legs=ItemSlotData(**inv_data.get("armor_legs", {})),
            armor_feet=ItemSlotData(**inv_data.get("armor_feet", {})),
            offhand=ItemSlotData(**inv_data.get("offhand", {})),
            selected_hotbar_slot=int(inv_data.get("selected_hotbar_slot", 0)),
        )

        entities = [
            PerceivedEntityData(
                entity_id=int(e.get("entity_id", 0)),
                canonical_type_id=int(e.get("canonical_type_id", e.get("type_id", 0))),
                category=str(e.get("category", "misc")),
                dx=float(e.get("dx", 0.0)),
                dy=float(e.get("dy", 0.0)),
                dz=float(e.get("dz", 0.0)),
                vx=float(e.get("vx", 0.0)),
                vy=float(e.get("vy", 0.0)),
                vz=float(e.get("vz", 0.0)),
                health=float(e.get("health", 0.0)),
                distance=float(e.get("distance", 0.0)),
                is_alive=bool(e.get("is_alive", True)),
            )
            for e in data.get("entities", [])
        ]

        aff_data = data.get("affordances", {})
        aff = MechanicalAffordanceState(
            targeted_block_canonical_id=int(aff_data.get("targeted_block_canonical_id", aff_data.get("target_block_id", 0))),
            targeted_block_distance=float(aff_data.get("targeted_block_distance", aff_data.get("target_block_distance", 0.0))),
            targeted_block_face=int(aff_data.get("targeted_block_face", 1)),
            can_mine_target=bool(aff_data.get("can_mine_target", aff_data.get("can_mine", False))),
            targeted_entity_idx=int(aff_data.get("targeted_entity_idx", -1)),
            light_level=int(aff_data.get("light_level", 0)),
            sky_light=int(aff_data.get("sky_light", 0)),
            open_container_type=str(aff_data.get("open_container_type", "none")),
        )

        last_res = ActionResult.from_dict(data.get("last_action_result", {}))
        val_mask = ActionValidityMask.from_dict(data.get("validity_mask", {}))

        return cls(
            voxels=[int(x) for x in data.get("voxels", [])],
            voxel_shape=data.get("voxel_shape", [11, 11, 11]),
            player_state=[float(x) for x in data.get("player_state", [])],
            inventory=inv,
            entities=entities,
            affordances=aff,
            last_action_result=last_res,
            validity_mask=val_mask,
            done=bool(data.get("done", False)),
            episode_id=int(data.get("episode_id", 0)),
            step_id=int(data.get("step_id", 0)),
            timestamp=float(data.get("timestamp", 0.0)),
            info=data.get("info", {}),
        )
