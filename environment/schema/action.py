from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, List, Optional
import numpy as np

class ActionCategory(str, Enum):
    IDLE = "idle"
    MOVEMENT = "movement"
    INTERACTION = "interaction"
    WORLD = "world"
    INVENTORY = "inventory"
    CRAFTING = "crafting"
    CONTAINER = "container"
    PROCESSING = "processing"
    VEHICLE = "vehicle"
    ENVIRONMENT = "environment"

class ActionPrimitive(str, Enum):
    # Tier 1 - Movement
    NOOP = "noop"
    MOVE_CAMERA = "move_camera"
    JUMP = "jump"
    SPRINT = "sprint"
    SNEAK = "sneak"
    SWIM = "swim"
    CLIMB = "climb"

    # Tier 2 - Interaction
    ATTACK_ENTITY = "attack_entity"
    INTERACT_ENTITY = "interact_entity"
    ACTIVATE_BLOCK = "activate_block"
    USE_ITEM = "use_item"

    # Tier 3 - World manipulation
    DIG = "dig"
    PLACE = "place"

    # Tier 4 - Inventory
    HOTBAR_SELECT = "hotbar_select"
    EQUIP = "equip"
    UNEQUIP = "unequip"
    SWAP_SLOTS = "swap_slots"
    DROP_HELD = "drop_held"
    DROP_STACK = "drop_stack"
    OFFHAND_SWAP = "offhand_swap"

    # Tier 5 - Crafting
    CRAFT = "craft"
    CRAFT_BATCH = "craft_batch"

    # Tier 6 - Container
    OPEN_CONTAINER = "open_container"
    CLOSE_CONTAINER = "close_container"
    CONTAINER_MOVE = "container_move"

    # Tier 7 - Processing
    SMELT = "smelt"

    # Tier 8 - Vehicles
    MOUNT = "mount"
    DISMOUNT = "dismount"

    # Tier 9 - Environmental
    SLEEP = "sleep"
    TOGGLE_SWITCH = "toggle_switch"

# List of all primitives for discrete categorical indexing
ALL_PRIMITIVES = list(ActionPrimitive)
PRIMITIVE_TO_IDX = {p.value: idx for idx, p in enumerate(ALL_PRIMITIVES)}
IDX_TO_PRIMITIVE = {idx: p.value for idx, p in enumerate(ALL_PRIMITIVES)}
NUM_PRIMITIVES = len(ALL_PRIMITIVES)

@dataclass
class ContinuousMotorControl:
    """Continuous locomotion & camera orientation updated on every tick."""
    move_x: float = 0.0      # -1.0 (left) to +1.0 (right)
    move_z: float = 0.0      # -1.0 (back) to +1.0 (forward)
    yaw_delta: float = 0.0   # Horizontal camera delta in radians
    pitch_delta: float = 0.0 # Vertical camera delta in radians
    jump: bool = False
    sprint: bool = False
    sneak: bool = False

@dataclass
class DiscreteActionCommand:
    """Hierarchical action primitive selected from the Action Ontology."""
    category: ActionCategory = ActionCategory.IDLE
    primitive: ActionPrimitive = ActionPrimitive.NOOP
    # Parameter slots
    target_entity_idx: int = 0      # 0 to 15 (nearest entity index)
    target_slot: int = 0            # 0 to 35
    target_slot_dest: int = 0       # 0 to 35
    recipe_name: str = ""           # Canonical recipe name
    target_block_face: int = 1      # 0=bottom, 1=top, 2=north, 3=south, 4=west, 5=east
    duration_ticks: int = 1         # Desired persistence / lock

@dataclass
class HierarchicalAction:
    """
    Unified Action command: Continuous motor locomotion + Discrete hierarchical primitive.
    """
    motor: ContinuousMotorControl = field(default_factory=ContinuousMotorControl)
    command: DiscreteActionCommand = field(default_factory=DiscreteActionCommand)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "motor": {
                "move_x": float(self.motor.move_x),
                "move_z": float(self.motor.move_z),
                "yaw_delta": float(self.motor.yaw_delta),
                "pitch_delta": float(self.motor.pitch_delta),
                "jump": bool(self.motor.jump),
                "sprint": bool(self.motor.sprint),
                "sneak": bool(self.motor.sneak),
            },
            "command": {
                "category": self.command.category.value,
                "primitive": self.command.primitive.value,
                "target_entity_idx": int(self.command.target_entity_idx),
                "target_slot": int(self.command.target_slot),
                "target_slot_dest": int(self.command.target_slot_dest),
                "recipe_name": str(self.command.recipe_name),
                "target_block_face": int(self.command.target_block_face),
                "duration_ticks": int(self.command.duration_ticks),
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HierarchicalAction":
        m = data.get("motor", {})
        c = data.get("command", {})
        
        motor = ContinuousMotorControl(
            move_x=float(m.get("move_x", 0.0)),
            move_z=float(m.get("move_z", 0.0)),
            yaw_delta=float(m.get("yaw_delta", 0.0)),
            pitch_delta=float(m.get("pitch_delta", 0.0)),
            jump=bool(m.get("jump", False)),
            sprint=bool(m.get("sprint", False)),
            sneak=bool(m.get("sneak", False)),
        )

        cat_val = c.get("category", ActionCategory.IDLE.value)
        prim_val = c.get("primitive", ActionPrimitive.NOOP.value)
        
        try:
            category = ActionCategory(cat_val)
        except ValueError:
            category = ActionCategory.IDLE

        try:
            primitive = ActionPrimitive(prim_val)
        except ValueError:
            primitive = ActionPrimitive.NOOP

        command = DiscreteActionCommand(
            category=category,
            primitive=primitive,
            target_entity_idx=int(c.get("target_entity_idx", 0)),
            target_slot=int(c.get("target_slot", 0)),
            target_slot_dest=int(c.get("target_slot_dest", 0)),
            recipe_name=str(c.get("recipe_name", "")),
            target_block_face=int(c.get("target_block_face", 1)),
            duration_ticks=int(c.get("duration_ticks", 1)),
        )

        return cls(motor=motor, command=command)

@dataclass
class ActionResult:
    """Action outcome returned by the environment (feedback on what succeeded/failed)."""
    action_primitive: str = ActionPrimitive.NOOP.value
    success: bool = True
    failure_reason: str = "NONE" # "NONE", "OUT_OF_RANGE", "NO_TOOL", "CANNOT_REACH", "INSUFFICIENT_ITEMS", etc.
    elapsed_ticks: int = 1
    state_delta: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionResult":
        return cls(
            action_primitive=str(data.get("action_primitive", ActionPrimitive.NOOP.value)),
            success=bool(data.get("success", True)),
            failure_reason=str(data.get("failure_reason", "NONE")),
            elapsed_ticks=int(data.get("elapsed_ticks", 1)),
            state_delta=data.get("state_delta", {}),
        )

@dataclass
class ActionValidityMask:
    """
    Mechanical validity mask provided by Minecraft (VALIDITY != STRATEGY).
    Informs the policy which actions are physically possible in the current state.
    """
    can_jump: bool = True
    can_sprint: bool = True
    can_sneak: bool = True
    can_attack_entity: bool = False
    can_dig_block: bool = False
    can_place_block: bool = False
    can_use_item: bool = False
    can_open_container: bool = False
    can_sleep: bool = False
    valid_primitives_mask: List[bool] = field(default_factory=lambda: [True] * NUM_PRIMITIVES)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionValidityMask":
        return cls(
            can_jump=bool(data.get("can_jump", True)),
            can_sprint=bool(data.get("can_sprint", True)),
            can_sneak=bool(data.get("can_sneak", True)),
            can_attack_entity=bool(data.get("can_attack_entity", False)),
            can_dig_block=bool(data.get("can_dig_block", False)),
            can_place_block=bool(data.get("can_place_block", False)),
            can_use_item=bool(data.get("can_use_item", False)),
            can_open_container=bool(data.get("can_open_container", False)),
            can_sleep=bool(data.get("can_sleep", False)),
            valid_primitives_mask=data.get("valid_primitives_mask", [True] * NUM_PRIMITIVES),
        )
