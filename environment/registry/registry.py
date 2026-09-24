from pathlib import Path
from typing import Dict, Any, List, Optional
import json

class CanonicalRegistry:
    """
    Loads the canonical Minecraft registry tables and universal dictionary.
    Guarantees stable integer indices across any Minecraft game version.
    """
    _instance: Optional["CanonicalRegistry"] = None

    def __init__(self, registry_dir: Optional[str] = None):
        base_dir = Path(registry_dir) if registry_dir else Path(__file__).parent
        self.base_dir = base_dir

        # Universal Dictionary (string -> integer index)
        dict_path = base_dir / "universal_dictionary.json"
        if not dict_path.exists():
            raise FileNotFoundError(f"Universal dictionary missing at {dict_path}")
        self.dict_data = json.loads(dict_path.read_text(encoding="utf-8"))
        self.vocab_version = self.dict_data.get("vocab_version", 1)
        self.ref_version = self.dict_data.get("reference_version", "unknown")

        self.block_to_idx: Dict[str, int] = self.dict_data["namespaces"]["blocks"]
        self.item_to_idx: Dict[str, int] = self.dict_data["namespaces"]["items"]
        self.entity_to_idx: Dict[str, int] = self.dict_data["namespaces"]["entities"]
        self.biome_to_idx: Dict[str, int] = self.dict_data["namespaces"]["biomes"]

        # Inverse mappings (index -> string)
        self.idx_to_block = {v: k for k, v in self.block_to_idx.items()}
        self.idx_to_item = {v: k for k, v in self.item_to_idx.items()}
        self.idx_to_entity = {v: k for k, v in self.entity_to_idx.items()}
        self.idx_to_biome = {v: k for k, v in self.biome_to_idx.items()}

        # Mechanical fact tables (the rulebook facts)
        self.blocks = json.loads((base_dir / "blocks.json").read_text(encoding="utf-8"))
        self.items = json.loads((base_dir / "items.json").read_text(encoding="utf-8"))
        self.entities = json.loads((base_dir / "entities.json").read_text(encoding="utf-8"))
        self.biomes = json.loads((base_dir / "biomes.json").read_text(encoding="utf-8"))
        self.recipes = json.loads((base_dir / "recipes.json").read_text(encoding="utf-8"))

    @classmethod
    def get_instance(cls, registry_dir: Optional[str] = None) -> "CanonicalRegistry":
        if cls._instance is None:
            cls._instance = cls(registry_dir)
        return cls._instance

    @property
    def block_vocab_size(self) -> int:
        return len(self.block_to_idx)

    @property
    def item_vocab_size(self) -> int:
        return len(self.item_to_idx)

    @property
    def entity_vocab_size(self) -> int:
        return len(self.entity_to_idx)

    @property
    def biome_vocab_size(self) -> int:
        return len(self.biome_to_idx)

    def block_index(self, name: str) -> int:
        """Returns stable integer index for block name (with fallback to 0=<UNKNOWN>)."""
        clean = name.split(":")[-1] if ":" in name else name
        return self.block_to_idx.get(clean, 0)

    def item_index(self, name: str) -> int:
        clean = name.split(":")[-1] if ":" in name else name
        return self.item_to_idx.get(clean, 0)

    def entity_index(self, name: str) -> int:
        clean = name.split(":")[-1] if ":" in name else name
        return self.entity_to_idx.get(clean, 0)

    def biome_index(self, name: str) -> int:
        clean = name.split(":")[-1] if ":" in name else name
        return self.biome_to_idx.get(clean, 0)

    def block_name(self, index: int) -> str:
        return self.idx_to_block.get(index, "<UNKNOWN>")

    def item_name(self, index: int) -> str:
        return self.idx_to_item.get(index, "<UNKNOWN>")

    def entity_name(self, index: int) -> str:
        return self.idx_to_entity.get(index, "<UNKNOWN>")

    def get_block_info(self, name: str) -> Dict[str, Any]:
        clean = name.split(":")[-1] if ":" in name else name
        return self.blocks.get(clean, {
            "name": clean, "hardness": 0.0, "diggable": False, "transparent": False, "boundingBox": "block"
        })

    def get_item_info(self, name: str) -> Dict[str, Any]:
        clean = name.split(":")[-1] if ":" in name else name
        return self.items.get(clean, {
            "name": clean, "stackSize": 64, "maxDurability": 0
        })
