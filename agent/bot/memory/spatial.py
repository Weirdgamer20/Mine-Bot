from typing import Dict, Tuple, List, Optional
import numpy as np
import time

class SpatialMemory:
    """
    Learned spatial memory of explored Minecraft regions.
    Does NOT store hardcoded diamond or resource coordinates.
    Maintains empirical visitation statistics, terrain embeddings, and observed consequences.
    """
    def __init__(self, chunk_size: int = 16):
        self.chunk_size = chunk_size
        self.regions: Dict[Tuple[int, int], Dict] = {} # (chunk_x, chunk_z) -> RegionData

    def _chunk_coords(self, x: float, z: float) -> Tuple[int, int]:
        return int(np.floor(x / self.chunk_size)), int(np.floor(z / self.chunk_size))

    def _heading_sector(self, yaw: float) -> int:
        """Discretizes continuous yaw into 8 compass sectors (0=South, 2=West, 4=North, 6=East)."""
        normalized_yaw = yaw % (2 * np.pi)
        sector = int(np.floor((normalized_yaw + np.pi / 8) / (np.pi / 4))) % 8
        return sector

    def record_visit(self, x: float, y: float, z: float, yaw: float, terrain_latent: np.ndarray, consequence_delta: float):
        chunk = self._chunk_coords(x, z)
        sector = self._heading_sector(yaw)
        now = time.time()

        if chunk not in self.regions:
            self.regions[chunk] = {
                "chunk": chunk,
                "first_visited": now,
                "last_visited": now,
                "visit_count": 1,
                "y_mean": float(y),
                "terrain_latent": np.copy(terrain_latent),
                "total_consequence": float(consequence_delta),
                "heading_sectors": {sector},
            }
        else:
            reg = self.regions[chunk]
            reg["last_visited"] = now
            reg["visit_count"] += 1
            reg["y_mean"] = 0.9 * reg["y_mean"] + 0.1 * float(y)
            reg["terrain_latent"] = 0.9 * reg["terrain_latent"] + 0.1 * terrain_latent
            reg["total_consequence"] += float(consequence_delta)
            if "heading_sectors" not in reg:
                reg["heading_sectors"] = set()
            reg["heading_sectors"].add(sector)

    def get_visitation_count(self, x: float, z: float) -> int:
        chunk = self._chunk_coords(x, z)
        return self.regions[chunk]["visit_count"] if chunk in self.regions else 0

    def get_spatial_novelty(self, x: float, z: float, yaw: float = 0.0) -> float:
        """Returns higher novelty bonus for unvisited chunks and unexplored compass headings."""
        chunk = self._chunk_coords(x, z)
        if chunk not in self.regions:
            return 1.5 # Full novelty for unvisited chunk
        reg = self.regions[chunk]
        count = reg["visit_count"]
        chunk_novelty = float(1.0 / np.sqrt(count + 1.0))
        sector = self._heading_sector(yaw)
        sectors = reg.get("heading_sectors", set())
        # Directional curiosity bonus if venturing in an unexplored compass direction
        directional_bonus = 0.5 if sector not in sectors else 0.0
        return chunk_novelty + directional_bonus

    def total_regions_discovered(self) -> int:
        return len(self.regions)

    def to_dict(self) -> Dict:
        return {
            f"{cx},{cz}": {
                "chunk": list(r["chunk"]),
                "visit_count": r["visit_count"],
                "y_mean": r["y_mean"],
                "total_consequence": r["total_consequence"],
            }
            for (cx, cz), r in self.regions.items()
        }

    def load_from_dict(self, data: Dict):
        for k, v in data.items():
            cx, cz = (int(x) for x in k.split(","))
            self.regions[(cx, cz)] = {
                "chunk": (cx, cz),
                "first_visited": time.time(),
                "last_visited": time.time(),
                "visit_count": v.get("visit_count", 1),
                "y_mean": v.get("y_mean", 64.0),
                "terrain_latent": np.zeros(64, dtype=np.float32),
                "total_consequence": v.get("total_consequence", 0.0),
            }
