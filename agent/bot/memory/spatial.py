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

    def record_visit(self, x: float, y: float, z: float, terrain_latent: np.ndarray, consequence_delta: float):
        chunk = self._chunk_coords(x, z)
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
            }
        else:
            reg = self.regions[chunk]
            reg["last_visited"] = now
            reg["visit_count"] += 1
            reg["y_mean"] = 0.9 * reg["y_mean"] + 0.1 * float(y)
            # Running average of terrain embedding
            reg["terrain_latent"] = 0.9 * reg["terrain_latent"] + 0.1 * terrain_latent
            reg["total_consequence"] += float(consequence_delta)

    def get_visitation_count(self, x: float, z: float) -> int:
        chunk = self._chunk_coords(x, z)
        return self.regions[chunk]["visit_count"] if chunk in self.regions else 0

    def get_spatial_novelty(self, x: float, z: float) -> float:
        """Returns higher novelty bonus for unvisited or rarely visited chunks."""
        count = self.get_visitation_count(x, z)
        return float(1.0 / np.sqrt(count + 1.0))

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
