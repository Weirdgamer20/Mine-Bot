from typing import Dict, Tuple, List, Optional
import numpy as np
import time

class SpatialMemory:
    """
    Learned 3D spatial memory of explored Minecraft regions across 16x16x16 sub-chunks (256x256x256 blocks).
    Tracks 3D subchunk empirical visit statistics, terrain latent embeddings, vertical cave/surface stratification,
    and observed consequences without relying on hardcoded coordinates.
    """
    def __init__(self, chunk_size: int = 16, macro_grid_span: int = 16):
        self.chunk_size = chunk_size          # 16 blocks per Minecraft sub-chunk dimension
        self.macro_grid_span = macro_grid_span # 16x16x16 chunk radius window (4,096 subchunks)
        self.regions: Dict[Tuple[int, int, int], Dict] = {} # (chunk_x, chunk_y, chunk_z) -> SubchunkData

    def _chunk_coords(self, x: float, z: float, y: float = 64.0) -> Tuple[int, int, int]:
        """Calculates 3D subchunk index (cx, cy, cz) for 16x16x16 block cubes."""
        cx = int(np.floor(x / self.chunk_size))
        cy = int(np.floor(y / self.chunk_size))
        cz = int(np.floor(z / self.chunk_size))
        return cx, cy, cz

    def _heading_sector(self, yaw: float) -> int:
        """Discretizes continuous yaw into 8 compass sectors (0=South, 2=West, 4=North, 6=East)."""
        normalized_yaw = yaw % (2 * np.pi)
        sector = int(np.floor((normalized_yaw + np.pi / 8) / (np.pi / 4))) % 8
        return sector

    def record_visit(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        terrain_latent: Optional[np.ndarray],
        consequence_delta: float = 0.0,
    ):
        """Records agent visitation into 3D subchunk coordinate space."""
        chunk = self._chunk_coords(x, z, y=y)
        sector = self._heading_sector(yaw)
        now = time.time()

        if chunk not in self.regions:
            self.regions[chunk] = {
                "chunk": chunk,
                "first_visited": now,
                "last_visited": now,
                "visit_count": 1,
                "y_mean": float(y),
                "terrain_latent": np.copy(terrain_latent) if terrain_latent is not None else None,
                "total_consequence": float(consequence_delta),
                "heading_sectors": {sector},
            }
        else:
            reg = self.regions[chunk]
            reg["last_visited"] = now
            reg["visit_count"] += 1
            reg["y_mean"] = 0.9 * reg["y_mean"] + 0.1 * float(y)
            if terrain_latent is not None:
                if reg["terrain_latent"] is not None:
                    reg["terrain_latent"] = 0.9 * reg["terrain_latent"] + 0.1 * terrain_latent
                else:
                    reg["terrain_latent"] = np.copy(terrain_latent)
            reg["total_consequence"] += float(consequence_delta)
            if "heading_sectors" not in reg:
                reg["heading_sectors"] = set()
            reg["heading_sectors"].add(sector)

    def get_visitation_count(self, x: float, z: float, y: float = 64.0) -> int:
        chunk = self._chunk_coords(x, z, y=y)
        return self.regions[chunk]["visit_count"] if chunk in self.regions else 0

    def get_spatial_novelty(self, x: float, z: float, y: float = 64.0, yaw: float = 0.0) -> float:
        """
        Returns spatial novelty bonus evaluated across 3D subchunk volume and compass orientation.
        Cave depths (negative Y) and mountain heights represent distinct 3D subchunks.
        """
        chunk = self._chunk_coords(x, z, y=y)
        if chunk not in self.regions:
            return 1.5 # Full novelty for unvisited 3D subchunk

        reg = self.regions[chunk]
        count = reg["visit_count"]
        chunk_novelty = float(1.0 / np.sqrt(count + 1.0))
        sector = self._heading_sector(yaw)
        sectors = reg.get("heading_sectors", set())

        # Directional curiosity bonus if venturing in an unexplored compass direction
        directional_bonus = 0.5 if sector not in sectors else 0.0
        return chunk_novelty + directional_bonus

    def get_macro_chunk_density(self, x: float, z: float, y: float = 64.0, radius_chunks: int = 8) -> float:
        """
        Calculates exploration density across surrounding 16x16x16 chunk macro volume (256x256x256 blocks).
        Returns ratio of explored subchunks in the local macro volume.
        """
        center_cx, center_cy, center_cz = self._chunk_coords(x, z, y=y)
        visited = 0
        total = (2 * radius_chunks) ** 3
        for dcx in range(-radius_chunks, radius_chunks):
            for dcy in range(-radius_chunks, radius_chunks):
                for dcz in range(-radius_chunks, radius_chunks):
                    if (center_cx + dcx, center_cy + dcy, center_cz + dcz) in self.regions:
                        visited += 1
        return float(visited / max(1, total))

    def total_regions_discovered(self) -> int:
        return len(self.regions)

    def to_dict(self) -> Dict:
        return {
            f"{cx},{cy},{cz}": {
                "chunk": list(r["chunk"]),
                "visit_count": r["visit_count"],
                "y_mean": r["y_mean"],
                "total_consequence": r["total_consequence"],
            }
            for (cx, cy, cz), r in self.regions.items()
        }

    def load_from_dict(self, data: Dict):
        for k, v in data.items():
            parts = k.split(",")
            if len(parts) == 3:
                cx, cy, cz = (int(x) for x in parts)
            elif len(parts) == 2:
                cx, cz = (int(x) for x in parts)
                cy = int(np.floor(v.get("y_mean", 64.0) / self.chunk_size))
            else:
                continue

            self.regions[(cx, cy, cz)] = {
                "chunk": (cx, cy, cz),
                "first_visited": time.time(),
                "last_visited": time.time(),
                "visit_count": v.get("visit_count", 1),
                "y_mean": v.get("y_mean", 64.0),
                "terrain_latent": np.zeros(64, dtype=np.float32),
                "total_consequence": v.get("total_consequence", 0.0),
                "heading_sectors": set(),
            }
