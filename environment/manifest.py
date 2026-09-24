from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import time

@dataclass
class EnvironmentManifest:
    """
    Locks and identifies the active Minecraft environment specifications.
    Saved into checkpoints and verified during handshake to prevent undefined behavior
    when checkpoints are transferred across different Minecraft/Mineflayer versions.
    """
    minecraft_version: str = "1.20.4"
    protocol_version: int = 765
    mineflayer_version: str = "4.25.0"
    minecraft_data_version: str = "3.78.0"
    environment_schema_version: int = 1
    registry_vocab_version: int = 1
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["timestamp"] == 0.0:
            d["timestamp"] = time.time()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnvironmentManifest":
        return cls(
            minecraft_version=str(data.get("minecraft_version", "unknown")),
            protocol_version=int(data.get("protocol_version", 0)),
            mineflayer_version=str(data.get("mineflayer_version", "unknown")),
            minecraft_data_version=str(data.get("minecraft_data_version", "unknown")),
            environment_schema_version=int(data.get("environment_schema_version", 1)),
            registry_vocab_version=int(data.get("registry_vocab_version", 1)),
            timestamp=float(data.get("timestamp", time.time())),
        )
