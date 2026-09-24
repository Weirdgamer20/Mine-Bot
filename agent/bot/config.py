from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class Config:
    # Dimensions
    hidden_dim: int = 256
    latent_dim: int = 64
    recurrent_dim: int = 256
    
    # Hierarchical Action Dimensions
    motor_dim: int = 7       # move_x, move_z, yaw_delta, pitch_delta, jump, sprint, sneak
    num_primitives: int = 27 # ActionPrimitive count
    max_slots: int = 36
    max_entities: int = 16
    
    # Canonical Vocabulary & Embeddings (matching Universal Dictionary)
    voxel_vocab: int = 1200
    voxel_emb_dim: int = 32
    item_vocab: int = 1500
    item_emb_dim: int = 32
    entity_vocab: int = 150
    entity_emb_dim: int = 32
    biome_vocab: int = 80
    biome_emb_dim: int = 16
    
    # State dimensions
    player_state_dim: int = 18
    affordance_dim: int = 8
    validity_mask_dim: int = 9
    
    # Training hyperparameters
    learning_rate: float = 3e-4
    batch_size: int = 16
    sequence_length: int = 16
    imagination_horizon: int = 12
    replay_capacity: int = 100_000
    gamma: float = 0.99
    lambda_gae: float = 0.95
    
    # Loss balance
    kl_weight: float = 0.1
    continuation_weight: float = 1.0
    rnd_weight: float = 1.0
    prediction_error_weight: float = 0.2
    entropy_weight: float = 0.01
    
    # Streaming Transport & Network
    stream_host: str = "0.0.0.0"
    stream_port: int = 9099
    
    # Persistence
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 500

    @classmethod
    def from_json(cls, path: str) -> "Config":
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text())
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
