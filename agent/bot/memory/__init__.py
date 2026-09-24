from .replay import PrioritizedSequenceBuffer
from .spatial import SpatialMemory
from .experience_graph import ExperienceGraph

# Backward-compatibility alias
TrajectoryBuffer = PrioritizedSequenceBuffer

__all__ = [
    "PrioritizedSequenceBuffer",
    "TrajectoryBuffer",
    "SpatialMemory",
    "ExperienceGraph",
]
