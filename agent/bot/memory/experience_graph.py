from typing import Dict, List, Tuple, Any, Optional
import numpy as np

class ExperienceGraph:
    """
    Learned topological Experience Graph.
    Nodes: Empirical state signatures, objects, and events.
    Edges: Actions taken, observed consequences, transition frequencies, and state deltas.
    Allows the agent to learn causal capability graphs without hardcoded strategies.
    """
    def __init__(self, max_nodes: int = 50000):
        self.max_nodes = max_nodes
        self.nodes: Dict[int, Dict[str, Any]] = {}
        self.edges: Dict[Tuple[int, int, str], Dict[str, Any]] = {} # (from_node, to_node, action_name) -> EdgeData

    def _state_hash(self, latent_state: np.ndarray) -> int:
        """Discretizes continuous latent state into a unique signature."""
        binned = (latent_state * 2.0).astype(np.int32)
        return int(hash(binned.tobytes()))

    def record_transition(
        self,
        from_latent: np.ndarray,
        action_name: str,
        to_latent: np.ndarray,
        consequence_delta: float,
        success: bool,
    ):
        src_id = self._state_hash(from_latent)
        dst_id = self._state_hash(to_latent)

        # 1. Update / create nodes
        if src_id not in self.nodes:
            if len(self.nodes) >= self.max_nodes:
                oldest = min(self.nodes.keys(), key=lambda k: self.nodes[k].get("visits", 0))
                del self.nodes[oldest]
                self.edges = {k: v for k, v in self.edges.items() if k[0] != oldest and k[1] != oldest}
            self.nodes[src_id] = {"id": src_id, "visits": 1}
        else:
            self.nodes[src_id]["visits"] += 1

        if dst_id not in self.nodes:
            if len(self.nodes) < self.max_nodes:
                self.nodes[dst_id] = {"id": dst_id, "visits": 1}

        # 2. Update edge
        edge_key = (src_id, dst_id, action_name)
        if edge_key not in self.edges:
            self.edges[edge_key] = {
                "from": src_id,
                "to": dst_id,
                "action": action_name,
                "traversals": 1,
                "success_count": 1 if success else 0,
                "mean_consequence": float(consequence_delta),
            }
        else:
            e = self.edges[edge_key]
            e["traversals"] += 1
            if success:
                e["success_count"] += 1
            e["mean_consequence"] = 0.9 * e["mean_consequence"] + 0.1 * float(consequence_delta)

    def get_candidate_actions(self, latent_state: np.ndarray) -> List[Tuple[str, float, float]]:
        """Returns known outgoing actions from current state signature: (action_name, success_rate, consequence)."""
        node_id = self._state_hash(latent_state)
        candidates = []
        for (src, dst, act), edge in self.edges.items():
            if src == node_id:
                success_rate = edge["success_count"] / max(1, edge["traversals"])
                candidates.append((act, success_rate, edge["mean_consequence"]))
        return candidates

    def num_nodes(self) -> int:
        return len(self.nodes)

    def num_edges(self) -> int:
        return len(self.edges)
