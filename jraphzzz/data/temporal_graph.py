from .graph import GraphsTuple
from typing import Optional

import jax.numpy as jnp
# ============================================================================
# Core Data Structure
# ============================================================================

class TemporalGraphsTuple(GraphsTuple):
    '''
    Inherits from GraphsTuple to include temporal information.
    additional fields:
    -----------
    node_times: jnp.ndarray
        Timestamps associated with each node.
    edge_times: jnp.ndarray
        Timestamps associated with each edge.
    -----------
    '''
    times: jnp.ndarray  # [total_num_edges] timestamps
    memory: Optional[jnp.ndarray] = None  # [total_num_nodes] memory states
    