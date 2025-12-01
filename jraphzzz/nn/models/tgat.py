import jax
import jax.numpy as jnp
from typing import Optional, Callable, NamedTuple
from ...data.temporal_graph import TemporalGraphsTuple
from ...utils import segment_sum, batch


# ============================================================================
# Neighbor Finding (Pure Functions)
# ============================================================================

class TemporalAdjacency(NamedTuple):
    """Temporal adjacency list representation (immutable)."""
    # CSR-like format for efficient neighbor lookup
    node_offsets: jnp.ndarray  # [num_nodes + 1] offsets into neighbors array
    neighbor_nodes: jnp.ndarray  # [num_interactions] flattened neighbor node ids
    neighbor_edges: jnp.ndarray  # [num_interactions] flattened edge ids
    neighbor_times: jnp.ndarray  # [num_interactions] flattened timestamps
    num_nodes: int


def build_temporal_adjacency(
    senders: jnp.ndarray,
    receivers: jnp.ndarray,
    edge_times: jnp.ndarray,
    edge_ids: jnp.ndarray,
    num_nodes: int
) -> TemporalAdjacency:
    """
    Build temporal adjacency structure from edge list.
    Pure function - no side effects.
    """
    # Sort by sender, then by time (descending for recency)
    sort_idx = jnp.lexsort((-edge_times, senders))
    
    sorted_senders = senders[sort_idx]
    sorted_receivers = receivers[sort_idx]
    sorted_times = edge_times[sort_idx]
    sorted_edge_ids = edge_ids[sort_idx]
    
    # Compute offsets (CSR format)
    node_offsets = jnp.zeros(num_nodes + 1, dtype=jnp.int32)
    for i in range(len(sorted_senders)):
        node_offsets = node_offsets.at[sorted_senders[i] + 1].add(1)
    node_offsets = jnp.cumsum(node_offsets)
    
    return TemporalAdjacency(
        node_offsets=node_offsets,
        neighbor_nodes=sorted_receivers,
        neighbor_edges=sorted_edge_ids,
        neighbor_times=sorted_times,
        num_nodes=num_nodes
    )


def sample_temporal_neighbors(
    adj: TemporalAdjacency,
    query_nodes: jnp.ndarray,
    cut_times: jnp.ndarray,
    num_neighbors: int,
    sampling_strategy: str = 'recent'
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Sample temporal neighbors before cutoff times.
    Pure function - deterministic or uses explicit random key.
    
    Args:
        adj: Temporal adjacency structure
        query_nodes: [batch_size] node indices to query
        cut_times: [batch_size] cutoff timestamps
        num_neighbors: number of neighbors to sample
        sampling_strategy: 'recent', 'uniform', or 'importance'
        
    Returns:
        neighbor_nodes: [batch_size, num_neighbors]
        neighbor_edges: [batch_size, num_neighbors]
        neighbor_times: [batch_size, num_neighbors]
    """    
    def sample_node_neighbors(node_idx, cut_time):
        """Sample neighbors for a single node."""
        start_idx = adj.node_offsets[node_idx]
        end_idx = adj.node_offsets[node_idx + 1]
        
        # Get all neighbors and filter by time
        node_neighbors = adj.neighbor_nodes[start_idx:end_idx]
        edge_ids = adj.neighbor_edges[start_idx:end_idx]
        times = adj.neighbor_times[start_idx:end_idx]
        
        # Filter by cutoff time
        valid_mask = times < cut_time
        valid_count = jnp.sum(valid_mask)
        
        # Take most recent num_neighbors (already sorted by time desc)
        take_count = jnp.minimum(valid_count, num_neighbors)
        
        # Pad if needed
        padded_neighbors = jnp.pad(
            node_neighbors[:take_count],
            (0, num_neighbors - take_count),
            constant_values=0
        )
        padded_edges = jnp.pad(
            edge_ids[:take_count],
            (0, num_neighbors - take_count),
            constant_values=0
        )
        padded_times = jnp.pad(
            times[:take_count],
            (0, num_neighbors - take_count),
            constant_values=0.0
        )
        
        return padded_neighbors, padded_edges, padded_times
    
    # Vectorize over batch
    neighbors, edges, times = jax.vmap(sample_node_neighbors)(
        query_nodes, cut_times
    )
    
    return neighbors, edges, times


# ============================================================================
# Time Encoding (Pure Functions)
# ============================================================================

def harmonic_time_encoding(
    timestamps: jnp.ndarray,
    dim: int,
    freq_scale: float = 10.0,
    max_freq_power: float = 9.0
) -> jnp.ndarray:
    """
    Sinusoidal time encoding (no learnable parameters).
    
    Args:
        timestamps: [...] timestamps to encode
        dim: encoding dimension
        freq_scale: base frequency scale
        max_freq_power: range of frequencies (10^0 to 10^max_freq_power)
        
    Returns:
        encoded: [..., dim] time encodings
    """
    # Create frequency basis
    basis_freq = 1.0 / (freq_scale ** jnp.linspace(0, max_freq_power, dim))
    
    # Broadcast and compute
    ts_expanded = jnp.expand_dims(timestamps, -1)
    map_ts = ts_expanded * basis_freq
    
    return jnp.cos(map_ts)


def relative_time_encoding(
    query_times: jnp.ndarray,
    event_times: jnp.ndarray,
    dim: int
) -> jnp.ndarray:
    """
    Encode relative time differences.
    
    Args:
        query_times: [...] query timestamps
        event_times: [...] event timestamps
        dim: encoding dimension
        
    Returns:
        encoded: [..., dim] relative time encodings
    """
    time_deltas = query_times - event_times
    return harmonic_time_encoding(time_deltas, dim)


# ============================================================================
# Temporal Graph Operations (Pure Functions, like jraph.GraphMapFeatures)
# ============================================================================

def temporal_graph_map_features(
    node_fn: Optional[Callable] = None,
    edge_fn: Optional[Callable] = None,
    global_fn: Optional[Callable] = None,
) -> Callable[[TemporalGraphsTuple], TemporalGraphsTuple]:
    """
    Returns function that applies transformations to temporal graph features.
    Analogous to jraph.GraphMapFeatures but includes temporal context.
    
    Example:
        # Transform nodes with time awareness
        def node_update(nodes, node_times):
            time_features = harmonic_time_encoding(node_times, 32)
            return jnp.concatenate([nodes, time_features], axis=-1)
        
        update_fn = temporal_graph_map_features(node_fn=node_update)
        new_graph = update_fn(graph)
    """
    def _apply(graph: TemporalGraphsTuple) -> TemporalGraphsTuple:
        updated = {}
        
        if node_fn is not None:
            updated['nodes'] = node_fn(graph.nodes, graph.node_times)
        
        if edge_fn is not None:
            # Provide sender/receiver nodes and times for context
            sender_nodes = graph.nodes[graph.senders]
            receiver_nodes = graph.nodes[graph.receivers]
            sender_times = graph.node_times[graph.senders]
            receiver_times = graph.node_times[graph.receivers]
            
            updated['edges'] = edge_fn(
                graph.edges,
                graph.edge_times,
                sender_nodes,
                receiver_nodes,
                sender_times,
                receiver_times
            )
        
        if global_fn is not None:
            updated['globals'] = global_fn(graph.globals, graph)
        
        return graph._replace(**updated)
    
    return _apply


def temporal_aggregate_neighbors(
    graph: TemporalGraphsTuple,
    adj: TemporalAdjacency,
    aggregation_fn: Callable,
    num_neighbors: int = 20,
    time_encoding_dim: int = 32
) -> jnp.ndarray:
    """
    Aggregate temporal neighborhood information for each node.
    Pure function version - all computations are deterministic.
    
    Args:
        graph: Input temporal graph
        adj: Precomputed temporal adjacency structure
        aggregation_fn: Function(node_feat, neighbor_feats, time_deltas) -> aggregated
        num_neighbors: Number of neighbors to sample
        time_encoding_dim: Dimension for time encoding
        
    Returns:
        aggregated_features: [num_nodes, feat_dim] aggregated node features
    """
    num_nodes = graph.nodes.shape[0]
    node_indices = jnp.arange(num_nodes)
    
    # Sample neighbors for all nodes
    neighbor_nodes, neighbor_edges, neighbor_times = sample_temporal_neighbors(
        adj,
        node_indices,
        graph.node_times,
        num_neighbors
    )
    
    # Gather neighbor features
    neighbor_features = graph.nodes[neighbor_nodes]  # [num_nodes, num_neighbors, feat_dim]
    neighbor_edge_features = graph.edges[neighbor_edges]
    
    # Compute time deltas and encode
    time_deltas = graph.node_times[:, jnp.newaxis] - neighbor_times
    time_encodings = harmonic_time_encoding(time_deltas, time_encoding_dim)
    
    # Create mask for padding
    mask = neighbor_nodes == 0
    
    # Apply aggregation (vectorized over nodes)
    def aggregate_single(node_feat, node_time, neigh_feats, edge_feats, time_enc, m):
        return aggregation_fn(node_feat, node_time, neigh_feats, edge_feats, time_enc, m)
    
    aggregated = jax.vmap(aggregate_single)(
        graph.nodes,
        graph.node_times,
        neighbor_features,
        neighbor_edge_features,
        time_encodings,
        mask
    )
    
    return aggregated


def temporal_message_passing(
    graph: TemporalGraphsTuple,
    message_fn: Callable,
    aggregation_fn: Callable = segment_sum,
    use_time_in_message: bool = True
) -> TemporalGraphsTuple:
    """
    General temporal message passing operation.
    Analogous to jraph's InteractionNetwork pattern.
    
    Args:
        graph: Input temporal graph
        message_fn: Computes messages from edges
        aggregation_fn: Aggregates messages (default: segment_sum)
        use_time_in_message: Whether to include time in messages
        
    Returns:
        Updated graph with aggregated messages
    """
    # Compute messages
    sender_features = graph.nodes[graph.senders]
    receiver_features = graph.nodes[graph.receivers]
    
    if use_time_in_message:
        # Include temporal context in messages
        time_deltas = graph.node_times[graph.receivers] - graph.edge_times
        time_encodings = harmonic_time_encoding(
            time_deltas,
            dim=graph.nodes.shape[-1]
        )
        
        messages = message_fn(
            sender_features,
            receiver_features,
            graph.edges,
            time_encodings
        )
    else:
        messages = message_fn(
            sender_features,
            receiver_features,
            graph.edges
        )
    
    # Aggregate messages to nodes
    num_nodes = graph.nodes.shape[0]
    aggregated = aggregation_fn(
        messages,
        graph.receivers,
        num_segments=num_nodes
    )
    
    return graph._replace(nodes=aggregated)


# ============================================================================
# Multi-hop Temporal Convolution (Pure Functional)
# ============================================================================

def temporal_convolution_layer(
    graph: TemporalGraphsTuple,
    adj: TemporalAdjacency,
    query_nodes: jnp.ndarray,
    query_times: jnp.ndarray,
    aggregation_fn: Callable,
    num_neighbors: int = 20
) -> jnp.ndarray:
    """
    Single layer of temporal convolution.
    Pure function - no state.
    
    Args:
        graph: Temporal graph with current layer features
        adj: Temporal adjacency structure
        query_nodes: [batch] nodes to compute features for
        query_times: [batch] query timestamps
        aggregation_fn: Aggregation function
        num_neighbors: Number of neighbors to sample
        
    Returns:
        features: [batch, feat_dim] node features
    """
    # Sample neighbors
    neighbor_nodes, neighbor_edges, neighbor_times = sample_temporal_neighbors(
        adj,
        query_nodes,
        query_times,
        num_neighbors
    )
    
    # Gather features
    query_features = graph.nodes[query_nodes]
    neighbor_features = graph.nodes[neighbor_nodes]
    edge_features = graph.edges[neighbor_edges]
    
    # Encode time
    time_deltas = query_times[:, jnp.newaxis] - neighbor_times
    time_encodings = harmonic_time_encoding(time_deltas, graph.nodes.shape[-1])
    
    # Mask for padding
    mask = neighbor_nodes == 0
    
    # Aggregate
    aggregated = jax.vmap(aggregation_fn)(
        query_features,
        neighbor_features,
        edge_features,
        time_encodings,
        mask
    )
    
    return aggregated


def multi_hop_temporal_conv(
    graph: TemporalGraphsTuple,
    adj: TemporalAdjacency,
    query_nodes: jnp.ndarray,
    query_times: jnp.ndarray,
    aggregation_fns: list[Callable],
    num_neighbors: int = 20
) -> jnp.ndarray:
    """
    Multi-hop temporal convolution (recursive neighborhood aggregation).
    Pure functional implementation.
    
    Args:
        graph: Input temporal graph
        adj: Temporal adjacency
        query_nodes: [batch] query node indices
        query_times: [batch] query timestamps
        aggregation_fns: List of aggregation functions (one per layer)
        num_neighbors: Neighbors per hop
        
    Returns:
        embeddings: [batch, feat_dim] final node embeddings
    """
    num_layers = len(aggregation_fns)
    
    def recursive_conv(layer_idx, curr_nodes, curr_times):
        """Recursively compute features at each layer."""
        if layer_idx == 0:
            # Base case: return raw features
            return graph.nodes[curr_nodes]
        
        # Sample neighbors
        neighbor_nodes, neighbor_edges, neighbor_times = sample_temporal_neighbors(
            adj, curr_nodes, curr_times, num_neighbors
        )
        
        # Recursively get neighbor features from previous layer
        batch_size = len(curr_nodes)
        neighbor_nodes_flat = neighbor_nodes.reshape(-1)
        neighbor_times_flat = neighbor_times.reshape(-1)
        
        neighbor_features = recursive_conv(
            layer_idx - 1,
            neighbor_nodes_flat,
            neighbor_times_flat
        )
        neighbor_features = neighbor_features.reshape(
            batch_size, num_neighbors, -1
        )
        
        # Get current node features
        curr_features = recursive_conv(layer_idx - 1, curr_nodes, curr_times)
        
        # Gather edge features and encode time
        edge_features = graph.edges[neighbor_edges]
        time_deltas = curr_times[:, jnp.newaxis] - neighbor_times
        time_encodings = harmonic_time_encoding(time_deltas, graph.nodes.shape[-1])
        
        # Aggregate
        mask = neighbor_nodes == 0
        aggregation_fn = aggregation_fns[layer_idx - 1]
        
        aggregated = jax.vmap(aggregation_fn)(
            curr_features,
            neighbor_features,
            edge_features,
            time_encodings,
            mask
        )
        
        return aggregated
    
    return recursive_conv(num_layers, query_nodes, query_times)


# ============================================================================
# Utility Functions
# ============================================================================

def temporal_batch(graphs: list[TemporalGraphsTuple]) -> TemporalGraphsTuple:
    """
    Batch multiple temporal graphs (extends jraph.batch).
    """
    # Use jraph's batch for base GraphsTuple fields
    base_batched = batch([g._replace(node_times=None, edge_times=None) for g in graphs])
    
    # Concatenate temporal fields
    node_times = jnp.concatenate([g.node_times for g in graphs])
    edge_times = jnp.concatenate([g.edge_times for g in graphs])
    
    return TemporalGraphsTuple(
        nodes=base_batched.nodes,
        edges=base_batched.edges,
        receivers=base_batched.receivers,
        senders=base_batched.senders,
        globals=base_batched.globals,
        n_node=base_batched.n_node,
        n_edge=base_batched.n_edge,
        node_times=node_times,
        edge_times=edge_times
    )


def create_temporal_graph_from_events(
    node_features: jnp.ndarray,
    edge_features: jnp.ndarray,
    events: list[tuple[int, int, float, int]],  # (src, dst, time, edge_id)
) -> TemporalGraphsTuple:
    """
    Create TemporalGraphsTuple from event stream.
    Pure function - deterministic construction.
    """
    num_nodes = node_features.shape[0]
    
    senders = jnp.array([e[0] for e in events])
    receivers = jnp.array([e[1] for e in events])
    edge_times = jnp.array([e[2] for e in events])
    edge_indices = jnp.array([e[3] for e in events])
    
    # Compute node times (last interaction)
    node_times = jnp.zeros(num_nodes)
    for src, dst, time, _ in events:
        node_times = node_times.at[src].set(jnp.maximum(node_times[src], time))
        node_times = node_times.at[dst].set(jnp.maximum(node_times[dst], time))
    
    return TemporalGraphsTuple(
        nodes=node_features,
        edges=edge_features[edge_indices],
        senders=senders,
        receivers=receivers,
        globals=None,
        n_node=jnp.array([num_nodes]),
        n_edge=jnp.array([len(events)]),
        node_times=node_times,
        edge_times=edge_times
    )


# ============================================================================
# Example Usage (Framework Agnostic)
# ============================================================================

def simple_attention_aggregation(
    query_feat: jnp.ndarray,
    neighbor_feats: jnp.ndarray,
    edge_feats: jnp.ndarray,
    time_encodings: jnp.ndarray,
    mask: jnp.ndarray
) -> jnp.ndarray:
    """
    Simple attention-based aggregation (no learnable parameters).
    Pure function using only JAX operations.
    """
    # Combine features
    combined = jnp.concatenate([neighbor_feats, edge_feats, time_encodings], axis=-1)
    
    # Simple dot-product attention
    query_expanded = jnp.expand_dims(query_feat, 0)
    scores = jnp.sum(combined * query_expanded, axis=-1)
    
    # Mask and normalize
    scores = jnp.where(mask, -1e10, scores)
    attn_weights = jax.nn.softmax(scores)
    
    # Aggregate
    aggregated = jnp.sum(
        neighbor_feats * attn_weights[:, jnp.newaxis],
        axis=0
    )
    
    return aggregated
