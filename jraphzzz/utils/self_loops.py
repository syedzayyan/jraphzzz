import jax.numpy as jnp
from typing import Optional, Tuple
from .utils import num_nodes as _num_nodes

def add_self_loops(
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        edge_attr: jnp.ndarray = None,
        fill_value = None,
        num_nodes: Optional[int] = None,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Adds self edges. Assumes self edges are not in the graph yet.

    Args:
        senders (jnp.ndarray): Array of sender node indices.
        receivers (jnp.ndarray): Array of receiver node indices.
        edge_attr (jnp.ndarray): Array of edge attributes or edge weights.
            (default: :obj:`None`)
        fill_value (float): Value to fill the edge features with.
            (default: :obj:`None`)
        num_nodes (int): Total number of nodes in the graph.
            (default: :obj:`None`)
    """
    N = _num_nodes(receivers, senders, num_nodes)

    loop_index = jnp.arange(N)
    loop_index = jnp.expand_dims(loop_index, axis=0)
    loop_index = jnp.tile(loop_index, (2, 1))

    if edge_attr is not None:
        if fill_value is None:
            shape = (N, ) + edge_attr.shape[1:]
            loop_attr = jnp.full(shape, 1.0)
        elif isinstance(fill_value, (int, float)):
            shape = (N,) + edge_attr.shape[1:]
            loop_attr = jnp.full(shape, fill_value)
        elif isinstance(fill_value, jnp.ndarray):
            loop_attr = edge_attr
            if edge_attr.ndim != loop_attr.ndim:
                loop_attr = jnp.expand_dims(loop_attr, axis=0)
            sizes = [N] + [1] * (edge_attr.ndim - 1)
            loop_attr = jnp.tile(loop_attr, sizes)
        elif isinstance(fill_value, str):
            #TODO: Implement scatter-like function or use _segment_update from jax.ops
            raise NotImplementedError("add_self_loops does not yet support string fill values.")
        else:
            raise AttributeError("Provided 'fill_value' values are not supported.")

        edge_attr = jnp.concatenate((edge_attr, loop_attr), axis=0)

    senders = jnp.concatenate((senders, loop_index[0]), axis=0)
    receivers = jnp.concatenate((receivers, loop_index[1]), axis=0)
    return senders, receivers, edge_attr

def remove_self_loops(senders: jnp.ndarray, receivers: jnp.ndarray, edge_weight: jnp.ndarray):
    """
    Removes self loops from a graph.
    """
    mask = senders != receivers
    senders = senders[mask]
    receivers = receivers[mask]
    if edge_weight is not None:
        edge_weight = edge_weight[mask]
    return senders, receivers, edge_weight