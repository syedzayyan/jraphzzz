from typing import Callable

import jax
import jax.numpy as jnp
import jax.tree_util as tree
from jraphzzz.utils import utils
from ..types import (
    NodeFeatures,
    AggregateEdgesToNodesFn,
)


def ChebyshevConvolution(
    update_node_fn: Callable[[NodeFeatures], NodeFeatures],
    K: int = 3,
    aggregate_nodes_fn: AggregateEdgesToNodesFn = utils.segment_sum,
    add_self_edges: bool = True,
    symmetric_normalization: bool = True,
):
    """Returns a method that applies a Chebyshev (spectral) Graph Convolution.

    Implements ChebNet (Defferrard et al.) style convolution using the
    recurrence for Chebyshev polynomials T_k(L˜) x, where L˜ is the scaled
    (normalized) Laplacian. The layer computes T_0..T_{K-1} and concatenates
    them along the feature axis, then passes the concatenated features through
    `update_node_fn`.

    Args:
      update_node_fn: function applied to the concatenated Chebyshev features.
        Typically this will be a linear layer (Dense) mapping features*K -> out.
      K: highest order (will compute orders 0..K-1). Must be >= 1.
      aggregate_nodes_fn: function used to aggregate sender node features.
      add_self_edges: whether to add self edges before building normalized adjacency.
      symmetric_normalization: if True uses symmetric normalization D^{-1/2} A D^{-1/2}.
        If False uses left-pre / right-post normalization like the provided GCN.

    Returns:
      A method that applies a Chebyshev convolution and returns the updated graph.
    """

    if K < 1:
        raise ValueError("K must be >= 1")

    def _ApplyL(x, conv_senders, conv_receivers, total_num_nodes, sender_degree, receiver_degree):
        """Applies the normalized adjacency multiplication A_norm @ x, then returns Lx = x - A_norm x."""
        # pre-normalize by sender degree depending on symmetric flag
        if symmetric_normalization:
            x_pre = tree.tree_map(
                lambda v: v * jax.lax.rsqrt(jnp.maximum(sender_degree, 1.0))[:, None],
                x,
            )
            # aggregate
            agg = tree.tree_map(
                lambda v: aggregate_nodes_fn(v[conv_senders], conv_receivers, total_num_nodes),
                x_pre,
            )
            # post-normalize
            a_x = tree.tree_map(
                lambda v: v * jax.lax.rsqrt(jnp.maximum(receiver_degree, 1.0))[:, None],
                agg,
            )
        else:
            x_pre = tree.tree_map(
                lambda v: v / jnp.maximum(sender_degree, 1.0)[:, None],
                x,
            )
            agg = tree.tree_map(
                lambda v: aggregate_nodes_fn(v[conv_senders], conv_receivers, total_num_nodes),
                x_pre,
            )
            a_x = tree.tree_map(
                lambda v: v / jnp.maximum(receiver_degree, 1.0)[:, None],
                agg,
            )
        # Lx = x - A_norm x
        return tree.tree_map(lambda v, ax: v - ax, x, a_x)

    def _ApplyCheb(graph):
        nodes, _, receivers, senders, _, _, _ = graph

        # Determine total number of nodes in the (batched or single) GraphsTuple
        total_num_nodes = tree.tree_leaves(nodes)[0].shape[0]

        if add_self_edges:
            conv_receivers = jnp.concatenate((receivers, jnp.arange(total_num_nodes)), axis=0)
            conv_senders = jnp.concatenate((senders, jnp.arange(total_num_nodes)), axis=0)
        else:
            conv_senders = senders
            conv_receivers = receivers

        # compute degrees for normalization
        def count_edges(x):
            return utils.segment_sum(jnp.ones_like(conv_senders), x, total_num_nodes)

        sender_degree = count_edges(conv_senders)
        receiver_degree = count_edges(conv_receivers)

        # Chebyshev recurrence:
        # T_0(x) = x
        # T_1(x) = L x
        # T_k(x) = 2 L T_{k-1}(x) - T_{k-2}(x)
        T_list = []
        T0 = nodes
        T_list.append(T0)

        if K >= 2:
            L_T0 = _ApplyL(T0, conv_senders, conv_receivers, total_num_nodes, sender_degree, receiver_degree)
            T1 = L_T0
            T_list.append(T1)

            for k in range(2, K):
                L_prev = _ApplyL(T_list[-1], conv_senders, conv_receivers, total_num_nodes, sender_degree, receiver_degree)
                Tk = tree.tree_map(lambda a, b: 2.0 * a - b, L_prev, T_list[-2])
                T_list.append(Tk)

        # Concatenate along feature axis: need to concatenate each node feature array across K orders.
        concatenated = tree.tree_map(lambda *arrays: jnp.concatenate(arrays, axis=-1), *T_list)

        # Pass concatenated features through updater (e.g., linear layer)
        updated_nodes = update_node_fn(concatenated)

        return graph._replace(nodes=updated_nodes)

    return _ApplyCheb
