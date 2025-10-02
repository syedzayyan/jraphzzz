from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax.tree_util as tree
from jraph import utils
from .types import (
    GATAttentionQueryFn,
    GATAttentionLogitFn,
    GATNodeUpdateFn,
    NodeFeatures,
    AggregateEdgesToNodesFn,
    AggregateNodesToGlobalsFn,
    Globals,
)
from .base import GraphNetwork


def GAT(
    attention_query_fn: GATAttentionQueryFn,
    attention_logit_fn: GATAttentionLogitFn,
    node_update_fn: Optional[GATNodeUpdateFn] = None,
):
    """Returns a method that applies a Graph Attention Network layer.

    Graph Attention message passing as described in
    https://arxiv.org/abs/1710.10903. This model expects node features as a
    jnp.array, may use edge features for computing attention weights, and
    ignore global features. It does not support nests.

    NOTE: this implementation assumes that the input graph has self edges. To
    recover the behavior of the referenced paper, please add self edges.

    Args:
      attention_query_fn: function that generates attention queries
        from sender node features.
      attention_logit_fn: function that converts attention queries into logits for
        softmax attention.
      node_update_fn: function that updates the aggregated messages. If None,
        will apply leaky relu and concatenate (if using multi-head attention).

    Returns:
      A function that applies a Graph Attention layer.
    """
    # pylint: disable=g-long-lambda
    if node_update_fn is None:
        # By default, apply the leaky relu and then concatenate the heads on the
        # feature axis.
        def node_update_fn(x):
            return jnp.reshape(jax.nn.leaky_relu(x), (x.shape[0], -1))

    def _ApplyGAT(graph):
        """Applies a Graph Attention layer."""
        nodes, edges, receivers, senders, _, _, _ = graph
        # Equivalent to the sum of n_node, but statically known.
        try:
            sum_n_node = nodes.shape[0]
        except IndexError:
            raise IndexError("GAT requires node features")  # pylint: disable=raise-missing-from

        # First pass nodes through the node updater.
        nodes = attention_query_fn(nodes)
        # pylint: disable=g-long-lambda
        # We compute the softmax logits using a function that takes the
        # embedded sender and receiver attributes.
        sent_attributes = nodes[senders]
        received_attributes = nodes[receivers]
        softmax_logits = attention_logit_fn(sent_attributes, received_attributes, edges)

        # Compute the softmax weights on the entire tree.
        weights = utils.segment_softmax(
            softmax_logits, segment_ids=receivers, num_segments=sum_n_node
        )
        # Apply weights
        messages = sent_attributes * weights
        # Aggregate messages to nodes.
        nodes = utils.segment_sum(messages, receivers, num_segments=sum_n_node)

        # Apply an update function to the aggregated messages.
        nodes = node_update_fn(nodes)
        return graph._replace(nodes=nodes)

    # pylint: enable=g-long-lambda
    return _ApplyGAT


def GraphConvolution(
    update_node_fn: Callable[[NodeFeatures], NodeFeatures],
    aggregate_nodes_fn: AggregateEdgesToNodesFn = utils.segment_sum,
    add_self_edges: bool = False,
    symmetric_normalization: bool = True,
):
    """Returns a method that applies a Graph Convolution layer.

    Graph Convolutional layer as in https://arxiv.org/abs/1609.02907,

    NOTE: This implementation does not add an activation after aggregation.
    If you are stacking layers, you may want to add an activation between
    each layer.

    Args:
      update_node_fn: function used to update the nodes. In the paper a single
        layer MLP is used.
      aggregate_nodes_fn: function used to aggregates the sender nodes.
      add_self_edges: whether to add self edges to nodes in the graph as in the
        paper definition of GCN. Defaults to False.
      symmetric_normalization: whether to use symmetric normalization. Defaults
        to True. Note that to replicate the fomula of the linked paper, the
        adjacency matrix must be symmetric. If the adjacency matrix is not
        symmetric the data is prenormalised by the sender degree matrix and post
        normalised by the receiver degree matrix.

    Returns:
      A method that applies a Graph Convolution layer.
    """

    def _ApplyGCN(graph):
        """Applies a Graph Convolution layer."""
        nodes, _, receivers, senders, _, _, _ = graph

        # First pass nodes through the node updater.
        nodes = update_node_fn(nodes)
        # Equivalent to jnp.sum(n_node), but jittable
        total_num_nodes = tree.tree_leaves(nodes)[0].shape[0]
        if add_self_edges:
            # We add self edges to the senders and receivers so that each node
            # includes itself in aggregation.
            # In principle, a `GraphsTuple` should partition by n_edge, but in
            # this case it is not required since a GCN is agnostic to whether
            # the `GraphsTuple` is a batch of graphs or a single large graph.
            conv_receivers = jnp.concatenate(
                (receivers, jnp.arange(total_num_nodes)), axis=0
            )
            conv_senders = jnp.concatenate(
                (senders, jnp.arange(total_num_nodes)), axis=0
            )
        else:
            conv_senders = senders
            conv_receivers = receivers

        # pylint: disable=g-long-lambda
        if symmetric_normalization:
            # Calculate the normalization values.
            def count_edges(x):
                return utils.segment_sum(
                    jnp.ones_like(conv_senders), x, total_num_nodes
                )

            sender_degree = count_edges(conv_senders)
            receiver_degree = count_edges(conv_receivers)

            # Pre normalize by sqrt sender degree.
            # Avoid dividing by 0 by taking maximum of (degree, 1).
            nodes = tree.tree_map(
                lambda x: x * jax.lax.rsqrt(jnp.maximum(sender_degree, 1.0))[:, None],
                nodes,
            )
            # Aggregate the pre normalized nodes.
            nodes = tree.tree_map(
                lambda x: aggregate_nodes_fn(
                    x[conv_senders], conv_receivers, total_num_nodes
                ),
                nodes,
            )
            # Post normalize by sqrt receiver degree.
            # Avoid dividing by 0 by taking maximum of (degree, 1).
            nodes = tree.tree_map(
                lambda x: (
                    x * jax.lax.rsqrt(jnp.maximum(receiver_degree, 1.0))[:, None]
                ),
                nodes,
            )
        else:
            nodes = tree.tree_map(
                lambda x: aggregate_nodes_fn(
                    x[conv_senders], conv_receivers, total_num_nodes
                ),
                nodes,
            )
        # pylint: enable=g-long-lambda
        return graph._replace(nodes=nodes)

    return _ApplyGCN


def DeepSets(
    update_node_fn: Callable[[NodeFeatures, Globals], NodeFeatures],
    update_global_fn: Callable[[NodeFeatures], Globals],
    aggregate_nodes_for_globals_fn: AggregateNodesToGlobalsFn = utils.segment_sum,
):
    """Returns a method that applies a DeepSets layer.

    Implementation for the model described in https://arxiv.org/abs/1703.06114
    (M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Poczos, R. Salakhutdinov, A. Smola).
    See also PointNet (https://arxiv.org/abs/1612.00593, C. Qi, H. Su, K. Mo,
    L. J. Guibas) for a related model.

    This module operates on sets, which can be thought of as graphs without
    edges. The nodes features are first updated based on their value and the
    globals features, and new globals features are then computed based on the
    updated nodes features.

    Args:
      update_node_fn: function used to update the nodes.
      update_global_fn: function used to update the globals.
      aggregate_nodes_for_globals_fn: function used to aggregate the nodes for the
        globals.
    """
    # DeepSets can be implemented with a GraphNetwork, with just a node
    # update function that takes nodes and globals, and a global update
    # function based on the updated node features.
    return GraphNetwork(
        update_edge_fn=None,
        update_node_fn=lambda n, s, r, g: update_node_fn(n, g),
        update_global_fn=lambda n, e, g: update_global_fn(n),
        aggregate_nodes_for_globals_fn=aggregate_nodes_for_globals_fn,
    )
