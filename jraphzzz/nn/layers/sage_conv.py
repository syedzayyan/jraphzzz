from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jraphzzz.utils import utils
from ...data.graph import GraphsTuple
from ..types import (
    NodeFeatures,
    AggregateEdgesToNodesFn,
)


def SageConv(
    update_node_fn_lft: Callable[[NodeFeatures], NodeFeatures],
    update_node_fn_rght: Callable[[NodeFeatures], NodeFeatures],
    aggregate_nodes_fn: AggregateEdgesToNodesFn = utils.segment_mean,
    update_node_fn_project: Optional[Callable[[NodeFeatures], NodeFeatures]] = None,
    add_self_edges: bool = False,
    normalize: bool = False,
    root_weight: bool = True,
):
    """
    The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    The node features are computed as follows:
    
    .. math::
        \mathbf{{h}}_{u}^{k}=\mathbf{W}_1 \cdot \mathbf{{h}}_{u}^{k-1} + \mathbf{W}_2 \cdot \text{CONCAT}(\mathbf{{h}}_{u}^{k-1}, \mathbf{{h}}_{\mathcal{N}(u)}^{k})

    having:
    
    .. math::
        \mathbf{{h}}_{\mathcal{N}(u)}^{k} = \text{AGGREGATE}(\{\mathbf{{h}}_{v}^{k-1}, \forall v \in \mathcal{N}(u)\})

    and :math:`AGGREGATE` being an aggregation operator (i.e. :obj:`"mean"`, :obj:`"max"`, or :obj:`"sum"`)

    If :obj:`project = True`, then :math:`\mathbf{{h}}_{u}^{k-1}` is first projected via:

    .. math::
        \mathbf{{h}}_{v}^{k-1}=\text{ReLU}(\mathbf{W}_3 \mathbf{{h}}_{v}^{k-1} + \mathbf{b})

    Args:
        normalize (bool, optional): If :obj:`True`, output features
            are :math:`\ell_2`-normalized.
            (default: :obj:`False`)
        root_weight (bool, optional): If :obj:`False` the linear transformed features
            :math:`\mathbf{W}_1 \cdot \mathbf{{h}}_{u}^{k-1}` are not added to the output features.
            (default: :obj:`True`)
        update_node_fn_project (function, optional): If :obj:`True`, neighbour features are projected before aggregation as
            explained above.
            (default: :obj:`False`)
        bias (bool, optional): If :obj:`True`, the layer will add
            an additive bias to the output.
            (default: :obj:`True`)
    """

    def _ApplySageConv(graph: GraphsTuple):
        """Applies a Graph Convolution layer."""
        nodes, _, receivers, senders, _, _, _ = graph
        
        total_num_nodes = jax.tree.leaves(nodes)[0].shape[0]
        
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

        def count_edges(x):
            return utils.segment_sum(
                jnp.ones_like(conv_senders), x, total_num_nodes
            )
        
        if update_node_fn_project is not None:
            h = update_node_fn_project(nodes)
        else:
            h = nodes

        # Pre normalize by sqrt sender degree.
        # Avoid dividing by 0 by taking maximum of (degree, 1).
        if normalize:
            sender_degree = count_edges(conv_senders)
            h = jax.tree.map(
                lambda x: x * jax.lax.rsqrt(jnp.maximum(sender_degree, 1.0))[:, None],
                h,
            )

        h = jax.tree.map(lambda x: aggregate_nodes_fn(x[conv_senders], conv_receivers, total_num_nodes), h)
        h = jnp.concatenate((nodes, h), axis=1)

        out = update_node_fn_lft(h)

        if root_weight:
            out = out + update_node_fn_rght(nodes)

        if normalize:
            # out /= jnp.linalg.norm(out, ord=2, axis=-1, keepdims=True)
            # Post normalize by sqrt receiver degree.
            # Avoid dividing by 0 by taking maximum of (degree, 1).
            receiver_degree = count_edges(conv_receivers)
            out = jax.tree.map(
                lambda x: (
                    x * jax.lax.rsqrt(jnp.maximum(receiver_degree, 1.0))[:, None]
                ),
                out,
            )

        return graph._replace(nodes=out)

    return _ApplySageConv
