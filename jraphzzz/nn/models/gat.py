# Copyright 2020 DeepMind Technologies Limited.


# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

from typing import Optional
from jraphzzz.utils import utils
from jraphzzz.nn.base import GraphNetwork
from jraphzzz.nn.types import (
    GNUpdateEdgeFn,
    GNUpdateNodeFn,
    AttentionLogitFn,
    AttentionReduceFn,
    GNUpdateGlobalFn,
    AggregateEdgesToNodesFn,
    AggregateNodesToGlobalsFn,
    AggregateEdgesToGlobalsFn,
)


def GraphNetGAT(
    update_edge_fn: GNUpdateEdgeFn,
    update_node_fn: GNUpdateNodeFn,
    attention_logit_fn: AttentionLogitFn,
    attention_reduce_fn: AttentionReduceFn,
    update_global_fn: Optional[GNUpdateGlobalFn] = None,
    aggregate_edges_for_nodes_fn: AggregateEdgesToNodesFn = utils.segment_sum,
    aggregate_nodes_for_globals_fn: AggregateNodesToGlobalsFn = utils.segment_sum,
    aggregate_edges_for_globals_fn: AggregateEdgesToGlobalsFn = utils.segment_sum,
):
    """Returns a method that applies a GraphNet with attention on edge features.

    Args:
      update_edge_fn: function used to update the edges.
      update_node_fn: function used to update the nodes.
      attention_logit_fn: function used to calculate the attention weights.
      attention_reduce_fn: function used to apply attention weights to the edge
        features.
      update_global_fn: function used to update the globals or None to deactivate
        globals updates.
      aggregate_edges_for_nodes_fn: function used to aggregate attention-weighted
        messages to each node.
      aggregate_nodes_for_globals_fn: function used to aggregate the nodes for the
        globals.
      aggregate_edges_for_globals_fn: function used to aggregate
        attention-weighted edges for the globals.

    Returns:
      A function that applies a GraphNet Graph Attention layer.
    """
    if (attention_logit_fn is None) or (attention_reduce_fn is None):
        raise ValueError(
            (
                "`None` value not supported for `attention_logit_fn` or "
                "`attention_reduce_fn` in a Graph Attention network."
            )
        )
    return GraphNetwork(
        update_edge_fn=update_edge_fn,
        update_node_fn=update_node_fn,
        update_global_fn=update_global_fn,
        attention_logit_fn=attention_logit_fn,
        attention_reduce_fn=attention_reduce_fn,
        aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn,
        aggregate_nodes_for_globals_fn=aggregate_nodes_for_globals_fn,
        aggregate_edges_for_globals_fn=aggregate_edges_for_globals_fn,
    )
