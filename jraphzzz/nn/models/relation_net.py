# Copyright 2020 DeepMind Technologies Limited.


# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A library of Graph Neural Network models."""

from typing import Callable, Optional
from jraphzzz.utils import utils
from jraphzzz.nn.base import GraphNetwork
from jraphzzz.nn.types import (
    EmbedEdgeFn,
    EmbedNodeFn,
    EmbedGlobalFn,
    SenderFeatures,
    ReceiverFeatures,
    EdgeFeatures,
    NodeFeatures,
    AggregateEdgesToGlobalsFn,
)


def GraphMapFeatures(
    embed_edge_fn: Optional[EmbedEdgeFn] = None,
    embed_node_fn: Optional[EmbedNodeFn] = None,
    embed_global_fn: Optional[EmbedGlobalFn] = None,
):
    """Returns function which embeds the components of a graph independently.

    Args:
      embed_edge_fn: function used to embed the edges.
      embed_node_fn: function used to embed the nodes.
      embed_global_fn: function used to embed the globals.
    """

    def identity(x):
        return x

    embed_edges_fn = embed_edge_fn if embed_edge_fn else identity
    embed_nodes_fn = embed_node_fn if embed_node_fn else identity
    embed_global_fn = embed_global_fn if embed_global_fn else identity

    def Embed(graphs_tuple):
        return graphs_tuple._replace(
            nodes=embed_nodes_fn(graphs_tuple.nodes),
            edges=embed_edges_fn(graphs_tuple.edges),
            globals=embed_global_fn(graphs_tuple.globals),
        )

    return Embed


def RelationNetwork(
    update_edge_fn: Callable[[SenderFeatures, ReceiverFeatures], EdgeFeatures],
    update_global_fn: Callable[[EdgeFeatures], NodeFeatures],
    aggregate_edges_for_globals_fn: AggregateEdgesToGlobalsFn = utils.segment_sum,
):
    """Returns a method that applies a Relation Network.

    See https://arxiv.org/abs/1706.01427 for more details.

    This implementation has one more argument, `aggregate_edges_for_globals_fn`,
    which changes how edge features are aggregated. The paper uses the default -
    `utils.segment_sum`.

    Args:
      update_edge_fn: function used to update the edges.
      update_global_fn: function used to update the globals.
      aggregate_edges_for_globals_fn: function used to aggregate the edges for the
        globals.
    """
    return GraphNetwork(
        update_edge_fn=lambda e, s, r, g: update_edge_fn(s, r),
        update_node_fn=None,
        update_global_fn=lambda n, e, g: update_global_fn(e),
        attention_logit_fn=None,
        aggregate_edges_for_globals_fn=aggregate_edges_for_globals_fn,
    )
