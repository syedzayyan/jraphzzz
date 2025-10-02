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

import functools
from typing import Any, Callable, Iterable, Mapping, Optional, Union

import jax
import jax.numpy as jnp
import jax.tree_util as tree
from jraph.data import graph as gn_graph
from jraph import utils



# def InteractionNetwork(
#     update_edge_fn: InteractionUpdateEdgeFn,
#     update_node_fn: Union[InteractionUpdateNodeFn,
#                           InteractionUpdateNodeFnNoSentEdges],
#     aggregate_edges_for_nodes_fn: AggregateEdgesToNodesFn = utils.segment_sum,
#     include_sent_messages_in_node_update: bool = False):
#   """Returns a method that applies a configured InteractionNetwork.

#   An interaction network computes interactions on the edges based on the
#   previous edges features, and on the features of the nodes sending into those
#   edges. It then updates the nodes based on the incoming updated edges.
#   See https://arxiv.org/abs/1612.00222 for more details.

#   This implementation adds an option not in https://arxiv.org/abs/1612.00222,
#   which is to include edge features for which a node is a sender in the
#   arguments to the node update function.

#   Args:
#     update_edge_fn: a function mapping a single edge update inputs to a single
#       edge feature.
#     update_node_fn: a function mapping a single node update input to a single
#       node feature.
#     aggregate_edges_for_nodes_fn: function used to aggregate messages to each
#       node.
#     include_sent_messages_in_node_update: pass edge features for which a node is
#       a sender to the node update function.
#   """
#   # An InteractionNetwork is a GraphNetwork without globals features,
#   # so we implement the InteractionNetwork as a configured GraphNetwork.

#   # An InteractionNetwork edge function does not have global feature inputs,
#   # so we filter the passed global argument in the GraphNetwork.
#   wrapped_update_edge_fn = lambda e, s, r, g: update_edge_fn(e, s, r)

#   # Similarly, we wrap the update_node_fn to ensure only the expected
#   # arguments are passed to the Interaction net.
#   if include_sent_messages_in_node_update:
#     wrapped_update_node_fn = lambda n, s, r, g: update_node_fn(n, s, r)  # pytype: disable=wrong-arg-count
#   else:
#     wrapped_update_node_fn = lambda n, s, r, g: update_node_fn(n, r)  # pytype: disable=wrong-arg-count
#   return GraphNetwork(
#       update_edge_fn=wrapped_update_edge_fn,
#       update_node_fn=wrapped_update_node_fn,
#       aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn)





# def GraphMapFeatures(embed_edge_fn: Optional[EmbedEdgeFn] = None,
#                      embed_node_fn: Optional[EmbedNodeFn] = None,
#                      embed_global_fn: Optional[EmbedGlobalFn] = None):
#   """Returns function which embeds the components of a graph independently.

#   Args:
#     embed_edge_fn: function used to embed the edges.
#     embed_node_fn: function used to embed the nodes.
#     embed_global_fn: function used to embed the globals.
#   """
#   identity = lambda x: x
#   embed_edges_fn = embed_edge_fn if embed_edge_fn else identity
#   embed_nodes_fn = embed_node_fn if embed_node_fn else identity
#   embed_global_fn = embed_global_fn if embed_global_fn else identity

#   def Embed(graphs_tuple):
#     return graphs_tuple._replace(
#         nodes=embed_nodes_fn(graphs_tuple.nodes),
#         edges=embed_edges_fn(graphs_tuple.edges),
#         globals=embed_global_fn(graphs_tuple.globals))

#   return Embed


# def RelationNetwork(
#     update_edge_fn: Callable[[SenderFeatures, ReceiverFeatures], EdgeFeatures],
#     update_global_fn: Callable[[EdgeFeatures], NodeFeatures],
#     aggregate_edges_for_globals_fn:
#         AggregateEdgesToGlobalsFn = utils.segment_sum):
#   """Returns a method that applies a Relation Network.

#   See https://arxiv.org/abs/1706.01427 for more details.

#   This implementation has one more argument, `aggregate_edges_for_globals_fn`,
#   which changes how edge features are aggregated. The paper uses the default -
#   `utils.segment_sum`.

#   Args:
#     update_edge_fn: function used to update the edges.
#     update_global_fn: function used to update the globals.
#     aggregate_edges_for_globals_fn: function used to aggregate the edges for the
#       globals.
#   """
#   return GraphNetwork(
#       update_edge_fn=lambda e, s, r, g: update_edge_fn(s, r),
#       update_node_fn=None,
#       update_global_fn=lambda n, e, g: update_global_fn(e),
#       attention_logit_fn=None,
#       aggregate_edges_for_globals_fn=aggregate_edges_for_globals_fn)




# def GraphNetGAT(
#     update_edge_fn: GNUpdateEdgeFn,
#     update_node_fn: GNUpdateNodeFn,
#     attention_logit_fn: AttentionLogitFn,
#     attention_reduce_fn: AttentionReduceFn,
#     update_global_fn: Optional[GNUpdateGlobalFn] = None,
#     aggregate_edges_for_nodes_fn: AggregateEdgesToNodesFn = utils.segment_sum,
#     aggregate_nodes_for_globals_fn: AggregateNodesToGlobalsFn = utils.
#     segment_sum,
#     aggregate_edges_for_globals_fn: AggregateEdgesToGlobalsFn = utils.
#     segment_sum
#     ):
#   """Returns a method that applies a GraphNet with attention on edge features.

#   Args:
#     update_edge_fn: function used to update the edges.
#     update_node_fn: function used to update the nodes.
#     attention_logit_fn: function used to calculate the attention weights.
#     attention_reduce_fn: function used to apply attention weights to the edge
#       features.
#     update_global_fn: function used to update the globals or None to deactivate
#       globals updates.
#     aggregate_edges_for_nodes_fn: function used to aggregate attention-weighted
#       messages to each node.
#     aggregate_nodes_for_globals_fn: function used to aggregate the nodes for the
#       globals.
#     aggregate_edges_for_globals_fn: function used to aggregate
#       attention-weighted edges for the globals.

#   Returns:
#     A function that applies a GraphNet Graph Attention layer.
#   """
#   if (attention_logit_fn is None) or (attention_reduce_fn is None):
#     raise ValueError(('`None` value not supported for `attention_logit_fn` or '
#                       '`attention_reduce_fn` in a Graph Attention network.'))
#   return GraphNetwork(
#       update_edge_fn=update_edge_fn,
#       update_node_fn=update_node_fn,
#       update_global_fn=update_global_fn,
#       attention_logit_fn=attention_logit_fn,
#       attention_reduce_fn=attention_reduce_fn,
#       aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn,
#       aggregate_nodes_for_globals_fn=aggregate_nodes_for_globals_fn,
#       aggregate_edges_for_globals_fn=aggregate_edges_for_globals_fn)



