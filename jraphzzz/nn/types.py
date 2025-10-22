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

from typing import Any, Callable, Iterable, Mapping, Union
import jax.numpy as jnp

# As of 04/2020 pytype doesn't support recursive types.
# pytype: disable=not-supported-yet
ArrayTree = Union[jnp.ndarray, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]

# All features will be an ArrayTree.
NodeFeatures = EdgeFeatures = SenderFeatures = ReceiverFeatures = Globals = ArrayTree

# Signature:
# (edges of each node to be aggregated, segment ids, number of segments) ->
# aggregated edges
AggregateEdgesToNodesFn = Callable[[EdgeFeatures, jnp.ndarray, int], NodeFeatures]

# Signature:
# (nodes of each graph to be aggregated, segment ids, number of segments) ->
# aggregated nodes
AggregateNodesToGlobalsFn = Callable[[NodeFeatures, jnp.ndarray, int], Globals]

# Signature:
# (edges of each graph to be aggregated, segment ids, number of segments) ->
# aggregated edges
AggregateEdgesToGlobalsFn = Callable[[EdgeFeatures, jnp.ndarray, int], Globals]

# Signature:
# (edge features, sender node features, receiver node features, globals) ->
# attention weights
AttentionLogitFn = Callable[
    [EdgeFeatures, SenderFeatures, ReceiverFeatures, Globals], ArrayTree
]

# Signature:
# (edge features, weights) -> edge features for node update
AttentionReduceFn = Callable[[EdgeFeatures, ArrayTree], EdgeFeatures]

# Signature:
# (edges to be normalized, segment ids, number of segments) ->
# normalized edges
AttentionNormalizeFn = Callable[[EdgeFeatures, jnp.ndarray, int], EdgeFeatures]

# Signature:
# (edge features, sender node features, receiver node features, globals) ->
# updated edge features
GNUpdateEdgeFn = Callable[
    [EdgeFeatures, SenderFeatures, ReceiverFeatures, Globals], EdgeFeatures
]

# Signature:
# (node features, outgoing edge features, incoming edge features,
#  globals) -> updated node features
GNUpdateNodeFn = Callable[
    [NodeFeatures, SenderFeatures, ReceiverFeatures, Globals], NodeFeatures
]

GNUpdateGlobalFn = Callable[[NodeFeatures, EdgeFeatures, Globals], Globals]


GATAttentionQueryFn = Callable[[NodeFeatures], NodeFeatures]
GATAttentionLogitFn = Callable[
    [SenderFeatures, ReceiverFeatures, EdgeFeatures], EdgeFeatures
]
GATNodeUpdateFn = Callable[[NodeFeatures], NodeFeatures]


InteractionUpdateNodeFn = Callable[
    [NodeFeatures, Mapping[str, SenderFeatures], Mapping[str, ReceiverFeatures]],
    NodeFeatures,
]
InteractionUpdateNodeFnNoSentEdges = Callable[
    [NodeFeatures, Mapping[str, ReceiverFeatures]], NodeFeatures
]

InteractionUpdateEdgeFn = Callable[
    [EdgeFeatures, SenderFeatures, ReceiverFeatures], EdgeFeatures
]


# Signature:
# edge features -> embedded edge features
EmbedEdgeFn = Callable[[EdgeFeatures], EdgeFeatures]

# Signature:
# node features -> embedded node features
EmbedNodeFn = Callable[[NodeFeatures], NodeFeatures]

# Signature:
# globals features -> embedded globals features
EmbedGlobalFn = Callable[[Globals], Globals]
