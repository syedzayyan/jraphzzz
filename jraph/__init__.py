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
"""Jraph."""


from jraph.data.graph import GraphsTuple
from jraph.nn.types import AggregateEdgesToGlobalsFn
from jraph.nn.types import AggregateEdgesToNodesFn
from jraph.nn.types import AggregateNodesToGlobalsFn
from jraph.nn.types import AttentionLogitFn
from jraph.nn.types import AttentionReduceFn
from jraph.nn.layers import DeepSets
# from jraph.nn.types import EmbedEdgeFn
# from jraph.nn.types import EmbedGlobalFn
# from jraph.nn.types import EmbedNodeFn
from jraph.nn.layers import GAT
from jraph.nn.types import GATAttentionLogitFn
from jraph.nn.types import GATAttentionQueryFn
from jraph.nn.types import GATNodeUpdateFn
from jraph.nn.types import GNUpdateEdgeFn
from jraph.nn.types import GNUpdateGlobalFn
from jraph.nn.types import GNUpdateNodeFn
from jraph.nn.layers import GraphConvolution
# from jraph.nn.types import GraphMapFeatures
# from jraph.nn.models import GraphNetGAT
from jraph.nn.base import GraphNetwork
# from jraph.nn.types import InteractionNetwork
from jraph.nn.types import InteractionUpdateEdgeFn
from jraph.nn.types import InteractionUpdateNodeFn
from jraph.nn.types import NodeFeatures
# from jraph.nn.types import RelationNetwork
from jraph.utils import ArrayTree
from jraph.utils import batch
from jraph.utils import batch_np
from jraph.utils import concatenated_args
from jraph.utils import dynamically_batch
from jraph.utils import get_edge_padding_mask
from jraph.utils import get_fully_connected_graph
from jraph.utils import get_graph_padding_mask
from jraph.utils import get_node_padding_mask
from jraph.utils import get_number_of_padding_with_graphs_edges
from jraph.utils import get_number_of_padding_with_graphs_graphs
from jraph.utils import get_number_of_padding_with_graphs_nodes
from jraph.utils import pad_with_graphs
from jraph.utils import partition_softmax
from jraph.utils import segment_max
from jraph.utils import segment_max_or_constant
from jraph.utils import segment_mean
from jraph.utils import segment_min
from jraph.utils import segment_min_or_constant
from jraph.utils import segment_normalize
from jraph.utils import segment_softmax
from jraph.utils import segment_sum
from jraph.utils import segment_variance
from jraph.utils import sparse_matrix_to_graphs_tuple
from jraph.utils import unbatch
from jraph.utils import unbatch_np
from jraph.utils import unpad_with_graphs
from jraph.utils import with_zero_out_padding_outputs
from jraph.utils import zero_out_padding


from jraph.data.downloads import Planetoid


__version__ = "0.0.6.dev0"

__all__ = ("ArrayTree", "DeepSets", "GraphConvolution", "GraphMapFeatures",
           "InteractionNetwork", "RelationNetwork", "GraphNetGAT", "GAT",
           "GraphsTuple", "GraphNetwork", "NodeFeatures",
           "AggregateEdgesToNodesFn", "AggregateNodesToGlobalsFn",
           "AggregateEdgesToGlobalsFn", "AttentionLogitFn", "AttentionReduceFn",
           "GNUpdateEdgeFn", "GNUpdateNodeFn", "GNUpdateGlobalFn",
           "InteractionUpdateNodeFn", "InteractionUpdateEdgeFn", "EmbedEdgeFn",
           "EmbedNodeFn", "EmbedGlobalFn", "GATAttentionQueryFn",
           "GATAttentionLogitFn", "GATNodeUpdateFn", "batch", "batch_np",
           "unbatch", "unbatch_np", "pad_with_graphs",
           "get_number_of_padding_with_graphs_graphs",
           "get_number_of_padding_with_graphs_nodes",
           "get_number_of_padding_with_graphs_edges", "unpad_with_graphs",
           "get_node_padding_mask", "get_edge_padding_mask",
           "get_graph_padding_mask", "segment_max", "segment_max_or_constant",
           "segment_min_or_constant", "segment_softmax", "segment_sum",
           "partition_softmax", "concatenated_args",
           "get_fully_connected_graph", "dynamically_batch",
           "with_zero_out_padding_outputs", "zero_out_padding",
           "sparse_matrix_to_graphs_tuple", "Planetoid")

#  _________________________________________
# / Please don't use symbols in `_src` they \
# \ are not part of the Jraph public API.    /
#  -----------------------------------------
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||
#
try:
  del _src  # pylint: disable=undefined-variable
except NameError:
  pass
