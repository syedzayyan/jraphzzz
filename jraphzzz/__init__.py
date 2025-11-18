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
"""Jraphzzz."""

from jraphzzz.data.graph import GraphsTuple
from jraphzzz.nn.types import AggregateEdgesToGlobalsFn
from jraphzzz.nn.types import AggregateEdgesToNodesFn
from jraphzzz.nn.types import AggregateNodesToGlobalsFn
from jraphzzz.nn.types import AttentionLogitFn
from jraphzzz.nn.types import AttentionReduceFn
from jraphzzz.nn.layers import DeepSets

# from jraph.nn.types import EmbedEdgeFn
# from jraph.nn.types import EmbedGlobalFn
# from jraph.nn.types import EmbedNodeFn
from jraphzzz.nn.layers import GAT
from jraphzzz.nn.types import GATAttentionLogitFn
from jraphzzz.nn.types import GATAttentionQueryFn
from jraphzzz.nn.types import GATNodeUpdateFn
from jraphzzz.nn.types import GNUpdateEdgeFn
from jraphzzz.nn.types import GNUpdateGlobalFn
from jraphzzz.nn.types import GNUpdateNodeFn
from jraphzzz.nn.layers import GraphConvolution

from jraphzzz.nn.base import GraphMapFeatures
# from jraph.nn.models import GraphNetGAT
from jraphzzz.nn.base import GraphNetwork

# from jraph.nn.types import InteractionNetwork
from jraphzzz.nn.types import InteractionUpdateEdgeFn
from jraphzzz.nn.types import InteractionUpdateNodeFn
from jraphzzz.nn.types import NodeFeatures

# from jraph.nn.types import RelationNetwork
from jraphzzz.utils.utils import ArrayTree
from jraphzzz.utils.utils import batch
from jraphzzz.utils.utils import batch_np
from jraphzzz.utils.utils import concatenated_args
from jraphzzz.utils.utils import dynamically_batch
from jraphzzz.utils.utils import get_edge_padding_mask
from jraphzzz.utils.utils import get_fully_connected_graph
from jraphzzz.utils.utils import get_graph_padding_mask
from jraphzzz.utils.utils import get_node_padding_mask
from jraphzzz.utils.utils import get_number_of_padding_with_graphs_edges
from jraphzzz.utils.utils import get_number_of_padding_with_graphs_graphs
from jraphzzz.utils.utils import get_number_of_padding_with_graphs_nodes
from jraphzzz.utils.utils import pad_with_graphs
from jraphzzz.utils.utils import partition_softmax
from jraphzzz.utils.utils import segment_max
from jraphzzz.utils.utils import segment_max_or_constant
from jraphzzz.utils.utils import segment_mean
from jraphzzz.utils.utils import segment_min
from jraphzzz.utils.utils import segment_min_or_constant
from jraphzzz.utils.utils import segment_normalize
from jraphzzz.utils.utils import segment_softmax
from jraphzzz.utils.utils import segment_sum
from jraphzzz.utils.utils import segment_variance
from jraphzzz.utils.utils import sparse_matrix_to_graphs_tuple
from jraphzzz.utils.utils import unbatch
from jraphzzz.utils.utils import unbatch_np
from jraphzzz.utils.utils import unpad_with_graphs
from jraphzzz.utils.utils import with_zero_out_padding_outputs
from jraphzzz.utils.utils import zero_out_padding

from jraphzzz.utils.laplacian import get_laplacian
from jraphzzz.utils.laplacian import get_laplacian_matrix

from jraphzzz.data.download.cora import Planetoid

from jraphzzz.utils.chem import from_smiles

__version__ = "0.0.7.dev0"

__all__ = (
    "ArrayTree",
    "DeepSets",
    "GraphConvolution",
    "GraphMapFeatures",
    "InteractionNetwork",
    "RelationNetwork",
    "GraphNetGAT",
    "GAT",
    "GraphsTuple",
    "GraphNetwork",
    "NodeFeatures",
    "AggregateEdgesToNodesFn",
    "AggregateNodesToGlobalsFn",
    "AggregateEdgesToGlobalsFn",
    "AttentionLogitFn",
    "AttentionReduceFn",
    "GNUpdateEdgeFn",
    "GNUpdateNodeFn",
    "GNUpdateGlobalFn",
    "InteractionUpdateNodeFn",
    "InteractionUpdateEdgeFn",
    "EmbedEdgeFn",
    "EmbedNodeFn",
    "EmbedGlobalFn",
    "GATAttentionQueryFn",
    "GATAttentionLogitFn",
    "GATNodeUpdateFn",
    "batch",
    "batch_np",
    "unbatch",
    "unbatch_np",
    "pad_with_graphs",
    "get_number_of_padding_with_graphs_graphs",
    "get_number_of_padding_with_graphs_nodes",
    "get_number_of_padding_with_graphs_edges",
    "unpad_with_graphs",
    "get_node_padding_mask",
    "get_edge_padding_mask",
    "get_graph_padding_mask",
    "segment_max",
    "segment_max_or_constant",
    "segment_min_or_constant",
    "segment_softmax",
    "segment_sum",
    "partition_softmax",
    "concatenated_args",
    "get_fully_connected_graph",
    "dynamically_batch",
    "with_zero_out_padding_outputs",
    "zero_out_padding",
    "sparse_matrix_to_graphs_tuple",
    "Planetoid",

    "from_smiles",
)
