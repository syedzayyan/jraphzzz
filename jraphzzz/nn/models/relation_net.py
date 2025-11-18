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

from typing import Callable
from jraphzzz.utils import utils
from jraphzzz.nn.base import GraphNetwork
from jraphzzz.nn.types import (
    SenderFeatures,
    ReceiverFeatures,
    EdgeFeatures,
    NodeFeatures,
    AggregateEdgesToGlobalsFn,
)



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
