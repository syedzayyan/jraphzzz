from typing import Callable
from jraphzzz.utils import utils
from ..types import (
    NodeFeatures,
    AggregateNodesToGlobalsFn,
    Globals,
)
from ..base import GraphNetwork


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
