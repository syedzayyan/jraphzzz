import jax.numpy as jnp
from typing import NamedTuple
from .graph import ArrayTree

class ShardedEdgesGraphsTuple(NamedTuple):
    """A `GraphsTuple` for use with `ShardedEdgesGraphNetwork`.

    NOTES:
    - A ShardedEdgesGraphNetwork is for use with `jax.pmap`. As such, it will have
      a leading axis of size `num_devices` on the host, but no such axis on
      device. Non-sharded data is replicated on each device. To achieve this with
      `jax.pmap` you can broadcast non-sharded data to have leading axis
      'num_devices' or use the 'in_axes' parameter, which will indicate which
      attributes should be replicated and which should not. Current helper methods
      use the first approach.
    - It is recommended that you constructed ShardedEdgesGraphsTuples with
      `graphs_tuple_to_broadcasted_sharded_grahs_tuple`.


    The values of `nodes`, `device_edges` and `globals` can be gn_graph.ArrayTree
    - nests of features with `jax` compatible values. For example, `nodes` in a
    graph may have more than one type of attribute.

    However, the ShardedEdgesGraphsTuple typically takes the following form for a
    batch of `n` graphs:

    - n_node: The number of nodes per graph. It is a vector of integers with shape
      `[n_graphs]`, such that `graph.n_node[i]` is the number of nodes in the i-th
      graph.

    - n_edge: The number of edges per graph. It is a vector of integers with shape
      `[n_graphs]`, such that `graph.n_edge[i]` is the number of edges in the i-th
      graph.

    - nodes: The nodes features. It is either `None` (the graph has no node
      features), or a vector of shape `[n_nodes] + node_shape`, where
      `n_nodes = sum(graph.n_node)` is the total number of nodes in the batch of
      graphs, and `node_shape` represents the shape of the features of each node.
      The relative index of a node from the batched version can be recovered from
      the `graph.n_node` property. For instance, the second node of the third
      graph will have its features in the
      `1 + graph.n_node[0] + graph.n_node[1]`-th slot of graph.nodes.
      Observe that having a `None` value for this field does not mean that the
      graphs have no nodes, only that they do not have node features.

    - receivers: The indices of the receiver nodes, for each edge. It is either
      `None` (if the graph has no edges), or a vector of integers of shape
      `[n_edges]`, such that `graph.receivers[i]` is the index of the node
      receiving from the i-th edge.

      Observe that the index is absolute (in other words, cumulative), i.e.
      `graphs.receivers` take value in `[0, n_nodes]`. For instance, an edge
      connecting the vertices with relative indices 2 and 3 in the second graph of
      the batch would have a `receivers` value of `3 + graph.n_node[0]`.
      If `graphs.receivers` is `None`, then `graphs.edges` and `graphs.senders`
      should also be `None`.

    - senders: The indices of the sender nodes, for each edge. It is either
      `None` (if the graph has no edges), or a vector of integers of shape
      `[n_edges]`, such that `graph.senders[i]` is the index of the node
      sending from the i-th edge.

      Observe that the index is absolute, i.e. `graphs.senders` take value in
      `[0, n_nodes]`. For instance, an edge connecting the vertices with relative
      indices 1 and 3 in the third graph of the batch would have a `senders` value
      of `1 + graph.n_node[0] + graph.n_node[1]`.

      If `graphs.senders` is `None`, then `graphs.edges` and `graphs.receivers`
      should also be `None`.

    - globals: The global features of the graph. It is either `None` (the graph
      has no global features), or a vector of shape `[n_graphs] + global_shape`
      representing graph level features.

    The ShardedEdgesGraphsTuple also contains device-local attributes that are
    used for data parallel computation. On the host, each of these attributes will
    have an additional leading axis of shape `num_devices` for use with
    `jax.pmap`, but this is ommited in the following documentation.

    - device_edges: The subset of the edge features that are on the device.
        It is either `None` (the graph has no edge features), or a vector of
        shape `[num_edges / num_devices] + edge_shape`

        Observe that having a `None` value for this field does not mean that the
        graph has no edges, only that they do not have edge features.

    - device_senders: The sender indices of edges on device. This is of length
        num_edges / num_devices.

    - device_receivers: The receiver indices of edge on device. This is of length
        num_edges / num_devices.

    - device_n_edge: The graph partitions of the edges on device. For example,
        say that there are 2 graphs in the original graphs tuple, with n_edge
        [1, 11], which has been split over 3 devices. The `device_n_edge`s would
        be [[1, 3], [4, 0], [4, 0]]. `0` valued entries that are padding values or
        graphs with zero edges are not distinguished. Since these attributes are
        used only for `repeat` purposes, the difference makes no difference to
        the implementation.

    - device_graph_idx: The indices of the graphs on device. For example, say
        that there are 5 graphs in the original graphs tuple, and these has been
        split over 3 devices, the device_graphs_idxs could be
        [[0, 1, 2], [2, 3, 0], [3, 4, 0]]. In this splitting, the third graph
        is split over 2 devices. If a `0` is the first in `device_graph_idx` then
        that indicates the first graph, otherwise it indicates a padding value.
    """

    nodes: ArrayTree
    device_edges: ArrayTree
    device_receivers: jnp.ndarray  # with integer dtype
    device_senders: jnp.ndarray  # with integer dtype
    receivers: jnp.ndarray  # with integer dtype
    senders: jnp.ndarray  # with integer dtype
    globals: ArrayTree
    device_n_edge: jnp.ndarray  # with integer dtype
    n_node: jnp.ndarray  # with integer dtype
    n_edge: jnp.ndarray  # with integer dtype
    device_graph_idx: jnp.ndarray  # with integer dtype

