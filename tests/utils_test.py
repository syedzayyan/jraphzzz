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
"""Tests for jraph.utils."""

import functools
import os
import jax
from jax._src import xla_bridge
import jax.numpy as jnp
import jax.tree_util as tree
from jraphzzz.data import graph
from jraphzzz.utils import utils
import numpy as np
import pytest


def _get_random_graph(
    max_n_graph=10,
    include_node_features=True,
    include_edge_features=True,
    include_globals=True,
):
    n_graph = np.random.randint(1, max_n_graph + 1)
    n_node = np.random.randint(0, 10, n_graph)
    n_edge = np.random.randint(0, 20, n_graph)
    # We cannot have any edges if there are no nodes.
    n_edge[n_node == 0] = 0

    senders = []
    receivers = []
    offset = 0
    for n_node_in_graph, n_edge_in_graph in zip(n_node, n_edge):
        if n_edge_in_graph != 0:
            senders += list(
                np.random.randint(0, n_node_in_graph, n_edge_in_graph) + offset
            )
            receivers += list(
                np.random.randint(0, n_node_in_graph, n_edge_in_graph) + offset
            )
        offset += n_node_in_graph
    if include_globals:
        global_features = jnp.asarray(np.random.random(size=(n_graph, 5)))
    else:
        global_features = None
    if include_node_features:
        nodes = jnp.asarray(np.random.random(size=(np.sum(n_node), 4)))
    else:
        nodes = None

    if include_edge_features:
        edges = jnp.asarray(np.random.random(size=(np.sum(n_edge), 3)))
    else:
        edges = None
    return graph.GraphsTuple(
        n_node=jnp.asarray(n_node),
        n_edge=jnp.asarray(n_edge),
        nodes=nodes,
        edges=edges,
        globals=global_features,
        senders=jnp.asarray(senders),
        receivers=jnp.asarray(receivers),
    )


def _make_nest(array):
    """Returns a nest given an array."""
    return {"a": array, "b": [jnp.ones_like(array), {"c": jnp.zeros_like(array)}]}


def _get_list_and_batched_graph():
    """Returns a list of individual graphs and a batched version.

    This test-case includes the following corner-cases:
      - single node,
      - multiple nodes,
      - no edges,
      - single edge,
      - and multiple edges.
    """
    batched_graph = graph.GraphsTuple(
        n_node=jnp.array([1, 3, 1, 0, 2, 0, 0]),
        n_edge=jnp.array([2, 5, 0, 0, 1, 0, 0]),
        nodes=_make_nest(jnp.arange(14).reshape(7, 2)),
        edges=_make_nest(jnp.arange(24).reshape(8, 3)),
        globals=_make_nest(jnp.arange(14).reshape(7, 2)),
        senders=jnp.array([0, 0, 1, 1, 2, 3, 3, 6]),
        receivers=jnp.array([0, 0, 2, 1, 3, 2, 1, 5]),
    )

    list_graphs = [
        graph.GraphsTuple(
            n_node=jnp.array([1]),
            n_edge=jnp.array([2]),
            nodes=_make_nest(jnp.array([[0, 1]])),
            edges=_make_nest(jnp.array([[0, 1, 2], [3, 4, 5]])),
            globals=_make_nest(jnp.array([[0, 1]])),
            senders=jnp.array([0, 0]),
            receivers=jnp.array([0, 0]),
        ),
        graph.GraphsTuple(
            n_node=jnp.array([3]),
            n_edge=jnp.array([5]),
            nodes=_make_nest(jnp.array([[2, 3], [4, 5], [6, 7]])),
            edges=_make_nest(
                jnp.array(
                    [[6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17], [18, 19, 20]]
                )
            ),
            globals=_make_nest(jnp.array([[2, 3]])),
            senders=jnp.array([0, 0, 1, 2, 2]),
            receivers=jnp.array([1, 0, 2, 1, 0]),
        ),
        graph.GraphsTuple(
            n_node=jnp.array([1]),
            n_edge=jnp.array([0]),
            nodes=_make_nest(jnp.array([[8, 9]])),
            edges=_make_nest(jnp.zeros((0, 3))),
            globals=_make_nest(jnp.array([[4, 5]])),
            senders=jnp.array([]),
            receivers=jnp.array([]),
        ),
        graph.GraphsTuple(
            n_node=jnp.array([0]),
            n_edge=jnp.array([0]),
            nodes=_make_nest(jnp.zeros((0, 2))),
            edges=_make_nest(jnp.zeros((0, 3))),
            globals=_make_nest(jnp.array([[6, 7]])),
            senders=jnp.array([]),
            receivers=jnp.array([]),
        ),
        graph.GraphsTuple(
            n_node=jnp.array([2]),
            n_edge=jnp.array([1]),
            nodes=_make_nest(jnp.array([[10, 11], [12, 13]])),
            edges=_make_nest(jnp.array([[21, 22, 23]])),
            globals=_make_nest(jnp.array([[8, 9]])),
            senders=jnp.array([1]),
            receivers=jnp.array([0]),
        ),
        graph.GraphsTuple(
            n_node=jnp.array([0]),
            n_edge=jnp.array([0]),
            nodes=_make_nest(jnp.zeros((0, 2))),
            edges=_make_nest(jnp.zeros((0, 3))),
            globals=_make_nest(jnp.array([[10, 11]])),
            senders=jnp.array([]),
            receivers=jnp.array([]),
        ),
        graph.GraphsTuple(
            n_node=jnp.array([0]),
            n_edge=jnp.array([0]),
            nodes=_make_nest(jnp.zeros((0, 2))),
            edges=_make_nest(jnp.zeros((0, 3))),
            globals=_make_nest(jnp.array([[12, 13]])),
            senders=jnp.array([]),
            receivers=jnp.array([]),
        ),
        graph.GraphsTuple(
            n_node=jnp.array([]),
            n_edge=jnp.array([]),
            nodes=_make_nest(jnp.zeros((0, 2))),
            edges=_make_nest(jnp.zeros((0, 3))),
            globals=_make_nest(jnp.zeros((0, 2))),
            senders=jnp.array([]),
            receivers=jnp.array([]),
        ),
    ]

    return list_graphs, batched_graph


def _get_list_matrix():
    """Returns a list of adjacency matrices, and its sparse and graph versions.

    This test-case includes the following corner-cases:
      - single node,
      - multiple nodes,
      - no edges,
      - single edge,
      - and multiple edges.
    """
    adj_matrices = [
        np.array([[2]]),
        np.array([[1, 1, 0], [0, 0, 1], [1, 1, 0]]),
        np.array([[0]]),
        np.array([[]]),
        np.array([[0, 0], [1, 0]]),
    ]
    # Sparse version of the above adjacency matrix.
    sparse_coo_matrices = [
        # (row, column, values, n_node)
        (np.array([0]), np.array([0]), np.array([2]), np.array([1])),
        (
            np.array([0, 0, 1, 2, 2]),
            np.array([0, 1, 2, 0, 1]),
            np.array([1, 1, 1, 1, 1]),
            np.array(3),
        ),
        (np.array([]), np.array([]), np.array([]), np.array(1)),
        (np.array([]), np.array([]), np.array([]), np.array(0)),
        (np.array([1]), np.array([0]), np.array([1]), np.array(2)),
    ]
    expected_graphs = [
        graph.GraphsTuple(
            n_node=jnp.array([1]),
            n_edge=jnp.array([2]),
            nodes=None,
            edges=None,
            globals=None,
            senders=jnp.array([0, 0]),
            receivers=jnp.array([0, 0]),
        ),
        graph.GraphsTuple(
            n_node=jnp.array([3]),
            n_edge=jnp.array([5]),
            nodes=None,
            edges=None,
            globals=None,
            senders=jnp.array([0, 0, 1, 2, 2]),
            receivers=jnp.array([0, 1, 2, 0, 1]),
        ),
        graph.GraphsTuple(
            n_node=jnp.array([1]),
            n_edge=jnp.array([0]),
            nodes=None,
            edges=None,
            globals=None,
            senders=jnp.array([]),
            receivers=jnp.array([]),
        ),
        graph.GraphsTuple(
            n_node=jnp.array([0]),
            n_edge=jnp.array([0]),
            nodes=None,
            edges=None,
            globals=None,
            senders=jnp.array([]),
            receivers=jnp.array([]),
        ),
        graph.GraphsTuple(
            n_node=jnp.array([2]),
            n_edge=jnp.array([1]),
            nodes=None,
            edges=None,
            globals=None,
            senders=jnp.array([1]),
            receivers=jnp.array([0]),
        ),
    ]
    return adj_matrices, sparse_coo_matrices, expected_graphs


_DB_NUM_NODES = (10, 15)
_DB_NODE_SHAPE = (3, 4, 1)
_DB_NUM_EDGES = (12, 17)
_DB_EDGE_SHAPE = (4, 3)
_DB_GLOBAL_SHAPE = (2, 3, 1, 4)


def _make_dynamic_batch_graph(
    add_globals,
    num_nodes=_DB_NUM_NODES,
    num_edges=_DB_NUM_EDGES,
):
    total_num_edges = sum(num_edges)
    total_num_nodes = sum(num_nodes)
    g_ = _make_nest(
        np.random.normal(size=_DB_GLOBAL_SHAPE)) if add_globals else None
    return graph.GraphsTuple(
        nodes=_make_nest(
            np.random.normal(size=(total_num_nodes,) + _DB_NODE_SHAPE)),
        edges=_make_nest(
            np.random.normal(size=(total_num_edges,) + _DB_EDGE_SHAPE)),
        n_edge=np.array(num_edges),
        n_node=np.array(num_nodes),
        senders=np.random.randint(
            0, total_num_nodes, size=total_num_edges, dtype=np.int32),
        receivers=np.random.randint(
            0, total_num_nodes, size=total_num_edges, dtype=np.int32),
        globals=g_)


def test_batch():
    """Tests batching of graph."""
    list_graphs_tuple, batched_graphs_tuple = _get_list_and_batched_graph()
    graphs_tuple = utils.batch(list_graphs_tuple)
    jax.tree_util.tree_map(
        np.testing.assert_allclose, graphs_tuple, batched_graphs_tuple
    )


def test_unbatch():
    """Tests unbatching of graph."""
    list_graphs_tuple, batched_graphs_tuple = _get_list_and_batched_graph()
    graphs_tuples = utils.unbatch(batched_graphs_tuple)
    # The final GraphsTuple does not contain a graph, and so shouldn't be
    # present in the result.
    jax.tree_util.tree_map(
        np.testing.assert_allclose, graphs_tuples, list_graphs_tuple[:-1]
    )


def test_batch_np():
    """Tests batching of graph in numpy."""
    (list_graphs_tuple, batched_graphs_tuple) = _get_list_and_batched_graph()
    graphs_tuple = utils.batch_np(list_graphs_tuple)
    jax.tree_util.tree_map(
        np.testing.assert_allclose, graphs_tuple, batched_graphs_tuple
    )


def test_unbatch_np():
    """Tests unbatching of graph in numpy."""
    (list_graphs_tuple, batched_graphs_tuple) = _get_list_and_batched_graph()
    graphs_tuples = utils.unbatch_np(batched_graphs_tuple)
    # The final GraphsTuple does not contain a graph, and so shouldn't be
    # present in the result.
    jax.tree_util.tree_map(
        np.testing.assert_allclose, graphs_tuples, list_graphs_tuple[:-1]
    )


@pytest.mark.parametrize(
    "include_globals,include_nodes,include_edges",
    [(True, True, False), (True, False, True), (False, True, True)],
)
def test_batch_unbatch_with_random_graphs(
    include_globals, include_nodes, include_edges
):
    """Tests batch(unbatch) is identity with random graphs."""
    np.random.seed(42)
    for _ in range(100):
        g = _get_random_graph(
            include_globals=include_globals,
            include_node_features=include_nodes,
            include_edge_features=include_edges,
        )
        jax.tree_util.tree_map(
            np.testing.assert_allclose, utils.batch(utils.unbatch(g)), g
        )

    for _ in range(10):

        def rg():
            return _get_random_graph(
                1, include_nodes, include_edges, include_globals
            )

        graphs1 = [rg() for _ in range(np.random.randint(1, 10))]
        graphs2 = utils.unbatch(utils.batch(graphs1))
        for g1, g2 in zip(graphs1, graphs2):
            jax.tree_util.tree_map(np.testing.assert_allclose, g1, g2)


def test_pad_with_graphs():
    """Tests padding of graph."""
    _, graphs_tuple = _get_list_and_batched_graph()
    padded_graphs_tuple = utils.pad_with_graphs(graphs_tuple, 10, 12, 9)
    expected_padded_graph = graph.GraphsTuple(
        n_node=jnp.concatenate([graphs_tuple.n_node, jnp.array([3, 0])]),
        n_edge=jnp.concatenate([graphs_tuple.n_edge, jnp.array([4, 0])]),
        nodes=tree.tree_map(
            lambda f: jnp.concatenate([f, jnp.zeros((3, 2), dtype=f.dtype)]),
            graphs_tuple.nodes,
        ),
        edges=tree.tree_map(
            lambda f: jnp.concatenate([f, jnp.zeros((4, 3), dtype=f.dtype)]),
            graphs_tuple.edges,
        ),
        globals=tree.tree_map(
            lambda f: jnp.concatenate([f, jnp.zeros((2, 2), dtype=f.dtype)]),
            graphs_tuple.globals,
        ),
        senders=jnp.concatenate([graphs_tuple.senders, jnp.array([7, 7, 7, 7])]),
        receivers=jnp.concatenate([graphs_tuple.receivers, jnp.array([7, 7, 7, 7])]),
    )
    jax.tree_util.tree_map(
        np.testing.assert_allclose, padded_graphs_tuple, expected_padded_graph
    )


def test_unpad():
    """Tests unpadding of graph."""
    _, graphs_tuple = _get_list_and_batched_graph()
    unpadded_graphs_tuple = utils.unpad_with_graphs(graphs_tuple)
    expected_unpadded_graph = graph.GraphsTuple(
        n_node=jnp.array([1, 3, 1, 0]),
        n_edge=jnp.array([2, 5, 0, 0]),
        nodes=_make_nest(jnp.arange(10).reshape(5, 2)),
        edges=_make_nest(jnp.arange(21).reshape(7, 3)),
        globals=_make_nest(jnp.arange(8).reshape(4, 2)),
        senders=jnp.array([0, 0, 1, 1, 2, 3, 3]),
        receivers=jnp.array([0, 0, 2, 1, 3, 2, 1]),
    )
    jax.tree_util.tree_map(
        np.testing.assert_allclose, unpadded_graphs_tuple, expected_unpadded_graph
    )


@pytest.mark.parametrize(
    "include_globals,include_nodes,include_edges",
    [(True, True, False), (True, False, True), (False, True, True)],
)
def test_pad_unpad_with_random_graphs(include_globals, include_nodes, include_edges):
    """Tests unpad(pad) is identity with random graphs."""
    np.random.seed(42)
    for _ in range(100):
        g = _get_random_graph(
            include_globals=include_globals,
            include_node_features=include_nodes,
            include_edge_features=include_edges,
        )
        jax.tree_util.tree_map(
            np.testing.assert_allclose,
            utils.unpad_with_graphs(utils.pad_with_graphs(g, 101, 200, 11)),
            g,
        )


def test_pad_unpad_with_graphs_exact_padding():
    """Tests unpad(pad) is identity with random graphs."""
    g = _get_random_graph(
        include_globals=True, include_node_features=True, include_edge_features=True
    )
    recovered_g = utils.unpad_with_graphs(
        utils.pad_with_graphs(
            g,
            n_node=g.n_node.sum() + 1,
            n_edge=g.n_edge.sum(),
            n_graph=g.n_node.shape[0] + 1,
        )
    )

    jax.tree_util.tree_map(np.testing.assert_allclose, recovered_g, g)


def test_get_number_of_padding_with_graphs_graphs():
    """Tests the number of padding graphs calculation."""
    _, graphs_tuple = _get_list_and_batched_graph()
    expected = 3
    with jax.disable_jit():
        jax.tree_util.tree_map(
            np.testing.assert_allclose,
            utils.get_number_of_padding_with_graphs_graphs(graphs_tuple),
            expected,
        )

    jax.tree_util.tree_map(
        np.testing.assert_allclose,
        jax.jit(utils.get_number_of_padding_with_graphs_graphs)(graphs_tuple),
        expected,
    )


def test_get_number_of_padding_with_graphs_nodes():
    """Tests the number of padding nodes calculation."""
    _, graphs_tuple = _get_list_and_batched_graph()
    expected = 2
    with jax.disable_jit():
        jax.tree_util.tree_map(
            np.testing.assert_allclose,
            utils.get_number_of_padding_with_graphs_nodes(graphs_tuple),
            expected,
        )

    jax.tree_util.tree_map(
        np.testing.assert_allclose,
        jax.jit(utils.get_number_of_padding_with_graphs_nodes)(graphs_tuple),
        expected,
    )


def test_get_number_of_padding_with_graphs_edges():
    """Tests the number of padding edges calculation."""
    _, graphs_tuple = _get_list_and_batched_graph()
    expected = 1
    with jax.disable_jit():
        jax.tree_util.tree_map(
            np.testing.assert_allclose,
            utils.get_number_of_padding_with_graphs_edges(graphs_tuple),
            expected,
        )

    jax.tree_util.tree_map(
        np.testing.assert_allclose,
        jax.jit(utils.get_number_of_padding_with_graphs_edges)(graphs_tuple),
        expected,
    )


def test_get_node_padding_mask():
    """Tests construction of node padding mask."""
    _, graphs_tuple = _get_list_and_batched_graph()
    expected_mask = jnp.array([1, 1, 1, 1, 1, 0, 0]).astype(bool)
    with jax.disable_jit():
        mask = utils.get_node_padding_mask(graphs_tuple)
        jax.tree_util.tree_map(np.testing.assert_array_equal, mask, expected_mask)

    mask = jax.jit(utils.get_node_padding_mask)(graphs_tuple)
    jax.tree_util.tree_map(np.testing.assert_array_equal, mask, expected_mask)


def test_get_edge_padding_mask():
    """Tests construction of edge padding mask."""
    _, graphs_tuple = _get_list_and_batched_graph()
    expected_mask = jnp.array([1, 1, 1, 1, 1, 1, 1, 0]).astype(bool)
    with jax.disable_jit():
        mask = utils.get_edge_padding_mask(graphs_tuple)
        np.testing.assert_array_equal(mask, expected_mask)

    mask = jax.jit(utils.get_edge_padding_mask)(graphs_tuple)
    np.testing.assert_array_equal(mask, expected_mask)


def test_get_graph_padding_mask():
    """Tests construction of graph padding mask."""
    _, graphs_tuple = _get_list_and_batched_graph()
    expected_mask = jnp.array([1, 1, 1, 1, 0, 0, 0]).astype(bool)
    with jax.disable_jit():
        mask = utils.get_graph_padding_mask(graphs_tuple)
        np.testing.assert_array_equal(mask, expected_mask)

    mask = jax.jit(utils.get_graph_padding_mask)(graphs_tuple)
    np.testing.assert_array_equal(mask, expected_mask)


def test_segment_sum():
    result = utils.segment_sum(jnp.arange(9), jnp.array([0, 1, 2, 0, 4, 0, 1, 1, 0]), 6)
    np.testing.assert_allclose(result, jnp.array([16, 14, 2, 0, 4, 0]))


def test_segment_sum_optional_num_segments():
    result = utils.segment_sum(jnp.arange(9), jnp.array([0, 1, 2, 0, 4, 0, 1, 1, 0]))
    np.testing.assert_allclose(result, jnp.array([16, 14, 2, 0, 4]))


@pytest.mark.parametrize("nan_data", [True, False])
def test_segment_mean(nan_data):
    data = jnp.arange(9, dtype=jnp.float32)
    expected_out = jnp.array([4, 14 / 3.0, 2, 0, 4, 0])
    segment_ids = jnp.array([0, 1, 2, 0, 4, 0, 1, 1, 0])
    if nan_data:
        data = data.at[0].set(jnp.nan)
        expected_out = expected_out.at[segment_ids[0]].set(jnp.nan)
    result = utils.segment_mean(data, segment_ids, 6)
    np.testing.assert_allclose(result, expected_out)


@pytest.mark.parametrize("nan_data", [True, False])
def test_segment_variance(nan_data):
    data = jnp.arange(8, dtype=jnp.float32)
    expected_out = jnp.stack(
        [jnp.var(jnp.arange(3)), jnp.var(jnp.arange(3, 5)), jnp.var(jnp.arange(5, 8))]
    )
    segment_ids = jnp.array([0, 0, 0, 1, 1, 2, 2, 2])
    if nan_data:
        data = data.at[0].set(jnp.nan)
        expected_out = expected_out.at[segment_ids[0]].set(jnp.nan)
    result = utils.segment_variance(data, segment_ids, 3)
    np.testing.assert_allclose(result, expected_out)


@pytest.mark.parametrize("nan_data", [True, False])
def test_segment_normalize(nan_data):
    def norm(x):
        return (x - jnp.mean(x)) * jax.lax.rsqrt(jnp.var(x))

    data = jnp.arange(8, dtype=jnp.float32)
    segment_ids = jnp.array([0, 0, 0, 1, 1, 2, 2, 2])
    expected_out = jnp.concatenate(
        [
            norm(jnp.arange(3, dtype=jnp.float32)),
            norm(jnp.arange(3, 5, dtype=jnp.float32)),
            norm(jnp.arange(5, 8, dtype=jnp.float32)),
        ]
    )
    if nan_data:
        data = data.at[0].set(jnp.nan)
        expected_out = expected_out.at[:3].set(jnp.nan)
    result = utils.segment_normalize(data, segment_ids, 3)
    np.testing.assert_allclose(result, expected_out)


@pytest.mark.parametrize(
    "indices_are_sorted,unique_indices",
    [(False, False), (True, False), (True, True), (False, True)],
)
def test_segment_max(indices_are_sorted, unique_indices):
    neg_inf = jnp.iinfo(jnp.int32).min
    if unique_indices:
        data = jnp.arange(6)
        if indices_are_sorted:
            segment_ids = jnp.array([0, 1, 2, 3, 4, 5])
            expected_out = jnp.array([0, 1, 2, 3, 4, 5])
            num_segments = 6
        else:
            segment_ids = jnp.array([1, 0, 2, 4, 3, -5])
            expected_out = jnp.array([1, 0, 2, 4, 3])
            num_segments = 5
    else:
        data = jnp.arange(9)
        if indices_are_sorted:
            segment_ids = jnp.array([0, 0, 0, 1, 1, 1, 2, 3, 4])
            expected_out = jnp.array([2, 5, 6, 7, 8, neg_inf])
            num_segments = 6
        else:
            segment_ids = jnp.array([0, 1, 2, 0, 4, 0, 1, 1, -6])
            expected_out = jnp.array([5, 7, 2, neg_inf, 4, neg_inf])
            num_segments = 6

    with jax.disable_jit():
        result = utils.segment_max(
            data, segment_ids, num_segments, indices_are_sorted, unique_indices
        )
        np.testing.assert_allclose(result, expected_out)
        result = utils.segment_max(
            data,
            segment_ids,
            indices_are_sorted=indices_are_sorted,
            unique_indices=unique_indices,
        )
        num_unique_segments = jnp.max(segment_ids) + 1
        np.testing.assert_allclose(result, expected_out[:num_unique_segments])

    result = jax.jit(utils.segment_max, static_argnums=(2, 3, 4))(
        data, segment_ids, num_segments, indices_are_sorted, unique_indices
    )
    np.testing.assert_allclose(result, expected_out)


@pytest.mark.parametrize(
    "indices_are_sorted,unique_indices",
    [(False, False), (True, False), (True, True), (False, True)],
)
def test_segment_max_or_constant(indices_are_sorted, unique_indices):
    if unique_indices:
        data = jnp.arange(6, dtype=jnp.float32)
        if indices_are_sorted:
            segment_ids = jnp.array([0, 1, 2, 3, 4, 5])
            expected_out = jnp.array([0, 1, 2, 3, 4, 5, 0], dtype=jnp.float32)
            num_segments = 7
        else:
            segment_ids = jnp.array([1, 0, 2, 4, 3, -5])
            expected_out = jnp.array([1, 0, 2, 4, 3], dtype=jnp.float32)
            num_segments = 5
    else:
        data = jnp.arange(9, dtype=jnp.float32)
        if indices_are_sorted:
            segment_ids = jnp.array([0, 0, 0, 1, 1, 1, 2, 3, 4])
            expected_out = jnp.array([2, 5, 6, 7, 8, 0], dtype=jnp.float32)
            num_segments = 6
        else:
            segment_ids = jnp.array([0, 1, 2, 0, 4, 0, 1, 1, -6])
            expected_out = jnp.array([5, 7, 2, 0, 4, 0], dtype=jnp.float32)
            num_segments = 6

    with jax.disable_jit():
        result = utils.segment_max_or_constant(
            data, segment_ids, num_segments, indices_are_sorted, unique_indices
        )
        np.testing.assert_allclose(result, expected_out)
        grad = jax.grad(lambda *x: jnp.sum(utils.segment_max_or_constant(*x)))(
            data, segment_ids, num_segments, indices_are_sorted, unique_indices
        )
        assert np.all(jnp.isfinite(grad))
        result = utils.segment_max_or_constant(
            data,
            segment_ids,
            indices_are_sorted=indices_are_sorted,
            unique_indices=unique_indices,
        )
        num_unique_segments = jnp.max(segment_ids) + 1
        np.testing.assert_allclose(result, expected_out[:num_unique_segments])

    result = jax.jit(utils.segment_max_or_constant, static_argnums=(2, 3, 4))(
        data, segment_ids, num_segments, indices_are_sorted, unique_indices
    )
    np.testing.assert_allclose(result, expected_out)
    grad_fn = jax.jit(
        jax.grad(lambda *x: jnp.sum(utils.segment_max_or_constant(*x))),
        static_argnums=(2, 3, 4),
    )
    grad = grad_fn(data, segment_ids, num_segments, indices_are_sorted, unique_indices)
    assert np.all(jnp.isfinite(grad))


@pytest.mark.parametrize(
    "indices_are_sorted,unique_indices",
    [(False, False), (True, False), (True, True), (False, True)],
)
def test_segment_max_or_constant_2d(indices_are_sorted, unique_indices):
    if unique_indices:
        data = jnp.stack([jnp.arange(6), jnp.arange(6, 0, -1)], axis=1)
        if indices_are_sorted:
            segment_ids = jnp.array([0, 1, 2, 3, 4, 5])
            expected_out = jnp.array(
                [[0, 6], [1, 5], [2, 4], [3, 3], [4, 2], [5, 1]])
            num_segments = 6
        else:
            segment_ids = jnp.array([1, 0, 2, 4, 3, -5])
            expected_out = jnp.array(
                [[1, 5], [0, 6], [2, 4], [4, 2], [3, 3]])
            num_segments = 5
    else:
        data = jnp.stack([jnp.arange(9), jnp.arange(9, 0, -1)], axis=1)
        if indices_are_sorted:
            segment_ids = jnp.array([0, 0, 0, 1, 1, 1, 2, 3, 4])
            expected_out = jnp.array(
                [[2, 9], [5, 6], [6, 3], [7, 2], [8, 1], [0, 0]])
            num_segments = 6
        else:
            segment_ids = jnp.array([0, 1, 2, 0, 4, 0, 1, 1, -6])
            expected_out = jnp.array(
                [[5, 9], [7, 8], [2, 7], [0, 0], [4, 5], [0, 0]])
            num_segments = 6

    with jax.disable_jit():
        result = utils.segment_max_or_constant(data, segment_ids, num_segments,
                                                indices_are_sorted, unique_indices)
        np.testing.assert_allclose(result, expected_out)
        result = utils.segment_max_or_constant(
            data,
            segment_ids,
            indices_are_sorted=indices_are_sorted,
            unique_indices=unique_indices)
        num_unique_segments = jnp.max(segment_ids) + 1
        np.testing.assert_allclose(result, expected_out[:num_unique_segments])
    
    result = jax.jit(utils.segment_max_or_constant, static_argnums=(2, 3, 4))(
        data, segment_ids, num_segments, indices_are_sorted, unique_indices)
    np.testing.assert_allclose(result, expected_out)


@pytest.mark.parametrize(
    "indices_are_sorted,unique_indices",
    [(False, False), (True, False), (True, True), (False, True)],
)
def test_segment_min(indices_are_sorted, unique_indices):
    inf = jnp.iinfo(jnp.int32).max
    if unique_indices:
        data = jnp.arange(6)
        if indices_are_sorted:
            segment_ids = jnp.array([0, 1, 2, 3, 4, 5])
            expected_out = jnp.array([0, 1, 2, 3, 4, 5])
            num_segments = 6
        else:
            segment_ids = jnp.array([1, 0, 2, 4, 3, -5])
            expected_out = jnp.array([1, 0, 2, 4, 3])
            num_segments = 5
    else:
        data = jnp.arange(9)
        if indices_are_sorted:
            segment_ids = jnp.array([0, 0, 0, 1, 1, 1, 2, 3, 4])
            expected_out = jnp.array([0, 3, 6, 7, 8, inf])
            num_segments = 6
        else:
            segment_ids = jnp.array([0, 1, 2, 0, 4, 0, 1, 1, -6])
            expected_out = jnp.array([0, 1, 2, inf, 4, inf])
            num_segments = 6

    with jax.disable_jit():
        result = utils.segment_min(data, segment_ids, num_segments,
                                    indices_are_sorted, unique_indices)
        np.testing.assert_allclose(result, expected_out)
        result = utils.segment_min(data, segment_ids,
                                    indices_are_sorted=indices_are_sorted,
                                    unique_indices=unique_indices)
        num_unique_segments = jnp.max(segment_ids) + 1
        np.testing.assert_allclose(result, expected_out[:num_unique_segments])
    
    result = jax.jit(utils.segment_min, static_argnums=(2, 3, 4))(
        data, segment_ids, num_segments, indices_are_sorted, unique_indices)
    np.testing.assert_allclose(result, expected_out)


@pytest.mark.parametrize(
    "indices_are_sorted,unique_indices",
    [(False, False), (True, False), (True, True), (False, True)],
)
def test_segment_min_or_constant(indices_are_sorted, unique_indices):
    if unique_indices:
        data = jnp.arange(6, dtype=jnp.float32)
        if indices_are_sorted:
            segment_ids = jnp.array([0, 1, 2, 3, 4, 5])
            expected_out = jnp.array([0, 1, 2, 3, 4, 5], dtype=jnp.float32)
            num_segments = 6
        else:
            segment_ids = jnp.array([1, 0, 2, 4, 3, -5])
            expected_out = jnp.array([1, 0, 2, 4, 3], dtype=jnp.float32)
            num_segments = 5
    else:
        data = jnp.arange(9, dtype=jnp.float32)
        if indices_are_sorted:
            segment_ids = jnp.array([0, 0, 0, 1, 1, 1, 2, 3, 4])
            expected_out = jnp.array([0, 3, 6, 7, 8, 0], dtype=jnp.float32)
            num_segments = 6
        else:
            segment_ids = jnp.array([0, 1, 2, 0, 4, 0, 1, 1, -6])
            expected_out = jnp.array([0, 1, 2, 0, 4, 0], dtype=jnp.float32)
            num_segments = 6

    with jax.disable_jit():
        result = utils.segment_min_or_constant(data, segment_ids, num_segments,
                                                indices_are_sorted, unique_indices)
        np.testing.assert_allclose(result, expected_out)
        grad = jax.grad(lambda *x: jnp.sum(utils.segment_min_or_constant(*x)))(
            data, segment_ids, num_segments, indices_are_sorted, unique_indices)
        assert np.all(jnp.isfinite(grad))
        result = utils.segment_min_or_constant(
            data,
            segment_ids,
            indices_are_sorted=indices_are_sorted,
            unique_indices=unique_indices)
        num_unique_segments = jnp.max(segment_ids) + 1
        np.testing.assert_allclose(result, expected_out[:num_unique_segments])
    
    result = jax.jit(
        utils.segment_min_or_constant,
        static_argnums=(2, 3, 4))(data, segment_ids, num_segments,
                                  indices_are_sorted, unique_indices)
    np.testing.assert_allclose(result, expected_out)
    grad_fn = jax.jit(
        jax.grad(lambda *x: jnp.sum(utils.segment_min_or_constant(*x))),
        static_argnums=(2, 3, 4))
    grad = grad_fn(data, segment_ids, num_segments, indices_are_sorted,
                   unique_indices)
    assert np.all(jnp.isfinite(grad))


@pytest.mark.parametrize(
    "indices_are_sorted,unique_indices",
    [(False, False), (True, False), (True, True), (False, True)],
)
def test_segment_min_or_constant_2d(indices_are_sorted, unique_indices):
    if unique_indices:
        data = jnp.stack([jnp.arange(6), jnp.arange(6, 0, -1)], axis=1)
        if indices_are_sorted:
            segment_ids = jnp.array([0, 1, 2, 3, 4, 5])
            expected_out = jnp.array(
                [[0, 6], [1, 5], [2, 4], [3, 3], [4, 2], [5, 1]])
            num_segments = 6
        else:
            segment_ids = jnp.array([1, 0, 2, 4, 3, -5])
            expected_out = jnp.array(
                [[1, 5], [0, 6], [2, 4], [4, 2], [3, 3]])
            num_segments = 5
    else:
        data = jnp.stack([jnp.arange(9), jnp.arange(9, 0, -1)], axis=1)
        if indices_are_sorted:
            segment_ids = jnp.array([0, 0, 0, 1, 1, 1, 2, 3, 4])
            expected_out = jnp.array(
                [[0, 7], [3, 4], [6, 3], [7, 2], [8, 1], [0, 0]])
            num_segments = 6
        else:
            segment_ids = jnp.array([0, 1, 2, 0, 4, 0, 1, 1, -6])
            expected_out = jnp.array(
                [[0, 4], [1, 2], [2, 7], [0, 0], [4, 5], [0, 0]])
            num_segments = 6

    with jax.disable_jit():
        result = utils.segment_min_or_constant(data, segment_ids, num_segments,
                                                indices_are_sorted, unique_indices)
        np.testing.assert_allclose(result, expected_out)
        result = utils.segment_min_or_constant(
            data,
            segment_ids,
            indices_are_sorted=indices_are_sorted,
            unique_indices=unique_indices)
        num_unique_segments = jnp.max(segment_ids) + 1
        np.testing.assert_allclose(result, expected_out[:num_unique_segments])
    
    result = jax.jit(utils.segment_min_or_constant, static_argnums=(2, 3, 4))(
        data, segment_ids, num_segments, indices_are_sorted, unique_indices)
    np.testing.assert_allclose(result, expected_out)


@pytest.mark.parametrize("nan_data", [True, False])
def test_segment_softmax(nan_data):
    data = jnp.arange(9, dtype=jnp.float32)
    segment_ids = jnp.array([0, 1, 2, 0, 4, 0, 1, 1, 0])
    num_segments = 6
    expected_out = jnp.array([3.1741429e-04, 1.8088353e-03, 1.0000000e+00,
                              6.3754367e-03, 1.0000000e+00, 4.7108460e-02,
                              2.6845494e-01, 7.2973621e-01, 9.4619870e-01])
    if nan_data:
        data = data.at[0].set(jnp.nan)
        expected_out = expected_out.at[np.array([0, 3, 5, 8])].set(jnp.nan)
    
    with jax.disable_jit():
        result = utils.segment_softmax(data, segment_ids, num_segments)
        np.testing.assert_allclose(result, expected_out)
        result = utils.segment_softmax(data, segment_ids)
        np.testing.assert_allclose(result, expected_out)
    
    result = jax.jit(
        utils.segment_softmax, static_argnums=2)(data, segment_ids,
                                                 num_segments)
    np.testing.assert_allclose(result, expected_out)


def test_partition_softmax():
    data = jnp.arange(9)
    partitions = jnp.array([3, 2, 4])
    sum_partitions = 9
    expected_out = np.array([0.090031, 0.244728, 0.665241, 0.268941, 0.731059,
                            0.032059, 0.087144, 0.236883, 0.643914])
    
    with jax.disable_jit():
        result = utils.partition_softmax(data, partitions, sum_partitions)
        jax.tree_util.tree_map(
            functools.partial(np.testing.assert_allclose, atol=1E-5, rtol=1E-5),
            result, expected_out)
        result = utils.partition_softmax(data, partitions)
        jax.tree_util.tree_map(
            functools.partial(np.testing.assert_allclose, atol=1E-5, rtol=1E-5),
            result, expected_out)
    
    result = jax.jit(utils.partition_softmax, static_argnums=2)(
        data, partitions, sum_partitions)
    jax.tree_util.tree_map(
        functools.partial(np.testing.assert_allclose, atol=1E-5, rtol=1E-5),
        result, expected_out)


@pytest.mark.parametrize(
    "n_node,n_graph,nodes,globals_",
    [
        (1, 1, False, False),
        (5, 5, False, False),
        (1, 1, True, False),
        (5, 5, False, True),
        (5, 5, True, True),
        (0, 1, False, False),
        (1, 0, False, False),
    ],
)
def test_fully_connected_graph(n_node, n_graph, nodes, globals_):
    node_feat = np.random.rand(n_node*n_graph, 32) if nodes else None
    global_feat = np.random.rand(n_graph, 32) if globals_ else None
    
    with jax.disable_jit():
        result = utils.get_fully_connected_graph(
            n_node, n_graph, node_feat, global_feat)
        if nodes:
            assert len(result.nodes) == n_node*n_graph
        if globals_:
            assert len(result.globals) == n_graph
        assert len(result.senders) == n_node**2 * n_graph
        assert len(result.receivers) == n_node**2 * n_graph
        np.testing.assert_allclose(result.n_node, jnp.array([n_node] * n_graph))
    
    result = jax.jit(utils.get_fully_connected_graph, static_argnums=[0, 1])(
        n_node, n_graph, node_feat, global_feat)
    if nodes:
        assert len(result.nodes) == n_node*n_graph
    if globals_:
        assert len(result.globals) == n_graph
    assert len(result.senders) == n_node**2 * n_graph
    assert len(result.receivers) == n_node**2 * n_graph
    np.testing.assert_allclose(result.n_node, jnp.array([n_node] * n_graph))

    # Test senders_receiver_indices
    if n_node > 0:
        # [0, 1, ..., n_node - 1]
        node_indices = jnp.arange(n_node)
        # [0, 1,..., n_node - 1] + [0, 1,..., n_node - 1] + ... n_node times
        # [0,..., 0, 1,..., 1,..., n_node - 1,..., n_node - 1] each n_node times
        expected_senders = np.concatenate([node_indices] * n_node, axis=0)
        expected_receivers = np.stack(
            [node_indices] * n_node, axis=-1).reshape([-1])
    else:
        expected_senders = np.array([], dtype=np.int32)
        expected_receivers = np.array([], dtype=np.int32)

    # Check sender and receivers on each graph in the batch.
    for result_graph in utils.unbatch(result):
        np.testing.assert_allclose(result_graph.senders, expected_senders)
        np.testing.assert_allclose(result_graph.receivers, expected_receivers)


@pytest.mark.parametrize(
    "n_node,n_graph",
    [(1, 1), (5, 5), (0, 1), (1, 0)],
)
def test_fully_connected_graph_no_self_edges(n_node, n_graph):
    # `test_fully_connected_graph` already tests the case `add_self_edges=True`
    # so all that is left to test is that if we set `add_self_edges=False` we
    # get the same edges, except for the self-edges (although order may differ).
    graph_with_self_edges = utils.get_fully_connected_graph(
        n_node, n_graph, add_self_edges=True)
    graph_without_self_edges = utils.get_fully_connected_graph(
        n_node, n_graph, add_self_edges=False)

    # We will use sets to compare the order, since the order is not preserved
    # due to the usage of `np.roll` (e.g. if you remove the self edges after
    # add_self_edges=True, the remaining edges are in a different order than if
    # add_self_edges=False).
    send_recv_actual = {
        (s, r) for s, r in zip(
            np.asarray(graph_without_self_edges.senders),
            np.asarray(graph_without_self_edges.receivers))}

    # Remove the self edges by hand from `graph_with_self_edges`
    mask_self_edges = (
        graph_with_self_edges.senders == graph_with_self_edges.receivers)
    send_recv_expected = {
        (s, r) for s, r in zip(
            np.asarray(graph_with_self_edges.senders[~mask_self_edges]),
            np.asarray(graph_with_self_edges.receivers[~mask_self_edges]))}
    assert send_recv_actual == send_recv_expected


@pytest.mark.parametrize("add_self_edges", [True, False])
def test_fully_connected_graph_order_edges(add_self_edges):
    # This helps documenting the order of the output edges, so we are aware
    # in case we accidentally change it.
    graph_batch = utils.get_fully_connected_graph(
        n_node_per_graph=3,
        n_graph=1,
        add_self_edges=add_self_edges)

    if add_self_edges:
        np.testing.assert_array_equal(
            graph_batch.senders, [0, 1, 2] * 3)
        np.testing.assert_array_equal(
            graph_batch.receivers, [0] * 3 + [1] * 3 + [2] * 3)
    else:
        np.testing.assert_array_equal(graph_batch.senders, [1, 2, 2, 0, 0, 1])
        np.testing.assert_array_equal(graph_batch.receivers, [0, 0, 1, 1, 2, 2])


@pytest.mark.parametrize(
    "args_shapes,kwargs_shapes,axis",
    [
        ([], {'a': np.array([10, 2])}, -1),
        ([np.array([10, 5])], {'a': np.array([10, 2])}, -1),
        ([np.array([10, 5]), np.array([10, 3])], {'a': np.array([10, 2])}, -1),
        ([np.array([10, 5]), np.array([10, 3])], {}, -1),
        ([{'a': np.array([10, 2]), 'b': np.array([10, 4])}],
         {'c': np.array([10, 3])}, 1),
        ([{'a': np.array([2, 10]), 'b': np.array([4, 10])}],
         {'c': np.array([3, 10])}, 0)
    ],
)
def test_concatenated_args_wrapper(args_shapes, kwargs_shapes, axis):
    args = tree.tree_map(lambda x: np.random.randn(*x), args_shapes)
    kwargs = {k: np.random.randn(*shape) for k, shape in kwargs_shapes.items()}

    @utils.concatenated_args(axis=axis)
    def update_fn(feat):
        return feat

    out = update_fn(*args, **kwargs)
    expected_out = jnp.concatenate(
        list(tree.tree_flatten(args)[0]) + list(tree.tree_flatten(kwargs)[0]),
        axis=axis)
    np.testing.assert_allclose(out, expected_out)


@pytest.fixture(scope="module")
def setup_xla():
    """Setup XLA for dynamic batch tests."""
    os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
    xla_bridge.get_backend.cache_clear()
    yield
    # Cleanup would go here if needed


@pytest.fixture
def global_graph():
    return _make_dynamic_batch_graph(add_globals=True)


@pytest.fixture
def global_small_graph():
    return _make_dynamic_batch_graph(
        add_globals=True, num_nodes=(5, 7), num_edges=(6, 8))


@pytest.mark.parametrize(
    "use_globals,batch_kwargs",
    [
        (True, {
            'n_node': sum(_DB_NUM_NODES) + 1,
            'n_edge': sum(_DB_NUM_EDGES) + 100,
            'n_graph': len(_DB_NUM_EDGES) + 100
        }),
        (True, {
            'n_node': sum(_DB_NUM_NODES) + 100,
            'n_edge': sum(_DB_NUM_EDGES),
            'n_graph': len(_DB_NUM_EDGES) + 100
        }),
        (True, {
            'n_node': sum(_DB_NUM_NODES) + 100,
            'n_edge': sum(_DB_NUM_EDGES) + 100,
            'n_graph': len(_DB_NUM_EDGES) + 1
        }),
        (False, {
            # Add enough padding so not enough for a single extra graph.
            'n_node': sum(_DB_NUM_NODES) + 5,
            'n_edge': sum(_DB_NUM_EDGES) + 5,
            'n_graph': len(_DB_NUM_EDGES) + 5
        }),
    ],
)
def test_dynamically_batch(setup_xla, use_globals, batch_kwargs):
    def graph_iterator():
        graphs = [
            _make_dynamic_batch_graph(add_globals=use_globals) for x in range(4)]
        return iter(graphs + utils.unbatch_np(graphs[-1]))

    batched_dataset = utils.dynamically_batch(graph_iterator(),
                                              **batch_kwargs)
    batched_graphs = []
    while True:
        try:
            batched_graphs.append(next(batched_dataset))
        except StopIteration:
            break

    assert len(batched_graphs) == 5
    for batch_graphs in batched_graphs:
        batch_nodes = jax.tree_util.tree_flatten(batch_graphs.nodes)[0]
        for nodes in batch_nodes:
            assert nodes.shape[0] == batch_kwargs['n_node']
        batch_edges = jax.tree_util.tree_flatten(batch_graphs.edges)[0]
        for edges in batch_edges:
            assert edges.shape[0] == batch_kwargs['n_edge']
        assert len(batch_graphs.n_node) == batch_kwargs['n_graph']
        assert (utils.get_number_of_padding_with_graphs_nodes(batch_graphs) ==
                batch_kwargs['n_node'] - sum(_DB_NUM_NODES))
        assert (utils.get_number_of_padding_with_graphs_edges(batch_graphs) ==
                batch_kwargs['n_edge'] - sum(_DB_NUM_EDGES))


def test_too_big_graphs_tuple(setup_xla, global_graph, global_small_graph):
    # test_too_big_nodes
    iterator = utils.dynamically_batch(
        iter([global_graph]), n_node=15, n_edge=50, n_graph=10)
    with pytest.raises(RuntimeError, match='Found graph bigger than batch size.*'):
        next(iterator)
    
    # test_too_big_edges
    iterator = utils.dynamically_batch(
        iter([global_graph]), n_node=26, n_edge=15, n_graph=10)
    with pytest.raises(RuntimeError, match='Found graph bigger than batch size.*'):
        next(iterator)
    
    # test_too_big_graphs
    iterator = utils.dynamically_batch(
        iter([global_graph]), n_node=50, n_edge=50, n_graph=1)
    with pytest.raises(ValueError, match='The number of graphs*'):
        next(iterator)
    
    # test_too_big_fails_gracefully
    # Ensure that dynamically_batch() returns the accumulated batch before
    # raising an exception.
    iterator = utils.dynamically_batch(
        iter([global_small_graph, global_graph]),
        n_node=15, n_edge=15, n_graph=10)
    next(iterator)
    with pytest.raises(RuntimeError, match='Found graph bigger than batch size.*'):
        next(iterator)


def test_not_enough_graphs(setup_xla, global_graph):
    iterator = utils.dynamically_batch(
        iter([global_graph]), n_node=5, n_edge=5, n_graph=1)
    with pytest.raises(ValueError, match='The number of graphs*'):
        next(iterator)


def _assert_values_for_graph(padded_graph, wrapper):
    # Make padded graph values non zero.
    padded_graph = padded_graph._replace(
        nodes=tree.tree_map(lambda x: x - 1., padded_graph.nodes),
        edges=tree.tree_map(lambda x: x - 1., padded_graph.edges),
        globals=tree.tree_map(lambda x: x - 1., padded_graph.globals))
    true_valid_graph = utils.unbatch(padded_graph)[0]
    if wrapper:
        zeroed_graph_net = utils.with_zero_out_padding_outputs(lambda x: x)
        zeroed_padded_graph = zeroed_graph_net(padded_graph)
    else:
        zeroed_padded_graph = utils.zero_out_padding(padded_graph)
    graphs = utils.unbatch(zeroed_padded_graph)
    valid_graph = graphs[0]
    padding_graphs = graphs[1:]
    tree.tree_map(np.testing.assert_array_equal, valid_graph.nodes,
                  true_valid_graph.nodes)
    for padding_graph in padding_graphs:
        tree.tree_map(
            lambda x: np.testing.assert_array_equal(x, jnp.zeros_like(x)),
            padding_graph.nodes)


@pytest.mark.parametrize("wrapper", [True, False])
def test_zero_padding_values(wrapper):
    g = _get_random_graph(max_n_graph=1)
    
    # test_all_padded_features
    _assert_values_for_graph(
        utils.pad_with_graphs(g, n_node=20, n_edge=20, n_graph=3),
        wrapper=wrapper)
    
    # test_no_edge_features
    _assert_values_for_graph(
        utils.pad_with_graphs(
            g, n_node=sum(g.n_node) + 1, n_edge=sum(g.n_edge), n_graph=3),
        wrapper=wrapper)
    
    # test_no_extra_graph_features
    _assert_values_for_graph(
        utils.pad_with_graphs(
            g, n_node=sum(g.n_node) + 1, n_edge=sum(g.n_edge), n_graph=2),
        wrapper=wrapper)


def test_sparse_matrix_to_graphs_tuple():
    """Tests sparse COO matrix is correctly converted to a GraphsTuple."""
    _, sparse_adj_matrices, expected_graphs = _get_list_matrix()
    for (sparse_matrix,
         expected_graph) in zip(sparse_adj_matrices, expected_graphs):
        senders, receivers, values, n_node = sparse_matrix
        from_sparse_graph = utils.sparse_matrix_to_graphs_tuple(
            senders, receivers, values, n_node)
        jax.tree_util.tree_map(np.testing.assert_allclose,
                               from_sparse_graph, expected_graph)