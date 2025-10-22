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
"""Tests for jraph.models."""

import jax
import jax.numpy as jnp
import jax.tree_util as tree

from jraphzzz.data import graph
from jraphzzz.nn.layers import GAT, DeepSets
from jraphzzz.nn.base import GraphNetwork
from jraphzzz.nn.models import (
    GraphNetGAT,
    InteractionNetwork,
    RelationNetwork,
    GraphMapFeatures,
)
from jraphzzz import utils
import numpy as np
import pytest


def _get_graph_network(graphs_tuple):
    # Our test update functions are just identity functions.
    def update_node_fn(n, se, re, g):
        return n

    def update_edge_fn(e, sn, rn, g):
        return e

    def update_global_fn(gn, ge, g):
        return g

    net = GraphNetwork(update_edge_fn, update_node_fn, update_global_fn)
    return net(graphs_tuple)


def _get_graph_network_no_global_update(graphs_tuple):
    # Our test update functions are just identity functions.
    def update_node_fn(n, se, re, g):
        return n

    def update_edge_fn(e, sn, rn, g):
        return e

    update_global_fn = None
    net = GraphNetwork(update_edge_fn, update_node_fn, update_global_fn)
    return net(graphs_tuple)


def _get_graph_network_no_node_update(graphs_tuple):
    # Our test update functions are just identity functions.
    update_node_fn = None

    def update_edge_fn(e, sn, rn, g):
        return e

    def update_global_fn(gn, ge, g):
        return g

    net = GraphNetwork(update_edge_fn, update_node_fn, update_global_fn)
    return net(graphs_tuple)


def _get_graph_network_no_edge_update(graphs_tuple):
    # Our test update functions are just identity functions.
    def update_node_fn(n, se, re, g):
        return n

    update_edge_fn = None

    def update_global_fn(gn, ge, g):
        return g

    net = GraphNetwork(update_edge_fn, update_node_fn, update_global_fn)
    return net(graphs_tuple)


def _get_attention_graph_network(graphs_tuple):
    # Our test update functions are just identity functions.
    def update_node_fn(n, se, re, g):
        return n

    def update_edge_fn(e, sn, rn, g):
        return e

    def update_global_fn(gn, ge, g):
        return g

    # Our attention logits are just one in this case.
    def attention_logit_fn(e, sn, rn, g):
        return jnp.array(1.0)

    # We use a custom apply function here, which just returns the edge unchanged.
    def attention_reduce_fn(e, w):
        return e

    net = GraphNetwork(
        update_edge_fn,
        update_node_fn,
        update_global_fn,
        attention_logit_fn=attention_logit_fn,
        attention_reduce_fn=attention_reduce_fn,
    )
    return net(graphs_tuple)


def _get_graph_gat(graphs_tuple):
    # Our test update functions are just identity functions.
    def update_node_fn(n, se, re, g):
        return n

    def update_edge_fn(e, sn, rn, g):
        return e

    def update_global_fn(gn, ge, g):
        return g

    # Our attention logits are just one in this case.
    def attention_logit_fn(e, sn, rn, g):
        return jnp.array(1.0)

    # We use a custom apply function here, which just returns the edge unchanged.
    def attention_reduce_fn(e, w):
        return e

    net = GraphNetGAT(
        update_edge_fn,
        update_node_fn,
        attention_logit_fn,
        attention_reduce_fn,
        update_global_fn,
    )
    return net(graphs_tuple)


def _get_multi_head_attention_graph_network(graphs_tuple):
    # Our test update functions are just identity functions.
    def update_node_fn(n, se, re, g):
        return n

    def update_global_fn(gn, ge, g):
        return g

    # With multi-head attention we have to return multiple edge features.
    # Here we define 3 heads, all with the same message.
    def update_edge_fn(e, unused_sn, unused_rn, unused_g):
        return tree.tree_map(lambda e_: jnp.stack([e_, e_, e_]), e)

    # Our attention logits are just the sum of the edge features of each head.
    def attention_logit_fn(e, unused_sn, unused_rn, unused_g):
        return tree.tree_map(lambda e_: jnp.sum(e_, axis=-1), e)

    # For multi-head attention we need a custom apply attention function.
    # In this we return the first edge.
    def attention_reduce_fn(e, unused_w):
        return tree.tree_map(lambda e_: e_[0], e)

    net = GraphNetwork(
        jax.vmap(update_edge_fn),
        jax.vmap(update_node_fn),
        update_global_fn,
        attention_logit_fn=jax.vmap(attention_logit_fn),
        attention_reduce_fn=jax.vmap(attention_reduce_fn),
    )
    return net(graphs_tuple)


def _get_interaction_network(graphs_tuple):
    def update_node_fn(n, r):
        return jnp.concatenate((n, r), axis=-1)

    def update_edge_fn(e, s, r):
        return jnp.concatenate((e, s, r), axis=-1)

    out = InteractionNetwork(update_edge_fn, update_node_fn)(graphs_tuple)
    nodes, edges, receivers, senders, _, _, _ = graphs_tuple
    expected_edges = jnp.concatenate((edges, nodes[senders], nodes[receivers]), axis=-1)
    aggregated_nodes = utils.segment_sum(
        expected_edges, receivers, num_segments=len(graphs_tuple.nodes)
    )
    expected_nodes = jnp.concatenate((nodes, aggregated_nodes), axis=-1)
    expected_out = graphs_tuple._replace(edges=expected_edges, nodes=expected_nodes)
    return out, expected_out


def _get_graph_independent(graphs_tuple):
    def embed_fn(x):
        return x * 2

    out = GraphMapFeatures(embed_fn, embed_fn, embed_fn)(graphs_tuple)
    expected_out = graphs_tuple._replace(
        nodes=graphs_tuple.nodes * 2,
        edges=graphs_tuple.edges * 2,
        globals=graphs_tuple.globals * 2,
    )
    return out, expected_out


def _get_relation_network(graphs_tuple):
    def edge_fn(s, r):
        return jnp.concatenate((s, r), axis=-1)

    def global_fn(e):
        return e * 2

    out = RelationNetwork(edge_fn, global_fn)(graphs_tuple)
    expected_edges = jnp.concatenate(
        (
            graphs_tuple.nodes[graphs_tuple.senders],
            graphs_tuple.nodes[graphs_tuple.receivers],
        ),
        axis=-1,
    )
    num_graphs = len(graphs_tuple.n_edge)
    edge_gr_idx = jnp.repeat(
        jnp.arange(num_graphs),
        graphs_tuple.n_edge,
        total_repeat_length=graphs_tuple.edges.shape[0],
    )
    aggregated_edges = utils.segment_sum(
        expected_edges, edge_gr_idx, num_segments=num_graphs
    )
    expected_out = graphs_tuple._replace(
        edges=expected_edges, globals=aggregated_edges * 2
    )
    return out, expected_out


def _get_deep_sets(graphs_tuple):
    def node_fn(n, g):
        return jnp.concatenate((n, g), axis=-1)

    def global_fn(n):
        return n * 2

    out = DeepSets(node_fn, global_fn)(graphs_tuple)
    num_graphs = len(graphs_tuple.n_node)
    num_nodes = len(graphs_tuple.nodes)
    broadcasted_globals = jnp.repeat(
        graphs_tuple.globals, graphs_tuple.n_node, total_repeat_length=num_nodes, axis=0
    )
    expected_nodes = jnp.concatenate((graphs_tuple.nodes, broadcasted_globals), axis=-1)
    node_gr_idx = jnp.repeat(
        jnp.arange(num_graphs), graphs_tuple.n_node, total_repeat_length=num_nodes
    )
    expected_out = graphs_tuple._replace(
        nodes=expected_nodes,
        globals=utils.segment_sum(expected_nodes, node_gr_idx, num_segments=num_graphs)
        * 2,
    )
    return out, expected_out


def _get_gat(graphs_tuple):
    # With multi-head attention we have to return multiple edge features.
    # Here we define 3 heads, all with the same message.
    def attention_query_fn(n):
        return tree.tree_map(lambda n_: jnp.stack([n_, n_, n_], axis=2), n)

    # Our attention logits 1 if a self edge
    def attention_logit_fn(s, r, e_):
        del e_
        return (s == r) * 1 + (s != r) * -1e10

    def node_update_fn(nodes):
        return jnp.mean(nodes, axis=2)

    net = GAT(attention_query_fn, attention_logit_fn, node_update_fn)

    # Cast nodes to floats since GAT will output floats from the softmax
    # attention.
    graphs_tuple = graphs_tuple._replace(
        nodes=jnp.array(graphs_tuple.nodes, jnp.float32)
    )
    return net(graphs_tuple), graphs_tuple


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
    ]

    return list_graphs, batched_graph


@pytest.mark.parametrize(
    "network_fn",
    [
        _get_graph_network,
        _get_graph_network_no_node_update,
        _get_graph_network_no_edge_update,
        _get_graph_network_no_global_update,
        _get_attention_graph_network,
        _get_multi_head_attention_graph_network,
        _get_graph_gat,
    ]
)
def test_connect_graphnetwork(network_fn):
    _, batched_graphs_tuple = _get_list_and_batched_graph()

    with jax.disable_jit():
        out = network_fn(batched_graphs_tuple)
        jax.tree_util.tree_map(np.testing.assert_allclose, out, batched_graphs_tuple)

    out = jax.jit(network_fn)(batched_graphs_tuple)
    jax.tree_util.tree_map(np.testing.assert_allclose, out, batched_graphs_tuple)


@pytest.mark.parametrize(
    "network_fn",
    [
        _get_graph_network,
        _get_graph_network_no_node_update,
        _get_graph_network_no_edge_update,
        _get_graph_network_no_global_update,
    ],
)
def test_connect_graphnetwork_nones(network_fn):
    batched_graphs_tuple = graph.GraphsTuple(
        n_node=jnp.array([1, 3, 1, 0, 2, 0, 0]),
        n_edge=jnp.array([2, 5, 0, 0, 1, 0, 0]),
        nodes=_make_nest(jnp.arange(14).reshape(7, 2)),
        edges=_make_nest(jnp.arange(24).reshape(8, 3)),
        globals=_make_nest(jnp.arange(14).reshape(7, 2)),
        senders=jnp.array([0, 0, 1, 1, 2, 3, 3, 6]),
        receivers=jnp.array([0, 0, 2, 1, 3, 2, 1, 5]),
    )

    for name, graphs_tuple in [
        ("no_globals", batched_graphs_tuple._replace(globals=None)),
        ("empty_globals", batched_graphs_tuple._replace(globals=[])),
        ("no_edges", batched_graphs_tuple._replace(edges=None)),
        ("empty_edges", batched_graphs_tuple._replace(edges=[])),
    ]:

        out = network_fn(graphs_tuple)
        jax.tree_util.tree_map(np.testing.assert_allclose, out, graphs_tuple)
        
        with jax.disable_jit():
            out = jax.jit(network_fn)(graphs_tuple)
            jax.tree_util.tree_map(np.testing.assert_allclose, out, graphs_tuple)


@pytest.mark.parametrize(
    "network_fn",
    [
        _get_interaction_network,
        _get_graph_independent,
        _get_gat,
        _get_relation_network,
        _get_deep_sets,
    ],
)
def test_connect_gnns(network_fn):
    batched_graphs_tuple = graph.GraphsTuple(
        n_node=jnp.array([1, 3, 1, 0, 2, 0, 0]),
        n_edge=jnp.array([1, 7, 1, 0, 3, 0, 0]),
        nodes=jnp.arange(14).reshape(7, 2),
        edges=jnp.arange(36).reshape(12, 3),
        globals=jnp.arange(14).reshape(7, 2),
        senders=jnp.array([0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 3, 6]),
        receivers=jnp.array([0, 1, 2, 3, 4, 5, 6, 2, 3, 2, 1, 5]),
    )
    with jax.disable_jit():
        out, expected_out = network_fn(batched_graphs_tuple)
        jax.tree_util.tree_map(np.testing.assert_allclose, out, expected_out)

    out, expected_out = jax.jit(network_fn)(batched_graphs_tuple)
    jax.tree_util.tree_map(np.testing.assert_allclose, out, expected_out)


# def test_graphnetwork_attention_error(self):
#     with self.assertRaisesRegex(
#         ValueError,
#         ("attention_logit_fn and attention_reduce_fn must both be supplied."),
#     ):
#         GraphNetwork(
#             update_edge_fn=None,
#             update_node_fn=None,
#             attention_logit_fn=lambda x: x,
#             attention_reduce_fn=None,
#         )
#     with self.assertRaisesRegex(
#         ValueError,
#         ("attention_logit_fn and attention_reduce_fn must both be supplied."),
#     ):
#         GraphNetwork(
#             update_edge_fn=None,
#             update_node_fn=None,
#             attention_logit_fn=None,
#             attention_reduce_fn=lambda x: x,
#         )
