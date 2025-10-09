Quick Start
===========

Representing Graphs - The ``GraphsTuple``
------------------------------------------

Jraph takes inspiration from the Tensorflow `graph_nets library <https://github.com/deepmind/graph_nets>`_
in defining a ``GraphsTuple`` data structure, which is a ``namedtuple`` that contains
one or more directed graphs.

.. code-block:: python

  import jraph
  import jax.numpy as jnp

  # Define a three node graph, each node has an integer as its feature.
  node_features = jnp.array([[0.], [1.], [2.]])

  # We will construct a graph fro which there is a directed edge between each node
  # and its successor. We define this with `senders` (source nodes) and `receivers`
  # (destination nodes).
  senders = jnp.array([0, 1, 2])
  receivers = jnp.array([1, 2, 0])

  # You can optionally add edge attributes.
  edges = jnp.array([[5.], [6.], [7.]])

  # We then save the number of nodes and the number of edges.
  # This information is used to make running GNNs over multiple graphs
  # in a GraphsTuple possible.
  n_node = jnp.array([3])
  n_edge = jnp.array([3])

  # Optionally you can add `global` information, such as a graph label.

  global_context = jnp.array([[1]]) # Same feature dimensions as nodes and edges.
  graph = jraph.GraphsTuple(nodes=node_features, senders=senders, receivers=receivers,
  edges=edges, n_node=n_node, n_edge=n_edge, globals=global_context)

A ``GraphsTuple`` can have more than one graph.

.. code-block:: python

  two_graph_graphstuple = jraph.batch([graph, graph])


The ``node`` and ``edge`` features are stacked on the leading axis.

.. code-block:: python

  jraph.batch([graph, graph]).nodes
  >> DeviceArray([[0.],
               [1.],
               [2.],
               [0.],
               [1.],
               [2.]], dtype=float32)


You can tell which nodes are from which graph by looking at ``n_node``.

.. code-block:: python

  jraph.batch([graph, graph]).n_node
  >> DeviceArray([3, 3], dtype=int32)


You can store nests of features in ``nodes``, ``edges`` and ``globals``. This makes
it possible to store multiple sets of features for each node, edge or graph, with
potentially different types and semantically different meanings (for example
'training' and 'testing' nodes). The only requirement if that all arrays within
each nest must have a common leading dimensions size, matching the total number
of nodes, edges or graphs within the ``Graphstuple`` respectively.

.. code-block:: python

  node_targets = jnp.array([[True], [False], [True]])
  graph = graph._replace(nodes={'inputs': graph.nodes, 'targets': node_targets})


Using the Model Zoo
-------------------

Jraph provides a set of implemented reference models for you to use.

A Jraph model defines a message passing algorithm between the nodes, edges and
global attributes of a graph. The user defines ``update`` functions that update graph features, which are typically neural networks but can be arbitrary jax functions.

Let's go through a ``GraphNetwork`` [(paper)](https://arxiv.org/abs/1806.01261) example.
A GraphNetwork's first update function updates the edges using ``edge`` features,
the node features of the ``sender`` and ``receiver`` and the ``global`` features.


.. code-block:: python

  # As one example, we just pass the edge features straight through.
  def update_edge_fn(edge, sender, receiver, globals_):
    return edge


Often we use the concatenation of these features, and ``jraph`` provides an easy
way of doing this with the ``concatenated_args`` decorator.

.. code-block:: python

  @jraph.concatenated_args
  def update_edge_fn(concatenated_features):
    return concatenated_features


Typically, a learned model such as a Multi-Layer Perceptron is used within an
update function.

The user similarly defines functions that update the nodes and globals. These
are then used to configure a `GraphNetwork`. To see the arguments to the node
and global `update_fns` please take a look at the model zoo.

.. code-block:: python

  net = jraph.GraphNetwork(update_edge_fn=update_edge_fn,
                           update_node_fn=update_node_fn,
                           update_global_fn=update_global_fn)


``net`` is a function that sends messages according to the ``GraphNetwork`` algorithm
and applies the ``update_fn``. It takes a graph, and returns a graph.

.. code-block:: python

  updated_graph = net(graph)

Contribute
----------

Please read ``CONTRIBUTING.md``.

- Issue tracker: https://github.com/deepmind/jraph/issues
- Source code: https://github.com/deepmind/jraph/tree/master

Support
-------

If you are having issues, please let us know by filing an issue on our
`issue tracker <https://github.com/deepmind/jraph/issues>`_.

License
-------

Jraph is licensed under the Apache 2.0 License.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`