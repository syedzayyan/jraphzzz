# %% [markdown]
# # Molecule Solubility

# %%
import jraphzzz
import random
from rdkit import Chem
import jax.numpy as jnp
import flax.nnx as nnx
import jax
from typing import Any, Dict, List, Tuple
import optax

# %%
def load_ds(train_sdf: str, test_sdf: str):
    train_mols = [m for m in Chem.SDMolSupplier(train_sdf) if m is not None]
    test_mols = [m for m in Chem.SDMolSupplier(test_sdf) if m is not None]

    sol_cls_dict = {"(A) low": 0, "(B) medium": 1, "(C) high": 2}

    def extract_label(mol):
        for prop in ("CLASS", "solubility_class", "SOL_CLASS", "class"):
            if mol.HasProp(prop):
                v = mol.GetProp(prop)
                if v in sol_cls_dict:
                    return sol_cls_dict[v]
                try:
                    return int(float(v))
                except Exception:
                    return 0
        return random.randint(0, 1)

    train_graphs = [jraphzzz.from_smiles(Chem.MolToSmiles(m)) for m in train_mols]
    train_labels = jnp.array([extract_label(m) for m in train_mols], dtype=jnp.float32)

    test_graphs = [jraphzzz.from_smiles(Chem.MolToSmiles(m)) for m in test_mols]
    test_labels = jnp.array([extract_label(m) for m in test_mols], dtype=jnp.float32)

    return (train_graphs, train_labels), (test_graphs, test_labels)

# %%
TRAIN_SDF = "./datasets/solubility.train.sdf"
TEST_SDF = "./datasets/solubility.test.sdf"
(train_graphs, train_labels), (test_graphs, test_labels) = load_ds(TRAIN_SDF, TEST_SDF)

# %%
train_graphs[0]

# %%
def _nearest_bigger_power_of_two(x: int) -> int:
  """Computes the nearest power of two greater than x for padding."""
  y = 2
  while y < x: 
    y *= 2
  return y


# %%
def pad_graph_to_nearest_power_of_two(
    graphs_tuple: jraphzzz.GraphsTuple) -> jraphzzz.GraphsTuple:
  """Pads a batched `GraphsTuple` to the nearest power of two.
  For example, if a `GraphsTuple` has 7 nodes, 5 edges and 3 graphs, this method
  would pad the `GraphsTuple` nodes and edges:
    7 nodes --> 8 nodes (2^3)
    5 edges --> 8 edges (2^3)
  And since padding is accomplished using `jraph.pad_with_graphs`, an extra
  graph and node is added:
    8 nodes --> 9 nodes
    3 graphs --> 4 graphs
  Args:
    graphs_tuple: a batched `GraphsTuple` (can be batch size 1).
  Returns:
    A graphs_tuple batched to the nearest power of two.
  """
  # Add 1 since we need at least one padding node for pad_with_graphs.
  pad_nodes_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_node)) + 1
  pad_edges_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_edge))
  # Add 1 since we need at least one padding graph for pad_with_graphs.
  # We do not pad to nearest power of two because the batch size is fixed.
  pad_graphs_to = graphs_tuple.n_node.shape[0] + 1
  return jraphzzz.pad_with_graphs(graphs_tuple, pad_nodes_to, pad_edges_to,
                               pad_graphs_to)

# %%
_nearest_bigger_power_of_two(jnp.sum(train_graphs[0].n_node)) + 1

# %%
def pad_graphs_to_max_size(
    graphs_list: list[jraphzzz.GraphsTuple]) -> list[jraphzzz.GraphsTuple]:
    """Pads a list of batched `GraphsTuple`s to the maximum size across all graphs.
    
    Finds the maximum number of nodes, edges, and graphs across all GraphsTuples
    in the list, then pads each GraphsTuple to match these maximum dimensions.
    
    Args:
        graphs_list: a list of batched `GraphsTuple` objects.
    
    Returns:
        A list of graphs_tuples padded to the maximum dimensions.
    """
    # Find maximum dimensions across all graphs
    pad_nodes_to = _nearest_bigger_power_of_two(max(int(jnp.sum(g.n_node)) for g in train_graphs))
    pad_edges_to = _nearest_bigger_power_of_two(max(int(jnp.sum(g.n_edge)) for g in train_graphs))
    
    # Pad each graph to the maximum dimensions
    padded_graphs = []
    for graph in graphs_list:
        pad_graphs_to = graph.n_node.shape[0] + 1
        padded_graph = jraphzzz.pad_with_graphs(
            graph, pad_nodes_to, pad_edges_to, pad_graphs_to
        )
        padded_graphs.append(padded_graph)
    
    return padded_graphs, {
        "max_nodes": pad_nodes_to,
        "max_edges" : pad_edges_to,
    }

# %%
padded_graphs, metadata = pad_graphs_to_max_size(train_graphs + test_graphs)

padded_train_graphs = padded_graphs[:len(train_graphs)]
padded_test_graphs = padded_graphs[len(test_graphs):]

# %%
class MoleculeGCN(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, node_in: int, edge_in: int, global_in: int):
        self.rngs = rngs

        # Fixed embedders for GraphMapFeatures
        self.node_embed = nnx.Linear(node_in, 128, rngs=self.rngs) if node_in > 0 else (lambda x: x)
        self.edge_embed = nnx.Linear(edge_in, 128, rngs=self.rngs) if edge_in > 0 else (lambda x: x)
        self.global_embed = nnx.Linear(global_in, 128, rngs=self.rngs) if global_in > 0 else (lambda x: x)

        # Fixed update MLPs for GraphNetwork
        self.edge_update_fn = nnx.Sequential(
            nnx.Linear(128 * 4, 128, rngs=self.rngs),
            jax.nn.relu,
            nnx.Linear(128, 128, rngs=self.rngs),
        )

        self.node_update_fn = nnx.Sequential(
            nnx.Linear(128 * 4, 128, rngs=self.rngs),
            jax.nn.relu,
            nnx.Linear(128, 128, rngs=self.rngs),
        )

        self.update_global_fn = nnx.Sequential(
            nnx.Linear(128 * 3, 128, rngs=self.rngs),
            jax.nn.relu,
            nnx.Linear(128, 3, rngs=self.rngs),
        )

    def __call__(self, graph: jraphzzz.GraphsTuple) -> jraphzzz.GraphsTuple:
        # Ensure a globals field exists
        graph = graph._replace(globals=jnp.zeros([graph.n_node.shape[0], 1]))

        # GraphMapFeatures expects (edges, nodes, globals)
        embedder = jraphzzz.GraphMapFeatures(
            self.edge_embed, self.node_embed, self.global_embed
        )

        @jraphzzz.concatenated_args
        def edge_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
            return self.edge_update_fn(feats)

        @jraphzzz.concatenated_args
        def node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
            return self.node_update_fn(feats)

        @jraphzzz.concatenated_args
        def update_global_fn(feats: jnp.ndarray) -> jnp.ndarray:
            return self.update_global_fn(feats)



        net = jraphzzz.GraphNetwork(
            update_node_fn=node_update_fn,
            update_edge_fn=edge_update_fn,
            update_global_fn=update_global_fn,
        )

        embedded_graph = embedder(graph)
        return net(embedded_graph)


# %%
gxn = MoleculeGCN(nnx.Rngs(0), 9, 3, 1)

# %%
def train(dataset: List[Dict[str, Any]], labels, num_train_steps: int):
    """Training loop."""
    # Initialize the network
    net = MoleculeGCN(nnx.Rngs(0), 9, 3, 1)
    # Create optimizer
    optimizer = nnx.Optimizer(net, optax.adam(1e-5), wrt = nnx.Param)
    
    @nnx.jit
    def train_step(model, optimizer, graph, label):
        def loss_fn(model):
            # Get output graph from the model      
            pred_graph = model(graph)
            logits = pred_graph.globals  # Shape: (num_graphs, 2)
            preds = jax.nn.log_softmax(logits, axis=-1)  # Add axis=-1
            targets = jax.nn.one_hot(label, 3)
            
            # Mask for padded graphs
            mask = jraphzzz.get_graph_padding_mask(pred_graph)
            
            # Cross entropy loss - sum over valid samples only, then divide by count
            loss = -jnp.sum(preds * targets * mask[:, None]) / jnp.sum(mask)
            
            # Accuracy taking into account the mask
            accuracy = jnp.sum(
                (jnp.argmax(logits, axis=1) == label) * mask) / jnp.sum(mask)
            
            return loss, accuracy
        
        (loss, acc), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads)  # FIXED: Don't pass model here
        return loss, acc
    
    for idx in range(num_train_steps):
        batched_graphs = jraphzzz.batch(dataset)
        labels_array = jnp.array(labels)
        num_batched_graphs = batched_graphs.n_node.shape[0]
        num_original_graphs = len(labels)
        num_padding_graphs = num_batched_graphs - num_original_graphs
        padded_labels = jnp.concatenate([
            labels_array, 
            jnp.zeros(num_padding_graphs, dtype=labels_array.dtype)
        ])
        
        loss, acc = train_step(net, optimizer, batched_graphs, padded_labels)
        
        if idx % 10 == 0:  # Print more often to see the trend
            print(f'step: {idx}, loss: {loss:.4f}, acc: {acc:.4f}')
    
    print('Training finished')
    return net


# %%
params = train(padded_train_graphs, train_labels, num_train_steps=100)

# %%
def evaluate(dataset: List[Dict[str, Any]], labels, net) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluation Script using the same style as the training loop."""
    
    # Batch all graphs and pad to nearest power of two
    batched_graphs = jraphzzz.batch(dataset)
    
    labels_array = jnp.array(labels)
    num_batched_graphs = batched_graphs.n_node.shape[0]
    num_original_graphs = len(labels)
    num_padding_graphs = num_batched_graphs - num_original_graphs
    
    # Pad labels with zeros if necessary
    padded_labels = jnp.concatenate([
        labels_array,
        jnp.zeros(num_padding_graphs, dtype=labels_array.dtype)
    ])
    
    @nnx.jit
    def eval_step(model, graph, label):
        pred_graph = model(graph)
        logits = pred_graph.globals  # Shape: (num_graphs, num_classes)
        preds = jax.nn.log_softmax(logits, axis=-1)
        targets = jax.nn.one_hot(label, logits.shape[-1])
        
        mask = jraphzzz.get_graph_padding_mask(pred_graph)
        
        # Loss over valid graphs only
        loss = -jnp.sum(preds * targets * mask[:, None]) / jnp.sum(mask)
        
        # Accuracy over valid graphs only
        accuracy = jnp.sum((jnp.argmax(logits, axis=1) == label) * mask) / jnp.sum(mask)
        
        return loss, accuracy

    # Evaluate on all graphs at once
    loss, acc = eval_step(net, batched_graphs, padded_labels)
    
    print('Completed evaluation.')
    print(f'Eval loss: {loss:.4f}, accuracy: {acc:.4f}')
    
    return loss, acc


# %%
evaluate(padded_test_graphs, test_labels, params)
