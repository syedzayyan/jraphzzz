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
!wget -P "./datasets" https://raw.githubusercontent.com/rdkit/rdkit/master/Docs/Book/data/solubility.train.sdf 
!wget -P "./datasets" https://raw.githubusercontent.com/rdkit/rdkit/master/Docs/Book/data/solubility.test.sdf
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
class MoleculeGCN(nnx.Module):
  def __init__(self, rngs: nnx.Rngs):
      self.rngs = rngs

  def __call__(self, graph: jraphzzz.GraphsTuple) -> jraphzzz.GraphsTuple:
      # ensure a globals field exists (1 feature per graph here)
      graph = graph._replace(globals=jnp.zeros([graph.n_node.shape[0], 1]))


      # infer dims (convert to int to avoid JAX tracers in prints)
      node_in = 0 if graph.nodes is None else int(graph.nodes.shape[1])
      edge_in = 0 if graph.edges is None else int(graph.edges.shape[1])
      global_in = 0 if graph.globals is None else int(graph.globals.shape[1])

      # Build per-slot embedders using inferred dims
      node_embed = nnx.Linear(node_in, 128, rngs=self.rngs) if node_in > 0 else (lambda x: x)
      edge_embed = nnx.Linear(edge_in, 128, rngs=self.rngs) if edge_in > 0 else (lambda x: x)
      global_embed = (
          nnx.Linear(global_in, 128, rngs=self.rngs) if global_in > 0 else (lambda x: x)
      )

      # ---- Construct update functions here so they can capture `rngs` ----
      # These create their Linear layers at *call* time but with rngs available.
      @jraphzzz.concatenated_args
      def edge_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
          net = nnx.Sequential(
              nnx.Linear(int(feats.shape[1]), 128, rngs=self.rngs),
              jax.nn.relu,
              nnx.Linear(128, 128, rngs=self.rngs),
          )
          return net(feats)

      @jraphzzz.concatenated_args
      def node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
          net = nnx.Sequential(
              nnx.Linear(int(feats.shape[1]), 128, rngs=self.rngs),
              jax.nn.relu,
              nnx.Linear(128, 128, rngs=self.rngs),
          )
          return net(feats)

      @jraphzzz.concatenated_args
      def update_global_fn(feats: jnp.ndarray) -> jnp.ndarray:
          net = nnx.Sequential(
              nnx.Linear(int(feats.shape[1]), 128, rngs=self.rngs),
              jax.nn.relu,
              nnx.Linear(128, 2, rngs=self.rngs),
          )
          return net(feats)

      # --------------------------------------------------------------------

      # IMPORTANT: GraphMapFeatures expects embedders in the order: (edges, nodes, globals)
      embedder = jraphzzz.GraphMapFeatures(edge_embed, node_embed, global_embed)

      net = jraphzzz.GraphNetwork(
          update_node_fn=node_update_fn,
          update_edge_fn=edge_update_fn,
          update_global_fn=update_global_fn,
      )

      embedded_graph = embedder(graph)  # should now succeed
      return net(embedded_graph)

# %%
# Adapted from https://github.com/deepmind/jraph/blob/master/jraph/ogb_examples/train.py
def _nearest_bigger_power_of_two(x: int) -> int:
  """Computes the nearest power of two greater than x for padding."""
  y = 2
  while y < x:
    y *= 2
  return y

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
def train(dataset: List[Dict[str, Any]], labels, num_train_steps: int):
    """Training loop."""
    # Initialize the network
    net = MoleculeGCN(nnx.Rngs(0))
    # Get a candidate graph to initialize
    graph = dataset[0]
    # Initialize network with forward pass
    _ = net(graph)
    # Create optimizer
    optimizer = nnx.Optimizer(net, optax.adam(0.01), wrt = nnx.Param)
    
    @nnx.jit
    def train_step(model, optimizer, graph, label):
        def loss_fn(model):
            # Get output graph from the model      
            pred_graph = model(graph)
            logits = pred_graph.globals  # Shape: (num_graphs, 2)
            preds = jax.nn.log_softmax(logits, axis=-1)  # Add axis=-1
            targets = jax.nn.one_hot(label, 2)
            
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
        batched_graphs = jraphzzz.batch([pad_graph_to_nearest_power_of_two(graph) for graph in dataset])
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
params = train(train_graphs, train_labels, num_train_steps=100)

# %%
def evaluate(dataset: List[Dict[str, Any]], graph_labels, net) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluation Script using the same style as the training loop."""
    
    @nnx.jit
    def eval_step(model, graph, label):
        output_graph = model(graph)
        logits = output_graph.globals
        
        if logits.ndim > 1:
            logits = jnp.squeeze(logits)
        
        loss = jnp.mean((logits - label) ** 2)
        acc = jnp.mean((logits > 0.5) == (label > 0.5))
        return loss, acc
    
    accumulated_loss = 0.0
    accumulated_accuracy = 0.0
    
    for idx, (graph, label) in enumerate(zip(dataset, graph_labels)):
        graph = pad_graph_to_nearest_power_of_two(graph)
        
        # Only append 0 if necessary
        if label.ndim == 0:
            label = jnp.concatenate([label.reshape(1,), jnp.array([0.0])])
        else:
            label = jnp.concatenate([label, jnp.array([0.0])])
        
        loss, acc = eval_step(net, graph, label)
        accumulated_loss += loss
        accumulated_accuracy += acc
        
        if idx % 100 == 0:
            print(f'Evaluated {idx + 1} graphs')
    
    num_items = len(dataset)
    avg_loss = accumulated_loss / num_items
    avg_accuracy = accumulated_accuracy / num_items
    
    print('Completed evaluation.')
    print(f'Eval loss: {avg_loss}, accuracy: {avg_accuracy}')
    
    return avg_loss, avg_accuracy


# %%
evaluate(test_graphs, test_labels, params)

# %%
