# %% [markdown]
# # Basic GCN on the Cora Dataset

# %% [markdown]
# This notebook implements a Graph Convolutional Network (GCN) for node classification on the Cora citation network. The Cora dataset consists of scientific papers (nodes) connected by citation relationships (edges), where each paper is classified into one of seven research topics.
# 
# ## What You'll Learn
# - Loading and preprocessing graph datasets with jraphzzz
# - Building a 2-layer GCN using Flax NNX
# - Training graph neural networks with JAX
# - Evaluating node classification performance

# %%
import jraphzzz
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

# %% [markdown]
# ## Loading the Cora Dataset
# 
# The Planetoid loader provides three standard benchmark datasets (Cora, CiteSeer, PubMed). We're using:
# - **Cora**: 2,708 scientific papers with 5,429 citation links
# - Each paper has a 1,433-dimensional bag-of-words feature vector
# - Papers are classified into 7 categories (e.g., Neural Networks, Probabilistic Methods)
# - The "public" split provides the standard train/validation/test division used in research
# 

# %%
ds = jraphzzz.Planetoid(root="datasets", name="Cora", split="public")

# %% [markdown]
# ## Dataset Metadata
# 
# We extract key information about the dataset structure:
# - **num_features**: Dimensionality of input node features (1,433 for Cora)
# - **num_classes**: Number of output categories to predict (7 for Cora)
# - **data_Cora[0]**: The graph object containing nodes, edges, features, and labels
# 

# %%
num_features = ds.num_features
num_classes = ds.num_classes
data_Cora = ds[0]  # Get the first graph object.

# %%
graph = data_Cora["graph"]
graph_train_mask = jnp.asarray([data_Cora["train_mask"]]).squeeze()
graph_val_mask = jnp.asarray([data_Cora["val_mask"]]).squeeze()
graph_test_mask = jnp.asarray([data_Cora["test_mask"]]).squeeze()
graph_labels = jnp.asarray([data_Cora["y"]]).squeeze()
one_hot_labels = jax.nn.one_hot(graph_labels, len(jnp.unique(graph_labels)))

# %% [markdown]
# ## GCN Model Architecture
# 
# Our Graph Convolutional Network consists of two GCN layers:
# 
# **Layer 1 (GCN + ReLU):**
# - Input: Node features (1,433 dimensions)
# - Output: Hidden representations (8 dimensions)
# - Aggregates information from 1-hop neighbors
# - Includes self-loops (nodes consider their own features)
# 
# **Layer 2 (GCN):**
# - Input: Hidden representations (8 dimensions)
# - Output: Class logits (7 dimensions, one per class)
# - Further aggregates information (now 2-hop neighborhood)

# %%
class GCN(nnx.Module):
    """Defines a GAT network using FLAX

    Args:
      graph: GraphsTuple the network processes.

    Returns:
      output graph with updated node values.
    """
    def __init__(self, n_nodes: int, gcn1_output_dim: int, output_dim: int, rngs: nnx.Rngs):
        self.gcn1_output_dim = gcn1_output_dim
        self.output_dim = output_dim
        self.rngs = rngs
        self.convolution_linear1 = nnx.Linear(n_nodes, gcn1_output_dim, rngs = rngs)
        self.convolution_linear2 = nnx.Linear(gcn1_output_dim, output_dim, rngs = rngs)
        

    def __call__(self, graphs: jraphzzz.GraphsTuple):
        gcn1 = jraphzzz.GraphConvolution(
            update_node_fn=self.convolution_linear1,
            add_self_edges=True,
        )
        gcn2 = jraphzzz.GraphConvolution(update_node_fn=self.convolution_linear2)
        return gcn2(gcn1(graphs))

# %%
model = GCN(graph.nodes.shape[1], 8, len(jnp.unique(graph_labels)), nnx.Rngs(0))

# %%
def compute_loss(model, graph, labels, one_hot_labels, mask):
    """Computes loss."""
    pred_graph = model(graph)
    preds = jax.nn.log_softmax(pred_graph.nodes)
    loss = optax.softmax_cross_entropy(preds, one_hot_labels)
    loss_mask = jnp.sum(jnp.where(mask, loss, 0)) / jnp.sum(mask)

    pred_labels = jnp.argmax(preds, axis=1)
    acc = pred_labels == labels
    acc_mask = jnp.sum(jnp.where(mask, acc, 0)) / jnp.sum(mask)
    return loss_mask, acc_mask

# %%
@nnx.jit  # Jit the function for efficiency
def train_step(model, optimizer, graph, graph_labels, one_hot_labels, train_mask):
    # Gradient function
    grad_fn = nnx.value_and_grad(
        compute_loss,  # Function to calculate the loss
        argnums=0,  # Parameters are first argument of the function
        has_aux=True,  # Function has additional outputs, here accuracy
    )
    # Determine gradients for current model, parameters and batch
    (loss, acc), grads = grad_fn(
        model, graph, graph_labels, one_hot_labels, train_mask
    )
    # Perform parameter update with gradients and optimizer
    optimizer.update(model, grads)
    # Return state and any other value we might want
    return model, loss, acc

# %%
def train_model(
    model, graph, graph_labels, one_hot_labels, train_mask, val_mask, num_epochs
):
    # Training loop
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt = nnx.Param)

    for epoch in range(num_epochs):
        state, loss, acc = train_step(
            model, optimizer, graph, graph_labels, one_hot_labels, train_mask
        )
        val_loss, val_acc = compute_loss(
            state, graph, graph_labels, one_hot_labels, val_mask
        )
        print(
            f"step: {epoch:03d}, train loss: {loss:.4f}, train acc: {acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}"
        )
    return state, acc, val_acc

# %%
trained_model_state, train_acc, val_acc = train_model(
    model,
    graph,
    graph_labels,
    one_hot_labels,
    graph_train_mask,
    graph_val_mask,
    num_epochs=1000,
)

# %%



