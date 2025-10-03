### A basic GCN

```{code-cell}
import jraphzzz
import jax
import jax.numpy as jnp
```

```{code-cell}
import flax.linen as nn
```

```{code-cell}
import optax
from flax.training import train_state
```

```{code-cell}
ds = jraphzzz.Planetoid(root='datasets', name='Cora', split='public')
```

```{code-cell}
num_features = ds.num_features
num_classes = ds.num_classes
data_Cora = ds[0]  # Get the first graph object.
```

```{code-cell}
graph = data_Cora["graph"]

graph_train_mask = jnp.asarray([data_Cora["train_mask"]]).squeeze()
graph_val_mask = jnp.asarray([data_Cora["val_mask"]]).squeeze()
graph_test_mask = jnp.asarray([data_Cora["test_mask"]]).squeeze()
graph_labels = jnp.asarray([data_Cora["y"]]).squeeze()

one_hot_labels = jax.nn.one_hot(graph_labels, len(jnp.unique(graph_labels)))
```

```{code-cell}
def make_embed_fn(latent_size):
  def embed(inputs):
    return nn.Dense(latent_size)(inputs)
  return embed

def _attention_logit_fn(sender_attr: jnp.ndarray,
                        receiver_attr: jnp.ndarray,
                        edges: jnp.ndarray) -> jnp.ndarray:
  del edges
  x = jnp.concatenate((sender_attr, receiver_attr), axis=1)
  return nn.Dense(1)(x)

class GCN(nn.Module):
  """Defines a GAT network using FLAX

  Args:
    graph: GraphsTuple the network processes.

  Returns:
    output graph with updated node values.
  """
  gcn1_output_dim: int
  output_dim: int

  @nn.compact
  def __call__(self, x):
    gcn1 = jraphzzz.GraphConvolution(update_node_fn=lambda n: jax.nn.relu(make_embed_fn(self.gcn1_output_dim)(n)),
                          add_self_edges=True)
    gcn2 = jraphzzz.GraphConvolution(update_node_fn=nn.Dense(self.output_dim))
    return gcn2(gcn1(x))

model = GCN(8, len(jnp.unique(graph_labels)))
model

GCN(
    # attributes
    gcn1_output_dim = 8,
    output_dim = 7
)
```

```{code-cell}
optimizer = optax.adam(learning_rate=0.01)

rng, inp_rng, init_rng = jax.random.split(jax.random.PRNGKey(142), 3)
params = model.init(jax.random.PRNGKey(142),graph)

model_state = train_state.TrainState.create(apply_fn=model.apply,
                                            params=params,
                                            tx=optimizer)
```

```{code-cell}
def compute_loss(state, params, graph, labels, one_hot_labels, mask):
  """Computes loss."""
  pred_graph = state.apply_fn(params, graph)
  preds = jax.nn.log_softmax(pred_graph.nodes)
  loss = optax.softmax_cross_entropy(preds, one_hot_labels)
  loss_mask = jnp.sum(jnp.where(mask, loss, 0)) / jnp.sum(mask)

  pred_labels = jnp.argmax(preds, axis=1)
  acc = (pred_labels == labels)
  acc_mask = jnp.sum(jnp.where(mask, acc, 0)) / jnp.sum(mask)
  return loss_mask, acc_mask
```

```{code-cell}
@jax.jit  # Jit the function for efficiency
def train_step(state, graph, graph_labels, one_hot_labels, train_mask):
  # Gradient function
  grad_fn = jax.value_and_grad(compute_loss,  # Function to calculate the loss
                                argnums=1,  # Parameters are second argument of the function
                                has_aux=True  # Function has additional outputs, here accuracy
                              )
  # Determine gradients for current model, parameters and batch
  (loss, acc), grads = grad_fn(state, state.params, graph, graph_labels, one_hot_labels, train_mask)
  # Perform parameter update with gradients and optimizer
  state = state.apply_gradients(grads=grads)
  # Return state and any other value we might want
  return state, loss, acc

def train_model(state, graph, graph_labels, one_hot_labels, train_mask, val_mask, num_epochs):
  # Training loop
  for epoch in range(num_epochs):
    state, loss, acc = train_step(state, graph, graph_labels, one_hot_labels, train_mask)
    val_loss, val_acc = compute_loss(state, state.params, graph, graph_labels, one_hot_labels, val_mask)
    print(f'step: {epoch:03d}, train loss: {loss:.4f}, train acc: {acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}')
  return state, acc, val_acc
```

```{code-cell}
accuracy_list = []
```

```{code-cell}
trained_model_state, train_acc, val_acc = train_model(model_state, graph, graph_labels, one_hot_labels, graph_train_mask, graph_val_mask, num_epochs=200)
accuracy_list.append(['Cora', 'train', float(train_acc)])
accuracy_list.append(['Cora', 'valid', float(val_acc)])
```
