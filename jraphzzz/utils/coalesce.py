import jax.numpy as jnp
import numpy as np

def _coalesce_undirected_edges(senders, receivers, edges):
    """Canonicalize undirected edges (min,max), coalesce duplicates and sum edge features if provided.
    Returns:
      uniq_senders, uniq_receivers, uniq_edges, mapping_old_to_new (numpy int array)
    """
    # canonical endpoints as tuple lists (min,max)
    a = jnp.minimum(senders, receivers)
    b = jnp.maximum(senders, receivers)
    # stack and convert to 2D NumPy for unique operation (unique not stable on jnp for axis=0)
    pairs = jnp.stack([a, b], axis=1)
    pairs_np = np.asarray(pairs)
    # use numpy unique to get mapping (it's fine inside python)
    uniq_pairs, idx_inv = np.unique(pairs_np, axis=0, return_inverse=True)
    uniq_senders_np = uniq_pairs[:, 0].astype(np.int32)
    uniq_receivers_np = uniq_pairs[:, 1].astype(np.int32)

    uniq_edges = None
    if edges is not None:
        edges_np = np.asarray(edges)
        # sum edges over duplicates according to idx_inv
        uniq_edges_np = np.zeros((uniq_pairs.shape[0],) + edges_np.shape[1:], dtype=edges_np.dtype)
        for old_idx, new_idx in enumerate(idx_inv):
            uniq_edges_np[new_idx] += edges_np[old_idx]
        uniq_edges = jnp.asarray(uniq_edges_np)

    return (jnp.asarray(uniq_senders_np, dtype=jnp.int32),
            jnp.asarray(uniq_receivers_np, dtype=jnp.int32),
            uniq_edges,
            jnp.asarray(idx_inv, dtype=jnp.int32))

