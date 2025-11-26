import jax.numpy as jnp
import jax
import numpy as np
from ..data.graph import GraphsTuple
from .coalesce import _coalesce_undirected_edges
from .self_loops import remove_self_loops

def line_graphs_tuple(graphs: GraphsTuple, *, force_directed: bool = False) -> GraphsTuple:
    """
    Convert a jraph.GraphsTuple to its line-graph (also a GraphsTuple).
    Uses add_self_loops/remove_self_loops for convenience where needed.
    """
    senders = graphs.senders
    receivers = graphs.receivers
    edges = graphs.edges
    globals_ = graphs.globals
    n_node = np.asarray(graphs.n_node)
    n_edge = np.asarray(graphs.n_edge)

    n_graphs = int(n_node.shape[0])
    node_offsets = np.cumsum(np.concatenate([[0], n_node[:-1]]), dtype=np.int32)
    edge_offsets = np.cumsum(np.concatenate([[0], n_edge[:-1]]), dtype=np.int32)

    new_nodes_list = []
    new_senders_chunks = []
    new_receivers_chunks = []
    new_n_node = []
    new_n_edge = []
    new_globals = []

    for g in range(n_graphs):
        node_off = node_offsets[g]
        edge_off = edge_offsets[g]
        num_nodes_g = int(n_node[g])
        num_edges_g = int(n_edge[g])

        # empty graph -> empty line graph
        if num_edges_g == 0:
            new_nodes_list.append(None if edges is None else jnp.zeros((0,) + jnp.asarray(edges).shape[1:], dtype=jnp.asarray(edges).dtype))
            new_senders_chunks.append(jnp.array([], dtype=jnp.int32))
            new_receivers_chunks.append(jnp.array([], dtype=jnp.int32))
            new_n_node.append(0)
            new_n_edge.append(0)
            new_globals.append(None if globals_ is None else jax.tree_map(lambda x: x[g:g+1], globals_))
            continue

        s = senders[edge_off:edge_off + num_edges_g] - node_off
        r = receivers[edge_off:edge_off + num_edges_g] - node_off
        e_attr = None if edges is None else edges[edge_off:edge_off + num_edges_g]

        if force_directed:
            # Directed: create an edge e1 -> e2 if receiver(e1) == sender(e2)
            s_np = np.asarray(s, dtype=np.int32)
            r_np = np.asarray(r, dtype=np.int32)
            outgoing_lists = [[] for _ in range(num_nodes_g)]
            for ei, sv in enumerate(s_np):
                outgoing_lists[int(sv)].append(int(ei))

            line_s = []
            line_r = []
            for ej in range(num_edges_g):
                target_node = int(r_np[ej])
                outgoing = outgoing_lists[target_node]
                for ok in outgoing:
                    line_s.append(ej)
                    line_r.append(ok)

            new_senders = jnp.asarray(line_s, dtype=jnp.int32) if len(line_s) else jnp.array([], dtype=jnp.int32)
            new_receivers = jnp.asarray(line_r, dtype=jnp.int32) if len(line_r) else jnp.array([], dtype=jnp.int32)
            new_n_nodes_g = num_edges_g
            new_node_features = None if e_attr is None else e_attr

        else:
            # Undirected: canonicalise and coalesce reciprocal edges (sum features)
            uniq_s, uniq_r, uniq_edges, idx_inv = _coalesce_undirected_edges(s, r, e_attr)
            num_uniq_edges = int(uniq_s.shape[0])

            # Build incident lists for each original node
            incident = [[] for _ in range(num_nodes_g)]
            for uniq_e_idx in range(num_uniq_edges):
                u = int(uniq_s[uniq_e_idx])
                v = int(uniq_r[uniq_e_idx])
                incident[u].append(uniq_e_idx)
                if v != u:
                    incident[v].append(uniq_e_idx)

            line_s = []
            line_r = []
            for incident_edges in incident:
                L = len(incident_edges)
                if L <= 1:
                    continue
                for i_idx in range(L):
                    for j_idx in range(L):
                        ei = incident_edges[i_idx]
                        ej = incident_edges[j_idx]
                        if ei == ej:
                            continue
                        line_s.append(ei)
                        line_r.append(ej)

            if len(line_s) == 0:
                new_senders = jnp.array([], dtype=jnp.int32)
                new_receivers = jnp.array([], dtype=jnp.int32)
            else:
                pairs_np = np.stack([np.asarray(line_s), np.asarray(line_r)], axis=1)
                uniq_pairs = np.unique(pairs_np, axis=0)
                new_senders = jnp.asarray(uniq_pairs[:, 0], dtype=jnp.int32)
                new_receivers = jnp.asarray(uniq_pairs[:, 1], dtype=jnp.int32)

            # Remove any self-loops that might remain using your remove_self_loops helper
            ns, nr, _ = remove_self_loops(new_senders, new_receivers, None)
            new_senders = ns
            new_receivers = nr

            new_n_nodes_g = num_uniq_edges
            new_node_features = None if uniq_edges is None else uniq_edges

        # Append per-graph result (we will offset node IDs when concatenating)
        new_nodes_list.append(new_node_features)
        new_senders_chunks.append(new_senders)
        new_receivers_chunks.append(new_receivers)
        new_n_node.append(int(new_n_nodes_g))
        new_n_edge.append(int(new_senders.shape[0]))
        new_globals.append(None if globals_ is None else jax.tree_map(lambda x, idx=g: x[idx], globals_))

    # Concatenate node features if any
    if new_nodes_list and new_nodes_list[0] is not None:
        nodes_concat = jnp.concatenate([n if n is not None else jnp.zeros((0,) + jnp.asarray(new_nodes_list[0]).shape[1:], dtype=jnp.asarray(new_nodes_list[0]).dtype) for n in new_nodes_list], axis=0)
    else:
        nodes_concat = None

    # Offset senders/receivers across batch
    sender_chunks = []
    receiver_chunks = []
    node_offset = 0
    for nn, ss, rr in zip(new_n_node, new_senders_chunks, new_receivers_chunks):
        if ss is None or ss.shape[0] == 0:
            sender_chunks.append(jnp.array([], dtype=jnp.int32))
            receiver_chunks.append(jnp.array([], dtype=jnp.int32))
        else:
            sender_chunks.append((ss + node_offset).astype(jnp.int32))
            receiver_chunks.append((rr + node_offset).astype(jnp.int32))
        node_offset += nn

    all_senders = jnp.concatenate(sender_chunks, axis=0) if any([c.shape[0] for c in sender_chunks]) else jnp.array([], dtype=jnp.int32)
    all_receivers = jnp.concatenate(receiver_chunks, axis=0) if any([c.shape[0] for c in receiver_chunks]) else jnp.array([], dtype=jnp.int32)

    # Build new globals (best-effort; preserve None)
    new_globals_concat = None
    if globals_ is not None:
        try:
            new_globals_concat = jax.tree_map(lambda x: jnp.stack([g for g in new_globals], axis=0), globals_)
        except Exception:
            new_globals_concat = None

    n_node_arr = jnp.asarray(new_n_node, dtype=jnp.int32)
    n_edge_arr = jnp.asarray(new_n_edge, dtype=jnp.int32)

    # As in the PyG implementation: edge features of the resulting line-graph are None.
    new_edges_feat = None

    return GraphsTuple(
        nodes=nodes_concat,
        edges=new_edges_feat,
        senders=all_senders,
        receivers=all_receivers,
        globals=new_globals_concat,
        n_node=n_node_arr,
        n_edge=n_edge_arr
    )
