from dataclasses import dataclass
from typing import Dict, Tuple, Callable, Optional, Any

import jax.numpy as jnp
from jaxtyping import Array

from ...data.temporal_graph import TemporalGraphsTuple
from ...utils.scatter import scatter
from ...utils.utils import segment_softmax, segment_sum
import jax.lax as lax

import jax

TGNMessageStoreType = Dict[int, Tuple[Array, Array, Array, Array]]


# @dataclass
# class TGNMemoryState:
#     memory: Array          # [num_nodes, memory_dim]
#     last_update: Array     # [num_nodes]
#     msg_s_store: TGNMessageStoreType
#     msg_d_store: TGNMessageStoreType

# def compute_msg(
#     state: TGNMemoryState,
#     n_id: Array,
#     msg_store: TGNMessageStoreType,
#     msg_module: Callable,
#     time_encoder: Callable[[Array], Array],
# ):
#     data = [msg_store[int(i)] for i in n_id.tolist()]
#     src, dst, t, raw_msg = list(zip(*data))

#     src = jnp.concatenate(src, axis=0)
#     dst = jnp.concatenate(dst, axis=0)
#     t   = jnp.concatenate(t,   axis=0)
#     raw_msg = [m for k, m in enumerate(raw_msg) if m.size > 0 or k == 0]
#     raw_msg = jnp.concatenate(raw_msg, axis=0)

#     t_rel = t - state.last_update[src]
#     t_enc = time_encoder(t_rel.astype(raw_msg.dtype))

#     msg = msg_module(state.memory[src], state.memory[dst], raw_msg, t_enc)
#     return msg, t, src, dst

# def get_updated_memory(
#     state: TGNMemoryState,
#     n_id: Array,
#     num_nodes: int,
#     message_module: Callable,
#     aggr_module: Callable,
#     gru: Callable,
#     time_encoder: Callable[[Array], Array],
# ) -> TGNMemoryState:
#     # assoc: global node id -> local index in n_id
#     assoc = -jnp.ones((num_nodes,), dtype=jnp.int32)
#     assoc = assoc.at[n_id].set(jnp.arange(n_id.shape[0], dtype=jnp.int32))

#     msg_s, t_s, src_s, dst_s = compute_msg(
#         state, n_id, state.msg_s_store, message_module, time_encoder
#     )
#     msg_d, t_d, src_d, dst_d = compute_msg(
#         state, n_id, state.msg_d_store, message_module, time_encoder
#     )

#     idx = jnp.concatenate([src_s, src_d], axis=0)
#     msg = jnp.concatenate([msg_s, msg_d], axis=0)
#     t   = jnp.concatenate([t_s, t_d], axis=0)

#     aggr = aggr_module(msg, assoc[idx], t, n_id.shape[0])
#     new_local_memory = gru(aggr, state.memory[n_id])

#     # Note: your scatter signature is scatter(input, dim, index, src, reduce)
#     last_update_all = scatter(
#         input=state.last_update,
#         dim=0,
#         index=idx,
#         src=t,
#         reduce="max",
#     )
#     new_local_last_update = last_update_all[n_id]

#     memory = state.memory.at[n_id].set(new_local_memory)
#     last_update = state.last_update.at[n_id].set(new_local_last_update)

#     return TGNMemoryState(
#         memory=memory,
#         last_update=last_update,
#         msg_s_store=state.msg_s_store,
#         msg_d_store=state.msg_d_store,
#     )

# def update_msg_store(
#     msg_store: TGNMessageStoreType,
#     src: Array,
#     dst: Array,
#     t: Array,
#     raw_msg: Array,
# ) -> TGNMessageStoreType:
#     n_id, perm = jnp.sort(src), jnp.argsort(src)
#     unique_ids, counts = jnp.unique(n_id, return_counts=True)

#     new_store = dict(msg_store)
#     start = 0
#     for i, c in zip(unique_ids.tolist(), counts.tolist()):
#         end = start + c
#         idx = perm[start:end]
#         new_store[int(i)] = (src[idx], dst[idx], t[idx], raw_msg[idx])
#         start = end
#     return new_store

# def reset_state(raw_msg_dim: int, num_nodes: int, memory_dim: int) -> TGNMemoryState:
#     memory = jnp.zeros((num_nodes, memory_dim))
#     last_update = jnp.zeros((num_nodes,), dtype=jnp.int64)

#     empty_i   = jnp.zeros((0,), dtype=jnp.int32)
#     empty_msg = jnp.zeros((0, raw_msg_dim))

#     msg_s = {j: (empty_i, empty_i, empty_i, empty_msg) for j in range(num_nodes)}
#     msg_d = {j: (empty_i, empty_i, empty_i, empty_msg) for j in range(num_nodes)}

#     return TGNMemoryState(
#         memory=memory,
#         last_update=last_update,
#         msg_s_store=msg_s,
#         msg_d_store=msg_d,
#     )

# def update_state(
#     state: TGNMemoryState,
#     src: Array,
#     dst: Array,
#     t: Array,
#     raw_msg: Array,
#     training: bool,
#     num_nodes: int,
#     message_module: Callable,
#     aggr_module: Callable,
#     gru: Callable,
#     time_encoder: Callable[[Array], Array],
# ) -> TGNMemoryState:
#     n_id = jnp.unique(jnp.concatenate([src, dst], axis=0))

#     if training:
#         state = get_updated_memory(
#             state, n_id, num_nodes, message_module, aggr_module, gru, time_encoder
#         )
#         msg_s = update_msg_store(state.msg_s_store, src, dst, t, raw_msg)
#         msg_d = update_msg_store(state.msg_d_store, dst, src, t, raw_msg)
#         state = TGNMemoryState(state.memory, state.last_update, msg_s, msg_d)
#     else:
#         msg_s = update_msg_store(state.msg_s_store, src, dst, t, raw_msg)
#         msg_d = update_msg_store(state.msg_d_store, dst, src, t, raw_msg)
#         state = TGNMemoryState(state.memory, state.last_update, msg_s, msg_d)
#         state = get_updated_memory(
#             state, n_id, num_nodes, message_module, aggr_module, gru, time_encoder
#         )

#     return state

# Generic temporal aggregator:
# (graph, n_id, aux_state) -> (graph_out, aux_state_out)
TemporalAggregator = Callable[
    [TemporalGraphsTuple, Array, Any],
    Tuple[TemporalGraphsTuple, Any]
]



def identity_message(z_src: Array, z_dst: Array, raw_msg: Array,
            t_enc: Array):
    return jnp.concatenate([z_src, z_dst, raw_msg, t_enc], axis=-1)

def last_aggregator(
    msg: Array, index: Array, t: Array, dim_size: int
) -> Array:
    """
    msg:   [E, D]
    index: [E] (destination indices in [0, dim_size))
    t:     [E] (timestamps)
    """
    # argmax over t per index, like scatter_argmax
    # build an initial argmax index array of -1
    init = -jnp.ones((dim_size,), dtype=jnp.int32)

    def body(acc, i):
        idx = index[i]
        # if this is the first or newer time, update
        cond = (acc[idx] < 0) | (t[i] > t[acc[idx]])
        new_acc = acc.at[idx].set(jnp.where(cond, i, acc[idx]))
        return new_acc, None

    argmax, _ = lax.scan(body, init, jnp.arange(t.shape[0]))

    out = jnp.zeros((dim_size, msg.shape[-1]), dtype=msg.dtype)
    mask = argmax >= 0
    out = out.at[mask].set(msg[argmax[mask]])
    return out

def mean_aggregator(
    msg: Array, index: Array, t: Array, dim_size: int
) -> Array:
    # Pure mean over msgs per index; t is ignored (kept for API compat)
    out = scatter(
        input=jnp.zeros((dim_size, msg.shape[-1]), dtype=msg.dtype),
        dim=0,
        index=index,
        src=msg,
        reduce="mean",
    )
    return out

def time_encoder(t: Array, time_encode_func: Callable) -> Array:
    return jnp.cos(time_encode_func(t.reshape(-1, 1)))

def temporal_neighbors_for_node(node, cut_time, num_neighbors,
                                senders, receivers, times, MAX_DEGREE = 1000):
    # 1) incident edges
    mask_inc = (senders == node) | (receivers == node)
    idx_inc = jnp.nonzero(mask_inc, size=MAX_DEGREE, fill_value=-1)[0]

    # drop padded -1s
    valid_inc_mask = (idx_inc >= 0)
    idx_inc = idx_inc[valid_inc_mask]

    # 2) filter by time
    inc_times = times[idx_inc]
    mask_time = (inc_times < cut_time)
    idx_valid = idx_inc[mask_time]
    valid_times = inc_times[mask_time]

    # if none, just return all -1
    def no_neighbors(_):
        return (
            jnp.full((num_neighbors,), -1, dtype=jnp.int32),
            jnp.full((num_neighbors,), -1, dtype=times.dtype),
            jnp.full((num_neighbors,), -1, dtype=jnp.int32),
        )

    def some_neighbors(_):
        # sort by time, take last K
        sort_idx = jnp.argsort(valid_times)
        idx_sorted = idx_valid[sort_idx]
        times_sorted = valid_times[sort_idx]

        k = num_neighbors
        n = idx_sorted.shape[0]
        start = jnp.maximum(n - k, 0)
        idx_lastk = idx_sorted[start:]
        times_lastk = times_sorted[start:]

        # pad at the front if fewer than k
        pad = k - idx_lastk.shape[0]
        idx_padded = jnp.pad(idx_lastk, (pad, 0), constant_values=-1)
        times_padded = jnp.pad(times_lastk, (pad, 0), constant_values=-1)

        # neighbor nodes: from senders/receivers; here choose "the other end"
        src = senders[idx_padded]
        dst = receivers[idx_padded]
        neigh_nodes = jnp.where(src == node, dst, src)

        return neigh_nodes, times_padded, idx_padded

    has_any = idx_valid.shape[0] > 0
    neigh_nodes, neigh_times, neigh_edge_indices = jax.lax.cond(
        has_any, some_neighbors, no_neighbors, operand=None
    )
    return neigh_nodes, neigh_times, neigh_edge_indices




def TemporalGATBase(
    num_nodes: int,
    time_encoder: Callable,
    attention_layer: Callable,
    last_linear_layer: Callable,
    memory_aggregator_module: Callable
):
 
    def ApplyTemporalGATBase(graph: TemporalGraphsTuple):
        node_feats, times, senders, receivers = graph.nodes, graph.times, graph.senders, graph.receivers

        # 1) Time encodings
        t_enc = time_encoder(times)

        # 2) Build attention messages
        src_h = node_feats[senders]
        dst_h = node_feats[receivers]
        
        msg_ij, log_alpha_ij = attention_layer(
            src_h, dst_h, t_enc
        )  # msg_ij: [E, d_h]

        # 3) Normalize attention per receiver
        # jraph.segment_softmax expects data and indices
        alpha_ij = segment_softmax(
            log_alpha_ij,
            receivers,
            num_segments=node_feats.shape[0],
        )  # [E]

        weighted_msgs = msg_ij * alpha_ij[:, None]  # [E, d_h]

        # 4) Aggregate over incoming edges
        agg_msgs = segment_sum(
            weighted_msgs,
            receivers,
            num_segments=node_feats.shape[0],
        )  # [N, d_h]

        # 5) Node update
        new_nodes = last_linear_layer(agg_msgs, node_feats)

        return graph._replace(nodes=new_nodes)
 
    return ApplyTemporalGATBase
