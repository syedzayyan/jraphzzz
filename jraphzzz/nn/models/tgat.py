# jraph_tgan.py
from typing import Optional, Callable, Tuple

import jax.numpy as jnp
import flax.nnx as nnx
import jraphzzz


# ----------------------------
# Utility: MergeLayer
# ----------------------------
class MergeLayer(nnx.Module):
    dim1: int
    dim2: int
    dim3: int
    dim4: int

    @nn.compact
    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        # x1, x2: [..., dim1] and [..., dim2] shapes
        x = jnp.concatenate([x1, x2], axis=-1)
        h = nnx.relu(nnx.Dense(self.dim3)(x))
        out = nnx.Dense(self.dim4)(h)
        return out


# ----------------------------
# Scaled Dot-Product Attention
# ----------------------------
class ScaledDotProductAttention(nnx.Module):
    temperature: float

    @nn.compact
    def __call__(self, q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, mask: Optional[jnp.ndarray] = None):
        # q: [B, Lq, d_k]; k: [B, Lk, d_k]; v: [B, Lv, d_v]
        attn_logits = jnp.einsum('bqd,bkd->bqk', q, k) / self.temperature
        if mask is not None:
            # mask shape should broadcast to [B, Lq, Lk], masked positions are True => set to -inf
            attn_logits = jnp.where(mask, -1e10, attn_logits)
        attn = nn.softmax(attn_logits, axis=-1)
        out = jnp.einsum('bqk,bkv->bqv', attn, v)
        return out, attn


# ----------------------------
# Multi-head attention
# ----------------------------
class MultiHeadAttention(nn.Module):
    n_head: int
    d_model: int
    d_k: int
    d_v: int
    dropout_rate: float = 0.1
    deterministic: bool = True

    @nn.compact
    def __call__(self, q, k, v, mask=None):
        B, Lq, _ = q.shape
        _, Lk, _ = k.shape
        n_head = self.n_head
        d_k, d_v = self.d_k, self.d_v

        w_q = nnx.Dense(n_head * d_k, use_bias=False, name='w_q')(q)
        w_k = nnx.Dense(n_head * d_k, use_bias=False, name='w_k')(k)
        w_v = nnx.Dense(n_head * d_v, use_bias=False, name='w_v')(v)

        # reshape to [B, n_head, L, head_dim]
        def split_heads(x, head_dim):
            x = x.reshape(B, -1, n_head, head_dim)
            return jnp.transpose(x, (0, 2, 1, 3))  # [B, n_head, L, head_dim]

        qh = split_heads(w_q, d_k)
        kh = split_heads(w_k, d_k)
        vh = split_heads(w_v, d_v)

        # merge batch & heads for dot product computation: [B * n_head, L, head_dim]
        qh = qh.reshape(B * n_head, Lq, d_k)
        kh = kh.reshape(B * n_head, Lk, d_k)
        vh = vh.reshape(B * n_head, Lk, d_v)

        # mask must be expanded to (B * n_head, Lq, Lk) if provided
        if mask is not None:
            mask = jnp.repeat(mask, repeats=n_head, axis=0)

        out, attn = ScaledDotProductAttention(temperature=jnp.sqrt(d_k))(qh, kh, vh, mask=mask)
        out = out.reshape(B, n_head, Lq, d_v)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(B, Lq, n_head * d_v)  # [B, Lq, n_head*d_v]

        out = nnx.Dense(self.d_model, name='out_fc')(out)
        out = nnx.LayerNorm()(out + q)  # residual + norm
        return out, attn.reshape(B, n_head, Lq, Lk)  # attn shaped [B, n_head, Lq, Lk]


# ----------------------------
# Map-based multi-head (custom affinity)
# ----------------------------
class MapBasedMultiHeadAttention(nnx.Module):
    n_head: int
    d_model: int
    d_k: int
    d_v: int
    dropout_rate: float = 0.1
    deterministic: bool = True

    @nn.compact
    def __call__(self, q, k, v, mask=None):
        # q: [B, Lq, D], k, v: [B, Lk, D]
        B, Lq, _ = q.shape
        _, Lk, _ = k.shape
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        wq = nnx.Dense(n_head * d_k, use_bias=False)(q)
        wk = nnx.Dense(n_head * d_k, use_bias=False)(k)
        wv = nnx.Dense(n_head * d_v, use_bias=False)(v)

        # reshape -> [B, n_head, L, head_dim]
        def reshape_heads(x, head_dim):
            return x.reshape(B, -1, n_head, head_dim).transpose((0, 2, 1, 3))

        qh = reshape_heads(wq, d_k)
        kh = reshape_heads(wk, d_k)
        vh = reshape_heads(wv, d_v)

        # create pairwise concatenation per head and compute learned score
        # qh: [B, n_head, Lq, d_k], kh: [B, n_head, Lk, d_k]
        # we want attn logits shaped [B * n_head, Lq, Lk]
        qh_exp = jnp.expand_dims(qh, axis=3)  # [B, n_head, Lq, 1, d_k]
        kh_exp = jnp.expand_dims(kh, axis=2)  # [B, n_head, 1, Lk, d_k]
        pair = jnp.concatenate([qh_exp.repeat(Lk, axis=3), kh_exp.repeat(Lq, axis=2)], axis=-1)
        # pair: [B, n_head, Lq, Lk, 2*d_k]
        # pass through a small MLP (linear) to produce scalar scores
        score_layer = nnx.Dense(1, use_bias=False)
        logits = score_layer(pair).squeeze(-1)  # [B, n_head, Lq, Lk]

        if mask is not None:
            # mask expected shape [B, 1, Lq, Lk] or broadcastable
            logits = jnp.where(mask, -1e10, logits)

        attn = nnx.softmax(logits, axis=-1)  # [B, n_head, Lq, Lk]
        # apply dropout? (skipped; can use nn.Dropout with deterministic flag)
        # compute weighted sum over v per head:
        # rearrange vh -> [B, n_head, Lk, d_v], attn -> [B, n_head, Lq, Lk]
        out_per_head = jnp.einsum('bhqk,bhkd->bhqd', attn, vh)  # [B, n_head, Lq, d_v]
        out = out_per_head.transpose((0, 2, 1, 3)).reshape(B, Lq, n_head * d_v)
        out = nnx.relu(nnx.Dense(self.d_model)(out))
        out = nnx.LayerNorm()(out + q)
        return out, attn


# ----------------------------
# Time encoders
# ----------------------------
class TimeEncode(nnx.Module):
    expand_dim: int
    factor: float = 5.0

    def setup(self):
        freqs = 1.0 / (10 ** jnp.linspace(0., 9., self.expand_dim))
        self.basis_freq = self.param('basis_freq', lambda rng, shape: freqs)
        self.phase = self.param('phase', lambda rng, shape: jnp.zeros(self.expand_dim))

    def __call__(self, ts: jnp.ndarray):
        # ts: [B, L] (float)
        B, L = ts.shape
        ts_ = ts[..., None]  # [B, L, 1]
        map_ts = ts_ * self.basis_freq[None, None, :] + self.phase[None, None, :]
        return jnp.cos(map_ts)  # [B, L, expand_dim]


class PosEncode(nnx.Module):
    expand_dim: int
    seq_len: int

    def setup(self):
        self.embedding = nnx.Embed(num_embeddings=self.seq_len, features=self.expand_dim)

    def __call__(self, ts: jnp.ndarray):
        # convert timestamps to ordering -> use argsort like behaviour is non-trivial in jax
        # here assume `ts` already contains integer indices indicating positions 0..(seq_len-1)
        idx = jnp.argsort(ts, axis=1)  # returns positions; still OK but watch differentiability
        return self.embedding(idx)


class EmptyEncode(nnx.Module):
    expand_dim: int

    def __call__(self, ts: jnp.ndarray):
        B, L = ts.shape
        return jnp.zeros((B, L, self.expand_dim))


# ----------------------------
# MeanPool (simple)
# ----------------------------
class MeanPool(nnx.Module):
    feat_dim: int
    edge_dim: int

    def setup(self):
        self.merger = MergeLayer(self.edge_dim + self.feat_dim, self.feat_dim, self.feat_dim, self.feat_dim)

    def __call__(self, src, src_t, seq, seq_t, seq_e, mask):
        # seq: [B, N, D]; seq_e: [B, N, De]
        seq_x = jnp.concatenate([seq, seq_e], axis=-1)
        hn = jnp.mean(seq_x, axis=1)  # [B, De + D]
        out = self.merger(hn, src)
        return out, None


# ----------------------------
# AttnModel using MultiHead / MapBased
# ----------------------------
class AttnModel(nnx.Module):
    feat_dim: int
    edge_dim: int
    time_dim: int
    attn_mode: str = 'prod'
    n_head: int = 2
    dropout: float = 0.1

    def setup(self):
        self.edge_in_dim = self.feat_dim + self.edge_dim + self.time_dim
        self.merger = MergeLayer(self.edge_in_dim, self.feat_dim, self.feat_dim, self.feat_dim)

        if self.attn_mode == 'prod':
            self.attn = MultiHeadAttention(n_head=self.n_head,
                                           d_model=self.edge_in_dim,
                                           d_k=self.edge_in_dim // self.n_head,
                                           d_v=self.edge_in_dim // self.n_head,
                                           dropout_rate=self.dropout)
        elif self.attn_mode == 'map':
            self.attn = MapBasedMultiHeadAttention(n_head=self.n_head,
                                                   d_model=self.edge_in_dim,
                                                   d_k=self.edge_in_dim // self.n_head,
                                                   d_v=self.edge_in_dim // self.n_head,
                                                   dropout_rate=self.dropout)
        else:
            raise ValueError('attn_mode must be "prod" or "map"')

    def __call__(self, src, src_t, seq, seq_t, seq_e, mask):
        # shapes: src: [B, D], src_t: [B, Dt], seq: [B, N, D], seq_t: [B, N, Dt], seq_e: [B, N, De], mask: [B, N] (bool)
        B, N, D = seq.shape
        src_ext = src[:, None, :]  # [B, 1, D]
        src_e_ph = jnp.zeros((B, 1, self.edge_dim))
        q = jnp.concatenate([src_ext, src_e_ph, src_t[:, None, :]], axis=-1)  # [B, 1, edge_in_dim]
        k = jnp.concatenate([seq, seq_e, seq_t], axis=-1)  # [B, N, edge_in_dim]

        # mask -> [B, 1, N] with True for masked positions
        mask_ = mask[:, None, :]
        # attn expect mask shape [B, Lq, Lk] -> here [B,1,N]
        out, attn = self.attn(q, k, k, mask=mask_)
        out = out.squeeze(axis=1)  # [B, edge_in_dim]
        out = self.merger(out, src)
        attn = attn.squeeze(axis=1)  # reduce Lq dim -> may be [B, n_head, N]
        return out, attn


# ----------------------------
# TGAN-ish wrapper with tem_conv using jraph message passing
# ----------------------------
class TGANJraph(nnx.Module):
    node_feat_table: jnp.ndarray  # pretrained node raw features as numpy/jnp array [V, D]
    edge_feat_table: jnp.ndarray  # pretrained edge raw features [E, D]
    num_layers: int = 3
    attn_mode: str = 'prod'
    use_time: str = 'time'
    seq_len: Optional[int] = None
    n_head: int = 2

    def setup(self):
        self.n_feat_th = self.variable('consts', 'n_feat_th', lambda: jnp.asarray(self.node_feat_table))
        self.e_feat_th = self.variable('consts', 'e_feat_th', lambda: jnp.asarray(self.edge_feat_table))
        self.feat_dim = self.n_feat_th.value.shape[1]

        self.merge_layer = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, self.feat_dim)

        # aggregator per layer
        self.agg_layers = [AttnModel(self.feat_dim, self.feat_dim, self.feat_dim,
                                     attn_mode=self.attn_mode, n_head=self.n_head)
                           for _ in range(self.num_layers)]

        if self.use_time == 'time':
            self.time_encoder = TimeEncode(expand_dim=self.feat_dim)
        elif self.use_time == 'pos':
            assert self.seq_len is not None
            self.time_encoder = PosEncode(expand_dim=self.feat_dim, seq_len=self.seq_len)
        else:
            self.time_encoder = EmptyEncode(expand_dim=self.feat_dim)

        self.affinity = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1)

    def __call__(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):
        """
        src_idx_l, target_idx_l: numpy arrays shaped [B] of node indices
        cut_time_l: numpy array shaped [B] of cut times (floats)
        """
        # convert incoming numpy arrays to jnp
        src_idx_l = jnp.asarray(src_idx_l).astype(jnp.int32)
        target_idx_l = jnp.asarray(target_idx_l).astype(jnp.int32)
        cut_time_l = jnp.asarray(cut_time_l).astype(jnp.float32)

        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        score = self.affinity(src_embed, target_embed).squeeze(-1)
        return score

    def tem_conv(self, src_idx_l, cut_time_l, curr_layers, num_neighbors):
        # NOTE: here we must rely on a neighbor finder that returns numpy arrays:
        # (B, num_neighbors) node ids, (B, num_neighbors) edge ids, (B, num_neighbors) timestamps
        # For this example we assume you pass them already or have a callable available.
        B = src_idx_l.shape[0]
        device = None

        # convert node indices -> features
        src_node_feat = self.n_feat_th.value[src_idx_l]  # [B, D]
        # src time embedding: query node time delta is zero
        src_node_t_embed = self.time_encoder(jnp.zeros((B, 1)))

        if curr_layers == 0:
            return src_node_feat

        # Recursively obtain neighbor conv features: here we expect an external neighbor finder function
        # This function must run outside of jax or be jittable; easiest is to precompute neighbors in numpy
        # For demo, assume neighbor_finder is available on module: self.get_temporal_neighbor(...)
        src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.get_temporal_neighbor(src_idx_l, cut_time_l, num_neighbors)

        # Convert to jnp
        src_ngh_node_batch = jnp.asarray(src_ngh_node_batch).astype(jnp.int32)  # [B, N]
        src_ngh_eidx_batch = jnp.asarray(src_ngh_eidx_batch).astype(jnp.int32)
        src_ngh_t_batch = jnp.asarray(src_ngh_t_batch).astype(jnp.float32)  # absolute neighbour times

        # relative delta
        src_ngh_t_batch_delta = cut_time_l[:, None] - src_ngh_t_batch  # [B, N]
        # flatten to compute prev layer features for neighbor nodes
        flat_nodes = src_ngh_node_batch.reshape(-1)
        flat_times = src_ngh_t_batch_delta.reshape(-1)
        # recursively compute neighbor embeddings from prev layer:
        prev_layer_feat = self.tem_conv(flat_nodes, flat_times, curr_layers - 1, num_neighbors)  # returns [B*N, D]
        src_ngh_feat = prev_layer_feat.reshape(B, num_neighbors, -1)

        # edge feature lookup and time enc
        src_ngn_edge_feat = self.e_feat_th.value[src_ngh_eidx_batch]  # [B, N, De]
        src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_delta)  # [B, N, Dt]

        # mask: neighbor index == 0 is padding (borrowed from your code)
        mask = (src_ngh_node_batch == 0)  # [B, N] boolean

        attn_m = self.agg_layers[curr_layers - 1]
        local, weight = attn_m(src_node_feat, src_node_t_embed, src_ngh_feat, src_ngh_t_embed, src_ngn_edge_feat, mask)
        return local

    # Placeholder for neighbor finder: expected to be overridden / passed in
    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors):
        """
        This function must be provided by the user/environment. In your pytorch code
        you used self.ngh_finder.get_temporal_neighbor(...). Keep the same behavior:
        return three numpy arrays shaped (B, num_neighbors): node ids, edge ids, timestamps.
        Here we raise to remind you to supply.
        """
        raise NotImplementedError("Provide get_temporal_neighbor that returns numpy arrays (B, N): nodes, eids, times")
