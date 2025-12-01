import jax
import logging
import jax.numpy as jnp
import flax.nnx as nnx


class MergeLayer(nnx.Module):
    def __init__(self, dim1, dim2, dim3, dim4, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(dim1 + dim2, dim3, rngs=rngs)
        self.fc2 = nnx.Linear(dim3, dim4, rngs=rngs)
        # Todo: xavier normal init

    def __call__(self, x1, x2):
        x = jnp.concatenate([x1, x2], axis=1)
        h = jax.nn.relu(self.fc1(x))
        return self.fc2(h)


class ScaledDotProductAttention(nnx.Module):
    def __init__(self, temperature, rngs: nnx.Rngs, attn_dropout=0.1):
        self.temperature = temperature
        self.dropout = nnx.Dropout(attn_dropout, rngs=rngs)

    def __call__(self, q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, mask=None):
        # q: (batch, lq, dk), k: (batch, lk, dk)
        attn = jnp.matmul(q, jnp.transpose(k, (0, 2, 1)))
        attn = attn / self.temperature

        if mask is not None:
            # mask True indicates positions to mask -> set very negative value
            attn = jnp.where(mask, -1e10, attn)

        attn = jax.nn.softmax(attn, axis=2)
        attn = self.dropout(attn)

        output = jnp.matmul(attn, v)

        return output, attn


class MultiHeadedAttention(nnx.Module):
    """Multi-Head Attention module"""

    def __init__(
        self,
        n_head: int,
        d_model: int,
        d_k: int,
        d_v,
        rngs: nnx.Rngs,
        dropout: float = 0.1,
    ):
        """
        :param n_head: number of attention heads
        :param d_model: dimension of model
        :param d_k: dimension of keys
        :param d_v: dimension of values
        :param dropout: dropout rate
        """
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nnx.Linear(d_model, n_head * d_k, rngs=rngs)
        self.w_ks = nnx.Linear(d_model, n_head * d_k, rngs=rngs)
        self.w_vs = nnx.Linear(d_model, n_head * d_v, rngs=rngs)

        # ToDo:
        # nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        # nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        # nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(
            temperature=jnp.power(d_k, 0.5), rngs=rngs, attn_dropout=dropout
        )
        self.layer_norm = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.fc = nnx.Linear(n_head * d_v, d_model, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, mask=None):
        '''
        :param q: queries shape (batch_size, len_q, d_model)
        :param k: keys shape (batch_size, len_k, d_model)
        :param v: values shape (batch_size, len_v, d_model)
        :param mask: mask shape (batch_size, len_q, len_k)
        :return: output shape (batch_size, len_q, d_model)
        '''
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        # https://github.com/pytorch/pytorch/issues/5544
        sz_b, len_q, _ = q.shape
        sz_b, len_k, _ = k.shape
        sz_b, len_v, _ = v.shape

        residual = q

        q = self.w_qs(q).reshape(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).reshape(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).reshape(sz_b, len_v, n_head, d_v)

        q = jnp.transpose(q, (2, 0, 1, 3)).reshape(-1, len_q, d_k)
        k = jnp.transpose(k, (2, 0, 1, 3)).reshape(-1, len_k, d_k)
        v = jnp.transpose(v, (2, 0, 1, 3)).reshape(-1, len_v, d_v)

        # Repeat mask along the head/batch dimension: (batch, lq, lk) -> (n_head * batch, lq, lk)
        if mask is not None:
            mask = jnp.repeat(mask, n_head, axis=0)

        output, attn = self.attention(q, k, v, mask=mask)
        output = output.reshape(n_head, sz_b, len_q, d_v)

        output = jnp.permute_dims(output, (1, 2, 0, 3)).reshape(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn
    
class MapBasedMultiHeadAttention(nnx.Module):
    def __init__(self, n_head, d_model, d_k, d_v, rngs: nnx.Rngs, dropout=0.1):

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.wq_node_transform = nnx.Linear(d_model, n_head * d_k, rngs=rngs)
        self.wk_node_transform = nnx.Linear(d_model, n_head * d_k, rngs=rngs)
        # values projection must produce n_head * d_v dims
        self.wv_node_transform = nnx.Linear(d_model, n_head * d_v, rngs=rngs)
        
        self.layer_norm = nnx.LayerNorm(d_model, rngs=rngs)

        self.fc = nnx.Linear(n_head * d_v, d_model, rngs=rngs)
        
        # self.act = nnx.leaky_relu(negative_slope=0.2)
        self.weight_map = nnx.Linear(2 * d_k, 1, rngs=rngs)
        
        # nn.init.xavier_normal_(self.fc.weight)
        
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        # self.softmax = torch.nn.Softmax(dim=2)


    def __call__(self, q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, mask=None):
        ''''
        param q: (batch_size, len_q, d_model)
        param k: (batch_size, len_k, d_model)
        param v: (batch_size, len_v, d_model)
        mask: (batch_size, len_q, len_k)
        '''
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.shape
        sz_b, len_k, _ = k.shape
        sz_b, len_v, _ = v.shape

        residual = q

        q = self.wq_node_transform(q).reshape(sz_b, len_q, n_head, d_k)
        k = self.wk_node_transform(k).reshape(sz_b, len_k, n_head, d_k)
        v = self.wv_node_transform(v).reshape(sz_b, len_v, n_head, d_v)

        q = jnp.transpose(q, (2, 0, 1, 3)).reshape(-1, len_q, d_k)  # (n*b) x lq x dk
        # expand and broadcast q to (n*b, lq, lk, dk)
        q = jnp.expand_dims(q, axis=2)  # (n*b, lq, 1, dk)
        q = jnp.broadcast_to(q, (q.shape[0], q.shape[1], len_k, q.shape[3]))

        k = jnp.transpose(k, (2, 0, 1, 3)).reshape(-1, len_k, d_k)  # (n*b) x lk x dk
        k = jnp.expand_dims(k, axis=1)  # (n*b, 1, lk, dk)
        k = jnp.broadcast_to(k, (k.shape[0], len_q, k.shape[2], k.shape[3]))  # (n*b, lq, lk, dk)

        v = jnp.transpose(v, (2, 0, 1, 3)).reshape(-1, len_v, d_v)  # (n*b) x lv x dv

        if mask is not None:
            mask = jnp.repeat(mask, n_head, axis=0)  # (n*b) x lq x lk

        q_k = jnp.concatenate([q, k], axis=-1)  # (n*b, lq, lk, dk*2)
        attn = self.weight_map(q_k).squeeze(-1)  # (n*b, lq, lk)

        if mask is not None:
            attn = jnp.where(mask, -1e10, attn)

        attn = jax.nn.softmax(attn, axis=2)  # (n*b, lq, lk)
        attn = self.dropout(attn)

        # (n*b, lq, lk) @ (n*b, lv, dv) -> (n*b, lq, dv)
        output = jnp.matmul(attn, v)
        output = output.reshape(n_head, sz_b, len_q, d_v)

        output = jnp.transpose(output, (1, 2, 0, 3)).reshape(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, attn
    

class TimeEncode(nnx.Module):
    def __init__(self, expand_dim, factor=5):
        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = nnx.Param((1 / 10 ** jnp.linspace(0, 9, time_dim)).astype(jnp.float32))
        self.phase = nnx.Param(jnp.zeros(time_dim).astype(jnp.float32))

    def __call__(self, ts):
        batch_size = ts.shape[0]
        seq_len = ts.shape[1]

        ts = ts.reshape(batch_size, seq_len, 1)
        map_ts = ts * self.basis_freq.reshape(1, 1, -1)

        harmonic = jnp.cos(map_ts)
        return harmonic

class PosEncode(nnx.Module):
    def __init__(self, expand_dim, seq_len, rngs: nnx.Rngs):
        super().__init__()
        
        self.pos_embeddings = nnx.Embed(num_embeddings=seq_len, features=expand_dim, rngs=rngs)
                
    def __call__(self, ts):
        # ts: [N, L]
        order = jnp.argsort(ts)
        ts_emb = self.pos_embeddings(order)
        return ts_emb
    

class EmptyEncode(nnx.Module):
    def __init__(self, expand_dim):
        super().__init__()
        self.expand_dim = expand_dim
        
    def __call__(self, ts):
        out = jnp.zeros_like(ts).astype(jnp.float32)
        out = jnp.expand_dims(out, axis=-1)
        out = jnp.broadcast_to(out, (out.shape[0], out.shape[1], self.expand_dim))
        return out


class LSTMPool(nnx.Module):
    def __init__(self, feat_dim, edge_dim, time_dim, rngs: nnx.Rngs):
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.edge_dim = edge_dim
        
        self.att_dim = feat_dim + edge_dim + time_dim
        
        self.lstm = nnx.LSTMCell(
            in_features=self.att_dim,
            hidden_features=self.feat_dim,
            rngs = rngs
        )

        self.merger = MergeLayer(feat_dim, feat_dim, feat_dim, feat_dim, rngs=rngs)

    def __call__(self, src, src_t, seq, seq_t, seq_e, mask):
        # seq [B, N, D]
        # mask [B, N]
        seq_x = jnp.concatenate([seq, seq_e, seq_t], axis=2)

        # nnx.LSTMCell usage varies; here we try to call it in a simple scan-like way
        # Initialize hidden / cell states as zeros
        batch = seq_x.shape[0]
        h = jnp.zeros((batch, self.feat_dim), dtype=jnp.float32)
        c = jnp.zeros((batch, self.feat_dim), dtype=jnp.float32)

        # simple recurrent loop over time dimension
        for t in range(seq_x.shape[1]):
            h, c = self.lstm(seq_x[:, t, :], (h, c))

        hn = h

        out = self.merger(hn, src)
        return out, None
    

class MeanPool(nnx.Module):
    def __init__(self, feat_dim, edge_dim, rngs: nnx.Rngs):
        super(MeanPool, self).__init__()
        self.edge_dim = edge_dim
        self.feat_dim = feat_dim
        self.merger = MergeLayer(edge_dim + feat_dim, feat_dim, feat_dim, feat_dim, rngs=rngs)
        
    def __call__(self, src, src_t, seq, seq_t, seq_e, mask):
        # seq [B, N, D]
        # mask [B, N]
        src_x = src
        seq_x = jnp.concatenate([seq, seq_e], axis=2) #[B, N, De + D]
        hn = seq_x.mean(axis=1) #[B, De + D]
        output = self.merger(hn, src_x)
        return output, None


class AttnModel(nnx.Module):
    """Attention based temporal layers
    """
    def __init__(self, feat_dim, edge_dim, time_dim, rngs: nnx.Rngs,
                 attn_mode='prod', n_head=2, drop_out=0.1):
        """
        args:
          feat_dim: dim for the node features
          edge_dim: dim for the temporal edge features
          time_dim: dim for the time encoding
          attn_mode: choose from 'prod' and 'map'
          n_head: number of heads in attention
          drop_out: probability of dropping a neural.
        """
        super(AttnModel, self).__init__()
        
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        
        self.edge_in_dim = (feat_dim + edge_dim + time_dim)
        self.model_dim = self.edge_in_dim
        #self.edge_fc = torch.nn.Linear(self.edge_in_dim, self.feat_dim, bias=False)

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim, rngs=rngs)

        #self.act = torch.nn.ReLU()
        
        # assert(self.model_dim % n_head == 0)
        self.logger = logging.getLogger(__name__)
        self.attn_mode = attn_mode
        
        if attn_mode == 'prod':
            self.multi_head_target = MultiHeadedAttention(n_head, 
                                             d_model=self.model_dim, 
                                             d_k=self.model_dim // n_head, 
                                             d_v=self.model_dim // n_head, 
                                             dropout=drop_out, rngs=rngs)
            self.logger.info('Using scaled prod attention')
            
        elif attn_mode == 'map':
            self.multi_head_target = MapBasedMultiHeadAttention(n_head, 
                                             d_model=self.model_dim, 
                                             d_k=self.model_dim // n_head, 
                                             d_v=self.model_dim // n_head, 
                                             dropout=drop_out, rngs=rngs)
            self.logger.info('Using map based attention')
        else:
            raise ValueError('attn_mode can only be prod or map')
        
        
    def __call__(self, src, src_t, seq, seq_t, seq_e, mask):
        """"Attention based temporal attention forward pass
        args:
          src: float Tensor of shape [B, D]
          src_t: float Tensor of shape [B, Dt], Dt == D
          seq: float Tensor of shape [B, N, D]
          seq_t: float Tensor of shape [B, N, Dt]
          seq_e: float Tensor of shape [B, N, De], De == D
          mask: boolean Tensor of shape [B, N], where the true value indicate a null value in the sequence.

        returns:
          output, weight

          output: float Tensor of shape [B, D]
          weight: float Tensor of shape [B, N]
        """

        src_ext = jnp.expand_dims(src, axis=1) # src [B, 1, D]
        src_e_ph = jnp.zeros_like(src_ext)
        q = jnp.concatenate([src_ext, src_e_ph, src_t], axis=2) # [B, 1, D + De + Dt]
        k = jnp.concatenate([seq, seq_e, seq_t], axis=2)

        mask = jnp.expand_dims(mask, axis=2) # mask [B, N, 1]
        mask = jnp.transpose(mask, (0, 2, 1)) #mask [B, 1, N]

        # target-attention
        output, attn = self.multi_head_target(q, k, k, mask=mask) # output: [B, 1, D], attn: possibly (n*b, 1, N)
        output = jnp.squeeze(output, axis=1)
        attn = jnp.squeeze(attn)

        output = self.merger(output, src)
        return output, attn


class TGAT(nnx.Module):
    def __init__(self, ngh_finder, n_feat, e_feat, rngs: nnx.Rngs,
                 attn_mode='prod', use_time='time', agg_method='attn', node_dim=None, time_dim=None,
                 num_layers=3, n_head=4, null_idx=0, num_heads=1, drop_out=0.1, seq_len=None):
        
        self.rngs = rngs
        self.num_layers = num_layers 
        self.ngh_finder = ngh_finder
        self.null_idx = null_idx
        self.logger = logging.getLogger(__name__)
        self.n_feat_th = nnx.Param(n_feat.astype(jnp.float32))
        self.e_feat_th = nnx.Param(e_feat.astype(jnp.float32))
        
        self.edge_raw_embed = nnx.Embed(
            num_embeddings=self.e_feat_th.shape[0],
            features=self.e_feat_th.shape[1],
            rngs=self.rngs
        )
        self.edge_raw_embed.embedding.copy_from(self.e_feat_th)

        self.node_raw_embed = nnx.Embed(
            num_embeddings=self.n_feat_th.shape[0],
            features=self.n_feat_th.shape[1],
            rngs=self.rngs
        )
        self.node_raw_embed.embedding.copy_from(self.n_feat_th)
        
        self.feat_dim = self.n_feat_th.shape[1]
        
        self.n_feat_dim = self.feat_dim
        self.e_feat_dim = self.feat_dim
        self.model_dim = self.feat_dim
        
        self.use_time = use_time
        self.merge_layer = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, self.feat_dim, rngs=rngs)
        
        if agg_method == 'attn':
            self.logger.info('Aggregation uses attention model')
            self.attn_model_list = nnx.List([AttnModel(self.feat_dim, 
                                                               self.feat_dim, 
                                                               self.feat_dim,
                                                               attn_mode=attn_mode, 
                                                               n_head=n_head, 
                                                               drop_out=drop_out, rngs=rngs) for _ in range(num_layers)])
        elif agg_method == 'lstm':
            self.logger.info('Aggregation uses LSTM model')
            self.attn_model_list = nnx.List([LSTMPool(self.feat_dim,
                                                                 self.feat_dim,
                                                                 self.feat_dim) for _ in range(num_layers)])
        elif agg_method == 'mean':
            self.logger.info('Aggregation uses constant mean model')
            self.attn_model_list = nnx.List([MeanPool(self.feat_dim,
                                                                 self.feat_dim, rngs=rngs) for _ in range(num_layers)])
        else:
        
            raise ValueError('invalid agg_method value, use attn or lstm')
        
        
        if use_time == 'time':
            self.logger.info('Using time encoding')
            self.time_encoder = TimeEncode(expand_dim=self.n_feat_th.shape[1])
        elif use_time == 'pos':
            assert(seq_len is not None)
            self.logger.info('Using positional encoding')
            self.time_encoder = PosEncode(expand_dim=self.n_feat_th.shape[1], seq_len=seq_len)
        elif use_time == 'empty':
            self.logger.info('Using empty encoding')
            self.time_encoder = EmptyEncode(expand_dim=self.n_feat_th.shape[1])
        else:
            raise ValueError('invalid time option!')
        
        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1, rngs=rngs) #torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)
        
    def __call__(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):
        
        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        
 
        score = self.affinity_score(src_embed, target_embed).squeeze(-1)
        
        return score

    def contrast(self, src_idx_l, target_idx_l, background_idx_l, cut_time_l, num_neighbors=20):
        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        background_embed = self.tem_conv(background_idx_l, cut_time_l, self.num_layers, num_neighbors)
        pos_score = self.affinity_score(src_embed, target_embed).squeeze(-1)
        neg_score = self.affinity_score(src_embed, background_embed).squeeze(-1)
        return jax.nn.sigmoid(pos_score), jax.nn.sigmoid(neg_score)
    

    def tem_conv(self, src_idx_l, cut_time_l, curr_layers, num_neighbors=20):
        assert(curr_layers >= 0)
    
        batch_size = len(src_idx_l)
        
        src_node_batch_th = src_idx_l
        cut_time_l_th = cut_time_l
        
        cut_time_l_th = jnp.expand_dims(cut_time_l_th, axis=1)
        # query node always has the start time -> time span == 0
        src_node_t_embed = self.time_encoder(jnp.zeros_like(cut_time_l_th))
        src_node_feat = self.node_raw_embed(src_node_batch_th)
        
        if curr_layers == 0:
            return src_node_feat
        else:
            src_node_conv_feat = self.tem_conv(src_idx_l, 
                                           cut_time_l,
                                           curr_layers=curr_layers - 1, 
                                           num_neighbors=num_neighbors)
            
            
            src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.ngh_finder.get_temporal_neighbor( 
                                                                    src_idx_l, 
                                                                    cut_time_l, 
                                                                    num_neighbors=num_neighbors)

            src_ngh_node_batch_th = src_ngh_node_batch
            src_ngh_eidx_batch = src_ngh_eidx_batch
            
            src_ngh_t_batch_delta = cut_time_l[:, jnp.newaxis] - src_ngh_t_batch
            src_ngh_t_batch_th = src_ngh_t_batch_delta
            
            # get previous layer's node features
            src_ngh_node_batch_flat = src_ngh_node_batch.flatten() #reshape(batch_size, -1)
            src_ngh_t_batch_flat = src_ngh_t_batch.flatten() #reshape(batch_size, -1)  
            src_ngh_node_conv_feat = self.tem_conv(src_ngh_node_batch_flat, 
                                                   src_ngh_t_batch_flat,
                                                   curr_layers=curr_layers - 1, 
                                                   num_neighbors=num_neighbors)
            src_ngh_feat = src_ngh_node_conv_feat.reshape(batch_size, num_neighbors, -1)
            
            # get edge time features and node features
            src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)
            src_ngn_edge_feat = self.edge_raw_embed(src_ngh_eidx_batch)

            # attention aggregation
            mask = src_ngh_node_batch_th == 0
            attn_m = self.attn_model_list[curr_layers - 1]
                        
            local, weight = attn_m(src_node_conv_feat, 
                                   src_node_t_embed,
                                   src_ngh_feat,
                                   src_ngh_t_embed, 
                                   src_ngn_edge_feat, 
                                   mask)
            return local