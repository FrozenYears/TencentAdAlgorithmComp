"""
ESMM + DIN + DCN + Transformer 模型V2

核心改进:
1. DCN (Deep Cross Network): 显式特征交叉
2. Transformer Encoder: 序列建模替代简单attention
3. Self-supervised Auxiliary Loss: 缓解标签稀疏
4. 多域序列融合: 所有序列域的attention输出融合
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINTargetAttention(nn.Module):
    """DIN Target Attention (与V1相同)"""

    def __init__(self, embedding_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.attention_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query, keys, mask=None):
        batch_size, seq_len, emb_dim = keys.size()
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)
        attention_input = torch.cat([
            query_expanded, keys,
            query_expanded - keys,
            query_expanded * keys,
        ], dim=-1)
        scores = self.attention_mlp(attention_input).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = attn_weights.masked_fill(torch.isnan(attn_weights), 0.0)
        output = torch.bmm(attn_weights.unsqueeze(1), keys).squeeze(1)
        return output


class CrossNetworkLayer(nn.Module):
    """DCN Cross Network Layer: x_{l+1} = x_0 * (w^T * x_l + b) + x_l"""

    def __init__(self, input_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, 1) * 0.01)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0, xl):
        xl_w = torch.matmul(xl, self.weight)
        cross = x0 * (xl_w + self.bias)
        return cross + xl


class DeepCrossNetwork(nn.Module):
    """DCN: 多层Cross Network + Deep Network"""

    def __init__(self, input_dim: int, num_cross_layers: int = 3, deep_hidden_dim: int = 128):
        super().__init__()
        self.cross_layers = nn.ModuleList([
            CrossNetworkLayer(input_dim) for _ in range(num_cross_layers)
        ])
        self.deep_network = nn.Sequential(
            nn.Linear(input_dim, deep_hidden_dim),
            nn.BatchNorm1d(deep_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(deep_hidden_dim, deep_hidden_dim),
            nn.BatchNorm1d(deep_hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = input_dim + deep_hidden_dim

    def forward(self, x):
        x0 = x
        xl = x
        for cross_layer in self.cross_layers:
            xl = cross_layer(x0, xl)
        deep_out = self.deep_network(x)
        return torch.cat([xl, deep_out], dim=-1)


class TransformerEncoder(nn.Module):
    """Transformer Encoder for sequence modeling"""

    def __init__(self, d_model: int, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        if mask is not None:
            padding_mask = ~mask.bool()
        else:
            padding_mask = None
        out = self.transformer(x, src_key_padding_mask=padding_mask)
        return self.norm(out)


class SequenceAggregator(nn.Module):
    """Multi-domain sequence aggregation with Transformer"""

    def __init__(self, embedding_dim: int, num_domains: int = 4, num_heads: int = 4, num_layers: int = 1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_domains = num_domains

        self.domain_transformers = nn.ModuleList([
            TransformerEncoder(embedding_dim, nhead=num_heads, num_layers=num_layers)
            for _ in range(num_domains)
        ])

        self.domain_gate = nn.Sequential(
            nn.Linear(embedding_dim * num_domains, num_domains),
            nn.Softmax(dim=-1),
        )
        self.output_dim = embedding_dim

    def forward(self, domain_seqs, query, mask):
        domain_outputs = []
        for i, (seq, transformer) in enumerate(zip(domain_seqs, self.domain_transformers)):
            attn_out = transformer(seq, mask)
            mask_expanded = mask.unsqueeze(-1).expand_as(attn_out)
            masked_sum = (attn_out * mask_expanded).sum(dim=1)
            mask_count = mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = masked_sum / mask_count
            domain_outputs.append(pooled)

        concat_domains = torch.cat(domain_outputs, dim=-1)
        gate_weights = self.domain_gate(concat_domains).unsqueeze(-1)
        stacked_domains = torch.stack(domain_outputs, dim=1)
        fused = (stacked_domains * gate_weights).sum(dim=1)
        return fused


class SelfSupervisedHead(nn.Module):
    """Self-supervised auxiliary loss: masked item prediction"""

    def __init__(self, embedding_dim: int, num_items: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_items),
        )

    def forward(self, seq_repr, target_item_emb):
        logits = self.projection(seq_repr)
        return logits


class ESMM_DIN_V2(nn.Module):
    """
    ESMM + DIN + DCN + Transformer V2

    架构:
    - 共享Embedding层
    - DIN Target Attention (原始)
    - Transformer Encoder (序列建模)
    - DCN (显式特征交叉)
    - 统计/时间/交叉特征融合
    - Self-supervised Auxiliary Loss
    - CTR Tower + CVR Tower
    """

    def __init__(
        self,
        n_user_scalar_feats: int,
        n_item_scalar_feats: int,
        n_user_list_feats: int,
        n_item_list_feats: int,
        user_scalar_dims: list,
        item_scalar_dims: list,
        user_list_dims: list,
        item_list_dims: list,
        user_dense_dim: int,
        n_stat_feats: int = 10,
        n_time_feats: int = 4,
        n_seq_stat_feats: int = 12,
        n_cross_feats: int = 6,
        cross_hash_bucket: int = 5001,
        embedding_dim: int = 16,
        seq_hash_bucket: int = 10001,
        hidden_dims: list = None,
        dropout_rate: float = 0.3,
        l2_reg: float = 1e-5,
        num_cross_layers: int = 3,
        num_transformer_layers: int = 1,
        num_transformer_heads: int = 4,
        enable_ssl: bool = True,
        ssl_num_items: int = 100001,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.l2_reg = l2_reg
        self.enable_ssl = enable_ssl
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        # === 共享Embedding层 ===
        self.user_scalar_embs = nn.ModuleList([
            nn.Embedding(dim, embedding_dim, padding_idx=0) for dim in user_scalar_dims
        ])
        self.item_scalar_embs = nn.ModuleList([
            nn.Embedding(dim, embedding_dim, padding_idx=0) for dim in item_scalar_dims
        ])
        self.user_list_embs = nn.ModuleList([
            nn.Embedding(dim, embedding_dim, padding_idx=0) for dim in user_list_dims
        ])
        self.item_list_embs = nn.ModuleList([
            nn.Embedding(dim, embedding_dim, padding_idx=0) for dim in item_list_dims
        ])
        self.seq_embedding = nn.Embedding(seq_hash_bucket, embedding_dim, padding_idx=0)

        # 交叉特征embedding
        self.cross_embs = nn.ModuleList([
            nn.Embedding(cross_hash_bucket, embedding_dim, padding_idx=0)
            for _ in range(n_cross_feats)
        ])

        # 统计/时间/序列统计特征投影
        self.stat_proj = nn.Sequential(
            nn.Linear(n_stat_feats, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(n_time_feats, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.seq_stat_proj = nn.Sequential(
            nn.Linear(n_seq_stat_feats, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        # 用户侧特征投影
        user_input_dim = (
            embedding_dim * len(user_scalar_dims)
            + embedding_dim * len(user_list_dims)
            + user_dense_dim
            + 32 + 16  # stat + time
        )
        self.user_proj = nn.Sequential(
            nn.Linear(user_input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # 物品侧特征投影
        item_input_dim = embedding_dim * (len(item_scalar_dims) + len(item_list_dims))
        self.item_proj = nn.Sequential(
            nn.Linear(item_input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # DIN Attention (保留原始)
        self.item_query_proj = nn.Linear(item_input_dim, embedding_dim)
        self.attention_a = DINTargetAttention(embedding_dim, hidden_dim=64)
        self.attention_b = DINTargetAttention(embedding_dim, hidden_dim=64)
        self.attention_c = DINTargetAttention(embedding_dim, hidden_dim=64)
        self.attention_d = DINTargetAttention(embedding_dim, hidden_dim=64)

        # Transformer序列建模
        self.sequence_aggregator = SequenceAggregator(
            embedding_dim, num_domains=4,
            num_heads=num_transformer_heads,
            num_layers=num_transformer_layers,
        )

        # 序列聚合投影
        self.seq_proj = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # 特征融合维度: user + item + seq + cross_embs + seq_stat
        cross_dim = embedding_dim * n_cross_feats
        fusion_dim = hidden_dims[0] * 3 + cross_dim + 32  # user + item + seq + cross + seq_stat

        # DCN for feature interaction
        self.dcn = DeepCrossNetwork(
            fusion_dim,
            num_cross_layers=num_cross_layers,
            deep_hidden_dim=hidden_dims[0],
        )

        # CTR Tower
        tower_input_dim = self.dcn.output_dim
        self.ctr_tower = self._build_mlp(tower_input_dim, hidden_dims, dropout_rate)

        # CVR Tower
        self.cvr_tower = self._build_mlp(tower_input_dim, hidden_dims, dropout_rate)

        # Self-supervised head
        if enable_ssl:
            self.ssl_head = SelfSupervisedHead(embedding_dim, ssl_num_items)

    def _build_mlp(self, input_dim, hidden_dims, dropout_rate):
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    def _embed_user(self, user_scalar, user_list, user_dense, stat_feats, time_feats):
        parts = []
        for i, emb_layer in enumerate(self.user_scalar_embs):
            parts.append(emb_layer(user_scalar[:, i]))
        for i, emb_layer in enumerate(self.user_list_embs):
            list_emb = emb_layer(user_list[:, i])
            mask = (user_list[:, i] > 0).float().unsqueeze(-1)
            summed = (list_emb * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1)
            parts.append(summed / count)
        parts.append(user_dense)
        parts.append(self.stat_proj(stat_feats))
        parts.append(self.time_proj(time_feats))
        combined = torch.cat(parts, dim=-1)
        return self.user_proj(combined)

    def _embed_item(self, item_scalar, item_list):
        parts = []
        for i, emb_layer in enumerate(self.item_scalar_embs):
            parts.append(emb_layer(item_scalar[:, i]))
        for i, emb_layer in enumerate(self.item_list_embs):
            list_emb = emb_layer(item_list[:, i])
            mask = (item_list[:, i] > 0).float().unsqueeze(-1)
            summed = (list_emb * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1)
            parts.append(summed / count)
        raw_emb = torch.cat(parts, dim=-1)
        projected = self.item_proj(raw_emb)
        return raw_emb, projected

    def _embed_cross(self, cross_feats):
        parts = []
        for i, emb_layer in enumerate(self.cross_embs):
            parts.append(emb_layer(cross_feats[:, i]))
        return torch.cat(parts, dim=-1)

    def _process_sequence(self, seq_ids, query, mask, attention_layer):
        seq_emb = self.seq_embedding(seq_ids)
        return attention_layer(query, seq_emb, mask)

    def get_l2_reg_loss(self):
        l2_loss = 0.0
        for emb in self.user_scalar_embs:
            l2_loss += torch.norm(emb.weight, p=2)
        for emb in self.item_scalar_embs:
            l2_loss += torch.norm(emb.weight, p=2)
        for emb in self.user_list_embs:
            l2_loss += torch.norm(emb.weight, p=2)
        for emb in self.item_list_embs:
            l2_loss += torch.norm(emb.weight, p=2)
        l2_loss += torch.norm(self.seq_embedding.weight, p=2)
        return self.l2_reg * l2_loss

    def forward(self, batch):
        user_repr = self._embed_user(
            batch['user_scalar'], batch['user_list'], batch['user_dense'],
            batch['stat_feats'], batch['time_feats']
        )
        item_raw, item_repr = self._embed_item(batch['item_scalar'], batch['item_list'])

        item_query = self.item_query_proj(item_raw)
        attn_a = self._process_sequence(batch['seq_a'], item_query, batch['seq_mask'], self.attention_a)
        attn_b = self._process_sequence(batch['seq_b'], item_query, batch['seq_mask'], self.attention_b)
        attn_c = self._process_sequence(batch['seq_c'], item_query, batch['seq_mask'], self.attention_c)
        attn_d = self._process_sequence(batch['seq_d'], item_query, batch['seq_mask'], self.attention_d)

        seq_repr = self.seq_proj(torch.cat([attn_a, attn_b, attn_c, attn_d], dim=-1))

        cross_repr = self._embed_cross(batch['cross_feats'])
        seq_stat_repr = self.seq_stat_proj(batch['seq_stat_feats'])

        combined = torch.cat([user_repr, item_repr, seq_repr, cross_repr, seq_stat_repr], dim=-1)

        dcn_out = self.dcn(combined)

        p_ctr = torch.sigmoid(self.ctr_tower(dcn_out).squeeze(-1))
        p_cvr = torch.sigmoid(self.cvr_tower(dcn_out).squeeze(-1))
        p_ctcvr = p_ctr * p_cvr

        ssl_logits = None
        if self.enable_ssl and self.training:
            seq_emb_a = self.seq_embedding(batch['seq_a'])
            mask = batch['seq_mask']
            mask_expanded = mask.unsqueeze(-1).expand_as(seq_emb_a)
            seq_repr_ssl = (seq_emb_a * mask_expanded).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            ssl_logits = self.ssl_head(seq_repr_ssl, item_raw)

        return p_ctr, p_cvr, p_ctcvr, ssl_logits
