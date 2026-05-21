"""
ESMM+DIN+DCN model adapted for official Tencent Angel Platform framework.

Key design decisions:
- Single-tower CVR prediction (no ESMM multi-task) since label is binary (label_type==2)
- DIN Target Attention on 4 sequence domains
- DCN Cross Network for explicit feature interaction
- Compatible with ModelInput NamedTuple from official framework
- Implements get_sparse_params()/get_dense_params() for dual optimizer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, NamedTuple


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict
    seq_lens: dict
    seq_time_buckets: dict


class CrossNetwork(nn.Module):
    def __init__(self, input_dim: int, num_layers: int = 3):
        super().__init__()
        self.cross_layers = nn.ModuleList([
            nn.Linear(input_dim, 1, bias=True) for _ in range(num_layers)
        ])

    def forward(self, x0):
        xl = x0
        for layer in self.cross_layers:
            xl = x0 * layer(xl) + xl
        return xl


class DINTargetAttention(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.attention_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query, keys, mask=None):
        B, S, D = keys.size()
        q = query.unsqueeze(1).expand(-1, S, -1)
        att_input = torch.cat([q, keys, q - keys, q * keys], dim=-1)
        scores = self.attention_mlp(att_input).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = weights.masked_fill(torch.isnan(weights), 0.0)
        return torch.bmm(weights.unsqueeze(1), keys).squeeze(1)


class ESMM_DIN_DCN(nn.Module):
    def __init__(
        self,
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: Dict[str, List[int]],
        user_ns_groups: List[List[int]],
        item_ns_groups: List[List[int]],
        d_model: int = 64,
        emb_dim: int = 32,
        num_queries: int = 1,
        num_hyformer_blocks: int = 2,
        num_heads: int = 4,
        seq_encoder_type: str = 'transformer',
        hidden_mult: int = 4,
        dropout_rate: float = 0.1,
        num_cross_layers: int = 3,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        action_num: int = 1,
        num_time_buckets: int = 65,
        rank_mixer_mode: str = 'full',
        use_rope: bool = False,
        rope_base: float = 10000.0,
        seq_top_k: int = 50,
        seq_causal: bool = False,
        ns_tokenizer_type: str = 'rankmixer',
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.emb_dim = emb_dim
        self.seq_domains = sorted(seq_vocab_sizes.keys())
        self.user_int_feature_specs = user_int_feature_specs
        self.item_int_feature_specs = item_int_feature_specs

        self.user_embs = nn.ModuleList()
        for vs, offset, length in user_int_feature_specs:
            self.user_embs.append(nn.Embedding(vs + 1, emb_dim, padding_idx=0))

        self.item_embs = nn.ModuleList()
        for vs, offset, length in item_int_feature_specs:
            self.item_embs.append(nn.Embedding(vs + 1, emb_dim, padding_idx=0))

        self.has_user_dense = user_dense_dim > 0
        self.has_item_dense = item_dense_dim > 0
        if self.has_user_dense:
            self.user_dense_proj = nn.Linear(user_dense_dim, d_model)
        if self.has_item_dense:
            self.item_dense_proj = nn.Linear(item_dense_dim, d_model)

        self._seq_embs = nn.ModuleDict()
        self._seq_proj = nn.ModuleDict()
        for domain in self.seq_domains:
            vocab_sizes = seq_vocab_sizes[domain]
            self._seq_embs[domain] = nn.ModuleList([
                nn.Embedding(vs + 1, emb_dim, padding_idx=0) for vs in vocab_sizes
            ])
            self._seq_proj[domain] = nn.Linear(emb_dim * len(vocab_sizes), emb_dim)

        user_int_dim = emb_dim * len(user_int_feature_specs)
        item_int_dim = emb_dim * len(item_int_feature_specs)
        self.user_proj = nn.Sequential(
            nn.Linear(user_int_dim + (d_model if self.has_user_dense else 0), d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.item_proj = nn.Sequential(
            nn.Linear(item_int_dim + (d_model if self.has_item_dense else 0), d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.item_query_proj = nn.Linear(d_model, emb_dim)

        self.attention_layers = nn.ModuleDict()
        for domain in self.seq_domains:
            self.attention_layers[domain] = DINTargetAttention(emb_dim, hidden_dim=64)

        self.seq_proj = nn.Sequential(
            nn.Linear(emb_dim * len(self.seq_domains), d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        tower_input = d_model * 3
        self.cross_network = CrossNetwork(tower_input, num_cross_layers)
        self.dnn = nn.Sequential(
            nn.Linear(tower_input, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(d_model // 2 + tower_input, 1)

        self._num_ns = sum(len(g) for g in user_ns_groups) + sum(len(g) for g in item_ns_groups)

    @property
    def num_ns(self) -> int:
        return self._num_ns

    def get_sparse_params(self) -> List[nn.Parameter]:
        params = []
        for emb in self.user_embs:
            params.extend(emb.parameters())
        for emb in self.item_embs:
            params.extend(emb.parameters())
        for domain in self.seq_domains:
            for emb in self._seq_embs[domain]:
                params.extend(emb.parameters())
        return params

    def get_dense_params(self) -> List[nn.Parameter]:
        sparse_ids = {id(p) for p in self.get_sparse_params()}
        return [p for p in self.parameters() if id(p) not in sparse_ids]

    def reinit_high_cardinality_params(self, threshold: int = 0) -> List[int]:
        reinit_ptrs = []
        for emb in self.user_embs:
            if emb.num_embeddings > threshold:
                nn.init.xavier_uniform_(emb.weight)
                if emb.padding_idx is not None:
                    with torch.no_grad():
                        emb.weight[emb.padding_idx].fill_(0)
                reinit_ptrs.append(emb.weight.data_ptr())
        for emb in self.item_embs:
            if emb.num_embeddings > threshold:
                nn.init.xavier_uniform_(emb.weight)
                if emb.padding_idx is not None:
                    with torch.no_grad():
                        emb.weight[emb.padding_idx].fill_(0)
                reinit_ptrs.append(emb.weight.data_ptr())
        return reinit_ptrs

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        user_feats = []
        for i, (vs, offset, length) in enumerate(self.user_int_feature_specs):
            feat_slice = inputs.user_int_feats[:, offset:offset+length]
            if length == 1:
                feat_slice = feat_slice.squeeze(1)
            emb = self.user_embs[i](feat_slice)
            if length > 1:
                emb = emb.mean(dim=1)
            user_feats.append(emb)

        if self.has_user_dense:
            user_dense_tok = F.silu(self.user_dense_proj(inputs.user_dense_feats))
            user_repr = self.user_proj(torch.cat(user_feats + [user_dense_tok], dim=-1))
        else:
            user_repr = self.user_proj(torch.cat(user_feats, dim=-1))

        item_feats = []
        for i, (vs, offset, length) in enumerate(self.item_int_feature_specs):
            feat_slice = inputs.item_int_feats[:, offset:offset+length]
            if length == 1:
                feat_slice = feat_slice.squeeze(1)
            emb = self.item_embs[i](feat_slice)
            if length > 1:
                emb = emb.mean(dim=1)
            item_feats.append(emb)

        if self.has_item_dense:
            item_dense_tok = F.silu(self.item_dense_proj(inputs.item_dense_feats))
            item_repr = self.item_proj(torch.cat(item_feats + [item_dense_tok], dim=-1))
        else:
            item_repr = self.item_proj(torch.cat(item_feats, dim=-1))

        item_query = self.item_query_proj(item_repr)

        seq_reprs = []
        for domain in self.seq_domains:
            seq_tensor = inputs.seq_data[domain]
            seq_ids = seq_tensor[:, 0, :]
            mask = (seq_ids > 0).float()

            parts = []
            for j, emb in enumerate(self._seq_embs[domain]):
                parts.append(emb(seq_tensor[:, j, :]))
            seq_emb_raw = torch.cat(parts, dim=-1)
            seq_emb = self._seq_proj[domain](seq_emb_raw)

            attn_out = self.attention_layers[domain](item_query, seq_emb, mask)
            seq_reprs.append(attn_out)

        seq_repr = self.seq_proj(torch.cat(seq_reprs, dim=-1))

        combined = torch.cat([user_repr, item_repr, seq_repr], dim=-1)
        cross_out = self.cross_network(combined)
        dnn_out = self.dnn(combined)
        logits = self.output_layer(torch.cat([dnn_out, cross_out], dim=-1))

        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(inputs)
        return logits, logits
