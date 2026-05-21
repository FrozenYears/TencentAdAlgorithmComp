"""
DIN+DCN model for CVR prediction, adapted for Tencent Angel Platform.

Key features:
- DIN Target Attention on 4 sequence domains
- DCN Cross Network for explicit feature interaction
- emb_skip_threshold: skip ultra-high-cardinality features to save GPU memory
- Compatible with ModelInput NamedTuple from official framework
- Implements get_sparse_params()/get_dense_params() for dual optimizer
"""

import logging
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
    def __init__(self, input_dim: int, num_layers: int = 2):
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
    def __init__(self, embedding_dim: int, hidden_dim: int = 32):
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


def _build_emb_list(
    feature_specs: List[Tuple[int, int, int]],
    emb_dim: int,
    emb_skip_threshold: int,
) -> Tuple[nn.ModuleList, List[int]]:
    """Build embedding tables with optional high-cardinality skipping.

    Returns:
        module_list: Embedding tables (only for non-skipped features).
        emb_index: Maps feature position -> index in module_list, or -1 if skipped.
    """
    embs_raw = []
    emb_index = []
    skipped = 0
    for vs, offset, length in feature_specs:
        skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
        if skip:
            emb_index.append(-1)
            skipped += 1
        else:
            emb_index.append(len(embs_raw))
            embs_raw.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
    if skipped > 0:
        logging.info(f"emb_skip_threshold={emb_skip_threshold}: skipped {skipped}/{len(feature_specs)} features")
    return nn.ModuleList(embs_raw), emb_index


def _build_seq_emb_list(
    vocab_sizes: List[int],
    emb_dim: int,
    emb_skip_threshold: int,
) -> Tuple[nn.ModuleList, List[int]]:
    """Build sequence embedding tables with optional high-cardinality skipping.

    Returns:
        module_list: Embedding tables (only for non-skipped features).
        emb_index: Maps feature position -> index in module_list, or -1 if skipped.
    """
    embs_raw = []
    emb_index = []
    skipped = 0
    for vs in vocab_sizes:
        skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
        if skip:
            emb_index.append(-1)
            skipped += 1
        else:
            emb_index.append(len(embs_raw))
            embs_raw.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
    if skipped > 0:
        logging.info(f"seq emb_skip_threshold={emb_skip_threshold}: skipped {skipped}/{len(vocab_sizes)} features")
    return nn.ModuleList(embs_raw), emb_index


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
        d_model: int = 32,
        emb_dim: int = 16,
        num_cross_layers: int = 2,
        dropout_rate: float = 0.1,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        num_queries: int = 1,
        num_hyformer_blocks: int = 2,
        num_heads: int = 4,
        seq_encoder_type: str = 'transformer',
        hidden_mult: int = 4,
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
        self.emb_skip_threshold = emb_skip_threshold

        # ========== User/Item int feature embeddings (with skip) ==========
        self.user_embs, self.user_emb_index = _build_emb_list(
            user_int_feature_specs, emb_dim, emb_skip_threshold)
        self.item_embs, self.item_emb_index = _build_emb_list(
            item_int_feature_specs, emb_dim, emb_skip_threshold)

        # ========== Dense feature projections ==========
        self.has_user_dense = user_dense_dim > 0
        self.has_item_dense = item_dense_dim > 0
        if self.has_user_dense:
            self.user_dense_proj = nn.Linear(user_dense_dim, d_model)
        if self.has_item_dense:
            self.item_dense_proj = nn.Linear(item_dense_dim, d_model)

        # ========== Sequence embeddings (with skip) ==========
        self._seq_embs = nn.ModuleDict()
        self._seq_emb_index = {}
        self._seq_proj = nn.ModuleDict()
        self._seq_num_features = {}
        for domain in self.seq_domains:
            vocab_sizes = seq_vocab_sizes[domain]
            embs, idx = _build_seq_emb_list(vocab_sizes, emb_dim, emb_skip_threshold)
            self._seq_embs[domain] = embs
            self._seq_emb_index[domain] = idx
            self._seq_num_features[domain] = len(vocab_sizes)
            self._seq_proj[domain] = nn.Linear(emb_dim * len(vocab_sizes), emb_dim)

        # ========== Feature projection layers ==========
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

        # ========== DIN Attention ==========
        self.item_query_proj = nn.Linear(d_model, emb_dim)
        attn_hidden = max(emb_dim, 32)  # proportional to emb_dim
        self.attention_layers = nn.ModuleDict()
        for domain in self.seq_domains:
            self.attention_layers[domain] = DINTargetAttention(emb_dim, hidden_dim=attn_hidden)

        self.seq_proj = nn.Sequential(
            nn.Linear(emb_dim * len(self.seq_domains), d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # ========== DCN + DNN Tower ==========
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

        # Log model stats
        total_params = sum(p.numel() for p in self.parameters())
        sparse_params = sum(p.numel() for p in self.get_sparse_params())
        dense_params = sum(p.numel() for p in self.get_dense_params())
        logging.info(f"ESMM_DIN_DCN: {total_params:,} total params "
                     f"({sparse_params:,} sparse + {dense_params:,} dense)")
        logging.info(f"  d_model={d_model}, emb_dim={emb_dim}, "
                     f"num_cross_layers={num_cross_layers}, "
                     f"emb_skip_threshold={emb_skip_threshold}")

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

    def _embed_int_features(
        self,
        feature_specs: List[Tuple[int, int, int]],
        embs: nn.ModuleList,
        emb_index: List[int],
        feats: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Embed int features, using zero vectors for skipped features."""
        result = []
        for i, (vs, offset, length) in enumerate(feature_specs):
            feat_slice = feats[:, offset:offset + length]
            if length == 1:
                feat_slice = feat_slice.squeeze(1)

            real_idx = emb_index[i]
            if real_idx == -1:
                result.append(torch.zeros(
                    feat_slice.shape[0], self.emb_dim,
                    dtype=torch.float32, device=feat_slice.device))
            else:
                emb = embs[real_idx]
                e = emb(feat_slice)
                if length > 1:
                    e = e.mean(dim=1)
                result.append(e)
        return result

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        # ========== User features ==========
        user_feats = self._embed_int_features(
            self.user_int_feature_specs, self.user_embs, self.user_emb_index,
            inputs.user_int_feats)

        if self.has_user_dense:
            user_dense_tok = F.silu(self.user_dense_proj(inputs.user_dense_feats))
            user_repr = self.user_proj(torch.cat(user_feats + [user_dense_tok], dim=-1))
        else:
            user_repr = self.user_proj(torch.cat(user_feats, dim=-1))

        # ========== Item features ==========
        item_feats = self._embed_int_features(
            self.item_int_feature_specs, self.item_embs, self.item_emb_index,
            inputs.item_int_feats)

        if self.has_item_dense:
            item_dense_tok = F.silu(self.item_dense_proj(inputs.item_dense_feats))
            item_repr = self.item_proj(torch.cat(item_feats + [item_dense_tok], dim=-1))
        else:
            item_repr = self.item_proj(torch.cat(item_feats, dim=-1))

        # ========== DIN Attention on sequences ==========
        item_query = self.item_query_proj(item_repr)

        seq_reprs = []
        for domain in self.seq_domains:
            seq_tensor = inputs.seq_data[domain]
            seq_ids = seq_tensor[:, 0, :]
            mask = (seq_ids > 0).float()

            # Embed each sideinfo feature, skip if needed
            emb_list = []
            emb_index = self._seq_emb_index[domain]
            for j in range(self._seq_num_features[domain]):
                real_idx = emb_index[j] if j < len(emb_index) else -1
                if real_idx == -1:
                    B, L = seq_tensor[:, j, :].shape
                    emb_list.append(torch.zeros(
                        B, L, self.emb_dim,
                        dtype=torch.float32, device=seq_tensor.device))
                else:
                    emb_list.append(self._seq_embs[domain][real_idx](seq_tensor[:, j, :]))

            seq_emb_raw = torch.cat(emb_list, dim=-1)
            seq_emb = self._seq_proj[domain](seq_emb_raw)

            attn_out = self.attention_layers[domain](item_query, seq_emb, mask)
            seq_reprs.append(attn_out)

        seq_repr = self.seq_proj(torch.cat(seq_reprs, dim=-1))

        # ========== DCN + DNN ==========
        combined = torch.cat([user_repr, item_repr, seq_repr], dim=-1)
        cross_out = self.cross_network(combined)
        dnn_out = self.dnn(combined)
        logits = self.output_layer(torch.cat([dnn_out, cross_out], dim=-1))

        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(inputs)
        return logits, logits
