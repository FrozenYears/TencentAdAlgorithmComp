"""
ESMM + DIN 模型

ESMM: CTCVR = pCTR * pCVR, 共享embedding层
DIN: Target Attention (candidate=Query, behavior_seq=Key/Value)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINTargetAttention(nn.Module):
    """
    DIN Target Attention

    Query: 候选物品embedding
    Key/Value: 用户行为序列embedding
    Attention: MLP(concat(q, k, q*k)) -> score -> weighted sum
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.attention_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query, keys, mask=None):
        """
        Args:
            query: [batch, embedding_dim] 候选物品
            keys: [batch, seq_len, embedding_dim] 行为序列
            mask: [batch, seq_len] 1=有效, 0=padding
        Returns:
            output: [batch, embedding_dim] attention加权后的用户表征
        """
        batch_size, seq_len, emb_dim = keys.size()

        # query扩展到seq_len维度
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)  # [B, S, D]

        # 构建attention输入: concat(q, k, q-k, q*k)
        attention_input = torch.cat([
            query_expanded,
            keys,
            query_expanded - keys,
            query_expanded * keys,
        ], dim=-1)  # [B, S, 4D]

        scores = self.attention_mlp(attention_input).squeeze(-1)  # [B, S]

        # mask: padding位置设为-inf
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # softmax归一化
        attn_weights = F.softmax(scores, dim=-1)  # [B, S]

        # 处理全mask行(全-inf导致nan)
        attn_weights = attn_weights.masked_fill(torch.isnan(attn_weights), 0.0)

        # 加权求和
        output = torch.bmm(attn_weights.unsqueeze(1), keys).squeeze(1)  # [B, D]

        return output


class ESMM_DIN(nn.Module):
    """
    ESMM多任务模型 + DIN Target Attention

    架构:
    - 共享Embedding层 (所有特征共享同一套embedding)
    - DIN Attention (4个domain的行为序列)
    - CTR Tower (MLP)
    - CVR Tower (MLP)
    - CTCVR = pCTR * pCVR
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
        embedding_dim: int = 16,
        seq_hash_bucket: int = 10001,
        hidden_dims: list = None,
        dropout_rate: float = 0.4,
        l2_reg: float = 1e-5,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.l2_reg = l2_reg
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        # === 共享Embedding层 ===
        # 用户标量特征embedding
        self.user_scalar_embs = nn.ModuleList([
            nn.Embedding(dim, embedding_dim, padding_idx=0)
            for dim in user_scalar_dims
        ])

        # 物品标量特征embedding
        self.item_scalar_embs = nn.ModuleList([
            nn.Embedding(dim, embedding_dim, padding_idx=0)
            for dim in item_scalar_dims
        ])

        # 用户列表特征embedding
        self.user_list_embs = nn.ModuleList([
            nn.Embedding(dim, embedding_dim, padding_idx=0)
            for dim in user_list_dims
        ])

        # 物品列表特征embedding
        self.item_list_embs = nn.ModuleList([
            nn.Embedding(dim, embedding_dim, padding_idx=0)
            for dim in item_list_dims
        ])

        # 序列embedding (4个domain共享同一个hash embedding)
        self.seq_embedding = nn.Embedding(seq_hash_bucket, embedding_dim, padding_idx=0)

        # 用户侧特征投影 (scalar_emb*N + list_emb*N + dense_dim -> hidden)
        user_input_dim = (
            embedding_dim * len(user_scalar_dims)
            + embedding_dim * len(user_list_dims)
            + user_dense_dim
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

        # DIN Attention (4个domain各一个)
        # item_query_proj: 将raw item embedding投影到embedding_dim空间用于attention
        self.item_query_proj = nn.Linear(item_input_dim, embedding_dim)
        self.attention_a = DINTargetAttention(embedding_dim, hidden_dim=64)
        self.attention_b = DINTargetAttention(embedding_dim, hidden_dim=64)
        self.attention_c = DINTargetAttention(embedding_dim, hidden_dim=64)
        self.attention_d = DINTargetAttention(embedding_dim, hidden_dim=64)

        # 序列聚合投影 (4个domain的attention输出拼接 -> hidden)
        self.seq_proj = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # CTR Tower
        tower_input_dim = hidden_dims[0] * 3  # user + item + seq
        self.ctr_tower = self._build_mlp(tower_input_dim, hidden_dims, dropout_rate)

        # CVR Tower (共享embedding, 独立MLP)
        self.cvr_tower = self._build_mlp(tower_input_dim, hidden_dims, dropout_rate)

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

    def _embed_user(self, user_scalar, user_list, user_dense):
        """用户特征 -> embedding向量"""
        parts = []
        for i, emb_layer in enumerate(self.user_scalar_embs):
            parts.append(emb_layer(user_scalar[:, i]))
        for i, emb_layer in enumerate(self.user_list_embs):
            list_emb = emb_layer(user_list[:, i])  # [B, max_list_len, D]
            # mean pooling (忽略padding=0)
            mask = (user_list[:, i] > 0).float().unsqueeze(-1)  # [B, max_list_len, 1]
            summed = (list_emb * mask).sum(dim=1)  # [B, D]
            count = mask.sum(dim=1).clamp(min=1)  # [B, 1]
            parts.append(summed / count)
        parts.append(user_dense)

        combined = torch.cat(parts, dim=-1)
        return self.user_proj(combined)

    def _embed_item(self, item_scalar, item_list):
        """物品特征 -> (raw_embedding, projected_embedding)"""
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

    def _process_sequence(self, seq_ids, query, mask, attention_layer):
        """单个domain序列的DIN attention"""
        seq_emb = self.seq_embedding(seq_ids)  # [B, S, D]
        return attention_layer(query, seq_emb, mask)  # [B, D]

    def get_l2_reg_loss(self):
        """计算所有embedding层的L2正则化损失"""
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
        """
        Args:
            batch: dict with keys:
                user_scalar, user_list, user_dense,
                item_scalar, item_list,
                seq_a, seq_b, seq_c, seq_d, seq_mask

        Returns:
            p_ctr: [B] CTR预测概率
            p_cvr: [B] CVR预测概率 (仅对有意义的样本)
            p_ctcvr: [B] CTCVR = pCTR * pCVR
        """
        user_repr = self._embed_user(
            batch['user_scalar'], batch['user_list'], batch['user_dense']
        )
        item_raw, item_repr = self._embed_item(batch['item_scalar'], batch['item_list'])

        # DIN attention: 将raw item embedding投影到embedding_dim作为query
        item_query = self.item_query_proj(item_raw)
        attn_a = self._process_sequence(batch['seq_a'], item_query, batch['seq_mask'], self.attention_a)
        attn_b = self._process_sequence(batch['seq_b'], item_query, batch['seq_mask'], self.attention_b)
        attn_c = self._process_sequence(batch['seq_c'], item_query, batch['seq_mask'], self.attention_c)
        attn_d = self._process_sequence(batch['seq_d'], item_query, batch['seq_mask'], self.attention_d)

        seq_repr = self.seq_proj(torch.cat([attn_a, attn_b, attn_c, attn_d], dim=-1))

        # 拼接用户、物品、序列表征
        combined = torch.cat([user_repr, item_repr, seq_repr], dim=-1)

        # CTR Tower
        p_ctr = torch.sigmoid(self.ctr_tower(combined).squeeze(-1))

        # CVR Tower
        p_cvr = torch.sigmoid(self.cvr_tower(combined).squeeze(-1))

        # ESMM: CTCVR = pCTR * pCVR
        p_ctcvr = p_ctr * p_cvr

        return p_ctr, p_cvr, p_ctcvr
