import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np
import math
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
import os

class AttentionPool(nn.Module):
    """
    어텐션 메커니즘을 사용하여 시퀀스 출력을 단일 벡터로 풀링합니다.
    
    Args:
        d_model (int): 인코더의 피처 차원 (feature dimension).
        n_heads (int): 멀티헤드 어텐션의 헤드 수.
    """
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.d_model = d_model
        # 학습 가능한 단일 쿼리 벡터를 생성합니다. 
        # 이 쿼리가 전체 시퀀스를 요약하는 방법을 학습합니다.
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 표준 MultiheadAttention 레이어를 사용합니다.
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 인코더의 전체 출력. 
                              shape: [Batch, Seq_Len, d_model]
        
        Returns:
            torch.Tensor: 풀링된 단일 컨텍스트 벡터.
                          shape: [Batch, d_model]
        """
        # MultiheadAttention은 (Seq_Len, Batch, d_model) 입력을 기대하므로, 차원을 바꿔줍니다.
        x = x.permute(1, 0, 2)  # [Seq_Len, Batch, d_model]
        
        B = x.size(1)
        query = self.query.expand(-1, B, -1) # [1, Batch, d_model]
        
        attn_output, _ = self.attention(
            query=query, 
            key=x, 
            value=x
        ) 
        
        return attn_output.squeeze(0) # [Batch, d_model]

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class SparseVariationalAttention(nn.Module):
    def __init__(self, feature_dim, n_heads=4, factor=5):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.head_dim = feature_dim // n_heads
        self.factor = factor
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

    def _prob_QK(self, Q, K, L_K):
        # Q [B, H, L, D]
        # K [B, H, L, D]
        B, H, L_Q, _ = Q.shape

        # Determine number of keys to sample: c*ln(L_K)
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        U_part = U_part if U_part < L_K else L_K

        # Randomly sample U_part keys for each query
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, self.head_dim)
        index_sample = torch.randint(L_K, (L_Q, U_part))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # Sparsity Measurement: max-avg
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), U_part)

        # Determine number of top queries: c*ln(L_Q)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()
        u = u if u < L_Q else L_Q
        M_top_index = M.topk(u, sorted=False)[1]

        # Use the selected top-u queries to calculate QK with all keys
        Q_reduce = Q[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None], M_top_index, :]

        # Calculate scores for selected queries only
        Q_K_scores = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K_scores, M_top_index, u

    def forward(self, x):
        # Project Q, K, V
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 1. Sparse Attention Calculation
        # Get scores and indices of top-u queries
        sparse_scores, top_query_indices, n_top = self._prob_QK(q, k, T)

        # 2. Variational Inference on Sparse Scores
        # Predict mu and sigma for the sparse scores
        scores_mu = sparse_scores / np.sqrt(self.head_dim)

        # Using a simple placeholder for log_sigma for demonstration
        scores_log_sigma = torch.zeros_like(scores_mu)
        scores_sigma = F.softplus(scores_log_sigma)

        # KL Divergence Loss
        kl_loss = 0.5 * torch.mean(scores_sigma.pow(2) + scores_mu.pow(2) - 1 - 2 * torch.log(scores_sigma + 1e-9))

        # 3. Sample scores and compute attention weights
        sampled_scores = scores_mu + torch.randn_like(scores_sigma) * scores_sigma
        attn_weights = F.softmax(sampled_scores, dim=-1)

        # 4. Compute and fill context
        # Initial context (e.g., mean of all values)
        context = v.mean(dim=-2).unsqueeze(-2).expand(B, self.n_heads, T, self.head_dim).clone()

        # Weighted sum of V for selected queries
        sparse_context = torch.matmul(attn_weights, v)

        # Fill the context for the selected queries
        context[torch.arange(B)[:, None, None], torch.arange(self.n_heads)[None, :, None], top_query_indices, :] = sparse_context.type_as(context)
        final_context = context.transpose(1, 2).reshape(B, T, C)

        return self.out_proj(final_context), kl_loss