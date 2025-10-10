import torch
import torch.nn as nn
import torch.nn.functional as F
from .TSF_TSM_layers import PatchingEmbedding
from .SelfAttention_Family import AttentionPool
from timm.models.layers import DropPath
import math

# TSF_TSM_experts_blocks.py 파일에 추가

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextualCouplingTSA(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, kernel_size=3, attn_temperature=1.0, s_max=2.0):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be an even number."
        self.d_model = d_model
        self.d_channel = d_model // 2
        self.n_heads = n_heads
        assert self.d_channel % n_heads == 0, "d_channel must be divisible by n_heads."
        self.d_head = self.d_channel // n_heads

        # Q/K/V는 x1 기준 (컨텍스트 생성 전용)
        self.q_proj = nn.Linear(self.d_channel, self.d_channel)
        self.k_proj = nn.Linear(self.d_channel, self.d_channel)
        self.v_proj = nn.Linear(self.d_channel, self.d_channel)

        pad = kernel_size // 2
        self.saliency_conv = nn.Conv1d(
            in_channels=self.d_channel,
            out_channels=n_heads,
            kernel_size=kernel_size,
            padding=pad
        )
        # saliency 스케일 (학습 가능)
        self.saliency_alpha = nn.Parameter(torch.tensor(1.0))

        # 컨텍스트 → (log_scale, shift)
        self.s_proj = nn.Linear(self.d_channel, self.d_channel)
        self.t_proj = nn.Linear(self.d_channel, self.d_channel)

        # 최종 출력 투영
        self.out_proj = nn.Linear(d_model, d_model)

        # 정규화/드롭아웃/파라미터
        self.pre_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_temperature = attn_temperature
        self.s_max = s_max  # log_scale clamp 범위

    def forward(self, src, attn_mask=None):
        """
        src: [B, L, d_model]
        반환: output [B, L, d_model], attention_weights [B, n_heads, L, L]
        """
        B, L, _ = src.shape

        # Pre-Norm은 레이어 외부 residual과 잘 맞음
        src = self.pre_norm(src)

        # 채널 절반으로 split
        x1, x2 = src.chunk(2, dim=-1)  # [B, L, d_channel] x 2

        # Q/K/V 생성 (x1 기반)
        Q = self.q_proj(x1).view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # [B,h,L,dh]
        K = self.k_proj(x1).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x1).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # Saliency scores: [B, n_heads, L] → [B, n_heads, 1, L] (모든 Query에 공통 bias)
        saliency_scores = self.saliency_conv(x1.transpose(1, 2))  # [B, h, L]
        saliency_scores = saliency_scores.unsqueeze(2)            # [B, h, 1, L]

        # Attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        # temperature & saliency
        attn_scores = attn_scores / max(self.attn_temperature, 1e-6)
        attn_scores = attn_scores + self.saliency_alpha * saliency_scores

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 컨텍스트: V에 대한 가중합
        context = torch.matmul(attn_weights, V)  # [B, h, L, dh]
        context = context.transpose(1, 2).contiguous().view(B, L, self.d_channel)  # [B, L, d_channel]

        # Affine 파라미터
        # log_scale은 clamp로 안정화 → scale = exp(log_scale)
        log_scale = self.s_proj(context)
        log_scale = torch.clamp(log_scale, min=-self.s_max, max=self.s_max)
        shift = self.t_proj(context)

        scale = torch.exp(log_scale)  # [B, L, d_channel]

        # y2 = scale * x2 + shift
        y2 = scale * x2 + shift

        # concat & project
        out = torch.cat([x1, y2], dim=-1)  # [B, L, d_model]
        out = self.out_proj(out)
        out = self.dropout(out)

        return out, attn_weights

class MoE(nn.Module):
    """Mixture of Experts FFN 레이어."""
    def __init__(self, d_model, d_ff, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
        self.gating_network = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # x shape: [B, L, C]
        logits = self.gating_network(x) # [B, L, num_experts]
        gates = F.softmax(logits, dim=-1)
        
        # 각 토큰에 대해 top-k 전문가 선택
        top_k_gates, top_k_indices = gates.topk(self.top_k, dim=-1) # [B, L, top_k]
        
        # top-k 가중치 정규화
        top_k_gates = top_k_gates / top_k_gates.sum(dim=-1, keepdim=True)
        
        # 전문가 계산
        output = torch.zeros_like(x)
        flat_x = x.view(-1, x.size(-1))
        flat_indices = top_k_indices.view(-1, self.top_k)

        for i in range(self.num_experts):
            # i번째 전문가를 선택한 토큰들의 마스크 생성
            mask, _ = (flat_indices == i).max(dim=-1)
            if mask.any():
                expert_input = flat_x[mask]
                expert_output = self.experts[i](expert_input)
                
                # 가중합을 위한 가중치 추출
                gate_values = top_k_gates.view(-1, self.top_k)[mask]
                indices_values = top_k_indices.view(-1, self.top_k)[mask]
                expert_gate = gate_values[indices_values == i]
                
                # 가중치를 적용하여 결과 저장
                output.view(-1, x.size(-1))[mask] += expert_output * expert_gate.unsqueeze(-1)
        return output

class MoETSAEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_experts, dropout=0.1):
        super().__init__()
        self.self_attn = ContextualCouplingTSA(d_model, n_heads, dropout=dropout)
        self.moe_ffn = MoE(d_model, d_ff, num_experts)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(0.1)

    def forward(self, src, attn_mask=None):
        norm_src = self.norm1(src)
        src2, attn_weights = self.self_attn(norm_src, attn_mask=attn_mask)
        src = src + self.drop_path(self.dropout(src2))

        norm_src = self.norm2(src)
        src2 = self.moe_ffn(norm_src)
        src = src + self.drop_path(self.dropout(src2))
        return src, attn_weights

class ContextEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.patching_embedding = PatchingEmbedding(configs.patch_len, configs.stride, configs.enc_in, configs.d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, (configs.seq_len - configs.patch_len) // configs.stride + 1, configs.d_model))
        self.encoder_layers = nn.ModuleList([
            MoETSAEncoderLayer(configs.d_model, configs.n_heads, configs.d_ff, configs.num_experts, configs.dropout)
            for _ in range(configs.e_layers)
        ])
        self.attention_pool = AttentionPool(d_model=configs.d_model, n_heads=configs.n_heads)

    def forward(self, x, get_attn=False):
        x_patched = self.patching_embedding(x)
        x_patched += self.pos_encoder

        attn_weights_list = []
        
        for layer in self.encoder_layers:
            x_patched, attn_weights  = layer(x_patched)
            if get_attn:
                attn_weights_list.append(attn_weights)
        
        summary_context = self.attention_pool(x_patched)

        if get_attn:
            return summary_context, attn_weights_list
        
        return summary_context